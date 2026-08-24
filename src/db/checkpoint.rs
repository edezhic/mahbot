//! WAL checkpointing for all Turso database stores.
//!
//! MahBot uses `multiprocess_wal` mode (configured in [`crate::db::EXPERIMENTAL_FEATURES`]),
//! which relies on a `.tshm` shared-memory file for WAL coordination between
//! connections.
//!
//! # WAL checkpoint hygiene
//!
//! Committed transactions are durable at COMMIT time: turso runs
//! synchronous=FULL (default), fsyncing the WAL before reporting a transaction
//! durable, so committed frames survive crash/SIGKILL without any checkpoint
//! (the durable `.tshm` index is never rebuilt or trimmed on a crash reopen;
//! only the exit-time TRUNCATE resets it — see the known-reopen defect
//! below). Checkpoints are hygiene, not the durability mechanism: they compact
//! committed frames
//! from the WAL into the main DB file, reclaim WAL space, and reset the shared
//! frame index. TRUNCATE at exit leaves a header-only WAL (clean store handoff;
//! also avoids the stale shared-index reopen hazard). TRUNCATE is avoided under
//! live writers because resetting the shared frame index is the documented
//! two-writer corruption vector; the periodic loop uses PASSIVE below the
//! 32 MiB cap instead. Only uncommitted in-flight transactions are lost on
//! crash — normal DB semantics, independent of checkpoints. One known turso
//! reopen defect (tracked by the WAL-race harness): a crash mid-transaction can
//! leave un-published frame-index entries that abort the next append ("shared
//! WAL frame ids must increase monotonically") — a defect in turso's WAL
//! reopen, not a loss of committed data; committed frames remain durable.
//!
//! This module provides the canonical checkpoint entry points:
//! [`checkpoint_all_databases`] (TRUNCATE, for exit-time paths — self-update
//! restart is single-writer (agents cancelled, browser sessions closed, shutdown
//! signaled before the checkpoint); GUI exit runs while background writers are
//! still live, but turso serializes via its checkpoint lock, so the practical
//! effect is busy→warn, not corruption) and
//! [`periodic_checkpoint_and_verify`] (non-truncating below the WAL-size cap,
//! TRUNCATE above it, plus integrity verification sharing one
//! coordination-state inspection per store — the auto-checkpoint loop spawned
//! by the binary's background task set).
//!
//! # Why keep `multiprocess_wal`?
//!
//! `multiprocess_wal` (via Turso) forces `NoLock` on all connections, which
//! affects locking, not fsync: committed data is durable at COMMIT regardless
//! (see above). The exit-time TRUNCATE is retained for the clean store handoff
//! (header-only WAL), not as a durability requirement.
//!
//! The feature is retained because `mahbot debug` (the CLI subcommand) opens
//! the same `.db` files while the daemon is running. Without
//! `multiprocess_wal`, the debug tool and the daemon would share a single
//! WAL file without coordination — strictly worse than the current approach
//! (all connections share a single WAL with `.tshm` coordination). A future
//! refactor could eliminate the debug CLI's need to access live databases
//! (e.g., via an IPC query endpoint), making `multiprocess_wal` removable.

use futures_util::future::{FutureExt, join_all};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use tracing::{debug, error, info, warn};

/// Cap on a store's on-disk `-wal` size (bytes) for the periodic checkpoint
/// mode. Below the cap the periodic loop runs non-truncating (PASSIVE)
/// checkpoints; a TRUNCATE runs only when the WAL exceeds the cap, bounding
/// WAL-file growth while keeping the frame-index reset that TRUNCATE causes
/// (the two-writer corruption vector) rare instead of every 5 minutes.
const WAL_CHECKPOINT_CAP_BYTES: u64 = 32 * 1024 * 1024;

/// Default minimum free disk space (bytes) below which TRUNCATE checkpoints
/// are skipped (only PASSIVE runs). Overridable via
/// `MAHBOT_CHECKPOINT_MIN_FREE_BYTES`; `0` disables the gate. ENOSPC is never
/// corruption — it is an actionable signal, not a quarantine/recreate trigger.
const DEFAULT_CHECKPOINT_MIN_FREE_BYTES: u64 = 64 * 1024 * 1024;

/// Parse the TRUNCATE-min-free-space threshold from the environment.
fn checkpoint_min_free_bytes() -> u64 {
    std::env::var("MAHBOT_CHECKPOINT_MIN_FREE_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_CHECKPOINT_MIN_FREE_BYTES)
}

/// Free bytes on the filesystem backing `path` (0 when unavailable).
#[cfg(unix)]
fn available_free_bytes(path: &Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return 0;
    };
    let mut stats = unsafe { std::mem::zeroed::<libc::statvfs>() };
    if unsafe { libc::statvfs(c_path.as_ptr(), std::ptr::addr_of_mut!(stats)) } != 0 {
        return 0;
    }
    u64::from(stats.f_bavail).saturating_mul(stats.f_frsize)
}

/// Free bytes on the filesystem backing `path` (0 when unavailable; Windows
/// has no direct free-space query via libc — the gate simply never trips).
#[cfg(not(unix))]
fn available_free_bytes(_path: &Path) -> u64 {
    u64::MAX
}

/// True when the store's disk has enough free space for a TRUNCATE checkpoint.
/// Below the threshold only PASSIVE checkpoints run; the condition is logged
/// (actionable signal — never classified as corruption).
fn truncate_allowed(root: &Path) -> bool {
    let min = checkpoint_min_free_bytes();
    if min == 0 {
        return true;
    }
    let free = available_free_bytes(root);
    let allowed = free >= min;
    if !allowed {
        warn!(
            free_bytes = free,
            min_free_bytes = min,
            "Free disk space below TRUNCATE checkpoint threshold — running PASSIVE only",
        );
    }
    allowed
}

/// Which checkpoint mode a store uses.
#[derive(Debug, Clone, Copy)]
enum CheckpointPolicy {
    /// TRUNCATE every checkpoint (exit-time paths; any live writers are
    /// serialized by turso's checkpoint lock → busy→warn, not corruption).
    Truncate,
    /// PASSIVE while the on-disk `-wal` stays under the cap; TRUNCATE above it.
    PassiveCapped(u64),
}

impl CheckpointPolicy {
    /// The periodic-loop policy: PASSIVE below the cap, TRUNCATE above it.
    fn periodic() -> Self {
        Self::PassiveCapped(WAL_CHECKPOINT_CAP_BYTES)
    }
}

/// Iterate all stores via [`crate::db::iter_checkpoint_stores`] and run an
/// async operation on each initialized store in parallel.
///
/// This is the shared iteration pattern used by
/// [`checkpoint_all_databases`] and
/// [`periodic_checkpoint_and_verify`]. Stores that
/// haven't been initialized yet (connection is `None`) are silently skipped.
///
/// The operation closure receives `(&'static str, &'static Connection)` — the
/// store name and the canonical connection — and should return a `Future` that
/// completes the operation and logs the result.
///
/// Each per-store operation is wrapped in `catch_unwind` (scoped per store,
/// not around the whole loop) so a storage-layer panic in one store's
/// checkpoint/integrity operation cannot abort the other stores' operations.
/// `catch_unwind` only catches panics raised on the same thread that polls the
/// future — the futures here are polled on Tokio worker threads, so this
/// covers the storage-layer panics that unwind through a poll.
async fn for_each_store<F, Fut>(op: F)
where
    F: Fn(&'static str, &'static crate::db::Connection) -> Fut,
    Fut: Future<Output = ()>,
{
    let futs: Vec<_> = crate::db::iter_checkpoint_stores()
        .filter_map(|(name, conn_opt)| {
            let conn = conn_opt?;
            let fut = AssertUnwindSafe(op(name, conn)).catch_unwind();
            Some(async move {
                if let Err(payload) = fut.await {
                    error!(
                        panic = %crate::util::panic_message(&*payload),
                        db = name,
                        "Store operation panicked — isolated to this store",
                    );
                }
            })
        })
        .collect();
    join_all(futs).await;
}

/// Checkpoint all Turso database stores before hard process termination.
///
/// `std::process::exit(0)` bypasses Rust destructors, so Turso WAL connections
/// are never properly closed. The TRUNCATE leaves a header-only WAL for a clean
/// store handoff; committed data is already fsync-durable at COMMIT.
///
/// Always runs TRUNCATE checkpoints — the exit-time path (self-update handoff
/// is single-writer; GUI shutdown runs while background writers are still live,
/// but turso's checkpoint lock serializes them, so the effect is busy→warn, not
/// corruption). Periodic checkpointing uses
/// [`periodic_checkpoint_and_verify`] instead, which avoids TRUNCATE under
/// live writers.
///
/// Skips stores that haven't been initialized yet, and stores whose WAL is
/// orphaned (on-disk `-wal` empty while `.tshm` advertises live frames) — a
/// checkpoint on an orphaned WAL would attempt a zero-fill. Logs and swallows
/// per-store errors to avoid blocking shutdown.
///
/// The store entries come from [`crate::db::iter_checkpoint_stores`] — the
/// single source of truth for which stores get checkpointed.
pub async fn checkpoint_all_databases() {
    checkpoint_stores(CheckpointPolicy::Truncate, false).await;
}

/// One 5-minute hygiene round: WAL checkpoint + integrity verification,
/// sharing a single coordination-state inspection per store — the checkpoint
/// and verify loops would otherwise each run a full-predicate `inspect_store`
/// back-to-back. Also the periodic-loop policy: PASSIVE below the WAL-size
/// cap, TRUNCATE above it — TRUNCATE resets the shared WAL frame index (the
/// two-writer corruption vector), so it is avoided under live writers; the
/// TRUNCATE-above-cap branch is the only mechanism that shrinks the WAL file
/// (turso's own auto-checkpoint is PASSIVE-only).
pub async fn periodic_checkpoint_and_verify() {
    checkpoint_stores(CheckpointPolicy::periodic(), true).await;
}

async fn checkpoint_stores(policy: CheckpointPolicy, verify: bool) {
    // The stores were opened under CONFIG's storage root — resolve the
    // artifact directory from the same source (identical to
    // default_config_dir() today, but canonical if a data-dir override lands).
    let root = crate::config::CONFIG.try_storage_root();
    let truncate_gate = root.as_deref().is_none_or(truncate_allowed);
    for_each_store(|name, conn| {
        let root = root.clone();
        async move {
            // Re-check the persistent sidecar fds against the paths: an
            // external process that replaced -wal/-tshm means any checkpoint
            // would touch a foreign file — suspend this store's checkpoints.
            if conn.check_coordination_identity() == Some(crate::db::SidecarIdentity::Replaced) {
                warn!(
                    db = %name,
                    "Skipping checkpoint: coordination files replaced by an external process",
                );
                return;
            }
            let status = root
                .as_deref()
                .map(|r| crate::db::wal_guard::inspect_store(r, name, conn.store_fds()));
            if let Some(status) = status.as_ref().filter(|s| s.class.blocks_checkpoint()) {
                warn!(
                    db = %name,
                    class = status.class.label(),
                    "Skipping checkpoint: blocking coordination state — checkpointing would \
                     zero-fill or touch a foreign WAL",
                );
                if verify {
                    info!(
                        db = %name,
                        "Database integrity check skipped (checkpoint-blocked coordination state)"
                    );
                }
                return;
            }
            let truncate = match policy {
                // Exit-time TRUNCATE (self-update handoff / shutdown): the
                // ENOSPC gate below applies here too — a TRUNCATE that runs
                // out of space mid-way is worse than the passive compaction
                // it skips. The handoff contract (release flock, spawn the
                // replacement) is unchanged; only the header-only-WAL clean
                // handoff is dropped, and that is a hygiene nicety, not a
                // durability requirement (committed frames are fsync-durable
                // at COMMIT regardless).
                CheckpointPolicy::Truncate => truncate_gate,
                CheckpointPolicy::PassiveCapped(cap) => {
                    // PASSIVE while the WAL is absent/unmeasurable (status
                    // None — unresolvable root or an unregistered store) or
                    // below the cap; TRUNCATE only above it. `truncate_gate`
                    // adds the free-space check for resolvable roots — it is
                    // moot when `status` is None, which is exactly the
                    // unresolvable-root case (no stores initialized anyway).
                    status.as_ref().is_some_and(|s| s.wal_size > cap) && truncate_gate
                }
            };
            let outcome = if truncate {
                conn.checkpoint().await
            } else {
                conn.checkpoint_passive().await
            };
            match outcome {
                Ok(o) if o.is_complete() => debug!(
                    db = %name,
                    log = o.log_frames,
                    checkpointed = o.checkpointed_frames,
                    "Database WAL checkpointed",
                ),
                Ok(o) => warn!(
                    db = %name,
                    busy = o.busy,
                    log = o.log_frames,
                    checkpointed = o.checkpointed_frames,
                    "Checkpoint busy or partial — WAL frames left uncheckpointed",
                ),
                Err(e) => warn!(error = %e, db = %name, "Failed to checkpoint database WAL"),
            }
            // Integrity verification shares the inspect above when this is
            // the periodic round. Log-only: runtime-detected btree/index
            // desync is healed at the next store init (boot), not in place
            // mid-run. (The known FTS count-mismatch false positive is
            // already filtered by the quick_check row scan.)
            if verify {
                match conn.quick_check().await {
                    Ok(()) => debug!(db = %name, "Database integrity check passed"),
                    Err(e) => error!(error = %e, db = %name, "Database integrity check failed"),
                }
            }
        }
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checkpoint/verify entry points are no-ops (no panic) when no stores
    /// are initialized (all `OnceCell`s are empty).
    #[tokio::test]
    async fn noop_when_no_stores() {
        checkpoint_all_databases().await;
        periodic_checkpoint_and_verify().await;
    }
}
