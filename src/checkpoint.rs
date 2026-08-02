//! WAL checkpointing for all Turso database stores.
//!
//! MahBot uses `multiprocess_wal` mode (configured in [`crate::turso::EXPERIMENTAL_FEATURES`]),
//! which relies on a `.tshm` shared-memory file for WAL coordination between
//! connections. On process crash or hard exit (`std::process::exit(0)` or signal
//! without cleanup), the `.tshm` file becomes stale and pending WAL frames are
//! permanently lost.
//!
//! This module provides the canonical checkpoint entry points:
//! [`checkpoint_all_databases`] (TRUNCATE, for controlled exit-time paths where
//! single-writer is guaranteed: self-update restart, GUI exit) and
//! [`periodic_checkpoint_all_databases`] (non-truncating below the WAL-size
//! cap, TRUNCATE above it — the auto-checkpoint loop spawned by the binary's
//! background task set).
//!
//! # Why keep `multiprocess_wal`?
//!
//! `multiprocess_wal` (via Turso) forces `NoLock` on all connections, making
//! explicit WAL checkpoints the only way to guarantee durability on exit
//! (the root cause this module works around). Removing it would eliminate the
//! data-loss mechanism entirely for normal (non-`exit(0)`) exits.
//!
//! The feature is retained because `mahbot debug` (the CLI subcommand) opens
//! the same `.db` files while the daemon is running. Without
//! `multiprocess_wal`, the debug tool and the daemon would share a single
//! WAL file without coordination — strictly worse than the current approach
//! (all connections share a single WAL with `.tshm` coordination). A future
//! refactor could eliminate the debug CLI's need to access live databases
//! (e.g., via an IPC query endpoint), making `multiprocess_wal` removable.
//!
//! In the meantime, the checkpoint orchestration in this module (exit-time +
//! periodic) ensures that under normal operation no writes are lost, and
//! crash data loss is bounded to the auto-checkpoint interval (5 minutes).

use futures_util::future::{FutureExt, join_all};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use tracing::{error, info, warn};

/// Default cap on a store's on-disk `-wal` size (bytes) for the periodic
/// checkpoint mode. Below the cap the periodic loop runs non-truncating
/// (PASSIVE) checkpoints; a TRUNCATE runs only when the WAL exceeds the cap,
/// bounding WAL-file growth while keeping the frame-index reset that TRUNCATE
/// causes (the two-writer corruption vector) rare instead of every 5 minutes.
const DEFAULT_WAL_CHECKPOINT_CAP_BYTES: u64 = 32 * 1024 * 1024;

/// Revert gate for the non-truncating periodic mode: setting
/// `MAHBOT_WAL_CHECKPOINT_CAP_BYTES=0` restores the previous always-TRUNCATE
/// periodic behavior (used while the repro harness validates the new mode).
fn periodic_checkpoint_cap_bytes() -> u64 {
    std::env::var("MAHBOT_WAL_CHECKPOINT_CAP_BYTES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_WAL_CHECKPOINT_CAP_BYTES)
}

/// Which checkpoint mode a store uses.
#[derive(Debug, Clone, Copy)]
enum CheckpointPolicy {
    /// TRUNCATE every checkpoint (controlled exit-time paths, single writer).
    Truncate,
    /// PASSIVE while the on-disk `-wal` stays under the cap; TRUNCATE above it.
    PassiveCapped(u64),
}

impl CheckpointPolicy {
    /// The periodic-loop policy, honoring the revert gate.
    fn periodic() -> Self {
        let cap = periodic_checkpoint_cap_bytes();
        if cap == 0 {
            Self::Truncate
        } else {
            Self::PassiveCapped(cap)
        }
    }
}

/// Iterate all stores via [`crate::turso::iter_checkpoint_stores`] and run an
/// async operation on each initialized store in parallel.
///
/// This is the shared iteration pattern used by both
/// [`checkpoint_all_databases`] and [`verify_all_databases`]. Stores that
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
    F: Fn(&'static str, &'static crate::turso::Connection) -> Fut,
    Fut: Future<Output = ()>,
{
    let futs: Vec<_> = crate::turso::iter_checkpoint_stores()
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
/// are never properly closed. Without this explicit checkpoint, pending WAL
/// writes are silently lost and `.tshm` coordination files are left inconsistent.
///
/// Always runs TRUNCATE checkpoints — this is the single-writer exit-time path
/// (self-update handoff, GUI shutdown). Periodic checkpointing uses
/// [`periodic_checkpoint_all_databases`] instead, which avoids TRUNCATE under
/// live writers.
///
/// Skips stores that haven't been initialized yet, and stores whose WAL is
/// orphaned (on-disk `-wal` empty while `.tshm` advertises live frames) — a
/// checkpoint on an orphaned WAL would attempt a zero-fill. Logs and swallows
/// per-store errors to avoid blocking shutdown.
///
/// The store entries come from [`crate::turso::iter_checkpoint_stores`] — the
/// single source of truth for which stores get checkpointed.
pub async fn checkpoint_all_databases() {
    checkpoint_stores(CheckpointPolicy::Truncate).await;
}

/// Periodic defense-in-depth checkpoint (auto-checkpoint loop).
///
/// Runs non-truncating (PASSIVE) checkpoints below the WAL-size cap and a
/// TRUNCATE only when a store's on-disk `-wal` exceeds it — TRUNCATE resets
/// the shared WAL frame index, which is the two-writer corruption vector when
/// live connections append afterward, so it is avoided during normal operation.
/// Set `MAHBOT_WAL_CHECKPOINT_CAP_BYTES=0` to revert to always-TRUNCATE.
pub async fn periodic_checkpoint_all_databases() {
    checkpoint_stores(CheckpointPolicy::periodic()).await;
}

async fn checkpoint_stores(policy: CheckpointPolicy) {
    // The stores were opened under CONFIG's storage root — resolve the
    // artifact directory from the same source (identical to
    // default_config_dir() today, but canonical if a data-dir override lands).
    let root = crate::config::CONFIG.try_storage_root();
    for_each_store(|name, conn| {
        let root = root.clone();
        async move {
            let status = root
                .as_deref()
                .map(|r| crate::wal_guard::inspect_store(r, name));
            if status.as_ref().is_some_and(|s| s.orphaned_wal) {
                warn!(
                    db = %name,
                    "Skipping checkpoint: orphaned WAL (on-disk -wal empty while .tshm \
                     advertises live frames) — checkpointing would zero-fill. Note: the \
                     orphaned predicate has a quiet window (max_frame reads 0 right after a \
                     checkpoint), so a store orphaned within that window is checkpointed \
                     instead — a TRUNCATE on its empty WAL is then a no-op.",
                );
                return;
            }
            let truncate = match policy {
                CheckpointPolicy::Truncate => true,
                CheckpointPolicy::PassiveCapped(cap) => {
                    // Safe default under uncertainty (root unresolvable) is
                    // PASSIVE — TRUNCATE is the corruption vector, and the
                    // periodic path's job is defense-in-depth, not exit-time
                    // durability.
                    status.as_ref().is_some_and(|s| s.wal_size > cap)
                }
            };
            let outcome = if truncate {
                conn.checkpoint().await
            } else {
                conn.checkpoint_passive().await
            };
            match outcome {
                Ok(o) if o.is_complete() => info!(
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
        }
    })
    .await;
}

/// Run PRAGMA quick_check on all initialized database stores.
///
/// Iterates all stores via [`crate::turso::iter_checkpoint_stores`], runs
/// `quick_check` on each in parallel, and logs the results. Corruption
/// errors are logged at `error!` level for operator visibility in the
/// dashboard Logs page, and a full `PRAGMA integrity_check` is
/// automatically triggered on the affected store so the complete
/// diagnostic report is available without manual intervention.
/// Successes are logged at `info!` level.
///
/// Skips stores that haven't been initialized yet. This is a fire-and-forget
/// function: all per-store errors are logged and swallowed to avoid blocking
/// the caller (matching the pattern of [`checkpoint_all_databases`]).
pub async fn verify_all_databases() {
    for_each_store(|name, conn| async move {
        match conn.quick_check().await {
            Ok(()) => info!(db = %name, "Database integrity check passed"),
            Err(e) => {
                // Run the full integrity_check to get the complete diagnostic
                // report so operators can triage without running debug CLI.
                match conn.integrity_check().await {
                    Ok(problems) if problems.is_empty() => {
                        // quick_check reported corruption but integrity_check
                        // found nothing — unexpected but handle gracefully.
                        warn!(
                            db = %name,
                            "Full integrity check returned no problems after quick_check failure"
                        );
                    }
                    Ok(problems) => {
                        let count = problems.len();
                        let problems_joined = problems.join("; ");
                        error!(
                            error = %e, db = %name, count,
                            problems = %problems_joined,
                            "Integrity check failed for {} ({} issue(s)): {}",
                            name, count, problems_joined,
                        );
                    }
                    Err(diag_err) => {
                        error!(
                            error = %diag_err, db = %name,
                            "Full integrity check also failed — database corruption may be severe"
                        );
                    }
                }
            }
        }
    })
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that both `checkpoint_all_databases` and `verify_all_databases` are
    /// no-ops (no panic) when no stores are initialized (all `OnceCell`s are empty).
    #[tokio::test]
    async fn noop_when_no_stores() {
        checkpoint_all_databases().await;
        periodic_checkpoint_all_databases().await;
        verify_all_databases().await;
    }
}
