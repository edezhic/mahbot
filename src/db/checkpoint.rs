//! WAL checkpointing for all Turso database stores.
//!
//! The daemon runs in turso's default single-process mode (no
//! `multiprocess_wal`, no `.tshm` coordination file) — a single process holds
//! the sole writer connection. Checkpoints are hygiene, not a durability
//! requirement: committed transactions are durable at COMMIT time (fsync'd
//! WAL frames), and checkpointing merely compacts those frames into the main
//! DB file, reclaims WAL space, and resets the shared frame index.
//!
//! TRUNCATE at exit leaves a header-only WAL for a clean store handoff; it is
//! avoided under live writers because resetting the shared WAL frame index
//! while a writer is live is the corruption vector — the periodic loop uses
//! PASSIVE below the 32 MiB cap instead. One known turso reopen defect: a
//! crash mid-transaction can leave un-published frame-index entries that abort
//! the next append ("shared WAL frame ids must increase monotonically") — a
//! defect in turso's WAL reopen, not a loss of committed data; committed
//! frames remain durable.
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
//! A failed periodic checkpoint triggers runtime corruption recovery: the
//! ticket-title FTS index is detect+repaired (see
//! [`crate::db::repair_ticket_title_fts_on_failed_checkpoint`]) and the
//! checkpoint retried; only an actual FTS rebuild decides the retry. On a
//! persistent failure the failure report is appended to `<root>/error.log`
//! (best-effort) and the graceful drain begins — the exit-time path never
//! recovers or drains, since the process is already exiting.

use futures_util::future::{FutureExt, join_all};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use tracing::{debug, error, info, warn};

use crate::db::{CheckpointOutcome, Connection, TicketTitleFtsRuntimeRepair};

/// Cap on a store's on-disk `-wal` size (bytes) for the periodic checkpoint
/// mode. Below the cap the periodic loop runs non-truncating (PASSIVE)
/// checkpoints; a TRUNCATE runs only when the WAL exceeds the cap, bounding
/// WAL-file growth while keeping the frame-index reset that TRUNCATE causes
/// (the live-writer corruption vector) rare instead of every 5 minutes.
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
/// structurally damaged (a corrupt main-DB header would make the checkpoint
/// fail loudly). Logs and swallows
/// per-store errors to avoid blocking shutdown.
///
/// The store entries come from [`crate::db::iter_checkpoint_stores`] — the
/// single source of truth for which stores get checkpointed.
pub async fn checkpoint_all_databases() {
    checkpoint_stores(CheckpointRound::Exit).await;
}

/// One 5-minute hygiene round: WAL checkpoint + integrity verification,
/// sharing a single store inspection per store — the checkpoint
/// and verify loops would otherwise each run an `inspect_store`
/// back-to-back. Also the periodic-loop policy: PASSIVE below the WAL-size
/// cap, TRUNCATE above it — TRUNCATE resets the shared WAL frame index (the
/// live-writer corruption vector), so it is avoided under live
/// writers; the TRUNCATE-above-cap branch is the only mechanism that shrinks
/// the WAL file (turso's own auto-checkpoint is PASSIVE-only). A checkpoint
/// failure on this round triggers the runtime FTS repair + graceful-shutdown
/// recovery (see [`recover_failed_checkpoint`]).
pub async fn periodic_checkpoint_and_verify() {
    checkpoint_stores(CheckpointRound::Periodic).await;
}

/// Discriminates the two [`checkpoint_stores`] rounds. Periodic implies
/// integrity verification plus runtime recovery on a checkpoint failure; Exit
/// is the TRUNCATE exit-time path and must never trigger recovery/shutdown
/// (the process is already exiting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckpointRound {
    /// Exit-time round: TRUNCATE, warn-only on failure.
    Exit,
    /// Periodic hygiene round: PASSIVE-capped, verify, and recover on failure.
    Periodic,
}

async fn checkpoint_stores(round: CheckpointRound) {
    // The stores were opened under CONFIG's storage root — resolve the
    // artifact directory from the same source (identical to
    // default_config_dir() today, but canonical if a data-dir override lands).
    let root = crate::config::CONFIG.try_storage_root();
    let truncate_gate = root.as_deref().is_none_or(truncate_allowed);
    let verify = matches!(round, CheckpointRound::Periodic);
    let policy = match round {
        CheckpointRound::Exit => CheckpointPolicy::Truncate,
        CheckpointRound::Periodic => CheckpointPolicy::periodic(),
    };
    for_each_store(|name, conn| {
        let root = root.clone();
        async move {
            // Single-process mode has no `.tshm` coordination and no external
            // writer, so there is no coordination identity to re-check. Only
            // the WAL size (for the TRUNCATE-vs-PASSIVE cap) needs the
            // store inspection below; a structurally-corrupt store's
            // checkpoint is attempted and its failure logged like any other.
            let status = root
                .as_deref()
                .map(|r| crate::db::wal_guard::inspect_store(r, name));
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
                Err(e) => {
                    warn!(error = %e, db = %name, "Failed to checkpoint database WAL");
                    // Periodic round only: attempt the runtime FTS repair and
                    // re-checkpoint; persistent failure drains (logs it). The
                    // exit-time round just logs — the process is already going
                    // away and must not trigger recovery/shutdown.
                    if matches!(round, CheckpointRound::Periodic) {
                        recover_failed_checkpoint(name, conn, &e, truncate, root.as_deref()).await;
                    }
                }
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

/// Recovery for a failed periodic checkpoint. Runs the runtime FTS repair
/// (reusing the boot-path code); retries the checkpoint after a rebuild. `root`
/// is the storage root the failure report is written under (threaded from the
/// caller — `None` when unresolvable). The continue-serving decision is made in
/// [`recover_failed_checkpoint_inner`] (the test-injectable seam) and is not
/// surfaced here — the production call site already discards it.
async fn recover_failed_checkpoint(
    name: &str,
    conn: &Connection,
    error: &anyhow::Error,
    truncate: bool,
    root: Option<&Path>,
) {
    let retry = async move {
        if truncate {
            conn.checkpoint().await
        } else {
            conn.checkpoint_passive().await
        }
    };
    recover_failed_checkpoint_inner(name, conn, error, root, retry).await;
}

/// Test-injectable core of [`recover_failed_checkpoint`]: `retry` is the
/// post-repair checkpoint attempt; `root` is the storage root the failure
/// report is written under (None when unresolvable). The returned bool is the
/// continue-serving decision, consumed by tests.
async fn recover_failed_checkpoint_inner(
    name: &str,
    conn: &Connection,
    error: &anyhow::Error,
    root: Option<&Path>,
    retry: impl Future<Output = anyhow::Result<CheckpointOutcome>>,
) -> bool {
    let repair = crate::db::repair_ticket_title_fts_on_failed_checkpoint(conn).await;
    // Re-verify only after an actual repair — a no-op repair means retrying a
    // deterministically-failing checkpoint is pointless. The retry is
    // panic-guarded like every probe on this path: a panic becomes the retry
    // error, so it flows into the report + drain instead of escaping to the
    // outer per-store catch (which would skip the drain).
    let retried = match &repair {
        TicketTitleFtsRuntimeRepair::Rebuilt(_) => {
            Some(match AssertUnwindSafe(retry).catch_unwind().await {
                Ok(result) => result,
                Err(panic) => Err(anyhow::anyhow!(
                    "retry checkpoint panicked: {}",
                    crate::util::panic_message(&panic)
                )),
            })
        }
        _ => None,
    };
    if let Some(Ok(o)) = &retried {
        info!(
            db = %name,
            repair = ?repair,
            complete = o.is_complete(),
            checkpointed = o.checkpointed_frames,
            "Checkpoint recovered after FTS repair — continuing"
        );
        return true;
    }
    // Persistent failure: append the full report to <root>/error.log, then
    // begin the graceful drain (shutdown::drain_begin; the drain cap bounds
    // stragglers). The write is synchronous and happens BEFORE the drain.
    let report = build_failure_report(
        name,
        error,
        &repair,
        retried.as_ref().and_then(|r| r.as_ref().err()),
        conn,
        root,
    )
    .await;
    match root {
        Some(root) => match write_checkpoint_error_log(root, &report) {
            Ok(path) => {
                error!(
                    db = %name,
                    log = %path.display(),
                    "Checkpoint failure persists after repair — error.log written, initiating graceful shutdown"
                );
            }
            Err(_) => {
                error!(
                    db = %name,
                    "Checkpoint failure persists after repair — error.log write FAILED, initiating graceful shutdown"
                );
            }
        },
        None => {
            error!(
                db = %name,
                "storage root unresolvable — error.log not written, initiating graceful shutdown"
            );
        }
    }
    crate::shutdown::drain_begin();
    false
}

/// Multi-line failure report for the persistent-checkpoint-failure terminal
/// path. Every diagnostic is best-effort — a failing OR panicking probe must
/// never prevent the report or its write: both probes are wrapped in panic
/// guards so a turso panic cannot escape to the outer `for_each_store`
/// `catch_unwind` (which would skip the drain and leave the service running
/// with a persistently failing checkpoint).
async fn build_failure_report(
    name: &str,
    error: &anyhow::Error,
    repair: &TicketTitleFtsRuntimeRepair,
    retry_error: Option<&anyhow::Error>,
    conn: &Connection,
    root: Option<&Path>,
) -> String {
    use std::fmt::Write;
    let mut body = String::new();
    let _ = writeln!(
        body,
        "MahBot checkpoint failure — {}",
        chrono::Utc::now().to_rfc3339()
    );
    let _ = writeln!(body, "store: {name}");
    match root {
        Some(r) => {
            let _ = writeln!(
                body,
                "db path: {}",
                crate::db::store_db_path(r, name).display()
            );
        }
        None => {
            let _ = writeln!(body, "db path: unresolvable (storage root unavailable)");
        }
    }
    let _ = writeln!(body, "checkpoint error: {error:#}");
    if let Some(re) = retry_error {
        let _ = writeln!(body, "retry error: {re:#}");
    }
    let _ = writeln!(body, "repair outcome: {}", repair.summary());
    match AssertUnwindSafe(conn.quick_check_problems())
        .catch_unwind()
        .await
    {
        Ok(Ok(problems)) if problems.is_empty() => {
            let _ = writeln!(body, "quick_check: ok");
        }
        Ok(Ok(problems)) => {
            let _ = writeln!(body, "quick_check problems: {}", problems.join("; "));
        }
        Ok(Err(e)) => {
            let _ = writeln!(body, "quick_check error: {e:#}");
        }
        Err(_) => {
            let _ = writeln!(body, "quick_check: probe panicked");
        }
    }
    if let Some(r) = root {
        let status = std::panic::catch_unwind(AssertUnwindSafe(|| {
            crate::db::wal_guard::inspect_store(r, name)
        }));
        match status {
            Ok(status) => {
                let _ = writeln!(
                    body,
                    "artifact state: store={} class={:?} wal_size={} has_stale_tshm={}",
                    status.store, status.class, status.wal_size, status.has_stale_tshm
                );
            }
            Err(_) => {
                let _ = writeln!(body, "artifact state: inspection panicked");
            }
        }
    }
    body
}

/// Append the failure report block to `<root>/error.log`, creating it if
/// absent. Returns the log path. Pure std::fs — no async, never panics on the
/// caller's behalf. The report + terminator go out as a single `write_all`
/// (one O_APPEND write in practice); even if the libc layer splits a large
/// buffer, each chunk is offset-atomic, so the worst case under concurrent
/// store failures is interleaved chunks, never torn bytes.
fn write_checkpoint_error_log(root: &Path, report: &str) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;
    let path = root.join("error.log");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    file.write_all(format!("{report}\n").as_bytes())?;
    Ok(path)
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

    /// Insert a minimal ticket row whose title is FTS-indexed.
    async fn insert_fts_ticket(conn: &Connection, id: &str, title: &str) {
        conn.execute(
            "INSERT INTO tickets (id, title, description, workspace_name, created_at, updated_at) \
             VALUES (?1, ?2, 'desc', 'ws', ?3, ?3)",
            crate::db::params![id.to_string(), title.to_string(), crate::db::now()],
        )
        .await
        .unwrap();
    }

    /// A checkpoint failure on a store whose title FTS index was corrupted is
    /// recovered: the FTS index is rebuilt (from the known DDL) and the
    /// checkpoint retried, so the service continues serving.
    #[tokio::test]
    async fn failed_checkpoint_with_broken_fts_repairs_and_retries() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .unwrap();
        insert_fts_ticket(&conn, "t-1", "Important bug fix one").await;
        insert_fts_ticket(&conn, "t-2", "Another relevant thing").await;

        // Break it: replace the FTS index with a same-named plain btree (the
        // same deterministic stand-in as the boot repair test).
        conn.execute_batch(
            "DROP INDEX idx_tickets_title_fts; \
             CREATE INDEX idx_tickets_title_fts ON tickets(title);",
        )
        .await
        .unwrap();
        assert!(
            !crate::db::is_fts_index(&conn, crate::db::TICKETS_FTS_INDEX_NAME).await,
            "index must be a btree before the recovery"
        );
        let before: i64 = conn
            .query_row("SELECT COUNT(*) FROM tickets", (), |r| r.get::<i64>(0))
            .await
            .unwrap();

        // A real post-repair retry (TRUNCATE, single-writer test env): the
        // production wrapper builds the retry from the failed attempt's mode.
        let retry = conn.checkpoint();
        let ok = recover_failed_checkpoint_inner(
            "core",
            &conn,
            &anyhow::anyhow!("injected checkpoint failure"),
            Some(tmp.path()),
            retry,
        )
        .await;
        assert!(
            ok,
            "the checkpoint must be retried and succeed after the repair"
        );

        assert!(
            crate::db::is_fts_index(&conn, crate::db::TICKETS_FTS_INDEX_NAME).await,
            "FTS index must be restored after the recovery"
        );
        let matched: String = conn
            .query_row(
                "SELECT id FROM tickets WHERE title MATCH ?1 LIMIT 1",
                crate::db::params![crate::db::sanitize_fts_query("Important bug fix one")],
                |r| r.get::<String>(0),
            )
            .await
            .unwrap();
        assert_eq!(
            matched, "t-1",
            "MATCH must find the known ticket after the rebuild"
        );
        let after: i64 = conn
            .query_row("SELECT COUNT(*) FROM tickets", (), |r| r.get::<i64>(0))
            .await
            .unwrap();
        assert_eq!(
            after, before,
            "ticket rows must be untouched by the FTS rebuild"
        );
    }

    /// The repair runs but the post-repair checkpoint still fails: the failure
    /// report is written to error.log and the graceful drain begins.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // retry_tests_lock serializes process-global test seams
    async fn repair_ran_but_checkpoint_still_fails_shuts_down() {
        let _lock = crate::util::test::retry_tests_lock();
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .unwrap();
        insert_fts_ticket(&conn, "t-1", "Important bug fix one").await;
        conn.execute_batch(
            "DROP INDEX idx_tickets_title_fts; \
             CREATE INDEX idx_tickets_title_fts ON tickets(title);",
        )
        .await
        .unwrap();

        let retry = async {
            Err::<crate::db::CheckpointOutcome, anyhow::Error>(anyhow::anyhow!(
                "injected persistent failure"
            ))
        };
        let ok = recover_failed_checkpoint_inner(
            "core",
            &conn,
            &anyhow::anyhow!("injected checkpoint failure"),
            Some(tmp.path()),
            retry,
        )
        .await;
        assert!(!ok, "a persistent retry failure must terminate the service");

        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.contains("injected checkpoint failure"),
            "the original checkpoint error must be in the report"
        );
        assert!(
            body.contains("injected persistent failure"),
            "the retry error must be in the report"
        );
        assert!(
            body.contains("quick_check"),
            "the report must carry a quick_check section"
        );
        assert!(
            crate::shutdown::is_draining(),
            "persistent failure must begin the graceful drain"
        );
        crate::shutdown::drain_clear();
    }

    /// A store with no ticket-title FTS index (repair is NotApplicable) skips
    /// the retry entirely yet still writes error.log and begins the drain.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // retry_tests_lock serializes process-global test seams
    async fn checkpoint_failure_without_fts_store_writes_error_log_and_drains() {
        let _lock = crate::util::test::retry_tests_lock();
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), "core"),
            "CREATE TABLE plain (id INTEGER PRIMARY KEY);",
        )
        .await
        .unwrap();

        // Panics if ever polled — repair is NotApplicable, so no retry happens.
        let retry = async { panic!("retry must not be polled for a non-FTS store") };
        let ok = recover_failed_checkpoint_inner(
            "core",
            &conn,
            &anyhow::anyhow!("injected checkpoint failure"),
            Some(tmp.path()),
            retry,
        )
        .await;
        assert!(
            !ok,
            "a non-FTS store must still terminate on a checkpoint failure"
        );

        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.contains("injected checkpoint failure"),
            "the checkpoint error must be in the report"
        );
        assert!(
            crate::shutdown::is_draining(),
            "persistent failure must begin the graceful drain"
        );
        crate::shutdown::drain_clear();
    }
}
