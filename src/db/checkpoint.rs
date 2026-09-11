//! WAL checkpointing for all Turso database stores.
//!
//! The daemon runs in turso's default single-process mode (no
//! `multiprocess_wal`) — a single process holds the sole writer connection.
//! Checkpoints are hygiene, not a durability requirement: committed
//! transactions are durable at COMMIT time (fsync'd WAL frames), and
//! checkpointing merely compacts those frames into the main DB file, reclaims
//! WAL space, and resets the shared frame index.
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
//! restart is single-writer (agents cancelled, chrome sessions closed, shutdown
//! signaled before the checkpoint); GUI exit runs while background writers are
//! still live, but turso serializes via its checkpoint lock, so the practical
//! effect is busy→warn, not corruption) and
//! [`periodic_checkpoint_and_verify`] (non-truncating below the WAL-size cap,
//! TRUNCATE above it, plus an independent per-store integrity verification —
//! the auto-checkpoint loop spawned by the binary's background task set).
//!
//! A failed periodic checkpoint triggers runtime corruption recovery: the
//! ticket-title FTS index is detect+repaired (see
//! [`crate::db::repair_ticket_title_fts_on_failed_checkpoint`]) and the
//! checkpoint retried; only an actual FTS rebuild decides the retry. On a
//! persistent failure the failure report is appended to `<root>/error.log`
//! (best-effort; the shared writer lives in [`crate::db::failure_record`]) and
//! the graceful drain begins — the exit-time path never recovers or drains,
//! since the process is already exiting.
//!
//! Both rounds and the periodic integrity check record durably: an exit-round
//! checkpoint failure (which cannot drain or recover) is its own block, and a
//! failing periodic `quick_check` records a block once per store per process —
//! every further failing round only warns with the running count, so a persistent
//! condition cannot append a block every 5 minutes.

use futures_util::future::{FutureExt, join_all};
use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use tracing::{debug, error, info, warn};

use crate::db::failure_record::{self, FailureKind, FailureReport};
use crate::db::{CheckpointOutcome, Connection, TicketTitleFtsRuntimeRepair};
use crate::util::UnwrapPoison;

/// Cap on a store's on-disk `-wal` size (bytes) for the periodic checkpoint
/// mode. Below the cap the periodic loop runs non-truncating (PASSIVE)
/// checkpoints; a TRUNCATE runs only when the WAL exceeds the cap, bounding
/// WAL-file growth while keeping the frame-index reset that TRUNCATE causes
/// (the live-writer corruption vector) rare instead of every 5 minutes.
const WAL_CHECKPOINT_CAP_BYTES: u64 = 32 * 1024 * 1024;

/// Default minimum free disk space (bytes) below which TRUNCATE checkpoints
/// are skipped (only PASSIVE runs). Overridable via
/// `MAHBOT_CHECKPOINT_MIN_FREE_BYTES`; `0` disables the gate. ENOSPC is never
/// corruption — it is an actionable signal, not evidence about the store.
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
/// This is the shared iteration pattern behind both public entry points
/// (via [`checkpoint_stores`]). Stores that
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
/// are never properly closed. Runs TRUNCATE checkpoints — the exit-time path
/// leaves a header-only WAL for a clean store handoff; why TRUNCATE is safe
/// here and avoided by the periodic loop is the module-level invariant at the
/// top of this file. The TRUNCATE is downgraded to PASSIVE when free disk
/// space is below the ENOSPC gate (`truncate_allowed`).
///
/// Stores that were never initialized (connection is `None`) are skipped; a
/// panicking store operation is isolated to that store (see [`for_each_store`])
/// so one store cannot abort the round; an exit-round checkpoint failure is
/// filed in the durable record ([`record_store_failure`]) because the exit path
/// can neither recover nor drain.
///
/// The store entries come from [`crate::db::iter_checkpoint_stores`] — the
/// single source of truth for which stores get checkpointed. Periodic
/// checkpointing uses [`periodic_checkpoint_and_verify`] instead.
pub async fn checkpoint_all_databases() {
    checkpoint_stores(CheckpointRound::Exit).await;
}

/// One 5-minute hygiene round: the periodic inspection — WAL checkpoint and an
/// independent `quick_check` per store.
///
/// Checkpoint policy: PASSIVE below the WAL-size cap, TRUNCATE above it —
/// TRUNCATE resets the shared WAL frame index (the live-writer corruption
/// vector), so it is avoided under live writers; the TRUNCATE-above-cap branch
/// is the only mechanism that shrinks the WAL file (turso's own auto-checkpoint
/// is PASSIVE-only). A checkpoint failure on this round triggers the runtime FTS
/// repair + graceful-shutdown recovery (see [`recover_failed_checkpoint`]).
pub async fn periodic_checkpoint_and_verify() {
    checkpoint_stores(CheckpointRound::Periodic).await;
}

/// Discriminates the two [`checkpoint_stores`] rounds. Periodic implies
/// integrity verification plus runtime recovery on a checkpoint failure; Exit
/// is the TRUNCATE exit-time path and must never trigger recovery/shutdown
/// (the process is already exiting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckpointRound {
    /// Exit-time round: TRUNCATE; a failure is recorded durably, never
    /// recovered.
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
            // There is no external writer in single-process mode, so there is
            // no coordination identity to re-check, and a structurally-corrupt
            // store's checkpoint is attempted and its failure logged like any
            // other. The only consumer of a store inspection is the periodic
            // round's TRUNCATE-vs-PASSIVE cap (WAL size), and that check must be
            // STAT-ONLY (wal_guard's lock rule); the exit round needs no
            // inspection at all.
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
                    // PASSIVE while the WAL is absent/unmeasurable (the
                    // inspection yields nothing — unresolvable root) or
                    // below the cap; TRUNCATE only above it. `truncate_gate`
                    // adds the free-space check for resolvable roots — it is
                    // moot when the inspection yields nothing, which is
                    // exactly the unresolvable-root case (no stores
                    // initialized anyway).
                    root.as_deref()
                        .map(|r| {
                            let db_path = crate::db::store_db_path(r, name);
                            crate::db::wal_guard::wal_size(&db_path)
                        })
                        .is_some_and(|wal_size| wal_size > cap)
                        && truncate_gate
                }
            };
            let outcome = conn.checkpoint_mode(truncate).await;
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
                    // exit-time round records the failure instead — the process
                    // is already going away, so the durable record is the only
                    // place it can surface and it must trigger no
                    // recovery/shutdown.
                    if verify {
                        recover_failed_checkpoint(name, conn, &e, truncate, root.as_deref()).await;
                    } else {
                        record_store_failure(
                            FailureKind::ExitCheckpointFailure,
                            "exit checkpoint failure",
                            name,
                            &e,
                            root.as_deref(),
                        );
                    }
                }
            }
            // Integrity verification is independent of the checkpoint. No
            // in-place action: runtime-detected btree/index desync is handled by
            // the next boot's store open, not mid-run, and the known FTS
            // count-mismatch false positive is already filtered by the
            // quick_check row scan. The failing rounds are recorded durably
            // (see [`record_integrity_failure`]).
            if verify {
                // The filtered problem list, not `quick_check()`: a condition
                // the boot path deliberately leaves report-only (an
                // unrecognised signature) is tolerated here too, and only a
                // check that cannot run or names actionable damage is recorded.
                match conn.quick_check_problems().await {
                    Ok(problems) if problems.is_empty() => {
                        debug!(db = %name, "Database integrity check passed");
                    }
                    Ok(problems) if crate::db::is_tolerated_integrity_report(&problems) => {
                        // A tolerated finding is deliberately neither filed in
                        // the durable record nor a stop — but it must not be
                        // silent either. `warn`, chosen deliberately: the
                        // signature may well be a corruption report the boot
                        // path has not learned yet, so it stays visible at a
                        // level operators watch, below the `error` of a filed
                        // failure.
                        warn!(
                            db = %name,
                            problems = %problems.join("; "),
                            "Database integrity check reported only tolerated findings — not recorded, not fatal",
                        );
                    }
                    Ok(problems) => record_integrity_failure(
                        name,
                        &anyhow::anyhow!("{}", problems.join("; ")),
                        root.as_deref(),
                    ),
                    Err(e) => record_integrity_failure(name, &e, root.as_deref()),
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
    name: &'static str,
    conn: &Connection,
    error: &anyhow::Error,
    truncate: bool,
    root: Option<&Path>,
) {
    let retry = conn.checkpoint_mode(truncate);
    recover_failed_checkpoint_inner(name, conn, error, root, retry).await;
}

/// The artifact state cannot be probed when the storage root is unresolvable;
/// the record says so instead of dropping the line.
const ARTIFACT_STATE_UNAVAILABLE: &str =
    "artifact state: not obtainable on this path (the storage root is unresolvable)";

/// The common body of a per-store runtime failure report: the store, its db
/// path, the reason, the environment note, and the artifact state — shared by
/// every per-store failure kind so their blocks cannot drift apart.
fn store_failure_report(
    kind: FailureKind,
    name: &'static str,
    e: &anyhow::Error,
    root: Option<&Path>,
) -> FailureReport {
    let report = FailureReport::new(kind)
        .store(name)
        .reason(format!("{e:#}"))
        .environment(crate::db::is_actionable_signal(e));
    with_store_file_state(report, name, root)
}

/// Add the store's db path and its stat-only artifact state — the `-wal` size
/// from [`crate::db::wal_guard::wal_size`], never a header read: wal_guard's
/// lock rule puts one out of reach in a process that already holds the store.
/// The storage root is unresolvable on some paths; the record says so instead of
/// dropping the line.
fn with_store_file_state(
    report: FailureReport,
    name: &'static str,
    root: Option<&Path>,
) -> FailureReport {
    match root {
        Some(root) => {
            let db_path = crate::db::store_db_path(root, name);
            let wal_size = crate::db::wal_guard::wal_size(&db_path);
            report
                .db_path(db_path)
                .extra(format!("artifact state: wal_size={wal_size}"))
        }
        None => report.extra(ARTIFACT_STATE_UNAVAILABLE),
    }
}

/// File one per-store runtime failure block and point the operator at the file
/// it landed in, on the log channel (the boot path prints the same pointer on
/// its own diagnostics channel before tracing is up).
fn record_store_failure(
    kind: FailureKind,
    what: &str,
    name: &'static str,
    e: &anyhow::Error,
    root: Option<&Path>,
) {
    let filed = failure_record::record(root, &store_failure_report(kind, name, e, root).render());
    if let Some(pointer) = failure_record::recorded_pointer(what, filed) {
        info!("{pointer}");
    }
}

/// Per-store count of periodic rounds whose integrity check has failed, for the
/// process's lifetime (see [`record_integrity_failure`]).
static INTEGRITY_FAILURE_ROUNDS: LazyLock<Mutex<HashMap<&'static str, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record a failed periodic integrity check: the first failing round of a store
/// files a [`FailureKind::RuntimeIntegrityFailure`] block, every further round
/// only warns. No drain and no behaviour change otherwise.
fn record_integrity_failure(name: &'static str, e: &anyhow::Error, root: Option<&Path>) {
    let further_rounds = {
        let mut rounds = INTEGRITY_FAILURE_ROUNDS.lock().unwrap_poison();
        let count = rounds.entry(name).or_insert(0);
        *count += 1;
        *count - 1
    };
    if further_rounds > 0 {
        warn!(
            error = %e,
            db = %name,
            further_rounds,
            "Database integrity check failed again — already recorded on its first failure",
        );
        return;
    }
    error!(error = %e, db = %name, "Database integrity check failed");
    record_store_failure(
        FailureKind::RuntimeIntegrityFailure,
        "runtime integrity failure",
        name,
        e,
        root,
    );
}

/// Test-injectable core of [`recover_failed_checkpoint`]: `retry` is the
/// post-repair checkpoint attempt; `root` is the storage root the failure
/// report is written under (None when unresolvable). The returned bool is the
/// continue-serving decision, consumed by tests.
async fn recover_failed_checkpoint_inner(
    name: &'static str,
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
    // Persistent failure: file the full report in <root>/error.log, then begin
    // the graceful drain (shutdown::drain_begin; the drain cap bounds
    // stragglers). The write is synchronous and happens BEFORE the drain;
    // `record` puts the block on stderr when the root is unresolvable or the
    // write itself fails.
    let report = build_failure_report(
        name,
        error,
        &repair,
        retried.as_ref().and_then(|r| r.as_ref().err()),
        conn,
        root,
    )
    .await;
    if let Some(path) = failure_record::record(root, &report) {
        error!(
            db = %name,
            record = %path.display(),
            "Checkpoint failure persists after repair — recorded, initiating graceful shutdown"
        );
    } else {
        error!(
            db = %name,
            "Checkpoint failure persists after repair — the durable record could not be written (block on stderr), initiating graceful shutdown"
        );
    }
    crate::shutdown::drain_begin();
    false
}

/// Multi-line failure report for the persistent-checkpoint-failure terminal
/// path. Every diagnostic is best-effort — a failing OR panicking probe must
/// never prevent the report or its write: the `quick_check` probe is wrapped in
/// a panic guard so a turso panic cannot escape to the outer `for_each_store`
/// `catch_unwind` (which would skip the drain and leave the service running
/// with a persistently failing checkpoint). The artifact state comes from the
/// stat-only [`crate::db::wal_guard::wal_size`] (wal_guard's lock rule: no
/// header read), so it reports the `-wal` size and deliberately no corruption
/// class — a header-level class is exactly what the lock rule puts out of reach,
/// and the `quick_check` section carries the integrity detail instead.
async fn build_failure_report(
    name: &'static str,
    error: &anyhow::Error,
    repair: &TicketTitleFtsRuntimeRepair,
    retry_error: Option<&anyhow::Error>,
    conn: &Connection,
    root: Option<&Path>,
) -> String {
    // The checkpoint error is the report's reason, rendered as its own
    // long-standing `checkpoint error:` line rather than a `reason:` line.
    let mut report = FailureReport::new(FailureKind::CheckpointFailure)
        .store(name)
        .environment(
            crate::db::is_actionable_signal(error)
                || retry_error.is_some_and(crate::db::is_actionable_signal),
        )
        .extra(format!("checkpoint error: {error:#}"));
    if let Some(re) = retry_error {
        report = report.extra(format!("retry error: {re:#}"));
    }
    report = report.extra(format!("repair outcome: {}", repair.summary()));
    report = report.extra(
        match AssertUnwindSafe(conn.quick_check_problems())
            .catch_unwind()
            .await
        {
            Ok(Ok(problems)) if problems.is_empty() => "quick_check: ok".to_string(),
            Ok(Ok(problems)) => format!("quick_check problems: {}", problems.join("; ")),
            Ok(Err(e)) => format!("quick_check error: {e:#}"),
            Err(_) => "quick_check: probe panicked".to_string(),
        },
    );
    report = with_store_file_state(report, name, root);
    report.render()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_support::{fts_corruption_ddl, insert_fts_ticket};

    /// The checkpoint/verify entry points are no-ops (no panic) when no stores
    /// are initialized (all `OnceCell`s are empty).
    #[tokio::test]
    async fn noop_when_no_stores() {
        checkpoint_all_databases().await;
        periodic_checkpoint_and_verify().await;
    }

    /// An exit-round checkpoint failure is recorded durably — the exit path can
    /// neither recover nor drain, so that block is the only place it can
    /// surface — with the stat-only artifact state and the environment note when
    /// the cause is an external condition.
    #[test]
    fn exit_checkpoint_failure_is_recorded() {
        let tmp = tempfile::TempDir::new().unwrap();
        record_store_failure(
            FailureKind::ExitCheckpointFailure,
            "exit checkpoint failure",
            "core",
            &anyhow::anyhow!("no space left on device"),
            Some(tmp.path()),
        );
        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        for needle in [
            "MahBot exit checkpoint failure",
            "store: core",
            "db path:",
            failure_record::ENVIRONMENT_CAUSE,
            "reason: no space left on device",
            "artifact state: wal_size=0",
        ] {
            assert!(
                body.contains(needle),
                "error.log must contain {needle:?}: {body}"
            );
        }
    }

    /// A failing periodic integrity check is recorded once per store per process;
    /// later rounds only warn.
    #[test]
    fn integrity_failure_is_recorded_once_then_counted() {
        let tmp = tempfile::TempDir::new().unwrap();
        // A store name no other test uses — the round count is process-global.
        let name = "integrity_probe";
        for _ in 0..3 {
            record_integrity_failure(
                name,
                &anyhow::anyhow!("the integrity check cannot run"),
                Some(tmp.path()),
            );
        }
        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert_eq!(
            body.matches("MahBot runtime integrity failure").count(),
            1,
            "only the first failing round may file a block: {body}"
        );
        for needle in [
            "store: integrity_probe",
            "db path:",
            "reason: the integrity check cannot run",
        ] {
            assert!(
                body.contains(needle),
                "error.log must contain {needle:?}: {body}"
            );
        }
    }

    /// Every field of a per-store runtime failure report is rendered, including
    /// the ones that cannot be obtained on the path.
    #[test]
    fn store_failure_report_renders_unobtainable_fields() {
        let report = store_failure_report(
            FailureKind::ExitCheckpointFailure,
            "core",
            &anyhow::anyhow!("disk gone"),
            None,
        )
        .render();
        for needle in [
            "MahBot exit checkpoint failure",
            "store: core",
            crate::db::failure_record::UNKNOWN_DB_PATH,
            "reason: disk gone",
            ARTIFACT_STATE_UNAVAILABLE,
        ] {
            assert!(
                report.contains(needle),
                "the report must contain {needle:?}: {report}"
            );
        }
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

        // Break it: replace the FTS index with a same-named plain btree.
        conn.execute_batch(&fts_corruption_ddl()).await.unwrap();
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
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn repair_ran_but_checkpoint_still_fails_shuts_down() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .unwrap();
        insert_fts_ticket(&conn, "t-1", "Important bug fix one").await;
        conn.execute_batch(&fts_corruption_ddl()).await.unwrap();

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
    /// the retry entirely yet still writes error.log and begins the drain; an
    /// environment-caused checkpoint error is marked as such in the record.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn checkpoint_failure_without_fts_store_writes_error_log_and_drains() {
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
            &anyhow::anyhow!("no space left on device"),
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
            body.contains("checkpoint error: no space left on device"),
            "the checkpoint error must be in the report: {body}"
        );
        assert!(
            body.contains(failure_record::ENVIRONMENT_CAUSE),
            "a resource-caused checkpoint failure must be marked environment-caused: {body}"
        );
        assert!(
            crate::shutdown::is_draining(),
            "persistent failure must begin the graceful drain"
        );
        crate::shutdown::drain_clear();
    }
}
