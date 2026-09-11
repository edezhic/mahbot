//! WAL checkpointing for all Turso database stores.
//!
//! The daemon runs in turso's default single-process mode (no
//! `multiprocess_wal`) — a single process holds the sole writer connection.
//! Checkpoints are hygiene, not a durability requirement: committed
//! transactions are durable at COMMIT time (fsync'd WAL frames), and
//! checkpointing merely compacts those frames into the main DB file, reclaims
//! WAL space, and resets the shared frame index.
//!
//! TRUNCATE at exit leaves the journal reclaimed (this engine truncates the
//! `-wal` to zero bytes, not to a header) for a clean store handoff; it is
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
//! A periodic round makes up to three attempts at a store's checkpoint — the
//! original plus at most two more — for EVERY store, whether or not it carries a
//! search index. What the engine answers decides the round, arm by arm
//! ([`periodic_checkpoint_inner`]), and only a genuine failure that no completion
//! follows stops the service. The exit-time round keeps its single attempt and
//! never recovers or drains ([`exit_checkpoint`]).
//!
//! The round's only recovery step runs after a first attempt that failed or was
//! answered busy — whatever held that attempt back, it can be what lets a retry
//! complete: the ticket-title FTS index is detect+repaired (see
//! [`crate::db::repair_ticket_title_fts_runtime`]) before the second attempt, and the
//! round is not decided by it.
//!
//! A reclaiming checkpoint the pre-shrink gate refuses is not one of those
//! attempts: the refusal ends the round, because the gate would refuse another
//! probe — and what a refusal is and is not is [`crate::db::shrink_gate`]'s to
//! state. A refusal never stops the service by itself: a round in which nothing
//! else ran keeps serving, while a genuine failure from an earlier attempt in the
//! same round still decides that round.
//!
//! Both rounds and the periodic integrity check record durably: an exit-round
//! checkpoint failure (which cannot drain or recover) is its own block, and a
//! failing periodic `quick_check` records a block once per store per process —
//! every further failing round only warns with the running count, so a persistent
//! condition cannot append a block every 5 minutes. A checkpoint-failure block is
//! by contrast filed on every round that reaches it, never deduplicated: such a
//! round is terminal (it stops the service), so that stop being recorded with its
//! own cause matters more than a filing rate the drain already bounds.
//!
//! # Reclaiming checkpoints are gated before they run
//!
//! Every reclaiming checkpoint this module issues — the exit round's attempt and
//! the periodic round's first attempt and each retry — goes through
//! [`checkpoint_attempt`], which runs the pre-shrink gate first;
//! [`crate::db::shrink_gate`] owns what it checks and what it accepts. The two
//! boot-time repair sites gate themselves before their own reclaiming
//! checkpoint, and `Connection::checkpoint_ungated()` is the raw engine call that
//! must not carry a reclaiming checkpoint outside the gate.
//!
//! # The engine's real cause is kept
//!
//! [`Connection::run_checkpoint`] arms a per-call sink so the engine's own reason
//! for a failed checkpoint travels with the checkpoint error — see
//! [`crate::db::checkpoint_cause`] for how it is obtained and what it costs. The
//! warn lines print that error's outermost message; the durable record renders the
//! whole chain. That reason is also the only thing that tells a blocked store —
//! which never stops the service — from a genuine pager error, which does
//! ([`crate::db::checkpoint_cause::is_blocked_checkpoint`]).

use futures_util::future::{FutureExt, join_all};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use crate::db::failure_record::{self, FailureKind, FailureReport, RoundCounter};
use crate::db::{
    CheckpointOutcome, Connection, TicketTitleFtsRuntimeRepair, checkpoint_cause, shrink_gate,
};

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

/// Total attempts a periodic checkpoint round makes (the original plus at most
/// two more) before it decides, for every store.
const CHECKPOINT_ATTEMPTS: usize = 3;

/// Pause before retrying a store that answered busy: it could not fold the journal
/// at that moment, so the retry gives whatever held it back this beat to clear. A
/// genuine failure is retried immediately — there is nothing to wait for.
const CHECKPOINT_RETRY_PAUSE: Duration = Duration::from_millis(100);

/// The `record` field's value when the durable block could not be filed: the
/// whole block already went to stderr (see [`failure_record::record`]).
const BLOCK_ON_STDERR: &str = "(could not be filed — block on stderr)";

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
/// leaves the journal reclaimed for a clean store handoff; why TRUNCATE is safe
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
/// is PASSIVE-only). A round's attempts, the runtime FTS repair it runs after a
/// first attempt that failed or was answered busy, and the genuine-failure-only
/// stop are [`periodic_checkpoint_inner`]'s.
pub async fn periodic_checkpoint_and_verify() {
    checkpoint_stores(CheckpointRound::Periodic).await;
}

/// Discriminates the two [`checkpoint_stores`] rounds. Periodic implies
/// integrity verification plus the retry/recovery rule
/// [`periodic_checkpoint_inner`] owns; Exit is the TRUNCATE exit-time path and
/// must never trigger recovery/shutdown (the process is already exiting).
#[derive(Clone, Copy)]
enum CheckpointRound {
    /// Exit-time round: TRUNCATE; a failure is recorded durably, never
    /// recovered.
    Exit,
    /// Periodic hygiene round: PASSIVE-capped, verify, and recover per
    /// [`periodic_checkpoint_inner`].
    Periodic,
}

async fn checkpoint_stores(round: CheckpointRound) {
    // The stores were opened under CONFIG's storage root — resolve the
    // artifact directory from the same source (identical to
    // default_config_dir() today, but canonical if a data-dir override lands).
    let root = crate::config::CONFIG.try_storage_root();
    let truncate_gate = root.as_deref().is_none_or(truncate_allowed);
    let policy = match round {
        CheckpointRound::Exit => CheckpointPolicy::Truncate,
        CheckpointRound::Periodic => CheckpointPolicy::periodic(),
    };
    for_each_store(|name, conn| {
        let root = root.clone();
        async move {
            // There is no external writer in single-process mode, so there is
            // no coordination identity to re-check, and no structural-corruption
            // classification decides here: a store's checkpoint is attempted and
            // its failure logged like any other, the pre-shrink gate being the
            // only condition under which a reclaiming attempt is withheld (see
            // [`crate::db::shrink_gate`]). The only consumer of a store
            // inspection is the periodic round's TRUNCATE-vs-PASSIVE cap (WAL
            // size), and that check must be STAT-ONLY (wal_guard's lock rule);
            // the exit round needs no inspection at all.
            let truncate = match policy {
                // Exit-time TRUNCATE (self-update handoff / shutdown): the
                // ENOSPC gate below applies here too — a TRUNCATE that runs
                // out of space mid-way is worse than the passive compaction
                // it skips. The handoff contract (release flock, spawn the
                // replacement) is unchanged; only the reclaimed-journal clean
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
            match round {
                CheckpointRound::Exit => {
                    exit_checkpoint(name, conn, truncate, root.as_deref()).await;
                }
                CheckpointRound::Periodic => {
                    periodic_checkpoint(name, conn, truncate, root.as_deref()).await;
                    verify_integrity(name, conn, root.as_deref()).await;
                }
            }
        }
    })
    .await;
}

/// The periodic round's integrity verification, independent of the checkpoint.
/// No in-place action: runtime-detected btree/index desync is handled by the
/// next boot's store open, not mid-run, and the known FTS count-mismatch false
/// positive is already filtered by the `quick_check` row scan. Failing rounds are
/// recorded durably (see [`record_integrity_failure`]).
async fn verify_integrity(name: &'static str, conn: &Connection, root: Option<&Path>) {
    // The filtered problem list, not `quick_check()`: a condition the boot path
    // deliberately leaves report-only (an unrecognised signature) is tolerated
    // here too, and only a check that cannot run or names actionable damage is
    // recorded.
    match conn.quick_check_problems().await {
        Ok(problems) if problems.is_empty() => {
            debug!(db = %name, "Database integrity check passed");
        }
        Ok(problems) if crate::db::is_tolerated_integrity_report(&problems) => {
            // A tolerated finding is deliberately neither filed in the durable
            // record nor a stop — but it must not be silent either. `warn`,
            // chosen deliberately: the signature may well be a corruption report
            // the boot path has not learned yet, so it stays visible at a level
            // operators watch, below the `error` of a filed failure.
            warn!(
                db = %name,
                problems = %problems.join("; "),
                "Database integrity check reported only tolerated findings — not recorded, not fatal",
            );
        }
        Ok(problems) => {
            record_integrity_failure(
                name,
                conn,
                &anyhow::anyhow!("{}", problems.join("; ")),
                root,
            );
        }
        Err(e) => record_integrity_failure(name, conn, &e, root),
    }
}

/// The result of one checkpoint attempt the round asked for.
enum Attempted {
    /// The attempt ran and the engine answered: a completed or partially folded
    /// checkpoint, or the engine's failure row — a genuine pager error or a
    /// blocked checkpoint, told apart by the reason the engine reports with it
    /// (see [`crate::db::checkpoint_cause`]).
    Ran(anyhow::Result<CheckpointOutcome>),
    /// The reclaiming checkpoint was refused by the pre-shrink gate: no
    /// checkpoint was issued (the gate recorded the refusal loudly).
    ShrinkRefused,
}

/// One checkpoint attempt, panic-guarded: a panic is a failed attempt (never
/// something that escapes the round and leaves the service running on a store
/// it has just proven it cannot write).
async fn guarded_checkpoint(
    attempt: impl Future<Output = anyhow::Result<CheckpointOutcome>>,
) -> anyhow::Result<CheckpointOutcome> {
    match AssertUnwindSafe(attempt).catch_unwind().await {
        Ok(result) => result,
        Err(panic) => Err(anyhow::anyhow!(
            "checkpoint attempt panicked: {}",
            crate::util::panic_message(&*panic)
        )),
    }
}

/// One checkpoint attempt: the pre-shrink gate first when the attempt is
/// reclaiming (see [`crate::db::shrink_gate`]), then the engine's own call
/// through the panic guard. Both rounds issue every reclaiming checkpoint through
/// here, so a retry is gated exactly like the round's first attempt (the
/// boot-time repair sites gate themselves — see the module doc).
async fn checkpoint_attempt(
    name: &'static str,
    conn: &Connection,
    truncate: bool,
    root: Option<&Path>,
) -> Attempted {
    if truncate && !shrink_gate::shrink_allowed(conn, name, root).await {
        return Attempted::ShrinkRefused;
    }
    Attempted::Ran(guarded_checkpoint(conn.checkpoint_mode(truncate)).await)
}

/// The exit/self-update round's one attempt: exactly the pre-existing exit
/// behaviour (one outcome, no retry, no drain). A failure is filed durably —
/// the process is going away, so the record is the only place it can surface.
///
/// The attempt is panic-guarded like the periodic round's ([`checkpoint_attempt`]):
/// a panic becomes this round's error and so reaches the durable record, instead
/// of only the per-store panic line `for_each_store` logs. Everything else about
/// the exit round is unchanged.
async fn exit_checkpoint(
    name: &'static str,
    conn: &Connection,
    truncate: bool,
    root: Option<&Path>,
) {
    match checkpoint_attempt(name, conn, truncate, root).await {
        Attempted::ShrinkRefused => {}
        Attempted::Ran(Ok(o)) if o.is_complete() => {
            debug!(
                db = %name,
                log = o.log_frames,
                checkpointed = o.checkpointed_frames,
                "Database WAL checkpointed",
            );
        }
        Attempted::Ran(Ok(o)) => {
            warn!(
                db = %name,
                busy = o.busy,
                log = o.log_frames,
                checkpointed = o.checkpointed_frames,
                "Checkpoint busy or partial — WAL frames left uncheckpointed",
            );
        }
        Attempted::Ran(Err(e)) => {
            warn!(error = %e, db = %name, "Failed to checkpoint database WAL");
            record_store_failure(
                FailureKind::ExitCheckpointFailure,
                "exit checkpoint failure",
                name,
                conn,
                &e,
                root,
            );
        }
    }
}

/// The periodic round: the three-attempt loop over the engine's real
/// checkpoint. `root` is the storage root the failure report is written under.
async fn periodic_checkpoint(
    name: &'static str,
    conn: &Connection,
    truncate: bool,
    root: Option<&Path>,
) {
    periodic_checkpoint_inner(name, conn, root, || {
        checkpoint_attempt(name, conn, truncate, root)
    })
    .await;
}

/// The periodic round's attempt loop. Its arms own the round's rule (what each
/// engine answer means) and [`crate::db::shrink_gate`] owns what a refusal is.
///
/// `attempt` is a parameter because this decision table has to be exercised
/// without an engine-side failing checkpoint, which is not reproducible
/// hermetically (see [`crate::db::checkpoint_cause`]): [`periodic_checkpoint`]
/// is the single caller and always passes the gate + engine attempt. It is the
/// round's only seam, and nothing outside this module can reach it.
///
/// The attempt future is `Send` because the periodic round runs inside the
/// process's spawned task set.
async fn periodic_checkpoint_inner<'a, Fut>(
    name: &'static str,
    conn: &'a Connection,
    root: Option<&Path>,
    mut attempt: impl FnMut() -> Fut,
) where
    Fut: Future<Output = Attempted> + Send + 'a,
{
    let mut failures: Vec<(usize, anyhow::Error)> = Vec::new();
    let mut repair: Option<TicketTitleFtsRuntimeRepair> = None;
    for attempt_no in 1..=CHECKPOINT_ATTEMPTS {
        let mut answered_busy = false;
        match attempt().await {
            // A refusal is never an attempt: the gate recorded it and no checkpoint
            // was issued. It never stops the service by itself, so with no genuine
            // failure the round ends here.
            Attempted::ShrinkRefused => {
                if failures.is_empty() {
                    return;
                }
                break;
            }
            // A completed checkpoint keeps the service serving, on any attempt: this
            // completion — never the repair's returned outcome — is what cures a
            // round whose earlier attempts failed.
            Attempted::Ran(Ok(o)) if o.is_complete() => {
                debug!(
                    db = %name,
                    log = o.log_frames,
                    checkpointed = o.checkpointed_frames,
                    "Database WAL checkpointed",
                );
                if !failures.is_empty() {
                    info!(
                        db = %name,
                        "Checkpoint completed on a retry after a genuine failure — continuing"
                    );
                }
                return;
            }
            // Not a completion, so a partially folded journal: normal — a reader
            // (this process's own reads included) capped how far the fold could go.
            // The round does nothing about it: no retry, no record, no warning, just
            // an INFO line (the level the log store retains). With no genuine failure
            // pending it ends the round; after one, the retries it opened still run. A
            // busy flag cannot ride a successful statement — the engine reports a
            // blocked store as the error below — so this arm is the whole
            // non-completion case.
            Attempted::Ran(Ok(o)) => {
                info!(
                    db = %name,
                    log = o.log_frames,
                    checkpointed = o.checkpointed_frames,
                    "Checkpoint folded the journal partially",
                );
                if failures.is_empty() {
                    return;
                }
            }
            // The engine's blocked answer: retried like a busy one, told from a
            // genuine pager error by the reason it carries (see
            // [`crate::db::checkpoint_cause`]). A store that stays blocked through the
            // budget keeps serving with its journal unfolded — the accepted trade for
            // never stopping on busy-ness — so its WAL keeps growing until it
            // unblocks.
            Attempted::Ran(Err(e)) if checkpoint_cause::is_blocked_checkpoint(&e) => {
                answered_busy = true;
                info!(attempt = attempt_no, error = %e, db = %name, "Checkpoint blocked — retrying");
            }
            // A genuine failure: kept for the report, retried immediately, and — when
            // nothing completes anywhere in the round — the reason the service stops.
            Attempted::Ran(Err(e)) => {
                warn!(attempt = attempt_no, error = %e, db = %name, "Failed to checkpoint database WAL");
                failures.push((attempt_no, e));
            }
        }
        // The runtime FTS repair runs here, after a first attempt that failed or was
        // answered busy; its outcome is reported, never used to decide.
        if attempt_no == 1 {
            repair = Some(crate::db::repair_ticket_title_fts_runtime(conn).await);
        }
        // Every iteration that reaches here leads to a retry (a completion and a
        // lone partial fold return, a refusal breaks); only a busy answer asks for
        // the pause first.
        if attempt_no < CHECKPOINT_ATTEMPTS && answered_busy {
            tokio::time::sleep(CHECKPOINT_RETRY_PAUSE).await;
        }
    }
    // The budget ran out with nothing but busy answers (a partially folded one
    // returns above): not a failure, nothing to record, nothing to warn about — the
    // next round tries again.
    if failures.is_empty() {
        info!(db = %name, "Checkpoint round ended without a completion — continuing");
        return;
    }
    // No completion anywhere in the round and at least one genuine failure: this is
    // the stop. The report is written (synchronously) before the drain, and the
    // round's repair outcome has no say in it. Nothing restarts the process after it
    // and the app shows no reason of its own for it, so this record and the log line
    // below are the operator's only trace.
    let report = build_failure_report(name, &failures, repair.as_ref(), conn).await;
    let pointer = failure_record::recorded_pointer(
        "checkpoint failure",
        failure_record::record(root, &report.render()),
    );
    error!(
        db = %name,
        record = pointer.as_deref().unwrap_or(BLOCK_ON_STDERR),
        "Genuine checkpoint failure with no completion — recorded, initiating graceful shutdown",
    );
    crate::shutdown::drain_begin();
}

/// The common body of a per-store runtime failure report: the store, its db
/// path, the reason, the environment note, and the artifact state — shared by
/// every per-store failure kind so their blocks cannot drift apart.
fn store_failure_report(
    kind: FailureKind,
    name: &'static str,
    e: &anyhow::Error,
    db_path: &Path,
) -> FailureReport {
    let report = FailureReport::new(kind)
        .store(name)
        .reason(format!("{e:#}"))
        .environment(crate::db::is_actionable_signal(e));
    with_store_file_state(report, db_path)
}

/// Add the store's db path and its stat-only artifact state — the `-wal` size
/// from [`crate::db::wal_guard::stat_size`], never a header read: wal_guard's
/// lock rule puts one out of reach in a process that already holds the store. A
/// stat that fails is reported as not obtainable, the same spelling the
/// pre-shrink gate's block uses. `db_path` is the file the failing connection has
/// open ([`Connection::db_path`]), so the record always names the very file whose
/// state it reports.
fn with_store_file_state(report: FailureReport, db_path: &Path) -> FailureReport {
    let wal_bytes = crate::db::wal_guard::stat_size(&crate::db::wal_path(db_path)).ok();
    report
        .db_path(db_path.to_path_buf())
        .extra(failure_record::artifact_state_line(wal_bytes))
}

/// File one per-store runtime failure block under the store's own measured file
/// ([`crate::db::failure_record::record_and_point`]).
fn record_store_failure(
    kind: FailureKind,
    what: &str,
    name: &'static str,
    conn: &Connection,
    e: &anyhow::Error,
    root: Option<&Path>,
) {
    let report = store_failure_report(kind, name, e, conn.db_path());
    failure_record::record_and_point(root, what, &report);
}

/// Per-store count of periodic rounds whose integrity check has failed, for the
/// process's lifetime (see [`record_integrity_failure`]).
static INTEGRITY_FAILURE_ROUNDS: RoundCounter = RoundCounter::new();

/// Record a failed periodic integrity check: the first failing round of a store
/// files a [`FailureKind::RuntimeIntegrityFailure`] block, every further round
/// only warns. No drain and no behaviour change otherwise.
fn record_integrity_failure(
    name: &'static str,
    conn: &Connection,
    e: &anyhow::Error,
    root: Option<&Path>,
) {
    let further_rounds = INTEGRITY_FAILURE_ROUNDS.prior_rounds(name);
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
        conn,
        e,
        root,
    );
}

/// Multi-line failure report for the genuine-failure terminal path. The engine's
/// own reason travels with each attempt's error when the engine reported one
/// (attached by [`Connection::run_checkpoint`] via
/// [`crate::db::checkpoint_cause`]), so the report needs no separate fold-in.
/// Every diagnostic is best-effort — a failing OR panicking probe must
/// never prevent the report or its write: the `quick_check` probe is wrapped in
/// a panic guard so a turso panic cannot escape to the outer `for_each_store`
/// `catch_unwind` (which would skip the drain and leave the service running
/// with a persistently failing checkpoint). The artifact state comes from the
/// stat-only [`crate::db::wal_guard::stat_size`] (wal_guard's lock rule: no
/// header read), so it reports the `-wal` size — or the shared not-obtainable
/// line when that stat fails — and deliberately no corruption class: a
/// header-level class is exactly what the lock rule puts out of reach, and the
/// `quick_check` section carries the integrity detail instead.
async fn build_failure_report(
    name: &'static str,
    failures: &[(usize, anyhow::Error)],
    repair: Option<&TicketTitleFtsRuntimeRepair>,
    conn: &Connection,
) -> FailureReport {
    // The checkpoint error is the report's reason, rendered as its own
    // long-standing `checkpoint error:` line rather than a `reason:` line; each
    // further attempt's error follows as its own `attempt N error:` line, N being
    // the attempt that produced it (attempts that were only answered busy, or that
    // folded the journal partially, leave a gap, deliberately). That first line is
    // the round's FIRST genuine failure and keeps the shape it has always had: an
    // attempt answered busy before it leaves no line of its own to carry a number,
    // so no number is claimed here.
    let mut report = FailureReport::new(FailureKind::CheckpointFailure)
        .store(name)
        .environment(
            failures
                .iter()
                .any(|(_, e)| crate::db::is_actionable_signal(e)),
        );
    // This report is built only after the attempts loop ended with a genuine
    // failure recorded, and the repair has always run by then — so both are
    // present by construction, not defensively.
    let (_, first) = failures
        .first()
        .expect("a failure report is built only after a failed attempt");
    report = report.extra(format!("checkpoint error: {first:#}"));
    for (attempt_no, e) in failures.iter().skip(1) {
        report = report.extra(format!("attempt {attempt_no} error: {e:#}"));
    }
    let repair = repair.expect("the round runs the repair before it can report");
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
    with_store_file_state(report, conn.db_path())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::checkpoint_cause;
    use crate::db::test_support::{
        build_empty_main_file_store, build_zero_page_count_store, build_zero_page_count_wal_store,
        fixture_rows, foreign_client_is_refused, fts_corruption_ddl, insert_fts_ticket, page_count,
    };
    use crate::db::wal_guard::{ShapeDefect, StoreShape};

    /// The checkpoint/verify entry points are no-ops (no panic) when no stores
    /// are initialized (all `OnceCell`s are empty).
    #[tokio::test]
    async fn noop_when_no_stores() {
        checkpoint_all_databases().await;
        periodic_checkpoint_and_verify().await;
    }

    /// A temp storage root holding one plain store named `name`, plus the
    /// connection that opened it (the file the durable record then names).
    async fn temp_store(name: &'static str) -> (tempfile::TempDir, Connection) {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), name),
            "CREATE TABLE plain (id INTEGER PRIMARY KEY);",
        )
        .await
        .unwrap();
        (tmp, conn)
    }

    /// A consolidated store whose ticket-title FTS index is corrupt — a same-named
    /// plain btree in its place — so the round's repair detects it and rebuilds it.
    async fn a_store_with_a_corrupt_fts_index(root: &Path) -> Connection {
        let conn = crate::db::open_consolidated_store(root).await.unwrap();
        insert_fts_ticket(&conn, "t-1", "Important bug fix one").await;
        insert_fts_ticket(&conn, "t-2", "Another relevant thing").await;
        conn.execute_batch(&fts_corruption_ddl()).await.unwrap();
        conn
    }

    /// The error a blocked checkpoint reaches the round as: the engine's own busy
    /// sentence — taken from [`checkpoint_cause::BLOCKED_REASON`], the constant the
    /// production classifier matches, so the two cannot drift — attached the way
    /// [`checkpoint_cause::CauseSink::attach`] attaches it around the product's
    /// constant text. The capture that produces the sentence, and the engine
    /// rendering it reads, are driven for real in `checkpoint_cause`'s own tests.
    fn blocked_checkpoint_error() -> anyhow::Error {
        anyhow::anyhow!("Unexpected result from PRAGMA wal_checkpoint").context(format!(
            "engine cause: {}",
            checkpoint_cause::BLOCKED_REASON
        ))
    }

    /// An exit-round checkpoint failure is recorded durably — the exit path can
    /// neither recover nor drain, so that block is the only place it can
    /// surface — with the stat-only artifact state and the environment note when
    /// the cause is an external condition.
    #[tokio::test]
    async fn exit_checkpoint_failure_is_recorded() {
        let (tmp, conn) = temp_store("core").await;
        record_store_failure(
            FailureKind::ExitCheckpointFailure,
            "exit checkpoint failure",
            "core",
            &conn,
            &anyhow::anyhow!("no space left on device"),
            Some(tmp.path()),
        );
        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        for needle in [
            "MahBot exit checkpoint failure",
            "store: core",
            failure_record::ENVIRONMENT_CAUSE,
            "reason: no space left on device",
            "artifact state: wal_size=",
        ] {
            assert!(
                body.contains(needle),
                "error.log must contain {needle:?}: {body}"
            );
        }
        assert!(
            body.contains(&format!("db path: {}", conn.db_path().display())),
            "the block must name the file the connection has open: {body}"
        );
    }

    /// A checkpoint failure's durable record carries the engine's own reason
    /// *and* the product's text as separate chain links — the product text
    /// alone reads the same for a corrupt store and for a healthy one (see
    /// [`crate::db::checkpoint_cause`]). The engine's reason is armed for the
    /// one call and attached to its error, so the record's reason line is the
    /// whole chain.
    #[tokio::test]
    async fn the_failure_record_carries_the_engines_own_reason() {
        use tracing_subscriber::layer::SubscriberExt;

        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::registry().with(checkpoint_cause::CauseCaptureLayer),
        );
        let (tmp, conn) = temp_store("core").await;
        let sink = checkpoint_cause::CauseSink::new();
        sink.scoped(async {
            tracing::debug!(
                target: "turso_core::vdbe::execute",
                "PRAGMA wal_checkpoint failed: engine-side detail"
            );
        })
        .await;
        let e = sink.attach(anyhow::anyhow!(
            "Unexpected result from PRAGMA wal_checkpoint"
        ));
        record_store_failure(
            FailureKind::ExitCheckpointFailure,
            "exit checkpoint failure",
            "core",
            &conn,
            &e,
            Some(tmp.path()),
        );

        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.contains(
                "reason: engine cause: PRAGMA wal_checkpoint failed: engine-side detail: \
                 Unexpected result from PRAGMA wal_checkpoint"
            ),
            "the reason line must be the full chain — the engine's own reason and the \
             product's text as separate links: {body}"
        );
    }

    /// A failing periodic integrity check is recorded once per store per process;
    /// later rounds only warn.
    #[tokio::test]
    async fn integrity_failure_is_recorded_once_then_counted() {
        // A store name no other test uses — the round count is process-global.
        let name = "integrity_probe";
        let (tmp, conn) = temp_store(name).await;
        for _ in 0..3 {
            record_integrity_failure(
                name,
                &conn,
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

    /// The report names the very file the connection has open and its stat-only
    /// artifact state — the record never has to re-derive the path from the root
    /// and the store name.
    #[tokio::test]
    async fn the_report_names_the_measured_file_and_its_artifact_state() {
        let (_tmp, conn) = temp_store("report_probe").await;
        let report = store_failure_report(
            FailureKind::ExitCheckpointFailure,
            "report_probe",
            &anyhow::anyhow!("disk gone"),
            conn.db_path(),
        )
        .render();
        for needle in [
            "MahBot exit checkpoint failure",
            "store: report_probe",
            "reason: disk gone",
            "artifact state: wal_size=",
        ] {
            assert!(
                report.contains(needle),
                "the report must contain {needle:?}: {report}"
            );
        }
        assert!(
            report.contains(&format!("db path: {}", conn.db_path().display())),
            "the report must name the file the connection has open: {report}"
        );
    }

    /// A checkpoint failure on a store whose title FTS index was corrupted is
    /// recovered: the FTS index is rebuilt (from the known DDL) and the next
    /// attempt completes a checkpoint, so the service continues serving — the
    /// completion is the cure, and the rebuild merely happens to be what made one
    /// possible.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn failed_checkpoint_with_broken_fts_repairs_and_retries() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = a_store_with_a_corrupt_fts_index(tmp.path()).await;
        assert!(
            !crate::db::is_fts_index(&conn, crate::db::TICKETS_FTS_INDEX_NAME).await,
            "index must be a btree before the recovery"
        );
        let before: i64 = conn
            .query_row("SELECT COUNT(*) FROM tickets", (), |r| r.get::<i64>(0))
            .await
            .unwrap();

        // The first attempt fails; the repair runs between it and the second,
        // which is a real checkpoint (TRUNCATE, single-writer test env).
        let mut attempt_no = 0;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempt_no += 1;
            let n = attempt_no;
            let conn = conn.clone();
            async move {
                Attempted::Ran(if n == 1 {
                    Err(anyhow::anyhow!("injected checkpoint failure"))
                } else {
                    conn.checkpoint_ungated().await
                })
            }
        })
        .await;
        assert_eq!(
            attempt_no, 2,
            "the failed first attempt must be retried exactly once after the repair"
        );
        assert!(
            !crate::shutdown::is_draining(),
            "a checkpoint completed by the retry must not begin the drain"
        );
        assert!(
            !tmp.path().join("error.log").exists(),
            "a checkpoint completed by the retry must not file a failure block"
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

    /// The repair rebuilt the index and every attempt still failed: the failure
    /// report is written to error.log with its real cause, and the service stops
    /// — a rebuilt index does not exempt the round, because no attempt completed
    /// a checkpoint.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_rebuilt_index_does_not_exempt_a_persistently_failing_round() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = a_store_with_a_corrupt_fts_index(tmp.path()).await;

        let mut attempt_no = 0;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempt_no += 1;
            let n = attempt_no;
            async move {
                let message = if n == 1 {
                    "injected checkpoint failure"
                } else {
                    "injected persistent failure"
                };
                Attempted::Ran(Err(anyhow::anyhow!(message)))
            }
        })
        .await;
        assert_eq!(
            attempt_no, CHECKPOINT_ATTEMPTS,
            "a persistent failure must run every attempt before the round decides"
        );

        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.contains("injected checkpoint failure"),
            "the original checkpoint error must be in the report: {body}"
        );
        assert!(
            body.contains("injected persistent failure"),
            "the later attempts' errors must be in the report: {body}"
        );
        assert!(
            body.contains("repair outcome: rebuilt after detection"),
            "the report must name what the round's repair returned: {body}"
        );
        assert!(
            crate::shutdown::is_draining(),
            "a rebuilt index must not exempt a round in which no attempt completed a checkpoint"
        );
        crate::shutdown::drain_clear();
    }

    /// A genuine failure, a repair that rebuilt the index, and attempts that
    /// follow it folding the journal only part of the way: the round still stops —
    /// a partially folded journal is not a completion, it does not cut the
    /// failure's budget short, and the repair's outcome decides nothing.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_rebuilt_index_does_not_exempt_a_partial_retry() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = a_store_with_a_corrupt_fts_index(tmp.path()).await;

        let mut attempt_no = 0;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempt_no += 1;
            let n = attempt_no;
            async move {
                if n == 1 {
                    Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure")))
                } else {
                    Attempted::Ran(Ok(CheckpointOutcome {
                        busy: false,
                        log_frames: 5,
                        checkpointed_frames: 1,
                    }))
                }
            }
        })
        .await;

        assert_eq!(
            attempt_no, CHECKPOINT_ATTEMPTS,
            "a partially folded attempt must not cut the failure's budget short"
        );
        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.contains("checkpoint error: injected checkpoint failure"),
            "the genuine failure must still be recorded with its real cause: {body}"
        );
        assert!(
            crate::shutdown::is_draining(),
            "a rebuilt index must not exempt a round in which a genuine failure had no completion"
        );
        crate::shutdown::drain_clear();
    }

    /// The round both repair variants below are run through: failing attempts and a
    /// completion on the last one. The service must serve and nothing may be filed —
    /// whatever the round's repair returned.
    async fn a_round_cured_by_its_last_attempt(conn: &Connection, root: &Path) {
        let mut attempt_no = 0;
        periodic_checkpoint_inner("core", conn, Some(root), || {
            attempt_no += 1;
            let n = attempt_no;
            async move {
                Attempted::Ran(if n == CHECKPOINT_ATTEMPTS {
                    Ok(CheckpointOutcome {
                        busy: false,
                        log_frames: 0,
                        checkpointed_frames: 0,
                    })
                } else {
                    Err(anyhow::anyhow!("injected checkpoint failure"))
                })
            }
        })
        .await;

        assert_eq!(
            attempt_no, CHECKPOINT_ATTEMPTS,
            "a completed last attempt ends the round without a fourth one"
        );
        assert!(
            !crate::shutdown::is_draining(),
            "a completed checkpoint after a failure must keep the service serving"
        );
        assert!(
            !root.join("error.log").exists(),
            "a round cured by its own retry must not file a failure block"
        );
    }

    /// The repair's outcome decides nothing: this store carries no ticket-title FTS
    /// index (so the repair found nothing to rebuild), and the round is cured by its
    /// own completed retry all the same.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_completed_retry_keeps_serving_whatever_the_repair_returned() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;
        assert!(
            matches!(
                crate::db::repair_ticket_title_fts_runtime(&conn).await,
                TicketTitleFtsRuntimeRepair::NotApplicable
            ),
            "this store must classify as a store with nothing to rebuild"
        );

        a_round_cured_by_its_last_attempt(&conn, tmp.path()).await;
    }

    /// A repair that fails outright decides nothing either: this store's
    /// ticket-title index cannot be rebuilt — its `tickets` table has no title
    /// column, so the MATCH probe reads it as corrupt and the rebuild's CREATE
    /// INDEX cannot succeed — and the round is cured by its own retry all the same.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_failed_repair_does_not_stop_a_round_its_retry_completes() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), "core"),
            "CREATE TABLE tickets (id INTEGER PRIMARY KEY);\
             CREATE INDEX idx_tickets_title_fts ON tickets (id);",
        )
        .await
        .unwrap();
        // The premise, asserted rather than assumed: this store makes the round's
        // repair fail, so the round's outcome is the `Failed` one (a rollback, so
        // the state the round then sees is unchanged).
        assert!(
            matches!(
                crate::db::repair_ticket_title_fts_runtime(&conn).await,
                TicketTitleFtsRuntimeRepair::Failed(_)
            ),
            "this store must classify as a store whose repair fails"
        );

        a_round_cured_by_its_last_attempt(&conn, tmp.path()).await;
    }

    /// A store with no ticket-title FTS index (repair is NotApplicable) still
    /// gets all three attempts, writes error.log and begins the drain; an
    /// environment-caused checkpoint error is marked as such in the record.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn checkpoint_failure_without_fts_store_writes_error_log_and_drains() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let mut attempts = 0usize;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempts += 1;
            async { Attempted::Ran(Err(anyhow::anyhow!("no space left on device"))) }
        })
        .await;
        assert_eq!(
            attempts, CHECKPOINT_ATTEMPTS,
            "every store gets all three attempts, indexed or not"
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

    /// A store whose ticket-title index is present and healthy still gets all
    /// three attempts, and a persistent failure still stops the service: the
    /// attempts are never gated on what the repair applies or what it returns.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_healthy_index_store_gets_all_three_attempts_and_stops() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .expect("open the consolidated store");
        insert_fts_ticket(&conn, "t-1", "Important bug fix one").await;

        let mut attempts = 0usize;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempts += 1;
            async { Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure"))) }
        })
        .await;
        assert_eq!(
            attempts, CHECKPOINT_ATTEMPTS,
            "an indexed store gets every attempt too",
        );

        let body = std::fs::read_to_string(tmp.path().join("error.log")).expect("error.log");
        assert!(
            body.contains("repair outcome: healthy"),
            "the report must name what the round's repair returned: {body}"
        );
        assert!(
            crate::shutdown::is_draining(),
            "a persistent failure on an indexed store must still stop the service",
        );
        crate::shutdown::drain_clear();
    }

    /// A partially folded journal is normal: the round does nothing about it —
    /// no retry, no record, no warning — and the service keeps serving.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_partially_folded_journal_is_normal_and_ends_the_round() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let mut attempts = 0usize;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempts += 1;
            async {
                Attempted::Ran(Ok(CheckpointOutcome {
                    busy: false,
                    log_frames: 5,
                    checkpointed_frames: 1,
                }))
            }
        })
        .await;

        assert_eq!(
            attempts, 1,
            "a partially folded journal must not be retried"
        );
        assert!(
            !crate::shutdown::is_draining(),
            "a partially folded journal must not begin the drain"
        );
        assert!(
            !tmp.path().join("error.log").exists(),
            "a partially folded journal must not file a failure block"
        );
    }

    /// A store that answers busy is retried after a pause, and a round that stays
    /// busy through the whole budget still stops nothing and records nothing:
    /// busy-ness is never a reason to stop.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_busy_store_is_retried_and_a_still_busy_round_keeps_serving() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let mut attempts = 0usize;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempts += 1;
            async { Attempted::Ran(Err(blocked_checkpoint_error())) }
        })
        .await;

        assert_eq!(
            attempts, CHECKPOINT_ATTEMPTS,
            "a blocked checkpoint must be retried up to the round's budget"
        );
        assert!(
            !crate::shutdown::is_draining(),
            "a round that stayed busy must keep the service serving"
        );
        assert!(
            !tmp.path().join("error.log").exists(),
            "a round that stayed busy must not file a failure block"
        );
    }

    /// A round whose first genuine failure comes after a busy attempt records it on
    /// the long-standing `checkpoint error:` line with no number on it — a busy
    /// attempt leaves no line of its own — while the failures that follow keep their
    /// own `attempt N error:` lines.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_busy_attempt_before_a_failure_leaves_the_cause_line_unmarked() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let mut attempt_no = 0usize;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempt_no += 1;
            let n = attempt_no;
            async move {
                if n == 1 {
                    Attempted::Ran(Err(blocked_checkpoint_error()))
                } else {
                    Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure")))
                }
            }
        })
        .await;

        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.lines()
                .any(|line| line == "checkpoint error: injected checkpoint failure"),
            "a busy first attempt leaves no line, so the cause line stays unmarked: {body}"
        );
        assert!(
            body.contains("attempt 3 error: injected checkpoint failure"),
            "the later failure must keep its own labelled line: {body}"
        );
        assert!(
            crate::shutdown::is_draining(),
            "a genuine failure with no completion must stop the service"
        );
        crate::shutdown::drain_clear();
    }

    /// A genuine failure followed by attempts the engine answers busy still stops:
    /// busy is not a completion, the failure's budget runs to its end, and the
    /// round's failure is the record's cause. Busy-ness never stops the service by
    /// itself, but it does not erase a genuine failure either.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_failure_followed_by_busy_attempts_still_stops() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let mut attempt_no = 0usize;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempt_no += 1;
            let n = attempt_no;
            async move {
                if n == 1 {
                    Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure")))
                } else {
                    Attempted::Ran(Err(blocked_checkpoint_error()))
                }
            }
        })
        .await;

        assert_eq!(
            attempt_no, CHECKPOINT_ATTEMPTS,
            "a busy attempt must not cut a genuine failure's budget short"
        );
        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.contains("checkpoint error: injected checkpoint failure"),
            "the round's real failure must be the report's reason: {body}"
        );
        assert!(
            crate::shutdown::is_draining(),
            "a genuine failure must still stop the service despite later busy attempts"
        );
        crate::shutdown::drain_clear();
    }

    /// A round with a gap — a busy attempt between two failures — still records the
    /// failures it has: the first keeps the long-standing unmarked `checkpoint
    /// error:` line, the later ones carry their own attempt number, and the attempt
    /// that produced no error contributes no line.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn the_report_numbers_each_failure_after_the_first() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let mut attempt_no = 0usize;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempt_no += 1;
            let n = attempt_no;
            async move {
                match n {
                    1 => Attempted::Ran(Err(anyhow::anyhow!("injected first failure"))),
                    2 => Attempted::Ran(Err(blocked_checkpoint_error())),
                    _ => Attempted::Ran(Err(anyhow::anyhow!("injected third failure"))),
                }
            }
        })
        .await;

        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.lines()
                .any(|line| line == "checkpoint error: injected first failure"),
            "attempt 1's cause line keeps its long-standing byte-for-byte shape: {body}"
        );
        assert!(
            body.contains("attempt 3 error: injected third failure"),
            "the later error must carry the attempt that produced it: {body}"
        );
        assert!(
            !body.contains("attempt 2 error:"),
            "the busy/partial attempt produced no error to record: {body}"
        );
        assert!(
            crate::shutdown::is_draining(),
            "the accumulated failure must still stop the service"
        );
        crate::shutdown::drain_clear();
    }

    /// A panicking attempt is a failed attempt: the round makes all three
    /// attempts, files the panic as the reason, and drains.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_panicking_attempt_is_a_failed_attempt_and_stops_after_three() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let mut attempts = 0usize;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempts += 1;
            async {
                Attempted::Ran(guarded_checkpoint(async { panic!("injected attempt panic") }).await)
            }
        })
        .await;
        assert_eq!(
            attempts, CHECKPOINT_ATTEMPTS,
            "a panic is a failed attempt, so every attempt must run"
        );
        assert!(
            crate::shutdown::is_draining(),
            "a panic must not escape the round without beginning the drain"
        );
        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.contains("checkpoint attempt panicked"),
            "the panic must be filed as the failed attempt's reason: {body}"
        );
        assert!(
            body.contains("injected attempt panic"),
            "the panicking attempt's message must be in the report: {body}"
        );
        crate::shutdown::drain_clear();
    }

    /// A refused shrink is neither a failure nor an attempt: the round makes
    /// exactly one attempt — a refusal is not retried — and it never reaches the
    /// failure/stop machinery. The store's bytes, its verdict and its lock are the
    /// end-to-end states' evidence, which drive this same round.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_refused_shrink_is_not_a_failure_and_not_an_attempt() {
        crate::shutdown::drain_clear();
        // The gate files its refusal block on the first refusal for a store file
        // in this process (the count is process-global but keyed by the checked
        // file), and this test's own temp dir makes the file unique to it.
        let name = "refused_shrink_probe";
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = crate::db::store_db_path(tmp.path(), name);
        build_zero_page_count_store(&db_path).await;
        let conn = Connection::open(&db_path)
            .await
            .expect("open the fixture store");
        let path = tmp.path();
        let conn_ref = &conn;

        let mut attempts = 0usize;
        periodic_checkpoint_inner(name, &conn, Some(path), || {
            attempts += 1;
            checkpoint_attempt(name, conn_ref, true, Some(path))
        })
        .await;

        assert_eq!(attempts, 1, "a refusal must not be retried");
        let body = std::fs::read_to_string(tmp.path().join("error.log"))
            .expect("the refusal must be filed in the round's root");
        assert!(
            body.contains("MahBot store shrink refused"),
            "the gate must refuse a shrink on a zero-page-count store and record it: {body}"
        );
        assert!(
            !body.contains("MahBot checkpoint failure"),
            "a refused shrink must not file a checkpoint failure: {body}"
        );
        assert!(
            !crate::shutdown::is_draining(),
            "a refused shrink must not begin the drain"
        );
    }

    /// The exit/self-update round is gated exactly like the periodic one, and its
    /// refusal is recorded on a path where the ordinary log may already be
    /// stopping: a refusal means no reclaiming checkpoint is issued, and it is
    /// never filed as the exit round's checkpoint failure — the process is
    /// exiting and a store the gate protected must not be reported as one it
    /// could not write.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn an_exit_round_refusal_is_recorded_and_is_not_an_exit_failure() {
        crate::shutdown::drain_clear();
        let name = "exit_refused_shrink_probe";
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = crate::db::store_db_path(tmp.path(), name);
        build_zero_page_count_store(&db_path).await;
        let conn = Connection::open(&db_path)
            .await
            .expect("open the fixture store");
        let before = std::fs::read(&db_path).unwrap();

        exit_checkpoint(name, &conn, true, Some(tmp.path())).await;

        let body = std::fs::read_to_string(tmp.path().join("error.log"))
            .expect("the refusal must be filed in the round's root");
        assert!(
            body.contains("MahBot store shrink refused"),
            "the exit round must record a refused shrink: {body}"
        );
        assert!(
            !body.contains("MahBot exit checkpoint failure"),
            "a refused shrink is not an exit-round checkpoint failure: {body}"
        );
        assert!(
            !crate::shutdown::is_draining(),
            "the exit round never drains — the process is already exiting"
        );
        assert_eq!(
            std::fs::read(&db_path).unwrap(),
            before,
            "the refused reclaiming checkpoint must not shrink the store"
        );
    }

    /// A refusal on a retry is not an attempt and never the reason the service
    /// stops — but the genuine failure of an earlier attempt in the same round
    /// still stops it, so the stop rule is not weakened.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_refusal_on_a_retry_still_stops_on_the_rounds_real_failure() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let mut attempt_no = 0usize;
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), || {
            attempt_no += 1;
            let n = attempt_no;
            async move {
                if n == 1 {
                    Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure")))
                } else {
                    Attempted::ShrinkRefused
                }
            }
        })
        .await;
        assert_eq!(
            attempt_no, 2,
            "the round stops at the refusal instead of retrying forever"
        );
        let body = std::fs::read_to_string(tmp.path().join("error.log")).unwrap();
        assert!(
            body.contains("checkpoint error: injected checkpoint failure"),
            "the report's reason must be the round's real failure: {body}"
        );
        assert!(
            body.contains("MahBot checkpoint failure"),
            "the round's stop must be filed as a checkpoint failure — a refusal is \
             not one and is never written as one: {body}"
        );
        assert!(
            crate::shutdown::is_draining(),
            "the round's real failure must still stop the service"
        );
        crate::shutdown::drain_clear();
    }

    /// The durable record a refused reclaiming round must leave: the refusal
    /// block, never a checkpoint failure, and no drain — a refusal is neither a
    /// failure nor an attempt.
    fn assert_refusal_recorded(root: &Path) {
        let body = std::fs::read_to_string(root.join("error.log"))
            .expect("the refusal must be filed in the round's root");
        assert!(
            body.contains("MahBot store shrink refused"),
            "the gate must refuse the shrink and record it: {body}"
        );
        for forbidden in [
            "MahBot checkpoint failure",
            "MahBot exit checkpoint failure",
        ] {
            assert!(
                !body.contains(forbidden),
                "a refused shrink must not read as {forbidden:?}: {body}"
            );
        }
        assert!(
            !crate::shutdown::is_draining(),
            "a refused shrink must not begin the drain",
        );
    }

    /// After a refused round the engine must still hold the store and have changed
    /// nothing in it: both foreign `sqlite3` probes are refused, the verdict
    /// [`crate::db::wal_guard`] defines for the file has not moved — the gate is a
    /// consistency check about one checkpoint's arithmetic, never a second
    /// definition of a damaged store — and the main file and journal are
    /// byte-identical to the bytes read before the store was opened. The probes
    /// come FIRST on purpose: reading the store's own bytes closes a descriptor,
    /// which drops this process's record locks on the file, so the foreign
    /// observation must happen before the comparison.
    fn assert_store_untouched_by_the_refusal(
        db_path: &Path,
        before_main: &[u8],
        before_wal: &[u8],
        verdict: StoreShape,
    ) {
        foreign_client_is_refused(db_path);
        assert_eq!(
            crate::db::wal_guard::classify_store_shape(db_path),
            verdict,
            "a refused shrink must not change what counts as a damaged store",
        );
        assert_eq!(
            std::fs::read(db_path).expect("read the fixture main file"),
            before_main,
            "a refused shrink must not change a byte of the store's main file",
        );
        assert_eq!(
            std::fs::read(crate::db::wal_path(db_path)).expect("read the fixture journal"),
            before_wal,
            "a refused shrink must not change a byte of the store's journal",
        );
    }

    /// The shared body of the destructive states whose main file is non-empty and
    /// whose rows stay readable — `state` names the page-1 image that declares no
    /// pages. Read the fixture's bytes, pin the file-level verdict, run the real
    /// reclaiming round, and prove it was refused with the rows still served,
    /// nothing moved and the engine's lock intact; then run the ungated reclaiming
    /// checkpoint the gate withheld, last because it destroys the fixture.
    async fn refused_state_lets_no_shrink_through(tmp: &Path, db_path: &Path, state: &str) {
        let before_main = std::fs::read(db_path).expect("read the fixture main file");
        let before_wal = std::fs::read(crate::db::wal_path(db_path)).expect("read the journal");
        assert!(
            !before_main.is_empty(),
            "the fixture must hold data in its main file",
        );
        // A file-level check sees a healthy store here, and must keep doing so:
        // only the store's own answer, compared with these files' sizes, refuses
        // the shrink.
        let verdict = crate::db::wal_guard::classify_store_shape(db_path);
        assert_eq!(
            verdict,
            StoreShape::Present,
            "the gate must add no file-level defect to a store it refuses",
        );

        let conn = Connection::open(db_path)
            .await
            .expect("open the fixture store");
        assert_eq!(page_count(&conn).await, 0, "{state} must declare no pages");
        assert_eq!(
            fixture_rows(&conn).await,
            1,
            "the fixture's rows must stay readable",
        );

        periodic_checkpoint("core", &conn, true, Some(tmp)).await;

        assert_refusal_recorded(tmp);
        assert_eq!(
            fixture_rows(&conn).await,
            1,
            "the refused round must leave the store serving",
        );

        assert_store_untouched_by_the_refusal(db_path, &before_main, &before_wal, verdict);

        conn.checkpoint_ungated()
            .await
            .expect("run the ungated reclaiming checkpoint");
        let main_len = std::fs::metadata(db_path)
            .expect("stat the fixture main file")
            .len();
        assert_eq!(
            main_len, 0,
            "an ungated reclaiming checkpoint must truncate the main file to nothing, \
             left {main_len} bytes",
        );
    }

    /// On a healthy store the reclaiming checkpoint is still issued, still in its
    /// reclaiming mode, and still reclaims the journal — the gate withholds only
    /// an answer it cannot trust; it never downgrades the mode, and it never skips
    /// the checkpoint on a store that answers consistently.
    #[tokio::test]
    async fn a_healthy_store_still_runs_its_reclaiming_checkpoint() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let db_path = crate::db::store_db_path(tmp.path(), "core");
        let conn =
            crate::db::open_with_schema(&db_path, "CREATE TABLE plain (id INTEGER PRIMARY KEY);")
                .await
                .expect("open the healthy store");
        conn.execute("INSERT INTO plain (id) VALUES (1)", ())
            .await
            .expect("commit a row so the journal holds frames");
        let wal = crate::db::wal_path(&db_path);
        let before = crate::db::wal_guard::stat_size(&wal).expect("stat the store's journal");
        assert!(
            before > crate::db::wal_guard::WAL_HEADER_BYTES,
            "the healthy store must have frames to reclaim, found {before} bytes",
        );

        let Attempted::Ran(Ok(outcome)) =
            checkpoint_attempt("core", &conn, true, Some(tmp.path())).await
        else {
            panic!("a healthy store must still run its reclaiming checkpoint");
        };
        assert!(
            outcome.is_complete(),
            "the reclaiming checkpoint must complete on a healthy store",
        );
        let after = crate::db::wal_guard::stat_size(&wal).expect("stat the reclaimed journal");
        assert_eq!(
            after, 0,
            "the reclaiming mode must still reclaim the journal, left {after} bytes",
        );
    }

    /// The gate's destructive state A end to end through the real reclaiming
    /// round: an empty main file with a non-empty journal. Not one byte moves and
    /// the store keeps the engine's lock, though the engine resolves an empty
    /// database in this state (`sqlite_schema` is empty, so the fixture's rows are
    /// NOT readable), so the demonstration rests on the files' bytes, the locked
    /// store and the row-free statement that still runs — exactly what the fixture
    /// builder documents. The closing proof, last because it destroys the fixture,
    /// shows what the refusal withheld: an ungated reclaiming checkpoint truncates
    /// the journal that holds the only copy of the committed frames.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn empty_main_file_with_a_journal_is_refused_end_to_end() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let db_path = crate::db::store_db_path(tmp.path(), "core");
        let wal = crate::db::wal_path(&db_path);
        build_empty_main_file_store(&db_path).await;
        let before_main = std::fs::read(&db_path).expect("read the fixture main file");
        let before_wal = std::fs::read(&wal).expect("read the fixture journal");
        assert!(
            before_main.is_empty(),
            "the fixture must leave an empty main file, found {} bytes",
            before_main.len(),
        );
        // The boot pre-flight keeps refusing this file on its own terms: the gate
        // adds no rule of its own about what a store may look like.
        let verdict = crate::db::wal_guard::classify_store_shape(&db_path);
        assert_eq!(
            verdict,
            StoreShape::Unusable(ShapeDefect::ZeroBytes),
            "an empty main file must stay a boot refusal, not a gate finding",
        );

        let conn = Connection::open(&db_path)
            .await
            .expect("open the fixture store");
        assert_eq!(
            page_count(&conn).await,
            0,
            "the engine must answer no pages for the empty main file",
        );

        periodic_checkpoint("core", &conn, true, Some(tmp.path())).await;

        assert_refusal_recorded(tmp.path());
        let schema_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_schema", (), |r| {
                r.get::<i64>(0)
            })
            .await
            .expect("the refused round must leave the store serving");
        assert_eq!(
            schema_rows, 0,
            "the engine resolves an empty database in this state",
        );

        assert_store_untouched_by_the_refusal(&db_path, &before_main, &before_wal, verdict);

        conn.checkpoint_ungated()
            .await
            .expect("run the ungated reclaiming checkpoint");
        let journal_len = std::fs::metadata(&wal)
            .expect("stat the fixture journal")
            .len();
        assert_eq!(
            journal_len, 0,
            "an ungated reclaiming checkpoint must truncate the journal that holds the only \
             copy of the committed frames, left {journal_len} bytes",
        );
    }

    /// The gate's destructive state B end to end through the real reclaiming
    /// round: a non-empty main file whose resolved page-1 header declares size
    /// zero. The round's reclaiming attempt is refused with nothing issued and
    /// nothing counted, the rows stay readable, the store keeps the engine's lock,
    /// and both files are byte-identical. The closing proof, last because it
    /// destroys the fixture, shows what the refusal withheld: an ungated
    /// reclaiming checkpoint truncates the main file to nothing.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn zero_page_count_store_is_refused_end_to_end() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let db_path = crate::db::store_db_path(tmp.path(), "core");
        build_zero_page_count_store(&db_path).await;

        let state = "the fixture's page-1 header";
        refused_state_lets_no_shrink_through(tmp.path(), &db_path, state).await;
    }

    /// The gate's destructive state C end to end through the real reclaiming
    /// round: a main file whose newest in-journal page-1 image declares size zero.
    /// The round's reclaiming attempt is refused with nothing issued and nothing
    /// counted, the rows stay readable, the store keeps the engine's lock, and both
    /// files are byte-identical. The closing proof, last because it destroys the
    /// fixture, shows what the refusal withheld: an ungated reclaiming checkpoint
    /// truncates the main file to nothing.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn zero_page_count_in_the_journal_is_refused_end_to_end() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let db_path = crate::db::store_db_path(tmp.path(), "core");
        build_zero_page_count_wal_store(&db_path).await;

        let state = "the journal's newest page-1 image";
        refused_state_lets_no_shrink_through(tmp.path(), &db_path, state).await;
    }
}
