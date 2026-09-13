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
//! original plus at most two more, spaced by [`CHECKPOINT_RETRY_PAUSE`] — for
//! EVERY store, whether or not it carries a search index. What the engine answers
//! decides the round, arm by arm ([`periodic_checkpoint_inner`]). The exit-time
//! round keeps its single attempt and never recovers or drains
//! ([`exit_checkpoint`]).
//!
//! # A failing round never stops the service by itself
//!
//! A genuine failure a completion does not follow opens (or extends) a per-store
//! *failure window*: the run of consecutive failing rounds the failure has lasted,
//! with its cumulative attempts and its opening cause. The round warns with the
//! window's running counts; only a window that has spanned both floors —
//! [`CHECKPOINT_FAILURE_MIN_ROUNDS`] consecutive rounds AND
//! [`CHECKPOINT_FAILURE_WINDOW`] of elapsed time — decides the stop, and then once,
//! filing those cumulative facts in the record. The window is in-memory and bounded,
//! and its retry pause races the daemon's own shutdown
//! ([`crate::shutdown::sleep_or_shutdown_or_drain`]), so a round the drain cuts short
//! decides nothing, and the store that drained the process can leave a sibling store's
//! open window unfiled.
//!
//! The only cure is a completed checkpoint, which closes the window wherever it
//! happens — first attempt, a retry, or a later round. A round that ends without a
//! genuine failure closes it too: a blocked or partially folded round is neither a
//! cure nor a failure, and counting it as failing would let non-consecutive failures
//! stop the service. The recorded cost of that rule is the mirror case — a store whose
//! rounds interleave genuine failures with blocked attempts closes its window every
//! time and keeps serving undecided, since busy-ness keeps its long-standing exemption
//! from stopping the service.
//!
//! Two consequences are deliberate. The classification and the stop rule are
//! unchanged, so a persistent refusal still exhausts the window and stops the
//! service: a retry policy cannot tell a transient engine refusal from a persistent
//! one except by observing whether a later attempt succeeds, and the terminal block
//! states that limit in as many words. And since no cause and no store is carved out,
//! EVERY genuine cause — ENOSPC, an I/O error, a removed volume — is served out for
//! the whole window before the stop: the stop is delayed by the window, never weakened
//! or dropped.
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
//! else ran keeps serving and closes any open window, while a genuine failure from
//! an earlier attempt in the same round still counts that round as a failing one.
//!
//! Both rounds and the periodic integrity check record durably: an exit-round
//! checkpoint failure (which cannot drain or recover) is its own block, and a
//! failing periodic `quick_check` records a block once per store per process —
//! every further failing round only warns with the running count, so a persistent
//! condition cannot append a block every 5 minutes. The checkpoint block follows
//! that shape for the same reason: every failing round short of the terminal one
//! would otherwise file a block 5 minutes apart.
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
//! whole chain. That reason is also what tells a blocked attempt, which never counts
//! towards a stop, from a genuine pager error
//! ([`crate::db::checkpoint_cause::is_blocked_checkpoint`]).

use futures_util::future::{FutureExt, join_all};
use std::collections::HashMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::db::failure_record::{self, FailureKind, FailureReport, RoundCounter};
use crate::db::{
    CheckpointOutcome, Connection, TicketTitleFtsRuntimeRepair, checkpoint_cause, shrink_gate,
};
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

/// Total attempts a periodic checkpoint round makes (the original plus at most
/// two more) before it decides, for every store.
const CHECKPOINT_ATTEMPTS: usize = 3;

/// Real pause before every retry, whichever answer opened it: a retry issued
/// immediately against the same in-memory engine state is the same attempt again,
/// so it is worth only what the spacing is worth. Two pauses per round stay far
/// below the round's 5-minute cadence.
const CHECKPOINT_RETRY_PAUSE: Duration = Duration::from_secs(1);

/// Consecutive failing rounds a store's failure window must span before a genuine
/// failure may decide the stop — one failing round never stops a healthy service.
const CHECKPOINT_FAILURE_MIN_ROUNDS: u64 = 2;

/// Span, since a window's first failure, the window must cover before a genuine
/// failure may decide the stop: the "several minutes" the rule asks for, made explicit
/// next to the round count. At the 5-minute cadence [`CHECKPOINT_FAILURE_MIN_ROUNDS`]
/// alone already spans more than this, so the floor is belt-and-braces — it binds only
/// for rounds run back to back, which no production path does, and the round's test
/// seam is where it is exercised.
const CHECKPOINT_FAILURE_WINDOW: Duration = Duration::from_secs(120);

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
    crate::util::disk::free_and_capacity(path).map_or(0, |(free, _)| free)
}

/// Windows has no direct free-space query via libc — the gate simply never trips.
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

/// The periodic round: the attempt loop over the engine's real checkpoint plus
/// the round's failure-window bookkeeping. `root` is the storage root the failure
/// report is written under.
async fn periodic_checkpoint(
    name: &'static str,
    conn: &Connection,
    truncate: bool,
    root: Option<&Path>,
) {
    periodic_checkpoint_inner(
        name,
        conn,
        root,
        CHECKPOINT_RETRY_PAUSE,
        CHECKPOINT_FAILURE_WINDOW,
        || checkpoint_attempt(name, conn, truncate, root),
    )
    .await;
}

/// What has accumulated for one store since the failure that opened its window:
/// the run of consecutive failing rounds a genuine checkpoint failure is decided
/// on (see the module header). Dropped by the round that closes it.
#[derive(Clone)]
struct FailureWindow {
    /// Consecutive failing rounds counted in this window, the current one
    /// included.
    rounds: u64,
    /// Checkpoint attempts those rounds made, blocked and partial ones included.
    attempts: u64,
    /// When the first failure of the window's first round was recorded, for the
    /// span.
    since: Instant,
    /// The engine's own reason for that first failure: the terminal record
    /// carries the window's cause, not only the deciding round's.
    first_failure: String,
}

impl FailureWindow {
    /// Count one more failing round that made `attempts` attempts.
    fn extend(&mut self, attempts: u64) {
        self.rounds += 1;
        self.attempts += attempts;
    }

    /// How long the window has been open.
    fn elapsed(&self) -> Duration {
        self.since.elapsed()
    }

    /// True when the window has spanned BOTH floors — the only condition under
    /// which a genuine failure decides the stop. `span` is the round's span floor.
    fn exhausted(&self, span: Duration) -> bool {
        self.rounds >= CHECKPOINT_FAILURE_MIN_ROUNDS && self.elapsed() >= span
    }

    /// The cumulative facts the terminal record and the terminal log line carry:
    /// how many attempts the failure took — [`Self::attempts`]'s tally, blocked and
    /// partial attempts included — and over how long, which is what tells the
    /// operator a transient condition from a persistent one.
    fn summary(&self) -> String {
        format!(
            "failure window: {} attempts (blocked and partial ones included) over {} consecutive \
             failing rounds, {:.1}s since the first failure",
            self.attempts,
            self.rounds,
            self.elapsed().as_secs_f64(),
        )
    }
}

/// Open failure windows, keyed by the failing store's own db path: unique per
/// store file, so two stores — or two tests' temp stores, which share store names
/// — never share a window.
static FAILURE_WINDOWS: LazyLock<Mutex<HashMap<PathBuf, FailureWindow>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The window key of the store `conn` has open (see [`FAILURE_WINDOWS`]).
fn failure_window_key(conn: &Connection) -> PathBuf {
    conn.db_path().to_path_buf()
}

/// Count one failing round for `key` and decide it in the same step, under one lock:
/// the round returns the window's state and whether it is exhausted, so a window can
/// never be decided while another round is extending it. An exhausted window is spent
/// and is not put back, so the block it decided is filed once for that window. `key` is
/// taken by value because it goes back into the map unless the window it decides is
/// exhausted; `since` and `first_failure` are the round's own first failure, kept only
/// when this round opens the window; `span` is the round's span floor, so a test can
/// compress it. Deciding and filing are outside the lock, which only makes two rounds
/// of the same store race there — and the daemon runs one round per store at a time.
fn extend_failure_window(
    key: PathBuf,
    attempts: u64,
    since: Instant,
    first_failure: &str,
    span: Duration,
) -> (FailureWindow, bool) {
    let mut windows = FAILURE_WINDOWS.lock().unwrap_poison();
    let mut window = windows.remove(&key).unwrap_or_else(|| FailureWindow {
        rounds: 0,
        attempts: 0,
        since,
        first_failure: first_failure.to_string(),
    });
    window.extend(attempts);
    let exhausted = window.exhausted(span);
    if !exhausted {
        windows.insert(key, window.clone());
    }
    (window, exhausted)
}

/// Drop the store's window when a round closes it — a completed checkpoint included
/// — warning with the cumulative facts it accumulated: a closed window is the whole
/// trace a failure that did not last leaves (only an exhausted window files a durable
/// block). Borrows the key, unlike [`extend_failure_window`]: a round that closes never
/// puts a window back. Returns whether one was open, which is what
/// [`cure_failure_window`] needs; the round's other closing arms ignore it.
fn close_failure_window(name: &'static str, key: &Path, reason: &'static str) -> bool {
    let Some(window) = FAILURE_WINDOWS.lock().unwrap_poison().remove(key) else {
        return false;
    };
    warn!(
        db = %name,
        window_attempts = window.attempts,
        window_rounds = window.rounds,
        elapsed_ms = window.elapsed().as_millis(),
        reason,
        "Checkpoint failure window closed — continuing",
    );
    true
}

/// A completed checkpoint is the cure: close the store's window if one is open and
/// warn with its cumulative facts. A round that had no window to close still warns
/// when it recorded genuine failures of its own — the completion is only a DEBUG line
/// and the log store retains INFO, so this warning is the whole trace a cured round
/// leaves. A plain healthy completion logs nothing but that DEBUG line.
fn cure_failure_window(name: &'static str, key: &Path, round_failures: usize) {
    if !close_failure_window(name, key, "a completed checkpoint") && round_failures > 0 {
        warn!(
            db = %name,
            round_failures,
            "Checkpoint completed on a retry after a genuine failure — continuing",
        );
    }
}

/// The periodic round's decision table: its arms own what each engine answer means
/// and [`crate::db::shrink_gate`] owns what a refusal is.
///
/// `attempt` is a parameter because this decision table has to be exercised
/// without an engine-side failing checkpoint, which is not reproducible
/// hermetically (see [`crate::db::checkpoint_cause`]): [`periodic_checkpoint`] is
/// the single caller and always passes the gate + engine attempt. It, and the two
/// timing parameters — `retry_pause`, the real pause before a retry, and `span`, how
/// long a store's failure window must have been open — are the round's only seam: the
/// tests compress both, production passes the two constants. Nothing outside this
/// module can reach it, which is also where the failure window's whole rule is
/// verified: that a real failure on a *healthy* store would have been cured by a
/// later attempt is not end-to-end observable, only that the later attempt is made.
///
/// The attempt future is `Send` because the periodic round runs inside the
/// process's spawned task set.
#[expect(clippy::too_many_lines)] // one arm per engine answer, each owning its own window step, reads as one flow
async fn periodic_checkpoint_inner<'a, Fut>(
    name: &'static str,
    conn: &'a Connection,
    root: Option<&Path>,
    retry_pause: Duration,
    span: Duration,
    mut attempt: impl FnMut() -> Fut,
) where
    Fut: Future<Output = Attempted> + Send + 'a,
{
    let key = failure_window_key(conn);
    let mut failures: Vec<(usize, anyhow::Error)> = Vec::new();
    let mut repair: Option<TicketTitleFtsRuntimeRepair> = None;
    let mut attempts_made = 0u64;
    // When this round's first genuine failure was seen (the round has no other use
    // for it): the instant a window it opens must be dated from, not the round's end.
    let mut first_failure_at: Option<Instant> = None;
    for attempt_no in 1..=CHECKPOINT_ATTEMPTS {
        match attempt().await {
            // A refusal is never an attempt: the gate recorded it and no checkpoint
            // was issued. It never stops the service by itself, so with no genuine
            // failure the round ends here — and closes the window, because a round
            // without a failure must not be counted as a failing one.
            Attempted::ShrinkRefused => {
                if failures.is_empty() {
                    close_failure_window(
                        name,
                        &key,
                        "round ended on a refused shrink with no failure",
                    );
                    return;
                }
                break;
            }
            // A completed checkpoint keeps the service serving, on any attempt and in
            // any round: this completion — never the repair's returned outcome — is
            // what cures a failure, wherever the failure was recorded.
            Attempted::Ran(Ok(o)) if o.is_complete() => {
                debug!(
                    db = %name,
                    log = o.log_frames,
                    checkpointed = o.checkpointed_frames,
                    "Database WAL checkpointed",
                );
                cure_failure_window(name, &key, failures.len());
                return;
            }
            // Not a completion, so a partially folded journal: normal — a reader
            // (this process's own reads included) capped how far the fold could go.
            // The round does nothing about it: no retry, no record, just an INFO line
            // (the level the log store retains) — and with no genuine failure pending
            // it ends the round, closing any window an earlier round opened, because a
            // round that did not fail must not be counted as a failing one. After a
            // genuine failure the retries it opened still run. A busy flag cannot ride
            // a successful statement — the engine reports a blocked attempt as the error
            // below — so this arm is the whole non-completion case.
            Attempted::Ran(Ok(o)) => {
                info!(
                    db = %name,
                    log = o.log_frames,
                    checkpointed = o.checkpointed_frames,
                    "Checkpoint folded the journal partially",
                );
                if failures.is_empty() {
                    close_failure_window(name, &key, "round folded partially with no failure");
                    return;
                }
            }
            // A blocked attempt: retried like any failure, told from a
            // genuine pager error by the reason it carries (see
            // [`crate::db::checkpoint_cause`]). A store that stays blocked through the
            // budget keeps serving with its journal unfolded — the accepted trade for
            // never stopping on busy-ness — so its WAL keeps growing until it
            // unblocks.
            Attempted::Ran(Err(e)) if checkpoint_cause::is_blocked_checkpoint(&e) => {
                info!(attempt = attempt_no, error = %e, db = %name, "Checkpoint blocked — retrying");
            }
            // A genuine failure: kept for the report, retried after the pause, and —
            // when the failure window is exhausted — the reason the service stops.
            Attempted::Ran(Err(e)) => {
                warn!(attempt = attempt_no, error = %e, db = %name, "Failed to checkpoint database WAL");
                first_failure_at.get_or_insert_with(Instant::now);
                failures.push((attempt_no, e));
            }
        }
        attempts_made += 1;
        // The runtime FTS repair runs here, after a first attempt that failed or was
        // answered busy; its outcome is reported, never used to decide.
        if attempt_no == 1 {
            repair = Some(crate::db::repair_ticket_title_fts_runtime(conn).await);
        }
        // Every iteration that reaches here leads to a retry (a completion and a
        // lone partial fold return, a refusal breaks), and the retry is spaced in
        // real time — never issued back to back against the same in-memory engine
        // state. The pause races the daemon's own shutdown, so a round the shutdown
        // or drain cut short decides nothing: the window is left exactly as it was,
        // and this round's failures reach no window at all, because a drain means the
        // process is on its way out and a window is a claim about one that keeps
        // serving.
        if attempt_no < CHECKPOINT_ATTEMPTS
            && !crate::shutdown::sleep_or_shutdown_or_drain(retry_pause).await
        {
            warn!(
                db = %name,
                round_attempts = attempts_made,
                "Checkpoint round cut short by the daemon's own shutdown — failure window not decided",
            );
            return;
        }
    }
    // The budget ran out with nothing but blocked/partial answers (a lone one
    // returns above): not a failure, nothing to record — the next round tries again,
    // and any window an earlier round opened closes here, because a round that did
    // not fail must not count as a failing one.
    if failures.is_empty() {
        info!(db = %name, "Checkpoint round ended without a completion — continuing");
        close_failure_window(name, &key, "round ended without a failure or a completion");
        return;
    }
    // The window records the failure that opened it: this round's first, which the
    // terminal block also renders as its `checkpoint error:` line.
    let (_, first) = failures
        .first()
        .expect("the window is only extended after a failed attempt");
    let first_failure = format!("{first:#}");
    let (window, exhausted) = extend_failure_window(
        key,
        attempts_made,
        first_failure_at.expect("a failed attempt set the round's first failure instant"),
        &first_failure,
        span,
    );
    // Not yet terminal: one failing round never stops a healthy service. Warnings
    // only — the durable block is filed once, when the window is exhausted and the
    // stop is decided.
    if !exhausted {
        warn!(
            db = %name,
            round_attempts = attempts_made,
            window_attempts = window.attempts,
            window_rounds = window.rounds,
            elapsed_ms = window.elapsed().as_millis(),
            "Genuine checkpoint failure with no completion — failure window not exhausted, continuing",
        );
        return;
    }
    // The window is exhausted: this is the stop. The report is written
    // (synchronously) before the drain, and the round's repair outcome has no say in
    // it. Nothing restarts the process after it and the app shows no reason of its
    // own for it, so this record and the log line below are the operator's only
    // trace.
    let report = build_failure_report(name, &failures, repair.as_ref(), &window, conn).await;
    let pointer = failure_record::recorded_pointer(
        "checkpoint failure",
        failure_record::record(root, &report.render()),
    );
    error!(
        db = %name,
        record = pointer.as_deref().unwrap_or(BLOCK_ON_STDERR),
        window_attempts = window.attempts,
        window_rounds = window.rounds,
        elapsed_ms = window.elapsed().as_millis(),
        "Genuine checkpoint failure with no completion across the failure window — recorded, \
         initiating graceful shutdown",
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
    window: &FailureWindow,
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
        // The environment note is the deciding round's: its failures are what the
        // block renders in full, and the window's opening cause is kept as text.
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
    let cause = format!("{first:#}");
    report = report.extra(format!("checkpoint error: {cause}"));
    for (attempt_no, e) in failures.iter().skip(1) {
        report = report.extra(format!("attempt {attempt_no} error: {e:#}"));
    }
    // The window's own cause and cumulative facts: this block is the only record of
    // the earlier failing rounds (they warn only), and what tells a reader whether
    // the failure was transient or persistent. The cause that opened the window is
    // carried only when it differs from this round's — usually it is the same chain,
    // and a duplicate buys nothing. The retry policy's stated limit belongs with the
    // facts, because the stop it decided is otherwise implicit.
    if window.first_failure != cause {
        report = report.extra(format!(
            "window first failure error: {}",
            window.first_failure
        ));
    }
    report = report.extra(window.summary());
    report = report.extra(
        "retry policy: a transient engine refusal cannot be told from a persistent one except by \
         whether a later attempt succeeds — a persistent refusal exhausts this window and the \
         service still stops, by design",
    );
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

    /// The retry pause the round's tests use: none, so a round's retries are instant.
    const TEST_PAUSE: Duration = Duration::ZERO;

    /// Whether the store `conn` has open currently carries an open failure window.
    fn window_open(conn: &Connection) -> bool {
        FAILURE_WINDOWS
            .lock()
            .unwrap_poison()
            .contains_key(&failure_window_key(conn))
    }

    /// [`CHECKPOINT_FAILURE_MIN_ROUNDS`] as the number of failing rounds a test runs
    /// to exhaust a store's window.
    fn window_rounds() -> usize {
        usize::try_from(CHECKPOINT_FAILURE_MIN_ROUNDS)
            .expect("the failure window's round floor fits a usize")
    }

    /// Run `rounds` failing periodic rounds for the store — every attempt answered
    /// with `error` — and return the total attempts made.
    async fn failing_rounds(
        name: &'static str,
        conn: &Connection,
        root: &Path,
        rounds: usize,
        error: &str,
        retry_pause: Duration,
        span: Duration,
    ) -> usize {
        let error = error.to_string();
        rounds_with_answers(name, conn, root, rounds, retry_pause, span, move |_| {
            let error = error.clone();
            async move { Attempted::Ran(Err(anyhow::anyhow!(error))) }
        })
        .await
    }

    /// Run `rounds` periodic rounds for the store, each attempt answered by
    /// `answer(attempt_no)` — what every test that varies the engine's answer per
    /// attempt drives its rounds through — and return the total attempts made.
    async fn rounds_with_answers<A, Fut>(
        name: &'static str,
        conn: &Connection,
        root: &Path,
        rounds: usize,
        retry_pause: Duration,
        span: Duration,
        mut answer: A,
    ) -> usize
    where
        A: FnMut(usize) -> Fut,
        Fut: Future<Output = Attempted> + Send,
    {
        let mut attempts = 0usize;
        for _ in 0..rounds {
            let mut attempt_no = 0usize;
            periodic_checkpoint_inner(name, conn, Some(root), retry_pause, span, || {
                attempt_no += 1;
                attempts += 1;
                answer(attempt_no)
            })
            .await;
        }
        attempts
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
    /// completion is the cure (it also closes the store's failure window, were one
    /// open), and the rebuild merely happens to be what made one possible.
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
        periodic_checkpoint_inner(
            "core",
            &conn,
            Some(tmp.path()),
            TEST_PAUSE,
            Duration::ZERO,
            || {
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
            },
        )
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

    /// The repair rebuilt the index and every attempt of every failing round still
    /// failed: once the window is exhausted the report is written to error.log with
    /// its real cause, and the service stops — a rebuilt index does not exempt the
    /// store, because no attempt completed a checkpoint.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_rebuilt_index_does_not_exempt_a_persistently_failing_round() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = a_store_with_a_corrupt_fts_index(tmp.path()).await;
        let store = &conn;

        let rounds = window_rounds();
        let total_attempts = rounds_with_answers(
            "core",
            store,
            tmp.path(),
            rounds,
            TEST_PAUSE,
            Duration::ZERO,
            |attempt_no| async move {
                // Each round's first attempt corrupts the index again: the round's own
                // repair is what rebuilds it, so the deciding round repairs a corrupted
                // index exactly as its predecessors did.
                if attempt_no == 1 {
                    store.execute_batch(&fts_corruption_ddl()).await.unwrap();
                    return Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure")));
                }
                Attempted::Ran(Err(anyhow::anyhow!("injected persistent failure")))
            },
        )
        .await;
        assert_eq!(
            total_attempts,
            CHECKPOINT_ATTEMPTS * rounds,
            "a persistent failure must run every attempt of every round before the window decides"
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

    /// A genuine failure, the round's repair on a store whose title FTS index was
    /// corrupted, and attempts that follow it folding the journal only part of the
    /// way: the round still counts as a failing one, and consecutive failing rounds
    /// stop the service — a partially folded attempt is not a completion, it does
    /// not cut the round's budget short, and the repair's outcome decides nothing.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_rebuilt_index_does_not_exempt_a_partial_retry() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = a_store_with_a_corrupt_fts_index(tmp.path()).await;

        let rounds = window_rounds();
        let total_attempts = rounds_with_answers(
            "core",
            &conn,
            tmp.path(),
            rounds,
            TEST_PAUSE,
            Duration::ZERO,
            |attempt_no| {
                let answer = if attempt_no == 1 {
                    Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure")))
                } else {
                    // A partially folded journal: not a completion, not a failure.
                    Attempted::Ran(Ok(CheckpointOutcome {
                        busy: false,
                        log_frames: 5,
                        checkpointed_frames: 1,
                    }))
                };
                async move { answer }
            },
        )
        .await;

        assert_eq!(
            total_attempts,
            CHECKPOINT_ATTEMPTS * rounds,
            "a partially folded attempt must not cut a failing round's budget short"
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
    /// the cure is the completion itself, since no failure window is open here (only a
    /// round that ends with a failure opens one) — whatever the round's repair
    /// returned.
    async fn a_round_cured_by_its_last_attempt(conn: &Connection, root: &Path) {
        let mut attempt_no = 0;
        periodic_checkpoint_inner("core", conn, Some(root), TEST_PAUSE, Duration::ZERO, || {
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

    /// A store with no ticket-title FTS index (repair is NotApplicable) still gets
    /// all three attempts on every failing round, and once the window is exhausted
    /// it writes error.log and begins the drain; an environment-caused checkpoint
    /// error is marked as such in the record.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn checkpoint_failure_without_fts_store_writes_error_log_and_drains() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let rounds = window_rounds();
        let attempts = failing_rounds(
            "core",
            &conn,
            tmp.path(),
            rounds,
            "no space left on device",
            TEST_PAUSE,
            Duration::ZERO,
        )
        .await;
        assert_eq!(
            attempts,
            CHECKPOINT_ATTEMPTS * rounds,
            "every store gets all three attempts of every failing round, indexed or not"
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
    /// three attempts on every failing round, and a persistent failure still stops
    /// the service once the window is exhausted: the attempts are never gated on
    /// what the repair applies or what it returns.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_healthy_index_store_gets_all_three_attempts_and_stops() {
        crate::shutdown::drain_clear();
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .expect("open the consolidated store");
        insert_fts_ticket(&conn, "t-1", "Important bug fix one").await;

        let rounds = window_rounds();
        let attempts = failing_rounds(
            "core",
            &conn,
            tmp.path(),
            rounds,
            "injected checkpoint failure",
            TEST_PAUSE,
            Duration::ZERO,
        )
        .await;
        assert_eq!(
            attempts,
            CHECKPOINT_ATTEMPTS * rounds,
            "an indexed store gets every attempt of every failing round too",
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
        periodic_checkpoint_inner(
            "core",
            &conn,
            Some(tmp.path()),
            TEST_PAUSE,
            Duration::ZERO,
            || {
                attempts += 1;
                async {
                    Attempted::Ran(Ok(CheckpointOutcome {
                        busy: false,
                        log_frames: 5,
                        checkpointed_frames: 1,
                    }))
                }
            },
        )
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
        periodic_checkpoint_inner(
            "core",
            &conn,
            Some(tmp.path()),
            TEST_PAUSE,
            Duration::ZERO,
            || {
                attempts += 1;
                async { Attempted::Ran(Err(blocked_checkpoint_error())) }
            },
        )
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
    /// own `attempt N error:` lines. Every failing round repeats the pattern until
    /// the window is exhausted.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_busy_attempt_before_a_failure_leaves_the_cause_line_unmarked() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        rounds_with_answers(
            "core",
            &conn,
            tmp.path(),
            window_rounds(),
            TEST_PAUSE,
            Duration::ZERO,
            |attempt_no| {
                let answer = if attempt_no == 1 {
                    Attempted::Ran(Err(blocked_checkpoint_error()))
                } else {
                    Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure")))
                };
                async move { answer }
            },
        )
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

    /// A genuine failure followed by attempts the engine answers busy still counts
    /// the round as failing: busy is not a completion, the round's budget runs to
    /// its end, and once the window is exhausted the round's failure is the record's
    /// cause and the service stops. Busy-ness never stops the service by itself, but
    /// it does not erase a genuine failure either.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_failure_followed_by_busy_attempts_still_stops() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let rounds = window_rounds();
        let attempts = rounds_with_answers(
            "core",
            &conn,
            tmp.path(),
            rounds,
            TEST_PAUSE,
            Duration::ZERO,
            |attempt_no| {
                let answer = if attempt_no == 1 {
                    Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure")))
                } else {
                    Attempted::Ran(Err(blocked_checkpoint_error()))
                };
                async move { answer }
            },
        )
        .await;

        assert_eq!(
            attempts,
            CHECKPOINT_ATTEMPTS * rounds,
            "a busy attempt must not cut a genuine failure's budget short — every attempt of every \
             failing round must run"
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
    /// that produced no error contributes no line. Every failing round repeats the
    /// pattern until the window is exhausted.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn the_report_numbers_each_failure_after_the_first() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        rounds_with_answers(
            "core",
            &conn,
            tmp.path(),
            window_rounds(),
            TEST_PAUSE,
            Duration::ZERO,
            |attempt_no| {
                let answer = match attempt_no {
                    1 => Attempted::Ran(Err(anyhow::anyhow!("injected first failure"))),
                    2 => Attempted::Ran(Err(blocked_checkpoint_error())),
                    _ => Attempted::Ran(Err(anyhow::anyhow!("injected third failure"))),
                };
                async move { answer }
            },
        )
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

    /// A panicking attempt is a failed attempt: every round makes all three
    /// attempts, files the panic as the reason, and once the window is exhausted the
    /// service drains.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_panicking_attempt_is_a_failed_attempt_and_stops_after_the_window() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let rounds = window_rounds();
        let attempts = rounds_with_answers(
            "core",
            &conn,
            tmp.path(),
            rounds,
            TEST_PAUSE,
            Duration::ZERO,
            |_| async {
                Attempted::Ran(guarded_checkpoint(async { panic!("injected attempt panic") }).await)
            },
        )
        .await;
        assert_eq!(
            attempts,
            CHECKPOINT_ATTEMPTS * rounds,
            "a panic is a failed attempt, so every attempt of every round must run"
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
        periodic_checkpoint_inner(name, &conn, Some(path), TEST_PAUSE, Duration::ZERO, || {
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
    /// stops — but the genuine failure of an earlier attempt in the same round still
    /// counts that round as failing, and consecutive failing rounds stop the
    /// service, so the stop rule is not weakened.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_refusal_on_a_retry_still_stops_on_the_rounds_real_failure() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let rounds = window_rounds();
        let attempts = rounds_with_answers(
            "core",
            &conn,
            tmp.path(),
            rounds,
            TEST_PAUSE,
            Duration::ZERO,
            |attempt_no| {
                let answer = if attempt_no == 1 {
                    Attempted::Ran(Err(anyhow::anyhow!("injected checkpoint failure")))
                } else {
                    Attempted::ShrinkRefused
                };
                async move { answer }
            },
        )
        .await;
        assert_eq!(
            attempts,
            2 * rounds,
            "every failing round must stop at the refusal instead of retrying forever — the total \
             is the failure and the refusal of each round"
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

    /// One failing round warns and never stops the service: it makes its whole
    /// attempt budget, leaves the store's window open, and files nothing — a single
    /// failing round is not evidence enough to stop a healthy service.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_single_failing_round_warns_and_never_stops() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let attempts = failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected checkpoint failure",
            TEST_PAUSE,
            Duration::ZERO,
        )
        .await;

        assert_eq!(
            attempts, CHECKPOINT_ATTEMPTS,
            "one failing round must still make its whole attempt budget"
        );
        assert!(
            !crate::shutdown::is_draining(),
            "one failing round must never stop the service"
        );
        assert!(
            window_open(&conn),
            "one failing round must open the store's failure window"
        );
        assert!(
            !tmp.path().join("error.log").exists(),
            "a window that is not exhausted must file no failure block"
        );
    }

    /// The second consecutive failing round exhausts the window (the fixture's span
    /// is zero) and stops the service with the window's cumulative facts: the
    /// deciding round's failures and the attempts and rounds the whole window took.
    /// The round's cause is the one that opened the window, so the record does not
    /// repeat it as the window's opening cause.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn the_second_consecutive_failing_round_stops_with_the_windows_cumulative_facts() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;

        let rounds = window_rounds();
        let attempts = failing_rounds(
            "core",
            &conn,
            tmp.path(),
            rounds,
            "injected checkpoint failure",
            TEST_PAUSE,
            Duration::ZERO,
        )
        .await;

        assert_eq!(
            attempts,
            CHECKPOINT_ATTEMPTS * rounds,
            "a persistent failure must run every attempt of every round before the window decides"
        );
        assert!(
            crate::shutdown::is_draining(),
            "an exhausted window must stop the service"
        );

        let body = std::fs::read_to_string(tmp.path().join("error.log")).expect("error.log");
        for needle in [
            "MahBot checkpoint failure",
            "checkpoint error: injected checkpoint failure",
            "retry policy:",
        ] {
            assert!(
                body.contains(needle),
                "error.log must contain {needle:?}: {body}"
            );
        }
        assert!(
            !body.contains("window first failure error:"),
            "the opening cause is not repeated when it is the deciding round's own: {body}"
        );
        let summary = format!(
            "failure window: {} attempts (blocked and partial ones included) over {} consecutive \
             failing rounds",
            CHECKPOINT_ATTEMPTS * rounds,
            rounds,
        );
        assert!(
            body.contains(&summary),
            "the record must carry the window's cumulative attempts and rounds — {summary:?}: {body}"
        );
        assert!(
            !window_open(&conn),
            "the exhausted window is spent: it must not be left for a later round to re-file"
        );
        crate::shutdown::drain_clear();
    }

    /// The terminal record carries the failure that opened the window when it is not
    /// the deciding round's own cause: the earlier failing rounds warn only, so this
    /// line is the only durable trace of what started the window.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn the_window_records_the_cause_that_opened_it() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;
        let (retry_pause, span) = (TEST_PAUSE, Duration::ZERO);

        // The first round opens the window with its own cause; the second exhausts it
        // with a different one.
        failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected first failure",
            retry_pause,
            span,
        )
        .await;
        failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected second failure",
            retry_pause,
            span,
        )
        .await;

        assert!(
            crate::shutdown::is_draining(),
            "the exhausted window must stop the service"
        );
        let body = std::fs::read_to_string(tmp.path().join("error.log")).expect("error.log");
        for needle in [
            "checkpoint error: injected second failure",
            "window first failure error: injected first failure",
        ] {
            assert!(
                body.contains(needle),
                "error.log must contain {needle:?}: {body}"
            );
        }
        crate::shutdown::drain_clear();
    }

    /// A completed checkpoint closes the store's failure window wherever it happens,
    /// and the next failing round opens a fresh one: that fresh window is one failing
    /// round, so the service keeps serving and files nothing.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_completed_checkpoint_closes_the_window_and_the_next_failure_starts_a_new_one() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;
        let (retry_pause, span) = (TEST_PAUSE, Duration::ZERO);

        failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected checkpoint failure",
            retry_pause,
            span,
        )
        .await;
        assert!(
            window_open(&conn),
            "the failing round must open the store's window"
        );

        // A round whose first attempt is a real, completing checkpoint: the completion
        // is what closes the window, not a retry.
        periodic_checkpoint_inner("core", &conn, Some(tmp.path()), retry_pause, span, || {
            let conn = conn.clone();
            async move { Attempted::Ran(conn.checkpoint_ungated().await) }
        })
        .await;
        assert!(
            !window_open(&conn),
            "a completed checkpoint must close the window"
        );
        assert!(
            !crate::shutdown::is_draining(),
            "a completion must keep the service serving"
        );
        assert!(
            !tmp.path().join("error.log").exists(),
            "a completion must not file a failure block"
        );

        failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected checkpoint failure",
            retry_pause,
            span,
        )
        .await;
        assert!(
            window_open(&conn),
            "the next failure must open a fresh window"
        );
        assert!(
            !crate::shutdown::is_draining(),
            "a fresh window is one failing round, so the service must keep serving"
        );
        assert!(
            !tmp.path().join("error.log").exists(),
            "a fresh, unexhausted window must file no failure block"
        );
    }

    /// A round that ended without a genuine failure closes the store's window:
    /// failing rounds separated by a round the engine answered busy are not
    /// consecutive, so the service keeps serving.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_round_that_did_not_fail_closes_the_window() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;
        let (retry_pause, span) = (TEST_PAUSE, Duration::ZERO);

        failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected checkpoint failure",
            retry_pause,
            span,
        )
        .await;
        assert!(
            window_open(&conn),
            "the failing round must open the store's window"
        );

        periodic_checkpoint_inner(
            "core",
            &conn,
            Some(tmp.path()),
            retry_pause,
            span,
            || async { Attempted::Ran(Err(blocked_checkpoint_error())) },
        )
        .await;
        assert!(
            !window_open(&conn),
            "a round that ended without a genuine failure must close the window"
        );

        failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected checkpoint failure",
            retry_pause,
            span,
        )
        .await;
        assert!(
            !crate::shutdown::is_draining(),
            "failing rounds separated by a round that did not fail are not consecutive, so the \
             fresh window must not stop the service"
        );
        assert!(
            !tmp.path().join("error.log").exists(),
            "an unexhausted window must file no failure block"
        );
    }

    /// The span floor alone blocks a stop the round floor would decide: rounds past
    /// the round floor but inside a span that has not elapsed keep the service
    /// serving, with the window still open and nothing filed.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn the_span_floor_alone_blocks_the_stop_the_round_floor_would_decide() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;
        let rounds = window_rounds() + 1;

        failing_rounds(
            "core",
            &conn,
            tmp.path(),
            rounds,
            "injected checkpoint failure",
            TEST_PAUSE,
            Duration::from_secs(3600),
        )
        .await;

        assert!(
            !crate::shutdown::is_draining(),
            "rounds past the round floor inside an unelapsed span must keep the service serving"
        );
        assert!(
            window_open(&conn),
            "the window must stay open until it has spanned both floors"
        );
        assert!(
            !tmp.path().join("error.log").exists(),
            "a window that has not spanned both floors must file no failure block"
        );
    }

    /// A window that outlives its span stops on the next failing round: the span
    /// floor is real elapsed time, not a round count.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_window_that_outlives_its_span_stops_on_the_next_failing_round() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;
        let (retry_pause, span) = (TEST_PAUSE, Duration::from_millis(100));

        failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected checkpoint failure",
            retry_pause,
            span,
        )
        .await;
        assert!(
            !crate::shutdown::is_draining(),
            "one failing round must never stop the service, whatever the span"
        );

        tokio::time::sleep(Duration::from_millis(150)).await;

        failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected checkpoint failure",
            retry_pause,
            span,
        )
        .await;
        assert!(
            crate::shutdown::is_draining(),
            "the window spans both floors once the span has really elapsed"
        );
        crate::shutdown::drain_clear();
    }

    /// A round the daemon's own drain cuts short decides nothing: the spaced retry is
    /// abandoned, so the round makes its first attempt only, leaves the store's
    /// window untouched — an interrupted round is not a failing one — and files
    /// nothing.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_round_cut_short_by_a_drain_decides_nothing() {
        crate::shutdown::drain_clear();
        let (tmp, conn) = temp_store("core").await;
        let (retry_pause, span) = (Duration::from_millis(10), Duration::ZERO);

        crate::shutdown::drain_begin();
        let attempts = failing_rounds(
            "core",
            &conn,
            tmp.path(),
            1,
            "injected checkpoint failure",
            retry_pause,
            span,
        )
        .await;

        assert_eq!(
            attempts, 1,
            "the drain must cut the spaced retry short after the round's first attempt"
        );
        assert!(
            !window_open(&conn),
            "an interrupted round is not a failing one, so the window must stay untouched"
        );
        assert!(
            !tmp.path().join("error.log").exists(),
            "a round that decided nothing must file no failure block"
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
