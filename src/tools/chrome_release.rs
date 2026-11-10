//! Chrome sessions and tab records runs leave behind: the ended-run releases, the live run's own
//! tab cap, and the per-agent tab record that cap counts from.
//!
//! chrome-use ≥1.5.101 preserves external Chrome tabs across daemon idle recycling, and a run
//! cannot close its own in its tail — that tail is skipped when its future is dropped — so every
//! run end queues its names from `RunEndCleanup`'s `Drop` and the `chrome-run-releases` task closes
//! them out of band through the browser extension ([`crate::tools::chrome_tabs::close_sessions`]),
//! not chrome-use's endpoint-bound `session stop`.
//!
//! A record ends only when the browser says every name is gone, when the record's routes cannot
//! close the rest ([`record_disposition`]), or when the cap evicts it — the record is lost, never a
//! tab. Leftovers no record names are reclaimed by a sweep of live `AGENT_TAB_PREFIX` groups
//! ([`reclaim_unfinished`]) that never runs on a timer and leaves [`ProtectedNamespaces`] alone.
//!
//! Reports name a run wherever durable state or [`EVICTED_RUNS`] does, and never blame the owner's
//! own tabs for what cannot be attributed ([`NO_OWNERSHIP_DOOR_REASON`]).

use super::chrome_daemon::cli_path;
use super::chrome_tabs::{self, RetryLeg, TabOutcome};
use crate::util::UnwrapPoison;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

/// Retry ladder for a leftover: doubles per attempt, capped at [`RELEASE_RETRY_CAP`].
const RELEASE_RETRY_BASE: Duration = Duration::from_secs(5);
const RELEASE_RETRY_CAP: Duration = Duration::from_mins(30);
/// Answers that this run's close left the tabs standing, before the close route concludes.
const MAX_LEFT_OPEN_ATTEMPTS: u32 = 3;
/// How long a held release waits, and the only bound on a run that never comes back: a restart
/// inside the window re-mints the hold from its own boot ([`restored_eligibility`]).
pub(crate) const RELEASE_HOLD_REDRIVEN_RUN: Duration = Duration::from_mins(30);
/// Which run ends hold their release back: a drain/shutdown cut and a workspace pause, after which
/// the same run is normally rebuilt under its own durable id, so its resumed segment must still
/// find its tabs. Nothing checks that a run is re-driven, so a run nothing comes back for pays the
/// hold as latency; `internal_cancel` is deliberately not held, which is why every attempt
/// re-checks per name. The hold is a floor: a namespace a non-terminalized job still names is left
/// alone anyway.
pub(crate) fn holds_run_end(classification: &str) -> bool {
    matches!(classification, "drain" | "shutdown" | "pause")
}

/// Floor on a restored record's eligibility: the runs the daemon re-drives at boot register after
/// the release task restores; a held record is re-minted to its whole hold instead.
const RELEASE_BOOT_GRACE: Duration = Duration::from_mins(1);
/// Durable record of the queue: a file, not a DB row, because the hand-off runs from a `Drop`.
const RELEASE_FILE_NAME: &str = "chrome-run-releases.json";
/// Hard cap on queued releases: a burst of runs ending while the browser side is down must not
/// accumulate without bound, so a record is evicted to make room ([`evict_oldest`]). Eviction
/// closes nothing and loses a record, never a tab — the reclaim sweep owns those tabs from its next
/// ask on and reports them itself, naming the run [`EVICTED_RUNS`] remembers.
const MAX_PENDING_RELEASES: usize = 256;
/// How long an evicted record's run is remembered for the sweep's report (see [`EVICTED_RUNS`]).
const EVICTED_RUN_TTL: Duration = Duration::from_hours(1);
/// Last-chance release budget on the shutdown/self-update path, below what one close pass may need
/// ([`crate::chrome::SESSION_STOP_TIMEOUT`]), so a slow pass is cut off and left for the next boot.
const SHUTDOWN_RELEASE_FLUSH_BUDGET: Duration = Duration::from_secs(10);

#[derive(Clone)]
struct PendingRunRelease {
    /// The identity records are merged by: a live run of the same agent id shares it.
    namespace: String,
    names: Vec<String>,
    attempts: u32,
    next_attempt_at: Instant,
    held: bool,
    run: String,
    /// Passes the browser answered for and left standing, concluded at [`MAX_LEFT_OPEN_ATTEMPTS`].
    left_open: u32,
}

/// One queued record as persisted: `Instant` is process-relative, so deadlines are epoch seconds.
#[derive(Serialize, Deserialize)]
struct PersistedRunRelease {
    namespace: String,
    names: Vec<String>,
    #[serde(default)]
    attempts: u32,
    /// Epoch seconds at which the record becomes eligible (`0` = now).
    #[serde(default)]
    next_attempt_at: u64,
    #[serde(default)]
    held: bool,
    /// `default` is mandatory on `run` and `left_open`: an old record file carries neither.
    #[serde(default)]
    run: String,
    #[serde(default)]
    left_open: u32,
}

/// A run whose record the cap evicted, so a report of its tabs can still name it.
struct EvictedRun {
    namespace: String,
    run: String,
    at: Instant,
}

static PENDING_RELEASES: OnceLock<Mutex<VecDeque<PendingRunRelease>>> = OnceLock::new();
/// The records passes in flight took out of the queue, tagged with their [`ReleasePass`] id.
/// [`persist_pending_releases`] writes them too, so a persist can never publish a file that loses a
/// batch a pass is mid-way through; each pass unparks exactly its own, since passes may overlap. An
/// entry outlives its pass only through a panic inside a finish closure, and it stays parked so the
/// next boot restores it.
static PARKED_RELEASES: OnceLock<Mutex<Vec<(u64, PendingRunRelease)>>> = OnceLock::new();
/// Source of [`ReleasePass`] ids, unique within one process.
static NEXT_PASS_ID: AtomicU64 = AtomicU64::new(0);
static RELEASE_WAKE: OnceLock<tokio::sync::Notify> = OnceLock::new();
/// Namespaces of the runs live RIGHT NOW, refcounted: a replacement run may overlap its
/// predecessor.
static LIVE_RUN_NAMESPACES: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();
/// When the sweep last asked, and how many asks went unanswered (see [`reclaim_due`]).
static RECLAIM: OnceLock<Mutex<ReclaimSchedule>> = OnceLock::new();
/// Namespaces the sweep concluded no route here closes, and when ([`stuck`]); in-process only.
static STUCK_NAMESPACES: OnceLock<Mutex<HashMap<String, Instant>>> = OnceLock::new();
/// Sessions a pass closed but had no budget to let go of, with their failed attempts.
static PENDING_FORGET: OnceLock<Mutex<Vec<(String, u32)>>> = OnceLock::new();
static EVICTED_RUNS: OnceLock<Mutex<VecDeque<EvictedRun>>> = OnceLock::new();

fn pending_releases() -> &'static Mutex<VecDeque<PendingRunRelease>> {
    PENDING_RELEASES.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn parked_releases() -> &'static Mutex<Vec<(u64, PendingRunRelease)>> {
    PARKED_RELEASES.get_or_init(|| Mutex::new(Vec::new()))
}

fn release_wake() -> &'static tokio::sync::Notify {
    RELEASE_WAKE.get_or_init(tokio::sync::Notify::new)
}

fn live_run_namespaces() -> &'static Mutex<HashMap<String, usize>> {
    LIVE_RUN_NAMESPACES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn evicted_runs() -> &'static Mutex<VecDeque<EvictedRun>> {
    EVICTED_RUNS.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Mark a run's namespace as live: an earlier record's release must never close a session a live
/// run re-attached. Empty namespaces (the shared non-agent instance) are ignored.
pub(crate) fn register_run_namespace(namespace: &str) {
    if namespace.is_empty() {
        return;
    }
    *live_run_namespaces()
        .lock()
        .unwrap_poison()
        .entry(namespace.to_string())
        .or_insert(0) += 1;
}

/// Drop one registration and wake the release task, so a record skipped as live is retried at once.
pub(crate) fn unregister_run_namespace(namespace: &str) {
    if namespace.is_empty() {
        return;
    }
    {
        let mut live = live_run_namespaces().lock().unwrap_poison();
        match live.get_mut(namespace) {
            Some(count) if *count > 1 => *count -= 1,
            Some(_) => {
                live.remove(namespace);
            }
            None => return, // never registered (or already deregistered)
        }
    }
    release_wake().notify_one();
}

fn run_namespace_is_live(namespace: &str) -> bool {
    !namespace.is_empty()
        && live_run_namespaces()
            .lock()
            .unwrap_poison()
            .contains_key(namespace)
}

/// Which of two eligibilities a merge keeps; the (deadline, hold) pair always travels whole.
#[derive(Clone, Copy)]
enum Eligibility {
    /// The incoming pair wins: a run end is the most recent statement about the run.
    RunEnd,
    /// The later deadline wins, so a pass putting records back cannot undercut an eligibility that
    /// arrived while it was in flight; a stale hold can win this way, by at most the hold.
    Retry,
}

impl Eligibility {
    fn keep(self, queued: (Instant, bool), incoming: (Instant, bool)) -> (Instant, bool) {
        match self {
            Self::RunEnd => incoming,
            // A retry keeps the queued pair unless the incoming one is later still — a tie must not
            // shorten what is already queued.
            Self::Retry if incoming.0 > queued.0 => incoming,
            Self::Retry => queued,
        }
    }
}

/// Fold `incoming` into `entry`: union of names, the higher counts, and the pair `keep` picks.
fn absorb(entry: &mut PendingRunRelease, incoming: PendingRunRelease, keep: Eligibility) {
    (entry.next_attempt_at, entry.held) = keep.keep(
        (entry.next_attempt_at, entry.held),
        (incoming.next_attempt_at, incoming.held),
    );
    entry.attempts = entry.attempts.max(incoming.attempts);
    entry.left_open = entry.left_open.max(incoming.left_open);
    if entry.run.is_empty() {
        entry.run = incoming.run;
    }
    // A membership set grown as the incoming names arrive, so no per-name linear scan.
    let mut seen: HashSet<String> = entry.names.iter().cloned().collect();
    for name in incoming.names {
        if seen.insert(name.clone()) {
            entry.names.push(name);
        }
    }
}

/// Whether a record is still inside its hold — see [`evict_oldest`] for what that protects.
fn within_hold(record: &PendingRunRelease, now: Instant) -> bool {
    record.held && record.next_attempt_at > now
}

/// The oldest record NOT inside a hold, or `None`: a held record's tabs may still be resumed.
fn evict_oldest(queue: &mut VecDeque<PendingRunRelease>) -> Option<PendingRunRelease> {
    let now = Instant::now();
    let index = queue.iter().position(|record| !within_hold(record, now))?;
    queue.remove(index)
}

/// Remember the run an evicted record came from, so the sweep's report can name it. Bounded like
/// the queue: entries past [`EVICTED_RUN_TTL`], and past the queue's own cap, go.
fn remember_evicted_run(namespace: &str, run: &str) {
    let now = Instant::now();
    let mut runs = evicted_runs().lock().unwrap_poison();
    runs.retain(|entry| now.saturating_duration_since(entry.at) < EVICTED_RUN_TTL);
    runs.push_back(EvictedRun {
        namespace: namespace.to_string(),
        run: run.to_string(),
        at: now,
    });
    while runs.len() > MAX_PENDING_RELEASES {
        runs.pop_front();
    }
}

/// The newest evicted run for `namespace` inside [`EVICTED_RUN_TTL`]: a namespace evicted twice
/// belongs to the later run.
fn evicted_run(namespace: &str) -> Option<String> {
    let now = Instant::now();
    evicted_runs()
        .lock()
        .unwrap_poison()
        .iter()
        .rev()
        .find(|entry| {
            entry.namespace == namespace
                && now.saturating_duration_since(entry.at) < EVICTED_RUN_TTL
        })
        .map(|entry| entry.run.clone())
}

/// Queue a record, merging into the record already queued for the same run — the union of the names
/// and the higher counts — instead of keeping two; `eligibility` resolves which pair wins.
fn merge_record(entry: PendingRunRelease, eligibility: Eligibility) {
    // A record without a namespace is never queued: records are merged, and matched against the
    // live runs, by it.
    if entry.namespace.is_empty() || entry.names.is_empty() {
        return;
    }
    let mut queue = pending_releases().lock().unwrap_poison();
    if let Some(existing) = queue.iter_mut().find(|e| e.namespace == entry.namespace) {
        absorb(existing, entry, eligibility);
        return;
    }
    let mut evicted: Vec<PendingRunRelease> = Vec::new();
    while queue.len() >= MAX_PENDING_RELEASES
        && let Some(record) = evict_oldest(&mut queue)
    {
        evicted.push(record);
    }
    queue.push_back(entry);
    drop(queue);
    for record in evicted {
        // Logged, not reported: the sweep owns these tabs from its next ask on and reports them
        // itself (see [`MAX_PENDING_RELEASES`]), with the run carried by the memory below. Armed
        // here because an eviction need not come from a run end.
        remember_evicted_run(&record.namespace, &record.run);
        arm_reclaim_at(Instant::now() + STUCK_REVISIT);
        info!(
            namespace = %record.namespace,
            run = %record.run,
            sessions = record.names.len(),
            "agent-run chrome release queue full — record evicted; the reclaim sweep owns its tabs"
        );
    }
}

/// Hand a run's ended-session record over for release. `held` is [`holds_run_end`]'s verdict; `run`
/// is the agent id, carried so a report can name the run. Sync and panic-free so it can run from a
/// `Drop`, and durable: the write below is what makes the release outlive this process. The names
/// are snapshotted here (`ChromeTool` records a session before dispatching), and a later run of the
/// same durable id re-attaches them, which the live-namespace guard covers.
pub(crate) fn queue_run_session_release(
    sessions: &super::chrome::ChromeRunSessions,
    held: bool,
    run: &str,
) {
    queue_run_session_release_after(
        sessions,
        if held {
            RELEASE_HOLD_REDRIVEN_RUN
        } else {
            Duration::ZERO
        },
        run,
    );
}

/// [`queue_run_session_release`] with the hold spelled out, for the tests that need another.
fn queue_run_session_release_after(
    sessions: &super::chrome::ChromeRunSessions,
    hold: Duration,
    run: &str,
) {
    let names = sessions.snapshot();
    if names.is_empty() {
        return; // most runs never touch chrome — nothing to queue
    }
    merge_record(
        PendingRunRelease {
            namespace: sessions.namespace().to_string(),
            names,
            attempts: 0,
            next_attempt_at: Instant::now() + hold,
            held: !hold.is_zero(),
            run: run.to_string(),
            left_open: 0,
        },
        Eligibility::RunEnd,
    );
    persist_pending_releases();
    // The run end is what brings a crashed run's leftover to light — a crash leaves exactly the
    // names no record has — and the pass that closes this record has already opened the session the
    // sweep reads in. A run that queued nothing must not arm one.
    schedule_reclaim_now();
    release_wake().notify_one();
}

/// Restore the records a previous process left queued, once, at the start of
/// [`run_session_release_queue`]: a pending release is retried rather than lost, a held record
/// survives the restart it waits for with its attempt count, and an unreadable file is ignored.
fn restore_pending_releases() {
    let Some(path) = release_settings().store else {
        return;
    };
    let Ok(json) = std::fs::read_to_string(&path) else {
        return;
    };
    let records: Vec<PersistedRunRelease> = match serde_json::from_str(&json) {
        Ok(records) => records,
        Err(error) => {
            info!(
                path = %path.display(),
                %error,
                "agent-run chrome release file unreadable — ignoring it"
            );
            return;
        }
    };
    let mut restored = 0usize;
    for record in records {
        // What [`merge_record`] would refuse is neither restored nor counted: it is no work this
        // process owes the browser a read for.
        if record.names.is_empty() || record.namespace.is_empty() {
            continue;
        }
        merge_record(
            PendingRunRelease {
                namespace: record.namespace,
                names: record.names,
                attempts: record.attempts,
                next_attempt_at: restored_eligibility(record.next_attempt_at, record.held),
                held: record.held,
                run: record.run,
                left_open: record.left_open,
            },
            // `Retry`, not `RunEnd`: a pre-dedupe file with two entries for one namespace keeps the
            // later pair, i.e. the hold.
            Eligibility::Retry,
        );
        restored += 1;
    }
    if restored > 0 {
        debug!(
            restored,
            "agent-run chrome releases restored from the previous process"
        );
        // Work a previous process left queued is exactly what the sweep is owed a read for,
        // including the names no record carries.
        schedule_reclaim_now();
        release_wake().notify_one();
    }
}

/// Drop the per-agent tab records no live run, no held release record and no resumable job claims
/// any more ([`crate::tools::chrome_tab_ledger`]). One durable read per boot, and fail-closed: a
/// read that could not be made prunes nothing, since deleting a record is not a step a missing
/// answer may authorize.
async fn prune_tab_records() {
    let Some(durable) = durable_resume_namespaces_checked().await else {
        return;
    };
    let protected = ProtectedNamespaces::snapshot(&durable);
    // A namespace is its own prefix, so `contains` answers whether anything still claims it.
    crate::tools::chrome_tab_ledger::prune(|namespace| protected.contains(namespace));
}

/// Write the whole queue — plus what the passes in flight hold ([`PARKED_RELEASES`]) — to the
/// record file (compact JSON, atomic tmp+rename), both locks held across the write because two
/// persists sharing one `.json.tmp` could publish a torn file. Fail-open: no storage root, a failed
/// write or a nameless record leaves the previous file in place, never an empty one that would read
/// as a drained queue.
fn persist_pending_releases() {
    let Some(path) = release_settings().store else {
        return;
    };
    let parked = parked_releases().lock().unwrap_poison();
    let queue = pending_releases().lock().unwrap_poison();
    // Indexed by namespace rather than searched per record: the fold runs over the whole queue
    // every pass, which is where the queue's size is paid for quadratically.
    let mut folded: Vec<PendingRunRelease> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for record in queue.iter().chain(parked.iter().map(|(_, record)| record)) {
        if let Some(at) = index.get(&record.namespace).copied() {
            absorb(&mut folded[at], record.clone(), Eligibility::Retry);
        } else {
            index.insert(record.namespace.clone(), folded.len());
            folded.push(record.clone());
        }
    }
    let records: Vec<PersistedRunRelease> = folded
        .iter()
        .filter(|entry| !entry.names.is_empty())
        .map(|entry| PersistedRunRelease {
            namespace: entry.namespace.clone(),
            names: entry.names.clone(),
            attempts: entry.attempts,
            next_attempt_at: eligibility_deadline(entry.next_attempt_at),
            held: entry.held,
            run: entry.run.clone(),
            left_open: entry.left_open,
        })
        .collect();
    let json = match serde_json::to_string(&records) {
        Ok(json) => json,
        Err(err) => {
            // Never fall back to an empty payload: one published over the live file would read as a
            // drained queue.
            warn!(error = %err, "agent-run chrome release record not written: serialization failed");
            return;
        }
    };
    if let Err(err) = crate::util::write_json_record(&path, &json) {
        warn!(error = %err, path = %path.display(), "agent-run chrome release record not written");
    }
}

fn epoch_secs() -> u64 {
    crate::util::unix_millis() / 1000
}

/// Absolute epoch-seconds deadline for a record's remaining eligibility — what the file stores,
/// since an `Instant` dies with the process that minted it.
fn eligibility_deadline(next_attempt_at: Instant) -> u64 {
    epoch_secs().saturating_add(
        next_attempt_at
            .saturating_duration_since(Instant::now())
            .as_secs(),
    )
}

/// A persisted record as this process's eligibility instant. A held record re-mints its whole hold
/// from THIS boot; every other is floored at the boot grace and clamped to the ladder's cap, since
/// a further-out deadline would panic the add and take the release task down for the whole boot.
fn restored_eligibility(deadline: u64, held: bool) -> Instant {
    if held {
        return Instant::now() + RELEASE_HOLD_REDRIVEN_RUN;
    }
    let remaining =
        Duration::from_secs(deadline.saturating_sub(epoch_secs())).min(RELEASE_RETRY_CAP);
    Instant::now() + release_settings().boot_grace.max(remaining)
}

/// Release-queue driver (spawned as `chrome-run-releases`), restoring what a previous process left
/// queued before it takes its first pass.
pub async fn run_session_release_queue() {
    release_queue(crate::shutdown::shutdown_token()).await;
}

/// The loop against an explicit token, so a test can drive the real entry point: wake on an
/// enqueue, on a live run's tracker going away, or on the next deadline, whichever comes first. It
/// does not gate passes on `shutdown::aborting()` — every pass releases runs that already ended, so
/// there is nothing a drain protects from it.
async fn release_queue(shutdown: CancellationToken) {
    restore_pending_releases();
    // The restored records are in the queue above, so their namespaces count as protected here: the
    // boot prune must not drop the tab record of a release still waiting on the browser.
    prune_tab_records().await;
    loop {
        tokio::select! {
            () = wait_for_release_work() => {}
            () = release_wake().notified() => {}
            () = shutdown.cancelled() => break,
        }
        release_due().await;
    }
}

/// The driver's wait: until the next deadline, or forever when there is none — the loop is then
/// woken by an enqueue, a live run's tracker going away, or shutdown.
async fn wait_for_release_work() {
    match next_release_deadline() {
        Some(deadline) => {
            tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
        }
        None => std::future::pending().await,
    }
}

/// When the release task must run again: the earliest queued retry not owned by a live run, the
/// sweep's scheduled ask ([`ReclaimSchedule::next_at`]) and the let-go queue's retry, whichever
/// comes first; `None` when none exists, and the loop then waits on a wake alone. There is no
/// standing interval — each is an event's ask or a leftover's own retry, spaced by the same capped
/// ladder.
fn next_release_deadline() -> Option<Instant> {
    let sweep = reclaim_schedule().lock().unwrap_poison().next_at;
    pending_releases()
        .lock()
        .unwrap_poison()
        .iter()
        .filter(|entry| !run_namespace_is_live(&entry.namespace))
        .map(|entry| entry.next_attempt_at)
        .min()
        .into_iter()
        .chain(sweep)
        .chain(forget_retry_at())
        .min()
}

/// Take the records a pass may attempt: those whose hold or backoff elapsed, skipping any whose
/// namespace a live run still owns ([`register_run_namespace`]), and any durable state still names
/// for a resumable job ([`ProtectedNamespaces`]).
fn take_releasable() -> Vec<PendingRunRelease> {
    let now = Instant::now();
    let mut queue = pending_releases().lock().unwrap_poison();
    let mut taken = Vec::new();
    let mut waiting = VecDeque::with_capacity(queue.len());
    while let Some(mut entry) = queue.pop_front() {
        let releasable = !run_namespace_is_live(&entry.namespace) && entry.next_attempt_at <= now;
        if releasable {
            // A held record taken here has spent its hold, so it goes back ordinary: from now on it
            // is a rung of the retry ladder and nothing protects it as if a run could return.
            entry.held = false;
            taken.push(entry);
        } else {
            waiting.push_back(entry);
        }
    }
    *queue = waiting;
    taken
}

struct RecordDisposition {
    /// Names the browser did not settle (`Retry`) or the pass never asked about (no binary to ask):
    /// the record stays queued for exactly these.
    retry: Vec<String>,
    /// Retried names the browser ANSWERED for and did not remove ([`RetryLeg::LeftOpen`]) — kept
    /// apart because their answer is what the record counts and reports.
    left_open: Vec<String>,
    /// Names the record is done with: not the extension's to remove, or left standing
    /// [`MAX_LEFT_OPEN_ATTEMPTS`] times. Reported with the run, then dropped — the reclaim sweep
    /// keeps examining their group at [`STUCK_REVISIT`].
    unclosable: Vec<String>,
}

/// Decide one record's fate from the pass's answers: a name settled as gone is dropped, one that is
/// not the extension's to remove is a leftover to report, one answered for and left open is retried
/// and reported, and anything else stays queued. A record's own count concludes the left-open
/// names: at most one such answer per pass, at [`MAX_LEFT_OPEN_ATTEMPTS`], never reset.
fn record_disposition(
    entry: &mut PendingRunRelease,
    outcomes: &HashMap<String, TabOutcome>,
) -> RecordDisposition {
    // One left-open answer per pass, however many of the record's names were answered that way:
    // what is counted is the browser's repeated answer about this run's close.
    if entry.names.iter().any(|name| {
        matches!(
            outcomes.get(name),
            Some(TabOutcome::Retry(RetryLeg::LeftOpen))
        )
    }) {
        entry.left_open = entry.left_open.saturating_add(1);
    }
    let concluded = entry.left_open >= MAX_LEFT_OPEN_ATTEMPTS;
    let mut retry = Vec::new();
    let mut left_open = Vec::new();
    let mut unclosable = Vec::new();
    for name in &entry.names {
        match outcomes.get(name) {
            Some(TabOutcome::Gone) => {}
            Some(TabOutcome::Unclosable) => unclosable.push(name.clone()),
            // The browser has answered the same way [`MAX_LEFT_OPEN_ATTEMPTS`] times: this route is
            // not going to close them, so they conclude like any other leftover.
            Some(TabOutcome::Retry(RetryLeg::LeftOpen)) if concluded => {
                unclosable.push(name.clone());
            }
            Some(TabOutcome::Retry(RetryLeg::LeftOpen)) => {
                left_open.push(name.clone());
                retry.push(name.clone());
            }
            Some(TabOutcome::Retry(RetryLeg::Silent)) | None => retry.push(name.clone()),
        }
    }
    RecordDisposition {
        retry,
        left_open,
        unclosable,
    }
}

/// The records one release pass took out of the queue, parked ([`PARKED_RELEASES`]) under the
/// pass's id so the durable file still describes them: a record out of both would be stranded — the
/// failure this mechanism exists to prevent. `Drop` puts back whatever it did not reconcile.
struct ReleasePass {
    /// This pass's identity in [`PARKED_RELEASES`]: it unparks its own records, never a peer's.
    id: u64,
    records: Vec<PendingRunRelease>,
    /// Namespace `finish` holds right now, so a panic inside it leaves exactly that record parked.
    in_flight: Option<String>,
    reconciled: bool,
}

impl ReleasePass {
    /// The records a pass may attempt, or `None` when nothing is due. They stay parked while the
    /// pass holds them out of the queue, and the parked lock is held across the pop too, so a
    /// persist never sees them in neither.
    fn take() -> Option<Self> {
        let mut parked = parked_releases().lock().unwrap_poison();
        let records = take_releasable();
        if records.is_empty() {
            return None;
        }
        let id = NEXT_PASS_ID.fetch_add(1, Ordering::Relaxed);
        parked.extend(records.iter().cloned().map(|record| (id, record)));
        Some(Self {
            id,
            records,
            in_flight: None,
            reconciled: false,
        })
    }

    /// Hand every record to `finish` with the pass's own answers, then disarm; whatever a panic
    /// left — including the record a panicking `finish` held — is dealt with by `Drop`.
    fn reconcile(
        &mut self,
        outcomes: &HashMap<String, TabOutcome>,
        mut finish: impl FnMut(PendingRunRelease, RecordDisposition),
    ) {
        // Popped one at a time, so a panic in `finish` still leaves the records it never reached
        // for the `Drop` guard. `finish` must not panic: only the parking keeps the record alive.
        while let Some(mut entry) = self.records.pop() {
            self.in_flight = Some(entry.namespace.clone());
            let disposition = record_disposition(&mut entry, outcomes);
            finish(entry, disposition);
            self.in_flight = None;
        }
        self.reconciled = true;
        self.unpark(None);
    }

    /// Stop parking this pass's records: the ones `finish` dropped are gone and the ones it
    /// re-queued are back in the queue, which is what the file carries. `keep` is the namespace a
    /// mid-panic `finish` held, the one record that must stay parked.
    fn unpark(&self, keep: Option<&str>) {
        parked_releases()
            .lock()
            .unwrap_poison()
            .retain(|(pass, record)| *pass != self.id || keep == Some(record.namespace.as_str()));
    }
}

impl Drop for ReleasePass {
    fn drop(&mut self) {
        if self.reconciled {
            return;
        }
        // The pass never finished: every record not yet handed to `finish` is put back (`Retry`
        // cannot shorten an eligibility that arrived meanwhile) and unparked, so the file carries
        // it again.
        for entry in self.records.drain(..) {
            merge_record(entry, Eligibility::Retry);
        }
        let in_flight = self.in_flight.take();
        self.unpark(in_flight.as_deref());
        persist_pending_releases();
    }
}

/// One pass over the due records plus the reclaim sweep: [`attempt_pass`] does the work, this only
/// collects the sweep's names, reads the durable resume set once and reports what stayed open.
async fn release_due() {
    let cli = release_cli();
    let deadline = Instant::now() + release_settings().attempt_timeout;
    let pass = ReleasePass::take();
    let taken = pass.is_some();
    let sweep_due = reclaim_due();
    // The one durable read of the pass, and only when something will use it: an idle wake is a
    // no-op.
    let durable = if taken || sweep_due {
        durable_resume_namespaces().await
    } else {
        Vec::new()
    };
    let mut protected = ProtectedNamespaces::snapshot(&durable);
    // The session this pass runs its browser reads in, decided once for the sweep and the close
    // alike.
    let session = read_session(pass.as_ref());
    let ask = reclaim_names(cli.as_deref(), &session, &protected, deadline, sweep_due).await;
    let issues = attempt_pass(
        cli.as_deref(),
        pass,
        &session,
        &ask,
        &mut protected,
        deadline,
    )
    .await;
    report_issues(issues).await;
}

/// The session one pass runs its browser reads in: a name the work itself already has, so a pass
/// with one opens no page of its own. The records the pass took come first — those are the tabs
/// being closed — then the first name any queued or parked record carries, which is the run's own
/// session. A page our read opens in that group is removed by the record's own close, unless the
/// namespace is protected then. [`OWN_SESSION`] is what is left for a pass with no name to borrow:
/// our own scratch group.
fn read_session(pass: Option<&ReleasePass>) -> String {
    if let Some(name) = pass
        .and_then(|pass| pass.records.first())
        .and_then(|record| record.names.first())
    {
        return name.clone();
    }
    let parked = parked_releases().lock().unwrap_poison();
    let queued = pending_releases().lock().unwrap_poison();
    parked
        .iter()
        .map(|(_, record)| record)
        .chain(queued.iter())
        .find_map(|record| record.names.first())
        .cloned()
        .unwrap_or_else(|| crate::tools::chrome_tabs::OWN_SESSION.to_string())
}

/// Close the browser's tabs for `names`, except the ones a run may own again right now. `protected`
/// is refreshed here with the live map and the held records as they are immediately before the
/// close: the sweep above spends a browser read (seconds), and a run that registers in that window
/// owns its tabs. (`close_sessions` re-asks at each act — see
/// [`crate::tools::chrome_tabs::OwnedAgain`].)
async fn close_unprotected(
    cli: Option<&Path>,
    names: &[String],
    session: &str,
    protected: &mut ProtectedNamespaces,
    deadline: Instant,
) -> HashMap<String, TabOutcome> {
    let Some(cli) = cli else {
        return HashMap::new();
    };
    protected.refresh();
    let unprotected: Vec<String> = names
        .iter()
        .filter(|name| !protected.contains(name))
        .cloned()
        .collect();
    // What the close asks right before each act: the live map and the held records as they are
    // then. The durable half is what the pass snapshotted once and cannot change inside one pass.
    let owned_again = ProtectedNamespaces::protects_live_or_held;
    chrome_tabs::close_sessions(cli, &unprotected, session, deadline, &owned_again)
        .await
        .into_iter()
        .collect()
}

/// Close one tab of a LIVE agent run because the run went over its per-agent tab cap, and say
/// whether the browser confirmed the group is gone. Only `session` is asked for and matched
/// exactly, so no other run's tabs and none of the owner's can be touched; the caller's namespace
/// is exempt from the live-run guard, or every eviction would be a no-op. Off the agent's path,
/// bounded by [`ATTEMPT_TIMEOUT`].
pub(crate) async fn evict_agent_tab(namespace: &str, session: &str, read_session: &str) -> bool {
    let Some(cli) = release_cli() else {
        return false;
    };
    let deadline = Instant::now() + release_settings().attempt_timeout;
    // The run's own namespace is ours to close; every other live run's and every held record's
    // stays protected.
    let owned_again = |name: &str| {
        !names_a_namespace(namespace, name) && ProtectedNamespaces::protects_live_or_held(name)
    };
    let name = session.to_string();
    let outcomes = chrome_tabs::close_sessions(
        &cli,
        std::slice::from_ref(&name),
        read_session,
        deadline,
        &owned_again,
    )
    .await;
    let gone = outcomes
        .first()
        .is_some_and(|(_, outcome)| matches!(outcome, chrome_tabs::TabOutcome::Gone));
    if gone {
        // The group is gone, but chrome-use's session record still pins the tab it created, and the
        // name must stay good for a new page: retire the record, or a later `open` on the name
        // fails on a target the relay no longer holds. A stop that does not settle is asked once
        // more — this runs off the agent's path, so the retry costs nothing. Past that, the run's
        // own release retires the record; the let-go queue cannot stop a live run's session.
        for _ in 0..2 {
            let outcome = chrome_tabs::forget_settled_sessions(
                &cli,
                std::slice::from_ref(&name),
                // The eviction's own deadline, not a fresh one: a stop is spawned with the whole
                // bound only when the budget still holds it, so `tried` keeps meaning the helper
                // answered.
                deadline,
                &|_: &str| false,
            )
            .await;
            if outcome.tried.is_empty() && outcome.deferred.is_empty() {
                break;
            }
        }
    }
    gone
}

/// Let go of the sessions earlier passes closed but had no budget to forget, plus `just_settled`,
/// and keep what must wait: chrome-use's per-session bookkeeping, this module's own scratch session
/// when its let-go was not confirmed, and never a run's tab.
///
/// [`LetGoOutcome`]'s classes decide: a stop not settled is charged another attempt, one given no
/// whole bound keeps its rung, one a run owns again leaves the queue (its own end hands over its
/// release), and [`chrome_tabs::OWN_SESSION`] waits for the call holding it and is dropped by
/// neither bound. The wait is the highest rung held, floored at one whole stop when anything was
/// deferred ([`BUSY_FORGET_FLOOR`]). The queue is never taken out of its slot, so an overlapping
/// pass takes no names with it.
async fn forget_settled(cli: &Path, just_settled: &[String], deadline: Instant) {
    let mut todo: Vec<(String, u32)> = pending_forget().lock().unwrap_poison().clone();
    // The names this pass takes out of the live queue, read BEFORE the cap below trims them: a
    // queued name the cap drops must not be written back by the merge at the end, or the cap would
    // be undone and the queue would churn instead of settling on the oldest names.
    let snapshot: HashSet<String> = todo.iter().map(|(name, _)| name.clone()).collect();
    // First-seen wins, so a name already waiting keeps the attempts it has run up.
    todo.extend(just_settled.iter().map(|name| (name.clone(), 0)));
    dedupe_and_bound_forgotten(&mut todo);
    if todo.is_empty() {
        // Nothing to try. Only a queue that is still empty loses its retry — a pass that overlapped
        // this one armed its own for whatever it queued.
        let queue = pending_forget().lock().unwrap_poison();
        if queue.is_empty() {
            *forget_retry().lock().unwrap_poison() = None;
        }
        return;
    }
    let names: Vec<String> = todo.iter().map(|(name, _)| name.clone()).collect();
    // Asked at each act inside: the run that owns a session again is what the stop would end, and a
    // let-go step must never race one.
    let owned_again = ProtectedNamespaces::protects_live_or_held;
    let left = chrome_tabs::forget_settled_sessions(cli, &names, deadline, &owned_again).await;
    let mut kept: Vec<(String, u32)> = Vec::new();
    for (name, attempts) in todo {
        if left.owned.contains(&name) {
            // A name a run owns again leaves this queue instead of climbing it: its own end hands
            // over its own release, so nothing here is waiting for us.
            continue;
        }
        if left.deferred.contains(&name) || left.busy.contains(&name) {
            // Nothing was asked of the helper — no whole bound fitted, or a call of ours held the
            // session — so nothing is charged and the rung stays as it is.
            kept.push((name, attempts));
            continue;
        }
        if !left.tried.contains(&name) {
            continue; // let go: the helper stopped the session and dropped its record
        }
        let attempts = attempts.saturating_add(1);
        // The one name the bound never drops is this module's own scratch session (see
        // [`MAX_FORGET_ATTEMPTS`]).
        if attempts >= MAX_FORGET_ATTEMPTS && name != chrome_tabs::OWN_SESSION {
            info!(
                session = %name,
                attempts,
                "a settled chrome session could not be let go — dropped from the let-go queue \
                 with chrome-use's own per-session record left behind; a pass that still owes \
                 this session's let-go queues it again"
            );
            continue;
        }
        kept.push((name, attempts));
    }
    // What is left drives its own retry ([`forget_retry_at`]): a session the helper would not stop
    // is tried again without an event arming one, and once nothing is left nothing is scheduled.
    let deferred_behind_call = !left.busy.is_empty();
    let mut queue = pending_forget().lock().unwrap_poison();
    // What the pass leaves: the names it held, minus what it let go or found a run owning again,
    // plus whatever an overlapping pass queued while the stops ran — the snapshot tells the two
    // apart, so the cap's own dropped names are not added back. The queue was never taken out, so
    // nothing else may be removed, and a count only ever climbs.
    let mut merged = kept;
    merged.extend(
        queue
            .iter()
            .filter(|(name, _)| !snapshot.contains(name))
            .cloned(),
    );
    for (name, attempts) in &mut merged {
        if let Some((_, live)) = queue.iter().find(|(queued, _)| queued == name) {
            *attempts = (*attempts).max(*live);
        }
    }
    *queue = merged;
    dedupe_and_bound_forgotten(&mut queue);
    let most_attempts = queue.iter().map(|(_, attempts)| *attempts).max();
    let retry_at = most_attempts.map(|attempts| Instant::now() + release_backoff(attempts));
    *forget_retry().lock().unwrap_poison() = if deferred_behind_call {
        // Nothing is owed until the call holding the session is done, and waking at the base gap
        // would only bring the driver back to defer again (see [`BUSY_FORGET_FLOOR`]).
        let floor = Instant::now() + BUSY_FORGET_FLOOR;
        Some(retry_at.map_or(floor, |at| at.max(floor)))
    } else {
        retry_at
    };
}

/// Put `name` in the let-go queue for the driver's next attempt, keeping the place and attempts a
/// name already has, and arm the queue's own retry deadline so a name queued by a pass whose tail
/// runs out still has an attempt behind it. While no chrome-use binary resolves no pass runs the
/// queue at all, so such a name waits for the next event — all a pass with no binary could do
/// anyway.
fn queue_forget(name: &str) {
    let mut queue = pending_forget().lock().unwrap_poison();
    if queue.iter().any(|(queued, _)| queued == name) {
        return;
    }
    queue.push((name.to_string(), 0));
    dedupe_and_bound_forgotten(&mut queue);
    *forget_retry().lock().unwrap_poison() = Some(Instant::now() + release_backoff(0));
}

/// When the let-go queue must be tried again; `None` while it is empty — a queue whose work is done
/// schedules nothing.
static FORGET_RETRY: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();

fn forget_retry() -> &'static Mutex<Option<Instant>> {
    FORGET_RETRY.get_or_init(|| Mutex::new(None))
}

/// The queue's own retry deadline, as [`next_release_deadline`] reads it: `None` while no
/// chrome-use binary is resolvable, because no pass can run the queue then. A deadline left over
/// from before the binary went away (a self-update swap, a removed install) would otherwise read as
/// due for the rest of the process and spin the driver. The stored deadline stays where it is, and
/// nothing wakes the driver for this alone.
fn forget_retry_at() -> Option<Instant> {
    release_cli()?;
    *forget_retry().lock().unwrap_poison()
}

/// Cap on the sessions waiting to be let go: a burst the helper will not stop must not grow the
/// queue without bound, so past the cap the oldest names go — chrome-use's own `~/.chrome-use`
/// bookkeeping, never a run's tab and never this module's own scratch session. Each stop is a
/// session daemon being shut down, so the queue is what keeps a burst waiting its turn.
const MAX_PENDING_FORGET: usize = 64;
/// Floor on the wake the queue arms when a pass found a call of ours still running in a name's
/// session ([`crate::tools::chrome_tabs::LetGoOutcome::busy`]): the pass holding it spends one
/// whole `session stop` right after, so waking at the base gap would only defer again.
const BUSY_FORGET_FLOOR: Duration = crate::chrome::SESSION_STOP_TIMEOUT;
/// How many attempts may fail to let one session go before it leaves the queue with one `info!`.
/// That trace is not a row the Issues view is owed — the scratch session's own row is filed by
/// `chrome_tabs`'s caller. [`chrome_tabs::OWN_SESSION`] is the one name the bound never drops: what
/// it leaves is a group in the owner's strip, so the queue's own deadline retries it.
const MAX_FORGET_ATTEMPTS: u32 = 5;

/// Keep `names` in first-seen order without repeats, bounded by [`MAX_PENDING_FORGET`]: what the
/// cap drops is chrome-use's per-session bookkeeping — never a run's tab.
fn dedupe_and_bound_forgotten(names: &mut Vec<(String, u32)>) {
    let mut seen: HashSet<String> = HashSet::with_capacity(names.len());
    names.retain(|(name, _)| seen.insert(name.clone()));
    let excess = names.len().saturating_sub(MAX_PENDING_FORGET);
    if excess == 0 {
        return;
    }
    let mut dropped = 0usize;
    names.retain(|(name, _)| {
        // The product's own scratch session is never the name the cap drops — one more goes
        // instead.
        if dropped == excess || name == chrome_tabs::OWN_SESSION {
            return true;
        }
        dropped += 1;
        false
    });
    debug!(
        dropped,
        "settled chrome sessions past the let-go cap were dropped — chrome-use records left \
         behind by it, never a run's tab"
    );
}

fn pending_forget() -> &'static Mutex<Vec<(String, u32)>> {
    PENDING_FORGET.get_or_init(|| Mutex::new(Vec::new()))
}

/// The names one pass is about: the reclaim set first, then every name its records still carry,
/// once.
fn pass_names(ask: &ReclaimAsk, pass: Option<&ReleasePass>) -> Vec<String> {
    let mut names = ask.names.clone();
    if let Some(pass) = pass {
        let mut seen: HashSet<String> = names.iter().cloned().collect();
        for name in record_names(pass) {
            if seen.insert(name.clone()) {
                names.push(name);
            }
        }
    }
    names
}

/// Every name the pass's records carry, in first-seen order.
fn record_names(pass: &ReleasePass) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for record in &pass.records {
        for name in &record.names {
            if seen.insert(name.clone()) {
                names.push(name.clone());
            }
        }
    }
    names
}

/// What the reclaim sweep's names came to: how many it closed, and the reports the rest owe. A
/// reclaimed group belongs to no record, so its namespace comes from the group's own session name
/// and its run from [`evicted_run`]. A name the browser did not settle is remembered as stuck with
/// its revisit armed, and an ask that answered nothing keeps one armed too. All answered leaves the
/// ladder at its floor.
fn reclaim_issues(
    reclaim: &[String],
    answered: bool,
    outcomes: &HashMap<String, TabOutcome>,
) -> (usize, Vec<Issue>) {
    let mut reclaimed = 0usize;
    let mut issues = Vec::new();
    let mut silent = false;
    for name in reclaim {
        let Some(outcome) = outcomes.get(name) else {
            continue;
        };
        let namespace = crate::tools::chrome::session_namespace(name).to_string();
        match outcome {
            TabOutcome::Gone => {
                reclaimed += 1;
                stuck_namespaces().lock().unwrap_poison().remove(&namespace);
            }
            // No route here closes these tabs: reported once, then skipped for a while.
            TabOutcome::Unclosable => {
                issues.push(reclaimed_issue(name, &namespace, IssueKind::Unclosable));
                park_stuck(&namespace);
            }
            // The browser answered and left its own tabs: a contradiction, retried later, reported
            // now.
            TabOutcome::Retry(RetryLeg::LeftOpen) => {
                issues.push(reclaimed_issue(name, &namespace, IssueKind::LeftOpen));
                park_stuck(&namespace);
            }
            // Nothing was established about that group, so the ask stays armed for it.
            TabOutcome::Retry(RetryLeg::Silent) => silent = true,
        }
    }
    // One ask, one rung: the sweep's failure is the pass, not how many names went unanswered. All
    // settled leaves the ladder at its floor; a failed read charged it itself.
    if silent {
        reclaim_unfinished();
    } else if answered {
        reset_reclaim_failures();
    }
    (reclaimed, issues)
}

/// Remember that nothing here is worth asking about `namespace` for a while: the entry expires
/// after [`STUCK_REVISIT`], and its revisit stays armed so a recovered browser is looked at again.
fn park_stuck(namespace: &str) {
    stuck_namespaces()
        .lock()
        .unwrap_poison()
        .insert(namespace.to_string(), Instant::now());
    schedule_reclaim_revisit();
}

/// One report about a reclaimed group: namespace from the name, run from [`evicted_run`].
fn reclaimed_issue(name: &str, namespace: &str, kind: IssueKind) -> Issue {
    Issue {
        kind,
        namespace: namespace.to_string(),
        run: evicted_run(namespace).unwrap_or_default(),
        name: name.to_string(),
    }
}

/// Let go of this product's own browser-read session ([`chrome_tabs::OWN_SESSION`]) if a pass of
/// ours used it, and surface the one state that is ours: `false` from
/// [`chrome_tabs::release_own_session`] means the browser did not confirm the scratch group gone —
/// a read may itself have failed — and the row says that. An unconfirmed let-go is queued for the
/// graceful `session stop` the let-go step runs a few lines later, the route that closes the tabs
/// the extension's ledger no longer reaches.
async fn let_go_own_session(cli: &Path, deadline: Instant) {
    if chrome_tabs::release_own_session(cli, deadline).await {
        return;
    }
    queue_forget(chrome_tabs::OWN_SESSION);
    if !chrome_tabs::own_scratch_group_possible() {
        return;
    }
    report_once(
        OWN_SESSION_MESSAGE,
        OWN_SESSION_REASON,
        serde_json::json!({
            "detail": format!(
                "a scratch group titled {} MAY be left in the owner's tab strip — this \
                 module's own about:blank read group, the session its browser reads run in — \
                 or the reads that look for it may themselves have failed, so the release is \
                 not confirmed; the group's own session is queued for a stop and the product's \
                 session sweeps close that family, and nothing of any run's is left behind by it",
                chrome_tabs::OWN_SESSION
            ),
        }),
    )
    .await;
}

/// One row for a host whose browser answers settle nothing this route acts on — filed at most once,
/// only after such an answer happened, since it settles no name and must not stay a `debug!` that
/// ages out.
async fn report_unreadable_answers() {
    if !chrome_tabs::saw_unreadable_answer() {
        return;
    }
    report_once(
        UNREADABLE_MESSAGE,
        UNREADABLE_REASON,
        serde_json::json!({
            "detail": "a read this cleanup decides from settled nothing: one of the extension's \
                       own answers about the browser — its status, the tab ledger it holds, or \
                       the browser's tab and group lists — carried a field no parser here will \
                       guess at, or a helper answer came back with no readable envelope, or with \
                       neither a success verdict nor a reason for failing. A partial or reasonless \
                       answer is never guessed at — a dropped tab would read as one that is gone — \
                       so nothing is concluded from it: another read, of this pass or a later one, \
                       still settles what it can",
        }),
    )
    .await;
}

/// Drop the tab record of a release that concluded with every name confirmed gone — unless the run
/// came alive again while the pass ran. A release with unclosable leftovers keeps its entry: its
/// tabs may stand.
fn conclude_released_record(namespace: &str) {
    if ProtectedNamespaces::protects_live_or_held(namespace) {
        return;
    }
    crate::tools::chrome_tab_ledger::forget_namespace(namespace);
}

/// One pass over the records a [`ReleasePass`] took plus the names the reclaim sweep found: every
/// name goes to the browser in one batched call, its answers decide each record's fate, and the
/// reports the Issues view is owed come back. [`ReleasePass`] puts back what the pass could not
/// finish. This product's own browser-read session, if used, is let go right after the close —
/// before the let-go step, which can spend what is left of the budget on children of its own.
async fn attempt_pass(
    cli: Option<&Path>,
    mut pass: Option<ReleasePass>,
    session: &str,
    ask: &ReclaimAsk,
    protected: &mut ProtectedNamespaces,
    deadline: Instant,
) -> Vec<Issue> {
    let names = pass_names(ask, pass.as_ref());
    if names.is_empty() {
        // Nothing due and nothing to reclaim: only the let-go steps, and the answered ask resets
        // the ladder.
        if ask.answered {
            reset_reclaim_failures();
        }
        if let Some(cli) = cli {
            let_go_own_session(cli, deadline).await;
            forget_settled(cli, &[], deadline).await;
        }
        // A read of THIS pass may have been the unreadable one: the row belongs to the pass that
        // met it.
        report_unreadable_answers().await;
        return Vec::new();
    }

    let outcomes = close_unprotected(cli, &names, session, protected, deadline).await;

    // Every browser read is behind us, so our own session is let go HERE, where it is affordable.
    if let Some(cli) = cli {
        let_go_own_session(cli, deadline).await;
    }

    // Let go of the settled sessions' chrome-use records and daemons — the close leaves both behind
    // — but never a name a run owns again by the time the browser answered. This is the pass's
    // THIRD read of the live map: a run re-dispatched while the close ran registers now, and its
    // tabs must not be let go under it.
    if let Some(cli) = cli {
        let live_now = ProtectedNamespaces::live_and_held();
        let settled: Vec<String> = names
            .iter()
            .filter(|name| {
                !live_now.contains(name)
                    && matches!(outcomes.get(name.as_str()), Some(TabOutcome::Gone))
            })
            .cloned()
            .collect();
        forget_settled(cli, &settled, deadline).await;
    }

    let mut issues: Vec<Issue> = Vec::new();
    let (reclaimed, reclaimed_issues) = reclaim_issues(&ask.names, ask.answered, &outcomes);
    issues.extend(reclaimed_issues);
    if reclaimed > 0 {
        info!(
            closed = reclaimed,
            "agent-run chrome leftovers reclaimed by the sweep"
        );
    }

    if let Some(pass) = pass.as_mut() {
        pass.reconcile(&outcomes, |mut entry, disposition| {
            let unclosable = disposition.unclosable.len();
            // One row per name, the kind being the browser's own answer about it.
            let reported = disposition
                .unclosable
                .into_iter()
                .map(|name| (IssueKind::Unclosable, name))
                .chain(
                    disposition
                        .left_open
                        .into_iter()
                        .map(|name| (IssueKind::LeftOpen, name)),
                );
            for (kind, name) in reported {
                issues.push(Issue {
                    kind,
                    namespace: entry.namespace.clone(),
                    run: entry.run.clone(),
                    name,
                });
            }
            // The record ends here: those tabs are `agent-tab-*` groups nothing claims, so arm the
            // sweep.
            if unclosable > 0 {
                arm_reclaim_at(Instant::now() + STUCK_REVISIT);
            }
            if disposition.retry.is_empty() {
                if unclosable == 0 {
                    conclude_released_record(&entry.namespace);
                    tracing::debug!(
                        namespace = %entry.namespace,
                        run = %entry.run,
                        sessions = entry.names.len(),
                        "agent-run chrome sessions released"
                    );
                } else {
                    info!(
                        namespace = %entry.namespace,
                        run = %entry.run,
                        unclosable,
                        left_open = entry.left_open,
                        "agent-run chrome session release settled — no route in this record \
                         closes the rest (reported)"
                    );
                }
                return;
            }
            // Charged on every pass that leaves names open — no bound, so retried while unanswered
            // — and spaced.
            entry.attempts = entry.attempts.saturating_add(1);
            entry.next_attempt_at = Instant::now() + release_backoff(entry.attempts);
            debug!(
                namespace = %entry.namespace,
                sessions = disposition.retry.len(),
                attempts = entry.attempts,
                "agent-run chrome session release still open — retrying"
            );
            entry.names = disposition.retry;
            merge_record(entry, Eligibility::Retry);
        });
        persist_pending_releases();
    }
    // A read of THIS pass may have been the unreadable one, so the host row is filed here.
    report_unreadable_answers().await;
    issues
}

/// Last-chance pass for the shutdown and self-update paths: spends at most `budget` closing every
/// record whose hold/retry elapsed, then returns. It runs no reclaim sweep and files no leftover
/// row — the next boot owns both — though it can owe the row for this module's own scratch session,
/// and reaches the let-go queue too (best-effort: the budget is below one whole stop). A held
/// record waits for a run the restart may re-drive, and the retry ladder's count is not charged; a
/// left-open answer the browser gave is, so a repeated one still concludes, and the record is kept
/// for the next boot to report.
async fn flush_pending_run_releases(budget: Duration) {
    let deadline = Instant::now() + budget;
    let cli = release_cli();
    let Some(mut pass) = ReleasePass::take() else {
        // No record is due, but the let-go queue may still hold a name: its last chance before
        // exit.
        if let Some(cli) = cli.as_deref() {
            forget_settled(cli, &[], deadline).await;
        }
        return;
    };
    let names = record_names(&pass);
    let durable = durable_resume_namespaces().await;
    let mut protected = ProtectedNamespaces::snapshot(&durable);
    let session = read_session(Some(&pass));
    let outcomes =
        close_unprotected(cli.as_deref(), &names, &session, &mut protected, deadline).await;
    if let Some(cli) = cli.as_deref() {
        // This product's own session, if the pass used it, goes before the process does.
        let_go_own_session(cli, deadline).await;
        // The let-go queue is reached here too: only a graceful stop closes the group an
        // unconfirmed let-go may have left, and the budget is below one whole stop, so the stop is
        // best-effort.
        forget_settled(cli, &[], deadline).await;
    }
    let mut released = 0usize;
    let mut still_open = 0usize;
    pass.reconcile(&outcomes, |mut entry, disposition| {
        // Only a `Gone` answer is settled; anything else stays in the record for the next boot.
        let keep: Vec<String> = disposition
            .retry
            .into_iter()
            .chain(disposition.unclosable)
            .collect();
        released += entry.names.len() - keep.len();
        if keep.is_empty() {
            return;
        }
        still_open += keep.len();
        entry.names = keep;
        merge_record(entry, Eligibility::Retry);
    });
    persist_pending_releases();
    debug!(
        released,
        still_open, "shutdown: last-chance agent-run chrome session release"
    );
}

/// The chrome half of a shutdown, called by both shutdown paths: release what ended runs left
/// queued, and close nothing else — a session ends by its own lifetime or an explicit `session
/// stop`.
pub async fn flush_ended_run_chrome_releases() {
    flush_pending_run_releases(SHUTDOWN_RELEASE_FLUSH_BUDGET).await;
}

/// Names of the live agent tab groups the sweep owns this pass: every group titled under
/// [`AGENT_TAB_PREFIX`] whose namespace no run may still own and none that already got a verdict
/// ([`stuck`]). `due` is the caller's own read of [`reclaim_due`], taken before the pass. Only the
/// names a queue or parked record still holds are left to that record ([`claimed_names`]); a name
/// the record dropped is the sweep's again, so a recovered browser still gets its group closed.
async fn reclaim_names(
    cli: Option<&Path>,
    session: &str,
    protected: &ProtectedNamespaces,
    deadline: Instant,
    due: bool,
) -> ReclaimAsk {
    if !due {
        return ReclaimAsk::unanswered();
    }
    // Every due sweep, before the browser is read: a verdict expires with its window even when
    // unread.
    prune_stuck();
    let Some(cli) = cli else {
        // No binary right now: the owed ask goes up the ladder rather than standing at once.
        reclaim_unfinished();
        return ReclaimAsk::unanswered();
    };
    // Built directly rather than through the live/held pair: it keeps a second driver off record
    // names.
    let claimed = claimed_names();
    match chrome_tabs::agent_group_names(cli, session, deadline).await {
        Ok(names) => {
            reclaim_answered();
            ReclaimAsk {
                names: names
                    .into_iter()
                    .filter(|name| {
                        !protected.contains(name)
                            && !claimed.contains(name)
                            && !stuck(crate::tools::chrome::session_namespace(name))
                    })
                    .collect(),
                answered: true,
            }
        }
        Err(reason) => {
            reclaim_unfinished();
            debug!(
                reason = %reason.text(),
                "agent-run chrome reclaim could not read the live tab groups"
            );
            // No ownership door: the sweep cannot enumerate groups, and the Issues view is owed
            // that.
            if chrome_tabs::ownership_door_missing(cli, session, deadline).await {
                report_once(
                    NO_DOOR_MESSAGE,
                    NO_OWNERSHIP_DOOR_REASON,
                    serde_json::json!({
                        "detail": "there is no ownership door on this browser side (no \
                                   extension installed, one installed but disabled, or one \
                                   older than the version that has it), so the sweep cannot \
                                   enumerate the browser's groups and a name a record holds is \
                                   left to chrome-use's own `session stop`",
                    }),
                )
                .await;
            }
            ReclaimAsk::unanswered()
        }
    }
}

/// What one reclaim ask came to; `answered: false` is a failed read or no binary, never a clean
/// ask.
struct ReclaimAsk {
    names: Vec<String>,
    answered: bool,
}

impl ReclaimAsk {
    fn unanswered() -> Self {
        Self {
            names: Vec::new(),
            answered: false,
        }
    }
}

/// When the reclaim sweep may ask the browser again, and how many consecutive asks it could not get
/// answered. `next_at` is `Some` exactly while something is unfinished or an event armed one;
/// `failures` is the ladder rung a failed ask is spaced by, kept so a downed browser is not
/// re-shovelled. The schedule starts with NOTHING armed: a start that finds nothing pending must
/// not open a page, and nothing here is timed.
#[derive(Default)]
struct ReclaimSchedule {
    next_at: Option<Instant>,
    failures: u32,
}

fn reclaim_schedule() -> &'static Mutex<ReclaimSchedule> {
    RECLAIM.get_or_init(|| Mutex::new(ReclaimSchedule::default()))
}

/// Arm an ask at `at`, keeping whichever ask is already sooner.
fn arm_reclaim_at(at: Instant) {
    let mut schedule = reclaim_schedule().lock().unwrap_poison();
    schedule.next_at = Some(schedule.next_at.map_or(at, |old| old.min(at)));
}

/// The ask the stuck verdicts still inside their window owe; strictly in the future by
/// construction.
fn next_reclaim_revisit() -> Option<Instant> {
    stuck_namespaces()
        .lock()
        .unwrap_poison()
        .values()
        .filter(|concluded| concluded.elapsed() < STUCK_REVISIT)
        .map(|concluded| *concluded + STUCK_REVISIT)
        .min()
}

/// Arm an ask now: a run end queued names, leaving the failure count to space the next ask.
fn schedule_reclaim_now() {
    arm_reclaim_at(Instant::now());
}

/// Arm a revisit [`STUCK_REVISIT`] away: a leftover the sweep could not close is owed another look.
fn schedule_reclaim_revisit() {
    arm_reclaim_at(Instant::now() + STUCK_REVISIT);
}

/// How long a namespace the sweep concluded nothing can close is left alone before the full close
/// is tried again: a verdict saves a stop per name plus two reads on every sweep, and still
/// expires.
const STUCK_REVISIT: Duration = Duration::from_mins(30);

fn stuck_namespaces() -> &'static Mutex<HashMap<String, Instant>> {
    STUCK_NAMESPACES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Whether the sweep may skip `namespace`: a verdict was reached less than [`STUCK_REVISIT`] ago.
fn stuck(namespace: &str) -> bool {
    let mut stuck = stuck_namespaces().lock().unwrap_poison();
    match stuck.get(namespace) {
        Some(concluded) if concluded.elapsed() < STUCK_REVISIT => true,
        Some(_) => {
            stuck.remove(namespace);
            false
        }
        None => false,
    }
}

/// Drop the verdicts whose [`STUCK_REVISIT`] window passed, so one never read again still expires.
fn prune_stuck() {
    let now = Instant::now();
    stuck_namespaces()
        .lock()
        .unwrap_poison()
        .retain(|_, concluded| now.saturating_duration_since(*concluded) < STUCK_REVISIT);
}

/// Whether the sweep may ask the browser this pass: only once the ask the schedule carries is due.
/// Nothing arms one by itself, so it exists only while something is unfinished or an event armed
/// one.
fn reclaim_due() -> bool {
    let mut schedule = reclaim_schedule().lock().unwrap_poison();
    match schedule.next_at {
        // The pass that takes an ask runs it: clearing it here keeps one armed meanwhile for the
        // next wake.
        Some(at) if at <= Instant::now() => {
            schedule.next_at = None;
            true
        }
        _ => false,
    }
}

/// A clean ask: only what the sweep's own leftovers owe stays scheduled — the earliest parked
/// revisit. Deriving it from the verdicts lets a revisit survive a clean ask in between.
fn reclaim_answered() {
    let revisit = next_reclaim_revisit();
    let mut schedule = reclaim_schedule().lock().unwrap_poison();
    if let Some(at) = revisit {
        schedule.next_at = Some(schedule.next_at.map_or(at, |armed| armed.min(at)));
    }
}

/// Leave the ladder at its floor: the last ask was answered and nothing of the sweep's is
/// unfinished.
fn reset_reclaim_failures() {
    reclaim_schedule().lock().unwrap_poison().failures = 0;
}

/// Charge the ladder and space the next ask by it, capped at [`RELEASE_RETRY_CAP`], keeping a
/// sooner ask.
fn reclaim_unfinished() {
    let failures = {
        let mut schedule = reclaim_schedule().lock().unwrap_poison();
        schedule.failures = schedule.failures.saturating_add(1);
        schedule.failures
    };
    arm_reclaim_at(Instant::now() + release_backoff(failures));
}

/// Namespaces durable state says a run may still be rebuilt under: the roster agent ids of every
/// job not yet terminalized, minus the slots that already finished. A cut round resumes IN PLACE
/// from its stored roster, so its remaining slots come back under the very chrome namespaces they
/// had. Nothing else durable does this: a parallel round's ids carry a random suffix, and a session
/// pin is continuity for the SESSION, not a promise the run comes back. Fail-open: no store or a
/// failed read is a `debug!`.
async fn durable_resume_namespaces() -> Vec<String> {
    durable_resume_namespaces_checked()
        .await
        .unwrap_or_default()
}

/// [`durable_resume_namespaces`], but `None` when no answer could be made: a caller that DELETES
/// durable state must not read a failed read as "nothing claims this" ([`prune_tab_records`]).
async fn durable_resume_namespaces_checked() -> Option<Vec<String>> {
    let store = crate::session::SESSIONS.get()?;
    match crate::jobs::resumable_roster_agent_ids(&store.conn).await {
        Ok(agent_ids) => Some(
            agent_ids
                .iter()
                .map(|agent_id| crate::tools::chrome::run_session_namespace(agent_id))
                .collect(),
        ),
        Err(error) => {
            debug!(
                %error,
                "could not read the resumable runs' agent ids — reclaiming without them"
            );
            None
        }
    }
}

/// The session names every record queued or in a pass still holds — matched EXACTLY, because record
/// names ARE the physical session names. A name the record still carries is left to its own pass.
fn claimed_names() -> HashSet<String> {
    let parked = parked_releases().lock().unwrap_poison();
    let queue = pending_releases().lock().unwrap_poison();
    queue
        .iter()
        .chain(parked.iter().map(|(_, record)| record))
        .flat_map(|record| record.names.iter().cloned())
        .collect()
}

/// The namespaces a close must leave alone, snapshotted once per pass: every run live right now,
/// every queued record still inside its hold, and `durable` — the ids of a not-yet-terminalized
/// job's unfinished slots. The live half is re-read before the close and at every act; the durable
/// half is bounded by the stale-job purge, where a frozen job is purge-immune. Members are
/// namespaces, matched as a prefix.
struct ProtectedNamespaces {
    namespaces: Vec<String>,
}

impl ProtectedNamespaces {
    /// Every namespace a close must leave alone this pass: the live map, the held records,
    /// `durable`.
    fn snapshot(durable: &[String]) -> Self {
        let now = Instant::now();
        // Each lock is taken alone: `take_releasable` holds the queue lock while reading the live
        // map.
        let live: Vec<String> = live_run_namespaces()
            .lock()
            .unwrap_poison()
            .keys()
            .cloned()
            .collect();
        let held: Vec<String> = pending_releases()
            .lock()
            .unwrap_poison()
            .iter()
            .filter(|record| within_hold(record, now))
            .map(|record| record.namespace.clone())
            .collect();
        let mut protected = Self { namespaces: live };
        protected.namespaces.extend(held);
        protected.namespaces.extend(durable.iter().cloned());
        protected
    }

    /// The live map and the held records only — the set a step that must not race a run re-reads in
    /// one go.
    fn live_and_held() -> Self {
        Self::snapshot(&[])
    }

    /// Whether the LIVE map or a HELD record protects `name` now — what a close asks at each act.
    fn protects_live_or_held(name: &str) -> bool {
        let live = live_run_namespaces().lock().unwrap_poison();
        if live
            .keys()
            .any(|namespace| names_a_namespace(namespace, name))
        {
            return true;
        }
        drop(live);
        let now = Instant::now();
        pending_releases()
            .lock()
            .unwrap_poison()
            .iter()
            .any(|record| within_hold(record, now) && names_a_namespace(&record.namespace, name))
    }

    fn refresh(&mut self) {
        self.namespaces.extend(Self::live_and_held().namespaces);
    }

    /// Whether `name` belongs to a namespace some run may still own — also asked of a namespace
    /// itself.
    fn contains(&self, name: &str) -> bool {
        self.namespaces
            .iter()
            .any(|prefix| names_a_namespace(prefix, name))
    }
}

/// Whether `name` belongs to `prefix`; an empty prefix matches nothing.
fn names_a_namespace(prefix: &str, name: &str) -> bool {
    !prefix.is_empty() && name.starts_with(prefix)
}

pub(crate) const ISSUE_TARGET: &str = "chrome-tabs";
/// The one durable message every leftover report carries; stable, because the Issues view dedupes.
const LEFTOVER_MESSAGE: &str = "agent-run browser tabs could not be closed automatically";
/// The report for a close the browser answered for and did not make (see [`RetryLeg::LeftOpen`]).
const LEFT_OPEN_MESSAGE: &str =
    "agent-run browser tabs were left open: the browser answered for them and did not close them";
/// The report for a host whose browser side has no ownership door (see chrome_tabs' Honest limits):
/// a fact about the host, not a leftover of any run. The listing every read runs on is missing, so
/// a record's names are left to chrome-use's own `session stop`; nothing is named and no close is
/// claimed to have failed.
const NO_DOOR_MESSAGE: &str = "the browser side here cannot list agent-run tab groups, so \
                               leftovers no record names cannot be closed";
/// The report for this module's own scratch group a let-go could not confirm gone: the one visible
/// tab this product can leave in the owner's strip, so it is surfaced rather than accepted quietly
/// — but the reads that look for the group can fail, so the row says only that the release was not
/// CONFIRMED. It is the one name neither queue bound drops, so it is retried on the queue's own
/// capped deadline while the process lives, and the session sweeps own its family across a restart.
const OWN_SESSION_MESSAGE: &str =
    "the product's own browser-read session could not be confirmed released";
/// Why that report is filed: a property of the module's own session, not of any run.
const OWN_SESSION_REASON: &str = "chrome-tabs:own-session";
/// Why a tab is reported as unclosable: its ledger lost it, it was never this session's, or the
/// browser answered this run's close and left it standing ([`MAX_LEFT_OPEN_ATTEMPTS`] passes) —
/// only ever said of a repeated answer. Every row states what the product could NOT do, never what
/// the browser holds now.
const UNCLOSABLE_DETAIL: &str = "the product could not close these tabs and keeps retrying \
                                 them: the browser extension answered that it does not hold \
                                 them as its own (it lost its ledger — updated, re-installed or \
                                 restarted since they were created — or never created them), or \
                                 the browser answered this run's close attempts and left the \
                                 tabs standing — the removal refused as not its own to make, or \
                                 answered while they still stood";
const LEFT_OPEN_DETAIL: &str = "the browser extension answered this run's close and the tabs \
                                were left standing: it refused the removal as not its own to \
                                make, or answered it while the tabs still stood";
/// The reason a host with no ownership door is reported under: a property of the host, so no
/// namespace.
const NO_OWNERSHIP_DOOR_REASON: &str = "chrome-tabs:no-ownership-door";
/// The report for a host whose browser answers this route cannot decide from
/// ([`chrome_tabs::saw_unreadable_answer`]): the door is there and an answer came back, but nothing
/// in it settles a name. It claims neither a failed close nor a host no route can settle, and names
/// nothing.
const UNREADABLE_MESSAGE: &str = "the browser's answers gave this cleanup no decision it could act on, so no name was concluded from them";
/// Why that report is filed: a fact about the host rather than about any one run or name.
const UNREADABLE_REASON: &str = "chrome-tabs:unreadable-answer";
/// Cap on the joined `session` field of a report, so one wide leftover group cannot bloat the row.
const SESSION_NAMES_MAX_CHARS: usize = 300;
/// Reasons already reported in THIS process, so a re-read leftover costs no store read.
static REPORTED_REASONS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

fn reported_reasons() -> &'static Mutex<HashSet<String>> {
    REPORTED_REASONS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Which fact one report carries: the first ends its record's retries, the second keeps retrying.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IssueKind {
    /// The record's routes are done with these tabs: reported with the run, examined by the sweep.
    Unclosable,
    /// The browser answered and left the tabs: the record keeps retrying them.
    LeftOpen,
}

impl IssueKind {
    fn message(self) -> &'static str {
        match self {
            Self::Unclosable => LEFTOVER_MESSAGE,
            Self::LeftOpen => LEFT_OPEN_MESSAGE,
        }
    }

    /// The reason prefix this kind's rows are deduped on. One reason per namespace, so a repeated
    /// pass over the same run is a duplicate, not news.
    fn reason_prefix(self) -> &'static str {
        match self {
            Self::Unclosable => "chrome-tabs:",
            Self::LeftOpen => "chrome-tabs-left-open:",
        }
    }

    /// One sentence per kind: every name of a group ran into the same fact.
    fn detail(self) -> &'static str {
        match self {
            Self::Unclosable => UNCLOSABLE_DETAIL,
            Self::LeftOpen => LEFT_OPEN_DETAIL,
        }
    }
}

/// One report the Issues view is owed: the namespace rows are deduped on, the run, and one leaked
/// name.
struct Issue {
    kind: IssueKind,
    namespace: String,
    run: String,
    name: String,
}

/// One namespace's issues, the shape the Issues view is written in: one row per `(kind,
/// namespace)`.
struct IssueGroup {
    kind: IssueKind,
    namespace: String,
    run: String,
    /// The session names, in first-seen order.
    names: Vec<String>,
}

/// Report every fact the Issues view is owed: one row per `(kind, namespace)`, once per `(message,
/// reason)`.
async fn report_issues(issues: Vec<Issue>) {
    let mut groups: Vec<IssueGroup> = Vec::new();
    for issue in issues {
        match groups
            .iter_mut()
            .find(|group| group.kind == issue.kind && group.namespace == issue.namespace)
        {
            Some(group) => {
                if group.run.is_empty() {
                    group.run.clone_from(&issue.run);
                }
                group.names.push(issue.name);
            }
            None => groups.push(IssueGroup {
                kind: issue.kind,
                namespace: issue.namespace,
                run: issue.run,
                names: vec![issue.name],
            }),
        }
    }
    for group in groups {
        let reason = format!("{}{}", group.kind.reason_prefix(), group.namespace);
        let fields = serde_json::json!({
            "detail": group.kind.detail(),
            "run": group.run,
            "sessions": group.names.len(),
            "session": crate::util::truncate(&group.names.join(", "), SESSION_NAMES_MAX_CHARS),
        });
        report_once(group.kind.message(), &reason, fields).await;
    }
}

/// Tell the Issues view one fact, once: the in-process set spares a re-reading sweep, and a reason
/// is marked only after the write landed. Fail-open — a fact lost to a down store is reported again
/// by the next pass that reads it, and the process never stops closing tabs over a broken log
/// store.
async fn report_once(message: &str, reason: &str, fields: serde_json::Value) {
    if reported_reasons().lock().unwrap_poison().contains(reason) {
        return;
    }
    // The seam captures the same `(message, reason, fields)` the real writer takes.
    #[cfg(test)]
    if let Some(sink) = release_settings().issues {
        sink.lock()
            .unwrap_poison()
            .push((message.to_string(), reason.to_string(), fields));
        return mark_reported(reason);
    }
    if matches!(
        crate::logs::record_issue_once(message, reason, ISSUE_TARGET, fields).await,
        // A reason the store already holds is as good as written: the fact is in the view.
        crate::logs::IssueWrite::Written | crate::logs::IssueWrite::AlreadyRecorded
    ) {
        mark_reported(reason);
    }
}

/// Remember that `reason` has been reported, sparing the next pass the store read.
fn mark_reported(reason: &str) {
    reported_reasons()
        .lock()
        .unwrap_poison()
        .insert(reason.to_string());
}

/// Test seam: when set, issue reports land here instead of the Issues store.
#[cfg(test)]
type CapturedIssues = std::sync::Arc<Mutex<Vec<(String, String, serde_json::Value)>>>;

/// The whole budget of one close attempt — a release pass or one cap eviction: two phases of
/// browser reads ([`READ_PHASE_RESERVE`] each) and one `session stop` between. The removal call and
/// its verification reads have no reserve, so on a slow pass those names stay retries, settled by a
/// later pass.
///
/// [`READ_PHASE_RESERVE`]: crate::tools::chrome_tabs::READ_PHASE_RESERVE
const ATTEMPT_TIMEOUT: Duration = crate::tools::chrome_tabs::READ_PHASE_RESERVE
    .saturating_mul(2)
    .saturating_add(crate::chrome::SESSION_STOP_TIMEOUT);

/// Every knob of the release path — one accessor so a test seam swaps the whole set at once. A
/// release build compiles no seam state and always reads the constants above plus the storage root;
/// the binary is resolved on demand by [`release_cli`], so a `Drop`-driven enqueue, a persist or a
/// restore never probes for it.
#[derive(Clone, Debug)]
struct ReleaseSettings {
    attempt_timeout: Duration,
    retry_base: Duration,
    boot_grace: Duration,
    /// The durable record file (`None` = nothing to write it to; see [`RELEASE_FILE_NAME`]).
    store: Option<PathBuf>,
    /// Stub binary the release tests spawn in place of the real `chrome-use`.
    #[cfg(test)]
    cli: Option<PathBuf>,
    /// The sink [`report_once`] writes to under test (see [`CapturedIssues`]).
    #[cfg(test)]
    issues: Option<CapturedIssues>,
}

fn release_settings() -> ReleaseSettings {
    #[cfg(test)]
    if let Some(settings) = test_release_settings() {
        return settings;
    }
    ReleaseSettings {
        attempt_timeout: ATTEMPT_TIMEOUT,
        retry_base: RELEASE_RETRY_BASE,
        boot_grace: RELEASE_BOOT_GRACE,
        store: crate::config::CONFIG
            .try_storage_root()
            .map(|root| root.join(RELEASE_FILE_NAME)),
        #[cfg(test)]
        cli: None,
        #[cfg(test)]
        issues: None,
    }
}

/// The `chrome-use` binary a release pass spawns, resolved once per pass from the [`cli_path`]
/// probe. An installed test seam decides on its own, so a test can put the path in a self-update
/// swap window.
fn release_cli() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(settings) = test_release_settings() {
        return settings.cli;
    }
    cli_path()
}

#[cfg(test)]
static RELEASE_SETTINGS: Mutex<Option<ReleaseSettings>> = Mutex::new(None);

#[cfg(test)]
fn test_release_settings() -> Option<ReleaseSettings> {
    RELEASE_SETTINGS.lock().unwrap_poison().clone()
}

/// Install release seams (a stub `chrome-use` plus shortened timings), returning the previous
/// value.
#[cfg(test)]
fn swap_release_settings(settings: ReleaseSettings) -> Option<ReleaseSettings> {
    RELEASE_SETTINGS.lock().unwrap_poison().replace(settings)
}

#[cfg(test)]
fn restore_release_settings(previous: Option<ReleaseSettings>) {
    *RELEASE_SETTINGS.lock().unwrap_poison() = previous;
}

/// Gap before retry `attempts`: base doubling per attempt, capped.
fn release_backoff(attempts: u32) -> Duration {
    release_settings()
        .retry_base
        .saturating_mul(2u32.saturating_pow(attempts.saturating_sub(1)))
        .min(RELEASE_RETRY_CAP)
}

#[cfg(test)]
fn pending_releases_snapshot() -> Vec<(Vec<String>, u32, Duration)> {
    let now = Instant::now();
    pending_releases()
        .lock()
        .unwrap_poison()
        .iter()
        .map(|entry| {
            (
                entry.names.clone(),
                entry.attempts,
                entry.next_attempt_at.saturating_duration_since(now),
            )
        })
        .collect()
}

/// Test-only: the queued records as `(names, attempts)`, the shape most assertions pin.
#[cfg(test)]
pub(crate) fn pending_names_and_attempts() -> Vec<(Vec<String>, u32)> {
    pending_releases_snapshot()
        .into_iter()
        .map(|(names, attempts, _)| (names, attempts))
        .collect()
}

#[cfg(test)]
pub(crate) fn pending_release_delays() -> Vec<Duration> {
    pending_releases_snapshot()
        .into_iter()
        .map(|(_, _, delay)| delay)
        .collect()
}

/// Test-only: empty everything the release path holds in memory — queues, schedule, verdicts,
/// latches.
#[cfg(test)]
pub(crate) fn clear_pending_releases() {
    // Parked first, the order [`ReleasePass::take`] acquires them in, so no lock cycle can form.
    parked_releases().lock().unwrap_poison().clear();
    pending_releases().lock().unwrap_poison().clear();
    reported_reasons().lock().unwrap_poison().clear();
    *reclaim_schedule().lock().unwrap_poison() = ReclaimSchedule::default();
    stuck_namespaces().lock().unwrap_poison().clear();
    pending_forget().lock().unwrap_poison().clear();
    *forget_retry().lock().unwrap_poison() = None;
    evicted_runs().lock().unwrap_poison().clear();
    // The chrome_tabs process latch is cleared too, so a met unreadable answer cannot report again.
    crate::tools::chrome_tabs::reset_unreadable_answer();
}

/// Test-only: the sessions waiting to be let go, with their failed attempts; read by the unix
/// tests.
#[cfg(all(test, unix))]
fn pending_forget_snapshot() -> Vec<(String, u32)> {
    pending_forget().lock().unwrap_poison().clone()
}

/// Keeps the release path from writing the record file while it lives, so no test leaves records
/// behind.
#[cfg(test)]
pub(crate) struct NoReleaseStore(Option<ReleaseSettings>);

#[cfg(test)]
#[must_use = "the store is turned back on when the guard drops"]
pub(crate) fn no_release_store() -> NoReleaseStore {
    let mut settings = release_settings();
    settings.store = None;
    NoReleaseStore(swap_release_settings(settings))
}

#[cfg(test)]
impl Drop for NoReleaseStore {
    fn drop(&mut self) {
        restore_release_settings(self.0.take());
    }
}

/// Unix only: every test drives the release stub through a `sh` child.
#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::tools::chrome::{AGENT_TAB_PREFIX, ChromeRunSessions};
    use crate::tools::chrome_tab_ledger;
    use crate::tools::chrome_tabs::OWN_SESSION;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;

    /// Re-arm the attempt bound of the settings a [`ReleaseGuard`] installed, so a later pass does
    /// not race a cold stub spawn.
    fn widen_attempt_bound(attempt_timeout: Duration) {
        let mut settings = release_settings();
        settings.attempt_timeout = attempt_timeout;
        swap_release_settings(settings);
    }

    /// Re-arm the retry gap, to a value no test waits out: a release that still lands promptly can
    /// then only be explained by a wake.
    fn widen_retry_base(retry_base: Duration) {
        let mut settings = release_settings();
        settings.retry_base = retry_base;
        swap_release_settings(settings);
    }

    fn record_file_names(path: &Path) -> Vec<String> {
        let json = fs::read_to_string(path).expect("record file");
        let mut names: Vec<String> = serde_json::from_str::<Vec<PersistedRunRelease>>(&json)
            .expect("the file stays a record file")
            .into_iter()
            .flat_map(|record| record.names)
            .collect();
        names.sort();
        names
    }

    /// The `left_open` count of each queued record — the browser's repeated answer that a close
    /// left the tabs standing (see [`MAX_LEFT_OPEN_ATTEMPTS`]), which
    /// [`pending_names_and_attempts`] leaves out.
    fn pending_left_open() -> Vec<u32> {
        pending_releases()
            .lock()
            .unwrap_poison()
            .iter()
            .map(|entry| entry.left_open)
            .collect()
    }

    /// The browser state the stub answers from, one line per tab and per group: `tab <id> <groupId>
    /// <owned>` and `group <id> <title>`.
    fn write_browser_state(path: &Path, names: &[String], owned: bool) {
        let each: Vec<(&str, bool)> = names.iter().map(|name| (name.as_str(), owned)).collect();
        write_browser_state_each(path, &each);
    }

    /// As [`write_browser_state`], one ownership flag per group: the shape a record's close sees
    /// when one of its names is a leftover no route can close while another is closeable.
    fn write_browser_state_each(path: &Path, owned: &[(&str, bool)]) {
        let index = |i: usize| i64::try_from(i).expect("a small test index");
        let mut lines: Vec<String> = owned
            .iter()
            .enumerate()
            .map(|(i, (_, held))| {
                format!(
                    "tab {} {} {}",
                    100 + index(i),
                    10 + index(i),
                    u8::from(*held)
                )
            })
            .collect();
        lines.extend(
            owned
                .iter()
                .enumerate()
                .map(|(i, (name, _))| format!("group {} {name}", 10 + index(i))),
        );
        fs::write(path, lines.join("\n")).expect("write browser state");
    }

    /// The stub `chrome-use` itself: answers the extension protocol from `state`, records argv to
    /// `log`, and takes its behaviour from `mode`: `refuse` (structured error, never the
    /// extension's own refusal sentence), `remove-sentence` (`tabs.remove` fails with that
    /// sentence), `remove-transient` (the same, never reaching the browser), `silent`, `keep`
    /// (answers and leaves its tabs open), `amnesia-once`, `garbled-answer`, `stop-refused`,
    /// `no-extension`, `disabled-extension`, and `ledger-unreadable` (a crafted shape whose status
    /// reads as no extension while the group query still answers, reaching the door-less fallback;
    /// its `session stop` refuses one name).
    #[expect(clippy::too_many_lines)] // one shell script, one stub: the modes are read against each other
    fn stub_script(log: &Path, mode: &Path, state: &Path, mark: &Path) -> String {
        r#"#!/bin/sh
printf '%s\n' "$*" >> __LOG__
mode="$(cat __MODE__)"
case "$mode" in
  refuse) printf '%s' '{"success":false,"error":"the browser extension refused"}'; exit 1 ;;
  silent) exec sleep 5 ;;
esac
case "$mode" in
  no-extension|disabled-extension)
    case "$*" in
      # The real envelope of a host with no usable extension: the host manifest IS
      # installed (this product writes it on every start) while nothing answers an
      # `extension call`. For `no-extension` both the live version and chromeExtension are
      # null (`installed` is not the extension's state); for `disabled-extension` the
      # extension is there with a version that passes the gate and a non-empty
      # `disableReasons`, which Chrome reports for one switched off in chrome://extensions.
      "extension status"*)
        if [ "$mode" = disabled-extension ]; then
          printf '%s' '{"success":true,"data":{"installed":true,"liveExtensionVersion":null,"chromeExtension":{"version":"0.5.25","disableReasons":["user"]}}}'
        else
          printf '%s' '{"success":true,"data":{"installed":true,"liveExtensionVersion":null,"chromeExtension":null}}'
        fi
        exit 0 ;;
      *) printf '%s' '{"success":false,"error":"no browser extension"}'; exit 1 ;;
    esac ;;
  ledger-unreadable)
    # `extension status` reads exactly like `no-extension`, while `tabGroups.query` still
    # answers from the state file (below). Both ownership reads fail with the state-less
    # envelope, so the close must fall back and map each live group back to its name. The
    # fallback's own `session stop` succeeds except for a name carrying `refuse`, so the
    # mapping has two different outcomes to place.
    case "$*" in
      "extension status"*) printf '%s' '{"success":true,"data":{"installed":true,"liveExtensionVersion":null,"chromeExtension":null}}'; exit 0 ;;
      "extension state"*|"extension call tabs.query"*) printf '%s' '{"success":false,"error":"no browser extension"}'; exit 1 ;;
      "session stop"*refuse*) printf '%s' '{"success":false,"error":"the session is wedged"}'; exit 1 ;;
    esac ;;
  garbled-answer)
    # A live extension that answers one read in a shape no parser can use: the group list
    # carries an element without an integer id, which this route refuses to guess a part of.
    case "$*" in
      "extension call tabGroups.query"*) printf '%s' '{"success":true,"data":{"result":[{"id":"one","title":"agent-tab-0badf00d0000-x"}]}}'; exit 0 ;;
    esac ;;
  stop-refused)
    # The shape a session whose browser the extension no longer reaches answers a stop with:
    # every read works, and `session stop` reports that a tab it created could not be closed
    # and that ownership was kept — chrome-use's own wording, which nothing else in this stub
    # produces, so a test can tell a stop that cannot close a group from one that never ran.
    case "$*" in
      "session stop"*) printf '%s' '{"success":false,"error":"stopped session daemon, but 1 tab it created could not be closed; ownership was preserved"}'; exit 1 ;;
    esac ;;
esac
case "$*" in
  "extension call tabGroups.query"*)
    printf '{"success":true,"data":{"result":[%s]}}' "$(sed -n 's/^group \([0-9]*\) \(.*\)$/{"id":\1,"title":"\2"}/p' __STATE__ | paste -sd, -)" ;;
  "extension call tabs.query"*)
    printf '{"success":true,"data":{"result":[%s]}}' "$(sed -n 's/^tab \([0-9]*\) \([0-9-]*\) .*/{"id":\1,"groupId":\2}/p' __STATE__ | paste -sd, -)" ;;
  "extension state"*)
    # `amnesia-once` answers the first ledger read with an empty one — the shape a
    # swallowed storage read takes, which reads exactly like a ledger that lost its tabs.
    if [ "$mode" = amnesia-once ] && [ ! -e __MARK__ ]; then
      : > __MARK__
      printf '%s' '{"success":true,"data":{"ownedTabs":[]}}'
    else
      printf '{"success":true,"data":{"ownedTabs":[%s]}}' "$(awk '$1=="tab" && $4=="1"{printf "%s%s", (o?",":""), $2; o=1}' __STATE__)"
    fi ;;
  "extension status"*)
    printf '%s' '{"success":true,"data":{"installed":true,"liveExtensionVersion":"0.5.25","chromeExtension":{"version":"0.5.25"}}}' ;;
  "extension call tabs.remove"*)
    # The ids must arrive as the call's own single argument, a nested array
    # ([[1,2]]): the extension spreads the JSON as chrome.tabs.remove's positional
    # arguments, so [1,2] would be read as a tab id plus a callback.
    case "$4" in
      "[["*) ;;
      *) printf '%s' '{"success":false,"error":"tabs.remove wants the tab ids as its first argument"}'; exit 1 ;;
    esac
    # The two cores of the close's promise, told apart purely by the leg's error text.
    # `remove-sentence` is the extension's own refusal sentence, relayed verbatim through
    # the helper — the one failing leg that counts as the browser answering about the tabs.
    # `remove-transient` is a leg that never reached the browser at all, so it must never be
    # read as the browser answering for the tabs. Both answer every read above normally, so the
    # ledger holds the group's tabs and this leg is actually reached.
    if [ "$mode" = remove-sentence ]; then
      printf '%s' '{"success":false,"error":"call: tabs.remove refused — tab 7 is not owned by this relay (agent-created or adopted tabs only)"}'
      exit 1
    fi
    if [ "$mode" = remove-transient ]; then
      printf '%s' "{\"success\":false,\"error\":\"relay isn't connected\"}"
      exit 1
    fi
    ids="$(printf '%s' "$4" | tr -d '[]' | tr ',' ' ')"
    # The real door refuses the WHOLE call unless every id is one it holds, so a
    # wrong or foreign id must fail the call here too.
    for id in $ids; do
      awk -v id="$id" '$1=="tab" && $2==id && $4=="1"{found=1} END{exit !found}' __STATE__ \
        || { printf '%s' '{"success":false,"error":"tab '"$id"' is not owned by this relay"}'; exit 1; }
    done
    # `keep` is the browser that answers and leaves its own tabs open anyway. Otherwise
    # exactly the tabs asked for go, and nothing else: a close that asks for too few — or
    # for a held tab of another session's group — must leave the browser changed
    # accordingly, so a test can catch it. A group whose last tab went no longer exists.
    if [ "$mode" != keep ]; then
      for id in $ids; do
        sed "/^tab $id /d" __STATE__ > __STATE__.new && mv __STATE__.new __STATE__
      done
      awk 'NR==FNR{if ($1=="tab") live[$3]=1; next} $1=="group" && !($2 in live){next} {print}' __STATE__ __STATE__ > __STATE__.new && mv __STATE__.new __STATE__
    fi
    printf '%s' '{"success":true}' ;;
  *)
    printf '%s' '{"success":true}' ;;
esac
exit 0
"#
        .replace("__LOG__", &log.display().to_string())
        .replace("__MODE__", &mode.display().to_string())
        .replace("__STATE__", &state.display().to_string())
        .replace("__MARK__", &mark.display().to_string())
    }

    /// A stub `chrome-use`, the [`ReleaseSettings`] pointing at it, a private record file and a
    /// browser-state file.
    struct ReleaseGuard {
        state: PathBuf,
        mode: PathBuf,
        log: PathBuf,
        issues: CapturedIssues,
        previous: Option<ReleaseSettings>,
        // Holds the stub binary, the record file and the state file for the guard's lifetime.
        dir: tempfile::TempDir,
    }

    impl ReleaseGuard {
        // Install the stub settings with no boot grace (every restored record is eligible at once)
        // and warm the spawn path once.
        async fn install(attempt_timeout: Duration) -> Self {
            Self::install_with(attempt_timeout, Duration::ZERO).await
        }

        // `boot_grace` is the floor a restored record's eligibility gets — a test injects a short
        // one instead of waiting a real minute.
        async fn install_with(attempt_timeout: Duration, boot_grace: Duration) -> Self {
            let dir = tempfile::tempdir().expect("release stub dir");
            let cli = dir.path().join("chrome-use");
            let mode = dir.path().join("mode");
            let log = dir.path().join("log");
            let state = dir.path().join("state.json");
            let script = stub_script(&log, &mode, &state, &dir.path().join("mark"));
            fs::write(&cli, script).expect("write release stub");
            crate::util::test::make_executable(&cli);
            Self::write_mode(&mode, "ok");
            write_browser_state(&state, &[], true);
            let issues = Arc::new(Mutex::new(Vec::new()));
            let previous = swap_release_settings(ReleaseSettings {
                cli: Some(cli.clone()),
                attempt_timeout,
                retry_base: Duration::ZERO,
                boot_grace,
                store: Some(dir.path().join(RELEASE_FILE_NAME)),
                issues: Some(Arc::clone(&issues)),
            });
            let guard = Self {
                state,
                mode,
                log,
                issues,
                previous,
                dir,
            };
            // Warm-up call; the log is reset so it cannot be mistaken for a real close pass, and
            // the session it read in is let go exactly as a pass would — otherwise the next pass
            // would find a read outstanding that it never made.
            let _ = chrome_tabs::close_sessions(
                &cli,
                &["warmup".to_string()],
                OWN_SESSION,
                Instant::now() + attempt_timeout,
                &|_: &str| false,
            )
            .await;
            chrome_tabs::release_own_session(&cli, Instant::now() + attempt_timeout).await;
            guard.clear_log();
            guard
        }

        fn set_mode(&self, mode: &str) {
            Self::write_mode(&self.mode, mode);
        }

        fn write_mode(path: &Path, mode: &str) {
            fs::write(path, mode).expect("write release mode");
        }

        // Write the browser state the stub answers from, one group per name.
        fn write_state(&self, names: &[String], owned: bool) {
            write_browser_state(&self.state, names, owned);
        }

        // Write the browser state with one ownership flag per group — the shape a close sees when
        // one of a record's names is unclosable and another is closeable.
        fn write_state_each(&self, owned: &[(&str, bool)]) {
            write_browser_state_each(&self.state, owned);
        }

        // The titles of the tab groups the stub's browser still holds (`group <id> <title>` lines),
        // so a test can assert what an eviction left open.
        fn group_titles(&self) -> Vec<String> {
            fs::read_to_string(&self.state)
                .unwrap_or_default()
                .lines()
                .filter_map(|line| line.strip_prefix("group "))
                .filter_map(|rest| rest.split_once(' ').map(|(_, title)| title.to_string()))
                .collect()
        }

        // Put the release path in the window a managed self-update swap creates: chrome-use
        // resolves to nothing.
        #[expect(
            clippy::unused_self,
            reason = "called as a guard method so every seam installs the same way"
        )]
        fn set_binary_absent(&self) {
            let mut settings = release_settings();
            settings.cli = None;
            swap_release_settings(settings);
        }

        // Arm the sweep as an event would — the schedule no longer arms a boot ask.
        #[expect(
            clippy::unused_self,
            reason = "called as a guard method so every seam installs the same way"
        )]
        fn arm_sweep(&self) {
            schedule_reclaim_now();
        }

        // One line per stub invocation (the argv the release path built).
        fn log_lines(&self) -> Vec<String> {
            fs::read_to_string(&self.log)
                .unwrap_or_default()
                .lines()
                .map(str::to_string)
                .collect()
        }

        // Whether any invocation's argv contained `needle`.
        fn invoked(&self, needle: &str) -> bool {
            self.log_lines().iter().any(|line| line.contains(needle))
        }

        // The tab ids the release path asked `tabs.remove` for, in the order asked — what the close
        // demanded of the browser, not what the browser answered.
        fn removed_tab_ids(&self) -> Vec<i64> {
            self.log_lines()
                .iter()
                .filter(|line| line.starts_with("extension call tabs.remove"))
                .flat_map(|line| {
                    let Some(start) = line.find("[[") else {
                        return Vec::new();
                    };
                    let rest = &line[start + 2..];
                    let Some(end) = rest.find("]]") else {
                        return Vec::new();
                    };
                    rest[..end]
                        .split(',')
                        .filter_map(|id| id.parse::<i64>().ok())
                        .collect()
                })
                .collect()
        }

        // Whether the only things the release path asked the browser were reads: no record was
        // attempted, nothing was closed, and no record's session was stopped.
        fn only_swept(&self) -> bool {
            self.log_lines().iter().all(|line| {
                !line.contains("tabs.remove")
                    && (!line.contains("session stop")
                        || line.contains(&format!("--session {OWN_SESSION}")))
            })
        }

        fn clear_log(&self) {
            fs::write(&self.log, "").expect("reset release log");
        }

        fn issues(&self) -> Vec<(String, String, serde_json::Value)> {
            self.issues.lock().unwrap_poison().clone()
        }

        fn clear_issues(&self) {
            self.issues.lock().unwrap_poison().clear();
        }

        // The guard's own durable record file (never the real `~/.mahbot` one).
        fn store_path(&self) -> PathBuf {
            self.dir.path().join(RELEASE_FILE_NAME)
        }
    }

    impl Drop for ReleaseGuard {
        fn drop(&mut self) {
            restore_release_settings(self.previous.take());
            clear_pending_releases();
        }
    }

    /// Seed one ended run's record through the production tracker (dropped, as a real run's tracker
    /// is at run end) and hand it over, eligible at once.
    fn queue_names(agent_id: &str, names: &[&str]) {
        queue_names_with_hold(agent_id, names, Duration::ZERO);
    }

    /// As [`queue_names`], with the hold the run-end hand-off would have passed
    /// ([`RELEASE_HOLD_REDRIVEN_RUN`] for a run that resumes under its own id).
    fn queue_names_with_hold(agent_id: &str, names: &[&str], hold: Duration) {
        let sessions = ChromeRunSessions::for_run(agent_id);
        for name in run_session_names(agent_id, names) {
            sessions.track(&name);
        }
        queue_run_session_release_after(&sessions, hold, agent_id);
    }

    /// Physical names as a run records them: the tracker's namespace plus each logical name — the
    /// shape the durable record file carries.
    fn run_session_names(agent_id: &str, logical: &[&str]) -> Vec<String> {
        let ns = ChromeRunSessions::for_run(agent_id).namespace().to_string();
        logical.iter().map(|name| format!("{ns}{name}")).collect()
    }

    /// Write the record file the way a previous process would have left it.
    fn write_record_file(path: &Path, records: &[PersistedRunRelease]) {
        fs::write(
            path,
            serde_json::to_string(records).expect("serialize release records"),
        )
        .expect("write release record file");
    }

    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_browsers_own_answers_settle_each_release_record() {
        // An ended run's record is dropped once the browser's own answer says every one of its
        // names is gone, and the close went through the browser's extension.
        {
            let case = "a verified release empties the record";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let names =
                run_session_names("run-verified", &["agent-tab-v-default", "agent-tab-v-docs"]);
            guard.write_state(&names, true);
            queue_names("run-verified", &["agent-tab-v-default", "agent-tab-v-docs"]);
            assert_eq!(pending_releases_snapshot().len(), 1, "case {case}");

            release_due().await;

            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: both names settled gone"
            );
            assert!(
                guard.invoked("extension call tabs.remove"),
                "case {case}: the close went through the browser's own extension"
            );
            assert_eq!(
                guard.removed_tab_ids(),
                vec![100, 101],
                "case {case}: exactly the run's own tabs were asked for: {:?}",
                guard.log_lines()
            );
            assert!(
                guard.invoked(&format!(
                    "session stop --force --json --session {}",
                    names[0]
                )),
                "case {case}: the record's own session the reads ran in is let go once its group is settled, so \
                 it cannot outlive the pass: {:?}",
                guard.log_lines()
            );
            assert!(
                !guard.only_swept(),
                "case {case}: a pass that asked tabs.remove is not a sweep, whatever session it ran in: {:?}",
                guard.log_lines()
            );
        }
        // A record whose groups the browser no longer holds is dropped — no `tabs.remove` is even
        // needed for it.
        {
            let case = "a record the browser reports gone is dropped";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            queue_names("run-gone", &["agent-tab-g-default"]);

            release_due().await;

            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: the browser holds no group by that name, so its tabs are gone"
            );
            assert!(!guard.invoked("extension call tabs.remove"), "case {case}");
        }
        // The `refuse` stub answers a structured error to EVERY call, so the browser never settles
        // the record: it stays queued past five attempts and is never dropped from the durable file
        // — there is no attempt cap on a record the browser did not answer for.
        {
            let case = "a record the browser never settles stays queued past five attempts";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            let names = run_session_names("run-refuse", &["agent-tab-r-stuck"]);
            queue_names("run-refuse", &["agent-tab-r-stuck"]);
            guard.set_mode("refuse");

            for _ in 0..6 {
                release_due().await;
                assert_eq!(
                    pending_names_and_attempts().len(),
                    1,
                    "case {case}: the record stays queued across attempts"
                );
                assert_eq!(
                    pending_names_and_attempts()[0].0,
                    names,
                    "case {case}: and keeps its names"
                );
            }

            let (queued, attempts) = &pending_names_and_attempts()[0];
            assert_eq!(queued, &names, "case {case}: the names are intact");
            assert!(
                *attempts > 5,
                "case {case}: past five attempts, still queued: {attempts}"
            );
            assert_eq!(
                record_file_names(&guard.store_path()),
                {
                    let mut sorted = names.clone();
                    sorted.sort();
                    sorted
                },
                "case {case}: the record file still describes it — nothing drops an unanswered record from the queue"
            );
        }
        // A record the browser reports as not the extension's to remove is reported on the Issues
        // view once and dropped: the queue never re-asks those names, and the reclaim pass keeps
        // re-reading the browser for such a group anyway.
        {
            let case = "a record the browser reports as not ours is reported once and dropped";
            // The shipped pass budget: the planning reads, a stop's full bound and the
            // confirmation's read reserve, which is what the confirmation step needs to run its
            // stop at all.
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let namespace = ChromeRunSessions::for_run("run-unclosable")
                .namespace()
                .to_string();
            let names = run_session_names("run-unclosable", &["agent-tab-u-default"]);
            let session = names[0].clone();
            // The group is live and holds a tab, but the extension does not own it.
            guard.write_state(&names, false);
            queue_names("run-unclosable", &["agent-tab-u-default"]);

            release_due().await;

            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: the record's route is done with those tabs, so they are not retried"
            );
            let issues = guard.issues();
            assert_eq!(
                issues.len(),
                1,
                "case {case}: the leftover is reported once"
            );
            assert_eq!(issues[0].0, LEFTOVER_MESSAGE, "case {case}");
            assert_eq!(
                issues[0].1,
                format!("chrome-tabs:{namespace}"),
                "case {case}"
            );
            assert_eq!(issues[0].2["run"], "run-unclosable", "case {case}");
            assert_eq!(issues[0].2["sessions"], 1, "case {case}");
            assert!(
                guard.invoked(&format!("session stop --json --session {session}")),
                "case {case}: the session's own route is asked before any tab is called unclosable: {:?}",
                guard.log_lines()
            );

            // A second pass has nothing left to report.
            release_due().await;
            assert_eq!(guard.issues().len(), 1, "case {case}: and never again");
        }
        // An empty ledger is one browser answer, not a verdict on its own: the extension's storage
        // read is not reported to us, so a swallowed one hands back exactly what a lost ledger
        // does.
        {
            let case = "a transient empty ledger does not write a run off";
            // The shipped pass budget, so the pass reaches the confirmation's second read rather
            // than being cut off before it.
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let names = run_session_names("run-amnesia", &["agent-tab-a-default"]);
            guard.write_state(&names, true);
            // The first ledger read comes back empty, every later one is the real ledger.
            guard.set_mode("amnesia-once");
            queue_names("run-amnesia", &["agent-tab-a-default"]);

            release_due().await;

            assert_eq!(
                pending_names_and_attempts().len(),
                1,
                "case {case}: an answer the second read did not repeat is not a verdict"
            );
            assert!(
                guard.issues().is_empty(),
                "case {case}: and nothing is announced about those tabs"
            );

            release_due().await;
            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: the next pass reads the real ledger and closes them"
            );
            assert!(guard.invoked("extension call tabs.remove"), "case {case}");
        }
        // A browser that answers and leaves its own tabs open is a contradiction between two of its
        // own answers, not a verdict from one pass: the record keeps retrying, and the
        // answered-and-left-open name is told durably — once, with the run it came from — rather
        // than ageing out as a `debug!` line nobody ever sees.
        {
            let case = "a close the browser leaves open is reported once and still retried";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            let namespace = ChromeRunSessions::for_run("run-kept")
                .namespace()
                .to_string();
            let names = run_session_names("run-kept", &["agent-tab-k-default"]);
            guard.write_state(&names, true);
            guard.set_mode("keep");
            queue_names("run-kept", &["agent-tab-k-default"]);

            release_due().await;

            assert_eq!(
                pending_names_and_attempts().len(),
                1,
                "case {case}: one left-open answer proves nothing, so the record stays queued"
            );
            assert_eq!(
                record_file_names(&guard.store_path()),
                {
                    let mut sorted = names.clone();
                    sorted.sort();
                    sorted
                },
                "case {case}: and stays in the durable file"
            );
            let issues = guard.issues();
            assert_eq!(
                issues.len(),
                1,
                "case {case}: the left-open answer is reported once"
            );
            assert_eq!(issues[0].0, LEFT_OPEN_MESSAGE, "case {case}");
            assert_eq!(
                issues[0].1,
                format!("chrome-tabs-left-open:{namespace}"),
                "case {case}"
            );
            assert_eq!(issues[0].2["run"], "run-kept", "case {case}");

            let before = pending_names_and_attempts()[0].1;
            release_due().await;
            assert_eq!(
                guard.issues().len(),
                1,
                "case {case}: and never again for the same run's namespace"
            );
            assert!(
                pending_names_and_attempts()[0].1 > before,
                "case {case}: the retry ladder still moves"
            );
        }
        // The extension's own ownership-refusal sentence, relayed under `tabs.remove`, is the only
        // failing leg that counts as the browser answering about the tabs — so it reaches the very
        // conclusion a kept tab does: after [`MAX_LEFT_OPEN_ATTEMPTS`] passes the record stops
        // retrying, and the leftover is reported with the run it came from (the reclaim sweep keeps
        // examining its group).
        {
            let case = "a remove the extension refuses concludes after the cap";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let namespace = ChromeRunSessions::for_run("run-refused-sentence")
                .namespace()
                .to_string();
            let names = run_session_names("run-refused-sentence", &["agent-tab-rs-default"]);
            guard.write_state(&names, true);
            guard.set_mode("remove-sentence");
            queue_names("run-refused-sentence", &["agent-tab-rs-default"]);

            for pass in 1..MAX_LEFT_OPEN_ATTEMPTS {
                release_due().await;
                assert_eq!(
                    pending_names_and_attempts().len(),
                    1,
                    "case {case}: a relayed left-open answer short of the cap keeps the record queued"
                );
                assert_eq!(
                    pending_left_open(),
                    vec![pass],
                    "case {case}: each pass is one browser answer, counted"
                );
            }
            release_due().await;

            assert!(
                pending_names_and_attempts().is_empty(),
                "case {case}: the cap concludes the close route: the record stops retrying"
            );
            let issues = guard.issues();
            let conclusion: Vec<_> = issues
                .iter()
                .filter(|issue| issue.1 == format!("chrome-tabs:{namespace}"))
                .collect();
            assert_eq!(
                conclusion.len(),
                1,
                "case {case}: the leftover is reported once as unclosable: {issues:?}"
            );
            assert_eq!(conclusion[0].0, LEFTOVER_MESSAGE, "case {case}");
            assert_eq!(
                conclusion[0].2["run"], "run-refused-sentence",
                "case {case}: and the conclusion names the run it came from"
            );
            assert!(
                issues
                    .iter()
                    .any(|issue| issue.1 == format!("chrome-tabs-left-open:{namespace}")),
                "case {case}: the browser's own answer was told early, before the cap: {issues:?}"
            );
        }
        // A `tabs.remove` leg that failed before the browser answered — here a relay that is not
        // connected — is NOT the browser answering for the tabs: it never increments `left_open`,
        // so no pass count ever concludes the record, and neither a left-open nor a leftover row is
        // filed for it.
        {
            let case = "a transient remove failure is never counted as an answer";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let namespace = ChromeRunSessions::for_run("run-transient-remove")
                .namespace()
                .to_string();
            let names = run_session_names("run-transient-remove", &["agent-tab-tr-default"]);
            guard.write_state(&names, true);
            guard.set_mode("remove-transient");
            queue_names("run-transient-remove", &["agent-tab-tr-default"]);

            for _ in 0..MAX_LEFT_OPEN_ATTEMPTS {
                release_due().await;
            }

            assert_eq!(
                pending_names_and_attempts(),
                vec![(names.clone(), MAX_LEFT_OPEN_ATTEMPTS)],
                "case {case}: the transient leg leaves the record queued, attempted once per pass"
            );
            assert_eq!(
                pending_left_open(),
                vec![0],
                "case {case}: a leg that never reached the browser is never counted as an answer"
            );
            assert!(
                guard.invoked("extension call tabs.remove"),
                "case {case}: the close did reach the removal leg — the transient failure is that leg's own \
                 answer: {:?}",
                guard.log_lines()
            );
            assert!(
                guard.issues().iter().all(|(_, reason, _)| {
                    reason != &format!("chrome-tabs-left-open:{namespace}")
                        && reason != &format!("chrome-tabs:{namespace}")
                }),
                "case {case}: a transient failure files neither the left-open nor the leftover row: {:?}",
                guard.issues()
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn every_leftover_is_reported_to_the_issues_view_once() {
        // A host with no browser extension at all has no ownership door, so nothing here can
        // enumerate the browser's groups and a leftover no record names is out of reach: the Issues
        // view is told once, with what to do about it, instead of a `debug!` line that ages out of
        // the log store.
        {
            let case = "a host without the ownership door is reported once";
            // The shipped pass budget: the fallback close is a `session stop` by name, and the
            // helper only spawns one with its whole bound left (chrome_tabs' own `stop_budget`).
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            guard.set_mode("no-extension");
            let names = run_session_names("run-no-door", &["agent-tab-n-default"]);
            let session = names[0].clone();
            guard.write_state(&names, true);
            queue_names("run-no-door", &["agent-tab-n-default"]);

            release_due().await;

            let host_rows = || -> Vec<(String, String, serde_json::Value)> {
                guard
                    .issues()
                    .into_iter()
                    .filter(|issue| issue.1 == NO_OWNERSHIP_DOOR_REASON)
                    .collect()
            };
            assert_eq!(
                host_rows().len(),
                1,
                "case {case}: the absent door is reported once: {:?}",
                guard.issues()
            );
            let issues = host_rows();
            assert_eq!(issues[0].0, NO_DOOR_MESSAGE, "case {case}");
            assert!(
                issues[0].2["detail"]
                    .as_str()
                    .is_some_and(|d| !d.is_empty()),
                "case {case}"
            );
            assert_eq!(
                pending_names_and_attempts().len(),
                1,
                "case {case}: a door-less host never writes the run's tabs off: the record keeps trying"
            );
            assert!(
                guard.invoked(&format!("session stop --json --session {session}")),
                "case {case}: the only route such a host has is the helper's own stop"
            );

            // The pass's reads run in the record's own session — the product's own scratch session
            // is not opened at all — so nothing is filed for it.
            assert!(
                guard
                    .issues()
                    .iter()
                    .all(|issue| issue.1 != OWN_SESSION_REASON),
                "case {case}: no scratch-group row: the reads no longer run in the product's own session: {:?}",
                guard.issues()
            );

            // The sweep keeps being woken and keeps failing to read; the report stays one.
            guard.arm_sweep();
            release_due().await;
            assert_eq!(host_rows().len(), 1, "case {case}: and never again");
            assert!(
                guard
                    .issues()
                    .iter()
                    .all(|issue| issue.1 != OWN_SESSION_REASON),
                "case {case}: and no scratch-group row either: the reads run in the record's session"
            );
        }
        // The one visible tab this product can leave in the owner's strip — its own scratch group,
        // when a let-go could not confirm it gone — is surfaced once rather than accepted quietly.
        {
            let case = "a scratch group the let go could not confirm is surfaced once";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            // A scratch group holding a tab the extension does not hold as its own: this route may
            // not remove it, so the let-go cannot confirm the group gone — and the session's own
            // stop is answered with chrome-use's "ownership was preserved" refusal, the shape that
            // leaves the group standing and the let-go unfinished.
            guard.set_mode("stop-refused");
            guard.write_state(&[OWN_SESSION.to_string()], false);
            // No record queued: the pass has nothing to borrow, so its reads run in the product's
            // own session — the shape a parked leftover's revisit takes.
            guard.arm_sweep();

            release_due().await;
            release_due().await;

            let rows: Vec<(String, String, serde_json::Value)> = guard
                .issues()
                .into_iter()
                .filter(|issue| issue.1 == OWN_SESSION_REASON)
                .collect();
            assert_eq!(
                rows.len(),
                1,
                "case {case}: one row for the product's own session: {:?}",
                guard.issues()
            );
            assert_eq!(rows[0].0, OWN_SESSION_MESSAGE, "case {case}");
            let detail = rows[0].2["detail"].as_str().unwrap_or_default();
            assert!(
                detail.contains("the release is not confirmed"),
                "case {case}: and it says only that the release was not confirmed: {detail}"
            );
            assert_eq!(
                pending_forget_snapshot(),
                vec![(OWN_SESSION.to_string(), 2)],
                "case {case}: and the session stays queued with one charged attempt per pass that asked its \
                 stop — the one route left that closes its group was answered with a refusal"
            );
        }
        // A host whose browser answers settle nothing this route can act on is a fact about the
        // HOST, filed once: every automatic close there is a retry with no conclusion, so it must
        // not stay a `debug!` that ages out (see [`UNREADABLE_MESSAGE`]).
        {
            let case = "a host whose answers are unreadable is reported once";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            guard.set_mode("garbled-answer");
            let names = run_session_names("run-garbled", &["agent-tab-garbled"]);
            guard.write_state(&names, true);
            queue_names("run-garbled", &["agent-tab-garbled"]);

            release_due().await;
            release_due().await;

            let rows: Vec<(String, String, serde_json::Value)> = guard
                .issues()
                .into_iter()
                .filter(|issue| issue.1 == UNREADABLE_REASON)
                .collect();
            assert_eq!(
                rows.len(),
                1,
                "case {case}: one row for the host, whatever the passes: {:?}",
                guard.issues()
            );
            assert_eq!(rows[0].0, UNREADABLE_MESSAGE, "case {case}");
            let detail = rows[0].2["detail"].as_str().unwrap_or_default();
            assert!(
                detail.contains("settled nothing")
                    && detail.contains("no parser here will guess at"),
                "case {case}: and it says what settled nothing, and why it is not guessed at: {detail}"
            );
            assert_eq!(
                pending_names_and_attempts().len(),
                1,
                "case {case}: while the record keeps the name queued: nothing was settled"
            );
        }
        // The same host-level fact, for the host shape that is easiest to mistake for a working
        // one: the extension IS installed (and new enough for the door), but Chrome reports it
        // disabled, so nothing answers an `extension call`.
        {
            let case = "a host with a disabled extension is reported once";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            guard.set_mode("disabled-extension");
            let names = run_session_names("run-disabled-ext", &["agent-tab-d-disabled"]);
            let session = names[0].clone();
            guard.write_state(&names, true);
            queue_names("run-disabled-ext", &["agent-tab-d-disabled"]);

            release_due().await;

            let issues: Vec<(String, String, serde_json::Value)> = guard
                .issues()
                .into_iter()
                .filter(|issue| issue.1 == NO_OWNERSHIP_DOOR_REASON)
                .collect();
            assert_eq!(
                issues.len(),
                1,
                "case {case}: a disabled extension's door is missing too, and reported once: {:?}",
                guard.issues()
            );
            assert_eq!(issues[0].0, NO_DOOR_MESSAGE, "case {case}");
            assert!(
                issues[0].2["detail"]
                    .as_str()
                    .is_some_and(|detail| detail.contains("disabled")),
                "case {case}: the row names the disabled case rather than only a missing extension: {issues:?}"
            );
            assert_eq!(
                pending_names_and_attempts().len(),
                1,
                "case {case}: and the record still keeps trying rather than being written off"
            );
            assert!(
                guard.invoked(&format!("session stop --json --session {session}")),
                "case {case}: the fallback close is attempted on this host too: {:?}",
                guard.log_lines()
            );
        }
        // A group the sweep claims but the browser will not close is reported with the group's own
        // namespace, even though no record names it: the record path reports its left-open names,
        // and a reclaimed one must not be the only kind that ages out.
        {
            let case = "a reclaimed group the browser leaves open is reported";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            let orphan = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan");
            guard.write_state(&[orphan], true);
            // The browser answers and leaves its own tab in place.
            guard.set_mode("keep");
            guard.arm_sweep();

            release_due().await;

            let issues = guard.issues();
            assert_eq!(
                issues.len(),
                1,
                "case {case}: the left-open answer is reported: {issues:?}"
            );
            assert_eq!(issues[0].0, LEFT_OPEN_MESSAGE, "case {case}");
            assert_eq!(
                issues[0].1,
                format!("chrome-tabs-left-open:{AGENT_TAB_PREFIX}0123456789ab-"),
                "case {case}"
            );
            assert_eq!(
                issues[0].2["run"], "",
                "case {case}: no record was evicted for a group nothing recorded, so no run can be named"
            );

            // The verdict is remembered: the next sweep skips the group it already answered for,
            // and the report stays one.
            guard.arm_sweep();
            release_due().await;
            assert_eq!(guard.issues().len(), 1, "case {case}: and never again");
        }
    }

    /// The fallback's mapping over an index slice, with more than one affected name and two
    /// DIFFERENT outcomes to place: the name whose stop the helper refuses must stay queued, the
    /// other must settle `Gone`, and the name with no live group must settle on the group read and
    /// never be asked for a stop.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_fallback_maps_each_live_name_back_to_its_own_slot() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        guard.set_mode("ledger-unreadable");
        let names = run_session_names(
            "run-ledger-unreadable",
            &["agent-tab-missing", "agent-tab-refuse", "agent-tab-live"],
        );
        // The first name has no live group (settled by the one group read); the other two do, and
        // the ownership reads that would map their ids back to names both fail.
        guard.write_state(&names[1..], true);
        queue_names(
            "run-ledger-unreadable",
            &["agent-tab-missing", "agent-tab-refuse", "agent-tab-live"],
        );

        release_due().await;

        assert!(
            guard.invoked(&format!("session stop --json --session {}", names[1])),
            "the fallback's stop is asked for every name with a live group: {:?}",
            guard.log_lines()
        );
        assert!(guard.invoked(&format!("session stop --json --session {}", names[2])));
        assert!(
            !guard.invoked(&format!("session stop --json --session {}", names[0])),
            "and never for the name with no live group, which the one group read settled"
        );
        assert_eq!(
            pending_names_and_attempts(),
            vec![(vec![names[1].clone()], 1)],
            "each name keeps its own outcome: the refused stop stays queued with its attempt, \
             the answered one settles"
        );
        assert!(
            guard.issues().is_empty(),
            "and no leftover row is filed — the fallback's retry is silent: {:?}",
            guard.issues()
        );
    }

    /// The sweep claims a live agent group nothing accounts for, and leaves alone every group a run
    /// may still come back to: one whose namespace is live right now, a held record's, an unheld
    /// record's own names, and one the roster of a job that has not been terminalized still names —
    /// a cut round resumes in place under the very same ids, and reads its tabs back by name.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_reclaim_spares_every_group_a_run_may_come_back_to() {
        crate::util::test::init_test_stores().await;
        let store = crate::session::store();
        // A roster slot's own id, in the shape the pipeline mints
        // (`ticket_{id}_{idx}_{suffix}_{role}`): a cut round resumes under the very id it was
        // working with, which is what the reclaim sweep has to spare.
        let resumable_id = "ticket_reclaim_0_resume_analyst";
        crate::util::test::JobRowBuilder::new(
            &store.conn,
            "job-reclaim-cu",
            "analyze",
            "analyst",
            "ws",
        )
        .timestamps("2026-01-01T00:00:00Z")
        .insert()
        .await
        .expect("insert a job that has not been terminalized");
        store
            .conn
            .execute(
                crate::jobs::AGENT_INSERT_SQL,
                crate::jobs::agent_params(
                    "job-reclaim-cu",
                    resumable_id,
                    crate::jobs::AgentKind::Analyst,
                    Some(0),
                    "task",
                ),
            )
            .await
            .expect("insert a roster row");
        // A slot of the same job that already FINISHED: a resume reconstructs it from its stored
        // outcome instead of re-dispatching it, so nothing will work under its id again and its
        // namespace is nothing to keep.
        let finished_id = "ticket_reclaim_1_resume_analyst";
        store
            .conn
            .execute(
                crate::jobs::AGENT_INSERT_SQL,
                crate::jobs::agent_params(
                    "job-reclaim-cu",
                    finished_id,
                    crate::jobs::AgentKind::Analyst,
                    Some(1),
                    "task",
                ),
            )
            .await
            .expect("insert a finished roster row");
        crate::jobs::write_agent_outcome(
            &store.conn,
            "job-reclaim-cu",
            finished_id,
            crate::jobs::RowStatus::Done,
            Some("{}"),
        )
        .await
        .expect("mark the roster slot done");

        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        let resumable = format!(
            "{}default",
            crate::tools::chrome::run_session_namespace(resumable_id)
        );
        let finished = format!(
            "{}default",
            crate::tools::chrome::run_session_namespace(finished_id)
        );
        let held = run_session_names("run-held-rec", &["agent-tab-h-default"]).remove(0);
        queue_names_with_hold(
            "run-held-rec",
            &["agent-tab-h-default"],
            Duration::from_hours(1),
        );
        let claimed = run_session_names("run-claimed-rec", &["agent-tab-c-default"]).remove(0);
        queue_names("run-claimed-rec", &["agent-tab-c-default"]);
        let live = ChromeRunSessions::for_run("run-live-rec");
        let live_name = format!("{}default", live.namespace());
        let orphan = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan");
        guard.write_state(
            &[
                resumable.clone(),
                finished.clone(),
                held.clone(),
                claimed.clone(),
                live_name.clone(),
                orphan.clone(),
            ],
            true,
        );

        let cli = release_cli();
        let protected = ProtectedNamespaces::snapshot(&durable_resume_namespaces().await);
        let reclaim = reclaim_names(
            cli.as_deref(),
            OWN_SESSION,
            &protected,
            Instant::now() + Duration::from_secs(10),
            true,
        )
        .await;

        // Undo the seeded protection BEFORE asserting: the shared test store must not keep naming a
        // run that may come back, or every later test in this process reads a resume set this one
        // left behind — including when an assertion below fails.
        drop(live);
        store
            .conn
            .execute(
                "DELETE FROM jobs WHERE id = ?1",
                crate::db::params!["job-reclaim-cu"],
            )
            .await
            .expect("delete the test job");
        assert!(
            reclaim.names.contains(&orphan),
            "the group nothing accounts for is the one to reclaim: {:?}",
            reclaim.names
        );
        for spared in [&resumable, &held, &claimed, &live_name] {
            assert!(
                !reclaim.names.contains(spared),
                "a name a run may come back to is never reclaimed: {spared}: {:?}",
                reclaim.names
            );
        }
        assert!(
            reclaim.names.contains(&finished),
            "a slot that already finished is never re-dispatched, so its group is reclaimed like \
             any other one nothing will work under again: {:?}",
            reclaim.names
        );
    }

    /// A stuck leftover is not interrogated again every sweep, and its verdict is dropped once
    /// expired so the group is examined again.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_stuck_verdict_expires_and_is_examined_again() {
        // A leftover the browser has already answered twice can never be closed is not interrogated
        // in full on every sweep — the verdict is remembered, so a stuck group cannot cost a
        // `session stop` per name on each pass — while the report stays one.
        {
            let case = "a stuck leftover is not interrogated again every sweep";
            // The shipped pass budget: the planning reads, a stop's full bound and the
            // confirmation's read reserve, which is what the confirmation step needs to run its
            // stop at all.
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let orphan = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan");
            // The group holds a tab the extension does not own: this route can never close it,
            // which the second read repeats.
            guard.write_state(std::slice::from_ref(&orphan), false);
            guard.arm_sweep();

            release_due().await;

            let issues = guard.issues();
            assert_eq!(
                issues.len(),
                1,
                "case {case}: the leftover is reported: {issues:?}"
            );
            assert_eq!(issues[0].0, LEFTOVER_MESSAGE, "case {case}");
            assert_eq!(
                issues[0].1,
                format!("chrome-tabs:{AGENT_TAB_PREFIX}0123456789ab-"),
                "case {case}"
            );

            guard.clear_log();
            guard.arm_sweep();
            release_due().await;
            assert!(
                !guard.invoked(&format!("session stop --json --session {orphan}")),
                "case {case}: a remembered verdict is not re-confirmed every sweep: {:?}",
                guard.log_lines()
            );
            assert_eq!(
                guard.issues().len(),
                1,
                "case {case}: and never reported again either"
            );
        }
        // A verdict is not forever: once its [`STUCK_REVISIT`] window has passed the prune drops
        // it, and the next armed sweep examines the group in full again — the stop route included —
        // instead of skipping it for the rest of the process.
        {
            let case = "an expired stuck verdict is dropped and its group examined again";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let orphan = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan");
            guard.write_state(std::slice::from_ref(&orphan), false);
            guard.arm_sweep();

            release_due().await;
            assert_eq!(
                guard.issues().len(),
                1,
                "case {case}: the leftover is reported once"
            );

            // The window has passed: the verdict's age is the sole thing that expires it.
            let expired = Instant::now()
                .checked_sub(STUCK_REVISIT)
                .expect("the process clock reaches back the revisit window");
            stuck_namespaces().lock().unwrap_poison().insert(
                crate::tools::chrome::session_namespace(&orphan).to_string(),
                expired,
            );

            guard.clear_log();
            guard.arm_sweep();
            release_due().await;

            assert!(
                guard.invoked(&format!("session stop --json --session {orphan}")),
                "case {case}: the group is examined again, not skipped by an expired verdict: {:?}",
                guard.log_lines()
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_let_go_queue_retires_each_scratch_session() {
        // The session a pass with no name to borrow reads in ([`OWN_SESSION`]) is let go by the
        // pass that made the reads, through the same door a run's tabs are closed by — so a
        // re-minted relay endpoint cannot orphan its scratch group — and a pass that read nothing
        // leaves it alone.
        {
            let case = "the pass lets go of the session its reads ran in";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            // The scratch group a reading pass opens, holding the tab its daemon's first command
            // opened in it.
            guard.write_state(&[OWN_SESSION.to_string()], true);
            let cli = release_cli().expect("the stub binary");
            let deadline = Instant::now() + Duration::from_secs(10);

            // A read is the only thing that starts the session, and the pass's teardown follows.
            chrome_tabs::agent_group_names(&cli, OWN_SESSION, deadline)
                .await
                .expect("the live groups");
            chrome_tabs::release_own_session(&cli, deadline).await;

            assert_eq!(
                guard.removed_tab_ids(),
                vec![100],
                "case {case}: the scratch group is closed through the extension's own door: {:?}",
                guard.log_lines()
            );
            assert!(
                guard.invoked(&format!(
                    "session stop --force --json --session {OWN_SESSION}"
                )),
                "case {case}: the group the ledger door confirmed gone is followed by the record-dropping stop: \
                 {:?}",
                guard.log_lines()
            );

            // Nothing outstanding: a pass that asked the browser nothing stops nothing.
            guard.clear_log();
            chrome_tabs::release_own_session(&cli, deadline).await;
            assert!(
                guard.log_lines().is_empty(),
                "case {case}: an idle pass spawns nothing: {:?}",
                guard.log_lines()
            );
        }
        // A scratch group holding a tab the extension does not hold as its own is not closed from
        // this module's ledger route — asking for an id the extension does not hold is refused
        // whole — so the let-go hands the name to the queue, and the pass's let-go step is the ONE
        // place a graceful `session stop` is asked for: it is the verb that closes the tabs a
        // session created.
        {
            let case = "a scratch group the ledger cannot close goes to its session stop";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            guard.set_mode("stop-refused");
            guard.write_state(&[OWN_SESSION.to_string()], false);
            let cli = release_cli().expect("the stub binary");
            let deadline = Instant::now() + ATTEMPT_TIMEOUT;

            // A read is what makes the let-go owe the session, so it does not return early on the
            // marker check.
            chrome_tabs::agent_group_names(&cli, OWN_SESSION, deadline)
                .await
                .expect("the live groups");
            let_go_own_session(&cli, deadline).await;

            assert!(
                guard.removed_tab_ids().is_empty(),
                "case {case}: a tab the extension does not hold is never asked for: {:?}",
                guard.log_lines()
            );
            assert!(
                !guard.invoked("session stop"),
                "case {case}: and the unconfirmed let-go does not stop the session itself — the pass's let-go \
                 step is the one stop site: {:?}",
                guard.log_lines()
            );
            assert!(
                pending_forget_snapshot() == vec![(OWN_SESSION.to_string(), 0)],
                "case {case}: it hands the session to the driver's retry path instead: {:?}",
                guard.log_lines()
            );

            // The pass's own let-go step: it asks the closing verb exactly once, and the answer it
            // hears is charged.
            guard.clear_log();
            forget_settled(&cli, &[], deadline).await;
            let stops = guard
                .log_lines()
                .iter()
                .filter(|line| line.starts_with("session stop"))
                .count();
            assert_eq!(
                stops,
                1,
                "case {case}: one stop for the pass, with the verb that closes the group: {:?}",
                guard.log_lines()
            );
            assert!(
                guard.invoked(&format!("session stop --json --session {OWN_SESSION}")),
                "case {case}: and it is the graceful form: {:?}",
                guard.log_lines()
            );
            assert!(
                !guard.invoked(&format!(
                    "session stop --force --json --session {OWN_SESSION}"
                )),
                "case {case}: never the form that would drop the record and leave the group standing: {:?}",
                guard.log_lines()
            );
            assert!(
                pending_forget_snapshot() == vec![(OWN_SESSION.to_string(), 1)],
                "case {case}: a whole-bound stop that could not close it charges one attempt and keeps the name \
                 queued: {:?}",
                guard.log_lines()
            );
        }
        // A scratch group the ledger route cannot close, whose own stop then lands, ends the whole
        // debt: that stop settled the session — its record and the group — so the reads it covered
        // are no longer owed.
        {
            let case = "a settled scratch session is not started again by the next pass";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            // A tab the extension does not hold as its own: the ledger route may not ask for it, so
            // the pass hands the session to the let-go queue and its own step asks the stop.
            guard.write_state(&[OWN_SESSION.to_string()], false);
            let cli = release_cli().expect("the stub binary");
            let deadline = Instant::now() + ATTEMPT_TIMEOUT;

            chrome_tabs::agent_group_names(&cli, OWN_SESSION, deadline)
                .await
                .expect("the live groups");
            let_go_own_session(&cli, deadline).await;
            assert_eq!(
                pending_forget_snapshot(),
                vec![(OWN_SESSION.to_string(), 0)],
                "case {case}: the unconfirmed let-go is queued"
            );

            guard.clear_log();
            forget_settled(&cli, &[], deadline).await;
            assert!(
                guard.invoked(&format!("session stop --json --session {OWN_SESSION}")),
                "case {case}: the let-go step asks the graceful stop: {:?}",
                guard.log_lines()
            );
            assert!(
                pending_forget_snapshot().is_empty(),
                "case {case}: which settles the session and drops the name"
            );

            guard.clear_log();
            chrome_tabs::release_own_session(&cli, deadline).await;
            assert!(
                guard.log_lines().is_empty(),
                "case {case}: and the next pass owes nothing for those reads: {:?}",
                guard.log_lines()
            );
        }
        // Two passes over the let-go queue can overlap — the driver and the shutdown flush — and
        // neither may lose the other's names: the queue is read and written back, never taken out.
        {
            let case = "overlapping let go passes keep each others names";
            let budget = Duration::from_millis(300);
            let guard = ReleaseGuard::install(budget).await;
            // A silent helper: both stops are cut short at the pass budget, so each pass keeps its
            // own name — and would write the other's away if it had taken the queue out wholesale.
            guard.set_mode("silent");
            let cli = release_cli().expect("the stub binary");
            let deadline = Instant::now() + budget;
            let first = run_session_names("run-forget-overlap-a", &["agent-tab-fo-a"]).remove(0);
            let second = run_session_names("run-forget-overlap-b", &["agent-tab-fo-b"]).remove(0);

            tokio::join!(
                forget_settled(&cli, std::slice::from_ref(&first), deadline),
                forget_settled(&cli, std::slice::from_ref(&second), deadline),
            );

            let queued: Vec<String> = pending_forget_snapshot()
                .into_iter()
                .map(|(name, _)| name)
                .collect();
            assert!(
                queued.contains(&first) && queued.contains(&second),
                "case {case}: both passes' names survive: {queued:?}"
            );
        }
        // The sessions waiting to be let go are kept without repeats and bounded: a burst of
        // settled sessions the helper will not stop must not grow the queue without bound.
        {
            let case = "the let go queue is deduped and bounded";
            let mut names = vec![
                ("s0".to_string(), 3),
                ("s1".to_string(), 0),
                ("s0".to_string(), 0),
                ("s2".to_string(), 1),
            ];
            dedupe_and_bound_forgotten(&mut names);
            assert_eq!(
                names,
                vec![
                    ("s0".to_string(), 3),
                    ("s1".to_string(), 0),
                    ("s2".to_string(), 1)
                ],
                "case {case}: a repeat keeps its first place, attempts and all"
            );

            let mut full: Vec<(String, u32)> = (0..MAX_PENDING_FORGET)
                .map(|i| (format!("s{i}"), 0))
                .collect();
            full.push(("newest".to_string(), 0));
            dedupe_and_bound_forgotten(&mut full);
            assert_eq!(
                full.len(),
                MAX_PENDING_FORGET,
                "case {case}: the queue is bounded"
            );
            assert_eq!(
                full[0],
                ("s1".to_string(), 0),
                "case {case}: the oldest name is what the bound drops"
            );
            assert_eq!(full.last(), Some(&("newest".to_string(), 0)), "case {case}");

            // The product's own scratch session is the one name the cap never drops, even from the
            // place the cap drops first: what it leaves when its stop fails is a group in the
            // owner's tab strip, not helper bookkeeping.
            let mut with_scratch: Vec<(String, u32)> = (0..MAX_PENDING_FORGET)
                .map(|i| (format!("s{i}"), 0))
                .collect();
            with_scratch.insert(0, (OWN_SESSION.to_string(), 0));
            dedupe_and_bound_forgotten(&mut with_scratch);
            assert_eq!(with_scratch[0], (OWN_SESSION.to_string(), 0), "case {case}");
            assert!(
                !with_scratch.iter().any(|(name, _)| name == "s0"),
                "case {case}: so the oldest bookkeeping name is what the cap drops instead"
            );
        }
        // A settled session the pass had no budget to let go is not lost: it waits — with the rung
        // it had, since the pass never asked the helper anything — and the next pass tries it
        // again.
        {
            let case = "a settled session left over is let go by a later pass";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let cli = release_cli().expect("the stub binary");
            let name = run_session_names("run-forget", &["agent-tab-f-default"]).remove(0);

            // A pass whose budget is already spent lets go of nothing and keeps the name — charging
            // it no attempt, because its stop was never asked.
            forget_settled(&cli, std::slice::from_ref(&name), Instant::now()).await;
            assert_eq!(
                pending_forget_snapshot(),
                vec![(name.clone(), 0)],
                "case {case}"
            );
            assert!(
                !guard.invoked("session stop"),
                "case {case}: and spawned nothing: {:?}",
                guard.log_lines()
            );
            assert!(
                forget_retry_at().is_some(),
                "case {case}: while the queue arms its own retry, so a budget-starved pass is not the last word"
            );

            // More starved passes than the bound: a stop the pass never asked is deferred work, not
            // a failure, so the name neither climbs toward `MAX_FORGET_ATTEMPTS` nor is dropped
            // while its group may still be standing.
            for _ in 0..MAX_FORGET_ATTEMPTS {
                forget_settled(&cli, &[], Instant::now()).await;
            }
            assert_eq!(
                pending_forget_snapshot(),
                vec![(name.clone(), 0)],
                "case {case}"
            );

            // The next pass has budget, so the leftover is let go and nothing waits.
            forget_settled(&cli, &[], Instant::now() + ATTEMPT_TIMEOUT).await;
            assert!(pending_forget_snapshot().is_empty(), "case {case}");
            assert!(
                guard.invoked("session stop"),
                "case {case}: the helper's own recipe is what clears the stale record: {:?}",
                guard.log_lines()
            );
        }
        // A stop the pass could only cut short is best-effort: the helper is asked (it often
        // answers well inside the bound) and a name it took leaves the queue, but a name it did not
        // take keeps the rung it had — a pass that could not hear a whole answer must not walk a
        // session toward [`MAX_FORGET_ATTEMPTS`] and drop it while its group may stand.
        {
            let case = "a stop cut to the pass is not charged an attempt";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            guard.set_mode("refuse");
            let cli = release_cli().expect("the stub binary");
            let name = run_session_names("run-cut-stop", &["agent-tab-cs-default"]).remove(0);
            *pending_forget().lock().unwrap_poison() = vec![(name.clone(), 2)];

            // Ten seconds of pass left: a stop, but not a whole one.
            forget_settled(&cli, &[], Instant::now() + Duration::from_secs(10)).await;

            assert!(
                guard.invoked("session stop"),
                "case {case}: the stop is still attempted: {:?}",
                guard.log_lines()
            );
            assert_eq!(
                pending_forget_snapshot(),
                vec![(name, 2)],
                "case {case}: and its refused answer charges nothing: the rung is where it was"
            );
        }
    }

    /// The let-go queue's own rules, each case in the order the queue meets it: the shutdown
    /// flush's last chance (its budget is below one whole stop, so only a helper that answers
    /// inside it lets a name go), the attempt bound that drops a name instead of retrying it for
    /// the whole process, the product's own scratch session that the bound never drops, a name
    /// behind a call of ours (deferred as busy, woken at one whole stop), a name a run owns again
    /// (leaves the queue unasked), and a retry that cannot wake a driver with no binary.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_let_go_queue_keeps_to_its_own_rules() {
        // The shutdown flush is the let-go queue's last chance before the process goes down, and it
        // has to reach the queue even when no record is due — the unconfirmed let-go of this
        // module's own scratch session is queued precisely when its own pass is ending.
        {
            let case = "the shutdown flush reaches the let go queue";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let name = run_session_names("run-flush-forget", &["agent-tab-ff-default"]).remove(0);
            queue_forget(&name);
            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: no record is due"
            );

            flush_pending_run_releases(SHUTDOWN_RELEASE_FLUSH_BUDGET).await;

            assert!(
                guard.invoked(&format!("session stop --force --json --session {name}")),
                "case {case}: the queued session is stopped before the process goes down: {:?}",
                guard.log_lines()
            );
            assert!(
                pending_forget_snapshot().is_empty(),
                "case {case}: and a helper that answers inside the flush's budget lets the name go"
            );
        }
        // A let-go that never lands is bounded rather than retried forever: after
        // [`MAX_FORGET_ATTEMPTS`] passes whose stop the helper did not settle, a run's session name
        // leaves the queue with one `info!` naming it — a log trace, not an Issues row
        // (chrome-use's own record under `~/.chrome-use` is all that is left behind by it, never a
        // tab).
        {
            let case = "the let go queue gives up after its attempt bound";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            // Every `session stop` fails, so no name is ever let go.
            guard.set_mode("refuse");
            let cli = release_cli().expect("the stub binary");
            let name = run_session_names("run-forget-bound", &["agent-tab-fb-default"]).remove(0);
            // A pass-length budget, because only a stop that got the whole bound may be charged a
            // failed attempt — that is what the bound counts.
            let deadline = Instant::now() + ATTEMPT_TIMEOUT;

            for attempts in 1..MAX_FORGET_ATTEMPTS {
                forget_settled(&cli, std::slice::from_ref(&name), deadline).await;
                assert_eq!(
                    pending_forget_snapshot(),
                    vec![(name.clone(), attempts)],
                    "case {case}: the name waits with one more failed attempt on it"
                );
            }
            forget_settled(&cli, &[], deadline).await;
            assert!(
                pending_forget_snapshot().is_empty(),
                "case {case}: the bound drops it instead of retrying it for the whole process"
            );
        }
        // The product's own scratch session is the one name the attempt bound never drops: what it
        // can leave behind is a group in the owner's tab strip rather than chrome-use's
        // bookkeeping, so once a stop has failed [`MAX_FORGET_ATTEMPTS`] times it stays in the
        // queue at the bound's rung and the queue's own deadline keeps retrying it — the
        // alternative would be the group waiting for the next process start.
        {
            let case = "the scratch session is retried past the attempt bound";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            guard.set_mode("stop-refused");
            let cli = release_cli().expect("the stub binary");
            let deadline = Instant::now() + ATTEMPT_TIMEOUT;

            for _ in 0..MAX_FORGET_ATTEMPTS {
                forget_settled(&cli, &[OWN_SESSION.to_string()], deadline).await;
            }
            assert_eq!(
                pending_forget_snapshot(),
                vec![(OWN_SESSION.to_string(), MAX_FORGET_ATTEMPTS)],
                "case {case}: the scratch session stays queued past the bound"
            );
            assert!(
                forget_retry_at().is_some(),
                "case {case}: and the queue's own deadline is what retries it, not a restart"
            );
        }
        // A name the pass could not ask a stop of because a call of ours was still running in its
        // session is deferred as busy, not failed: the helper is not asked, nothing is charged, and
        // the queue's own wake is floored at one whole stop — nothing is owed until that call is
        // done, and the pass holding it covers the name at its own let-go step, so waking at the
        // base gap would only bring the driver back to defer again.
        {
            let case = "a let go behind a call of ours waits the stop bound";
            let _guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let cli = release_cli().expect("the stub binary");
            let deadline = Instant::now() + ATTEMPT_TIMEOUT;
            // A call of ours, held for the whole test: what a pass overlapping this one is running.
            let call = chrome_tabs::hold_door_call_in_flight();

            let outcome = chrome_tabs::forget_settled_sessions(
                &cli,
                std::slice::from_ref(&OWN_SESSION.to_string()),
                deadline,
                &|_: &str| false,
            )
            .await;
            assert_eq!(outcome.busy, vec![OWN_SESSION.to_string()], "case {case}");
            assert!(
                outcome.deferred.is_empty() && outcome.tried.is_empty(),
                "case {case}: and it is neither failed nor a deferral of a bound that did not fit"
            );

            forget_settled(&cli, &[OWN_SESSION.to_string()], deadline).await;
            assert_eq!(
                pending_forget_snapshot(),
                vec![(OWN_SESSION.to_string(), 0)],
                "case {case}: the name is kept, at the rung it had"
            );
            let wait = forget_retry_at()
                .expect("while the call holds the session the queue arms its own retry")
                .saturating_duration_since(Instant::now());
            assert!(
                BUSY_FORGET_FLOOR.saturating_sub(wait) < Duration::from_secs(1),
                "case {case}: and the wake waits out one whole stop, not the base gap: {wait:?}"
            );
            drop(call);
        }
        // A name a run owns again leaves the let-go queue instead of climbing it: stopping that
        // session is what would end the run, whose own end hands over its own release — so the
        // helper is never asked, and the name is not charged a failed attempt for it.
        {
            let case = "a session a run owns again leaves the let go queue";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            guard.set_mode("refuse");
            let cli = release_cli().expect("the stub binary");
            let name = run_session_names("run-owned-again", &["agent-tab-ro-default"]).remove(0);
            let deadline = Instant::now() + Duration::from_secs(10);
            // The run's tracker is held for the whole test, so its namespace is live: `owned_again`
            // is the rule a resumed run is protected by.
            let tracker = ChromeRunSessions::for_run("run-owned-again");

            forget_settled(&cli, std::slice::from_ref(&name), deadline).await;

            assert!(
                !guard.invoked("session stop"),
                "case {case}: the stop would end the run's own session: {:?}",
                guard.log_lines()
            );
            assert!(
                pending_forget_snapshot().is_empty(),
                "case {case}: and nothing waits for a name the run owns"
            );
            drop(tracker);
        }
        // A let-go retry left behind by an earlier pass must not keep the driver awake while no
        // chrome-use binary is resolvable: nothing can run the queue then, so a deadline already in
        // the past would be read as due by every loop turn, with a pass that can do nothing — a
        // spin, not a retry.
        {
            let case = "a let go retry cannot wake a driver with no binary";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            // A retry the queue left behind before the binary went away: already due.
            *forget_retry().lock().unwrap_poison() = Some(Instant::now());

            assert!(
                next_release_deadline().is_some(),
                "case {case}: with a binary to run it, the retry is a real deadline"
            );

            guard.set_binary_absent();
            assert_eq!(
                next_release_deadline(),
                None,
                "case {case}: with none, nothing reads the queue's past deadline as work"
            );
        }
    }

    /// The sweep's own gate: the ask is not on a timer — a pass woken by an enqueue does not ask
    /// the browser again once the last ask is done with, so a quiet queue cannot cost a chrome-use
    /// child per wake; an event arms the next ask, and only then does the sweep read the browser
    /// again.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_sweep_is_armed_by_events_and_revisits() {
        {
            let case = "the reclaim asks the browser once per armed ask";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let orphan = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan");
            guard.write_state(&[orphan], true);

            // Arm the ask a run end would; the sweep does not arm one at boot.
            guard.arm_sweep();
            release_due().await;
            assert!(
                guard.invoked("extension call tabs.remove"),
                "case {case}: the first pass reclaims"
            );
            assert!(
                next_release_deadline().is_none(),
                "case {case}: a clean ask leaves nothing scheduled — no ask, no queued retry"
            );
            guard.clear_log();

            release_due().await;
            assert!(
                guard.log_lines().is_empty(),
                "case {case}: a successful ask schedules nothing, so no interval re-asks: {:?}",
                guard.log_lines()
            );

            schedule_reclaim_now();
            release_due().await;
            assert!(
                guard.invoked("extension call tabGroups.query"),
                "case {case}: an event — a run end queuing names — is what arms the next ask"
            );
        }
        // A record whose leftover no route here can close is not abandoned with it: once the record
        // is dropped and the leftover reported, the sweep is armed to look at the group again at
        // its own slower cadence, not at once.
        {
            let case = "a record settled unclosable arms the sweep";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let names = run_session_names("run-unclosable", &["agent-tab-u-default"]);
            // The group holds a tab the extension does not own, so no route here can close it.
            guard.write_state(&names, false);
            queue_names("run-unclosable", &["agent-tab-u-default"]);

            release_due().await;

            assert!(
                pending_names_and_attempts().is_empty(),
                "case {case}: the record stops retrying the names no route can close"
            );
            let issues = guard.issues();
            assert_eq!(issues.len(), 1, "case {case}: {issues:?}");
            assert_eq!(issues[0].0, LEFTOVER_MESSAGE, "case {case}");
            assert_eq!(issues[0].2["run"], "run-unclosable", "case {case}");
            let remaining = next_release_deadline()
                .expect("the driver wakes for the group the record left behind")
                .saturating_duration_since(Instant::now());
            assert!(
                STUCK_REVISIT.saturating_sub(remaining) < Duration::from_secs(1),
                "case {case}: the sweep is armed for the revisit, not at once: {remaining:?}"
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn silent_sweeps_climb_one_rung_per_pass() {
        // A sweep whose read succeeds but whose close leg comes back Silent is unfinished, and the
        // ladder has to see that: repeated Silent passes climb the capped backoff instead of
        // re-arming at the first rung every few seconds.
        {
            let case = "repeated silent sweeps climb the ladder";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            // A non-zero base: the guard ships `retry_base` as zero so an ordinary retry can be
            // exercised without waiting, and this test is about the spacing the ladder derives.
            widen_retry_base(Duration::from_secs(5));
            let orphan = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan");
            // An owned tab whose remove leg never reaches the browser: the read succeeds, the close
            // is a Silent retry, so the ask is unfinished rather than answered.
            guard.write_state(&[orphan], true);
            guard.set_mode("remove-transient");
            guard.arm_sweep();

            release_due().await;
            let after_first = reclaim_schedule().lock().unwrap_poison().failures;
            let first = next_release_deadline()
                .expect("an unfinished ask keeps the next one armed")
                .saturating_duration_since(Instant::now());

            reclaim_schedule().lock().unwrap_poison().next_at = Some(Instant::now());
            release_due().await;
            let second = next_release_deadline()
                .expect("the second unfinished ask keeps one armed too")
                .saturating_duration_since(Instant::now());

            assert_eq!(
                after_first, 1,
                "case {case}: the Silent answer charged the ladder's first rung"
            );
            assert!(
                first > Duration::ZERO,
                "case {case}: the first ask is spaced by the base, not at the floor: {first:?}"
            );
            assert!(
                second > first + Duration::from_secs(2),
                "case {case}: the second Silent pass took a further rung rather than re-arming at the floor: \
                 {first:?} then {second:?}"
            );

            // A later ask the browser answers cleanly leaves the ladder at its floor again.
            guard.set_mode("");
            reclaim_schedule().lock().unwrap_poison().next_at = Some(Instant::now());
            release_due().await;
            assert_eq!(
                reclaim_schedule().lock().unwrap_poison().failures,
                0,
                "case {case}: a clean answered ask resets the ladder"
            );
        }
        // One ask, one rung: the sweep's failure is the pass, not how many of its names the browser
        // left unanswered — two Silent leftovers in one pass take a single rung.
        {
            let case = "one pass with several silent leftovers takes one rung";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            widen_retry_base(Duration::from_secs(5));
            let first = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan-one");
            let second = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan-two");
            guard.write_state(&[first, second], true);
            guard.set_mode("remove-transient");
            guard.arm_sweep();

            release_due().await;
            assert_eq!(
                reclaim_schedule().lock().unwrap_poison().failures,
                1,
                "case {case}: one pass, one rung, however many names it left unanswered"
            );
        }
        // The one thing a sweep that found a leftover it could not close leaves behind is that
        // leftover's revisit: nothing is periodic, so a pass right after asks nothing while the ask
        // the driver waits on is still parked, and the armed ask is what asks the browser again
        // once it comes due — no new event needed.
        {
            let case = "a leftover the sweep could not settle keeps an ask armed";
            let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
            let orphan = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan");
            // The group holds a tab the extension does not own, so this route can never close it.
            guard.write_state(std::slice::from_ref(&orphan), false);
            guard.arm_sweep();

            release_due().await;

            let parked = reclaim_schedule()
                .lock()
                .unwrap_poison()
                .next_at
                .expect("an ask is armed");
            let revisit = next_reclaim_revisit().expect("the parked leftover owes a revisit");
            let drift = parked
                .saturating_duration_since(revisit)
                .max(revisit.saturating_duration_since(parked));
            assert!(
                drift < Duration::from_secs(1),
                "case {case}: the parked leftover's revisit is what the ask carries: {parked:?} vs {revisit:?}"
            );
            assert!(
                parked > Instant::now(),
                "case {case}: and it is in the future — the sweep does not spin on it"
            );
            assert!(
                next_release_deadline().is_some(),
                "case {case}: the driver wakes for that ask by itself, with no event"
            );
            guard.clear_log();
            release_due().await;
            assert!(
                guard.log_lines().is_empty(),
                "case {case}: and nothing asks before it is due: {:?}",
                guard.log_lines()
            );

            // The only thing time does to an armed ask: bring it due.
            reclaim_schedule().lock().unwrap_poison().next_at = Some(Instant::now());
            guard.clear_log();
            release_due().await;
            assert!(
                guard.invoked("extension call tabGroups.query"),
                "case {case}: the armed ask is what asks the browser again, with no event: {:?}",
                guard.log_lines()
            );
        }
    }

    /// The sweep leaves a record's own work alone by exact name, so a record that settled one name
    /// unclosable and left another retrying keeps only the retrying name — and the name it dropped
    /// is examined by the sweep at the revisit, so a browser that recovered still gets that group
    /// closed.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_name_a_record_dropped_is_still_swept() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        // Two names of one run: the first holds a tab the extension does not own (unclosable by
        // this route), the second a closeable one whose remove leg never reaches the browser (a
        // silent retry), so the record ends the pass holding only the second name.
        let names = run_session_names("run-mixed", &["agent-tab-m-stuck", "agent-tab-m-keep"]);
        guard.write_state_each(&[(&names[0], false), (&names[1], true)]);
        guard.set_mode("remove-transient");
        queue_names("run-mixed", &["agent-tab-m-stuck", "agent-tab-m-keep"]);

        release_due().await;

        assert_eq!(
            pending_names_and_attempts(),
            vec![(vec![names[1].clone()], 1)],
            "the record drops the unclosable name and keeps retrying the other"
        );

        // The browser recovered: the abandoned group's tab is closeable again, so the sweep is the
        // route that closes it.
        guard.write_state(std::slice::from_ref(&names[0]), true);
        guard.set_mode("ok");
        guard.clear_log();
        reclaim_schedule().lock().unwrap_poison().next_at = Some(Instant::now());
        release_due().await;

        assert!(
            guard.removed_tab_ids().contains(&100),
            "the sweep examines the group the record dropped instead of skipping it as \
             claimed: {:?}",
            guard.log_lines()
        );
        assert!(
            pending_names_and_attempts().is_empty(),
            "the retried name's group is gone as well, so the record settles"
        );
    }

    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_cap_evicts_only_an_unheld_record() {
        // The queue cap is kept and the record an overflow drops is the oldest unheld one: it is
        // logged, its run is remembered so the sweep's report of the tabs can still name it, and
        // the tabs themselves are left to the reclaim sweep, which owns an `agent-tab-*` group no
        // live run, no record and no resumable job claims.
        {
            let case = "an overflow eviction drops the oldest unheld record";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            clear_pending_releases();
            for i in 0..MAX_PENDING_RELEASES {
                queue_names(&format!("run-evict-{i}"), &["agent-tab-e-default"]);
            }
            guard.clear_issues();

            queue_names("run-evict-overflow", &["agent-tab-e-overflow"]);
            // No binary: the pass deals with the eviction without attempting 256 closes.
            guard.set_binary_absent();
            release_due().await;

            let queued = pending_names_and_attempts();
            assert_eq!(
                queued.len(),
                MAX_PENDING_RELEASES,
                "case {case}: the cap is kept"
            );
            let evicted = ChromeRunSessions::for_run("run-evict-0")
                .namespace()
                .to_string();
            assert!(
                !queued
                    .iter()
                    .any(|(names, _)| names.iter().any(|name| name.starts_with(&evicted))),
                "case {case}: the oldest unheld record was the one evicted"
            );
            assert!(
                queued.iter().any(|(names, _)| names
                    .iter()
                    .any(|name| name.ends_with("agent-tab-e-overflow"))),
                "case {case}: the record the cap made room for is queued"
            );
            assert_eq!(
                evicted_run(&evicted).as_deref(),
                Some("run-evict-0"),
                "case {case}: the evicted record's run is remembered for the sweep's report"
            );
            assert!(
                guard.issues().is_empty(),
                "case {case}: an eviction closes nothing and reports nothing: the sweep owns those tabs"
            );
            clear_pending_releases();
        }
        // The run an eviction remembered is what the sweep's own report names: the evicted record
        // cannot name its tabs any more, so a leftover it could not close would otherwise be
        // reported with no run at all.
        {
            let case = "an evicted records run is named by the sweeps report";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            clear_pending_releases();
            for i in 0..MAX_PENDING_RELEASES {
                queue_names(&format!("run-evict-{i}"), &["agent-tab-e-default"]);
            }
            queue_names("run-evict-overflow", &["agent-tab-e-overflow"]);
            // The evicted record's group is live, and the browser answers the sweep's close by
            // leaving its own tab standing — an answer the sweep reports.
            let evicted = run_session_names("run-evict-0", &["agent-tab-e-default"]);
            guard.write_state(&evicted, true);
            guard.set_mode("keep");

            release_due().await;

            let namespace = ChromeRunSessions::for_run("run-evict-0")
                .namespace()
                .to_string();
            let issues = guard.issues();
            assert_eq!(issues.len(), 1, "case {case}: {issues:?}");
            assert_eq!(issues[0].0, LEFT_OPEN_MESSAGE, "case {case}");
            assert_eq!(
                issues[0].1,
                format!("chrome-tabs-left-open:{namespace}"),
                "case {case}"
            );
            assert_eq!(
                issues[0].2["run"], "run-evict-0",
                "case {case}: the eviction's remembered run is what the sweep's report carries"
            );
            clear_pending_releases();
        }
        // The attribution an eviction leaves is what the sweep's own report carries — the record
        // cannot name its tabs any more, so a leftover it could not close would otherwise be
        // reported with no run at all — and it is bounded: a stale entry is not read, and the
        // newest entry for a namespace is the one a re-eviction leaves.
        {
            let case = "an evicted run is remembered for the sweeps report";
            clear_pending_releases();
            assert_eq!(
                evicted_run("agent-tab-x-"),
                None,
                "case {case}: nothing remembered yet"
            );

            remember_evicted_run("agent-tab-x-", "run-x");
            remember_evicted_run("agent-tab-y-", "run-y");
            remember_evicted_run("agent-tab-x-", "run-x-again");
            assert_eq!(
                evicted_run("agent-tab-x-").as_deref(),
                Some("run-x-again"),
                "case {case}"
            );
            assert_eq!(
                evicted_run("agent-tab-y-").as_deref(),
                Some("run-y"),
                "case {case}"
            );
            assert_eq!(evicted_run("agent-tab-z-"), None, "case {case}");

            // A remembered run expires with its window rather than with the process.
            let stale = Instant::now()
                .checked_sub(EVICTED_RUN_TTL)
                .expect("the process clock reaches back an hour");
            evicted_runs().lock().unwrap_poison()[1].at = stale;
            remember_evicted_run("agent-tab-z-", "run-z");
            assert_eq!(
                evicted_run("agent-tab-y-"),
                None,
                "case {case}: the stale entry is gone"
            );
            assert_eq!(
                evicted_run("agent-tab-z-").as_deref(),
                Some("run-z"),
                "case {case}"
            );
            clear_pending_releases();
        }
        // The cap never takes a record that is still inside its hold: one that is not always goes
        // first, and with every record inside a hold nothing is evicted at all — dropping one would
        // lose the tabs of a run that may still come back to them, which is worse than a queue that
        // grows until one of those holds elapses.
        {
            let case = "eviction prefers an unheld record over a held one";
            let record = |held: bool, until: Instant| PendingRunRelease {
                namespace: "agent-tab-h-".to_string(),
                names: vec!["agent-tab-h-default".to_string()],
                attempts: 0,
                next_attempt_at: until,
                held,
                run: String::new(),
                left_open: 0,
            };
            let held = Instant::now() + Duration::from_hours(1);
            let mut mixed = VecDeque::from([record(true, held), record(false, Instant::now())]);
            let evicted = evict_oldest(&mut mixed).expect("the unheld record is evicted first");
            assert!(
                !evicted.held,
                "case {case}: the unheld record is evicted first"
            );
            let mut held_only = VecDeque::from([record(true, held), record(true, held)]);
            assert!(
                evict_oldest(&mut held_only).is_none(),
                "case {case}: with only held records, nothing is evicted"
            );
            let mut elapsed = VecDeque::from([record(true, Instant::now())]);
            assert!(
                evict_oldest(&mut elapsed).is_some(),
                "case {case}: a hold that has elapsed is evictable again"
            );
        }
        // A full queue of held records grows instead of losing one: the cap is what bounds a burst
        // of ends, and it never outweighs the tabs of a run that may still come back.
        {
            let case = "the queue cap never evicts a held record";
            clear_pending_releases();
            let held = |namespace: String| PendingRunRelease {
                namespace: namespace.clone(),
                names: vec![format!("{namespace}default")],
                attempts: 0,
                next_attempt_at: Instant::now() + Duration::from_hours(1),
                held: true,
                run: String::new(),
                left_open: 0,
            };
            for i in 0..MAX_PENDING_RELEASES {
                merge_record(held(format!("held-{i}")), Eligibility::RunEnd);
            }
            merge_record(held("held-overflow".to_string()), Eligibility::RunEnd);

            assert_eq!(
                pending_releases_snapshot().len(),
                MAX_PENDING_RELEASES + 1,
                "case {case}: the cap yields to a queue in which every record is held"
            );
            clear_pending_releases();
        }
    }

    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn an_attempt_that_could_not_reach_the_browser_keeps_the_record_queued() {
        // A close pass that outlives its attempt bound is one failed attempt: the name stays
        // queued, and the shorter bound is cut off without waiting the stub out.
        {
            let case = "a timed out release counts as one failed attempt";
            let guard = ReleaseGuard::install(Duration::from_secs(1)).await;
            queue_names("run-slow", &["agent-tab-s-slow"]);
            guard.set_mode("silent");

            let started = Instant::now();
            release_due().await;
            assert!(
                started.elapsed() < Duration::from_secs(3),
                "case {case}: the 1 s attempt bound must cut the silent child off"
            );
            assert_eq!(
                pending_names_and_attempts(),
                vec![(run_session_names("run-slow", &["agent-tab-s-slow"]), 1)],
                "case {case}: a timed-out attempt is one failed attempt, and the name stays queued"
            );

            // The short bound has served its purpose; the success that follows must not race a cold
            // or loaded stub spawn.
            guard.set_mode("ok");
            widen_attempt_bound(ATTEMPT_TIMEOUT);
            release_due().await;
            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: dropped on the next pass, when the browser answers"
            );
        }
        // With no chrome-use to run, a due release is skipped rather than failed — the record stays
        // queued and nothing is spawned — but it is still charged the ladder so the driver cannot
        // spin on it.
        {
            let case = "a release with no chrome use binary keeps the record queued";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            queue_names("run-absent", &["agent-tab-a-default"]);
            guard.set_binary_absent();

            release_due().await;

            assert_eq!(
                pending_names_and_attempts(),
                vec![(run_session_names("run-absent", &["agent-tab-a-default"]), 1)],
                "case {case}: nothing the browser could answer, so the name stays queued"
            );
            assert!(
                guard.log_lines().is_empty(),
                "case {case}: no child was spawned"
            );
        }
    }

    /// Two passes may overlap (the driver and the shutdown flush): neither may erase the other's
    /// records from the file — the reason the queue and the file are two different views of a
    /// pass's records — and unparking is per pass.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn overlapping_passes_keep_each_others_records_in_the_file() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-overlap-a", &["agent-tab-oa-default"]);
        let first = ReleasePass::take().expect("the first record is due");
        queue_names("run-overlap-b", &["agent-tab-ob-default"]);
        let second = ReleasePass::take().expect("the second record is due");

        // Both records' names, as the file holds them (one entry per namespace).
        let mut both = run_session_names("run-overlap-a", &["agent-tab-oa-default"]);
        both.extend(run_session_names(
            "run-overlap-b",
            &["agent-tab-ob-default"],
        ));
        both.sort();

        persist_pending_releases();
        assert_eq!(
            record_file_names(&guard.store_path()),
            both,
            "both passes' records are in the file"
        );

        // The first pass ends (as the shutdown flush does while the driver's pass is still in
        // flight): its record goes back to the queue, and the second pass's record is untouched by
        // its unpark.
        drop(first);
        assert_eq!(
            record_file_names(&guard.store_path()),
            both,
            "the pass that ended left the other pass's record in the file"
        );
        assert_eq!(
            pending_names_and_attempts(),
            vec![(
                run_session_names("run-overlap-a", &["agent-tab-oa-default"]),
                0
            )]
        );

        drop(second);
    }

    /// A queued record survives the process that queued it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_persisted_record_is_restored_and_released() {
        // A record queued by one process is retried by the next: the write at enqueue/after-a-pass
        // is all the durable state, and restoring it brings back the names AND the attempt count.
        {
            let case = "a queued release survives the process that queued it";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            queue_names("run-durable", &["agent-tab-d-default"]);
            guard.set_mode("refuse");
            release_due().await;
            assert_eq!(
                pending_names_and_attempts(),
                vec![(
                    run_session_names("run-durable", &["agent-tab-d-default"]),
                    1
                )],
                "case {case}: one unanswered pass, re-queued"
            );

            // Everything the process held in memory is gone; the file is what a restart has left.
            clear_pending_releases();
            assert!(pending_releases_snapshot().is_empty(), "case {case}");

            restore_pending_releases();
            assert_eq!(
                pending_names_and_attempts(),
                vec![(
                    run_session_names("run-durable", &["agent-tab-d-default"]),
                    1
                )],
                "case {case}: names and attempt count restored from the record file"
            );
        }
        // A record file that cannot be read — or that carries an absurd deadline — is never fatal:
        // an unreadable one restores nothing, and a parseable one whose deadline is absurd is
        // restored with it clamped rather than trusted.
        {
            let case = "a corrupt record file is ignored";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            fs::write(guard.store_path(), "not a release record{")
                .expect("write corrupt record file");

            restore_pending_releases();

            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: a corrupt file restores nothing and fails nothing"
            );

            // Parseable but absurd input: unlike the invalid file above it IS restored, with its
            // deadline clamped rather than trusted (the add would panic on it).
            fs::write(
                guard.store_path(),
                r#"[{"namespace":"agent-tab-absurd-","names":["agent-tab-absurd-default"],"next_attempt_at":18446744073709551615}]"#,
            )
            .expect("write absurd record file");

            restore_pending_releases();

            let delays = pending_release_delays();
            assert_eq!(
                delays.len(),
                1,
                "case {case}: the absurd record is restored"
            );
            assert!(
                delays[0] <= RELEASE_RETRY_CAP,
                "case {case}: its deadline is clamped to the ladder's cap: {delays:?}"
            );
            clear_pending_releases();

            // An entry the queue itself would refuse — no namespace, or no names — is not restored,
            // and must not count as restored either: a start with nothing actually queued asks the
            // browser nothing at all.
            fs::write(
                guard.store_path(),
                r#"[{"namespace":"","names":["agent-tab-ns-less-default"]},
                    {"namespace":"agent-tab-nameless-","names":[]}]"#,
            )
            .expect("write refused record file");

            restore_pending_releases();

            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: neither entry is a record the queue would hold"
            );
            assert!(
                reclaim_schedule().lock().unwrap_poison().next_at.is_none(),
                "case {case}: and none of them arms a read: a start with no work opens no page"
            );
        }
        // The record file is this queue's own state, not input to be judged: a record naming
        // sessions outside the family this queue opens is restored with those names untouched, and
        // the browser's answer settles them like any other.
        {
            let case = "a restored record releases its names exactly as written";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            clear_pending_releases();
            fs::write(
                guard.store_path(),
                r#"[{"namespace":"agent-tab-written-","names":["default","link-enricher-x","","agent-tab-other-default","agent-tab-written-kept"]},{"namespace":"agent-tab-nameless-","names":[]}]"#,
            )
            .expect("write record file naming foreign sessions");

            restore_pending_releases();

            let restored: Vec<String> = [
                "default",
                "link-enricher-x",
                "",
                "agent-tab-other-default",
                "agent-tab-written-kept",
            ]
            .iter()
            .map(|name| (*name).to_string())
            .collect();
            assert_eq!(
                pending_names_and_attempts(),
                vec![(restored, 0)],
                "case {case}: every name the entry carries is restored as written; the entry left with no names is not queued"
            );

            release_due().await;

            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: the browser holds no group by those names, so the record is settled"
            );
            clear_pending_releases();
        }
        // A restored record waits out the boot grace before anything may be attempted — the runs
        // the daemon re-drives at boot are built after this restore, so the live-namespace guard
        // cannot protect them yet.
        {
            let case = "a restored release waits out the boot grace";
            let grace = Duration::from_millis(800);
            let guard = ReleaseGuard::install_with(Duration::from_secs(10), grace).await;
            queue_names("run-grace", &["agent-tab-g-default"]);
            clear_pending_releases();

            restore_pending_releases();
            release_due().await;
            assert_eq!(
                pending_releases_snapshot().len(),
                1,
                "case {case}: restored at boot, the record is not attempted before the runs it may belong to exist"
            );
            assert!(
                guard.only_swept(),
                "case {case}: nothing was asked for the record: {:?}",
                guard.log_lines()
            );

            tokio::time::sleep(grace * 2).await;
            release_due().await;
            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: settled once the boot grace elapsed"
            );
            assert!(
                guard.invoked("extension call tabGroups.query"),
                "case {case}"
            );
        }
        // A held record is skipped untried until its hold elapses (the unheld end is covered
        // above), and the run-end merge rule lets a later, unheld end of the same namespace release
        // it without waiting that hold out.
        {
            let case = "a held redriven run release waits out its hold";
            let hold = Duration::from_millis(800);
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;

            queue_names_with_hold("run-held", &["agent-tab-h-default"], hold);
            release_due().await;
            assert_eq!(
                pending_releases_snapshot().len(),
                1,
                "case {case}: a held record is left alone, and the skip is not an attempt"
            );
            assert!(
                guard.only_swept(),
                "case {case}: no close while the run it belongs to may still come back: {:?}",
                guard.log_lines()
            );

            tokio::time::sleep(hold * 2).await;
            release_due().await;
            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: settled once the hold elapsed"
            );
            guard.clear_log();

            // Same namespace, queued held first, then re-queued unheld: the shorter eligibility
            // must win, so the re-queue is not blocked by the hold set.
            queue_names_with_hold("run-shrunk", &["agent-tab-s-default"], hold * 4);
            queue_names("run-shrunk", &["agent-tab-s-default"]);
            release_due().await;
            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: a re-queue without a hold makes the record eligible at once"
            );
        }
    }

    /// A hold is spent once the record is taken for an attempt: from then on the record is an
    /// ordinary one of the retry ladder, so the retry that re-arms it must not leave it permanently
    /// un-evictable or keep its namespace protected as if a run could come back.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_spent_hold_stops_protecting_its_record() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        // A retry gap longer than this test, so only a hold left standing could keep the re-armed
        // record inside one and the assertions below meaningful.
        widen_retry_base(Duration::from_secs(600));
        let names = run_session_names("run-spent-hold", &["agent-tab-sh-default"]);
        guard.write_state(&names, true);
        // The hold has already elapsed by the time the pass takes the record.
        queue_names_with_hold(
            "run-spent-hold",
            &["agent-tab-sh-default"],
            Duration::from_millis(1),
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
        // The browser answers and leaves its own tabs open, so the ladder re-arms the record.
        guard.set_mode("keep");

        release_due().await;

        assert_eq!(
            pending_names_and_attempts(),
            vec![(names.clone(), 1)],
            "the browser answered and left the tabs, so the record is retried"
        );
        assert!(
            !ProtectedNamespaces::live_and_held().contains(&names[0]),
            "a spent hold no longer protects its namespace"
        );
        assert!(
            queue_has_evictable_record(),
            "and the record is an ordinary, evictable one again"
        );
    }

    /// Whether any queued record is outside a hold — the cap's own view of what it may evict.
    fn queue_has_evictable_record() -> bool {
        let mut queue = pending_releases().lock().unwrap_poison().clone();
        evict_oldest(&mut queue).is_some()
    }

    /// A record whose run is live again is skipped untried — that run owns the sessions — and
    /// becomes releasable the moment it is gone; a run resumed while a pass is in flight has
    /// already re-attached the very same names, so the close set is re-checked per name right
    /// before the browser is asked.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_live_run_keeps_its_own_sessions() {
        {
            let case = "a record whose run is live again is never attempted";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            let live = ChromeRunSessions::for_run("run-live");
            live.track(&format!("{}default", live.namespace()));
            queue_run_session_release(&live, false, "run-live");

            release_due().await;

            assert_eq!(
                pending_releases_snapshot().len(),
                1,
                "case {case}: a live run's record stays queued and is not an attempt"
            );
            assert!(
                guard.only_swept(),
                "case {case}: no close pass is ever run for a live run's sessions: {:?}",
                guard.log_lines()
            );

            drop(live);
            release_due().await;
            assert!(
                pending_releases_snapshot().is_empty(),
                "case {case}: settled once the run that owns them is gone"
            );
            assert!(
                guard.invoked("extension call tabGroups.query"),
                "case {case}"
            );
        }
        // A run resumed while a pass is in flight has already re-attached the very same names, so
        // the close set is re-checked per name right before the browser is asked: a name a live run
        // owns again is left out of the close and its record keeps it rather than closing the
        // session out from under the run — unless the browser holds no group of that title any
        // more, which is settled gone whoever owns it.
        {
            let case = "a run that comes back mid pass keeps its sessions";
            let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
            let names = run_session_names("run-midpass", &["agent-tab-m-default"]);
            guard.write_state(&names, true);
            // The record a pass would have taken just before the run came back.
            queue_names("run-midpass", &["agent-tab-m-default"]);
            let pass = ReleasePass::take().expect("the record is due before the run comes back");
            // The same durable id is re-dispatched inside the pass window and re-attaches the very
            // same names.
            let live = ChromeRunSessions::for_run("run-midpass");

            let cli = release_cli();
            let mut protected = ProtectedNamespaces::snapshot(&durable_resume_namespaces().await);
            let issues = attempt_pass(
                cli.as_deref(),
                Some(pass),
                OWN_SESSION,
                &ReclaimAsk::unanswered(),
                &mut protected,
                Instant::now() + Duration::from_secs(10),
            )
            .await;

            assert_eq!(
                pending_names_and_attempts(),
                vec![(names, 1)],
                "case {case}: a run that came back mid-pass keeps its sessions, retried on the next pass"
            );
            assert!(
                guard.log_lines().is_empty(),
                "case {case}: no child was spawned for a run that is live again"
            );
            assert!(issues.is_empty(), "case {case}");
            drop(live);
        }
    }

    /// The production entry point ([`run_session_release_queue`], driven here through its loop with
    /// a local token): a queued record is closed, and a record whose run is live again waits — not
    /// on a poll interval, on the wake that run's own tracker sends when it goes away.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_driver_releases_a_queued_record_and_waits_for_a_live_run_to_go() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        // A retry gap far longer than this test: a release that still lands promptly can only be
        // explained by the wake.
        widen_retry_base(Duration::from_secs(600));
        let live = ChromeRunSessions::for_run("run-driver");
        let names = run_session_names("run-driver", &["agent-tab-driver-default"]);
        live.track(&names[0]);
        queue_run_session_release(&live, false, "run-driver");

        let shutdown = CancellationToken::new();
        let driver = tokio::spawn(release_queue(shutdown.clone()));

        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            guard.only_swept(),
            "the driver leaves a live run's session alone: {:?}",
            guard.log_lines()
        );
        assert!(
            guard.invoked(&format!("--session {}", names[0])),
            "and runs its own read in the run's session, not the product's own: {:?}",
            guard.log_lines()
        );
        assert!(
            !guard.invoked(&format!("--session {OWN_SESSION}")),
            "so a run end opens nothing of ours at all: {:?}",
            guard.log_lines()
        );
        assert_eq!(pending_releases_snapshot().len(), 1);

        drop(live);
        // The record settles on the tracker's own wake, not on a poll interval.
        tokio::time::timeout(Duration::from_secs(5), async {
            while !record_file_names(&guard.store_path()).is_empty() {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the driver settles the record once the run that owned it is gone");
        assert!(
            pending_releases_snapshot().is_empty(),
            "and the queue holds nothing either"
        );
        assert!(
            guard.invoked("extension call tabGroups.query"),
            "the driver asked the browser for the record's groups"
        );

        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), driver)
            .await
            .expect("the driver stops on the token")
            .expect("the driver task does not panic");
    }

    /// One record per run: a second end of the same (agent-id-derived) namespace adds only the
    /// names it recorded and consumes no attempt.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_requeue_of_the_same_run_does_not_double_count() {
        // The merge path persists too, so the record file needs the guard's private dir (and the
        // queue is drained for the next test).
        let _guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        clear_pending_releases();
        let first = ChromeRunSessions::for_run("run-merge");
        let names = run_session_names("run-merge", &["agent-tab-m-default", "agent-tab-m-docs"]);
        first.track(&names[0]);
        queue_run_session_release(&first, false, "run-merge");
        let second = ChromeRunSessions::for_run("run-merge");
        second.track(&names[0]);
        second.track(&names[1]);
        queue_run_session_release(&second, false, "run-merge");
        drop((first, second));

        assert_eq!(
            pending_names_and_attempts(),
            vec![(names, 0)],
            "one record carrying both ends' names"
        );
        clear_pending_releases();
    }

    /// The shutdown flush spends its budget on the records whose hold (or retry backoff) elapsed
    /// and leaves the held ones alone: a held record waits for a run the restart may re-drive, and
    /// being durable it is picked up next boot.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_shutdown_flush_releases_the_queue_within_its_budget() {
        // The held record's hold outlasts the whole test: no sleeping.
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-flush", &["agent-tab-f-default"]);
        queue_names_with_hold(
            "run-flush-held",
            &["agent-tab-f-held"],
            Duration::from_hours(1),
        );

        // A silent browser side cannot hold the shutdown flush past its budget.
        guard.set_mode("silent");
        let started = Instant::now();
        flush_pending_run_releases(Duration::from_millis(500)).await;
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "the flush stops at its budget"
        );
        assert_eq!(
            pending_releases_snapshot().len(),
            2,
            "an unsettled name stays queued"
        );

        guard.set_mode("ok");
        guard.clear_log();
        flush_pending_run_releases(Duration::from_secs(10)).await;
        assert_eq!(
            pending_names_and_attempts(),
            vec![(
                run_session_names("run-flush-held", &["agent-tab-f-held"]),
                0
            )],
            "the flush drains what it can settle and leaves the held record for its resumed run"
        );
        assert!(
            guard.invoked("extension call tabGroups.query"),
            "one browser-driven close pass, and never one for the held record"
        );
    }

    /// A run end that arrives while a pass holds the record must not be lost by the file: queue and
    /// parked copies are folded, so the later (held) pair survives a process death in that window
    /// instead of the stale unheld snapshot the pass took.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_hold_queued_while_a_pass_holds_the_record_survives_in_the_file() {
        let guard = ReleaseGuard::install_with(Duration::from_secs(10), Duration::ZERO).await;
        queue_names("run-fold", &["agent-tab-f-default"]);
        let pass = ReleasePass::take().expect("a due record is taken");

        // The same run is cut off while the pass holds it: the queue now says the release must be
        // held, the pass's parked snapshot still says otherwise.
        queue_names_with_hold(
            "run-fold",
            &["agent-tab-f-default"],
            Duration::from_mins(30),
        );
        assert_eq!(
            record_file_names(&guard.store_path()),
            run_session_names("run-fold", &["agent-tab-f-default"]),
            "one entry per namespace, so a restore cannot pick the stale pair"
        );

        // What the next boot reads back.
        clear_pending_releases();
        restore_pending_releases();
        let delays = pending_release_delays();
        assert_eq!(delays.len(), 1, "the record is restored once");
        assert!(
            delays[0] >= RELEASE_HOLD_REDRIVEN_RUN.saturating_sub(Duration::from_secs(5)),
            "the file kept the hold rather than the stale unheld snapshot: {:?}",
            delays[0]
        );
        drop(pass);
    }

    /// A `finish` that panics hands nothing back, yet the record it was holding is still parked and
    /// persisted, so the next boot restores it instead of leaving those sessions stranded
    /// everywhere.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_panicking_finish_leaves_its_record_for_the_next_boot() {
        let _guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-panic", &["agent-tab-p-default"]);

        let mut pass = ReleasePass::take().expect("a due record is taken");
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pass.reconcile(&HashMap::new(), |_, _| panic!("finish panicked"));
        }))
        .is_err();
        drop(pass);
        assert!(panicked, "the panic reached the pass");

        assert!(
            pending_releases_snapshot().is_empty(),
            "the record is out of the live queue"
        );
        restore_pending_releases();
        assert_eq!(
            pending_names_and_attempts(),
            vec![(run_session_names("run-panic", &["agent-tab-p-default"]), 0)],
            "but the file kept it, so the next boot restores it"
        );
    }

    /// A pass dropped mid-attempt (shutdown token, panic) puts back what it took and counts no
    /// attempt: a record left out of the queue would also be left out of the next persist,
    /// stranding its sessions.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn an_aborted_pass_puts_its_records_back() {
        let _guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-aborted", &["agent-tab-a-default"]);

        let pass = ReleasePass::take().expect("a due record is taken");
        assert!(
            pending_releases_snapshot().is_empty(),
            "the pass holds the record"
        );
        drop(pass);

        assert_eq!(
            pending_names_and_attempts(),
            vec![(
                run_session_names("run-aborted", &["agent-tab-a-default"]),
                0
            )],
            "a dropped pass puts the record back untried"
        );
    }

    /// A held record re-mints its hold from the boot that restores it — the run it waits for is
    /// re-dispatched after the restart, so the deadline the previous process minted says nothing.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_restored_held_record_holds_from_boot() {
        let guard = ReleaseGuard::install_with(Duration::from_secs(10), Duration::ZERO).await;

        // The file a previous process left: its own namespace, and the physical names under it.
        let namespace = ChromeRunSessions::for_run("run-resume")
            .namespace()
            .to_string();
        write_record_file(
            &guard.store_path(),
            &[PersistedRunRelease {
                namespace,
                names: run_session_names("run-resume", &["agent-tab-r-default"]),
                attempts: 0,
                // Long past: the hold the previous process minted has expired.
                next_attempt_at: 1,
                held: true,
                run: String::new(),
                left_open: 0,
            }],
        );
        restore_pending_releases();

        let delays = pending_release_delays();
        assert_eq!(delays.len(), 1, "the held record is restored");
        assert!(
            delays[0] >= RELEASE_HOLD_REDRIVEN_RUN.saturating_sub(Duration::from_secs(5)),
            "the hold is measured from this boot, not from the expired deadline: {:?}",
            delays[0]
        );
    }

    /// The cap's eviction closes exactly the named tab of the live run it is asked for, leaves
    /// another live run's tab standing, and retires the settled session's chrome-use record.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn an_over_cap_eviction_closes_the_named_tab_and_leaves_others_alone() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        // Both runs are live: the victim's own namespace is exempt from the live-run guard (it is
        // the run asking for its own tab), while every other live namespace stays protected.
        let ours = ChromeRunSessions::for_run("evict-ours");
        let ours_session = format!("{}victim", ours.namespace());
        let other = ChromeRunSessions::for_run("evict-other");
        let other_session = format!("{}bystander", other.namespace());
        guard.write_state(&[ours_session.clone(), other_session.clone()], true);

        let gone = evict_agent_tab(ours.namespace(), &ours_session, &other_session).await;

        assert!(gone, "the browser confirmed the victim's group gone");
        assert!(
            !guard.group_titles().contains(&ours_session),
            "the victim's group is gone: {:?}",
            guard.group_titles()
        );
        assert!(
            guard.group_titles().contains(&other_session),
            "another live run's group — the close's own read session — is untouched: {:?}",
            guard.group_titles()
        );
        assert!(
            guard.invoked("session stop --force"),
            "the settled session's chrome-use record was retired: {:?}",
            guard.log_lines()
        );
    }

    /// An eviction the browser answers for but does not perform is not confirmed: the group is
    /// still standing and the caller is told so.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn an_eviction_the_browser_leaves_open_is_not_confirmed() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        guard.set_mode("keep");
        let ours = ChromeRunSessions::for_run("evict-kept");
        let ours_session = format!("{}victim", ours.namespace());
        guard.write_state(std::slice::from_ref(&ours_session), true);

        let gone = evict_agent_tab(ours.namespace(), &ours_session, &ours_session).await;

        assert!(
            !gone,
            "the browser answered and left the group standing: {:?}",
            guard.log_lines()
        );
        assert!(
            guard.group_titles().contains(&ours_session),
            "the group is still in the browser"
        );
    }

    /// A confirmed eviction whose `session stop` does not settle is asked again, because the name
    /// chrome-use still pins is one an `open` under it would fail on — and the retry runs off the
    /// agent's path, so it costs the agent nothing.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn an_eviction_whose_session_stop_fails_is_retried() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        guard.set_mode("stop-refused");
        let ours = ChromeRunSessions::for_run("evict-stop-refused");
        let ours_session = format!("{}victim", ours.namespace());
        guard.write_state(std::slice::from_ref(&ours_session), true);

        let gone = evict_agent_tab(ours.namespace(), &ours_session, &ours_session).await;

        assert!(gone, "the group itself was closed: {:?}", guard.log_lines());
        let stops = guard
            .log_lines()
            .iter()
            .filter(|line| line.starts_with("session stop"))
            .count();
        assert_eq!(stops, 2, "the stop that did not settle was asked again");
    }

    /// The tool's cap wiring end to end, minus only the browser dispatch the tool itself would
    /// make: the sixth appearance names the least recently addressed tab on the record, the close
    /// the tool spawns off its own path really takes that group out of the browser (the stub stands
    /// in for chrome-use), and the confirmed close comes back to the record as a closed tab.
    #[tokio::test]
    #[serial_test::serial(chrome_release, chrome_tabs)]
    async fn the_tools_sixth_appearance_closes_the_least_recently_addressed_tab() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        let _ledger = chrome_tab_ledger::TestLedgerGuard::install();
        let sessions = ChromeRunSessions::for_run("cap-tool-wiring");
        let tool = crate::tools::chrome::ChromeTool::new(Arc::clone(&sessions));
        let namespace = sessions.namespace().to_string();
        let victim_session = format!("{namespace}a");
        guard.write_state(std::slice::from_ref(&victim_session), true);

        for tab in ["a", "b", "c", "d", "e"] {
            tool.record_addressed(tab, None);
        }
        assert!(
            !chrome_tab_ledger::is_closed(&namespace, "a"),
            "the fifth tab is inside the cap"
        );
        tool.record_addressed("f", None);

        // The close runs off the agent's path, so wait for it to land.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !chrome_tab_ledger::is_closed(&namespace, "a") && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            chrome_tab_ledger::is_closed(&namespace, "a"),
            "the run's least recently addressed tab is the one the cap closed"
        );
        assert!(
            !guard.group_titles().contains(&victim_session),
            "the confirmed close took the tab out of the browser: {:?}",
            guard.group_titles()
        );
    }

    /// The boot prune drops a namespace no live run, no held release record and no resumable job
    /// claims, and keeps one a live run holds.
    #[tokio::test]
    #[serial_test::serial(chrome_release, chrome_tabs)]
    async fn a_namespace_no_run_claims_is_dropped_at_boot() {
        crate::util::test::init_test_stores().await;
        let ledger = chrome_tab_ledger::TestLedgerGuard::install();
        // A real run namespace, registered only for the call that minted the name (dropping the
        // tracker deregisters it): none of the claims the prune protects applies to it.
        let namespace = ChromeRunSessions::for_run("prune-orphan")
            .namespace()
            .to_string();
        assert!(
            chrome_tab_ledger::note_addressed(&namespace, "tab", true).is_none(),
            "one tab is inside the cap"
        );

        prune_tab_records().await;
        assert!(
            !ledger.holds(&namespace),
            "no run claims the namespace: its record is dropped"
        );

        // With the run live again the record survives the same prune.
        let live = ChromeRunSessions::for_run("prune-orphan");
        assert!(
            chrome_tab_ledger::note_addressed(&namespace, "tab", true).is_none(),
            "one tab is inside the cap"
        );
        prune_tab_records().await;
        assert!(
            ledger.holds(&namespace),
            "a live run's record survives the boot prune"
        );
        drop(live);
    }
}
