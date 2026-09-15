//! Ended-run chrome session releases.
//!
//! chrome-use ≥1.5.101 preserves external Chrome tabs across daemon idle
//! recycling, so the sessions an agent run opened must be released explicitly —
//! and a run cannot do it in its own tail, which is skipped whenever its future
//! is dropped. Every run end therefore queues the names
//! it recorded for itself, from its run-end guard's `Drop`
//! (`crate::agent::RunEndCleanup`), and the `chrome-run-releases` task releases
//! them out of band: one verified `session stop` per name, from the run's own
//! recorded set — what that proves and what it costs is stated in the
//! live-verified behaviours block of [`crate::tools::chrome_daemon`]. The record
//! file `chrome-run-releases.json` is what makes a queued release outlive the
//! process, and `MAX_PENDING_RELEASES` bounds the queue.
//!
//! The rules that decide a release — and the limits of each — are stated where
//! they live: which run ends keep their tabs is `holds_run_end`, the bounded
//! give-ups are `RELEASE_MAX_ATTEMPTS` (attempts spent), `RELEASE_MAX_SKIPS`
//! (chrome-use unrunnable) and `MAX_PENDING_RELEASES` (queue overflow), what an
//! attempt trusts is `release_one`, the one delay a release cannot bound is
//! `take_releasable`, and the two cases that leave nothing to release are
//! `crate::agent::RunEndCleanup` and `crate::tools::chrome::run_session_namespace`.
//!
//! Out of scope: the sessions an agent mints by shelling out to `mahbot chrome`
//! (`mahbot-chrome-ephemeral-*`, closed best-effort by that call and the boot sweep).

use super::chrome_daemon::{cli_path, run_cli_json_at};
use crate::util::UnwrapPoison;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

/// Bound for one release attempt: one `session stop`, whose whole legitimate cost is
/// the ≈28 s the live-verified behaviours block of [`crate::tools::chrome_daemon`]
/// states — about twice that, so the estimate drifting with a chrome-use release
/// cannot charge a working stop as a failure, at the price of a degraded path that
/// takes twice as long to give up (the ladder is [`RELEASE_MAX_ATTEMPTS`]).
const RELEASE_ATTEMPT_TIMEOUT: Duration = Duration::from_mins(1);
/// Concurrent `chrome-use` children per pass.
const RELEASE_CONCURRENCY: usize = 8;
/// Attempts one ended run's record gets before it is given up on and dropped with one
/// INFO line — the price of a bounded record, and the terminal policy for a browser side
/// that stays degraded. Attempts are spaced by [`release_backoff`] (≈75 s in gaps), and
/// each one costs up to [`RELEASE_ATTEMPT_TIMEOUT`] per [`RELEASE_CONCURRENCY`] of the
/// record's names: minutes for a small record, tens of minutes for one with dozens.
///
/// The give-up is silent: the record is dropped and those tabs stay open in the
/// browser, with the INFO line below the only trace of it. Nothing user-facing reports
/// it — no notification, message or board state — which is deliberate: this mechanism
/// adds no user-visible surface of its own.
const RELEASE_MAX_ATTEMPTS: u32 = 5;
/// Skips one record gets while chrome-use cannot be run at all (absent, mid-swap) before
/// it is dropped. An unrunnable binary is not the browser side refusing, so it does not
/// spend [`RELEASE_MAX_ATTEMPTS`] — but it is still charged here, so a record nothing can
/// act on cannot sit in the queue forever, on the same ladder (≈4 min of gaps only):
/// its give-up is the one [`RELEASE_MAX_ATTEMPTS`] describes.
const RELEASE_MAX_SKIPS: u32 = 8;
/// First retry (or skip) gap and its cap: the gap doubles per step.
const RELEASE_RETRY_BASE: Duration = Duration::from_secs(5);
const RELEASE_RETRY_CAP: Duration = Duration::from_mins(1);
/// How long a held release waits, and the only bound on a run that never comes back.
/// A restart inside the window re-mints the whole hold from that boot (see
/// [`restored_eligibility`]); which ends hold is [`holds_run_end`]'s rule.
pub(crate) const RELEASE_HOLD_REDRIVEN_RUN: Duration = Duration::from_mins(30);
/// Which run ends hold their session release back: a drain/shutdown cut and a
/// workspace-pause freeze, held for [`RELEASE_HOLD_REDRIVEN_RUN`]. Those are the ends
/// after which the same run is normally rebuilt under its own durable id (the
/// dispatches re-read a durable id, and a paused phase job waits for the unpause
/// re-drive), so its resumed segment must still find the tabs it was working with.
/// The hold is a property of the end alone — nothing checks that a run is actually
/// re-driven — so a run nothing comes back for (a research round member mints a fresh
/// id per dispatch) is held too, and pays the hold as latency.
///
/// `internal_cancel` is a cooperative end too, but deliberately not held: it may
/// release while a replacement segment starts up on the same names, which is why every
/// attempt re-checks per name.
///
/// Every other end releases at once, including the ones that never reach a trailing
/// statement: a run aborted at a round/research deadline or lost to a panic ends
/// through its run-end guard's `Drop` (`crate::agent::RunEndCleanup`), which queues
/// the record unheld — no hold is applied to it.
pub(crate) fn holds_run_end(classification: &str) -> bool {
    matches!(classification, "drain" | "shutdown" | "pause")
}

/// Floor on a restored record's eligibility: the runs the daemon re-drives at boot
/// are constructed (and their namespaces registered as live) after the release
/// task restores, so a restored record must not be attempted before they exist.
/// A held record is not floored here — its hold is re-minted from boot instead
/// (see [`restored_eligibility`]).
const RELEASE_BOOT_GRACE: Duration = Duration::from_mins(1);
/// Durable record of the queue, at the storage root. A file rather than a DB row
/// because the hand-off runs from a `Drop` (a dropped future is an interrupted
/// run's only hook), where a DB write during a shutdown is not dependable — this
/// is one small synchronous write.
const RELEASE_FILE_NAME: &str = "chrome-run-releases.json";
/// Hard cap on queued releases: a burst of runs ending while the browser side is
/// down must not accumulate records without bound. The oldest record is dropped to
/// make room, giving up its tabs the way [`RELEASE_MAX_ATTEMPTS`] describes.
const MAX_PENDING_RELEASES: usize = 64;
/// Last-chance release budget on the shutdown/self-update path, in the spirit of the
/// chrome-side `SHUTDOWN_CLEANUP_TIMEOUT`: a degraded browser side cannot be waited out
/// while the process is going down. It sits below what one `session stop` may
/// legitimately need (see [`RELEASE_ATTEMPT_TIMEOUT`]), so a slow but working stop is
/// cut off here and its record left for the next boot — these attempts are uncharged and
/// the record is durable, so the shortfall costs a retry, not the release.
const SHUTDOWN_RELEASE_FLUSH_BUDGET: Duration = Duration::from_secs(10);

/// One ended run's unreleased session names.
#[derive(Clone)]
struct PendingRunRelease {
    /// Namespace of the run that opened them — the identity a live run of the
    /// same agent id shares, which records are merged by.
    namespace: String,
    names: Vec<String>,
    attempts: u32,
    /// Times the record was due while chrome-use could not be run at all (see
    /// [`RELEASE_MAX_SKIPS`]) — the ladder its retry gap grows on.
    skips: u32,
    next_attempt_at: Instant,
    /// Whether [`holds_run_end`] held this record's eligibility back — see
    /// [`restored_eligibility`] for what that does to it.
    held: bool,
}

/// One queued record as persisted. `Instant` is process-relative, so the
/// remaining eligibility is stored as an absolute epoch-seconds deadline and
/// re-based onto the reading process's clock at restore.
#[derive(Serialize, Deserialize)]
struct PersistedRunRelease {
    namespace: String,
    names: Vec<String>,
    #[serde(default)]
    attempts: u32,
    #[serde(default)]
    skips: u32,
    /// Epoch seconds at which the record becomes eligible (`0` = now).
    #[serde(default)]
    next_attempt_at: u64,
    /// Whether the record's eligibility is a hold ([`PendingRunRelease::held`]).
    #[serde(default)]
    held: bool,
}

static PENDING_RELEASES: OnceLock<Mutex<VecDeque<PendingRunRelease>>> = OnceLock::new();
/// The records the passes in flight have taken out of the queue, each tagged with the
/// [`ReleasePass`] holding it. [`persist_pending_releases`] writes them too, so a persist
/// from any path can never publish a file that loses a batch a pass is mid-way through
/// when the process exits (self-update's `exit(0)`, a kill); each pass unparks exactly
/// its own, because passes may overlap (the driver and the shutdown flush). The one way
/// an entry outlives its pass is a panic inside a finish closure, and it is parked
/// precisely so the file keeps it until the next boot restores it.
static PARKED_RELEASES: OnceLock<Mutex<Vec<(u64, PendingRunRelease)>>> = OnceLock::new();
/// Source of [`ReleasePass`] ids, unique within one process.
static NEXT_PASS_ID: AtomicU64 = AtomicU64::new(0);
static RELEASE_WAKE: OnceLock<tokio::sync::Notify> = OnceLock::new();
/// Namespaces of the runs live RIGHT NOW, refcounted: a replacement run for the
/// same agent id (self-update or daemon-restart resume) can overlap the run it
/// replaces, so both may hold the same namespace at once.
static LIVE_RUN_NAMESPACES: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();

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

/// Mark a run's session namespace as live. The release of an earlier run's record
/// must never close a session whose run is live again: a resumed run re-attaches
/// the very same names, so it owns them until its own release. Empty namespaces
/// (the shared non-agent instance) are ignored — it never owns run-scoped sessions.
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

/// Drop one registration of a run's namespace. Waking the release task here is
/// what makes a record skipped as live releasable the moment that run's tracker is
/// gone — a live skip is not re-armed, so there is no poll interval.
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

/// Whether some live run owns `namespace` right now.
fn run_namespace_is_live(namespace: &str) -> bool {
    !namespace.is_empty()
        && live_run_namespaces()
            .lock()
            .unwrap_poison()
            .contains_key(namespace)
}

/// Which of two eligibilities a merge keeps when the run already has a record
/// queued. An eligibility is the pair (when the record may be attempted, whether it
/// is a hold) and the two travel together, so the pair kept is always whole.
#[derive(Clone, Copy)]
enum Eligibility {
    /// The incoming pair wins: a run end is the most recent statement about the run,
    /// so it replaces an earlier cut's hold with what it says now.
    RunEnd,
    /// The later deadline wins: a pass putting records back must not undercut an
    /// eligibility that arrived while it was in flight. A stale hold can win this way
    /// and delay a release its run no longer needs, by at most the hold.
    Retry,
}

impl Eligibility {
    /// The pair a merge keeps.
    fn keep(self, queued: (Instant, bool), incoming: (Instant, bool)) -> (Instant, bool) {
        match self {
            Self::RunEnd => incoming,
            // A retry keeps the queued pair unless the incoming one is later
            // still — a tie must not shorten what is already queued.
            Self::Retry if incoming.0 > queued.0 => incoming,
            Self::Retry => queued,
        }
    }
}

/// Fold `incoming` into `entry`: the union of the names, the higher counts and the
/// eligibility pair `keep` picks among the two.
fn absorb(entry: &mut PendingRunRelease, incoming: PendingRunRelease, keep: Eligibility) {
    (entry.next_attempt_at, entry.held) = keep.keep(
        (entry.next_attempt_at, entry.held),
        (incoming.next_attempt_at, incoming.held),
    );
    entry.attempts = entry.attempts.max(incoming.attempts);
    entry.skips = entry.skips.max(incoming.skips);
    for name in incoming.names {
        if !entry.names.contains(&name) {
            entry.names.push(name);
        }
    }
}

/// Put a record in the queue, merging into the record already queued for the same run
/// instead of keeping two: the merge keeps the union of the names and the higher counts,
/// so a re-queue cannot discard what an earlier end recorded. `eligibility` resolves
/// which pair wins (see [`Eligibility`]); the live-namespace guard, not the deadline, is
/// what keeps a run that is back from losing its tabs. Enforces the queue cap (oldest
/// dropped).
fn merge_record(entry: PendingRunRelease, eligibility: Eligibility) {
    // Only a run's own namespace can ever be addressed again, so a record without one is
    // never queued: releasing it could close a session this queue does not own.
    if entry.namespace.is_empty() || entry.names.is_empty() {
        return;
    }
    let mut queue = pending_releases().lock().unwrap_poison();
    if let Some(existing) = queue.iter_mut().find(|e| e.namespace == entry.namespace) {
        absorb(existing, entry, eligibility);
        return;
    }
    let mut dropped = 0usize;
    while queue.len() >= MAX_PENDING_RELEASES {
        queue.pop_front();
        dropped += 1;
    }
    if dropped > 0 {
        info!(
            dropped,
            "agent-run chrome release queue full — oldest records dropped"
        );
    }
    queue.push_back(entry);
}

/// Hand a run's ended-session record over for release. `held` is [`holds_run_end`]'s
/// verdict for how the run ended, and keeps the release back for
/// [`RELEASE_HOLD_REDRIVEN_RUN`]. Sync and panic-free so it can run from a `Drop` (an
/// aborted or panicking run's only hook), and durable: the write below is what makes the
/// release survive this process. The names and the namespace are snapshotted here —
/// `ChromeTool` records a session before it dispatches, so the set is complete once the
/// run is gone — and a later run of the same durable id re-attaches them, which the
/// live-namespace guard covers (see [`take_releasable`]).
pub(crate) fn queue_run_session_release(sessions: &super::chrome::ChromeRunSessions, held: bool) {
    queue_run_session_release_after(
        sessions,
        if held {
            RELEASE_HOLD_REDRIVEN_RUN
        } else {
            Duration::ZERO
        },
    );
}

/// [`queue_run_session_release`] with the hold spelled out, for the tests that need one
/// other than the constant's.
fn queue_run_session_release_after(sessions: &super::chrome::ChromeRunSessions, hold: Duration) {
    let names = sessions.snapshot();
    if names.is_empty() {
        return; // most runs never touch chrome — nothing to queue
    }
    merge_record(
        PendingRunRelease {
            namespace: sessions.namespace().to_string(),
            names,
            attempts: 0,
            skips: 0,
            next_attempt_at: Instant::now() + hold,
            held: !hold.is_zero(),
        },
        Eligibility::RunEnd,
    );
    persist_pending_releases();
    release_wake().notify_one();
}

/// Restore the records a previous process left queued, once at the start of
/// [`run_session_release_queue`]: a release still pending when the release task
/// stopped is retried after the restart instead of being lost, and a held record
/// survives the restart it is waiting for — with its attempt count, so the bound stays
/// bounded across restarts. An unreadable or corrupt file is ignored rather than
/// treated as a failure, and a name it carries is released only if it is an agent-run
/// session under that record's own namespace: losing or rejecting a record can leave
/// tabs open, never close a session this queue does not own.
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
    for mut record in records.into_iter().take(MAX_PENDING_RELEASES) {
        // The file is input from outside this process: it is not trusted to name the
        // sessions this queue may stop.
        record.names.retain(|name| {
            crate::tools::chrome::is_agent_tab_session(name) && name.starts_with(&record.namespace)
        });
        if record.names.is_empty() {
            continue;
        }
        merge_record(
            PendingRunRelease {
                namespace: record.namespace,
                names: record.names,
                attempts: record.attempts,
                skips: record.skips,
                next_attempt_at: restored_eligibility(record.next_attempt_at, record.held),
                held: record.held,
            },
            // `Retry`, not `RunEnd`: a file with two entries for one namespace (only a
            // pre-dedupe file can) keeps the later pair, i.e. the hold, rather than
            // blindly letting the last line win.
            Eligibility::Retry,
        );
        restored += 1;
    }
    if restored > 0 {
        debug!(
            restored,
            "agent-run chrome releases restored from the previous process"
        );
    }
}

/// Write the whole queue — plus what the passes in flight hold ([`PARKED_RELEASES`]) —
/// to the record file (compact JSON, atomic tmp+rename, like the chat-draft file):
/// fail-open, a no-op with no storage root, and a record with no names is never written.
/// Both locks are held across the write as well as the snapshot — parked before queue,
/// the order [`ReleasePass::take`] parks in — because two persists sharing the one
/// `.json.tmp` could publish a torn file the next boot ignores, stranding a batch. Queue
/// and parked can hold the same record while a pass reconciles, so the file carries one
/// entry per namespace folded with [`Eligibility::Retry`], keeping the later (held) pair
/// rather than letting a restore's last-wins resurrect a stale one.
fn persist_pending_releases() {
    let Some(path) = release_settings().store else {
        return;
    };
    let parked = parked_releases().lock().unwrap_poison();
    let queue = pending_releases().lock().unwrap_poison();
    let mut folded: Vec<PendingRunRelease> = Vec::new();
    for record in queue.iter().chain(parked.iter().map(|(_, record)| record)) {
        match folded
            .iter_mut()
            .find(|entry| entry.namespace == record.namespace)
        {
            Some(entry) => absorb(entry, record.clone(), Eligibility::Retry),
            None => folded.push(record.clone()),
        }
    }
    let records: Vec<PersistedRunRelease> = folded
        .iter()
        .filter(|entry| !entry.names.is_empty())
        .map(|entry| PersistedRunRelease {
            namespace: entry.namespace.clone(),
            names: entry.names.clone(),
            attempts: entry.attempts,
            skips: entry.skips,
            next_attempt_at: eligibility_deadline(entry.next_attempt_at),
            held: entry.held,
        })
        .collect();
    let json = serde_json::to_string(&records).unwrap_or_default();
    let tmp = path.with_extension("json.tmp");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&tmp, json);
    let _ = std::fs::rename(&tmp, &path);
}

/// Whole epoch seconds — the unit the record file stores deadlines in.
fn epoch_secs() -> u64 {
    crate::util::unix_millis() / 1000
}

/// Absolute epoch-seconds deadline for a record's remaining eligibility, which
/// is what the file stores: an `Instant` is process-relative and dies with the
/// process that minted it.
fn eligibility_deadline(next_attempt_at: Instant) -> u64 {
    epoch_secs().saturating_add(
        next_attempt_at
            .saturating_duration_since(Instant::now())
            .as_secs(),
    )
}

/// A persisted record as this process's eligibility instant. A held record
/// re-mints its whole hold from THIS boot ([`RELEASE_HOLD_REDRIVEN_RUN`]): the run
/// it waits for is re-dispatched only after the restart, so the deadline the
/// previous process minted says nothing about when that run will exist — trusting
/// it would downgrade a restart that outlasted the hold to the boot grace and close
/// the tabs of a run about to resume. Re-minting also means every boot inside the
/// window defers the release again, and the deferral itself is unbounded for as long
/// as restarts keep landing inside that window: no attempt is charged while a hold
/// runs ([`MAX_PENDING_RELEASES`] bounds how many records can exist, not how long one
/// waits). Every other record is floored at the boot grace — with the shipped
/// settings the stored deadline never wins, both being the same minute — and the
/// deadline is clamped to the ladder's cap, since a deadline further out than the
/// backoff can mint is corrupt input, and trusting it would panic the add and take
/// the release task down for the whole boot.
fn restored_eligibility(deadline: u64, held: bool) -> Instant {
    if held {
        return Instant::now() + RELEASE_HOLD_REDRIVEN_RUN;
    }
    let remaining =
        Duration::from_secs(deadline.saturating_sub(epoch_secs())).min(RELEASE_RETRY_CAP);
    Instant::now() + release_settings().boot_grace.max(remaining)
}

/// Release-queue driver (spawned as `chrome-run-releases`), restoring what a
/// previous process left queued before it takes its first pass.
pub async fn run_session_release_queue() {
    release_queue(crate::shutdown::shutdown_token()).await;
}

/// The loop itself against an explicit token, so a test can drive the real entry
/// point without the process-wide one: wake on an enqueue or a live run's tracker
/// going away, otherwise sleep until the earliest record that is due. Unlike the
/// other background loops it does not gate passes on `shutdown::aborting()` —
/// every pass releases runs that have already ended, so there is nothing a drain
/// needs to protect from it.
async fn release_queue(shutdown: CancellationToken) {
    restore_pending_releases();
    loop {
        tokio::select! {
            () = sleep_until_due(next_release_deadline()) => {}
            () = release_wake().notified() => {}
            () = shutdown.cancelled() => break,
        }
        release_due().await;
    }
}

/// Sleep until `deadline`, or forever when there is none — a queue whose every
/// record belongs to a live run has nothing to wait out, only a wake to answer.
async fn sleep_until_due(deadline: Option<Instant>) {
    match deadline {
        Some(at) => tokio::time::sleep_until(tokio::time::Instant::from_std(at)).await,
        None => std::future::pending::<()>().await,
    }
}

/// When the release task must run again: the earliest queued retry that is not
/// owned by a live run. `None` when nothing can be attempted, including a queue
/// holding only live-run records (a past deadline there would spin the driver).
fn next_release_deadline() -> Option<Instant> {
    pending_releases()
        .lock()
        .unwrap_poison()
        .iter()
        .filter(|entry| !run_namespace_is_live(&entry.namespace))
        .map(|entry| entry.next_attempt_at)
        .min()
}

/// Take the records a pass may attempt out of the queue: the ones whose hold or
/// retry backoff elapsed. A record whose namespace is live stays queued untouched —
/// a run owns those sessions, so its release waits, whether the holder is the run
/// itself, one still held by a consolidated round, or a later run of the same
/// durable singleton id (the earlier run's tabs stay until the chain ends). This is
/// the one delay a release does not bound: for as long as any holder keeps the
/// namespace live, the ended run's release lags its end by that much (see
/// [`register_run_namespace`]), and nothing shortens it but the holder going away.
fn take_releasable() -> Vec<PendingRunRelease> {
    let now = Instant::now();
    let mut queue = pending_releases().lock().unwrap_poison();
    let mut taken = Vec::new();
    let mut waiting = VecDeque::with_capacity(queue.len());
    while let Some(entry) = queue.pop_front() {
        let releasable = !run_namespace_is_live(&entry.namespace) && entry.next_attempt_at <= now;
        if releasable {
            taken.push(entry);
        } else {
            waiting.push_back(entry);
        }
    }
    *queue = waiting;
    taken
}

/// Outcome of one release attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReleaseOutcome {
    /// The session's persisted created-tab record is empty — its tabs are gone.
    Released,
    /// Not attempted. Never charged against `RELEASE_MAX_ATTEMPTS`; a
    /// [`SkipReason::NoBinary`] name climbs `RELEASE_MAX_SKIPS` instead.
    Skipped(SkipReason),
    /// Attempted and not verified.
    Failed,
}

/// Why a name was not attempted at all — the record's skip ladder keys on this rather
/// than on a re-derived guess about why the pass came up empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SkipReason {
    /// `chrome-use` cannot be run right now (absent, mid-swap): the record climbs the
    /// skip ladder for this one.
    NoBinary,
    /// The pass deliberately did not try — a run of the same agent id owns those sessions
    /// (the tracker wake re-drives the release), or the shutdown flush spent its budget.
    /// Put back untouched.
    Untried,
}

/// One `session stop` per name of every record, at most [`RELEASE_CONCURRENCY`]
/// children in flight ACROSS all of them, in round-robin order over the records: the
/// bound is global, so a record with dozens of names must not sit in front of every
/// other ended run's first attempt. Reports `(record index, name, outcome)` per name.
async fn attempt_releases(
    records: &[PendingRunRelease],
    per_name: impl Fn() -> Duration,
) -> Vec<(usize, String, ReleaseOutcome)> {
    let max_names = records
        .iter()
        .map(|record| record.names.len())
        .max()
        .unwrap_or(0);
    let mut jobs: Vec<(usize, String)> = Vec::new();
    for name_index in 0..max_names {
        for (index, record) in records.iter().enumerate() {
            if let Some(name) = record.names.get(name_index) {
                jobs.push((index, name.clone()));
            }
        }
    }
    futures_util::stream::iter(jobs)
        .map(|(index, name)| {
            let timeout = per_name();
            async move {
                // Liveness is re-checked here, per name: a pass can stay in
                // flight for minutes (one attempt bound × RELEASE_CONCURRENCY),
                // and a run of the same agent id resumed inside that window
                // re-attaches the very same names. `take_releasable`'s check
                // cannot cover it — it ran on a snapshot this pass outlived.
                if run_namespace_is_live(&records[index].namespace) {
                    return (index, name, ReleaseOutcome::Skipped(SkipReason::Untried));
                }
                // A spent deadline (the shutdown flush) must not spawn a child
                // that is killed on arrival — the module convention is an early
                // return (see `stop_session_daemon` in `chrome_daemon`).
                if timeout.is_zero() {
                    return (index, name, ReleaseOutcome::Skipped(SkipReason::Untried));
                }
                let outcome = release_one(&name, timeout).await;
                (index, name, outcome)
            }
        })
        .buffer_unordered(RELEASE_CONCURRENCY)
        .collect()
        .await
}

/// What one pass left of each record, indexed like `records`.
struct PassResult {
    /// The names the record still holds.
    open: Vec<String>,
    /// Whether the pass ran a `session stop` for the record at all. One attempted name
    /// charges the record — the bound is on the run's record, not per name — while a
    /// name skipped because its run is live again is not an attempt.
    attempted: bool,
    /// Whether `chrome-use` could not be run at all, which is what climbs the record's
    /// skip ladder (see [`SkipReason::NoBinary`]).
    unavailable: bool,
}

fn results_by_record(
    records: &[PendingRunRelease],
    outcomes: Vec<(usize, String, ReleaseOutcome)>,
) -> Vec<PassResult> {
    let mut results: Vec<PassResult> = records
        .iter()
        .map(|_| PassResult {
            open: Vec::new(),
            attempted: false,
            unavailable: false,
        })
        .collect();
    for (index, name, outcome) in outcomes {
        match outcome {
            ReleaseOutcome::Released => {}
            ReleaseOutcome::Failed => {
                results[index].open.push(name);
                results[index].attempted = true;
            }
            ReleaseOutcome::Skipped(SkipReason::NoBinary) => {
                results[index].open.push(name);
                results[index].unavailable = true;
            }
            ReleaseOutcome::Skipped(_) => results[index].open.push(name),
        }
    }
    results
}

/// The records one release pass took out of the queue, parked ([`PARKED_RELEASES`])
/// under this pass's id while it holds them so the durable file still describes them:
/// a pass can be dropped mid-attempt (the shutdown token aborts the release task, a
/// panic unwinds) or exit with the process, and a record out of both the queue and
/// the file would be stranded — the failure this whole mechanism exists to prevent.
/// [`ReleasePass::reconcile`] accounts for them; `Drop` puts back whatever it did not.
struct ReleasePass {
    /// This pass's identity in [`PARKED_RELEASES`], so it unparks its own records
    /// and never those of a pass overlapping it.
    id: u64,
    records: Vec<PendingRunRelease>,
    /// Namespace of the record `finish` is holding right now, so a panic inside it
    /// leaves exactly that record parked rather than dropping it everywhere.
    in_flight: Option<String>,
    reconciled: bool,
}

impl ReleasePass {
    /// The records a pass may attempt now, or `None` when nothing is due. They are
    /// parked for as long as the pass holds them out of the queue, so the file still
    /// describes them; the parked lock is held across the pop as well, so a persist can
    /// never look at a moment where they are in neither.
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

    /// Hand every record the pass took to `finish` (which either drops it or
    /// re-queues it) and disarm. Whatever a panic did not reach — and the record a
    /// panicking `finish` was holding — is dealt with by `Drop`.
    fn reconcile(
        &mut self,
        outcomes: Vec<(usize, String, ReleaseOutcome)>,
        mut finish: impl FnMut(PendingRunRelease, PassResult),
    ) {
        let mut results = results_by_record(&self.records, outcomes);
        // Popped together, from the end, so a panic in `finish` still leaves exactly
        // the records it never reached for the `Drop` guard. `finish` itself must not
        // panic: the record it holds is already out of the queue, and only the fact
        // that it is still parked keeps the file — and the next boot — describing it.
        while let Some((entry, result)) = self.records.pop().zip(results.pop()) {
            self.in_flight = Some(entry.namespace.clone());
            finish(entry, result);
            self.in_flight = None;
        }
        self.reconciled = true;
        self.unpark(None);
    }

    /// Stop parking this pass's records — the ones `finish` dropped are gone for
    /// good and the ones it re-queued are back in the queue, which is what the file
    /// carries. `keep` is the namespace a mid-panic `finish` was holding, the one
    /// record that has to stay parked.
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
        // The pass never finished: every record it had not handed to `finish` is put
        // back (`Retry` cannot shorten an eligibility that arrived meanwhile) and
        // unparked — the queue describes it again, so the file does too.
        for entry in self.records.drain(..) {
            merge_record(entry, Eligibility::Retry);
        }
        let in_flight = self.in_flight.take();
        self.unpark(in_flight.as_deref());
        persist_pending_releases();
    }
}

/// One pass over the records whose hold/retry is due. Records are taken out of the
/// queue by a [`ReleasePass`], which puts back whatever the pass could not finish;
/// the file is rewritten once per pass that took anything.
async fn release_due() {
    let Some(mut pass) = ReleasePass::take() else {
        return;
    };
    let timeout = release_settings().attempt_timeout;
    let outcomes = attempt_releases(&pass.records, || timeout).await;
    pass.reconcile(outcomes, |mut entry, result| {
        if result.open.is_empty() {
            debug!(
                namespace = %entry.namespace,
                sessions = entry.names.len(),
                "agent-run chrome sessions released"
            );
            return;
        }
        if !result.attempted {
            // Nothing was attempted. A record whose names were all skipped because its
            // run is live again is put back as it is: `unregister_run_namespace` wakes
            // the driver the moment that run's tracker is gone, so re-arming it would
            // only delay the release. A record skipped because chrome-use could not be
            // run climbs the skip ladder instead — a due record cannot then make the
            // driver take a pass and rewrite the file over and over, and one nothing can
            // act on cannot sit in the queue forever.
            entry.names = result.open;
            if result.unavailable {
                // Saturating: a corrupt file can carry any u32 here (see
                // `restore_pending_releases`).
                entry.skips = entry.skips.saturating_add(1);
                entry.next_attempt_at = Instant::now() + release_backoff(entry.skips);
                if entry.skips >= RELEASE_MAX_SKIPS {
                    info!(
                        namespace = %entry.namespace,
                        sessions = entry.names.len(),
                        skips = entry.skips,
                        "giving up on releasing agent-run chrome sessions — chrome-use was never available to run"
                    );
                    return;
                }
            }
            merge_record(entry, Eligibility::Retry);
            return;
        }
        entry.attempts = entry.attempts.saturating_add(1);
        entry.next_attempt_at = Instant::now() + release_backoff(entry.attempts);
        if entry.attempts >= RELEASE_MAX_ATTEMPTS {
            info!(
                namespace = %entry.namespace,
                sessions = result.open.len(),
                attempts = entry.attempts,
                "giving up on releasing agent-run chrome sessions — their tabs stay open in the browser"
            );
            return;
        }
        debug!(
            namespace = %entry.namespace,
            sessions = result.open.len(),
            attempts = entry.attempts,
            "agent-run chrome session release failed — retrying"
        );
        entry.names = result.open;
        merge_record(entry, Eligibility::Retry);
    });
    persist_pending_releases();
}

/// Last-chance pass for the shutdown and self-update paths: spends at most `budget`
/// attempting every record whose hold/retry has elapsed (same live-run guard, same
/// verified `session stop`; each attempt gets what is left of the budget), then
/// returns. A held record waits for a run the restart may re-drive, and the record is
/// durable either way, so whatever is left is picked up next boot. These attempts are
/// NOT charged against `RELEASE_MAX_ATTEMPTS`: the flush is the last chance before
/// the process goes down, not the bound that ends a record's life.
async fn flush_pending_run_releases(budget: Duration) {
    let deadline = Instant::now() + budget;
    let Some(mut pass) = ReleasePass::take() else {
        return;
    };
    let outcomes = attempt_releases(&pass.records, || {
        deadline.saturating_duration_since(Instant::now())
    })
    .await;
    let mut released = 0usize;
    let mut still_open = 0usize;
    pass.reconcile(outcomes, |mut entry, result| {
        released += entry.names.len() - result.open.len();
        if result.open.is_empty() {
            return;
        }
        still_open += result.open.len();
        entry.names = result.open;
        merge_record(entry, Eligibility::Retry);
    });
    persist_pending_releases();
    debug!(
        released,
        still_open, "shutdown: last-chance agent-run chrome session release"
    );
}

/// The chrome half of a shutdown, as one entry point: release what ended runs left
/// queued, then run the closing sweep. Both shutdown paths (GUI exit, self-update)
/// call this, so neither can do half of it — and an exit with a registered record
/// can spend both budgets back to back (`SHUTDOWN_RELEASE_FLUSH_BUDGET` here, the
/// sweep's own `SHUTDOWN_CLEANUP_TIMEOUT` after it).
pub async fn flush_and_close_all_chrome_sessions() {
    flush_pending_run_releases(SHUTDOWN_RELEASE_FLUSH_BUDGET).await;
    crate::tools::chrome::close_all_chrome_sessions().await;
}

/// One verified release attempt for one session — `Released` only on the chrome-use
/// verdict envelope with exit 0, whose proof is the live-verified `session stop`
/// behaviours in [`crate::tools::chrome_daemon`]. The adoption caveat cannot bite: an
/// agent-run session only ever drove tabs it opened itself, and that ownership record
/// survives a daemon death, so a later chrome-use changing things degrades to a
/// give-up, not a wrong close.
///
/// What this trusts, and all it trusts: the browser side's own report that the tabs it
/// created are closed — nothing here re-checks the browser.
///
/// [`SkipReason::NoBinary`] when there is no binary to run, so an absent or mid-swap
/// chrome-use does not spend the bounded attempts.
async fn release_one(name: &str, timeout: Duration) -> ReleaseOutcome {
    let Some(cli) = release_cli() else {
        debug!(
            session = name,
            "agent-run chrome session release skipped — chrome-use is not available"
        );
        return ReleaseOutcome::Skipped(SkipReason::NoBinary);
    };
    match run_cli_json_at(&cli, &["session", "stop"], Some(name), timeout).await {
        Ok(_) => {
            debug!(session = name, "agent-run chrome session released");
            ReleaseOutcome::Released
        }
        Err(err) => {
            debug!(
                session = name,
                error = err.as_deref().unwrap_or("no response"),
                "agent-run chrome session release failed — the session still owns its tabs, or chrome-use could not answer"
            );
            ReleaseOutcome::Failed
        }
    }
}

/// Every knob of the release path — one accessor so the test seam swaps the whole set
/// at once. A release build compiles no seam state (the `cli` stub below is
/// `#[cfg(test)]`) and always reads the constants above plus the storage root; the
/// real binary is resolved on demand by [`release_cli`], so a `Drop`-driven enqueue,
/// a persist or a restore never probes for it.
#[derive(Clone, Debug)]
struct ReleaseSettings {
    attempt_timeout: Duration,
    retry_base: Duration,
    /// Floor on a restored record's eligibility ([`RELEASE_BOOT_GRACE`]).
    boot_grace: Duration,
    /// The durable record file (`None` = nothing to write it to; see
    /// [`RELEASE_FILE_NAME`]).
    store: Option<PathBuf>,
    /// Stub binary the release tests spawn in place of the real `chrome-use`
    /// (see [`release_cli`]).
    #[cfg(test)]
    cli: Option<PathBuf>,
}

fn release_settings() -> ReleaseSettings {
    #[cfg(test)]
    if let Some(settings) = test_release_settings() {
        return settings;
    }
    ReleaseSettings {
        attempt_timeout: RELEASE_ATTEMPT_TIMEOUT,
        retry_base: RELEASE_RETRY_BASE,
        boot_grace: RELEASE_BOOT_GRACE,
        store: crate::config::CONFIG
            .try_storage_root()
            .map(|root| root.join(RELEASE_FILE_NAME)),
        #[cfg(test)]
        cli: None,
    }
}

/// The `chrome-use` binary one release attempt spawns (`None` = not available right
/// now). Resolved per name rather than once per pass, so a binary swapped out mid-pass
/// is seen — the [`cli_path`] probe behind it is cached (a lock plus an executable
/// check, not a filesystem search). An installed test seam decides on its own —
/// including "there is none" — so a test can put the release path in the window a
/// managed self-update swap creates.
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

/// Install release seams (a stub `chrome-use` plus shortened timings, so the
/// release path is exercised against a slow or failing browser side without a
/// browser and without real waits), returning the previous value so an RAII
/// guard can restore it on drop — including during a panic.
#[cfg(test)]
fn swap_release_settings(settings: ReleaseSettings) -> Option<ReleaseSettings> {
    RELEASE_SETTINGS.lock().unwrap_poison().replace(settings)
}

/// Restore a previously swapped-out release settings.
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

/// Test-only: one entry per queued record — its names, its attempt count and how
/// long it still has to wait before it may be attempted.
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

/// Test-only: the queued records as `(names, attempts)`, the shape most assertions
/// pin (a record's remaining eligibility gets its own assertion).
#[cfg(test)]
pub(crate) fn pending_names_and_attempts() -> Vec<(Vec<String>, u32)> {
    pending_releases_snapshot()
        .into_iter()
        .map(|(names, attempts, _)| (names, attempts))
        .collect()
}

/// Test-only: how long each queued record still has to wait before it may be
/// attempted.
#[cfg(test)]
pub(crate) fn pending_release_delays() -> Vec<Duration> {
    pending_releases_snapshot()
        .into_iter()
        .map(|(_, _, delay)| delay)
        .collect()
}

#[cfg(test)]
pub(crate) fn clear_pending_releases() {
    // Parked first, the order [`ReleasePass::take`] and [`persist_pending_releases`]
    // acquire the two in, so no lock cycle can form against a live pass.
    parked_releases().lock().unwrap_poison().clear();
    pending_releases().lock().unwrap_poison().clear();
}

/// Keeps the release path from writing the durable record file while it lives — for a
/// test that drives a real run end without a store of its own (a `ReleaseGuard` has
/// one), so it cannot leave records in the shared test storage root.
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
    use crate::tools::chrome::ChromeRunSessions;
    use std::fs;
    use std::path::Path;

    // -----------------------------------------------------------------------
    // Ended-run session releases
    // -----------------------------------------------------------------------

    /// Re-arm the attempt bound of the settings a [`ReleaseGuard`] installed — for
    /// a test whose short bound has served its purpose and whose later, successful
    /// pass must not race a cold or loaded stub spawn.
    fn widen_attempt_bound(attempt_timeout: Duration) {
        let mut settings = release_settings();
        settings.attempt_timeout = attempt_timeout;
        swap_release_settings(settings);
    }

    /// Re-arm the retry gap, to a value no test waits out: a release that still lands
    /// promptly can then only be explained by a wake.
    fn widen_retry_base(retry_base: Duration) {
        let mut settings = release_settings();
        settings.retry_base = retry_base;
        swap_release_settings(settings);
    }

    /// Poll until the guard's stub has been invoked at least `lines` times — the
    /// driver's progress, asserted where it cannot be driven directly.
    async fn wait_for_invocations(guard: &ReleaseGuard, lines: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while guard.log_lines().len() < lines {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the stub was invoked");
    }

    /// The names the record file describes, sorted.
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

    /// A stub `chrome-use`, the [`ReleaseSettings`] pointing at it, and a
    /// private record file. Restores the previous settings and drains the queue
    /// on drop, including during a panic.
    struct ReleaseGuard {
        mode: PathBuf,
        log: PathBuf,
        previous: Option<ReleaseSettings>,
        /// Holds the stub binary and the record file for the guard's lifetime.
        dir: tempfile::TempDir,
    }

    impl ReleaseGuard {
        /// Install the stub settings with no boot grace (every restored record is
        /// eligible at once) and warm the spawn path once: the first tokio child
        /// spawn in a fresh test binary costs ~300 ms (process machinery + cold
        /// exec), which under a short attempt bound would make a completed stub
        /// look like a timeout.
        async fn install(attempt_timeout: Duration) -> Self {
            Self::install_with(attempt_timeout, Duration::ZERO).await
        }

        /// `boot_grace` is the floor a restored record's eligibility gets — a
        /// test injects a short one instead of waiting a real minute. A record's
        /// hold is passed to [`queue_run_session_release`] by the caller (that is
        /// where the run end decides it), so a test injects that one directly.
        /// The record file lives in the guard's own temp dir, so the real
        /// `~/.mahbot/chrome-run-releases.json` is never touched.
        async fn install_with(attempt_timeout: Duration, boot_grace: Duration) -> Self {
            let dir = tempfile::tempdir().expect("release stub dir");
            let cli = dir.path().join("chrome-use");
            let mode = dir.path().join("mode");
            let log = dir.path().join("log");
            // The stub records every invocation's argv; `mode` decides whether
            // it answers with the verified envelope, refuses, or hangs. An exit
            // code alone is deliberately NOT a success mode — the release gates
            // on the envelope.
            fs::write(
                &cli,
                format!(
                    "#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\ncase \"$(cat {mode})\" in\n  \
                     hang) exec sleep 5 ;;\n  \
                     fail) printf '%s' '{{\"success\":false,\"error\":\"boom\"}}'; exit 1 ;;\n  \
                     ok) printf '%s' '{{\"success\":true}}' ;;\n\
                     esac\nexit 0\n",
                    log = log.display(),
                    mode = mode.display(),
                ),
            )
            .expect("write release stub");
            crate::util::test::make_executable(&cli);
            Self::write_mode(&mode, "ok");
            let previous = swap_release_settings(ReleaseSettings {
                cli: Some(cli.clone()),
                attempt_timeout,
                retry_base: Duration::ZERO,
                boot_grace,
                store: Some(dir.path().join(RELEASE_FILE_NAME)),
            });
            let guard = Self {
                mode,
                log,
                previous,
                dir,
            };
            // Warm-up call; the log is reset so it cannot be mistaken for a
            // real release attempt.
            let _ = release_one("warmup", attempt_timeout).await;
            guard.clear_log();
            guard
        }

        fn set_mode(&self, mode: &str) {
            Self::write_mode(&self.mode, mode);
        }

        fn write_mode(path: &Path, mode: &str) {
            fs::write(path, mode).expect("write release mode");
        }

        /// One line per stub invocation (the argv the release path built).
        fn log_lines(&self) -> Vec<String> {
            fs::read_to_string(&self.log)
                .unwrap_or_default()
                .lines()
                .map(str::to_string)
                .collect()
        }

        fn clear_log(&self) {
            fs::write(&self.log, "").expect("reset release log");
        }

        /// The guard's own durable record file (never the real `~/.mahbot` one).
        fn store_path(&self) -> PathBuf {
            self.dir.path().join(RELEASE_FILE_NAME)
        }
    }

    /// Stub log lines as a sorted multiset: one attempt pass runs its children
    /// concurrently, so their append order is not deterministic.
    fn sorted(mut lines: Vec<String>) -> Vec<String> {
        lines.sort();
        lines
    }

    impl Drop for ReleaseGuard {
        fn drop(&mut self) {
            restore_release_settings(self.previous.take());
            clear_pending_releases();
        }
    }

    /// Seed one ended run's record through the production tracker (dropped, as a
    /// real run's tracker is at run end) and hand it over, eligible at once. The
    /// names given are logical: the record carries them under the tracker's
    /// namespace, which [`run_session_names`] resolves for assertions.
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
        queue_run_session_release_after(&sessions, hold);
    }

    /// Physical names as a run records them: the tracker's namespace plus each logical
    /// name — the shape the durable record file carries. `for_run` derives the namespace
    /// from the agent id alone, so this is exactly what [`queue_names_with_hold`]
    /// recorded for that id.
    fn run_session_names(agent_id: &str, logical: &[&str]) -> Vec<String> {
        let ns = ChromeRunSessions::for_run(agent_id).namespace().to_string();
        logical.iter().map(|name| format!("{ns}{name}")).collect()
    }

    /// The release path's only invocation is `session stop --session <the run's
    /// own name>` per recorded name.
    fn stop_invocations(names: &[String]) -> Vec<String> {
        sorted(
            names
                .iter()
                .map(|name| format!("session stop --json --session {name}"))
                .collect(),
        )
    }

    /// Write the record file the way a previous process would have left it.
    fn write_record_file(path: &Path, records: &[PersistedRunRelease]) {
        fs::write(
            path,
            serde_json::to_string(records).expect("serialize release records"),
        )
        .expect("write release record file");
    }

    /// An ended run's record is done once every name verifies, and nothing but
    /// the run's own `session stop` invocations is ever spawned for it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_verified_release_empties_the_record() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-verified", &["agent-tab-v-default", "agent-tab-v-docs"]);
        assert_eq!(pending_releases_snapshot().len(), 1);

        release_due().await;

        assert!(
            pending_releases_snapshot().is_empty(),
            "both names released"
        );
        // Created-only: an enumeration sweep would show up here as an extra
        // line or a different verb.
        assert_eq!(
            sorted(guard.log_lines()),
            stop_invocations(&run_session_names(
                "run-verified",
                &["agent-tab-v-default", "agent-tab-v-docs"]
            ))
        );
    }

    /// A refused `session stop` keeps the record queued for exactly the bounded
    /// number of attempts, then the record is given up on.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_refused_release_is_retried_then_given_up_on_after_bounded_attempts() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-refused", &["agent-tab-r-stuck"]);
        guard.set_mode("fail");

        // retry_base is zero, so every pass is due.
        release_due().await;
        assert_eq!(
            pending_names_and_attempts(),
            vec![(run_session_names("run-refused", &["agent-tab-r-stuck"]), 1)],
            "a refused release stays queued as one failed attempt"
        );

        let mut passes = 1;
        while !pending_releases_snapshot().is_empty() {
            release_due().await;
            passes += 1;
            assert!(
                passes <= RELEASE_MAX_ATTEMPTS,
                "the record must be given up on within the bounded attempts"
            );
        }
        assert_eq!(passes, RELEASE_MAX_ATTEMPTS, "exactly the bounded attempts");
        assert_eq!(
            guard.log_lines().len(),
            RELEASE_MAX_ATTEMPTS as usize,
            "one `session stop` invocation per attempt"
        );
    }

    /// A `session stop` that outlives the attempt bound (slow browser side) is a
    /// failed attempt, not a release.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_timed_out_release_counts_as_one_failed_attempt() {
        let guard = ReleaseGuard::install(Duration::from_secs(1)).await;
        queue_names("run-slow", &["agent-tab-s-slow"]);
        guard.set_mode("hang");

        let started = Instant::now();
        release_due().await;
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "the 1 s attempt bound must cut the hanging child off"
        );
        assert_eq!(
            pending_names_and_attempts(),
            vec![(run_session_names("run-slow", &["agent-tab-s-slow"]), 1)],
            "a timed-out attempt is one failed attempt, and the name stays queued"
        );

        // The short bound has served its purpose; the success that follows must
        // not race a cold or loaded stub spawn.
        guard.set_mode("ok");
        widen_attempt_bound(Duration::from_secs(10));
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "released on the next pass"
        );
    }

    /// With no chrome-use to run, a due release is skipped rather than failed: a
    /// managed self-update swap leaves the binary unresolvable for a moment, and
    /// spending the bounded attempts on work that never happened would drop the
    /// record. It is re-armed on its own ladder instead, so the driver cannot spin on
    /// it either (that ladder is terminal — see
    /// `a_record_nothing_can_release_is_dropped_after_bounded_skips`).
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_release_with_no_chrome_use_binary_is_not_charged_an_attempt() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-absent", &["agent-tab-a-default"]);
        let mut settings = release_settings();
        settings.cli = None;
        settings.retry_base = Duration::from_secs(30);
        swap_release_settings(settings);

        release_due().await;

        assert_eq!(
            pending_names_and_attempts(),
            vec![(run_session_names("run-absent", &["agent-tab-a-default"]), 0)],
            "nothing was attempted, so nothing is charged"
        );
        assert!(guard.log_lines().is_empty(), "no child was spawned");
        assert!(
            pending_release_delays()[0] >= Duration::from_secs(25),
            "re-armed by a skip gap rather than left instantly due"
        );
    }

    /// The skip ladder is terminal: a record nothing can act on is dropped after
    /// [`RELEASE_MAX_SKIPS`], so the queue never holds one forever — and no attempt
    /// was charged for any of it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_record_nothing_can_release_is_dropped_after_bounded_skips() {
        // retry_base zero, so every skip re-arms to instantly due and the whole
        // ladder can be driven without waiting.
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-absent", &["agent-tab-a-default"]);
        let mut settings = release_settings();
        settings.cli = None;
        swap_release_settings(settings);

        for _ in 1..RELEASE_MAX_SKIPS {
            release_due().await;
            assert_eq!(
                pending_names_and_attempts(),
                vec![(run_session_names("run-absent", &["agent-tab-a-default"]), 0)],
                "still queued, and no attempt charged for an unavailable binary"
            );
        }
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "the record is dropped after its bounded skips"
        );
        assert!(
            guard.log_lines().is_empty(),
            "and nothing was ever spawned for it"
        );
    }

    /// Two passes may overlap (the driver and the shutdown flush): neither may erase
    /// the other's records from the file — the reason the queue and the file are two
    /// different views of a pass's records — and unparking is per pass. The put-back a
    /// dropped pass does is `an_aborted_pass_puts_its_records_back`'s subject.
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

        // The first pass ends (as the shutdown flush does while the driver's pass is
        // still in flight): its record goes back to the queue, and the second pass's
        // record is untouched by its unpark.
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

    /// A record queued by one process is retried by the next: the write at
    /// enqueue/after-a-pass is all the durable state, and restoring it brings
    /// back the names AND the attempt count (so the bound stays bounded across
    /// restarts).
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_queued_release_survives_the_process_that_queued_it() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-durable", &["agent-tab-d-default"]);
        guard.set_mode("fail");
        release_due().await;
        assert_eq!(
            pending_names_and_attempts(),
            vec![(
                run_session_names("run-durable", &["agent-tab-d-default"]),
                1
            )],
            "one refused attempt, re-queued"
        );

        // Everything the process held in memory is gone; the file is what a
        // restart has left.
        clear_pending_releases();
        assert!(pending_releases_snapshot().is_empty());

        restore_pending_releases();
        assert_eq!(
            pending_names_and_attempts(),
            vec![(
                run_session_names("run-durable", &["agent-tab-d-default"]),
                1
            )],
            "names and attempt count restored from the record file"
        );
    }

    /// A record file that cannot be read — or cannot be trusted — is never fatal: an
    /// unreadable one restores nothing, and a parseable one keeps only what is
    /// legitimate (a clamped deadline, and names under the record's own namespace).
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_corrupt_record_file_is_ignored() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        fs::write(guard.store_path(), "not a release record{").expect("write corrupt record file");

        restore_pending_releases();

        assert!(
            pending_releases_snapshot().is_empty(),
            "a corrupt file restores nothing and fails nothing"
        );

        // Parseable but absurd input: unlike the invalid file above it IS restored,
        // with its deadline clamped rather than trusted (the add would panic on it).
        fs::write(
            guard.store_path(),
            r#"[{"namespace":"agent-tab-absurd-","names":["agent-tab-absurd-default"],"next_attempt_at":18446744073709551615}]"#,
        )
        .expect("write absurd record file");

        restore_pending_releases();

        let delays = pending_release_delays();
        assert_eq!(delays.len(), 1, "the absurd record is restored");
        assert!(
            delays[0] <= RELEASE_RETRY_CAP,
            "its deadline is clamped to the ladder's cap: {delays:?}"
        );

        // The file is outside input: a name it carries is released only when it is an
        // agent-run session under the record's own namespace, so a foreign name cannot
        // point the queue at a session this record does not own.
        clear_pending_releases();
        fs::write(
            guard.store_path(),
            r#"[{"namespace":"agent-tab-absurd-","names":["default","link-enricher-x","","agent-tab-other-default","agent-tab-absurd-kept"]},{"namespace":"agent-tab-empty-","names":["default"]}]"#,
        )
        .expect("write foreign-names record file");

        restore_pending_releases();

        assert_eq!(
            pending_names_and_attempts(),
            vec![(vec!["agent-tab-absurd-kept".to_string()], 0)],
            "only names under the record's own namespace survive; a record left with none is not queued"
        );
        clear_pending_releases();
    }

    /// A restored record waits out the boot grace before anything may be
    /// attempted — the runs the daemon re-drives at boot are built after this
    /// restore, so the live-namespace guard cannot protect them yet.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_restored_release_waits_out_the_boot_grace() {
        let grace = Duration::from_millis(800);
        let guard = ReleaseGuard::install_with(Duration::from_secs(10), grace).await;
        queue_names("run-grace", &["agent-tab-g-default"]);
        clear_pending_releases();

        restore_pending_releases();
        release_due().await;
        assert_eq!(
            pending_releases_snapshot().len(),
            1,
            "restored at boot, the record is not attempted before the runs it may belong to exist"
        );
        assert!(guard.log_lines().is_empty());

        tokio::time::sleep(grace * 2).await;
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "released once the boot grace elapsed"
        );
        assert_eq!(
            sorted(guard.log_lines()),
            stop_invocations(&run_session_names("run-grace", &["agent-tab-g-default"]))
        );
    }

    /// A held record is skipped untried until its hold elapses
    /// (`a_verified_release_empties_the_record` covers the unheld end), and the
    /// run-end merge rule lets a later, unheld end of the same namespace release it
    /// without waiting that hold out.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_held_redriven_run_release_waits_out_its_hold() {
        let hold = Duration::from_millis(800);
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;

        queue_names_with_hold("run-held", &["agent-tab-h-default"], hold);
        release_due().await;
        assert_eq!(
            pending_releases_snapshot().len(),
            1,
            "a held record is left alone, and the skip is not an attempt"
        );
        assert!(
            guard.log_lines().is_empty(),
            "no `session stop` while the run it belongs to may still come back"
        );

        tokio::time::sleep(hold * 2).await;
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "released once the hold elapsed"
        );
        assert_eq!(
            sorted(guard.log_lines()),
            stop_invocations(&run_session_names("run-held", &["agent-tab-h-default"]))
        );
        guard.clear_log();

        // Same namespace, queued held first, then re-queued unheld: the shorter
        // eligibility must win, so the re-queue is not blocked by the hold set.
        queue_names_with_hold("run-shrunk", &["agent-tab-s-default"], hold * 4);
        queue_names("run-shrunk", &["agent-tab-s-default"]);
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "a re-queue without a hold makes the record eligible at once"
        );
        assert_eq!(
            sorted(guard.log_lines()),
            stop_invocations(&run_session_names("run-shrunk", &["agent-tab-s-default"]))
        );
    }

    /// A record whose run is live again is skipped untried — that run owns the
    /// sessions — and becomes releasable the moment it is gone.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_record_whose_run_is_live_again_is_never_attempted() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        let live = ChromeRunSessions::for_run("run-live");
        live.track(&format!("{}default", live.namespace()));
        queue_run_session_release(&live, false);

        release_due().await;

        assert_eq!(
            pending_releases_snapshot().len(),
            1,
            "a live run's record stays queued and is not an attempt"
        );
        assert!(
            guard.log_lines().is_empty(),
            "no `session stop` is ever spawned for a live run's sessions"
        );

        drop(live);
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "released once the run that owns them is gone"
        );
        assert_eq!(guard.log_lines().len(), 1);
    }

    /// The production entry point ([`run_session_release_queue`], driven here through
    /// its loop with a local token): a queued record is released, and a record whose
    /// run is live again waits — not on a poll interval, on the wake that run's own
    /// tracker sends when it goes away.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_driver_releases_a_queued_record_and_waits_for_a_live_run_to_go() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        // A retry gap far longer than this test: a release that still lands promptly
        // can only be explained by the wake.
        widen_retry_base(Duration::from_secs(600));
        let live = ChromeRunSessions::for_run("run-driver");
        let names = run_session_names("run-driver", &["agent-tab-driver-default"]);
        live.track(&names[0]);
        queue_run_session_release(&live, false);

        let shutdown = CancellationToken::new();
        let driver = tokio::spawn(release_queue(shutdown.clone()));

        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(
            guard.log_lines().is_empty(),
            "the driver leaves a live run's session alone"
        );
        assert_eq!(pending_releases_snapshot().len(), 1);

        drop(live);
        wait_for_invocations(&guard, 1).await;
        assert_eq!(
            guard.log_lines(),
            stop_invocations(&names),
            "the driver releases it through the run's own `session stop`"
        );
        assert!(
            pending_releases_snapshot().is_empty(),
            "and drops the record once it is released"
        );

        shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(5), driver)
            .await
            .expect("the driver stops on the token")
            .expect("the driver task does not panic");
    }

    /// One record per run: a second end of the same (agent-id-derived) namespace
    /// adds only the names it recorded and consumes no attempt.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_requeue_of_the_same_run_does_not_double_count() {
        // The merge path persists too, so the record file needs the guard's
        // private dir (and the queue is drained for the next test).
        let _guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        clear_pending_releases();
        let first = ChromeRunSessions::for_run("run-merge");
        let names = run_session_names("run-merge", &["agent-tab-m-default", "agent-tab-m-docs"]);
        first.track(&names[0]);
        queue_run_session_release(&first, false);
        let second = ChromeRunSessions::for_run("run-merge");
        second.track(&names[0]);
        second.track(&names[1]);
        queue_run_session_release(&second, false);
        drop((first, second));

        assert_eq!(
            pending_names_and_attempts(),
            vec![(names, 0)],
            "one record carrying both ends' names"
        );
        clear_pending_releases();
    }

    /// The shutdown flush spends its budget on the records whose hold (or retry
    /// backoff) elapsed and leaves the held ones alone: a held record waits for a
    /// run the restart may re-drive, and being durable it is picked up next boot.
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

        // A hanging browser side cannot hold the shutdown flush past its budget.
        guard.set_mode("hang");
        let started = Instant::now();
        flush_pending_run_releases(Duration::from_millis(500)).await;
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "the flush stops at its budget"
        );
        assert_eq!(
            pending_releases_snapshot().len(),
            2,
            "an unreleased name stays queued"
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
            "the flush drains what it can release and leaves the held record for its resumed run"
        );
        assert_eq!(
            sorted(guard.log_lines()),
            stop_invocations(&run_session_names("run-flush", &["agent-tab-f-default"])),
            "one verified `session stop` per name, and never one for the held record"
        );
    }

    /// A run resumed while a pass is in flight re-attaches the very same names,
    /// so the attempt is re-checked per name and skipped — never released out
    /// from under the run that owns it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_run_that_comes_back_mid_pass_keeps_its_sessions() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        let live = ChromeRunSessions::for_run("run-midpass");
        let name = format!("{}default", live.namespace());
        live.track(&name);
        // The record a pass would have taken just before the run came back.
        let record = PendingRunRelease {
            namespace: live.namespace().to_string(),
            names: vec![name.clone()],
            attempts: 0,
            skips: 0,
            next_attempt_at: Instant::now(),
            held: false,
        };

        let outcomes =
            attempt_releases(std::slice::from_ref(&record), || Duration::from_secs(10)).await;

        assert_eq!(
            outcomes,
            vec![(0, name, ReleaseOutcome::Skipped(SkipReason::Untried))]
        );
        assert!(
            guard.log_lines().is_empty(),
            "no `session stop` is spawned for a run that is live again"
        );
        drop(live);
    }

    /// A run end that arrives while a pass holds the record must not be lost by the file:
    /// queue and parked copies are folded, so the later (held) pair survives a process
    /// death in that window instead of the stale unheld snapshot the pass took.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_hold_queued_while_a_pass_holds_the_record_survives_in_the_file() {
        let guard = ReleaseGuard::install_with(Duration::from_secs(10), Duration::ZERO).await;
        queue_names("run-fold", &["agent-tab-f-default"]);
        let pass = ReleasePass::take().expect("a due record is taken");

        // The same run is cut off while the pass holds it: the queue now says the release
        // must be held, the pass's parked snapshot still says otherwise.
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

    /// A `finish` that panics hands nothing back, yet the record it was holding is
    /// still parked and persisted, so the next boot restores it instead of leaving
    /// those sessions stranded everywhere.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_panicking_finish_leaves_its_record_for_the_next_boot() {
        let _guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-panic", &["agent-tab-p-default"]);

        let mut pass = ReleasePass::take().expect("a due record is taken");
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pass.reconcile(Vec::new(), |_, _| panic!("finish panicked"));
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

    /// A pass dropped mid-attempt (shutdown token, panic) puts back what it took
    /// and counts no attempt: a record left out of the queue would also be left
    /// out of the next persist, stranding its sessions.
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

    /// A held record re-mints its hold from the boot that restores it — the run
    /// it waits for is re-dispatched after the restart, so the deadline the
    /// previous process minted says nothing.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_restored_held_record_holds_from_boot() {
        let guard = ReleaseGuard::install_with(Duration::from_secs(10), Duration::ZERO).await;

        // The file a previous process left: its own namespace, and the physical
        // names under it.
        let namespace = ChromeRunSessions::for_run("run-resume")
            .namespace()
            .to_string();
        write_record_file(
            &guard.store_path(),
            &[PersistedRunRelease {
                namespace,
                names: run_session_names("run-resume", &["agent-tab-r-default"]),
                attempts: 0,
                skips: 0,
                // Long past: the hold the previous process minted has expired.
                next_attempt_at: 1,
                held: true,
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
}
