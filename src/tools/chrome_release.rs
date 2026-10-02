//! Ended-run chrome session releases.
//!
//! chrome-use ≥1.5.101 preserves external Chrome tabs across daemon idle recycling, so the
//! sessions an agent run opened must be closed explicitly — a run cannot do it in its own
//! tail, which is skipped whenever its future is dropped. Every run end queues the names it
//! recorded, from its run-end guard's `Drop` (`crate::agent::RunEndCleanup`), and the
//! `chrome-run-releases` task closes them out of band.
//!
//! What closes them is the browser's own extension, not chrome-use's endpoint-bound
//! `session stop`: one batched call into [`crate::tools::chrome_tabs::close_sessions`] per
//! pass, over every due record's names and the reclaim set below, where the browser's own
//! answers are the only proof (that module owns the reading).
//!
//! A record is not dropped on a retry: it ends only when the browser says every name is
//! gone, or that the record's own routes cannot close the rest — the extension's ledger no
//! longer holding them, or an answer that this run's close left them standing repeated
//! [`MAX_LEFT_OPEN_ATTEMPTS`] times ([`record_disposition`], which reports both with the run
//! they came from) — or when the queue cap evicts it, which loses the record and never the
//! tabs ([`MAX_PENDING_RELEASES`]). A name the browser answered for and left open is reported
//! once and retried (see [`RetryLeg::LeftOpen`]); once the browser has answered that way
//! [`MAX_LEFT_OPEN_ATTEMPTS`] times the record is dropped with the leftover reported, and the
//! still-open groups are left to the reclaim sweep, which re-examines them at
//! [`STUCK_REVISIT`]. Anything else it did not answer for stays queued on a growing, capped
//! backoff and no attempt cap. A settled session is then let go
//! ([`crate::tools::chrome_tabs::forget_settled_sessions`]), and that let-go queue is the one
//! piece of this path that lives in memory only — what it holds is chrome-use's own
//! bookkeeping plus this module's scratch session. The scratch session is a
//! `mahbot-chrome-ephemeral-*` name, which the product's session sweeps own for what the queue
//! cannot settle; a settled run's session record is the one thing nothing else sweeps, and it is
//! bookkeeping alone, never a tab. The shutdown flush reaches that queue
//! ([`flush_pending_run_releases`]) without draining it: its budget is below one whole stop, so a
//! name its cut-short stop could not settle dies with the process. While the process lives, an
//! unconfirmed let-go of our own session is queued ([`queue_forget`]) and retried on the queue's
//! own deadline, never left for the next start. What this path can never lose is a run's tab.
//!
//! The browser reads run in a chrome-use session this module BORROWS wherever the work has one
//! ([`read_session`]): the sessions the records name are the run's own — the normal run-end case,
//! where the run's session is the one whose tabs are being closed — so a pass with one to borrow
//! mints no session of ours; a run whose session is still there gains nothing from the read, and
//! one whose session chrome-use had already recycled gains the page that session works in, in its
//! own group (see [`read_session`] for what removes it). Only a pass with no name
//! to borrow at all falls back to this product's own session
//! ([`crate::tools::chrome_tabs::OWN_SESSION`]), and every pass that used it lets it go
//! ([`crate::tools::chrome_tabs::release_own_session`]) through the same connection-independent
//! route the runs are closed by, so its scratch group cannot be stranded by an endpoint re-mint.
//! That session is this module's one member of the `mahbot-chrome-ephemeral-*` family, which the
//! product's session sweeps own for what its members could not close — including a scratch group
//! whose pass died before its let-go. The one state that let-go can establish against us — the
//! browser not confirming that group gone, a read that looks for it included — is surfaced once
//! ([`OWN_SESSION_REASON`]) rather than accepted quietly, and the session is queued for a stop
//! ([`queue_forget`]) so the group cannot sit in the owner's strip until the next start. The one
//! page any of this can add is the one chrome-use opens when a read starts a session whose daemon
//! had been recycled: in a borrowed session it lands in that session's own group, which the record
//! holding the name closes at its own pass — unless that namespace is protected then, when the page
//! waits in that run's group for a later pass — and for the fallback it is the scratch group the
//! same pass lets go.
//!
//! [`MAX_PENDING_RELEASES`] is the one other way a record leaves the queue: its eviction
//! closes nothing and needs no report of its own, because those tabs are then `agent-tab-*`
//! groups no live run, no record and no resumable job claims — what the sweep owns from its
//! next ask on, and reports itself if it cannot close them. The run the evicted record came
//! from is remembered for a while ([`EVICTED_RUNS`]), so that report can still name it.
//!
//! Leftovers nobody recorded — a run lost to a crash, force quit, hard kill or self-update,
//! and the residue of earlier runs and versions — are reclaimed unprompted by a sweep of the
//! browser's live tab groups whose title starts with `crate::tools::chrome::AGENT_TAB_PREFIX`
//! ([`crate::tools::chrome_tabs::agent_group_names`]), so the product does not depend on a
//! run-end hook having fired. It asks at the start of the process only when the durable record
//! file handed it something, and after that only while something is unfinished or an event arms
//! one ([`schedule_reclaim_now`] — a run end or a queue-cap eviction — a parked verdict's
//! revisit ([`schedule_reclaim_revisit`]), a record concluding with leftovers no route closed,
//! [`reclaim_unfinished`]) — never on a timer, so an idle product asks the browser nothing and a
//! start with nothing pending opens no page. Its reads borrow the sessions the records name
//! ([`read_session`]), so the product's own scratch group appears only for a leftover no record
//! names any more. [`ProtectedNamespaces`] is the set it leaves alone, and it does not
//! re-interrogate what it has already concluded. The product's other scratch families are NOT
//! this sweep's: a `mahbot-chrome-ephemeral-*` leftover and the link enricher's sessions are the
//! product's own session sweeps' to close. That is because this sweep can only protect what it
//! can attribute, and a live fetch whose daemon has been idle-recycled is one it cannot tell from
//! a leftover: chrome-use keeps tabs across recycling, so a live fetch's session can be
//! unregistered while its tabs and its work are live. That family is therefore swept at the
//! moments nothing is in flight for it — product start — and closed per fetch, and a leftover of
//! it whose session record is gone is a pre-existing, documented limit (see
//! [`crate::tools::chrome_daemon`]'s own note on the
//! dead-orphan case) rather than something this sweep may guess at.
//!
//! Reporting is truthful about its own reach: what this module can attribute to a session is
//! reported once per `(message, reason)` and with the run wherever durable state still names one
//! — or where the queue cap remembered it ([`EVICTED_RUNS`]). What is deliberately not in it is
//! what is not a leftover — a namespace [`ProtectedNamespaces`] still keeps for a run, and the
//! classes a route closing by exact group title cannot ATTRIBUTE (a rename off
//! `crate::tools::chrome::AGENT_TAB_PREFIX`, tabs moved out of a session's group) — because
//! reporting them would blame the owner's own tabs. A browser side with no ownership door is
//! reported once as a fact about the host, with no run and no session names
//! ([`NO_OWNERSHIP_DOOR_REASON`]).
//!

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
use tracing::{debug, info};

/// First retry gap and its cap: the gap doubles per attempt. A record whose tabs are
/// still there is retried on that ladder — a leftover must not cost a spawn every few
/// seconds — until the browser either answers that they are gone or has answered
/// [`MAX_LEFT_OPEN_ATTEMPTS`] times that this run's close left them standing (see
/// [`record_disposition`]).
const RELEASE_RETRY_BASE: Duration = Duration::from_secs(5);
const RELEASE_RETRY_CAP: Duration = Duration::from_mins(30);
/// How many passes may get the browser's own answer that this run's close left the tabs
/// standing before the close route concludes. An answer repeated that often, each after a full
/// close attempt and fresh reads, is the strongest evidence any route here can produce that it
/// is not going to close them — and a record must never be retried forever without reaching a
/// conclusion (see [`record_disposition`] for what the conclusion costs).
const MAX_LEFT_OPEN_ATTEMPTS: u32 = 3;
/// How long a held release waits, and the only bound on a run that never comes back.
/// A restart inside the window re-mints the whole hold from that boot (see
/// [`restored_eligibility`]); which ends hold is [`holds_run_end`]'s rule.
pub(crate) const RELEASE_HOLD_REDRIVEN_RUN: Duration = Duration::from_mins(30);
/// Which run ends hold their session release back: a drain/shutdown cut and a workspace-pause
/// freeze, held for [`RELEASE_HOLD_REDRIVEN_RUN`] — the ends after which the same run is
/// normally rebuilt under its own durable id (the dispatches re-read a durable id, and a paused
/// phase job waits for the unpause re-drive), so its resumed segment must still find the tabs it
/// was working with. The hold is a property of the end alone — nothing checks that a run is
/// actually re-driven — so a run nothing comes back for pays it as latency. `internal_cancel` is
/// a cooperative end too, but deliberately not held: it may release while a replacement segment
/// starts up on the same names, which is why every attempt re-checks per name. Every other end
/// releases at once — including a run aborted at a round/research deadline or lost to a panic,
/// which ends through its run-end guard's `Drop` with the record queued unheld.
///
/// The hold is a floor, not the whole protection: a namespace a non-terminalized job still
/// names is left alone whatever the hold says (see [`ProtectedNamespaces`]).
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
/// down must not accumulate records without bound, so a record is evicted to make room
/// (see [`evict_oldest`]). Eviction closes nothing and is logged, not reported: the evicted
/// record's tabs stay plain `agent-tab-*` groups no live run, no record and no resumable job
/// claims any more, so the reclaim sweep owns them from its next ask on and reports them itself
/// if it cannot close them — with the run the record came from, remembered for a while
/// ([`EVICTED_RUNS`]).
const MAX_PENDING_RELEASES: usize = 256;
/// How long an evicted record's run is remembered for the sweep's report of its tabs.
/// A leftover the sweep can close is closed on its next ask, so the attribution is only
/// ever read for one that survived a sweep or two; an hour covers the browser side being
/// brought back by hand without keeping the map alive for the whole process.
const EVICTED_RUN_TTL: Duration = Duration::from_hours(1);
/// Last-chance release budget on the shutdown/self-update path, in the spirit of the
/// chrome-side `SHUTDOWN_CLEANUP_TIMEOUT`: a degraded browser side cannot be waited out while
/// the process is going down. It sits below what one close pass may legitimately need (see
/// [`crate::chrome::SESSION_STOP_TIMEOUT`]), so a slow but working pass is cut off here and its
/// record left for the next boot — whose pass owns the charging rules (see
/// [`flush_pending_run_releases`]).
const SHUTDOWN_RELEASE_FLUSH_BUDGET: Duration = Duration::from_secs(10);

/// One ended run's unreleased session names.
#[derive(Clone)]
struct PendingRunRelease {
    /// Namespace of the run that opened them — the identity a live run of the
    /// same agent id shares, which records are merged by.
    namespace: String,
    names: Vec<String>,
    attempts: u32,
    next_attempt_at: Instant,
    /// Whether [`holds_run_end`] held this record's eligibility back — see
    /// [`restored_eligibility`] for what that does to it.
    held: bool,
    /// The agent id the run ended under, carried so an issue report can name the run
    /// it came from. Empty for a record restored from a file written before this field
    /// existed.
    run: String,
    /// Passes that asked the browser to remove this record's tabs and got the answer that this
    /// run's close left them standing ([`crate::tools::chrome_tabs::RetryLeg::LeftOpen`]).
    /// At [`MAX_LEFT_OPEN_ATTEMPTS`] the close route stops retrying and the leftovers are
    /// reported (see [`record_disposition`]).
    left_open: u32,
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
    /// Epoch seconds at which the record becomes eligible (`0` = now).
    #[serde(default)]
    next_attempt_at: u64,
    /// Whether the record's eligibility is a hold ([`PendingRunRelease::held`]).
    #[serde(default)]
    held: bool,
    /// The run the record belongs to ([`PendingRunRelease::run`]). `default` is
    /// mandatory here: an old record file carries no `run`, and without it the whole
    /// pending queue would stop parsing and be silently dropped.
    #[serde(default)]
    run: String,
    /// [`PendingRunRelease::left_open`], defaulted for the same reason as `run`.
    #[serde(default)]
    left_open: u32,
}

/// One run whose record the queue cap evicted. The record itself is gone — its tabs are the
/// sweep's from then on — but the run it belonged to is what lets the sweep's report of a
/// leftover it cannot close name the run instead of nobody (see [`evicted_run`]).
struct EvictedRun {
    namespace: String,
    run: String,
    at: Instant,
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
/// When the reclaim sweep last asked the browser and how many consecutive asks it could
/// not get answered, which is what spaces its attempts out (see [`reclaim_due`]).
static RECLAIM: OnceLock<Mutex<ReclaimSchedule>> = OnceLock::new();
/// Namespaces the sweep has already concluded no route here closes, and when it did
/// (see [`stuck`]). In-process only: a restart tries once more, which is what a fresh
/// look at a possibly repaired browser is worth.
static STUCK_NAMESPACES: OnceLock<Mutex<HashMap<String, Instant>>> = OnceLock::new();
/// Sessions a pass closed but had no budget left to let go of, each with the number of
/// passes that have failed to let it go (see [`forget_settled`]).
static PENDING_FORGET: OnceLock<Mutex<Vec<(String, u32)>>> = OnceLock::new();
/// The runs whose records the queue cap evicted, so the sweep's own report of the tabs they
/// left can still name them (see [`evicted_run`]).
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

/// Fold `incoming` into `entry`: the union of the names, the higher counts (attempts and
/// left-open answers alike, so a re-queue cannot restart a count the browser has already run
/// up) and the eligibility pair `keep` picks among the two.
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
    // A membership set built from the record's current names and grown as the incoming ones
    // arrive: a record's names are gathered from every merge (a re-queued end, a restore, the
    // queue's own fold), so the duplicate check must not be a linear scan per name.
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

/// Take out the record to evict to make room for one more: the oldest NOT inside a hold, or
/// `None` when every record is inside one. Dropping such a record would lose the tabs of a run
/// that may still come back to them, which is worse than a queue that grows until one of those
/// holds elapses.
fn evict_oldest(queue: &mut VecDeque<PendingRunRelease>) -> Option<PendingRunRelease> {
    let now = Instant::now();
    let index = queue.iter().position(|record| !within_hold(record, now))?;
    queue.remove(index)
}

/// Remember the run an evicted record came from, so the sweep's own report of the leftover
/// tabs can still name it. Bounded like the queue the record left: an entry older than
/// [`EVICTED_RUN_TTL`] is dropped, and so is the oldest once the deque is as long as the
/// queue's own cap.
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

/// The run an evicted record's namespace came from, when one was evicted inside the TTL —
/// the newest such entry, since a namespace evicted twice belongs to the later run.
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

/// Put a record in the queue, merging into the record already queued for the same run instead
/// of keeping two: the merge keeps the union of the names and the higher counts, so a re-queue
/// cannot discard what an earlier end recorded. `eligibility` resolves which pair wins (see
/// [`Eligibility`]).
fn merge_record(entry: PendingRunRelease, eligibility: Eligibility) {
    // The namespace is what a record is identified by — records are merged, and
    // matched against the runs live right now, by it — so a record without one is
    // never queued.
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
        // Logged and not reported: the sweep owns these tabs from its next ask on and
        // reports them itself (see [`MAX_PENDING_RELEASES`]) — with the run the record came
        // from, which is what the memory below carries to it. The ask is armed here because
        // an eviction need not come from a run end (a pass putting its records back, a
        // restore): the sweep has to be what picks the group up either way.
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

/// Hand a run's ended-session record over for release. `held` is [`holds_run_end`]'s verdict for
/// how the run ended; `run` is the agent id, carried so a report can name the run the leftover
/// tabs came from. Sync and panic-free so it can run from a `Drop`, and durable: the write below
/// is what makes the release survive this process. The names are snapshotted here (`ChromeTool`
/// records a session before it dispatches, so the set is complete once the run is gone), and a
/// later run of the same durable id re-attaches them, which the live-namespace guard covers.
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

/// [`queue_run_session_release`] with the hold spelled out, for the tests that need one
/// other than the constant's.
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
    // The run end is the event that brings a crashed run's leftover to light: names nothing
    // recorded are exactly what a crashed or force-quit run leaves, and a sweep asks the
    // browser about them at no extra cost to the owner — the pass that closes this record's
    // tabs has already opened the session the sweep's own read runs in. A run that queued
    // nothing must not arm one ([`unregister_run_namespace`] deliberately does not).
    schedule_reclaim_now();
    release_wake().notify_one();
}

/// Restore the records a previous process left queued, once at the start of
/// [`run_session_release_queue`]: a release still pending when the release task stopped is
/// retried after the restart instead of being lost, a held record survives the restart it waits
/// for (with its attempt count, so the bound stays bounded across restarts), and an unreadable
/// file is ignored rather than treated as a failure.
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
        // What [`merge_record`] would refuse is not restored — and must not count as restored
        // either: it is not work this process owes the browser a read for (a corrupt file's
        // nameless record would otherwise arm a sweep with nothing queued).
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
        // Work a previous process left queued is exactly what the sweep is owed a read for —
        // including the names no record carries, which a crash or a force quit leaves behind.
        schedule_reclaim_now();
        release_wake().notify_one();
    }
}

/// Write the whole queue — plus what the passes in flight hold ([`PARKED_RELEASES`]) — to the
/// record file (compact JSON, atomic tmp+rename): fail-open, a no-op with no storage root, and a
/// record with no names is never written. Both locks are held across the write as well as the
/// snapshot (parked before queue, the order [`ReleasePass::take`] parks in), because two
/// persists sharing the one `.json.tmp` could publish a torn file the next boot ignores.
fn persist_pending_releases() {
    let Some(path) = release_settings().store else {
        return;
    };
    let parked = parked_releases().lock().unwrap_poison();
    let queue = pending_releases().lock().unwrap_poison();
    // Indexed by namespace rather than searched per record: the fold runs over the whole
    // queue every pass, which is the one place the queue's size is paid for quadratically.
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

/// A persisted record as this process's eligibility instant. A held record re-mints its whole
/// hold from THIS boot ([`RELEASE_HOLD_REDRIVEN_RUN`]): the deadline the previous process
/// minted says nothing about when the run it waits for will exist, and trusting it would
/// downgrade a restart that outlasted the hold to the boot grace. Every other record is floored
/// at the boot grace (with the shipped settings the stored deadline never wins, both being the
/// same minute) and clamped to the ladder's cap, since a further-out deadline is corrupt input
/// the add would panic on, taking the release task down for the whole boot.
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

/// The loop itself against an explicit token, so a test can drive the real entry point without
/// the process-wide one: wake on an enqueue, on a live run's tracker going away, or on the next
/// deadline the schedule carries, whichever comes first. Unlike the other background loops it
/// does not gate passes on `shutdown::aborting()` — every pass releases runs that have already
/// ended, so there is nothing a drain needs to protect from it.
async fn release_queue(shutdown: CancellationToken) {
    restore_pending_releases();
    loop {
        tokio::select! {
            () = wait_for_release_work() => {}
            () = release_wake().notified() => {}
            () = shutdown.cancelled() => break,
        }
        release_due().await;
    }
}

/// The driver's own wait: until the next deadline, or forever when there is none — the loop
/// is then woken by an enqueue ([`queue_run_session_release`], [`schedule_reclaim_now`]), by a
/// live run's tracker going away ([`unregister_run_namespace`]), or by shutdown.
async fn wait_for_release_work() {
    match next_release_deadline() {
        Some(deadline) => {
            tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
        }
        None => std::future::pending().await,
    }
}

/// When the release task must run again: the earliest queued retry not owned by a live run, the
/// reclaim sweep's own scheduled ask ([`ReclaimSchedule::next_at`]) and the let-go queue's retry
/// ([`forget_retry_at`]), whichever comes first; `None` when none exists, and the loop then waits
/// on a wake alone. There is deliberately no standing interval behind it — each of the three is
/// either an event's ask or a leftover's own retry, spaced by the same capped ladder, and the
/// let-go queue's element is what makes a session nothing will stop (`~/.chrome-use` bookkeeping,
/// or this module's own scratch session) be tried again rather than wait for the next start.
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

/// Take the records a pass may attempt out of the queue: the ones whose hold or retry
/// backoff elapsed, skipping any whose namespace a live run still owns — its tabs stay until
/// the chain ends, the one delay a release does not bound (see [`register_run_namespace`]).
/// A namespace durable state still names for a resumable job is left alone at close time
/// instead — see [`ProtectedNamespaces`].
fn take_releasable() -> Vec<PendingRunRelease> {
    let now = Instant::now();
    let mut queue = pending_releases().lock().unwrap_poison();
    let mut taken = Vec::new();
    let mut waiting = VecDeque::with_capacity(queue.len());
    while let Some(mut entry) = queue.pop_front() {
        let releasable = !run_namespace_is_live(&entry.namespace) && entry.next_attempt_at <= now;
        if releasable {
            // A record is taken only when its eligibility elapsed; for a held record that
            // means the hold has been spent, so it goes back ordinary — from here on it is
            // just a rung of the retry ladder, and nothing may protect it (or its namespace)
            // as if a run could still come back to it.
            entry.held = false;
            taken.push(entry);
        } else {
            waiting.push_back(entry);
        }
    }
    *queue = waiting;
    taken
}

/// What one record's browser answers left.
struct RecordDisposition {
    /// Names still open that the browser did not settle (`Retry`) or the pass never
    /// asked about (no binary to ask): the record stays queued for exactly these.
    retry: Vec<String>,
    /// The retried names the browser ANSWERED for and did not remove (see
    /// [`RetryLeg::LeftOpen`]): kept apart from the silent ones because their answer is what
    /// the record counts and reports.
    left_open: Vec<String>,
    /// The names the record is done with: the browser answered as not the extension's to remove,
    /// or answered this run's close and left them standing [`MAX_LEFT_OPEN_ATTEMPTS`] times.
    /// Reported with the run, then dropped from the record — the reclaim sweep keeps examining
    /// their group at [`STUCK_REVISIT`].
    unclosable: Vec<String>,
}

/// Decide one record's fate from the pass's browser answers: a name the browser settled as gone
/// is dropped; one it says is not the extension's to remove is a leftover to report; one it
/// answered for and left open is retried and reported; anything else — including every name of a
/// pass whose reads settled nothing — stays queued.
///
/// What concludes those left-open names is this record's own count: at most
/// [`MAX_LEFT_OPEN_ATTEMPTS`] of them, one per pass, never reset, and about this run's close
/// however many of the record's names the browser answered that way. A name left open in the
/// pass that reaches that count is a leftover too (reported under the same kind) instead of
/// being retried forever; the early `chrome-tabs-left-open:` row still tells the owner as soon
/// as the browser first answers that way.
fn record_disposition(
    entry: &mut PendingRunRelease,
    outcomes: &HashMap<String, TabOutcome>,
) -> RecordDisposition {
    // One left-open answer per pass, however many of the record's names were answered that
    // way: what is counted is the browser's repeated answer about this run's close.
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
            // The browser has answered the same way [`MAX_LEFT_OPEN_ATTEMPTS`] times: this
            // route is not going to close them, so they conclude like any other leftover.
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
/// pass's id so the durable file still describes them: a record out of both the queue and the
/// file would be stranded — the failure this whole mechanism exists to prevent.
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
    /// re-queues it) with the pass's own browser answers, and disarm. Whatever a panic
    /// did not reach — and the record a panicking `finish` was holding — is dealt with
    /// by `Drop`.
    fn reconcile(
        &mut self,
        outcomes: &HashMap<String, TabOutcome>,
        mut finish: impl FnMut(PendingRunRelease, RecordDisposition),
    ) {
        // Popped one at a time, so a panic in `finish` still leaves exactly the
        // records it never reached for the `Drop` guard. `finish` itself must not
        // panic: the record it holds is already out of the queue, and only the fact
        // that it is still parked keeps the file — and the next boot — describing it.
        while let Some(mut entry) = self.records.pop() {
            self.in_flight = Some(entry.namespace.clone());
            let disposition = record_disposition(&mut entry, outcomes);
            finish(entry, disposition);
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

/// One pass over the records whose hold/retry is due, plus the reclaim sweep. All the
/// work is [`attempt_pass`]'s; this only collects the sweep's names, reads the durable
/// resume set once, and reports what the pass could not close.
async fn release_due() {
    let cli = release_cli();
    let deadline = Instant::now() + release_settings().attempt_timeout;
    let pass = ReleasePass::take();
    let taken = pass.is_some();
    let sweep_due = reclaim_due();
    // The one durable read of the pass, and only when something will use it: the sweep
    // is due, or records are waiting on the browser. An idle wake (an enqueue that finds
    // the namespace live, a retry that is not due yet) stays a no-op.
    let durable = if taken || sweep_due {
        durable_resume_namespaces().await
    } else {
        Vec::new()
    };
    let mut protected = ProtectedNamespaces::snapshot(&durable);
    // The session this pass runs its browser reads in, decided once for the sweep's read and the
    // close alike: a name the work itself already has wherever one exists (see [`read_session`]).
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
/// being closed — then the first name any record still queued or parked carries, which is the
/// run's own session even when the run's tracker still holds its namespace (the run-end case: the
/// record is queued the moment the run ends, and the namespace unregisters a moment later) or
/// when its release is held or restored. A borrowed name is always one a record holds, so a page
/// our read opens in its group (the session had been idle-recycled while its tabs survived) is
/// removed by that record's own close: the name is one of its own — bar the case where that
/// namespace is still protected when the close runs (its run holds it again, or a resumable job
/// claims it), which leaves the page inside that run's group until a later pass takes it.
/// [`chrome_tabs`](crate::tools::chrome_tabs)' Honest limits carry the whole of it, including the
/// page a stop-then-read pass mints.
///
/// [`OWN_SESSION`] is what is left for a pass with no name to borrow at all — a sweep armed for a
/// leftover no record names any more (a parked verdict's revisit, a queue-cap eviction). That is
/// the one pass that shows the product's own scratch group, and the only reason it still exists.
///
/// The queue's names may belong to a live run, and are borrowed anyway: the read is one call on a
/// session that run already has, serialized by chrome-use behind whatever it is doing, while
/// minting a session of ours instead would be a page in the owner's browser (which is the whole
/// footprint this choice exists to remove). Nothing of a live run's is closed by it — the live
/// half of [`ProtectedNamespaces`] keeps its names out of the close.
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

/// Close the browser's tabs for `names` — except the ones a run may own again right now.
/// `protected` is the pass's snapshot of every namespace that must be left alone, refreshed
/// here with the live map and held records as they are immediately before the close: the
/// reclaim sweep above spends a browser read (seconds) before this runs, and a run that
/// registers its namespace in that window owns its tabs — the pass's own snapshot would miss
/// it, and the close would take them from under the run. (`close_sessions` re-asks at each act
/// it performs — see [`crate::tools::chrome_tabs::OwnedAgain`].)
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
    // What the close asks right before each act: the live map and the held records as they
    // are at that moment. Only those two — the durable half is what the pass snapshotted
    // once, and it cannot change inside one pass.
    let owned_again = ProtectedNamespaces::protects_live_or_held;
    chrome_tabs::close_sessions(cli, &unprotected, session, deadline, &owned_again)
        .await
        .into_iter()
        .collect()
}

/// Let go of the sessions earlier passes closed but had no budget to forget, plus `just_settled`,
/// and keep what still has to wait: chrome-use's own per-session bookkeeping, plus this module's
/// own scratch session when its let-go could not be confirmed — whose stop is what closes its
/// group — and never a run's tab.
///
/// Which names wait is [`LetGoOutcome`]'s four classes: one whose stop the helper did not settle
/// is charged one more failed attempt, one the pass could not give a whole bound — either no
/// stop fitted or the one it spawned was cut short — keeps the rung it had, because it never
/// heard a real answer, one a run owns again leaves the queue outright, because stopping that
/// session is what would end the run, whose own end hands over its own release, and this module's
/// own scratch session waits behind a call of ours still running in it, owing nothing until that
/// call is done. The roster a pass
/// carries is bounded both ways: a name past [`MAX_PENDING_FORGET`] is trimmed, and one that
/// reaches [`MAX_FORGET_ATTEMPTS`] failed stops is dropped with one `info!` naming it and the
/// count — except this module's own scratch session, which neither bound drops (see both
/// constants). A name the queue keeps is retried by the attempt its own deadline arms
/// ([`forget_retry_at`]), which is how a session nothing answers for is tried again without an
/// event in between.
///
/// The wait it comes back on is the highest rung the queue holds ([`release_backoff`]) — one
/// batch, retried at the rung of its most-charged member, so a burst is never hammered and a
/// fresh name may wait a charged one's rung — floored at one whole stop when anything was
/// deferred behind a call of ours ([`BUSY_FORGET_FLOOR`]), since nothing then is owed until the
/// pass holding that call has spent its own let-go step. Deferring is deferred work, not a
/// failure — the pass that deferred it spent its budget on a close, and the next pass tries the
/// let-go with a fresh one.
///
/// The queue is never taken out of its slot: this pass reads it, runs the stops, and writes back
/// what it did not let go — so a pass overlapping this one, or a pass whose future is dropped
/// mid-stop (a shutdown cutting the flush), can never take names away with it. What the queue
/// gains meanwhile is kept after the names this pass held. One interleaving is not covered: a name
/// this pass itself let go is dropped from the queue even if a pass overlapping it kept that name
/// — which follows this pass's own observation that the stop landed, so the session is settled and
/// nothing is lost; the opposite order can leave a settled name queued for one redundant stop.
async fn forget_settled(cli: &Path, just_settled: &[String], deadline: Instant) {
    let mut todo: Vec<(String, u32)> = pending_forget().lock().unwrap_poison().clone();
    // The names this pass takes out of the live queue, read BEFORE the cap below trims them: a
    // queued name the cap drops must not be written back by the merge at the end, or the cap
    // would be undone and the queue would churn instead of settling on the oldest names.
    let snapshot: HashSet<String> = todo.iter().map(|(name, _)| name.clone()).collect();
    // First-seen wins, so a name already waiting keeps the attempts it has run up.
    todo.extend(just_settled.iter().map(|name| (name.clone(), 0)));
    dedupe_and_bound_forgotten(&mut todo);
    if todo.is_empty() {
        // Nothing to try. Only a queue that is still empty loses its retry — a pass that
        // overlapped this one armed its own for whatever it queued.
        let queue = pending_forget().lock().unwrap_poison();
        if queue.is_empty() {
            *forget_retry().lock().unwrap_poison() = None;
        }
        return;
    }
    let names: Vec<String> = todo.iter().map(|(name, _)| name.clone()).collect();
    // Asked at each act inside: the run that owns a session again is what the stop would
    // end, and a let-go step must never race one.
    let owned_again = ProtectedNamespaces::protects_live_or_held;
    let left = chrome_tabs::forget_settled_sessions(cli, &names, deadline, &owned_again).await;
    let mut kept: Vec<(String, u32)> = Vec::new();
    for (name, attempts) in todo {
        if left.owned.contains(&name) {
            // A name a run owns again leaves this queue instead of climbing it: its own end
            // hands over its own release, so nothing here is waiting for us.
            continue;
        }
        if left.deferred.contains(&name) || left.busy.contains(&name) {
            // Nothing was asked of the helper — for a deferral its whole bound did not fit, for
            // a busy name a call of ours held the session — so nothing is charged and the rung
            // it has stays as it is.
            kept.push((name, attempts));
            continue;
        }
        if !left.tried.contains(&name) {
            continue; // let go: the helper stopped the session and dropped its record
        }
        let attempts = attempts.saturating_add(1);
        // The one name the bound never drops is this module's own scratch session (see
        // [`MAX_FORGET_ATTEMPTS`]); every other name here leaves the queue.
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
    // What is left drives its own retry: the attempt a name has run up is the rung the queue
    // waits on ([`forget_retry_at`]), so a session the helper would not stop is tried again
    // without a run end or a sweep arming one — and once nothing is left, nothing is scheduled.
    let deferred_behind_call = !left.busy.is_empty();
    let mut queue = pending_forget().lock().unwrap_poison();
    // What the pass leaves in the queue: the names it held, minus what it let go (or found a run
    // owning again), plus whatever a pass overlapping it queued while the stops ran — the
    // snapshot above is what tells the two apart, so the cap's own dropped names are not added
    // back here. The queue was never taken out, so nothing else may be removed. A rung already
    // written by an overlapping pass is the high-water one: a count only ever climbs.
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
        // Nothing is owed until the call that held the session is done, and waking at the base
        // gap would only bring the driver back to defer again while it runs: the wake waits for
        // the covering pass's own let-go step (see [`BUSY_FORGET_FLOOR`]).
        let floor = Instant::now() + BUSY_FORGET_FLOOR;
        Some(retry_at.map_or(floor, |at| at.max(floor)))
    } else {
        retry_at
    };
}

/// Put `name` in the let-go queue for the driver's next attempt, keeping the place (and the
/// attempts) a name already has — and arm the queue's own retry deadline here
/// ([`forget_retry_at`]), so a name queued by a pass whose tail runs out before the let-go step
/// (or by the process's last pass) still has an attempt behind it rather than waiting for an
/// unrelated event. One case that deadline cannot cover: while no chrome-use binary resolves,
/// [`forget_retry_at`] answers `None` and no pass runs the queue at all — the attempt then waits
/// for the next event, which is the only thing a pass with no binary could do anyway.
fn queue_forget(name: &str) {
    let mut queue = pending_forget().lock().unwrap_poison();
    if queue.iter().any(|(queued, _)| queued == name) {
        return;
    }
    queue.push((name.to_string(), 0));
    dedupe_and_bound_forgotten(&mut queue);
    *forget_retry().lock().unwrap_poison() = Some(Instant::now() + release_backoff(0));
}

/// When the let-go queue must be tried again. `None` while the queue is empty — a queue whose
/// work is done schedules nothing.
static FORGET_RETRY: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();

fn forget_retry() -> &'static Mutex<Option<Instant>> {
    FORGET_RETRY.get_or_init(|| Mutex::new(None))
}

/// The queue's own retry deadline, as [`next_release_deadline`] reads it: `None` while no
/// chrome-use binary is resolvable, because no pass can run the queue then. A deadline left
/// over from before the binary went away (a managed self-update swap, a removed install) would
/// otherwise be read as due for the rest of the process, and the driver would come back from
/// its sleep into a pass that can do nothing — a spin, not a retry. The stored deadline stays
/// where it is: a binary that resolves again finds the queue still at its own rung, and it is run
/// by whatever comes next — a run end, a record's own retry, the shutdown flush, whose budget is
/// below one whole stop but reaches the queue all the same. Nothing wakes the driver for this
/// alone.
fn forget_retry_at() -> Option<Instant> {
    release_cli()?;
    *forget_retry().lock().unwrap_poison()
}

/// Cap on the sessions waiting to be let go: a burst the helper will not stop must not grow
/// the queue without bound, so past the cap the oldest names go — chrome-use's own bookkeeping
/// under `~/.chrome-use`, never a run's tab and never this module's own scratch session (see
/// both constants). A full queue clears over one or more passes: a pass asks as many stops as
/// its budget holds, and each of those is a session daemon being shut down, so the queue is what
/// keeps a burst from growing while it waits its turn.
const MAX_PENDING_FORGET: usize = 64;
/// Floor on the wake the queue arms when a pass found a call of ours still running in a name's
/// session ([`crate::tools::chrome_tabs::LetGoOutcome::busy`]). The pass holding that call spends
/// its own let-go step on the name right after, a step whose dominant cost is one whole
/// `session stop`, so the queue comes back at that stop's bound instead of at the base gap, which
/// would only bring the driver back to defer again while the step runs.
const BUSY_FORGET_FLOOR: Duration = crate::chrome::SESSION_STOP_TIMEOUT;
/// How many attempts may fail to let one session go before it leaves the queue with one
/// `info!` naming it. That trace is not the report the Issues view is owed — the scratch
/// session's own row is filed once by [`crate::tools::chrome_tabs`]'s caller, and what this
/// queue's attempts own is chrome-use's bookkeeping rather than a run's tab — and the attempt
/// bound, not the trace, is what stops the retry.
///
/// This module's own scratch session ([`chrome_tabs::OWN_SESSION`]) is the one name the bound
/// never drops: what IT can leave behind is a group in the owner's tab strip rather than helper
/// bookkeeping, and a name that then waits for the next process start is exactly what the queue
/// exists to avoid. It stays at the bound's rung and later, retried by the queue's own deadline
/// for as long as the helper cannot settle it.
const MAX_FORGET_ATTEMPTS: u32 = 5;

/// Keep `names` in first-seen order without repeats, bounded by [`MAX_PENDING_FORGET`]:
/// what a pass carries is what it will try to stop, so a queue nothing drains must not
/// grow into one that a pass can never work through. What the cap drops is chrome-use's
/// own per-session bookkeeping — never a run's tab.
fn dedupe_and_bound_forgotten(names: &mut Vec<(String, u32)>) {
    let mut seen: HashSet<String> = HashSet::with_capacity(names.len());
    names.retain(|(name, _)| seen.insert(name.clone()));
    let excess = names.len().saturating_sub(MAX_PENDING_FORGET);
    if excess == 0 {
        return;
    }
    let mut dropped = 0usize;
    names.retain(|(name, _)| {
        // The product's own scratch session is never the name the cap drops — see
        // [`MAX_FORGET_ATTEMPTS`] — so one more bookkeeping name goes in its place.
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

/// The names one pass is about: the reclaim set first, then every name the records it took
/// still carry, each once.
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

/// Every name the pass's records carry, in first-seen order — the set the driver's
/// pass and the shutdown flush both work from.
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

/// What the reclaim sweep's own names came to: how many it closed, and the reports the rest owe.
/// A reclaimed group belongs to no record any more (the cap's eviction kept the run, not the
/// record), so its namespace comes from the group's own session name and its run is whatever
/// [`evicted_run`] has, empty when nothing does.
///
/// A name the browser did not settle — Unclosable, or left open as the browser's own — is
/// remembered as stuck so the next sweeps skip it until its verdict is worth re-checking, and
/// its revisit stays armed ([`park_stuck`]); a name the browser never answered keeps an ask
/// armed too ([`reclaim_unfinished`]); either kind expires with its window.
///
/// The retry ladder is reset here from the sweep's own answers: a pass whose names were all
/// answered for leaves it at the floor, a pass with a Silent outcome keeps the rung
/// [`reclaim_unfinished`] just charged.
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
            // No route here can close these tabs: a leftover, reported and done with — and
            // not interrogated in full again until the verdict is worth re-checking.
            TabOutcome::Unclosable => {
                issues.push(reclaimed_issue(name, &namespace, IssueKind::Unclosable));
                park_stuck(&namespace);
            }
            // The browser answered and left its own tabs: a contradiction, not a verdict —
            // a later sweep retries them, and the report keeps them from ageing out unseen.
            TabOutcome::Retry(RetryLeg::LeftOpen) => {
                issues.push(reclaimed_issue(name, &namespace, IssueKind::LeftOpen));
                park_stuck(&namespace);
            }
            // Nothing was established about that group, so it is not abandoned: the pass
            // keeps an ask armed for it below, rather than letting it vanish with this pass.
            TabOutcome::Retry(RetryLeg::Silent) => silent = true,
        }
    }
    // One ask, one rung: the sweep's failure is the pass, not how many of its names the
    // browser left unanswered, so the ladder is charged once however many came back Silent.
    // An ask the browser answered whose names all settled (or were parked) is clean and
    // leaves the ladder at its floor; a failed read resets nothing here — it charged the
    // ladder itself.
    if silent {
        reclaim_unfinished();
    } else if answered {
        reset_reclaim_failures();
    }
    (reclaimed, issues)
}

/// Remember that nothing else here is worth asking about `namespace` for a while: the entry
/// expires on its own after [`STUCK_REVISIT`], whether or not any group is ever read again
/// (see [`stuck`]). This leftover's revisit is also what the sweep leaves scheduled behind
/// it ([`schedule_reclaim_revisit`]), so a browser that recovers is looked at again.
fn park_stuck(namespace: &str) {
    stuck_namespaces()
        .lock()
        .unwrap_poison()
        .insert(namespace.to_string(), Instant::now());
    schedule_reclaim_revisit();
}

/// One report about a reclaimed group — the same row a record's own name would produce,
/// except that the namespace comes from the name and the run is only known when the cap
/// evicted the record that carried it (see [`reclaim_issues`]).
fn reclaimed_issue(name: &str, namespace: &str, kind: IssueKind) -> Issue {
    Issue {
        kind,
        namespace: namespace.to_string(),
        run: evicted_run(namespace).unwrap_or_default(),
        name: name.to_string(),
    }
}

/// Let go of this product's own browser-read session ([`chrome_tabs::OWN_SESSION`]) if a pass of
/// ours actually used it — a pass that borrowed a name the work itself has has nothing of ours
/// open — and surface the one state that
/// is ours: `false` from [`chrome_tabs::release_own_session`] means this pass made the reads and
/// the browser did not confirm the scratch group gone — a read may itself have failed, so the
/// group may or may not be there — and the row says exactly that. Reported once, through the
/// same `(message, reason)` de-dupe as every other row; the normal path (nothing owed, or the
/// group confirmed gone) says nothing, and a host where no call of ours was ever answered gets
/// no row either ([`chrome_tabs::own_scratch_group_possible`]).
///
/// An unconfirmed let-go is handed on rather than left for the next start: the session goes into
/// the let-go queue ([`queue_forget`]), and the let-go step that runs a few lines later in this
/// same pass is the ONE place the graceful `session stop` is asked for — the route that closes
/// the tabs a session created even when the extension's ledger no longer reaches them. A pass
/// that runs out of budget before that step leaves the name queued with its retry armed, and the
/// product's session sweeps stay the backstop.
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

/// One row for a host whose browser answers settle nothing this route can act on — filed at most
/// once, and only after such an answer actually happened in this process: such an answer settles
/// no name, which must not stay a `debug!` that ages out. The latch is one atomic load, so an
/// ordinary pass pays nothing for it.
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

/// One pass over the records a [`ReleasePass`] took, plus the names the reclaim sweep
/// found: every name the pass is about goes to the browser in one batched call, the
/// browser's own answers decide each record's fate, and the reports the Issues view is
/// owed come back. [`ReleasePass`] puts back whatever the pass could not finish.
///
/// This product's own browser-read session, if the pass used it, is let go right after the
/// close — before the let-go step below, which can spend what is left of the budget on children
/// of its own — so the scratch group it opened cannot outlive a pass that ran out of budget in
/// its tail.
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
        // Nothing due and nothing to reclaim: the only work left is letting go of the
        // session the reclaim's own read ran in, and of the sessions an earlier pass closed
        // but had no budget to forget. The ask that found nothing to reclaim was still an
        // ask the browser answered, so it leaves the ladder at its floor.
        if ask.answered {
            reset_reclaim_failures();
        }
        if let Some(cli) = cli {
            let_go_own_session(cli, deadline).await;
            forget_settled(cli, &[], deadline).await;
        }
        // A read of THIS pass may have been the unreadable one (the ask that found nothing to
        // reclaim): the row belongs to the pass that met it, not to whenever one runs next.
        report_unreadable_answers().await;
        return Vec::new();
    }

    let outcomes = close_unprotected(cli, &names, session, protected, deadline).await;

    // Every browser read of this pass is behind us, and the let-go step below can spend what
    // is left of the budget on children of its own — so this product's own session, if the pass
    // borrowed no name and used it, is let go HERE, where it is still affordable. It is asked
    // for only when a call of ours is outstanding, so a pass that borrowed a session spawns
    // nothing for it.
    if let Some(cli) = cli {
        let_go_own_session(cli, deadline).await;
    }

    // Let go of the settled sessions' chrome-use records and daemons — the close leaves
    // both behind — but never a name a run owns again by the time the browser answered.
    // This is the pass's THIRD read of the live map, and the one moment the close's own
    // fresh read cannot cover: a run re-dispatched while the close was running registers
    // its namespace now, and its tabs must not be let go under it.
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
            // The record ends here, so those tabs are now `agent-tab-*` groups no record, live
            // run or resumable job claims: arm the sweep so the still-open group is examined
            // there — and parked like any other leftover once the sweep answers.
            if unclosable > 0 {
                arm_reclaim_at(Instant::now() + STUCK_REVISIT);
            }
            if disposition.retry.is_empty() {
                if unclosable == 0 {
                    info!(
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
            // Charged on every pass that leaves names open — no bound, so the record is
            // retried for as long as the browser has not answered — and spaced by the
            // capped ladder so a browser side that stays degraded is not hammered.
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
    // A read of THIS pass may have been the unreadable one, so the host row is filed here rather
    // than waiting for whichever pass runs next.
    report_unreadable_answers().await;
    issues
}

/// Last-chance pass for the shutdown and self-update paths: spends at most `budget`
/// closing every record whose hold/retry has elapsed (same live-run guard, same
/// browser-driven close; the whole sequence gets what is left of the budget), then
/// returns. It runs no reclaim sweep and files no leftover row of its own — the next boot's
/// normal pass owns both — though it can owe the one row for this module's own scratch
/// session ([`OWN_SESSION_MESSAGE`]), which is let go here, and it reaches the let-go queue
/// here too (best-effort: this budget is below one whole stop, so only a stop the
/// helper settles inside it lets a name go), before the process goes down; a name it could not
/// let go goes down with the process — the product's
/// own scratch family is what the next start's session sweep reaches, and a settled run's
/// session name is left as its chrome-use record alone, which nothing in this product sweeps.
/// A held record waits for a run the restart may re-drive, and the record is durable either
/// way, so whatever is left is picked up next boot. What is uncharged here is the retry
/// ladder's attempt count: the flush is the last chance before the process goes down, not a
/// step on that ladder, so a record it re-queues keeps the count it had. What IS charged is a
/// left-open answer the browser did give: it is counted here ([`record_disposition`]) so a
/// repeated one still reaches its conclusion, and the record is kept for the next boot to
/// report.
async fn flush_pending_run_releases(budget: Duration) {
    let deadline = Instant::now() + budget;
    let cli = release_cli();
    let Some(mut pass) = ReleasePass::take() else {
        // No record is due, but the let-go queue may still hold a name — a session let go of
        // but not stopped, this module's own scratch session among them — and this is its last
        // chance before the process goes down. Within this budget's stop bound, which is why
        // the stop is attempted best-effort ([`forget_settled_sessions`]).
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
        // This product's own session, if the pass used it, goes before the process does; the
        // closing sweep after this one owns the name's family, but its own route is
        // endpoint-bound.
        let_go_own_session(cli, deadline).await;
        // And the let-go queue is reached here rather than left to the next start: an
        // unconfirmed let-go just queued that scratch session, and only a graceful stop closes
        // the group it may have left. The flush's own budget is below one whole stop, so the
        // stop is best-effort — a name it could not settle stays in the queue, which the
        // process is about to drop (see this function's own doc for what that leaves).
        forget_settled(cli, &[], deadline).await;
    }
    let mut released = 0usize;
    let mut still_open = 0usize;
    pass.reconcile(&outcomes, |mut entry, disposition| {
        // Only a `Gone` answer is settled; an `Unclosable` or unanswered name is left in
        // the record — off the retry ladder and untouched, keeping the attempt count it had —
        // for the next boot's pass to report or retry.
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

/// The chrome half of a shutdown, as one entry point: release what ended runs left
/// queued, then run the closing sweep. Both shutdown paths (GUI exit, self-update)
/// call this, so neither can do half of it — and an exit with a registered record
/// can spend both budgets back to back (`SHUTDOWN_RELEASE_FLUSH_BUDGET` here, the
/// sweep's own `SHUTDOWN_CLEANUP_TIMEOUT` after it).
pub async fn flush_and_close_all_chrome_sessions() {
    flush_pending_run_releases(SHUTDOWN_RELEASE_FLUSH_BUDGET).await;
    crate::tools::chrome::close_all_chrome_sessions().await;
}

/// Names of the live agent tab groups the sweep owns this pass: every group the browser
/// titles under [`AGENT_TAB_PREFIX`] whose namespace no run may still own — and none whose
/// namespace already got a verdict ([`stuck`]). `due` is the caller's own read of
/// [`reclaim_due`], taken before the pass so the same answer can gate letting the door
/// session go afterwards. Only the session names a queue or parked record still holds are left
/// to that record, matched exactly ([`claimed_names`]): its own pass retries them on its own
/// ladder, while a name the record dropped — one it settled unclosable — is the sweep's again,
/// so a browser that recovers still gets that group closed.
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
    // Every due sweep, before the browser is read: a verdict expires with its window even
    // when nothing reads it again (see [`stuck`]) — and even when the binary is not
    // resolvable, so a verdict cannot outlive its window on a host without chrome-use.
    prune_stuck();
    let Some(cli) = cli else {
        // Nothing to ask the browser with right now (a managed self-update swap window, a
        // host without chrome-use): the ask that was owed goes up the ladder rather than
        // being left standing at once, so a binary that is not resolvable yet — an install
        // in flight, the swap window — does not cost the process its ask.
        reclaim_unfinished();
        return ReclaimAsk::unanswered();
    };
    // The names the queue's records still hold, built directly rather than through the
    // live/held pair: it is what keeps a second driver off the names a record's own pass
    // retries, while a name a record dropped is examined here.
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
            // Nothing on this pass, and on a host with no ownership door the sweep cannot
            // enumerate the browser's groups at all: that host-level gap is what the Issues
            // view is owed rather than a debug line that ages out. Told through the same
            // de-dupe as every other report, so a sweep that keeps failing costs one row.
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

/// What one reclaim ask came to: the names the browser answered with, and whether the browser
/// answered at all. `answered: false` is a read that failed or a binary that was not resolvable
/// — the asking legs charge the ladder for those themselves, and nothing may be read as a clean
/// ask from one.
struct ReclaimAsk {
    names: Vec<String>,
    answered: bool,
}

impl ReclaimAsk {
    /// An ask that was never put to the browser.
    fn unanswered() -> Self {
        Self {
            names: Vec::new(),
            answered: false,
        }
    }
}

/// When the reclaim sweep may ask the browser again, and how many consecutive asks it could
/// not get answered. `next_at` is the ask the schedule carries: `Some` exactly while something
/// is unfinished or an event armed one, `None` after a clean ask that left no leftover behind —
/// nothing here re-arms on an interval (see [`schedule_reclaim_now`] and [`reclaim_unfinished`]).
/// `failures` is the ladder rung a failed ask is spaced by, kept across asks so a browser side
/// that stays down is not re-shovelled.
///
/// The schedule starts with NOTHING armed: a process with no queued record and no leftover it
/// knows about has no reason to read the browser at all, and a start that finds nothing pending
/// must not open a page (`restore_pending_releases` arms the ask when there IS something).
/// Nothing here is timed, so the derived default is the whole constructor.
#[derive(Default)]
struct ReclaimSchedule {
    next_at: Option<Instant>,
    failures: u32,
}

fn reclaim_schedule() -> &'static Mutex<ReclaimSchedule> {
    RECLAIM.get_or_init(|| Mutex::new(ReclaimSchedule::default()))
}

/// Arm an ask at `at`, keeping whichever ask is already sooner: an event that arrives while a
/// leftover's revisit is parked must not push that revisit back, and a revisit parked while
/// an event's ask is armed must not push that ask back either.
fn arm_reclaim_at(at: Instant) {
    let mut schedule = reclaim_schedule().lock().unwrap_poison();
    schedule.next_at = Some(schedule.next_at.map_or(at, |old| old.min(at)));
}

/// The ask the leftovers the sweep could not close still owe: the earliest `at +
/// [`STUCK_REVISIT`]` over the stuck verdicts still INSIDE their window. Strictly in the
/// future by construction, so deriving an ask from it can never arm one in the past.
fn next_reclaim_revisit() -> Option<Instant> {
    stuck_namespaces()
        .lock()
        .unwrap_poison()
        .values()
        .filter(|concluded| concluded.elapsed() < STUCK_REVISIT)
        .map(|concluded| *concluded + STUCK_REVISIT)
        .min()
}

/// Arm an ask now: a run end queued names, which is the event that brings a crashed run's
/// leftover to light. The ladder's failure count is left alone, so this asks once now and a
/// failure still spaces the next ask out as it did.
fn schedule_reclaim_now() {
    arm_reclaim_at(Instant::now());
}

/// Arm a revisit [`STUCK_REVISIT`] away: a leftover the sweep could not close is owed another
/// look rather than left with nothing scheduled behind it.
fn schedule_reclaim_revisit() {
    arm_reclaim_at(Instant::now() + STUCK_REVISIT);
}

/// How long a namespace the sweep already concluded nothing can close — or that the browser
/// keeps answering for, leaving its own tabs open — is left alone before the full close is tried
/// again. A verdict is remembered so a stuck leftover cannot cost a `session stop` per name plus
/// two browser reads on every sweep, while a browser that recovered still gets another chance:
/// the entry expires with that window ([`prune_stuck`]), and the window is also the revisit a
/// successful sweep leaves armed ([`next_reclaim_revisit`]).
const STUCK_REVISIT: Duration = Duration::from_mins(30);

fn stuck_namespaces() -> &'static Mutex<HashMap<String, Instant>> {
    STUCK_NAMESPACES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Whether the sweep may skip `namespace` this pass: a verdict was reached for it less than
/// [`STUCK_REVISIT`] ago. The entry expires for real after that window ([`prune_stuck`] drops
/// the expired ones, and a read drops its own), so a namespace nothing reads again does not keep
/// its verdict for the whole process.
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

/// Drop the verdicts whose [`STUCK_REVISIT`] window has passed — the sweep calls this before
/// it reads the browser, so a namespace only ever read through [`stuck`] still loses its
/// verdict in time.
fn prune_stuck() {
    let now = Instant::now();
    stuck_namespaces()
        .lock()
        .unwrap_poison()
        .retain(|_, concluded| now.saturating_duration_since(*concluded) < STUCK_REVISIT);
}

/// Whether the sweep may ask the browser this pass: only once the ask the schedule carries is
/// due. Nothing arms one by itself — a start with nothing pending opens no page — so an ask
/// exists only while something is unfinished or an event armed one ([`restore_pending_releases`],
/// [`schedule_reclaim_now`], a parked verdict's revisit); [`reclaim_answered`] and
/// [`reclaim_unfinished`] are what an ask leaves behind.
fn reclaim_due() -> bool {
    let mut schedule = reclaim_schedule().lock().unwrap_poison();
    match schedule.next_at {
        // The pass that takes an ask is the one that runs it: clearing it here is what lets an
        // ask armed while that pass runs (a run end, a parked verdict) stay a new one for the
        // next wake instead of being consumed by the pass already in flight.
        Some(at) if at <= Instant::now() => {
            schedule.next_at = None;
            true
        }
        _ => false,
    }
}

/// A clean ask: the browser answered, so nothing stays scheduled except what the sweep's
/// own leftovers still owe — the earliest revisit of a parked verdict ([`next_reclaim_revisit`]).
/// Deriving the ask from the verdicts is what lets an already-parked revisit survive an ask
/// that answered cleanly in between, and what makes a revisit firing a hair inside its window
/// re-arm instead of vanishing; an ask armed while this pass ran is kept.
fn reclaim_answered() {
    let revisit = next_reclaim_revisit();
    let mut schedule = reclaim_schedule().lock().unwrap_poison();
    if let Some(at) = revisit {
        schedule.next_at = Some(schedule.next_at.map_or(at, |armed| armed.min(at)));
    }
}

/// Leave the ladder at its floor: the sweep's last ask was answered and nothing of its own is
/// unfinished, so the next failure starts from the first rung again.
fn reset_reclaim_failures() {
    reclaim_schedule().lock().unwrap_poison().failures = 0;
}

/// An ask that could not be run or was left unfinished: charge the ladder and space the next
/// ask by it, capped at [`RELEASE_RETRY_CAP`]. The failure count is kept across asks, so a
/// browser side that stays down is not re-shovelled, and the arming keeps an ask already
/// parked sooner.
fn reclaim_unfinished() {
    let failures = {
        let mut schedule = reclaim_schedule().lock().unwrap_poison();
        schedule.failures = schedule.failures.saturating_add(1);
        schedule.failures
    };
    arm_reclaim_at(Instant::now() + release_backoff(failures));
}

/// Namespaces durable state says a run may still be rebuilt under: the roster agent ids of
/// every job that has not been terminalized, minus the slots that already finished
/// ([`crate::jobs::resumable_roster_agent_ids`]). A cut round resumes IN PLACE from its stored
/// roster, so the slots it still has to run come back under the very ids — and the very chrome
/// namespaces — they were working with, which is what protects a run whose end was deferred past
/// the hold; a slot that finished is never re-dispatched, so its namespace is nothing to keep.
/// Nothing else durable does this job: a parallel round's agent ids carry a random suffix, so its
/// roster rows are the only place they exist before the run does, and a stage round's
/// deterministic id ([`crate::jobs::session_pin_id`]) is named by the row its stage job writes for
/// as long as that job can still resume — the row a settled ticket keeps as its session pin is
/// continuity for the SESSION, not a promise the run comes back, and is deliberately not read
/// here.
///
/// Fail-open: a store that is not up yields no protection beyond the in-memory registry and the
/// held records, and a read that fails is a `debug!` — the sweep then runs on what it does know.
async fn durable_resume_namespaces() -> Vec<String> {
    let Some(store) = crate::session::SESSIONS.get() else {
        return Vec::new();
    };
    match crate::jobs::resumable_roster_agent_ids(&store.conn).await {
        Ok(agent_ids) => agent_ids
            .iter()
            .map(|agent_id| crate::tools::chrome::run_session_namespace(agent_id))
            .collect(),
        Err(error) => {
            debug!(
                %error,
                "could not read the resumable runs' agent ids — reclaiming without them"
            );
            Vec::new()
        }
    }
}

/// The session names every record in the queue or in a pass right now still holds — held or
/// not — matched EXACTLY (record names ARE the physical session names,
/// `agent-tab-<12hex>-<tab>`). This is what keeps a second driver off the names a record's
/// own pass retries: a name the record still carries is left to that pass, while one it
/// dropped (a name it settled unclosable) belongs to the sweep again, so its group is still
/// examined there.
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
/// every queued record still inside its hold (a run rebuilt under its own durable id), and
/// whatever `durable` adds — the ids of a not-yet-terminalized job's slots that have not finished,
/// i.e. the ones a round resuming in place would re-dispatch. Both the record's own release and the
/// reclaim sweep pass that set: a record names a run that has ENDED, but a job row that outlives
/// the end still says the round may come back, so its tabs must survive the hold. The live half
/// is re-read before the close ([`close_unprotected`]), at every act it performs
/// ([`crate::tools::chrome_tabs::OwnedAgain`]) and once more for the let-go step. The durable
/// half is bounded by the stale-job purge ([`crate::jobs::purge_stale_jobs`]): a phase job idle
/// for [`crate::jobs::PURGE_CUTOFF_HOURS`] is deleted, a frozen one is purge-immune and keeps its
/// tabs (a paused round comes back), and only an abandoned non-phase job can outlive that — the
/// boot scan re-drives those, which is the safe direction to err in.
///
/// Every member is a namespace and matched as a prefix, so a live run protects every name under
/// it; the names a queue or parked record still holds are NOT this set (the sweep excludes those
/// by exact name, [`claimed_names`]).
struct ProtectedNamespaces {
    namespaces: Vec<String>,
}

impl ProtectedNamespaces {
    /// Every namespace a close must leave alone this pass: the live map, the held records
    /// and `durable`.
    fn snapshot(durable: &[String]) -> Self {
        let now = Instant::now();
        // Each lock is taken and released on its own: `take_releasable` holds the queue
        // lock while it reads the live map, so the two must never be nested in the
        // opposite order.
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

    /// The live map and the held records only — the set a step that must not race a run
    /// re-reads in one go.
    fn live_and_held() -> Self {
        Self::snapshot(&[])
    }

    /// Whether the LIVE map or a HELD record protects `name` right now, under its own
    /// locks and building nothing: this is the question a close asks at each act it
    /// performs ([`crate::tools::chrome_tabs::OwnedAgain`]), so it must see the map as it
    /// is at that moment — seconds after the pass's own snapshot — and cost nothing worth
    /// avoiding.
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

    /// Add the live map and the held records as they are now to this set.
    fn refresh(&mut self) {
        self.namespaces.extend(Self::live_and_held().namespaces);
    }

    /// Whether `name` belongs to a namespace some run may still own. Namespaces are
    /// matched as prefixes (they all end in `-`), never re-derived from the agent id.
    fn contains(&self, name: &str) -> bool {
        self.namespaces
            .iter()
            .any(|prefix| names_a_namespace(prefix, name))
    }
}

/// Whether a session `name` belongs to the namespace `prefix`: matched as a prefix,
/// never re-derived from the agent id, and an empty prefix matches nothing.
fn names_a_namespace(prefix: &str, name: &str) -> bool {
    !prefix.is_empty() && name.starts_with(prefix)
}

/// The `target` every row this module files with the Issues view carries.
const ISSUE_TARGET: &str = "chrome-tabs";
/// The one durable message every leftover report carries — stable, because the Issues
/// view dedupes on the `(message, reason)` pair.
const LEFTOVER_MESSAGE: &str = "agent-run browser tabs could not be closed automatically";
/// The report for a close the browser answered for and did not make (see
/// [`RetryLeg::LeftOpen`]).
const LEFT_OPEN_MESSAGE: &str =
    "agent-run browser tabs were left open: the browser answered for them and did not close them";
/// The report for a host whose browser side has no ownership door at all (see the chrome_tabs
/// module doc's Honest limits): a fact about the host, not a leftover of any run. What is missing
/// is the browser-side listing every read here runs on — so the sweep cannot enumerate agent-run
/// groups, and what a record names is left to chrome-use's own `session stop`. What that route
/// cannot settle is left unnamed rather than written off as one run's, since the host cannot
/// establish which tabs are whose. No run, no session and no tab is ever named, and the message
/// must not claim a close failed (there may have been nothing to close).
const NO_DOOR_MESSAGE: &str = "the browser side here cannot list agent-run tab groups, so leftovers no record names cannot be closed";
/// The report for this module's own scratch group a let-go could not confirm gone: the one
/// visible tab this product can leave in the owner's strip, so it must be surfaced rather than
/// accepted quietly — but the reads that look for the group can themselves fail, so the row says
/// only that the release was not CONFIRMED, never that a group was seen. Not a run's leftover: it
/// is the one name neither queue bound drops, so a helper that will not settle it is retried on
/// the queue's own capped deadline for as long as this process lives, and the product's session
/// sweeps own the `mahbot-chrome-ephemeral-*` family across a restart.
const OWN_SESSION_MESSAGE: &str =
    "the product's own browser-read session could not be confirmed released";
/// Why that report is filed; stable, because the Issues view dedupes on the `(message, reason)`
/// pair and this is a property of the module's own session rather than of any run.
const OWN_SESSION_REASON: &str = "chrome-tabs:own-session";
/// Why a tab left behind is reported as unclosable: either its ledger lost it (the extension was
/// updated, re-installed or restarted since it was created) or it was never this session's, or the
/// browser answered this run's close and left it standing — the record's own conclusion
/// ([`MAX_LEFT_OPEN_ATTEMPTS`] passes). Naming every one keeps the report true whichever it was,
/// and it is only said about an answer that repeated ([`crate::tools::chrome_tabs`]'s own
/// confirmation), never about one read. The record drops these names; the reclaim sweep keeps
/// examining their group at [`STUCK_REVISIT`].
///
/// Every one of these rows states what the product could NOT do and that something of ours keeps
/// after those tabs — the record's next pass, or the sweep's own revisit — never what the browser
/// holds right now: a later pass that does close the tabs makes the rows old, not untrue, and the
/// Issues view is an append-only record of what was answered.
const UNCLOSABLE_DETAIL: &str = "the product could not close these tabs and keeps retrying \
                                 them: the browser extension answered that it does not hold \
                                 them as its own (it lost its ledger — updated, re-installed or \
                                 restarted since they were created — or never created them), or \
                                 the browser answered this run's close attempts and left the \
                                 tabs standing — the removal refused as not its own to make, or \
                                 answered while they still stood";
/// The one detail a left-open report carries: the browser's answer that left the tabs standing,
/// as the past fact it is (see [`UNCLOSABLE_DETAIL`] for why). Both shapes the leg covers are
/// named, because either can be the answer the record's own count concluded on.
const LEFT_OPEN_DETAIL: &str = "the browser extension answered this run's close and the tabs \
                                were left standing: it refused the removal as not its own to \
                                make, or answered it while the tabs still stood";
/// The reason a host whose browser side has no ownership door is reported under: it is a
/// property of the host, not of any one run, so it carries no namespace.
const NO_OWNERSHIP_DOOR_REASON: &str = "chrome-tabs:no-ownership-door";
/// The report for a host whose browser answers this route cannot decide from (see
/// [`chrome_tabs::saw_unreadable_answer`]): unlike a missing door, the door is there and an answer
/// came back, but nothing in it settles a name — a field a read parser refuses to guess at,
/// because a dropped tab would read as one that is gone, bytes with no envelope at all, or an
/// envelope carrying neither a success verdict nor a reason for failing. The row must claim
/// neither a close that failed nor a host no route can ever close, and must not say what became
/// of the names the answer covered — another read settles what it can — because it is a fact about
/// the host rather than about one name, and like [`NO_DOOR_MESSAGE`] it names no run and no
/// session.
const UNREADABLE_MESSAGE: &str = "the browser's answers gave this cleanup no decision it could act on, so no name was concluded from them";
/// Why that report is filed; stable, because the Issues view dedupes on the `(message, reason)`
/// pair, and the fact is about the host rather than about any one run or name.
const UNREADABLE_REASON: &str = "chrome-tabs:unreadable-answer";
/// Cap on the joined `session` field of a report, so one wide leftover group cannot bloat
/// the row.
const SESSION_NAMES_MAX_CHARS: usize = 300;
/// Reasons already reported in THIS process, so a leftover the sweep keeps re-reading does
/// not cost a store read on every pass. The durable dedupe (`record_issue_once`) stays the
/// authority — including across restarts.
static REPORTED_REASONS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

fn reported_reasons() -> &'static Mutex<HashSet<String>> {
    REPORTED_REASONS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Which fact one report carries — the two are told apart in the Issues view because the first
/// ends its record's retries while the second is retried on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IssueKind {
    /// The record's routes are done with these tabs: reported with the run, while the reclaim
    /// sweep keeps examining their group.
    Unclosable,
    /// The browser answered and left the tabs: the record keeps retrying them.
    LeftOpen,
}

impl IssueKind {
    /// The message the Issues view shows for this kind.
    fn message(self) -> &'static str {
        match self {
            Self::Unclosable => LEFTOVER_MESSAGE,
            Self::LeftOpen => LEFT_OPEN_MESSAGE,
        }
    }

    /// The reason prefix this kind's rows are deduped on. One reason per namespace, so a
    /// repeated pass over the same run is a duplicate, not news.
    fn reason_prefix(self) -> &'static str {
        match self {
            Self::Unclosable => "chrome-tabs:",
            Self::LeftOpen => "chrome-tabs-left-open:",
        }
    }

    /// What the report says about the sessions it carries: one sentence per kind, because
    /// every name of a group ran into the same fact.
    fn detail(self) -> &'static str {
        match self {
            Self::Unclosable => UNCLOSABLE_DETAIL,
            Self::LeftOpen => LEFT_OPEN_DETAIL,
        }
    }
}

/// One report the Issues view is owed: the run namespace it belongs to (which the rows are
/// deduped on), the run that produced it, and one leaked session name.
struct Issue {
    kind: IssueKind,
    namespace: String,
    run: String,
    name: String,
}

/// One namespace's issues, the shape the Issues view is written in: one row per
/// `(kind, namespace)`, carrying every name that namespace ran into.
struct IssueGroup {
    kind: IssueKind,
    namespace: String,
    run: String,
    /// The session names, in first-seen order.
    names: Vec<String>,
}

/// Report every fact the Issues view is owed, one row per `(kind, namespace)` and at most
/// once per `(message, reason)` — across this process and, by
/// [`crate::logs::record_issue_once`], across restarts.
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

/// Tell the Issues view one fact, once: the in-process set is what a sweep that keeps
/// re-reading the same leftover is spared, and it is only marked after the write landed —
/// a write that failed must not suppress the report for the rest of the process. Every row
/// goes under [`ISSUE_TARGET`].
///
/// The write is fail-open by design ([`crate::logs::record_issue_once`] reports the failure
/// and writes nothing): a fact lost to a down store is not lost for good — whatever produced
/// it is still true of the browser, so the pass that reads it again reports it again — and
/// the process never stops closing tabs because its log store is broken.
async fn report_once(message: &str, reason: &str, fields: serde_json::Value) {
    if reported_reasons().lock().unwrap_poison().contains(reason) {
        return;
    }
    // The seam captures the same `(message, reason, fields)` the real writer takes, and a
    // test with no seam installed goes through the real writer.
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

/// Remember that `reason` has been reported: the next pass over the same leftover is then
/// spared the store read.
fn mark_reported(reason: &str) {
    reported_reasons()
        .lock()
        .unwrap_poison()
        .insert(reason.to_string());
}

/// Test seam: when set, issue reports land here instead of the Issues store, so a
/// test can assert a leftover report without a logs store.
#[cfg(test)]
type CapturedIssues = std::sync::Arc<Mutex<Vec<(String, String, serde_json::Value)>>>;

/// One pass's whole budget: the two phases of browser reads ([`READ_PHASE_RESERVE`] each — the
/// planning reads that decide what the close is about, and the confirmation's own deciding
/// reads) and one `session stop` ([`crate::chrome::SESSION_STOP_TIMEOUT`]) in between. The
/// removal call and its verification reads have no reserve of their own — they spend what the
/// planning share left over — so on a slow pass they are what pushes the confirmation's stop
/// out: the names that stop would have covered then stay retries, settled by no read of that
/// pass, and a later pass whose budget holds a whole stop concludes them (see `chrome_tabs`'
/// own reserve).
///
/// [`READ_PHASE_RESERVE`]: crate::tools::chrome_tabs::READ_PHASE_RESERVE
const ATTEMPT_TIMEOUT: Duration = crate::tools::chrome_tabs::READ_PHASE_RESERVE
    .saturating_mul(2)
    .saturating_add(crate::chrome::SESSION_STOP_TIMEOUT);

/// Every knob of the release path — one accessor so the test seam swaps the whole set
/// at once. A release build compiles no seam state (the `cli` and `issues` stubs below
/// are `#[cfg(test)]`) and always reads the constants above plus the storage root; the
/// real binary is resolved on demand by [`release_cli`], so a `Drop`-driven enqueue, a
/// persist or a restore never probes for it.
#[derive(Clone, Debug)]
struct ReleaseSettings {
    /// One pass's whole budget ([`ATTEMPT_TIMEOUT`]).
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

/// The `chrome-use` binary a release pass spawns (`None` = not available right now),
/// resolved once per pass from the [`cli_path`] probe behind it (a cached lock plus an
/// executable check, not a filesystem search). An installed test seam decides on its own —
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

/// Test-only: empty everything the release path holds in memory — the queue, the
/// passes' parking, the sweep's schedule and verdicts, the evicted runs it remembers,
/// the reasons already reported and the sessions waiting to be let go.
#[cfg(test)]
pub(crate) fn clear_pending_releases() {
    // Parked first, the order [`ReleasePass::take`] and [`persist_pending_releases`]
    // acquire the two in, so no lock cycle can form against a live pass.
    parked_releases().lock().unwrap_poison().clear();
    pending_releases().lock().unwrap_poison().clear();
    reported_reasons().lock().unwrap_poison().clear();
    *reclaim_schedule().lock().unwrap_poison() = ReclaimSchedule::default();
    stuck_namespaces().lock().unwrap_poison().clear();
    pending_forget().lock().unwrap_poison().clear();
    *forget_retry().lock().unwrap_poison() = None;
    evicted_runs().lock().unwrap_poison().clear();
    // The chrome_tabs process latches this module's passes set, so a test that met an
    // unreadable answer cannot make the next one report the host row for it.
    crate::tools::chrome_tabs::reset_unreadable_answer();
}

/// Test-only: the sessions waiting for the next pass to let them go, with the passes that
/// have failed to let each one go.
#[cfg(test)]
fn pending_forget_snapshot() -> Vec<(String, u32)> {
    pending_forget().lock().unwrap_poison().clone()
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
    use crate::tools::chrome::{AGENT_TAB_PREFIX, ChromeRunSessions};
    use crate::tools::chrome_tabs::OWN_SESSION;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // Ended-run session releases
    // -----------------------------------------------------------------------

    /// Re-arm the attempt bound of the settings a [`ReleaseGuard`] installed — for
    /// a test whose short bound has served its purpose and whose later, successful
    /// pass must not race a cold or loaded stub spawn. The shipped budget
    /// ([`ATTEMPT_TIMEOUT`]) is the value that can reach every step of a pass, the
    /// confirmation's stop included.
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

    /// The `left_open` count of each queued record — the browser's repeated answer that a
    /// close left the tabs standing (see [`MAX_LEFT_OPEN_ATTEMPTS`]), which
    /// [`pending_names_and_attempts`] leaves out.
    fn pending_left_open() -> Vec<u32> {
        pending_releases()
            .lock()
            .unwrap_poison()
            .iter()
            .map(|entry| entry.left_open)
            .collect()
    }

    /// The browser state the stub answers from, one line per tab and per group:
    /// `tab <id> <groupId> <owned>` and `group <id> <title>`. `owned` decides whether the
    /// extension holds that tab (a closeable one) or not (an unclosable one).
    fn write_browser_state(path: &Path, names: &[String], owned: bool) {
        let each: Vec<(&str, bool)> = names.iter().map(|name| (name.as_str(), owned)).collect();
        write_browser_state_each(path, &each);
    }

    /// As [`write_browser_state`], one ownership flag per group: the shape a record's close
    /// sees when one of its names is a leftover no route can close while another is closeable.
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

    /// The stub `chrome-use` itself: it answers the extension protocol from `state`,
    /// records every invocation's argv to `log`, and takes its behaviour from `mode` —
    /// `refuse` (a structured error for every call, none of which ever carries the
    /// extension's own refusal sentence), `remove-sentence` (every read answers normally,
    /// and the `tabs.remove` leg fails with the extension's own refusal sentence, relayed
    /// exactly as the extension sends it), `remove-transient` (the same, with a `tabs.remove`
    /// leg whose failure never reached the browser), `silent` (never answers), `keep`
    /// (answers, and leaves its tabs open anyway), `amnesia-once` (one empty ledger read),
    /// `garbled-answer` (one read answers in a shape no parser can use), `stop-refused` (every
    /// read answers normally and `session stop` reports a tab it could not close, the shape a
    /// session whose browser the extension no longer reaches has), `no-extension` (a host
    /// with no extension in the driving profile), `disabled-extension` (one installed but
    /// disabled), `ledger-unreadable` (a CRAFTED shape, impossible on a real host: `extension
    /// status` reads as no extension while `tabGroups.query` still answers from the state file
    /// and both ownership reads fail — it exists to reach the door-less fallback with live group
    /// names in play, and its `session stop` refuses exactly one name, the one carrying
    /// `refuse`). An exit code alone is
    /// deliberately NOT a success mode — the close gates on the browser's own answer.
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

    /// A stub `chrome-use`, the [`ReleaseSettings`] pointing at it, a private record
    /// file and a browser-state file. Restores the previous settings and drains the
    /// queue on drop, including during a panic.
    struct ReleaseGuard {
        state: PathBuf,
        mode: PathBuf,
        log: PathBuf,
        issues: CapturedIssues,
        previous: Option<ReleaseSettings>,
        /// Holds the stub binary, the record file and the state file for the guard's
        /// lifetime.
        dir: tempfile::TempDir,
    }

    impl ReleaseGuard {
        /// Install the stub settings with no boot grace (every restored record is
        /// eligible at once) and warm the spawn path once.
        async fn install(attempt_timeout: Duration) -> Self {
            Self::install_with(attempt_timeout, Duration::ZERO).await
        }

        /// `boot_grace` is the floor a restored record's eligibility gets — a test
        /// injects a short one instead of waiting a real minute. The record file and
        /// state file live in the guard's own temp dir, so the real
        /// `~/.mahbot/chrome-run-releases.json` is never touched.
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
            // Warm-up call; the log is reset so it cannot be mistaken for a real
            // close pass, and the session it read in is let go exactly as a pass would —
            // otherwise the next pass would find a read outstanding that it never made.
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

        /// Write the browser state the stub answers from, one group per name.
        fn write_state(&self, names: &[String], owned: bool) {
            write_browser_state(&self.state, names, owned);
        }

        /// Write the browser state with one ownership flag per group — the shape a close sees
        /// when one of a record's names is unclosable and another is closeable.
        fn write_state_each(&self, owned: &[(&str, bool)]) {
            write_browser_state_each(&self.state, owned);
        }

        /// Put the release path in the window a managed self-update swap creates:
        /// chrome-use resolves to nothing.
        #[expect(
            clippy::unused_self,
            reason = "called as a guard method so every seam installs the same way"
        )]
        fn set_binary_absent(&self) {
            let mut settings = release_settings();
            settings.cli = None;
            swap_release_settings(settings);
        }

        /// Arm the sweep as an event would — the schedule no longer arms a boot ask. The
        /// shipped product arms one from an event ([`schedule_reclaim_now`]); a test does it
        /// to ask without one.
        #[expect(
            clippy::unused_self,
            reason = "called as a guard method so every seam installs the same way"
        )]
        fn arm_sweep(&self) {
            schedule_reclaim_now();
        }

        /// One line per stub invocation (the argv the release path built).
        fn log_lines(&self) -> Vec<String> {
            fs::read_to_string(&self.log)
                .unwrap_or_default()
                .lines()
                .map(str::to_string)
                .collect()
        }

        /// Whether any invocation's argv contained `needle`.
        fn invoked(&self, needle: &str) -> bool {
            self.log_lines().iter().any(|line| line.contains(needle))
        }

        /// The tab ids the release path asked `tabs.remove` for, in the order asked — what
        /// the close demanded of the browser, not what the browser answered.
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

        /// Whether the only things the release path asked the browser were reads: no record was
        /// attempted, nothing was closed, and no record's session was stopped. The sweep's read
        /// borrows the work's own session where one exists, so a line naming a session other than
        /// [`OWN_SESSION`] is not itself a close — a `tabs.remove` is, wherever it runs, and so is
        /// a `session stop` of any name but the door's own.
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

        /// The guard's own durable record file (never the real `~/.mahbot` one).
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
        queue_run_session_release_after(&sessions, hold, agent_id);
    }

    /// Physical names as a run records them: the tracker's namespace plus each logical
    /// name — the shape the durable record file carries. `for_run` derives the namespace
    /// from the agent id alone, so this is exactly what [`queue_names_with_hold`]
    /// recorded for that id.
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

    /// An ended run's record is dropped once the browser's own answer says every one
    /// of its names is gone, and the close went through the browser's extension.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_verified_release_empties_the_record() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        let names = run_session_names("run-verified", &["agent-tab-v-default", "agent-tab-v-docs"]);
        guard.write_state(&names, true);
        queue_names("run-verified", &["agent-tab-v-default", "agent-tab-v-docs"]);
        assert_eq!(pending_releases_snapshot().len(), 1);

        release_due().await;

        assert!(
            pending_releases_snapshot().is_empty(),
            "both names settled gone"
        );
        assert!(
            guard.invoked("extension call tabs.remove"),
            "the close went through the browser's own extension"
        );
        assert_eq!(
            guard.removed_tab_ids(),
            vec![100, 101],
            "exactly the run's own tabs were asked for: {:?}",
            guard.log_lines()
        );
        assert!(
            guard.invoked(&format!(
                "session stop --force --json --session {}",
                names[0]
            )),
            "the record's own session the reads ran in is let go once its group is settled, so \
             it cannot outlive the pass: {:?}",
            guard.log_lines()
        );
        assert!(
            !guard.only_swept(),
            "a pass that asked tabs.remove is not a sweep, whatever session it ran in: {:?}",
            guard.log_lines()
        );
    }

    /// A record whose groups the browser no longer holds is dropped — no `tabs.remove`
    /// is even needed for it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_record_the_browser_reports_gone_is_dropped() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-gone", &["agent-tab-g-default"]);

        release_due().await;

        assert!(
            pending_releases_snapshot().is_empty(),
            "the browser holds no group by that name, so its tabs are gone"
        );
        assert!(!guard.invoked("extension call tabs.remove"));
    }

    /// The `refuse` stub answers a structured error to EVERY call, so the browser never
    /// settles the record: it stays queued past five attempts and is never dropped from the
    /// durable file — there is no attempt cap on a record the browser did not answer for.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_record_the_browser_never_settles_stays_queued_past_five_attempts() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        let names = run_session_names("run-refuse", &["agent-tab-r-stuck"]);
        queue_names("run-refuse", &["agent-tab-r-stuck"]);
        guard.set_mode("refuse");

        for _ in 0..6 {
            release_due().await;
            assert_eq!(
                pending_names_and_attempts().len(),
                1,
                "the record stays queued across attempts"
            );
            assert_eq!(
                pending_names_and_attempts()[0].0,
                names,
                "and keeps its names"
            );
        }

        let (queued, attempts) = &pending_names_and_attempts()[0];
        assert_eq!(queued, &names, "the names are intact");
        assert!(
            *attempts > 5,
            "past five attempts, still queued: {attempts}"
        );
        assert_eq!(
            record_file_names(&guard.store_path()),
            {
                let mut sorted = names.clone();
                sorted.sort();
                sorted
            },
            "the record file still describes it — nothing drops an unanswered record from the queue"
        );
    }

    /// A record the browser reports as not the extension's to remove is reported on the
    /// Issues view once and dropped: the queue never re-asks those names, and the
    /// reclaim pass keeps re-reading the browser for such a group anyway.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_record_the_browser_reports_as_not_ours_is_reported_once_and_dropped() {
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
            "the record's route is done with those tabs, so they are not retried"
        );
        let issues = guard.issues();
        assert_eq!(issues.len(), 1, "the leftover is reported once");
        assert_eq!(issues[0].0, LEFTOVER_MESSAGE);
        assert_eq!(issues[0].1, format!("chrome-tabs:{namespace}"));
        assert_eq!(issues[0].2["run"], "run-unclosable");
        assert_eq!(issues[0].2["sessions"], 1);
        assert!(
            guard.invoked(&format!("session stop --json --session {session}")),
            "the session's own route is asked before any tab is called unclosable: {:?}",
            guard.log_lines()
        );

        // A second pass has nothing left to report.
        release_due().await;
        assert_eq!(guard.issues().len(), 1, "and never again");
    }

    /// An empty ledger is one browser answer, not a verdict on its own: the extension's
    /// storage read is not reported to us, so a swallowed one hands back exactly what a
    /// lost ledger does. The verdict is only reached after the session's own route was
    /// tried and the browser was read again — here that second read finds the tab, so the
    /// record survives and the next pass closes it normally.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_transient_empty_ledger_does_not_write_a_run_off() {
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
            "an answer the second read did not repeat is not a verdict"
        );
        assert!(
            guard.issues().is_empty(),
            "and nothing is announced about those tabs"
        );

        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "the next pass reads the real ledger and closes them"
        );
        assert!(guard.invoked("extension call tabs.remove"));
    }

    /// A browser that answers and leaves its own tabs open is a contradiction between two of
    /// its own answers, not a verdict from one pass: the record keeps retrying, and the
    /// answered-and-left-open name is told durably — once, with the run it came from — rather
    /// than ageing out as a `debug!` line nobody ever sees.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_close_the_browser_leaves_open_is_reported_once_and_still_retried() {
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
            "one left-open answer proves nothing, so the record stays queued"
        );
        assert_eq!(
            record_file_names(&guard.store_path()),
            {
                let mut sorted = names.clone();
                sorted.sort();
                sorted
            },
            "and stays in the durable file"
        );
        let issues = guard.issues();
        assert_eq!(issues.len(), 1, "the left-open answer is reported once");
        assert_eq!(issues[0].0, LEFT_OPEN_MESSAGE);
        assert_eq!(issues[0].1, format!("chrome-tabs-left-open:{namespace}"));
        assert_eq!(issues[0].2["run"], "run-kept");

        let before = pending_names_and_attempts()[0].1;
        release_due().await;
        assert_eq!(
            guard.issues().len(),
            1,
            "and never again for the same run's namespace"
        );
        assert!(
            pending_names_and_attempts()[0].1 > before,
            "the retry ladder still moves"
        );
    }

    /// The extension's own ownership-refusal sentence, relayed under `tabs.remove`, is the only
    /// failing leg that counts as the browser answering about the tabs — so it reaches the very
    /// conclusion a kept tab does: after [`MAX_LEFT_OPEN_ATTEMPTS`] passes the record stops
    /// retrying, and the leftover is reported with the run it came from (the reclaim sweep
    /// keeps examining its group).
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_remove_the_extension_refuses_concludes_after_the_cap() {
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
                "a relayed left-open answer short of the cap keeps the record queued"
            );
            assert_eq!(
                pending_left_open(),
                vec![pass],
                "each pass is one browser answer, counted"
            );
        }
        release_due().await;

        assert!(
            pending_names_and_attempts().is_empty(),
            "the cap concludes the close route: the record stops retrying"
        );
        let issues = guard.issues();
        let conclusion: Vec<_> = issues
            .iter()
            .filter(|issue| issue.1 == format!("chrome-tabs:{namespace}"))
            .collect();
        assert_eq!(
            conclusion.len(),
            1,
            "the leftover is reported once as unclosable: {issues:?}"
        );
        assert_eq!(conclusion[0].0, LEFTOVER_MESSAGE);
        assert_eq!(
            conclusion[0].2["run"], "run-refused-sentence",
            "and the conclusion names the run it came from"
        );
        assert!(
            issues
                .iter()
                .any(|issue| issue.1 == format!("chrome-tabs-left-open:{namespace}")),
            "the browser's own answer was told early, before the cap: {issues:?}"
        );
    }

    /// A `tabs.remove` leg that failed before the browser answered — here a relay that is not
    /// connected — is NOT the browser answering for the tabs: it never increments `left_open`, so
    /// no pass count ever concludes the record, and neither a left-open nor a leftover row is
    /// filed for it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_transient_remove_failure_is_never_counted_as_an_answer() {
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
            "the transient leg leaves the record queued, attempted once per pass"
        );
        assert_eq!(
            pending_left_open(),
            vec![0],
            "a leg that never reached the browser is never counted as an answer"
        );
        assert!(
            guard.invoked("extension call tabs.remove"),
            "the close did reach the removal leg — the transient failure is that leg's own \
             answer: {:?}",
            guard.log_lines()
        );
        assert!(
            guard.issues().iter().all(|(_, reason, _)| {
                reason != &format!("chrome-tabs-left-open:{namespace}")
                    && reason != &format!("chrome-tabs:{namespace}")
            }),
            "a transient failure files neither the left-open nor the leftover row: {:?}",
            guard.issues()
        );
    }

    /// A host with no browser extension at all has no ownership door, so nothing here can
    /// enumerate the browser's groups and a leftover no record names is out of reach: the
    /// Issues view is told once, with what to do about it, instead of a `debug!` line that
    /// ages out of the log store. The record path is told too — with the absence that decides
    /// it (`chromeExtension`, not the host manifest the product writes itself), and it keeps
    /// retrying the helper's own stop rather than writing the tabs off.
    /// Nothing is filed for the product's own scratch group: the pass's reads run in the
    /// record's own session, not the product's own, so that group is never in play here.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_host_without_the_ownership_door_is_reported_once() {
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
            "the absent door is reported once: {:?}",
            guard.issues()
        );
        let issues = host_rows();
        assert_eq!(issues[0].0, NO_DOOR_MESSAGE);
        assert!(
            issues[0].2["detail"]
                .as_str()
                .is_some_and(|d| !d.is_empty())
        );
        assert_eq!(
            pending_names_and_attempts().len(),
            1,
            "a door-less host never writes the run's tabs off: the record keeps trying"
        );
        assert!(
            guard.invoked(&format!("session stop --json --session {session}")),
            "the only route such a host has is the helper's own stop"
        );

        // The pass's reads run in the record's own session — the product's own scratch session
        // is not opened at all — so nothing is filed for it.
        assert!(
            guard
                .issues()
                .iter()
                .all(|issue| issue.1 != OWN_SESSION_REASON),
            "no scratch-group row: the reads no longer run in the product's own session: {:?}",
            guard.issues()
        );

        // The sweep keeps being woken and keeps failing to read; the report stays one.
        guard.arm_sweep();
        release_due().await;
        assert_eq!(host_rows().len(), 1, "and never again");
        assert!(
            guard
                .issues()
                .iter()
                .all(|issue| issue.1 != OWN_SESSION_REASON),
            "and no scratch-group row either: the reads run in the record's session"
        );
    }

    /// The one visible tab this product can leave in the owner's strip — its own scratch group,
    /// when a let-go could not confirm it gone — is surfaced once rather than accepted quietly.
    /// The reads that look for the group can themselves fail, so the row claims only that the
    /// release is not CONFIRMED, and a host where no call of ours was ever answered gets none.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_scratch_group_the_let_go_could_not_confirm_is_surfaced_once() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        // A scratch group holding a tab the extension does not hold as its own: this route may
        // not remove it, so the let-go cannot confirm the group gone — and the session's own
        // stop is answered with chrome-use's "ownership was preserved" refusal, the shape that
        // leaves the group standing and the let-go unfinished.
        guard.set_mode("stop-refused");
        guard.write_state(&[OWN_SESSION.to_string()], false);
        // No record queued: the pass has nothing to borrow, so its reads run in the product's own
        // session — the shape a parked leftover's revisit takes.
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
            "one row for the product's own session: {:?}",
            guard.issues()
        );
        assert_eq!(rows[0].0, OWN_SESSION_MESSAGE);
        let detail = rows[0].2["detail"].as_str().unwrap_or_default();
        assert!(
            detail.contains("the release is not confirmed"),
            "and it says only that the release was not confirmed: {detail}"
        );
        assert_eq!(
            pending_forget_snapshot(),
            vec![(OWN_SESSION.to_string(), 2)],
            "and the session stays queued with one charged attempt per pass that asked its \
             stop — the one route left that closes its group was answered with a refusal"
        );
    }

    /// The fallback's mapping over an index slice, with more than one affected name and two
    /// DIFFERENT outcomes to place: the name whose stop the helper refuses must stay queued, the
    /// other must settle `Gone`, and the name with no live group must settle on the group read and
    /// never be asked for a stop. A positional mix-up (results written by slot instead of by
    /// index) moves a result onto another name's place, which is what the queued set catches —
    /// and the affected names are not the leading ones precisely so it can.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_fallback_maps_each_live_name_back_to_its_own_slot() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        guard.set_mode("ledger-unreadable");
        let names = run_session_names(
            "run-ledger-unreadable",
            &["agent-tab-missing", "agent-tab-refuse", "agent-tab-live"],
        );
        // The first name has no live group (settled by the one group read); the other two do,
        // and the ownership reads that would map their ids back to names both fail.
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

    /// A host whose browser answers settle nothing this route can act on is a fact about the HOST,
    /// filed once: every automatic close there is a retry with no conclusion, so it must not stay
    /// a `debug!` that ages out (see [`UNREADABLE_MESSAGE`]). The record itself keeps trying.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_host_whose_answers_are_unreadable_is_reported_once() {
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
            "one row for the host, whatever the passes: {:?}",
            guard.issues()
        );
        assert_eq!(rows[0].0, UNREADABLE_MESSAGE);
        let detail = rows[0].2["detail"].as_str().unwrap_or_default();
        assert!(
            detail.contains("settled nothing") && detail.contains("no parser here will guess at"),
            "and it says what settled nothing, and why it is not guessed at: {detail}"
        );
        assert_eq!(
            pending_names_and_attempts().len(),
            1,
            "while the record keeps the name queued: nothing was settled"
        );
    }

    /// The same host-level fact, for the host shape that is easiest to mistake for a working
    /// one: the extension IS installed (and new enough for the door), but Chrome reports it
    /// disabled, so nothing answers an `extension call`. That still establishes no ownership
    /// door — the row is filed once, and the fallback close is still attempted.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_host_with_a_disabled_extension_is_reported_once() {
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
            "a disabled extension's door is missing too, and reported once: {:?}",
            guard.issues()
        );
        assert_eq!(issues[0].0, NO_DOOR_MESSAGE);
        assert!(
            issues[0].2["detail"]
                .as_str()
                .is_some_and(|detail| detail.contains("disabled")),
            "the row names the disabled case rather than only a missing extension: {issues:?}"
        );
        assert_eq!(
            pending_names_and_attempts().len(),
            1,
            "and the record still keeps trying rather than being written off"
        );
        assert!(
            guard.invoked(&format!("session stop --json --session {session}")),
            "the fallback close is attempted on this host too: {:?}",
            guard.log_lines()
        );
    }

    /// The sweep claims a live agent group nothing accounts for, and leaves alone every
    /// group a run may still come back to: one whose namespace is live right now, a held
    /// record's, an unheld record's own names, and one the roster of a job that has not
    /// been terminalized still names — a cut round resumes in place under the very same
    /// ids, and reads its tabs back by name. A slot of that same roster that already
    /// finished is claimed with the orphan: a resume reconstructs it instead of
    /// re-dispatching it, so nothing will work under its id again.
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

        // Undo the seeded protection BEFORE asserting: the shared test store must not keep
        // naming a run that may come back, or every later test in this process reads a
        // resume set this one left behind — including when an assertion below fails.
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
                "{spared} belongs to a run that may come back: {:?}",
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

    /// A group the sweep claims but the browser will not close is reported with the group's
    /// own namespace, even though no record names it: the record path reports its left-open
    /// names, and a reclaimed one must not be the only kind that ages out.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_reclaimed_group_the_browser_leaves_open_is_reported() {
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
            "the left-open answer is reported: {issues:?}"
        );
        assert_eq!(issues[0].0, LEFT_OPEN_MESSAGE);
        assert_eq!(
            issues[0].1,
            format!("chrome-tabs-left-open:{AGENT_TAB_PREFIX}0123456789ab-")
        );
        assert_eq!(
            issues[0].2["run"], "",
            "no record was evicted for a group nothing recorded, so no run can be named"
        );

        // The verdict is remembered: the next sweep skips the group it already answered
        // for, and the report stays one.
        guard.arm_sweep();
        release_due().await;
        assert_eq!(guard.issues().len(), 1, "and never again");
    }

    /// A leftover the browser has already answered twice can never be closed is not
    /// interrogated in full on every sweep — the verdict is remembered, so a stuck group
    /// cannot cost a `session stop` per name on each pass — while the report stays one.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_stuck_leftover_is_not_interrogated_again_every_sweep() {
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
        assert_eq!(issues.len(), 1, "the leftover is reported: {issues:?}");
        assert_eq!(issues[0].0, LEFTOVER_MESSAGE);
        assert_eq!(
            issues[0].1,
            format!("chrome-tabs:{AGENT_TAB_PREFIX}0123456789ab-")
        );

        guard.clear_log();
        guard.arm_sweep();
        release_due().await;
        assert!(
            !guard.invoked(&format!("session stop --json --session {orphan}")),
            "a remembered verdict is not re-confirmed every sweep: {:?}",
            guard.log_lines()
        );
        assert_eq!(guard.issues().len(), 1, "and never reported again either");
    }

    /// A verdict is not forever: once its [`STUCK_REVISIT`] window has passed the prune drops
    /// it, and the next armed sweep examines the group in full again — the stop route
    /// included — instead of skipping it for the rest of the process.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn an_expired_stuck_verdict_is_dropped_and_its_group_examined_again() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        let orphan = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan");
        guard.write_state(std::slice::from_ref(&orphan), false);
        guard.arm_sweep();

        release_due().await;
        assert_eq!(guard.issues().len(), 1, "the leftover is reported once");

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
            "the group is examined again, not skipped by an expired verdict: {:?}",
            guard.log_lines()
        );
    }

    /// The session a pass with no name to borrow reads in ([`OWN_SESSION`]) is let go by the pass
    /// that made the reads, through the same door a run's tabs are closed by — so a re-minted
    /// relay endpoint cannot orphan its scratch group — and a pass that read nothing leaves it
    /// alone. A pass with real work borrows a record's own session instead of opening this one.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_pass_lets_go_of_the_session_its_reads_ran_in() {
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
            "the scratch group is closed through the extension's own door: {:?}",
            guard.log_lines()
        );
        assert!(
            guard.invoked(&format!(
                "session stop --force --json --session {OWN_SESSION}"
            )),
            "the group the ledger door confirmed gone is followed by the record-dropping stop: \
             {:?}",
            guard.log_lines()
        );

        // Nothing outstanding: a pass that asked the browser nothing stops nothing.
        guard.clear_log();
        chrome_tabs::release_own_session(&cli, deadline).await;
        assert!(
            guard.log_lines().is_empty(),
            "an idle pass spawns nothing: {:?}",
            guard.log_lines()
        );
    }

    /// A scratch group holding a tab the extension does not hold as its own is not closed from
    /// this module's ledger route — asking for an id the extension does not hold is refused
    /// whole — so the let-go hands the name to the queue, and the pass's let-go step is the ONE
    /// place a graceful `session stop` is asked for: it is the verb that closes the tabs a
    /// session created. One stop per pass, never the `--force` form (which drops the record and
    /// leaves the group exactly where it is), and the failure it hears is charged.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_scratch_group_the_ledger_cannot_close_goes_to_its_session_stop() {
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
            "a tab the extension does not hold is never asked for: {:?}",
            guard.log_lines()
        );
        assert!(
            !guard.invoked("session stop"),
            "and the unconfirmed let-go does not stop the session itself — the pass's let-go \
             step is the one stop site: {:?}",
            guard.log_lines()
        );
        assert!(
            pending_forget_snapshot() == vec![(OWN_SESSION.to_string(), 0)],
            "it hands the session to the driver's retry path instead: {:?}",
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
            "one stop for the pass, with the verb that closes the group: {:?}",
            guard.log_lines()
        );
        assert!(
            guard.invoked(&format!("session stop --json --session {OWN_SESSION}")),
            "and it is the graceful form: {:?}",
            guard.log_lines()
        );
        assert!(
            !guard.invoked(&format!(
                "session stop --force --json --session {OWN_SESSION}"
            )),
            "never the form that would drop the record and leave the group standing: {:?}",
            guard.log_lines()
        );
        assert!(
            pending_forget_snapshot() == vec![(OWN_SESSION.to_string(), 1)],
            "a whole-bound stop that could not close it charges one attempt and keeps the name \
             queued: {:?}",
            guard.log_lines()
        );
    }

    /// A scratch group the ledger route cannot close, whose own stop then lands, ends the whole
    /// debt: that stop settled the session — its record and the group — so the reads it covered
    /// are no longer owed. Without the marker being advanced there, the next pass would start the
    /// session again and mint a fresh blank group for no new reason, which is the one page this
    /// whole choice of session exists to keep out of the owner's browser.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_settled_scratch_session_is_not_started_again_by_the_next_pass() {
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
            "the unconfirmed let-go is queued"
        );

        guard.clear_log();
        forget_settled(&cli, &[], deadline).await;
        assert!(
            guard.invoked(&format!("session stop --json --session {OWN_SESSION}")),
            "the let-go step asks the graceful stop: {:?}",
            guard.log_lines()
        );
        assert!(
            pending_forget_snapshot().is_empty(),
            "which settles the session and drops the name"
        );

        guard.clear_log();
        chrome_tabs::release_own_session(&cli, deadline).await;
        assert!(
            guard.log_lines().is_empty(),
            "and the next pass owes nothing for those reads: {:?}",
            guard.log_lines()
        );
    }

    /// Two passes over the let-go queue can overlap — the driver and the shutdown flush — and
    /// neither may lose the other's names: the queue is read and written back, never taken out.
    /// (A pass whose future is dropped mid-stop is the same shape from the queue's side: it
    /// writes nothing back, and its names were never removed from it.)
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn overlapping_let_go_passes_keep_each_others_names() {
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
            "both passes' names survive: {queued:?}"
        );
    }

    /// The sessions waiting to be let go are kept without repeats and bounded: a burst of
    /// settled sessions the helper will not stop must not grow the queue without bound.
    #[test]
    fn the_let_go_queue_is_deduped_and_bounded() {
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
            "a repeat keeps its first place, attempts and all"
        );

        let mut full: Vec<(String, u32)> = (0..MAX_PENDING_FORGET)
            .map(|i| (format!("s{i}"), 0))
            .collect();
        full.push(("newest".to_string(), 0));
        dedupe_and_bound_forgotten(&mut full);
        assert_eq!(full.len(), MAX_PENDING_FORGET, "the queue is bounded");
        assert_eq!(
            full[0],
            ("s1".to_string(), 0),
            "the oldest name is what the bound drops"
        );
        assert_eq!(full.last(), Some(&("newest".to_string(), 0)));

        // The product's own scratch session is the one name the cap never drops, even from the
        // place the cap drops first: what it leaves when its stop fails is a group in the
        // owner's tab strip, not helper bookkeeping.
        let mut with_scratch: Vec<(String, u32)> = (0..MAX_PENDING_FORGET)
            .map(|i| (format!("s{i}"), 0))
            .collect();
        with_scratch.insert(0, (OWN_SESSION.to_string(), 0));
        dedupe_and_bound_forgotten(&mut with_scratch);
        assert_eq!(with_scratch[0], (OWN_SESSION.to_string(), 0));
        assert!(
            !with_scratch.iter().any(|(name, _)| name == "s0"),
            "so the oldest bookkeeping name is what the cap drops instead"
        );
    }

    /// A settled session the pass had no budget to let go is not lost: it waits — with the rung
    /// it had, since the pass never asked the helper anything — and the next pass tries it again.
    /// The record itself is dropped as soon as the browser says the tabs are gone, so nothing else
    /// would ever come back for that name.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_settled_session_left_over_is_let_go_by_a_later_pass() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        let cli = release_cli().expect("the stub binary");
        let name = run_session_names("run-forget", &["agent-tab-f-default"]).remove(0);

        // A pass whose budget is already spent lets go of nothing and keeps the name — charging
        // it no attempt, because its stop was never asked.
        forget_settled(&cli, std::slice::from_ref(&name), Instant::now()).await;
        assert_eq!(pending_forget_snapshot(), vec![(name.clone(), 0)]);
        assert!(
            !guard.invoked("session stop"),
            "and spawned nothing: {:?}",
            guard.log_lines()
        );
        assert!(
            forget_retry_at().is_some(),
            "while the queue arms its own retry, so a budget-starved pass is not the last word"
        );

        // More starved passes than the bound: a stop the pass never asked is deferred work, not
        // a failure, so the name neither climbs toward `MAX_FORGET_ATTEMPTS` nor is dropped
        // while its group may still be standing.
        for _ in 0..MAX_FORGET_ATTEMPTS {
            forget_settled(&cli, &[], Instant::now()).await;
        }
        assert_eq!(pending_forget_snapshot(), vec![(name.clone(), 0)]);

        // The next pass has budget, so the leftover is let go and nothing waits.
        forget_settled(&cli, &[], Instant::now() + ATTEMPT_TIMEOUT).await;
        assert!(pending_forget_snapshot().is_empty());
        assert!(
            guard.invoked("session stop"),
            "the helper's own recipe is what clears the stale record: {:?}",
            guard.log_lines()
        );
    }

    /// A stop the pass could only cut short is best-effort: the helper is asked (it often
    /// answers well inside the bound) and a name it took leaves the queue, but a name it did
    /// not take keeps the rung it had — a pass that could not hear a whole answer must not walk
    /// a session toward [`MAX_FORGET_ATTEMPTS`] and drop it while its group may stand.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_stop_cut_to_the_pass_is_not_charged_an_attempt() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        guard.set_mode("refuse");
        let cli = release_cli().expect("the stub binary");
        let name = run_session_names("run-cut-stop", &["agent-tab-cs-default"]).remove(0);
        *pending_forget().lock().unwrap_poison() = vec![(name.clone(), 2)];

        // Ten seconds of pass left: a stop, but not a whole one.
        forget_settled(&cli, &[], Instant::now() + Duration::from_secs(10)).await;

        assert!(
            guard.invoked("session stop"),
            "the stop is still attempted: {:?}",
            guard.log_lines()
        );
        assert_eq!(
            pending_forget_snapshot(),
            vec![(name, 2)],
            "and its refused answer charges nothing: the rung is where it was"
        );
    }

    /// The shutdown flush is the let-go queue's last chance before the process goes down, and it
    /// has to reach the queue even when no record is due — the unconfirmed let-go of this
    /// module's own scratch session is queued precisely when its own pass is ending. Its budget is
    /// below one whole stop, so only a helper that answers inside it lets a name go here; one that
    /// does not leaves the name where the process leaves everything the queue holds in memory.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_shutdown_flush_reaches_the_let_go_queue() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        let name = run_session_names("run-flush-forget", &["agent-tab-ff-default"]).remove(0);
        queue_forget(&name);
        assert!(pending_releases_snapshot().is_empty(), "no record is due");

        flush_pending_run_releases(SHUTDOWN_RELEASE_FLUSH_BUDGET).await;

        assert!(
            guard.invoked(&format!("session stop --force --json --session {name}")),
            "the queued session is stopped before the process goes down: {:?}",
            guard.log_lines()
        );
        assert!(
            pending_forget_snapshot().is_empty(),
            "and a helper that answers inside the flush's budget lets the name go"
        );
    }

    /// A let-go that never lands is bounded rather than retried forever: after
    /// [`MAX_FORGET_ATTEMPTS`] passes whose stop the helper did not settle, a run's session name
    /// leaves the queue with one `info!` naming it — a log trace, not an Issues row (chrome-use's
    /// own record under `~/.chrome-use` is all that is left behind by it, never a tab). A pass
    /// that does let go of its name drops it without reaching the bound.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_let_go_queue_gives_up_after_its_attempt_bound() {
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
                "the name waits with one more failed attempt on it"
            );
        }
        forget_settled(&cli, &[], deadline).await;
        assert!(
            pending_forget_snapshot().is_empty(),
            "the bound drops it instead of retrying it for the whole process"
        );
    }

    /// The product's own scratch session is the one name the attempt bound never drops: what it
    /// can leave behind is a group in the owner's tab strip rather than chrome-use's bookkeeping,
    /// so once a stop has failed [`MAX_FORGET_ATTEMPTS`] times it stays in the queue at the
    /// bound's rung and the queue's own deadline keeps retrying it — the alternative would be the
    /// group waiting for the next process start.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_scratch_session_is_retried_past_the_attempt_bound() {
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
            "the scratch session stays queued past the bound"
        );
        assert!(
            forget_retry_at().is_some(),
            "and the queue's own deadline is what retries it, not a restart"
        );
    }

    /// A name the pass could not ask a stop of because a call of ours was still running in its
    /// session is deferred as busy, not failed: the helper is not asked, nothing is charged, and
    /// the queue's own wake is floored at one whole stop — nothing is owed until that call is
    /// done, and the pass holding it covers the name at its own let-go step, so waking at the base
    /// gap would only bring the driver back to defer again.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_let_go_behind_a_call_of_ours_waits_the_stop_bound() {
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
        assert_eq!(outcome.busy, vec![OWN_SESSION.to_string()]);
        assert!(
            outcome.deferred.is_empty() && outcome.tried.is_empty(),
            "and it is neither failed nor a deferral of a bound that did not fit"
        );

        forget_settled(&cli, &[OWN_SESSION.to_string()], deadline).await;
        assert_eq!(
            pending_forget_snapshot(),
            vec![(OWN_SESSION.to_string(), 0)],
            "the name is kept, at the rung it had"
        );
        let wait = forget_retry_at()
            .expect("while the call holds the session the queue arms its own retry")
            .saturating_duration_since(Instant::now());
        assert!(
            BUSY_FORGET_FLOOR.saturating_sub(wait) < Duration::from_secs(1),
            "and the wake waits out one whole stop, not the base gap: {wait:?}"
        );
        drop(call);
    }

    /// A name a run owns again leaves the let-go queue instead of climbing it: stopping that
    /// session is what would end the run, whose own end hands over its own release — so the
    /// helper is never asked, and the name is not charged a failed attempt for it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_session_a_run_owns_again_leaves_the_let_go_queue() {
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
            "the stop would end the run's own session: {:?}",
            guard.log_lines()
        );
        assert!(
            pending_forget_snapshot().is_empty(),
            "and nothing waits for a name the run owns"
        );
        drop(tracker);
    }

    /// A let-go retry left behind by an earlier pass must not keep the driver awake while no
    /// chrome-use binary is resolvable: nothing can run the queue then, so a deadline already in
    /// the past would be read as due by every loop turn, with a pass that can do nothing — a
    /// spin, not a retry. The stored deadline is kept, so a binary that resolves again puts the
    /// queue back on the ladder.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_let_go_retry_cannot_wake_a_driver_with_no_binary() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        // A retry the queue left behind before the binary went away: already due.
        *forget_retry().lock().unwrap_poison() = Some(Instant::now());

        assert!(
            next_release_deadline().is_some(),
            "with a binary to run it, the retry is a real deadline"
        );

        guard.set_binary_absent();
        assert_eq!(
            next_release_deadline(),
            None,
            "with none, nothing reads the queue's past deadline as work"
        );
    }

    /// The sweep's own gate: the ask is not on a timer — a pass woken by an enqueue does not
    /// ask the browser again once the last ask is done with, so a quiet queue cannot cost a
    /// chrome-use child per wake; an event arms the next ask, and only then does the sweep
    /// read the browser again.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn the_reclaim_asks_the_browser_once_per_armed_ask() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        let orphan = format!("{AGENT_TAB_PREFIX}0123456789ab-orphan");
        guard.write_state(&[orphan], true);

        // Arm the ask a run end would; the sweep does not arm one at boot.
        guard.arm_sweep();
        release_due().await;
        assert!(
            guard.invoked("extension call tabs.remove"),
            "the first pass reclaims"
        );
        assert!(
            next_release_deadline().is_none(),
            "a clean ask leaves nothing scheduled — no ask, no queued retry"
        );
        guard.clear_log();

        release_due().await;
        assert!(
            guard.log_lines().is_empty(),
            "a successful ask schedules nothing, so no interval re-asks: {:?}",
            guard.log_lines()
        );

        schedule_reclaim_now();
        release_due().await;
        assert!(
            guard.invoked("extension call tabGroups.query"),
            "an event — a run end queuing names — is what arms the next ask"
        );
    }

    /// A sweep whose read succeeds but whose close leg comes back Silent is unfinished, and the
    /// ladder has to see that: repeated Silent passes climb the capped backoff instead of
    /// re-arming at the first rung every few seconds. The count is reset only where the sweep's
    /// answers are known ([`reclaim_issues`]), so a Silent pass keeps the rung it charged.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn repeated_silent_sweeps_climb_the_ladder() {
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
            "the Silent answer charged the ladder's first rung"
        );
        assert!(
            first > Duration::ZERO,
            "the first ask is spaced by the base, not at the floor: {first:?}"
        );
        assert!(
            second > first + Duration::from_secs(2),
            "the second Silent pass took a further rung rather than re-arming at the floor: \
             {first:?} then {second:?}"
        );

        // A later ask the browser answers cleanly leaves the ladder at its floor again.
        guard.set_mode("");
        reclaim_schedule().lock().unwrap_poison().next_at = Some(Instant::now());
        release_due().await;
        assert_eq!(
            reclaim_schedule().lock().unwrap_poison().failures,
            0,
            "a clean answered ask resets the ladder"
        );
    }

    /// One ask, one rung: the sweep's failure is the pass, not how many of its names the browser
    /// left unanswered — two Silent leftovers in one pass take a single rung.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn one_pass_with_several_silent_leftovers_takes_one_rung() {
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
            "one pass, one rung, however many names it left unanswered"
        );
    }

    /// The one thing a sweep that found a leftover it could not close leaves behind is that
    /// leftover's revisit: nothing is periodic, so a pass right after asks nothing while the
    /// ask the driver waits on is still parked, and the armed ask is what asks the browser
    /// again once it comes due — no new event needed.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_leftover_the_sweep_could_not_settle_keeps_an_ask_armed() {
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
            "the parked leftover's revisit is what the ask carries: {parked:?} vs {revisit:?}"
        );
        assert!(
            parked > Instant::now(),
            "and it is in the future — the sweep does not spin on it"
        );
        assert!(
            next_release_deadline().is_some(),
            "the driver wakes for that ask by itself, with no event"
        );
        guard.clear_log();
        release_due().await;
        assert!(
            guard.log_lines().is_empty(),
            "and nothing asks before it is due: {:?}",
            guard.log_lines()
        );

        // The only thing time does to an armed ask: bring it due.
        reclaim_schedule().lock().unwrap_poison().next_at = Some(Instant::now());
        guard.clear_log();
        release_due().await;
        assert!(
            guard.invoked("extension call tabGroups.query"),
            "the armed ask is what asks the browser again, with no event: {:?}",
            guard.log_lines()
        );
    }

    /// A record whose leftover no route here can close is not abandoned with it: once the
    /// record is dropped and the leftover reported, the sweep is armed to look at the group
    /// again at its own slower cadence, not at once.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_record_settled_unclosable_arms_the_sweep() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        let names = run_session_names("run-unclosable", &["agent-tab-u-default"]);
        // The group holds a tab the extension does not own, so no route here can close it.
        guard.write_state(&names, false);
        queue_names("run-unclosable", &["agent-tab-u-default"]);

        release_due().await;

        assert!(
            pending_names_and_attempts().is_empty(),
            "the record stops retrying the names no route can close"
        );
        let issues = guard.issues();
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].0, LEFTOVER_MESSAGE);
        assert_eq!(issues[0].2["run"], "run-unclosable");
        let remaining = next_release_deadline()
            .expect("the driver wakes for the group the record left behind")
            .saturating_duration_since(Instant::now());
        assert!(
            STUCK_REVISIT.saturating_sub(remaining) < Duration::from_secs(1),
            "the sweep is armed for the revisit, not at once: {remaining:?}"
        );
    }

    /// The sweep leaves a record's own work alone by exact name, so a record that settled one
    /// name unclosable and left another retrying keeps only the retrying name — and the name
    /// it dropped is examined by the sweep at the revisit, so a browser that recovered still
    /// gets that group closed. This is the regression a namespace-prefix exclusion caused: the
    /// reclaimed group shares the surviving record's namespace, but is no longer its name.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_name_a_record_dropped_is_still_swept() {
        let guard = ReleaseGuard::install(ATTEMPT_TIMEOUT).await;
        // Two names of one run: the first holds a tab the extension does not own (unclosable
        // by this route), the second a closeable one whose remove leg never reaches the
        // browser (a silent retry), so the record ends the pass holding only the second name.
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

        // The browser recovered: the abandoned group's tab is closeable again, so the sweep is
        // the route that closes it. Drive the revisit the record's own conclusion armed.
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

    /// A group whose namespace is a live run's, or a held record's, is never touched by
    /// the reclaim pass: the run may still be using it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_protected_name_is_never_reclaimed() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        guard.arm_sweep();
        // A live run's group, and a held record's group (its run is rebuilt under the
        // same durable id and must still find its tabs).
        let live = ChromeRunSessions::for_run("run-live-prot");
        let live_name = format!("{}default", live.namespace());
        let held = run_session_names("run-held-prot", &["agent-tab-h-prot"]).remove(0);
        queue_names_with_hold(
            "run-held-prot",
            &["agent-tab-h-prot"],
            Duration::from_hours(1),
        );
        guard.write_state(&[live_name, held], true);

        release_due().await;

        assert!(
            !guard.invoked("extension call tabs.remove"),
            "a protected group is never closed: {:?}",
            guard.log_lines()
        );
        drop(live);
    }

    /// The queue cap is kept and the record an overflow drops is the oldest unheld one: it
    /// is logged, its run is remembered so the sweep's report of the tabs can still name it,
    /// and the tabs themselves are left to the reclaim sweep, which owns an `agent-tab-*`
    /// group no live run, no record and no resumable job claims.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn an_overflow_eviction_drops_the_oldest_unheld_record() {
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
        assert_eq!(queued.len(), MAX_PENDING_RELEASES, "the cap is kept");
        let evicted = ChromeRunSessions::for_run("run-evict-0")
            .namespace()
            .to_string();
        assert!(
            !queued
                .iter()
                .any(|(names, _)| names.iter().any(|name| name.starts_with(&evicted))),
            "the oldest unheld record was the one evicted"
        );
        assert!(
            queued.iter().any(|(names, _)| names
                .iter()
                .any(|name| name.ends_with("agent-tab-e-overflow"))),
            "the record the cap made room for is queued"
        );
        assert_eq!(
            evicted_run(&evicted).as_deref(),
            Some("run-evict-0"),
            "the evicted record's run is remembered for the sweep's report"
        );
        assert!(
            guard.issues().is_empty(),
            "an eviction closes nothing and reports nothing: the sweep owns those tabs"
        );
        clear_pending_releases();
    }

    /// The run an eviction remembered is what the sweep's own report names: the evicted
    /// record cannot name its tabs any more, so a leftover it could not close would
    /// otherwise be reported with no run at all.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn an_evicted_records_run_is_named_by_the_sweeps_report() {
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
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].0, LEFT_OPEN_MESSAGE);
        assert_eq!(issues[0].1, format!("chrome-tabs-left-open:{namespace}"));
        assert_eq!(
            issues[0].2["run"], "run-evict-0",
            "the eviction's remembered run is what the sweep's report carries"
        );
        clear_pending_releases();
    }

    /// The attribution an eviction leaves is what the sweep's own report carries — the record
    /// cannot name its tabs any more, so a leftover it could not close would otherwise be
    /// reported with no run at all — and it is bounded: a stale entry is not read, and the
    /// newest entry for a namespace is the one a re-eviction leaves.
    #[test]
    #[serial_test::serial(chrome_release)]
    fn an_evicted_run_is_remembered_for_the_sweeps_report() {
        clear_pending_releases();
        assert_eq!(evicted_run("agent-tab-x-"), None, "nothing remembered yet");

        remember_evicted_run("agent-tab-x-", "run-x");
        remember_evicted_run("agent-tab-y-", "run-y");
        remember_evicted_run("agent-tab-x-", "run-x-again");
        assert_eq!(evicted_run("agent-tab-x-").as_deref(), Some("run-x-again"));
        assert_eq!(evicted_run("agent-tab-y-").as_deref(), Some("run-y"));
        assert_eq!(evicted_run("agent-tab-z-"), None);

        // A remembered run expires with its window rather than with the process.
        let stale = Instant::now()
            .checked_sub(EVICTED_RUN_TTL)
            .expect("the process clock reaches back an hour");
        evicted_runs().lock().unwrap_poison()[1].at = stale;
        remember_evicted_run("agent-tab-z-", "run-z");
        assert_eq!(evicted_run("agent-tab-y-"), None, "the stale entry is gone");
        assert_eq!(evicted_run("agent-tab-z-").as_deref(), Some("run-z"));
        clear_pending_releases();
    }

    /// The cap never takes a record that is still inside its hold: one that is not always
    /// goes first, and with every record inside a hold nothing is evicted at all — dropping
    /// one would lose the tabs of a run that may still come back to them, which is worse
    /// than a queue that grows until one of those holds elapses. A hold that HAS elapsed is
    /// evictable again, like any other record.
    #[test]
    fn eviction_prefers_an_unheld_record_over_a_held_one() {
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
        assert!(!evicted.held, "the unheld record is evicted first");
        let mut held_only = VecDeque::from([record(true, held), record(true, held)]);
        assert!(
            evict_oldest(&mut held_only).is_none(),
            "with only held records, nothing is evicted"
        );
        let mut elapsed = VecDeque::from([record(true, Instant::now())]);
        assert!(
            evict_oldest(&mut elapsed).is_some(),
            "a hold that has elapsed is evictable again"
        );
    }

    /// A full queue of held records grows instead of losing one: the cap is what bounds a
    /// burst of ends, and it never outweighs the tabs of a run that may still come back.
    #[test]
    #[serial_test::serial(chrome_release)]
    fn the_queue_cap_never_evicts_a_held_record() {
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
            "the cap yields to a queue in which every record is held"
        );
        clear_pending_releases();
    }

    /// A close pass that outlives its attempt bound is one failed attempt: the name
    /// stays queued, and the shorter bound is cut off without waiting the stub out.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_timed_out_release_counts_as_one_failed_attempt() {
        let guard = ReleaseGuard::install(Duration::from_secs(1)).await;
        queue_names("run-slow", &["agent-tab-s-slow"]);
        guard.set_mode("silent");

        let started = Instant::now();
        release_due().await;
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "the 1 s attempt bound must cut the silent child off"
        );
        assert_eq!(
            pending_names_and_attempts(),
            vec![(run_session_names("run-slow", &["agent-tab-s-slow"]), 1)],
            "a timed-out attempt is one failed attempt, and the name stays queued"
        );

        // The short bound has served its purpose; the success that follows must
        // not race a cold or loaded stub spawn.
        guard.set_mode("ok");
        widen_attempt_bound(ATTEMPT_TIMEOUT);
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "dropped on the next pass, when the browser answers"
        );
    }

    /// With no chrome-use to run, a due release is skipped rather than failed — the
    /// record stays queued and nothing is spawned — but it is still charged the ladder
    /// so the driver cannot spin on it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_release_with_no_chrome_use_binary_keeps_the_record_queued() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        queue_names("run-absent", &["agent-tab-a-default"]);
        guard.set_binary_absent();

        release_due().await;

        assert_eq!(
            pending_names_and_attempts(),
            vec![(run_session_names("run-absent", &["agent-tab-a-default"]), 1)],
            "nothing the browser could answer, so the name stays queued"
        );
        assert!(guard.log_lines().is_empty(), "no child was spawned");
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
    /// back the names AND the attempt count.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_queued_release_survives_the_process_that_queued_it() {
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
            "one unanswered pass, re-queued"
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

    /// A record file that cannot be read — or that carries an absurd deadline — is
    /// never fatal: an unreadable one restores nothing, and a parseable one whose
    /// deadline is absurd is restored with it clamped rather than trusted.
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
            "neither entry is a record the queue would hold"
        );
        assert!(
            reclaim_schedule().lock().unwrap_poison().next_at.is_none(),
            "and none of them arms a read: a start with no work opens no page"
        );
    }

    /// The record file is this queue's own state, not input to be judged: a record
    /// naming sessions outside the family this queue opens is restored with those
    /// names untouched, and the browser's answer settles them like any other. An entry
    /// with no names is still not queued.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_restored_record_releases_its_names_exactly_as_written() {
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
            "every name the entry carries is restored as written; the entry left with no names is not queued"
        );

        release_due().await;

        assert!(
            pending_releases_snapshot().is_empty(),
            "the browser holds no group by those names, so the record is settled"
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
        assert!(
            guard.only_swept(),
            "nothing was asked for the record: {:?}",
            guard.log_lines()
        );

        tokio::time::sleep(grace * 2).await;
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "settled once the boot grace elapsed"
        );
        assert!(guard.invoked("extension call tabGroups.query"));
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
            guard.only_swept(),
            "no close while the run it belongs to may still come back: {:?}",
            guard.log_lines()
        );

        tokio::time::sleep(hold * 2).await;
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "settled once the hold elapsed"
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
    }

    /// A hold is spent once the record is taken for an attempt: from then on the record is an
    /// ordinary one of the retry ladder, so the retry that re-arms it must not leave it
    /// permanently un-evictable or keep its namespace protected as if a run could come back.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_spent_hold_stops_protecting_its_record() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        // A retry gap longer than this test, so only a hold left standing could keep the
        // re-armed record inside one and the assertions below meaningful.
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

    /// A record whose run is live again is skipped untried — that run owns the
    /// sessions — and becomes releasable the moment it is gone.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_record_whose_run_is_live_again_is_never_attempted() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        let live = ChromeRunSessions::for_run("run-live");
        live.track(&format!("{}default", live.namespace()));
        queue_run_session_release(&live, false, "run-live");

        release_due().await;

        assert_eq!(
            pending_releases_snapshot().len(),
            1,
            "a live run's record stays queued and is not an attempt"
        );
        assert!(
            guard.only_swept(),
            "no close pass is ever run for a live run's sessions: {:?}",
            guard.log_lines()
        );

        drop(live);
        release_due().await;
        assert!(
            pending_releases_snapshot().is_empty(),
            "settled once the run that owns them is gone"
        );
        assert!(guard.invoked("extension call tabGroups.query"));
    }

    /// A run resumed while a pass is in flight has already re-attached the very same
    /// names, so the close set is re-checked per name right before the browser is asked:
    /// a name a live run owns again is left out of the close and its record keeps it
    /// rather than closing the session out from under the run — unless the browser holds
    /// no group of that title any more, which is settled gone whoever owns it.
    #[tokio::test]
    #[serial_test::serial(chrome_release)]
    async fn a_run_that_comes_back_mid_pass_keeps_its_sessions() {
        let guard = ReleaseGuard::install(Duration::from_secs(10)).await;
        let names = run_session_names("run-midpass", &["agent-tab-m-default"]);
        guard.write_state(&names, true);
        // The record a pass would have taken just before the run came back.
        queue_names("run-midpass", &["agent-tab-m-default"]);
        let pass = ReleasePass::take().expect("the record is due before the run comes back");
        // The same durable id is re-dispatched inside the pass window and re-attaches
        // the very same names.
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
            "a run that came back mid-pass keeps its sessions, retried on the next pass"
        );
        assert!(
            guard.log_lines().is_empty(),
            "no child was spawned for a run that is live again"
        );
        assert!(issues.is_empty());
        drop(live);
    }

    /// The production entry point ([`run_session_release_queue`], driven here through
    /// its loop with a local token): a queued record is closed, and a record whose
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
        // The record settles on the tracker's own wake, not on a poll interval. Wait on
        // the durable file — the queue empties as soon as a pass parks the record, so
        // neither the queue nor the log says whether that pass finished.
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

    /// The shutdown flush spends its budget on the records whose hold (or retry
    /// backoff) elapsed and leaves the held ones alone: a held record waits for a
    /// run the restart may re-drive, and being durable it is picked up next boot.
    /// A silent browser side is cut off at the budget and the record is left for that
    /// next boot.
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
        // names under it. It predates the `run` field, so it carries none.
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
}
