//! The stop *signals* of a run, and the one place they are fired: the per-run cancel, the urgent
//! interrupt and the workspace pause live here as one [`Stop`], and the process-wide drain and
//! shutdown are process *facts* this module only reads from [`crate::shutdown`], never duplicates.
//!
//! The registry below holds the [`Stop`] of every live run a cancel action can reach. It exists for
//! the rounds that are not an [`crate::Agent`] — a deep-research run, an analyze quick round, a
//! durable round resumed at boot — which have no stop of their own to hand to a caller outside
//! them: [`register`] hands out the [`RunGuard`] that keeps the round's entry alive for its
//! lifetime, and [`fire`] stops the round through that entry's own [`Stop`], so its boundary gates,
//! its completion gate and its delivery path all read one token of one type.
//!
//! What this module is not: a second set of signals beside the ones it owns. Every stop a run can
//! suffer is fired through [`Stop::fire`] — the registry's sweeps, a cancel a user asked for, the
//! pause request — and every question about one goes through [`Stop::reason`], so a run is stopped
//! in exactly one place and its ending is named in exactly one precedence order. A fired cancel
//! leaves its registry entry in place until the run's guard drops, so [`cancelled`] keeps answering
//! true through the run's whole unwinding tail.
//!
//! The distinctions survive because they mean different things: an interrupt ends the *turn* and
//! the run continues ([`StopReason::is_aborting`] is false), while a cancel, a pause, a drain and a
//! shutdown abort the run; a pause is re-drivable at unpause, so its live request is cleared while
//! the run's record of having obeyed it is not.
//!
//! The invariant: the reason is a *function* of the signals, never stored beside them, so the
//! classifier, the failure reason and the cut-short gate cannot disagree about one run.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use tokio_util::sync::CancellationToken;

use crate::util::UnwrapPoison;

/// Why a run stopped, in one vocabulary — named apart from [`crate::tools::shell::Stop`], which is
/// how a *process* ended. [`crate::Agent::stop`] checks the signals in their own order (a drain
/// outranks a token a force-cancel fired), so the variants carry no ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StopReason {
    /// An urgent message cut the current request; the turn ends and the run continues.
    Interrupted,
    /// A deliberate cancel: the GUI's stop, a re-dispatch, a phase transition.
    Cancelled,
    /// A workspace pause: the run stops at its next round boundary and is re-driven on unpause.
    Paused,
    /// The graceful drain: in-flight work finishes, nothing new starts.
    Drain,
    /// The process is shutting down.
    Shutdown,
}

impl StopReason {
    /// The process-wide reason, from the two global facts — the drain first, because a force-cancel
    /// fires the token mid-drain and the drain is the one that names the stop then.
    #[must_use]
    pub(crate) const fn process(draining: bool, shutting_down: bool) -> Option<Self> {
        if draining {
            Some(Self::Drain)
        } else if shutting_down {
            Some(Self::Shutdown)
        } else {
            None
        }
    }

    /// The stable classification label the agent-failure log and the telemetry columns carry — the
    /// exact strings the classifier returned before this type existed, so a stored row still reads
    /// the same.
    #[must_use]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Interrupted => "interrupted",
            Self::Cancelled => "internal_cancel",
            Self::Paused => "pause",
            Self::Drain => "drain",
            Self::Shutdown => "shutdown",
        }
    }

    /// Whether this stop aborts the run rather than ending one turn — the cut-short rule: an
    /// urgent interrupt ends the turn and the run carries on, every other reason is a stop.
    #[must_use]
    pub(crate) const fn is_aborting(self) -> bool {
        !matches!(self, Self::Interrupted)
    }
}

/// One run's stop: the three signals a run can be stopped by, owned in one place. The registry
/// holds the same `Arc`, so a sweep fires the run's own token instead of a copy of it.
///
/// The drain and the shutdown are process-wide *facts* ([`crate::shutdown`]) — [`Self::reason`]
/// reads them, this struct holds nothing for them.
#[derive(Default)]
pub(crate) struct Stop {
    cancel: CancellationToken,
    interrupt: CancellationToken,
    /// The workspace-pause stop of this run: the registry's live request, shared with the agent.
    pause_requested: AtomicBool,
    /// The run's own record of having obeyed the pause — which an unpause must not be able to
    /// erase, or a deliberately stopped run would be classified a hard failure.
    pause_obeyed: AtomicBool,
}

impl Stop {
    /// Fire a stop of one kind, returning whether this call changed anything so a caller publishes
    /// only on a real change. The drain and the shutdown are the process's to fire, never a run's:
    /// asking for one here changes nothing.
    pub(crate) fn fire(&self, reason: StopReason) -> bool {
        let token = match reason {
            StopReason::Interrupted => &self.interrupt,
            StopReason::Cancelled => &self.cancel,
            StopReason::Paused => return !self.pause_requested.swap(true, Ordering::SeqCst),
            StopReason::Drain | StopReason::Shutdown => return false,
        };
        let already = token.is_cancelled();
        token.cancel();
        !already
    }

    /// Whether this stop has fired — the question every round boundary and every finalizer asks.
    /// A pause answers from the run's immutable record of having obeyed it, so a resumed workspace
    /// cannot un-stop a run that already bailed.
    #[must_use]
    pub(crate) fn is_fired(&self, reason: StopReason) -> bool {
        match reason {
            StopReason::Interrupted => self.interrupt.is_cancelled(),
            StopReason::Cancelled => self.cancel.is_cancelled(),
            StopReason::Paused => self.pause_obeyed.load(Ordering::SeqCst),
            StopReason::Drain => crate::shutdown::is_draining(),
            StopReason::Shutdown => crate::shutdown::shutdown_token().is_cancelled(),
        }
    }

    /// The one place a run's stop reason is derived: the highest-priority signal that is set, the
    /// process-wide pair first and in their own order (a drain outranks a token a force-cancel
    /// fired mid-drain), then the pause's immutable record, the cancel and the turn's interrupt.
    #[must_use]
    pub(crate) fn reason(&self) -> Option<StopReason> {
        StopReason::process(
            crate::shutdown::is_draining(),
            crate::shutdown::shutdown_token().is_cancelled(),
        )
        .or_else(|| {
            self.is_fired(StopReason::Paused)
                .then_some(StopReason::Paused)
        })
        .or_else(|| {
            self.is_fired(StopReason::Cancelled)
                .then_some(StopReason::Cancelled)
        })
        .or_else(|| {
            self.is_fired(StopReason::Interrupted)
                .then_some(StopReason::Interrupted)
        })
    }

    /// The turn's own interrupt, for the loop that has to await it beside a model call.
    #[must_use]
    pub(crate) fn interrupt_token(&self) -> CancellationToken {
        self.interrupt.clone()
    }

    /// Whether a pause is currently requested — the round-top check and the GUI card, which must
    /// see a request the run has not obeyed yet.
    #[must_use]
    pub(crate) fn pause_requested(&self) -> bool {
        self.pause_requested.load(Ordering::SeqCst)
    }

    /// Drop the live pause request on unpause. Deliberately leaves the obeyed record alone: a run
    /// that already bailed on the pause stays a stop, not a failure.
    pub(crate) fn clear_pause(&self) -> bool {
        self.pause_requested.swap(false, Ordering::SeqCst)
    }

    /// Record, once and for good, that this run obeyed the pause — the authoritative read for a
    /// finalizer that may run after the workspace was resumed.
    pub(crate) fn obey_pause(&self) {
        self.pause_obeyed.store(true, Ordering::SeqCst);
    }
}

/// How many unregistered-cancel marks are kept — see [`RunState::fired`].
const FIRED_MARKS: usize = 32;

/// Every live run's [`Stop`], keyed by the id a cancel action addresses — see the module doc.
static RUNS: LazyLock<RunRegistry> = LazyLock::new(RunRegistry::default);

#[derive(Default)]
struct RunRegistry {
    inner: Mutex<RunState>,
}

#[derive(Default)]
struct RunState {
    /// One entry per live invocation, keyed by run id.
    runs: HashMap<String, Entry>,
    /// Ids a cancel fired for while NOTHING was registered — the one window where a canceller and a
    /// registrant cross: a boot resume registers only after the scan that picked its run. The
    /// registration that follows starts pre-fired; marks nothing claims expire by cap.
    fired: VecDeque<String>,
}

struct Entry {
    stop: Arc<Stop>,
    /// The dispatching session's agent id, when the registration carried one.
    caller: Option<String>,
}

impl RunRegistry {
    fn register(&self, job_id: &str, caller: Option<&str>) -> RunGuard {
        let mut state = self.inner.lock().unwrap_poison();
        // A cancel that fired while this id had no entry is picked up here: the run starts
        // pre-fired and observes it at its first boundary instead of delivering to a caller told it
        // would not come.
        let pre_fired = match state.fired.iter().position(|id| id == job_id) {
            Some(at) => {
                state.fired.remove(at);
                true
            }
            None => false,
        };
        let stop = Arc::new(Stop::default());
        if pre_fired {
            stop.fire(StopReason::Cancelled);
        }
        state.runs.insert(
            job_id.to_string(),
            Entry {
                stop,
                caller: caller.map(str::to_string),
            },
        );
        RunGuard {
            job_id: job_id.to_string(),
        }
    }

    /// Fire a run's cancel through its own [`Stop`] (the entry stays — see the module doc). An id
    /// with no entry leaves a mark for the registration racing this cancel, and no stop changed.
    fn fire(&self, job_id: &str) -> bool {
        let mut state = self.inner.lock().unwrap_poison();
        if let Some(entry) = state.runs.get(job_id) {
            return entry.stop.fire(StopReason::Cancelled);
        }
        if !state.fired.iter().any(|id| id == job_id) {
            if state.fired.len() == FIRED_MARKS {
                let _ = state.fired.pop_front();
            }
            state.fired.push_back(job_id.to_string());
        }
        false
    }

    fn cancelled(&self, job_id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_poison()
            .runs
            .get(job_id)
            .is_some_and(|entry| entry.stop.is_fired(StopReason::Cancelled))
    }

    /// The run's cancel token, for a caller that waits on it instead of polling.
    fn cancel_token(&self, job_id: &str) -> Option<CancellationToken> {
        self.inner
            .lock()
            .unwrap_poison()
            .runs
            .get(job_id)
            .map(|entry| entry.stop.cancel.clone())
    }

    fn is_registered(&self, job_id: &str) -> bool {
        self.inner.lock().unwrap_poison().runs.contains_key(job_id)
    }

    fn registered_caller(&self, job_id: &str) -> Option<String> {
        self.inner
            .lock()
            .unwrap_poison()
            .runs
            .get(job_id)
            .and_then(|entry| entry.caller.clone())
    }

    fn unregister(&self, job_id: &str) {
        self.inner.lock().unwrap_poison().runs.remove(job_id);
    }
}

/// RAII guard: removes the run's registry entry when the invocation exits — on every path, panics
/// included.
pub(crate) struct RunGuard {
    job_id: String,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        RUNS.unregister(&self.job_id);
    }
}

/// Register the run's [`Stop`] for one invocation (fresh dispatch or boot resume); `caller` is the
/// dispatching session's agent id. The caller holds the guard for the whole invocation.
pub(crate) fn register(job_id: &str, caller: Option<&str>) -> RunGuard {
    RUNS.register(job_id, caller)
}

/// Fire the run's cancel through its [`Stop`] — the user cancelled a research run from the Running
/// Agents page, or an analyze round's caller cancelled it — and report whether this call changed
/// it. Must be called BEFORE any row deletion so the invocation observes it at its next boundary.
/// An id that has not registered yet is marked, so the registration that follows starts cancelled
/// ([`RunState::fired`]) and nothing changed here.
pub(crate) fn fire(job_id: &str) -> bool {
    RUNS.fire(job_id)
}

/// Whether the run's cancel has fired — consulted by the boundary gates, the completion gate and
/// the delivery path.
pub(crate) fn cancelled(job_id: &str) -> bool {
    RUNS.cancelled(job_id)
}

/// Resolve when the run's cancel fires — the phase-level stop for a round's own model calls. Never
/// resolves for an unregistered id: such an id is marked, not signalled.
pub(crate) async fn await_cancel(job_id: &str) {
    match RUNS.cancel_token(job_id) {
        Some(token) => token.cancelled().await,
        None => std::future::pending::<()>().await,
    }
}

/// Whether a live invocation is registered for the run: registered means it is alive and will
/// observe a fired signal at its next boundary.
pub(crate) fn is_registered(job_id: &str) -> bool {
    RUNS.is_registered(job_id)
}

/// The dispatching session's agent id recorded at [`register`] time — lets a cancel action prove
/// ownership of a run whose jobs row the dispatch has not written yet.
pub(crate) fn registered_caller(job_id: &str) -> Option<String> {
    RUNS.registered_caller(job_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pause_stop_records_the_request_and_the_obedience_apart() {
        let stop = Stop::default();
        assert!(!stop.pause_requested(), "a fresh run is not paused");
        assert!(
            stop.fire(StopReason::Paused),
            "the first request changes the state"
        );
        assert!(stop.pause_requested());
        assert!(
            !stop.fire(StopReason::Paused),
            "a repeated request changes nothing"
        );
        assert!(
            !stop.is_fired(StopReason::Paused),
            "the request is not yet obeyed"
        );

        stop.obey_pause();
        assert!(
            stop.is_fired(StopReason::Paused),
            "the record survives the unpause"
        );

        assert!(stop.clear_pause(), "the unpause clears the live request");
        assert!(!stop.pause_requested());
        assert!(!stop.clear_pause(), "a second unpause changes nothing");
        assert!(
            stop.is_fired(StopReason::Paused),
            "the obeyed record is never cleared"
        );
    }

    #[test]
    fn a_drain_outranks_a_token_a_force_cancel_fired_mid_drain() {
        assert_eq!(StopReason::process(false, false), None);
        assert_eq!(StopReason::process(true, false), Some(StopReason::Drain));
        assert_eq!(StopReason::process(false, true), Some(StopReason::Shutdown));
        assert_eq!(
            StopReason::process(true, true),
            Some(StopReason::Drain),
            "a force-cancel fires the token mid-drain; the drain names the stop"
        );
    }

    #[test]
    fn the_labels_are_the_ones_the_failure_log_carries() {
        assert_eq!(StopReason::Interrupted.label(), "interrupted");
        assert_eq!(StopReason::Cancelled.label(), "internal_cancel");
        assert_eq!(StopReason::Paused.label(), "pause");
        assert_eq!(StopReason::Drain.label(), "drain");
        assert_eq!(StopReason::Shutdown.label(), "shutdown");
    }

    #[test]
    fn only_an_urgent_interrupt_leaves_the_run_alive() {
        assert!(
            !StopReason::Interrupted.is_aborting(),
            "the turn ends, the run goes on"
        );
        assert!(StopReason::Cancelled.is_aborting());
        assert!(StopReason::Paused.is_aborting());
        assert!(StopReason::Drain.is_aborting());
        assert!(StopReason::Shutdown.is_aborting());
    }

    /// The registered run's lifecycle: an unregistered id reports no signal, a registered one
    /// reports its signal until its guard drops, and the caller recorded at registration time is
    /// readable back (so a cancel can prove ownership before the run's row exists). A cancel of an
    /// unregistered id is not lost — the registration racing it starts pre-fired.
    #[tokio::test]
    async fn cancel_fires_the_registered_stop_and_the_guard_removes_the_entry_on_drop() {
        // Unregistered id: the cancel leaves a mark for a registration racing it (see the module
        // doc), and `cancelled` — which only reports a registered run's signal — stays false.
        assert!(!fire("stop_unregistered_run"));
        assert!(!cancelled("stop_unregistered_run"));

        // The ownership accessor is empty for an id nothing registered.
        assert!(
            !is_registered("stop_caller_run"),
            "nothing is running for an unregistered id"
        );
        assert_eq!(
            registered_caller("stop_caller_run"),
            None,
            "an unregistered id has no recorded dispatching session"
        );

        let caller_guard = register("stop_caller_run", Some("caller_session"));
        assert!(
            is_registered("stop_caller_run"),
            "a registered invocation is visible"
        );
        assert_eq!(
            registered_caller("stop_caller_run"),
            Some("caller_session".to_string()),
            "the dispatching session's id is recorded for the cancel action"
        );
        drop(caller_guard);
        assert_eq!(
            registered_caller("stop_caller_run"),
            None,
            "guard drop removes the recorded caller too"
        );

        let guard = register("stop_signal_run", None);
        assert!(!cancelled("stop_signal_run"));
        assert!(fire("stop_signal_run"), "the first cancel changes the stop");
        assert!(cancelled("stop_signal_run"), "fired signal stays visible");
        assert!(
            !fire("stop_signal_run"),
            "a second cancel changes nothing, but the entry stays so the signal stays visible"
        );
        assert!(cancelled("stop_signal_run"));
        drop(guard);
        assert!(
            !cancelled("stop_signal_run"),
            "guard drop removes the entry"
        );

        // A cancel of an id nothing registered is not lost: the registration that races it (a boot
        // resume, which registers only after the scan that picked its run) starts pre-fired, so the
        // run cannot run and deliver to a caller already answered as stopped.
        assert!(!is_registered("stop_late_register"));
        fire("stop_late_register");
        let late = register("stop_late_register", None);
        assert!(
            cancelled("stop_late_register"),
            "a registration racing a cancel starts pre-fired"
        );
        drop(late);
        assert!(
            !cancelled("stop_late_register"),
            "the mark is consumed by the registration it was waiting for"
        );
    }
}
