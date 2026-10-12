//! The in-process per-round cancel signals shared by every cancellable durable
//! round — deep-research runs and the analyze tool's quick rounds — plus the
//! RAII guard that keeps a round's entry alive for its lifetime.
//!
//! An entry exists from the moment a round starts (fresh dispatch or boot
//! resume) until it exits (any exit — terminal, abort, cancel, panic — via the
//! guard's drop). [`cancel`] FIRES the token but deliberately leaves the entry
//! in place so [`is_cancelled`] keeps returning true for the round's boundary
//! gates, its completion gate and its delivery path until the round actually
//! stops. The entry also records the dispatching session's agent id when there
//! is one ([`registered_caller`]), so a cancel action can prove ownership of a
//! round whose jobs row the dispatch has not written yet. A [`cancel`] of an id
//! nothing had registered leaves a bounded mark instead of being lost, so a
//! registration racing that cancel is pre-fired — see [`CancelState::fired`].

use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

use tokio_util::sync::CancellationToken;

use crate::util::UnwrapPoison;

/// How many unregistered-cancel marks are kept — see [`CancelState::fired`].
const FIRED_MARKS: usize = 32;

/// Per-round cancel signals — see the module doc.
static CANCELS: LazyLock<CancelRegistry> = LazyLock::new(CancelRegistry::default);

#[derive(Default)]
struct CancelRegistry {
    inner: Mutex<CancelState>,
}

#[derive(Default)]
struct CancelState {
    /// One entry per live invocation, keyed by round id.
    rounds: HashMap<String, Entry>,
    /// Ids a cancel fired for while NOTHING was registered — the one window where
    /// a canceller and a registrant cross: a boot resume registers only after the
    /// scan that picked its round, so a cancel landing in between fires into
    /// nothing. The registration it was waiting for picks the mark up and starts
    /// pre-fired; the marks nothing claims (a cancel of a round that is already
    /// gone and will never register again) expire by cap.
    fired: VecDeque<String>,
}

/// One round's registry entry.
struct Entry {
    /// The round's cancel signal.
    token: CancellationToken,
    /// The dispatching session's agent id, when the registration carried one.
    caller: Option<String>,
}

impl CancelRegistry {
    fn register(&self, job_id: &str, caller: Option<&str>) -> CancelGuard {
        let mut state = self.inner.lock().unwrap_poison();
        // A cancel that fired while this id had no entry (a boot resume racing a
        // cancel) is picked up here: the round starts pre-fired and observes it
        // at its first boundary instead of running to a delivery its caller was
        // told would not come.
        let pre_fired = match state.fired.iter().position(|id| id == job_id) {
            Some(at) => {
                state.fired.remove(at);
                true
            }
            None => false,
        };
        let token = CancellationToken::new();
        if pre_fired {
            token.cancel();
        }
        state.rounds.insert(
            job_id.to_string(),
            Entry {
                token,
                caller: caller.map(str::to_string),
            },
        );
        CancelGuard {
            job_id: job_id.to_string(),
        }
    }

    /// Fire a round's cancel signal (the entry stays — see the module doc). An id
    /// with no entry leaves a mark for the registration racing this cancel.
    fn cancel(&self, job_id: &str) {
        let mut state = self.inner.lock().unwrap_poison();
        if let Some(entry) = state.rounds.get(job_id) {
            entry.token.cancel();
            return;
        }
        if !state.fired.iter().any(|id| id == job_id) {
            if state.fired.len() == FIRED_MARKS {
                let _ = state.fired.pop_front();
            }
            state.fired.push_back(job_id.to_string());
        }
    }

    /// Whether the round's cancel signal has been fired.
    fn is_cancelled(&self, job_id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_poison()
            .rounds
            .get(job_id)
            .is_some_and(|entry| entry.token.is_cancelled())
    }

    /// The round's cancel signal, for a caller that waits on it instead of
    /// polling.
    fn token(&self, job_id: &str) -> Option<CancellationToken> {
        self.inner
            .lock()
            .unwrap_poison()
            .rounds
            .get(job_id)
            .map(|entry| entry.token.clone())
    }

    /// Whether a live invocation is registered for the round.
    fn is_registered(&self, job_id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_poison()
            .rounds
            .contains_key(job_id)
    }

    /// The dispatching session's agent id recorded at registration, if any.
    fn registered_caller(&self, job_id: &str) -> Option<String> {
        self.inner
            .lock()
            .unwrap_poison()
            .rounds
            .get(job_id)
            .and_then(|entry| entry.caller.clone())
    }

    fn unregister(&self, job_id: &str) {
        self.inner.lock().unwrap_poison().rounds.remove(job_id);
    }
}

/// RAII guard: removes the round's cancel-signal entry when the invocation
/// exits — on every path, including panics (Drop runs during unwind).
pub(crate) struct CancelGuard {
    job_id: String,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        CANCELS.unregister(&self.job_id);
    }
}

/// Register the round's cancel signal for the duration of one invocation
/// (fresh dispatch or boot resume). `caller` is the dispatching session's
/// agent id when there is one. Callers hold the returned guard for the whole
/// invocation.
pub(crate) fn register(job_id: &str, caller: Option<&str>) -> CancelGuard {
    CANCELS.register(job_id, caller)
}

/// Fire the round's cancel signal — the user cancelled a research run from the
/// Running Agents page, or an analyze round's caller cancelled it with the
/// analyze tool's cancel action. Must be called BEFORE any row deletion so the
/// invocation observes it at its next boundary and stops. An id whose invocation
/// has not registered yet is marked, so the registration that follows starts
/// cancelled ([`CancelState::fired`]).
pub(crate) fn cancel(job_id: &str) {
    CANCELS.cancel(job_id);
}

/// Whether the round's cancel signal has been fired. Consulted by the
/// invocation's boundary gates, its completion gate and its delivery path.
pub(crate) fn is_cancelled(job_id: &str) -> bool {
    CANCELS.is_cancelled(job_id)
}

/// Resolve when the round's cancel signal fires — the phase-level stop for a
/// round's own model calls. Never resolves for an id nothing registered: such an
/// id is marked, not signalled ([`cancel`]), so nothing can complete that wait.
pub(crate) async fn cancelled(job_id: &str) {
    match CANCELS.token(job_id) {
        Some(token) => token.cancelled().await,
        None => std::future::pending::<()>().await,
    }
}

/// Whether a live invocation is registered for the round: a registered id means
/// it is alive and will observe a fired signal at its next boundary, an
/// unregistered one means nothing is running for it (or it has already exited).
pub(crate) fn is_registered(job_id: &str) -> bool {
    CANCELS.is_registered(job_id)
}

/// The dispatching session's agent id recorded at [`register`] time, if any —
/// lets a cancel action prove ownership of a round whose jobs row the dispatch
/// has not written yet.
pub(crate) fn registered_caller(job_id: &str) -> Option<String> {
    CANCELS.registered_caller(job_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The signal's lifecycle and the ownership accessor behind it: an
    /// unregistered id reports no signal, a registered one reports its fired
    /// signal until its guard drops, and the caller recorded at registration time
    /// is readable back (so a cancel can prove ownership before the round's row
    /// exists). A cancel of an unregistered id is not lost — the registration
    /// racing it starts pre-fired.
    #[tokio::test]
    async fn cancel_fires_signal_and_guard_removes_entry_on_drop() {
        // Unregistered id: the cancel leaves a mark for a registration racing it
        // (see the module doc), and `is_cancelled` — which only reports a
        // registered round's signal — stays false.
        cancel("nope");
        assert!(!is_cancelled("nope"));

        // The ownership accessor is empty for an id nothing registered.
        assert!(
            !is_registered("run_caller_1"),
            "nothing is running for an unregistered id"
        );
        assert_eq!(
            registered_caller("run_caller_1"),
            None,
            "an unregistered id has no recorded dispatching session"
        );

        let caller_guard = register("run_caller_1", Some("caller_session"));
        assert!(
            is_registered("run_caller_1"),
            "a registered invocation is visible"
        );
        assert_eq!(
            registered_caller("run_caller_1"),
            Some("caller_session".to_string()),
            "the dispatching session's id is recorded for the cancel action"
        );
        drop(caller_guard);
        assert_eq!(
            registered_caller("run_caller_1"),
            None,
            "guard drop removes the recorded caller too"
        );

        let guard = register("run_signal_1", None);
        assert!(!is_cancelled("run_signal_1"));
        cancel("run_signal_1");
        assert!(is_cancelled("run_signal_1"), "fired signal stays visible");
        // A second cancel is idempotent.
        cancel("run_signal_1");
        assert!(is_cancelled("run_signal_1"));
        drop(guard);
        assert!(
            !is_cancelled("run_signal_1"),
            "guard drop removes the entry"
        );

        // A cancel of an id nothing registered is not lost: the registration
        // that races it (a boot resume, which registers only after the scan that
        // picked its round) starts pre-fired, so the round cannot run and
        // deliver to a caller already answered as stopped.
        assert!(!is_registered("run_late_register"));
        cancel("run_late_register");
        let late = register("run_late_register", None);
        assert!(
            is_cancelled("run_late_register"),
            "a registration racing a cancel starts pre-fired"
        );
        drop(late);
        assert!(
            !is_cancelled("run_late_register"),
            "the mark is consumed by the registration it was waiting for"
        );
    }
}
