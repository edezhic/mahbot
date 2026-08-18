//! Ticket phase transition buffer — piggyback non-critical transitions
//! onto critical notifications and user messages.
//!
//! Non-critical transitions (called with `NotifyPolicy::Buffer`) are buffered here and drained when a critical
//! notification triggers `notify_ticket` or when the user sends a
//! Manager message. This ensures the Manager sees all ticket state changes
//! without requiring a fresh `build_board_context` snapshot on every turn.
//!
//! Every entry carries its transition timestamp and origin (pipeline flow
//! vs user action). On drain the Manager receives the full chronological
//! sequence accumulated since the last drain — every hop, never coalesced.

use std::collections::{HashMap, VecDeque};
use std::fmt::Write;
use std::sync::{Mutex, OnceLock};

use crate::board::TicketPhase;
use crate::util::UnwrapPoison;

/// Who caused a ticket phase transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransitionOrigin {
    /// Automated board pipeline (poller, dispatch, verdict machinery).
    Pipeline,
    /// A user action (GUI board actions).
    User,
}

impl std::fmt::Display for TransitionOrigin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransitionOrigin::Pipeline => f.write_str("pipeline flow"),
            TransitionOrigin::User => f.write_str("user action"),
        }
    }
}

/// A single buffered ticket phase transition entry.
#[derive(Debug)]
struct Entry {
    id: String,
    source: TicketPhase,
    target: TicketPhase,
    at: String,
    origin: TransitionOrigin,
}

/// Global ticket transition buffer, keyed by workspace name.
static TICKET_BUFFER: OnceLock<Mutex<HashMap<String, VecDeque<Entry>>>> = OnceLock::new();

/// Initialize the global ticket buffer. Must be called during startup.
///
/// Idempotent — uses [`OnceLock::get_or_init`] so tests that reset the
/// buffer and production bootstrap cannot race each other's initialization.
pub fn init_global() {
    TICKET_BUFFER.get_or_init(|| Mutex::new(HashMap::new()));
}

/// Access the underlying mutex, panicking if the buffer is not initialized.
fn buffer() -> &'static Mutex<HashMap<String, VecDeque<Entry>>> {
    TICKET_BUFFER
        .get()
        .expect("ticket_buffer not initialized — call init_global() first")
}

/// Push a non-critical ticket transition into the buffer.
///
/// The entry is timestamped at push time and tagged with its origin.
///
/// # Panics
///
/// Panics if the buffer has not been initialized via [`init_global`].
pub(crate) fn push(
    workspace_name: &str,
    id: &str,
    source: TicketPhase,
    target: TicketPhase,
    origin: TransitionOrigin,
) {
    let mut map = buffer().lock().unwrap_poison();
    let deque = map.entry(workspace_name.to_string()).or_default();
    deque.push_back(Entry {
        id: id.to_string(),
        source,
        target,
        at: crate::turso::now(),
        origin,
    });
}

/// Drain all buffered entries for a workspace.
///
/// Returns a formatted string ready for insertion into a notification,
/// or an empty string if no entries are buffered.
///
/// # Panics
///
/// Panics if the buffer has not been initialized via [`init_global`].
///
/// Format:
/// ```text
/// Ticket updates:
/// • mahbot-1: in_development → in_diagnostics (pipeline flow, 2026-08-07T11:00:00+00:00)
/// • mahbot-2: in_diagnostics → diagnostics_done (user action, 2026-08-07T11:05:00+00:00)
/// ```
#[must_use]
pub(crate) fn drain(workspace_name: &str) -> String {
    let mut map = buffer().lock().unwrap_poison();
    let Some(entries) = map.remove(workspace_name) else {
        return String::new();
    };
    let mut out = String::from("<ticket-updates>");
    for entry in entries {
        let _ = writeln!(
            out,
            "• {}: {} → {} ({}, {})",
            entry.id, entry.source, entry.target, entry.origin, entry.at
        );
    }
    let _ = writeln!(out, "</ticket-updates>");
    out
}

/// Reset all buffers for test isolation.
#[cfg(test)]
pub fn reset() {
    init_global();
    buffer().lock().unwrap_poison().clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::TicketPhase;
    use std::sync::Mutex;

    /// Test serialization guard — Rust runs tests in parallel by default,
    /// but the global `TICKET_BUFFER` is shared state. Each test that
    /// mutates the buffer must hold this lock.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn push_entry(ws: &str, id: &str, source: TicketPhase, target: TicketPhase) {
        push(ws, id, source, target, TransitionOrigin::Pipeline);
    }

    #[test]
    fn push_and_drain_ordered() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        push_entry(
            "ws-a",
            "mahbot-1",
            TicketPhase::Backlog,
            TicketPhase::Analysis,
        );
        push(
            "ws-a",
            "mahbot-2",
            TicketPhase::Analysis,
            TicketPhase::Planning,
            TransitionOrigin::User,
        );
        push_entry(
            "ws-a",
            "mahbot-3",
            TicketPhase::InDevelopment,
            TicketPhase::InDiagnostics,
        );
        let result = drain("ws-a");
        assert!(result.contains("mahbot-1: backlog → analysis"));
        assert!(result.contains("mahbot-2: analysis → planning"));
        assert!(result.contains("mahbot-3: in_development → in_diagnostics"));
        let pos1 = result.find("mahbot-1").unwrap();
        let pos2 = result.find("mahbot-2").unwrap();
        let pos3 = result.find("mahbot-3").unwrap();
        assert!(pos1 < pos2 && pos2 < pos3);
    }

    #[test]
    fn drain_nonexistent_returns_empty() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        assert_eq!(drain("nonexistent"), "");
    }

    #[test]
    fn workspace_isolation() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        push_entry(
            "ws-a",
            "mahbot-1",
            TicketPhase::Backlog,
            TicketPhase::Analysis,
        );
        push_entry(
            "ws-b",
            "mahbot-2",
            TicketPhase::ReadyForDevelopment,
            TicketPhase::InDevelopment,
        );
        let result_a = drain("ws-a");
        assert!(result_a.contains("mahbot-1"));
        assert!(!result_a.contains("mahbot-2"));
        let result_b = drain("ws-b");
        assert!(result_b.contains("mahbot-2"));
        assert!(!result_b.contains("mahbot-1"));
    }

    #[test]
    fn drain_consumes_entries() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        push_entry(
            "ws-a",
            "mahbot-1",
            TicketPhase::Backlog,
            TicketPhase::Analysis,
        );
        let first = drain("ws-a");
        assert!(!first.is_empty());
        let second = drain("ws-a");
        assert!(second.is_empty());
    }
}
