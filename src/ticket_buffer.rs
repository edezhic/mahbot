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
//! Consecutive transitions of the same ticket with the same origin are
//! grouped into a single labeled block so a ticket bouncing through many
//! phases reads as one block instead of repeated identical bullets; a new
//! block starts whenever the (ticket, origin) pair changes (run-based in
//! buffer order), so no hop, timestamp, or origin is lost or reordered.

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
/// Format: consecutive transitions of the same ticket with the same origin
/// are grouped into one labeled block (a new block starts on any
/// (ticket, origin) change); each hop keeps its full RFC 3339 UTC timestamp.
/// ```text
/// <ticket-updates>
/// • mahbot-1 (pipeline flow):
///     in_development → in_diagnostics (2026-08-07T11:00:00+00:00)
///     in_diagnostics → in_review (2026-08-07T11:05:00+00:00)
/// • mahbot-2 (user action):
///     analysis → planning (2026-08-07T11:10:00+00:00)
/// </ticket-updates>
/// ```
#[must_use]
pub(crate) fn drain(workspace_name: &str) -> String {
    let mut map = buffer().lock().unwrap_poison();
    let Some(entries) = map.remove(workspace_name) else {
        return String::new();
    };
    let mut out = String::from("<ticket-updates>\n");
    let mut current_block: Option<(&str, TransitionOrigin)> = None;
    for entry in &entries {
        let block_key = (entry.id.as_str(), entry.origin);
        if current_block != Some(block_key) {
            let _ = writeln!(out, "• {} ({}):", entry.id, entry.origin);
            current_block = Some(block_key);
        }
        let _ = writeln!(
            out,
            "    {} → {} ({})",
            entry.source, entry.target, entry.at
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

    /// Push an entry with a deterministic timestamp (tests pin the exact
    /// rendered format; [`push`] timestamps via [`crate::turso::now`]).
    fn push_raw(
        ws: &str,
        id: &str,
        source: TicketPhase,
        target: TicketPhase,
        origin: TransitionOrigin,
        at: &str,
    ) {
        buffer()
            .lock()
            .unwrap_poison()
            .entry(ws.to_string())
            .or_default()
            .push_back(Entry {
                id: id.to_string(),
                source,
                target,
                at: at.to_string(),
                origin,
            });
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
        // Envelope markup preserved.
        assert!(result.starts_with("<ticket-updates>\n"));
        assert!(result.ends_with("</ticket-updates>\n"));
        // Each (id, origin) run gets a labeled header; hop lines keep the
        // source → target arrow plus the full RFC 3339 timestamp.
        assert!(result.contains("• mahbot-1 (pipeline flow):"));
        assert!(result.contains("• mahbot-2 (user action):"));
        assert!(result.contains("• mahbot-3 (pipeline flow):"));
        assert!(result.contains("    backlog → analysis ("));
        assert!(result.contains("    analysis → planning ("));
        assert!(result.contains("    in_development → in_diagnostics ("));
        let pos1 = result.find("• mahbot-1").unwrap();
        let pos2 = result.find("• mahbot-2").unwrap();
        let pos3 = result.find("• mahbot-3").unwrap();
        assert!(pos1 < pos2 && pos2 < pos3);
    }

    #[test]
    fn single_origin_ticket_groups_all_hops_into_one_block() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        push_raw(
            "ws-a",
            "mahbot-1736",
            TicketPhase::InDevelopment,
            TicketPhase::InDiagnostics,
            TransitionOrigin::Pipeline,
            "2026-08-17T08:11:34.225709+00:00",
        );
        push_raw(
            "ws-a",
            "mahbot-1736",
            TicketPhase::InDiagnostics,
            TicketPhase::InReview,
            TransitionOrigin::Pipeline,
            "2026-08-17T08:21:19.225709+00:00",
        );
        push_raw(
            "ws-a",
            "mahbot-1736",
            TicketPhase::InReview,
            TicketPhase::InQa,
            TransitionOrigin::Pipeline,
            "2026-08-17T08:32:01.225709+00:00",
        );
        let result = drain("ws-a");
        // One labeled header — the repeated-id noise is gone.
        assert_eq!(result.matches("• mahbot-1736").count(), 1);
        assert!(result.contains("• mahbot-1736 (pipeline flow):"));
        // Every hop keeps its full RFC 3339 timestamp, in push order.
        assert!(
            result
                .contains("    in_development → in_diagnostics (2026-08-17T08:11:34.225709+00:00)")
        );
        assert!(
            result.contains("    in_diagnostics → in_review (2026-08-17T08:21:19.225709+00:00)")
        );
        assert!(result.contains("    in_review → in_qa (2026-08-17T08:32:01.225709+00:00)"));
        let hop1 = result.find("    in_development").unwrap();
        let hop2 = result.find("    in_diagnostics → in_review").unwrap();
        let hop3 = result.find("    in_review → in_qa").unwrap();
        assert!(hop1 < hop2 && hop2 < hop3);
    }

    #[test]
    fn origin_change_mid_run_starts_new_labeled_block() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        push_raw(
            "ws-a",
            "mahbot-1736",
            TicketPhase::InDevelopment,
            TicketPhase::InDiagnostics,
            TransitionOrigin::Pipeline,
            "2026-08-17T08:11:34.225709+00:00",
        );
        push_raw(
            "ws-a",
            "mahbot-1736",
            TicketPhase::InDiagnostics,
            TicketPhase::InReview,
            TransitionOrigin::Pipeline,
            "2026-08-17T08:21:19.225709+00:00",
        );
        push_raw(
            "ws-a",
            "mahbot-1736",
            TicketPhase::InReview,
            TicketPhase::ReadyForDevelopment,
            TransitionOrigin::User,
            "2026-08-17T09:02:11.225709+00:00",
        );
        let result = drain("ws-a");
        // Two runs → two labeled headers for the same ticket.
        assert_eq!(result.matches("• mahbot-1736").count(), 2);
        assert!(result.contains("• mahbot-1736 (pipeline flow):"));
        assert!(result.contains("• mahbot-1736 (user action):"));
        // Pipeline hops sit under the pipeline header, the user hop under
        // the user header; nothing is lost or reordered.
        let pipeline_header = result.find("• mahbot-1736 (pipeline flow)").unwrap();
        let user_header = result.find("• mahbot-1736 (user action)").unwrap();
        let pipeline_hop = result.find("    in_development → in_diagnostics").unwrap();
        let user_hop = result
            .find("    in_review → ready_for_development")
            .unwrap();
        assert!(pipeline_header < pipeline_hop);
        assert!(pipeline_hop < user_header);
        assert!(user_header < user_hop);
        // Full timestamps preserved for every hop.
        assert!(result.contains("(2026-08-17T08:11:34.225709+00:00)"));
        assert!(result.contains("(2026-08-17T08:21:19.225709+00:00)"));
        assert!(result.contains("(2026-08-17T09:02:11.225709+00:00)"));
    }

    #[test]
    fn interleaved_tickets_are_run_grouped_not_merged() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        // A-p, B-p, A-p: run-based grouping must produce three blocks, with
        // A's second run under its own header — merging non-consecutive
        // hops would reorder them across tickets.
        push_raw(
            "ws-a",
            "mahbot-A",
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            TransitionOrigin::Pipeline,
            "2026-08-17T08:00:00+00:00",
        );
        push_raw(
            "ws-a",
            "mahbot-B",
            TicketPhase::Analysis,
            TicketPhase::Planning,
            TransitionOrigin::Pipeline,
            "2026-08-17T08:01:00+00:00",
        );
        push_raw(
            "ws-a",
            "mahbot-A",
            TicketPhase::Planning,
            TicketPhase::ReadyForDevelopment,
            TransitionOrigin::Pipeline,
            "2026-08-17T08:02:00+00:00",
        );
        let result = drain("ws-a");
        assert_eq!(result.matches("• mahbot-A").count(), 2);
        assert_eq!(result.matches("• mahbot-B").count(), 1);
        // Chronological hop order preserved across tickets.
        let hop_a1 = result.find("    backlog → analysis").unwrap();
        let hop_b = result.find("    analysis → planning").unwrap();
        let hop_a2 = result.find("    planning → ready_for_development").unwrap();
        assert!(hop_a1 < hop_b && hop_b < hop_a2);
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
