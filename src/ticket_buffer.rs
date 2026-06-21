//! Ticket status accumulation buffer — piggyback non-critical transitions
//! onto critical notifications and user messages.
//!
//! Non-critical transitions (where `transition_ticket` is called with
//! `notify: false`) are buffered here and drained when a critical
//! notification triggers `notify_ticket` or when the user sends a
//! Manager message. This ensures the Manager sees all ticket state changes
//! without requiring a fresh `build_board_context` snapshot on every turn.
//!

use std::collections::{HashMap, VecDeque};
use std::fmt::Write;
use std::sync::{Mutex, OnceLock};
use tracing::warn;

use crate::util::UnwrapPoison;

/// Maximum number of buffered entries per workspace before oldest are dropped.
const PER_WORKSPACE_CAPACITY: usize = 100;

/// A single buffered ticket status transition entry.
#[derive(Clone)]
struct Entry {
    id: String,
    old_status: String,
    new_status: String,
}

/// Global ticket transition buffer, keyed by workspace name.
static TICKET_BUFFER: OnceLock<Mutex<HashMap<String, VecDeque<Entry>>>> = OnceLock::new();

/// Initialize the global ticket buffer. Must be called during startup.
pub fn init_global() {
    TICKET_BUFFER
        .set(Mutex::new(HashMap::new()))
        .map_err(|_| "TICKET_BUFFER already initialized")
        .expect("TICKET_BUFFER already initialized");
}

/// Push a non-critical ticket transition into the buffer.
///
/// If the per-workspace capacity (`PER_WORKSPACE_CAPACITY`) is exceeded,
/// the oldest entry for that workspace is dropped and a warning is emitted.
///
/// If the buffer has not been initialized via [`init_global`], this is a
/// silent no-op with a warning log. This forgiving behavior exists because
/// some call sites (e.g., [`crate::board::BoardStore::supersede_and_create`]) are exercised
/// by tests that don't run the full startup sequence. Under normal operation
/// the buffer is always initialized before any caller reaches it, so a
/// warning here would indicate a genuine startup-order issue.
pub fn push(workspace_name: &str, id: &str, old_status: &str, new_status: &str) {
    let Some(mutex) = TICKET_BUFFER.get() else {
        warn!("ticket_buffer not initialized — call init_global() first");
        return;
    };
    let mut map = mutex.lock().unwrap_poison();
    let deque = match map.get_mut(workspace_name) {
        Some(d) => d,
        None => map.entry(workspace_name.to_string()).or_default(),
    };
    if deque.len() >= PER_WORKSPACE_CAPACITY {
        warn!(
            workspace = %workspace_name,
            capacity = PER_WORKSPACE_CAPACITY,
            "Ticket buffer overflow — dropping oldest entry"
        );
        deque.pop_front();
    }
    deque.push_back(Entry {
        id: id.to_string(),
        old_status: old_status.to_string(),
        new_status: new_status.to_string(),
    });
}

/// Drain all buffered entries for a workspace.
///
/// Returns a formatted string ready for insertion into a notification,
/// or an empty string if no entries are buffered (or if the buffer has
/// not been initialized).
///
/// Format:
/// ```text
/// Ticket updates:
/// • mahbot-42: in_development → in_diagnostics
/// • mahbot-43: in_diagnostics → diagnostics_done
/// ```
pub fn drain(workspace_name: &str) -> String {
    let Some(mutex) = TICKET_BUFFER.get() else {
        return String::new();
    };
    let mut map = mutex.lock().unwrap_poison();
    let Some(entries) = map.remove(workspace_name) else {
        return String::new();
    };
    if entries.is_empty() {
        return String::new();
    }
    let mut out = String::from("Ticket updates:\n");
    for entry in entries {
        let _ = writeln!(
            out,
            "• {}: {} → {}",
            entry.id, entry.old_status, entry.new_status
        );
    }
    out
}

/// Reset all buffers for test isolation.
#[cfg(test)]
pub fn reset() {
    match TICKET_BUFFER.get() {
        Some(mutex) => {
            mutex.lock().unwrap_poison().clear();
        }
        None => {
            let _ = TICKET_BUFFER.set(Mutex::new(HashMap::new()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Test serialization guard — Rust runs tests in parallel by default,
    /// but the global `TICKET_BUFFER` is shared state. Each test that
    /// mutates the buffer must hold this lock.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn push_and_drain_ordered() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        push("ws-a", "mahbot-1", "backlog", "analysis");
        push("ws-a", "mahbot-2", "analysis", "planning");
        push("ws-a", "mahbot-3", "in_development", "in_diagnostics");
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
    fn overflow_drops_oldest() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        for i in 0..101 {
            push("ws-b", &format!("mahbot-{i}"), "backlog", "analysis");
        }
        let result = drain("ws-b");
        // mahbot-0 should be dropped (oldest), mahbot-1 through mahbot-100 retained
        assert!(!result.contains("mahbot-0"));
        assert!(result.contains("mahbot-1"));
        assert!(result.contains("mahbot-100"));
        // header + 100 entries
        assert_eq!(result.lines().count(), 101);
    }

    #[test]
    fn workspace_isolation() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        push("ws-a", "mahbot-1", "backlog", "analysis");
        push(
            "ws-b",
            "mahbot-2",
            "ready_for_development",
            "in_development",
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
        push("ws-a", "mahbot-1", "backlog", "analysis");
        let first = drain("ws-a");
        assert!(!first.is_empty());
        let second = drain("ws-a");
        assert!(second.is_empty());
    }
}
