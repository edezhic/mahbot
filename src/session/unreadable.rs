//! The record of a message that arrived and could not be read.
//!
//! When a recognised person sends something the product cannot read, two things are owed: the
//! fixed line the sender received ([`NOTICE`]) has to land in the conversation's transcript, so the
//! owner sees that something arrived and was not read, and the fact — when, who, of what kind,
//! never any content — has to reach that conversation's next turn. Both wait for that turn, which
//! is the conversation's only writer: a row appended from the listener would sit outside the turn's
//! own append, so a live session's `history[..persisted_len]` would no longer mirror its rows in
//! `id` order, and a trailing `system` row would make an interrupted turn look finished to the
//! dead-session recovery. The turn writes the line as a row of its own, ahead of its message, in
//! the same transaction as that message. The row is a `system` row — the product speaking, not the
//! person, the same shape the compaction summary already takes — and it needs no answer. A
//! conversation that never speaks again keeps its chat line and its warning but writes no row: the
//! turn is the one place a row can be added without breaking either guarantee above.
//!
//! The queue is in memory, like the product's other show-once bridges: it survives until the turn
//! that carries it, and a restart loses a fact that no turn was ever told about. It is bounded per
//! conversation and forgotten when the conversation ends, like the announced-media-failure notes.

use std::collections::{HashMap, VecDeque};
use std::sync::{LazyLock, Mutex};

use crate::prompt::{load_prompt, substitute};
use crate::util::UnwrapPoison;

/// The one line a sender whose message could not be read receives, and the one line the transcript
/// keeps. Fixed and the same for every unreadable kind: it names no kind, promises no retry, and
/// never repeats what the message said.
pub(crate) const NOTICE: &str = "MahBot could not read that message.";

/// How many facts one conversation remembers. Past this the oldest is forgotten.
const PENDING_PER_CONVERSATION: usize = 64;

/// One unreadable message as the conversation's next turn is told about it.
struct Fact {
    /// When it arrived, rendered by [`crate::session::render_timestamp`].
    at: String,
    /// The account that sent it.
    person: String,
    /// What kind of thing arrived — never any of its content.
    kind: &'static str,
}

/// The facts queued per conversation, keyed by agent id, in arrival order.
static PENDING: LazyLock<Mutex<HashMap<String, VecDeque<Fact>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// What a conversation's next turn carries from here.
pub(crate) struct Pending {
    /// One transcript row per unreadable message, to be written ahead of the turn's message.
    pub(crate) lines: Vec<crate::ChatMessage>,
    /// The note to attach to that message; empty when nothing is queued.
    pub(crate) note: String,
}

/// Whether this row is the fixed line an unreadable message left behind.
pub(crate) fn is_notice(message: &crate::ChatMessage) -> bool {
    message.role == crate::ChatRole::System && message.content == NOTICE
}

/// Queue a message that could not be read for `agent_id`'s next turn. Nothing is written here: the
/// turn owns the conversation's history (see the module header).
pub(crate) fn record(agent_id: &str, person: &str, kind: &'static str) {
    let mut pending = PENDING.lock().unwrap_poison();
    let facts = pending.entry(agent_id.to_string()).or_default();
    facts.push_back(Fact {
        at: crate::session::render_timestamp(),
        person: person.to_string(),
        kind,
    });
    if facts.len() > PENDING_PER_CONVERSATION {
        facts.pop_front();
    }
}

/// Forget what `agent_id`'s conversation has been told but no turn has carried yet: the session
/// store calls it when a conversation's session ends (deleted or reset, including the bulk purge),
/// so a fact never rides a new conversation's first turn.
pub(crate) fn forget(agent_id: &str) {
    PENDING.lock().unwrap_poison().remove(agent_id);
}

/// The lines a conversation is owed but its next turn has not written yet, oldest first. The
/// Sessions page shows them, so the owner does not have to wait for a turn to see that something
/// arrived and was not read; the turn replaces them with their rows.
pub(crate) fn pending_lines(agent_id: &str) -> Vec<crate::ChatMessage> {
    PENDING
        .lock()
        .unwrap_poison()
        .get(agent_id)
        .map(|facts| notice_lines(facts.len()))
        .unwrap_or_default()
}

/// One transcript row per unreadable message, for the turn that writes them.
fn notice_lines(count: usize) -> Vec<crate::ChatMessage> {
    vec![crate::ChatMessage::system(NOTICE); count]
}

/// Take what `agent_id`'s conversation has not been told yet, oldest first: one transcript row per
/// fact, and the note the turn's message carries.
pub(crate) fn drain(agent_id: &str) -> Pending {
    let Some(facts) = PENDING.lock().unwrap_poison().remove(agent_id) else {
        return Pending {
            lines: Vec::new(),
            note: String::new(),
        };
    };
    let asset = load_prompt("context/unreadable_message.md");
    let note = facts
        .iter()
        .map(|fact| {
            substitute(
                &asset,
                &[
                    ("{{at}}", fact.at.as_str()),
                    ("{{person}}", fact.person.as_str()),
                    ("{{kind}}", fact.kind),
                ],
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    Pending {
        lines: notice_lines(facts.len()),
        note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn unique_key() -> String {
        format!("unreadable{}", TEST_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// A fact is told once: it rides the next turn and the one after carries nothing.
    #[test]
    fn a_recorded_fact_rides_the_next_turn_exactly_once() {
        let agent_id = unique_key();
        record(&agent_id, "Ada", "a picture");

        let pending = drain(&agent_id);
        assert!(
            pending.note.starts_with("<unreadable-message>"),
            "got {:?}",
            pending.note
        );
        assert!(
            pending.note.contains("Ada sent a picture"),
            "got {:?}",
            pending.note
        );
        assert_eq!(pending.lines.len(), 1, "one transcript row per message");
        assert_eq!(pending.lines[0].content, NOTICE);
        assert_eq!(pending.lines[0].role, crate::ChatRole::System);

        let pending = drain(&agent_id);
        assert!(pending.note.is_empty(), "a fact is told once");
        assert!(pending.lines.is_empty());
    }

    /// The queue is per conversation: one agent id's facts never reach another's turn.
    #[test]
    fn facts_never_cross_conversations() {
        let first = unique_key();
        let second = unique_key();
        record(&first, "Ada", "a picture");
        record(&second, "Bo", "a voice message");

        let pending = drain(&first);
        assert!(
            pending.note.contains("Ada sent a picture"),
            "{:?}",
            pending.note
        );
        assert!(!pending.note.contains("Bo"), "{:?}", pending.note);

        let pending = drain(&second);
        assert!(
            pending.note.contains("Bo sent a voice message"),
            "{:?}",
            pending.note
        );
        assert!(!pending.note.contains("Ada"), "{:?}", pending.note);
    }

    /// Nothing queued is the identity — a conversation that never missed a message pays nothing.
    #[test]
    fn nothing_queued_carries_nothing() {
        let pending = drain("unreadable_never_queued");
        assert!(pending.note.is_empty());
        assert!(pending.lines.is_empty());
    }

    /// The queue is bounded: the oldest facts fall off rather than accumulating forever.
    #[test]
    fn a_conversation_remembers_a_bounded_number_of_facts() {
        let agent_id = unique_key();
        for index in 0..=PENDING_PER_CONVERSATION {
            record(&agent_id, &format!("person{index}"), "a picture");
        }

        let pending = drain(&agent_id);
        assert_eq!(
            pending.note.matches("<unreadable-message>").count(),
            PENDING_PER_CONVERSATION,
            "the queue must stay bounded"
        );
        assert_eq!(pending.lines.len(), PENDING_PER_CONVERSATION);
        assert!(
            !pending.note.contains("person0 "),
            "the oldest fact fell off"
        );
        assert!(pending.note.contains("person1 "), "the rest are kept");
    }

    /// A conversation that ends takes what no turn carried with it, so nothing rides the next
    /// conversation's first turn.
    #[test]
    fn a_forgotten_conversation_carries_nothing_into_its_next_one() {
        let agent_id = unique_key();
        record(&agent_id, "Ada", "a picture");
        forget(&agent_id);
        assert!(drain(&agent_id).note.is_empty());
    }

    /// The Sessions page is offered the line before the turn that writes it, and nothing after.
    #[test]
    fn a_pending_line_is_offered_until_the_turn_takes_it() {
        let agent_id = unique_key();
        record(&agent_id, "Ada", "a picture");
        let lines = pending_lines(&agent_id);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].content, NOTICE);
        assert_eq!(lines[0].role, crate::ChatRole::System);

        drain(&agent_id);
        assert!(pending_lines(&agent_id).is_empty());
    }
}
