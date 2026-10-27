//! The user-activity line at the head of the admin Assistant's messages: where
//! the user is now, shown only when that differs from what that assistant was
//! last shown ([`prepend`]).
//!
//! The reference is a `session_metadata` column of the session, so it survives a
//! restart and goes with the session when that is cleared; a session's first
//! message sets it silently. The line never travels on its own and nothing about
//! activity is logged — it rides the head of a message that is being sent anyway.

use crate::Role;
use crate::prompt::{load_prompt, substitute};
use crate::user_activity::{self, Location, State};

/// Prepend the current activity state when it differs from the one the admin
/// Assistant was last shown, returning the state to record once the message
/// carrying it is persisted.
///
/// Only the admin Assistant is told: a guest assistant, any other role and a
/// machine that cannot tell never get a line.
pub(crate) async fn prepend(
    msg: &str,
    agent_id: &str,
    role: &Role,
    is_admin: bool,
) -> (String, Option<Location>) {
    if !is_admin || !matches!(role, Role::Assistant) {
        return (msg.to_string(), None);
    }
    let Some(current) = user_activity::read() else {
        return (msg.to_string(), None);
    };
    let shown = crate::session::store().user_activity(agent_id).await;
    line(shown.as_deref(), current, msg)
}

/// The rule: the line rides the message only when the location moved since the
/// assistant was last shown one, and the state is recorded whenever it is known
/// — silently, without a line, when nothing was shown yet.
fn line(shown: Option<&str>, current: State, msg: &str) -> (String, Option<Location>) {
    if shown == Some(current.location.as_str()) {
        return (msg.to_string(), None);
    }
    let content = match shown {
        Some(_) => format!("{}\n\n{msg}", render(current)),
        None => msg.to_string(),
    };
    (content, Some(current.location))
}

/// The line itself, as the assistant is sent it.
fn render(state: State) -> String {
    substitute(
        &load_prompt("context/user_activity.md"),
        &[("{{state}}", state.label())],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at_computer() -> State {
        State {
            location: Location::AtComputer,
            by_voice: false,
        }
    }

    fn away() -> State {
        State {
            location: Location::Away,
            by_voice: false,
        }
    }

    /// Nothing was ever shown, so the first message carries no line — it only
    /// establishes what the assistant has seen.
    #[test]
    fn a_first_message_sets_the_reference_silently() {
        assert_eq!(
            line(None, at_computer(), "hi"),
            ("hi".to_string(), Some(Location::AtComputer))
        );
    }

    /// A moved location opens the message with its own line, ahead of everything
    /// else the turn carries.
    #[test]
    fn a_change_rides_the_head_of_the_message() {
        let (content, state) = line(Some("absent"), at_computer(), "hi");
        assert!(
            content.starts_with("<user-activity>active</user-activity>\n\n"),
            "{content}"
        );
        assert!(content.ends_with("hi"), "{content}");
        assert_eq!(state, Some(Location::AtComputer));

        let (content, state) = line(Some("active"), away(), "hi");
        assert!(
            content.starts_with("<user-activity>absent</user-activity>\n\n"),
            "{content}"
        );
        assert!(content.ends_with("hi"), "{content}");
        assert_eq!(state, Some(Location::Away));
    }

    /// A spoken message typed while the assistant already knows the user is
    /// present has nothing new to say.
    #[test]
    fn an_unchanged_state_shows_nothing() {
        let spoken = State {
            location: Location::AtComputer,
            by_voice: true,
        };
        assert_eq!(line(Some("active"), spoken, "hi"), ("hi".to_string(), None));
    }

    /// Speaking aloud is a change worth reporting: it is the one thing that
    /// tells the assistant the user is waiting to hear an answer.
    #[test]
    fn a_spoken_change_says_so() {
        let spoken = State {
            location: Location::AtComputer,
            by_voice: true,
        };
        let (content, state) = line(Some("absent"), spoken, "hi");
        assert!(
            content.starts_with("<user-activity>active, by voice</user-activity>\n\n"),
            "{content}"
        );
        assert_eq!(state, Some(Location::AtComputer));
    }
}
