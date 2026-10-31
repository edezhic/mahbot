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

    fn spoken() -> State {
        State {
            location: Location::AtComputer,
            by_voice: true,
        }
    }

    /// The one rule at every entry point: the line rides the message only when
    /// the location moved since the assistant was last shown one, ahead of
    /// everything else the turn carries and saying so when speech is what put
    /// the user there; the state is recorded whenever it is known — silently,
    /// without a line, on the first message, and not at all when it did not
    /// change.
    #[test]
    fn the_line_rides_a_change_and_the_reference_always_moves() {
        type Case = (
            Option<&'static str>,
            State,
            Option<&'static str>,
            Option<Location>,
        );
        let cases: [Case; 5] = [
            // Nothing was ever shown: only what the assistant has seen is set.
            (None, at_computer(), None, Some(Location::AtComputer)),
            (
                Some("absent"),
                at_computer(),
                Some("<user-activity>active</user-activity>\n\n"),
                Some(Location::AtComputer),
            ),
            (
                Some("active"),
                away(),
                Some("<user-activity>absent</user-activity>\n\n"),
                Some(Location::Away),
            ),
            (
                Some("absent"),
                spoken(),
                Some("<user-activity>active, by voice</user-activity>\n\n"),
                Some(Location::AtComputer),
            ),
            // A spoken message while the assistant already knows the user is
            // present has nothing new to say.
            (Some("active"), spoken(), None, None),
        ];
        for (shown, current, head, recorded) in cases {
            let (content, state) = line(shown, current, "hi");
            match head {
                Some(head) => assert!(
                    content.starts_with(head) && content.ends_with("hi"),
                    "a change opens the message: {content:?}"
                ),
                None => assert_eq!(content, "hi", "no line for this state"),
            }
            assert_eq!(state, recorded);
        }
    }
}
