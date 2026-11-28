//! What one agent invocation works for: the single descriptor every dispatch path builds, in
//! place of the ticket, the parent key and the parent label threaded separately through the
//! session, the prompt, the registry and the message router. A ticket is one kind of work, not a
//! special case: a stage round, an analyze round, a research run, a branch launch, a chain step
//! and an ungrouped turn are the others.
//!
//! Two things are deliberately not universal. Grouping is not: [`WorkUnit::Turn`] is an ordinary
//! turn — a direct chat, a discovery pass, a maintenance pass — and [`WorkUnit::parent`] answers
//! `None` for it, which the Running Agents view reads as a workspace singleton. The board row is
//! not either: a round another agent dispatched inside a ticket carries the id and the title (so
//! the grouping, the log columns and the tool attribution name its ticket) but not the row,
//! because only the dispatcher's own turn renders the ticket block and re-reads the board.
//!
//! The constructors that decide ticket-ness are the only places that decide it: [`WorkUnit::of_ticket`]
//! for the run holding the row, [`WorkUnit::of_parent`] for a round a caller dispatched inside one.
//! Every other kind is its variant, and nothing outside this module inspects the variants to tell a
//! ticket from any other kind of work.

use std::sync::Arc;

use crate::agent::registry::ParentKey;
use crate::pipeline::board::Ticket;

pub(crate) enum WorkUnit {
    /// A board ticket: a stage round, or a round dispatched inside one. `board` is the row when
    /// the dispatcher holds it — the prompt's ticket block, the ticket-scoped log columns and the
    /// tool attribution read it — shared, never copied: one stage's roster of agents reads one
    /// row. A round another agent dispatched inside a ticket carries the id and title but not the
    /// row.
    Ticket {
        id: String,
        title: Option<String>,
        board: Option<Arc<Ticket>>,
    },
    /// A parallel analyze round; `label` is the question it answers.
    AnalyzeRound { id: String, label: String },
    /// A research run; `label` is its question.
    Research { id: String, label: String },
    /// A branch launch (a worker or a custom role); `id` is the launch's agent id.
    Launch { id: String, label: String },
    /// One step of a chain; `label` is the chain's original task.
    Chain { id: String, label: String },
    /// An ordinary turn — a direct chat, a discovery pass, a maintenance pass: nothing grouped.
    Turn,
}

impl WorkUnit {
    /// The board row, when the dispatcher holds it — the prompt's ticket block and every reader
    /// that needs the ticket's own fields. A round dispatched inside a ticket has none.
    #[must_use]
    pub(crate) fn ticket(&self) -> Option<&Ticket> {
        match self {
            Self::Ticket { board, .. } => board.as_deref(),
            _ => None,
        }
    }

    /// The ticket's id, whether or not the row is held — the ticket-scoped log columns and the
    /// tool attribution read it. `None` for every other kind of work, whose id is not a ticket.
    #[must_use]
    pub(crate) fn ticket_id(&self) -> Option<&str> {
        match self {
            Self::Ticket { id, .. } => Some(id),
            _ => None,
        }
    }

    /// The presentational label of the work (ticket title, analyze or research question, launch
    /// label, chain task) — the Running Agents group header. `None` for a plain turn, and for a
    /// dispatched round whose caller carried no label: an empty label is no label.
    #[must_use]
    pub(crate) fn label(&self) -> Option<&str> {
        let label = match self {
            Self::Ticket { title, .. } => title.as_deref(),
            Self::AnalyzeRound { label, .. }
            | Self::Research { label, .. }
            | Self::Launch { label, .. }
            | Self::Chain { label, .. } => Some(label.as_str()),
            Self::Turn => None,
        }?;
        (!label.is_empty()).then_some(label)
    }

    /// The DIRECT PARENT INVOCATION grouping key the registry publishes and group cancellation
    /// matches; `None` for a plain turn.
    #[must_use]
    pub(crate) fn parent(&self) -> Option<ParentKey> {
        match self {
            Self::Ticket { id, .. } => Some(ParentKey::Ticket(id.clone())),
            Self::AnalyzeRound { id, .. } => Some(ParentKey::AnalyzeRound(id.clone())),
            Self::Research { id, .. } => Some(ParentKey::Research(id.clone())),
            Self::Launch { id, .. } => Some(ParentKey::Launch(id.clone())),
            Self::Chain { id, .. } => Some(ParentKey::Chain(id.clone())),
            Self::Turn => None,
        }
    }

    /// The work a dispatcher holding the ticket's row hands to an agent. The row is shared, not
    /// cloned: the dispatcher already holds an `Arc`, and a stage's roster of agents reads it
    /// without a copy each.
    pub(crate) fn of_ticket(ticket: Arc<Ticket>) -> Self {
        Self::Ticket {
            id: ticket.id.clone(),
            title: Some(ticket.title.clone()),
            board: Some(ticket),
        }
    }

    /// The work a dispatched round inherits from its caller: the caller's grouping key and label,
    /// carried into this run's own descriptor. A ticket-parented round gets the id and title but
    /// not the board row — its caller holds the row, this run only names it. A caller with no
    /// label leaves the label empty, which [`Self::label`] reads as none; no caller grouping at
    /// all is a plain turn.
    pub(crate) fn of_parent(parent: Option<ParentKey>, label: Option<String>) -> Self {
        match parent {
            Some(ParentKey::Ticket(id)) => Self::Ticket {
                id,
                title: label,
                board: None,
            },
            Some(ParentKey::AnalyzeRound(id)) => Self::AnalyzeRound {
                id,
                label: label.unwrap_or_default(),
            },
            Some(ParentKey::Research(id)) => Self::Research {
                id,
                label: label.unwrap_or_default(),
            },
            Some(ParentKey::Launch(id)) => Self::Launch {
                id,
                label: label.unwrap_or_default(),
            },
            Some(ParentKey::Chain(id)) => Self::Chain {
                id,
                label: label.unwrap_or_default(),
            },
            None => Self::Turn,
        }
    }
}

/// Whether a run writes the workspace folder — the one line between reading and writing that the
/// folder's single-writer hold and the workspace freeze both draw. A ticket stage, a launch and a
/// chain step write. A separately launched analysis ([`WorkUnit::AnalyzeRound`]) and a discovery
/// pass ([`WorkUnit::Turn`] as the Discovery role) read the folder whatever roles they run, and so
/// do a research run's analysts and its synthesis worker, so none of them holds the folder or stops
/// on a freeze. The line is the work, not the role: the pipeline's analysis stage runs an analyst
/// under a ticket and does write, and a research run's cleanup step deletes the run's folder, so
/// that one writes too. The Manager's own triage writes nothing either, and is exempt so the notice
/// explaining the pause stays deliverable. An unknown role string writes wherever the role is
/// consulted; a launched analysis never consults it, because the round is a reading round whatever
/// its members do.
#[must_use]
pub(crate) fn writes_folder(parent: Option<&ParentKey>, role: &str) -> bool {
    if role == crate::Role::Manager.as_str() {
        return false;
    }
    // The one role test both arms share, so a reading role is named once.
    let reads = |reading: &[crate::Role]| {
        matches!(
            role.parse::<crate::Role>(),
            Ok(parsed) if reading.contains(&parsed)
        )
    };
    match parent {
        Some(ParentKey::AnalyzeRound(_)) => false,
        // A research run's analysts and synthesis worker read; its cleanup step, the one run under
        // this parent that deletes the run's folder, writes — and so does an unknown role string.
        Some(ParentKey::Research(_)) => !reads(&[crate::Role::Analyst, crate::Role::Worker]),
        Some(_) => true,
        None => !reads(&[crate::Role::Discovery]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal board row — only the fields the descriptor reads are meaningful here.
    fn ticket(id: &str, title: &str) -> Ticket {
        crate::util::test::TicketFixture::new(
            id,
            crate::pipeline::board::TicketPhase::InDevelopment,
        )
        .title(title)
        .build()
    }

    /// The dispatcher's own round holds the board row; a round it dispatched carries the same id
    /// and title but no row, so only the dispatcher renders the ticket block.
    #[test]
    fn a_ticket_round_holds_its_row_and_a_dispatched_round_only_names_it() {
        let own = WorkUnit::of_ticket(Arc::new(ticket("t1", "Fix the thing")));
        assert_eq!(own.ticket().map(|t| t.id.as_str()), Some("t1"));
        assert_eq!(own.ticket_id(), Some("t1"));
        assert_eq!(own.label(), Some("Fix the thing"));
        assert_eq!(own.parent(), Some(ParentKey::Ticket("t1".to_string())));

        let dispatched = WorkUnit::of_parent(
            Some(ParentKey::Ticket("t1".to_string())),
            Some("Fix the thing".to_string()),
        );
        assert!(dispatched.ticket().is_none(), "the caller holds the row");
        assert_eq!(dispatched.ticket_id(), Some("t1"));
        assert_eq!(dispatched.label(), Some("Fix the thing"));
        assert_eq!(
            dispatched.parent(),
            Some(ParentKey::Ticket("t1".to_string()))
        );
    }

    /// Every grouped kind names the same parent key and label it was built from, so the registry
    /// card and the grouping are unchanged by the consolidation.
    #[test]
    fn every_grouped_kind_keeps_its_parent_key_and_label() {
        let cases = [
            (
                WorkUnit::AnalyzeRound {
                    id: "r1".to_string(),
                    label: "which crate?".to_string(),
                },
                ParentKey::AnalyzeRound("r1".to_string()),
                "which crate?",
            ),
            (
                WorkUnit::Research {
                    id: "j1".to_string(),
                    label: "how does it work?".to_string(),
                },
                ParentKey::Research("j1".to_string()),
                "how does it work?",
            ),
            (
                WorkUnit::Launch {
                    id: "w1".to_string(),
                    label: "worker on main".to_string(),
                },
                ParentKey::Launch("w1".to_string()),
                "worker on main",
            ),
            (
                WorkUnit::Chain {
                    id: "c1".to_string(),
                    label: "the chain's task".to_string(),
                },
                ParentKey::Chain("c1".to_string()),
                "the chain's task",
            ),
        ];
        for (unit, parent, label) in cases {
            assert_eq!(unit.parent(), Some(parent.clone()));
            assert_eq!(unit.label(), Some(label));
            assert!(
                unit.ticket().is_none() && unit.ticket_id().is_none(),
                "{parent:?} is not a ticket"
            );
        }
    }

    /// An ordinary turn is nothing grouped and nothing ticketed.
    #[test]
    fn a_plain_turn_is_neither_grouped_nor_a_ticket() {
        let turn = WorkUnit::of_parent(None, None);
        assert!(matches!(turn, WorkUnit::Turn));
        assert!(turn.parent().is_none());
        assert!(turn.label().is_none());
        assert!(turn.ticket().is_none());
        assert!(turn.ticket_id().is_none());
    }

    /// The line between reading and writing is the work, not the role: the pipeline's analysis
    /// stage runs an analyst under a ticket and writes, while the same role in a separately
    /// launched round reads. A discovery pass and the Manager's triage write nothing; the
    /// maintainer writes.
    #[test]
    fn the_writer_line_is_drawn_by_the_work_not_the_role() {
        let ticket = Some(ParentKey::Ticket("t1".to_string()));
        let round = Some(ParentKey::AnalyzeRound("r1".to_string()));
        let run = Some(ParentKey::Research("run1".to_string()));
        assert!(writes_folder(
            ticket.as_ref(),
            crate::Role::Analyst.as_str()
        ));
        assert!(!writes_folder(
            round.as_ref(),
            crate::Role::Analyst.as_str()
        ));
        assert!(!writes_folder(run.as_ref(), crate::Role::Worker.as_str()));
        assert!(
            writes_folder(run.as_ref(), crate::Role::Sanitation.as_str()),
            "a research run's cleanup deletes the run's folder: it writes"
        );
        assert!(!writes_folder(None, crate::Role::Discovery.as_str()));
        assert!(writes_folder(None, crate::Role::Maintainer.as_str()));
        assert!(!writes_folder(None, crate::Role::Manager.as_str()));
        assert!(writes_folder(None, "no-such-role"), "the rule fails closed");
    }

    /// A caller that carried no label leaves none: an empty string would win the Running Agents
    /// group header over a later real label.
    #[test]
    fn a_caller_with_no_label_leaves_no_label() {
        let dispatched = WorkUnit::of_parent(Some(ParentKey::Chain("c1".to_string())), None);
        assert_eq!(
            dispatched.parent(),
            Some(ParentKey::Chain("c1".to_string()))
        );
        assert!(dispatched.label().is_none());
    }
}
