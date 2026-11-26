//! InDevelopment phase module — the engineer implements the ticket.

use std::fmt::Write;
use std::sync::Arc;

use crate::pipeline::board::Ticket;
use crate::prompt::{load_prompt, substitute};
use crate::{Agent, Role, Workspace};

use super::{
    StageRunKind, TicketPhase, TransitionCtx, comment_and_transition_or_bail, guard_job_phase,
    guard_stage, info, notice, pause_freezing, run_stage_agent, sync_phase_job_task, warn,
};

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if guard_job_phase(&ticket.id, TicketPhase::InDevelopment, &job_id).await {
        return;
    }
    dispatch_engineer(ticket, ws, &job_id).await;
}

/// Build the engineer's work prompt: comments that arrived after the engineer's last comment
/// are bounce feedback to address, otherwise the plain implement prompt.
///
/// On a first dispatch there is no prior engineer comment, so pre-existing comments are
/// baseline context already injected via the `<current-ticket>` block.
fn engineer_work_message(ticket: &Ticket) -> String {
    let feedback: Vec<&str> = ticket
        .comments
        .iter()
        .rposition(|c| c.role == Role::Engineer.as_str())
        .map(|i| {
            ticket
                .comments
                .iter()
                .skip(i + 1)
                .map(|c| c.content.as_str())
                .collect()
        })
        .unwrap_or_default();

    if feedback.is_empty() {
        load_prompt("implement.md")
    } else {
        substitute(
            &load_prompt("pipeline/bounce_feedback.md"),
            &[("{{feedback}}", &feedback.join("\n---\n"))],
        )
    }
}

async fn dispatch_engineer(ticket: Arc<Ticket>, ws: Workspace, job_id: &str) {
    let message = engineer_work_message(&ticket);
    let conn = &crate::session::store().conn;
    sync_phase_job_task(conn, &ticket.id, job_id, &message).await;
    run_stage_agent(&ticket, &ws, job_id, &message, StageRunKind::Engineer).await;
}

async fn engineer_comment_text(agent: &mut Agent, raw: &str) -> String {
    let ticket_id = agent
        .ticket
        .as_ref()
        .map_or_else(|| "?".to_string(), |t| t.id.clone());
    let policy = crate::retry::RetryPolicy::comment();
    let extraction_prompt = load_prompt("extraction/engineer.md");
    let summary = match agent
        .extract_verdict::<crate::EngineerSummary>(&extraction_prompt, None, Some(&policy))
        .await
    {
        Ok(summary) => summary,
        Err(e) => {
            warn!(
                ticket = %ticket_id,
                error = %e,
                "Engineer summary extraction failed — using raw response for ticket comment"
            );
            return raw.to_string();
        }
    };
    let items: Vec<&str> = summary
        .items
        .iter()
        .map(String::as_str)
        .filter(|s| !s.trim().is_empty())
        .collect();
    if items.is_empty() {
        warn!(
            ticket = %ticket_id,
            "Engineer summary extraction returned no usable items — using raw response for ticket comment"
        );
        return raw.to_string();
    }
    let mut out = String::new();
    for item in items {
        let _ = write!(out, "\n- {}", item.replace('\n', " "));
    }
    let reported: Vec<&str> = summary
        .reported
        .iter()
        .map(String::as_str)
        .filter(|s| !s.trim().is_empty())
        .collect();
    if !reported.is_empty() {
        out.push_str("\n\n### Reported");
        for entry in reported {
            let _ = write!(out, "\n- {}", entry.replace('\n', " "));
        }
    }
    let synopsis = summary.summary.as_deref().unwrap_or("").trim();
    if !synopsis.is_empty() {
        let _ = write!(out, "\n\n### Summary\n{synopsis}");
    }
    crate::util::failure_detail(&out, "engineer summary comment")
}

/// Shared engineer post-run tail: phase/drain guards, failure handling, pause,
/// transition, and job terminalization.
pub(crate) async fn finalize_engineer_stage(
    ticket: &Ticket,
    agent: &mut Agent,
    response: Option<&str>,
    job_id: &str,
    paused: bool,
) {
    if guard_stage(
        &ticket.id,
        TicketPhase::InDevelopment,
        "Engineer",
        response,
        job_id,
    )
    .await
    {
        return;
    }

    // Success: transition to Verification; the puller creates its phase job next tick.
    if let Some(text) = response {
        let comment_text = engineer_comment_text(agent, text).await;
        comment_and_transition_or_bail(
            TransitionCtx::buffered(
                ticket,
                TicketPhase::InDevelopment,
                TicketPhase::Verification,
                "Engineer",
                Role::Engineer.as_str(),
            ),
            Role::Engineer.as_str(),
            &comment_text,
            "Engineer finished — transitioned ticket",
        )
        .await;
        let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
        return;
    }

    // Past the guards above, response None here is a real failure — classify,
    // pause, and freeze/fail.
    handle_engineer_failure(ticket, agent, job_id, paused).await;
}

async fn handle_engineer_failure(ticket: &Ticket, agent: &Agent, job_id: &str, paused: bool) {
    // A workspace-pause (strict freeze) is NOT a failure — leave the job for the unpause
    // re-drive. Uses the immutable bail-time snapshot, so it survives an unpause race.
    if paused {
        pause_freezing(ticket, job_id).await;
        return;
    }

    // A code-driven cancellation — re-dispatch, replacement, supersede, or the GUI cancel —
    // leaves the ticket for the replacement run.
    if agent.is_cancelled() && !crate::shutdown::aborting() {
        info!(ticket = %ticket.id, "Engineer run interrupted by a code-driven cancellation — leaving the ticket for the replacement run");
        return;
    }

    // Hard failure: the pause and its notice are one commit. The ticket itself gets no comment —
    // ticket history carries stage conclusions only. This death has its own notice, so it gains
    // no second one.
    let terms = match agent.failure.clone() {
        Some(error) => notice::Terms::new(
            agent.failure_class,
            agent.failure_attempts,
            crate::util::failure_detail(&error, "engineer failure"),
        ),
        None => notice::Terms::plain("the engineer produced no output"),
    };
    notice::stop_workspace(
        ticket,
        notice::Stage::Development,
        notice::Cause::Engineer { terms },
        job_id,
    )
    .await;

    if crate::shutdown::aborting() {
        info!(
            ticket = %ticket.id,
            "Engineer failure cut short by shutdown/drain after the pause — job stays launched for boot resume",
        );
        return;
    }

    info!(
        ticket = %ticket.id,
        "Engineer hard failure — workspace paused, ticket reset for a fresh development attempt"
    );
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
}
