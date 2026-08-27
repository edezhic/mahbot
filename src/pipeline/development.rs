//! InDevelopment phase module — the engineer implements the ticket.

use std::fmt::Write;
use std::sync::Arc;

use crate::pipeline::board::Ticket;
use crate::prompt::{load_prompt, substitute};
use crate::{Agent, Role, Workspace};

use super::{
    RETRY_EXHAUSTION_MARKER, SYSTEM_ROLE, StageRunKind, TicketPhase, TransitionCtx, board,
    comment_and_transition_or_bail, guard_stage, info, is_ticket_in_phase, manager_agent_id,
    message_router, pause_freezing, pause_workspace_on_failure, paused_workspace_sentence,
    run_stage_agent, sync_phase_job_task, warn,
};

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if !is_ticket_in_phase(&ticket.id, TicketPhase::InDevelopment).await {
        let _ = crate::jobs::complete_ticket_job(&crate::session::store().conn, &job_id).await;
        return;
    }
    dispatch_engineer(ticket, ws, &job_id).await;
}

/// Build the engineer's work prompt: comments that arrived after the
/// engineer's last comment are bounce feedback to address, otherwise fall
/// back to the plain implement prompt.
///
/// On a first InDevelopment dispatch there is no prior engineer comment, so
/// there is no post-round feedback — pre-existing comments are baseline
/// context already injected via the `<current-ticket>` block.
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

/// Dispatch the Engineer agent to implement the ticket on the phase job.
async fn dispatch_engineer(ticket: Arc<Ticket>, ws: Workspace, job_id: &str) {
    let message = engineer_work_message(&ticket);
    let conn = &crate::session::store().conn;
    sync_phase_job_task(conn, job_id, &message).await;
    run_stage_agent(&ticket, &ws, job_id, &message, StageRunKind::Engineer).await;
}

/// Extract a concise structured summary of the engineer's work for the ticket
/// comment.
async fn engineer_comment_text(agent: &Agent, raw: &str) -> String {
    let ticket_id = agent.ticket.as_ref().map_or("?", |t| t.id.as_str());
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
            return crate::util::scrub_credentials(raw);
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
        return crate::util::scrub_credentials(raw);
    }
    let mut out = String::new();
    for item in items {
        let _ = write!(out, "\n- {}", item.replace('\n', " "));
    }
    let synopsis = summary.summary.as_deref().unwrap_or("").trim();
    if !synopsis.is_empty() {
        let _ = write!(out, "\n\n### Summary\n{synopsis}");
    }
    crate::util::truncate_sandwich(
        &crate::util::scrub_credentials(&out),
        crate::util::FAILURE_DETAIL_CAP,
        "engineer summary comment",
    )
}

/// Build the failure comment for a failed Engineer run.
fn engineer_failure_comment(shutdown: bool, error: Option<&str>) -> String {
    if shutdown {
        return "Engineer failed: service shutting down — the run was interrupted \
                by process shutdown."
            .to_string();
    }
    let Some(detail) = error else {
        return load_prompt("pipeline/engineer_failed.md");
    };
    let detail = crate::util::scrub_credentials(detail);
    let detail = crate::util::truncate_sandwich(
        &detail,
        crate::util::FAILURE_DETAIL_CAP,
        "engineer failure",
    );
    if detail.contains(RETRY_EXHAUSTION_MARKER) {
        format!("Engineer failed: LLM provider retry exhaustion.\n\n{detail}")
    } else {
        format!("Engineer failed.\n\n{detail}")
    }
}

/// Notify the Manager that a workspace was paused because of an engineer hard
/// failure.
fn notify_engineer_pause(ws: &Workspace, failure_details: &str, paused: bool) {
    let workspace_status = if paused {
        format!("The workspace is paused — {}.", paused_workspace_sentence())
    } else {
        "The workspace was not paused — remaining queued tickets may still be claimed.".to_string()
    };
    let warning = substitute(
        &load_prompt("pipeline/engineer_pause_notification.md"),
        &[
            ("{{failure_details}}", failure_details),
            ("{{workspace_status}}", &workspace_status),
        ],
    );
    let agent_id = manager_agent_id(&ws.name);
    message_router::route(
        &agent_id,
        message_router::AgentJob {
            content: warning,
            workspace_name: ws.name.clone(),
            user_name: "system".to_string(),
            channel: String::new(),
            kind: message_router::MessageKind::UserMessage,
            role: Role::Manager,
            reply_target: None,
            pending_job_id: None,
        },
    );
}

/// Shared engineer post-run tail: phase/drain guards, failure handling, pause,
/// transition, and job terminalization.
pub(crate) async fn finalize_engineer_stage(
    ticket: &Ticket,
    agent: &Agent,
    response: Option<&str>,
    job_id: &str,
    ws: &Workspace,
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

    // Success path: engineer produced output — transition to InDiagnostics.
    // The puller creates the InDiagnostics phase job on the next tick.
    if let Some(text) = response {
        let comment_text = engineer_comment_text(agent, text).await;
        comment_and_transition_or_bail(
            TransitionCtx::buffered(
                ticket,
                TicketPhase::InDevelopment,
                TicketPhase::InDiagnostics,
                "Engineer",
            ),
            Role::Engineer.as_str(),
            &comment_text,
            "Engineer finished — transitioned ticket",
        )
        .await;
        let _ = crate::jobs::complete_ticket_job(&crate::session::store().conn, job_id).await;
        return;
    }

    // Past the guards above, response None here is a real failure — classify,
    // pause, and freeze/fail.
    handle_engineer_failure(ticket, agent, job_id, ws, paused).await;
}

/// Handle the engineer failure tail (response `None` past the guards).
async fn handle_engineer_failure(
    ticket: &Ticket,
    agent: &Agent,
    job_id: &str,
    ws: &Workspace,
    paused: bool,
) {
    // A workspace-pause (strict freeze) is NOT a failure — leave the job in
    // place for the unpause re-drive. Uses the immutable bail-time snapshot
    // captured on the agent, so it survives a workspace-unpause race.
    if paused {
        pause_freezing(ticket, job_id).await;
        return;
    }

    // A code-driven (internal) cancellation — re-dispatch, register
    // replacement, phase transition/supersede, or the GUI cancel (whose
    // authoritative Cancelled transition is handled by the GUI, not here) —
    // leaves the ticket for the replacement run.
    if agent.is_cancelled() && !crate::shutdown::aborting() {
        info!(ticket = %ticket.id, "Engineer run interrupted by a code-driven cancellation — leaving the ticket for the replacement run");
        return;
    }

    let pause_occurred = pause_workspace_on_failure(ticket, "engineer agent failure").await;

    if crate::shutdown::aborting() {
        info!(
            ticket = %ticket.id,
            "Engineer failure cut short by shutdown/drain after the pause — job stays launched for boot resume",
        );
        return;
    }

    let failure_comment = engineer_failure_comment(
        crate::shutdown::shutdown_token().is_cancelled(),
        agent.failure.as_deref(),
    );
    let conn = &crate::session::store().conn;

    // Hard failure: pause is already committed. Leave an explanatory
    // comment and delete the phase job; the puller creates a FRESH
    // InDevelopment attempt on unpause. The pause is silent in the ticket
    // comment — notify_engineer_pause reports it separately.
    if let Err(e) = board()
        .add_comment(&ticket.id, SYSTEM_ROLE, &failure_comment)
        .await
    {
        warn!(ticket = %ticket.id, error = %e, "Failed to comment engineer hard failure");
    }
    let workspace_paused = ws.paused || pause_occurred;
    notify_engineer_pause(ws, &failure_comment, workspace_paused);
    info!(
        ticket = %ticket.id,
        "Engineer hard failure — workspace paused, ticket reset for a fresh development attempt"
    );
    let _ = crate::jobs::complete_ticket_job(conn, job_id).await;
}
