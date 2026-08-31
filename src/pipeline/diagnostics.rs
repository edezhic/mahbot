//! InDiagnostics phase module — runs discovered diagnostics commands.

use std::fmt::Write;
use std::sync::Arc;

use crate::pipeline::board::Ticket;
use crate::prompt::load_prompt;
use crate::tools::shell::{ShellMode, ShellTool};
use crate::{DiagnosticsCommands, Workspace};

use super::{
    DIAGNOSTICS_ROLE, TicketPhase, TransitionCtx, bounce_to_development,
    comment_and_transition_or_bail, guard_job_phase, reset_phase_attempt, warn,
};

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if guard_job_phase(&ticket.id, TicketPhase::InDiagnostics, &job_id).await {
        return;
    }
    dispatch_diagnostics(ticket, ws, &job_id).await;
}

/// Run diagnostics commands sequentially, collecting output and pass/fail status.
async fn run_diagnostics_commands(diag: &DiagnosticsCommands, ws: &Workspace) -> (String, bool) {
    let mut comment = String::new();
    let mut all_passed = true;
    let mut failed_at: &str = "";

    for (label, cmd_opt) in diag.commands() {
        let Some(cmd) = cmd_opt else {
            continue;
        };

        let mut mark_failed = |comment: &mut String, body: String| {
            let _ = write!(comment, "\n\n{label} ({cmd}):\n");
            comment.push_str(&body);
            all_passed = false;
            failed_at = label;
        };

        let started = std::time::Instant::now();
        match ShellTool::new(ShellMode::Full)
            .execute_with_status(ws, serde_json::json!({"command": cmd}))
            .await
        {
            Ok((_output, Some(0))) => {
                let _ = write!(
                    comment,
                    "\n\n{label} ({cmd}): PASSED in {:.1}s",
                    started.elapsed().as_secs_f64(),
                );
            }
            Ok((output, _exit_code)) => {
                let display = if output.is_empty() {
                    "(no output)".to_string()
                } else {
                    output
                };
                mark_failed(&mut comment, display);
                break;
            }
            Err(e) => {
                mark_failed(&mut comment, e.to_string());
                break;
            }
        }
    }

    if all_passed {
        comment.push_str("\n\n---\n");
        comment.push_str(&load_prompt("pipeline/diagnostics_passed.md"));
    } else {
        let _ = write!(
            comment,
            "\n\n---\n{} {failed_at}",
            load_prompt("pipeline/diagnostics_failed.md"),
        );
    }

    crate::tools::shell::cleanup_agent_spills(DIAGNOSTICS_ROLE);
    (comment.trim_start_matches('\n').to_string(), all_passed)
}

/// Conclude a successful diagnostics run — transition to InReview and delete
/// the phase job (the puller creates the InReview job).
async fn conclude_diagnostics_success(
    ticket: &Ticket,
    job_id: &str,
    comment: &str,
    log_label: &str,
) {
    comment_and_transition_or_bail(
        TransitionCtx::buffered(
            ticket,
            TicketPhase::InDiagnostics,
            TicketPhase::InReview,
            "Diagnostics",
        ),
        DIAGNOSTICS_ROLE,
        comment,
        log_label,
    )
    .await;
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
}

/// Conclude a failed diagnostics run — unified bounce back to development.
async fn conclude_diagnostics_failure(ticket: &Ticket, job_id: &str, comment: &str) {
    bounce_to_development(
        ticket,
        TicketPhase::InDiagnostics,
        "Diagnostics",
        /* drains_siblings */ true,
        DIAGNOSTICS_ROLE,
        comment,
        job_id,
    )
    .await;
}

/// Run diagnostics commands after the engineer completes development.
async fn dispatch_diagnostics(ticket: Arc<Ticket>, ws: Workspace, job_id: &str) {
    // Register a synthetic in-flight roster marker so the phase job's
    // re-dispatch guard blocks while diagnostics execute. Cleared when the
    // job is deleted at phase completion.
    let diag_agent_id = format!("ticket_{}_diagnostics", ticket.id);
    if let Err(e) = crate::jobs::upsert_job_agent(
        &crate::session::store().conn,
        job_id,
        &diag_agent_id,
        crate::jobs::AgentKind::Diagnostics,
        crate::jobs::RowStatus::Launched,
    )
    .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Failed to register diagnostics in-flight marker — diagnostics may re-dispatch",
        );
    }

    match crate::workspace::store().get_diagnostics(&ws.name).await {
        Ok(Some(cmds)) if !cmds.is_empty() => {
            let (comment, all_passed) = run_diagnostics_commands(&cmds, &ws).await;

            if guard_job_phase(&ticket.id, TicketPhase::InDiagnostics, job_id).await {
                return;
            }

            if all_passed {
                conclude_diagnostics_success(
                    &ticket,
                    job_id,
                    &comment,
                    "Diagnostics finished — transitioned ticket",
                )
                .await;
            } else {
                conclude_diagnostics_failure(&ticket, job_id, &comment).await;
            }
        }
        Ok(_) => {
            conclude_diagnostics_success(
                &ticket,
                job_id,
                "No diagnostics commands are configured for this workspace \
                 — diagnostics skipped.",
                "Diagnostics skipped — transitioned ticket",
            )
            .await;
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to load diagnostics for workspace — resetting for a fresh attempt",
            );
            reset_phase_attempt(
                &ticket,
                TicketPhase::InDiagnostics,
                job_id,
                "diagnostics load failure",
                &format!("Could not load diagnostics commands due to a database error: {e}"),
            )
            .await;
        }
    }
}
