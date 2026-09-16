//! InSanitation phase module — inspect new/untracked files before commit.

use std::sync::Arc;

use crate::pipeline::board::Ticket;
use crate::prompt::{load_prompt, substitute};
use crate::{Agent, Role, Workspace};

use super::{
    BoardStore, FinalizeOutcome, StageRunKind, TicketPhase, TransitionCtx, board,
    bounce_to_development, clear_implementation_roster, determine_notify_policy, error,
    guard_job_phase, guard_stage, info, list_new_or_untracked_files, notify_manager_system,
    pause_freezing, pause_status_sentence, reset_phase_attempt, run_git_status, run_stage_agent,
    sync_phase_job_task, warn, with_comment_and_transition,
};

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if guard_job_phase(&ticket.id, TicketPhase::InSanitation, &job_id).await {
        return;
    }
    dispatch_sanitation(ticket, ws, &job_id).await;
}

/// Transition a ticket to Done without writing a comment row, and delete any
/// launched phase jobs. Git-related Done cases (no git, not a repo, clean
/// tree) use this so the fallback is silent in ticket history; `reason`
/// preserves the specific cause in the log.
async fn transition_ticket_to_done_no_comment(ticket: &Ticket, source: TicketPhase, reason: &str) {
    let notify_policy = determine_notify_policy(&ticket.workspace_name, &ticket.id).await;
    if matches!(
        with_comment_and_transition(
            TransitionCtx::new(
                ticket,
                source,
                TicketPhase::Done,
                notify_policy,
                "Finalize",
                Role::Sanitation.as_str(),
            ),
            async |_tx| Ok(()),
        )
        .await,
        FinalizeOutcome::Applied
    ) {
        info!(ticket = %ticket.id, "{reason}");
        let _ = crate::jobs::complete_ticket_phase_jobs(&crate::session::store().conn, &ticket.id)
            .await;
    }
}

/// Why the workspace has no usable repository — git not installed, or the root
/// not a repo — or `None` when it has one. Both halves are the precondition for
/// every git read and for the commit in this module; each caller states its own
/// behaviour for the no-repo case.
async fn missing_repo_reason(ws: &Workspace) -> Option<&'static str> {
    if !crate::git::commands::git_is_installed().await {
        return Some("Git not installed — moving to Done without commit");
    }
    if !crate::git::commands::is_git_repo(ws.as_path()) {
        return Some("Not a git repo — moving to Done without commit");
    }
    None
}

/// Shared sanitation finalize tail: clear the implementation roster and
/// finalize the ticket at Done via its git status. A Done transition that
/// declines — the ticket was moved externally (a cancel), or the transition
/// errored — is not a round failure and is not stopped here.
///
/// `Err` carries the full failure chain (including the git process output) when
/// the repository state could not be read or the commit failed — the caller
/// stops the round. `Ok(())` means the finalize ran to its end: the Done
/// transition was attempted, with or without a commit (a clean tree and a
/// workspace without a repository move to Done without one).
async fn finalize_sanitation_ticket(
    ticket: &Ticket,
    ws: &Workspace,
    job_id: &str,
) -> Result<(), String> {
    clear_implementation_roster(&crate::session::store().conn, job_id, &ticket.id).await;
    let phase = TicketPhase::InSanitation;

    if let Some(reason) = missing_repo_reason(ws).await {
        transition_ticket_to_done_no_comment(ticket, phase, reason).await;
        return Ok(());
    }

    let porcelain = run_git_status(ws.as_path())
        .await
        .map_err(|e| format!("{e:#}"))?;

    if porcelain.trim().is_empty() {
        transition_ticket_to_done_no_comment(
            ticket,
            phase,
            "Clean working tree — moving to Done without commit",
        )
        .await;
        return Ok(());
    }

    match crate::git::commands::run_git_commit(ws.as_path(), &ticket.title).await {
        Ok(commit_info) => {
            // A pipeline auto-commit is a ref-only change the file watcher
            // never reports, so notify the GUI to refresh the footer promptly
            // instead of waiting for the periodic remote timer.
            crate::git::commands::notify_git_commit(ws.as_path());
            finalize_commit_and_transition(ticket, commit_info, phase).await;
            Ok(())
        }
        Err(e) => Err(format!("{e:#}")),
    }
}

/// The round's tail: finalize the ticket at Done, or stop the round when the
/// repository state could not be read or the commit failed. Stopping is the
/// shared technical-failure path — a round that cannot finish must never leave
/// its phase job behind for the puller to re-drive.
async fn finish_or_stop(ticket: &Ticket, ws: &Workspace, job_id: &str) {
    if let Err(cause) = finalize_sanitation_ticket(ticket, ws, job_id).await {
        stop_sanitation_round(ticket, ws, job_id, &cause).await;
    }
}

/// Stop a sanitation round that could not finish: the full cause in the log and
/// in a ticket comment, the workspace frozen, the phase job deleted. The ticket
/// stays in `InSanitation` and the round is replayed from scratch once the
/// workspace is unpaused — the uncommitted changes are preserved.
async fn stop_sanitation_round(ticket: &Ticket, ws: &Workspace, job_id: &str, cause: &str) {
    error!(
        ticket = %ticket.id,
        error = %cause,
        "Sanitation round could not finish — stopping the round and freezing the workspace"
    );
    let detail = crate::util::failure_detail(cause, "sanitation failure");
    let comment = substitute(
        &load_prompt("pipeline/sanitation_stop_comment.md"),
        &[("{{failure_details}}", &detail)],
    );

    // Read before the freeze lands: `reset_phase_attempt` reports whether THIS
    // stop paused the workspace, so an already-set flag means the freeze is not
    // this failure's (a human pause, or a failure stop on another ticket) and
    // still has to be described in the notice. The notice itself is never
    // suppressed: a stop cannot repeat inside one freeze episode (it deletes the
    // phase job and the freeze holds the poll gate), so the cause always reaches
    // the Manager.
    let was_frozen = matches!(
        crate::workspace::store().get_by_name(&ws.name).await,
        Ok(Some(live)) if live.paused
    );

    let pause_occurred = reset_phase_attempt(
        ticket,
        TicketPhase::InSanitation,
        job_id,
        "sanitation failure",
        &comment,
    )
    .await;

    let content = substitute(
        &load_prompt("pipeline/sanitation_stop_notification.md"),
        &[
            ("{{ticket_id}}", &ticket.id),
            ("{{failure_details}}", &detail),
            (
                "{{workspace_status}}",
                &pause_status_sentence(pause_occurred || was_frozen),
            ),
        ],
    );
    notify_manager_system(&ws.name, content);
}

/// After a successful `git commit`, persist the metadata and transition the
/// ticket to Done atomically.
async fn finalize_commit_and_transition(
    ticket: &Ticket,
    commit_info: crate::git::commands::CommitInfo,
    source: TicketPhase,
) {
    let phase_label = source.as_ref();

    crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(&ticket.id);

    let notify_policy = determine_notify_policy(&ticket.workspace_name, &ticket.id).await;

    let log_label = format!(
        "finalize Done transition from {phase_label} ({})",
        commit_info.short_hash(),
    );

    if matches!(
        with_comment_and_transition(
            TransitionCtx::new(
                ticket,
                source,
                TicketPhase::Done,
                notify_policy,
                &log_label,
                Role::Sanitation.as_str(),
            ),
            async |tx| {
                BoardStore::set_commit_info_tx(
                    tx,
                    &ticket.id,
                    &commit_info.hash,
                    commit_info.lines_added,
                    commit_info.lines_removed,
                )
                .await?;
                Ok(())
            },
        )
        .await,
        FinalizeOutcome::Applied
    ) {
        info!(ticket = %ticket.id, "Committed {}, moving to Done", commit_info.short_hash());
        let _ = crate::jobs::complete_ticket_phase_jobs(&crate::session::store().conn, &ticket.id)
            .await;
    }
}

/// Absorb the post-run tail for the sanitation stage: phase/drain guards,
/// response-None failure block, verdict extraction, and job terminalization.
pub(crate) async fn finalize_sanitation_stage(
    ticket: &Ticket,
    agent: &Agent,
    response: Option<&str>,
    job_id: &str,
    ws: &Workspace,
    paused: bool,
) {
    if guard_stage(
        &ticket.id,
        TicketPhase::InSanitation,
        "Sanitation",
        response,
        job_id,
    )
    .await
    {
        return;
    }
    // A cooperative pause-freeze (only possible when the agent produced no
    // response) leaves the job in place — never discards a completed round.
    if paused {
        pause_freezing(ticket, job_id).await;
        return;
    }
    if response.is_none() {
        warn!(
            ticket = %ticket.id,
            "Sanitation agent returned no output — resetting for a fresh attempt"
        );
        reset_phase_attempt(
            ticket,
            TicketPhase::InSanitation,
            job_id,
            "sanitation failure",
            "Sanitation could not complete the round (the agent did not respond).",
        )
        .await;
        return;
    }

    let extraction_prompt = crate::prompt::load_prompt("extraction/sanitation.md");
    match agent
        .extract_verdict::<crate::SanitationVerdict>(&extraction_prompt, None, None)
        .await
    {
        Ok(verdict) => {
            process_sanitation_verdict(ticket, job_id, verdict, ws).await;
        }
        Err(failure) => {
            warn!(
                ticket = %ticket.id,
                error = %failure,
                "Failed to extract sanitation verdict — resetting for a fresh attempt"
            );
            reset_phase_attempt(
                ticket,
                TicketPhase::InSanitation,
                job_id,
                "sanitation failure",
                "Sanitation could not complete the round (the verdict could not be extracted).",
            )
            .await;
        }
    }
}

/// Run the sanitation agent to inspect new/untracked files in the workspace.
async fn dispatch_sanitation(ticket: Arc<Ticket>, ws: Workspace, job_id: &str) {
    let untracked_files = match list_new_or_untracked_files(ws.as_path()).await {
        Ok(files) if files.is_empty() => {
            // No new/untracked files — skip the sanitation agent entirely and
            // commit straight to Done (no bounce budget consumed). The skip is
            // silent in ticket history.
            finish_or_stop(&ticket, &ws, job_id).await;
            return;
        }
        Ok(files) => files.join("\n"),
        Err(e) => {
            let Some(reason) = missing_repo_reason(&ws).await else {
                // A genuine repository-state read failure: stop the round rather
                // than let the agent inspect a placeholder list and commit the
                // tree uninspected.
                stop_sanitation_round(&ticket, &ws, job_id, &format!("{e:#}")).await;
                return;
            };
            // No usable repository: keep the dedicated silent Done fallback —
            // the agent round proceeds without a file list and the finalize
            // then moves the ticket to Done without a commit.
            warn!(
                ticket = %ticket.id,
                error = %e,
                reason,
                "Failed to list untracked files — proceeding without a file list",
            );
            String::from("(could not list untracked files)")
        }
    };

    let prompt = substitute(
        &crate::prompt::load_prompt("sanitation.md"),
        &[
            ("{{ticket_title}}", &ticket.title),
            ("{{ticket_description}}", &ticket.description),
            ("{{untracked_files}}", &untracked_files),
        ],
    );

    let conn = &crate::session::store().conn;
    sync_phase_job_task(conn, job_id, &prompt).await;

    run_stage_agent(&ticket, &ws, job_id, &prompt, StageRunKind::Sanitation).await;
}

/// Process the result of a sanitation agent inspection.
async fn process_sanitation_verdict(
    ticket: &Ticket,
    job_id: &str,
    verdict: crate::SanitationVerdict,
    ws: &Workspace,
) {
    if verdict.pass {
        let passed_suffix = if verdict.garbage_files.is_empty() {
            ""
        } else {
            " (files reviewed)"
        };
        let comment = format!(
            "🧹 Sanitation passed{passed_suffix}: {rationale}",
            rationale = verdict.rationale
        );
        if let Err(e) = board()
            .add_comment(&ticket.id, Role::Sanitation.as_str(), &comment)
            .await
        {
            warn!(ticket = %ticket.id, error = %e, "Failed to record sanitation pass comment");
        }
        finish_or_stop(ticket, ws, job_id).await;
    } else {
        let garbage_list = verdict.garbage_files.join("\n- ");
        let comment = substitute(
            &load_prompt("pipeline/sanitation_failed_comment.md"),
            &[
                ("{{garbage_list}}", &garbage_list),
                ("{{rationale}}", &verdict.rationale),
            ],
        );
        bounce_to_development(
            ticket,
            TicketPhase::InSanitation,
            "Sanitation",
            Role::Sanitation.as_str(),
            Role::Sanitation.as_str(),
            &comment,
            job_id,
        )
        .await;
    }
}
