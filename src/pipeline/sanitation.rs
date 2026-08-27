//! InSanitation phase module — inspect new/untracked files before commit.

use std::sync::Arc;

use crate::pipeline::board::Ticket;
use crate::prompt::{load_prompt, substitute};
use crate::{Agent, Role, Workspace};

use super::{
    BoardStore, FinalizeOutcome, SYSTEM_ROLE, StageRunKind, TicketPhase, TransitionCtx, board,
    bounce_to_development, clear_implementation_roster, comment_and_transition,
    determine_notify_policy, error, guard_stage, info, is_ticket_in_phase,
    list_new_or_untracked_files, pause_freezing, reset_phase_attempt, run_git_status,
    run_stage_agent, sync_phase_job_task, warn, with_comment_and_transition,
};

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    if !is_ticket_in_phase(&ticket.id, TicketPhase::InSanitation).await {
        let _ = crate::jobs::complete_ticket_job(&crate::session::store().conn, &job_id).await;
        return;
    }
    dispatch_sanitation(ticket, ws, &job_id).await;
}

/// Transition a ticket to Done and delete any launched phase jobs.
async fn transition_ticket_to_done(ticket: &Ticket, source: TicketPhase, comment: &str) {
    let notify_policy = determine_notify_policy(&ticket.workspace_name, &ticket.id).await;
    if matches!(
        comment_and_transition(
            TransitionCtx::new(ticket, source, TicketPhase::Done, notify_policy, "Finalize"),
            SYSTEM_ROLE,
            comment,
        )
        .await,
        FinalizeOutcome::Applied
    ) {
        info!(ticket = %ticket.id, "{comment}");
        let _ = crate::jobs::complete_ticket_phase_jobs(&crate::session::store().conn, &ticket.id)
            .await;
    }
}

/// Ensure git is available and run `git status --porcelain`.
async fn ensure_git_or_done_and_get_status(
    ticket: &Ticket,
    ws: &Workspace,
    phase: TicketPhase,
    error_context: &'static str,
) -> Option<String> {
    let repo_path = ws.as_path();

    if !crate::git::commands::git_is_installed().await {
        transition_ticket_to_done(
            ticket,
            phase,
            "Git not installed — moving to Done without commit",
        )
        .await;
        return None;
    }
    if !crate::git::commands::is_git_repo(repo_path) {
        transition_ticket_to_done(
            ticket,
            phase,
            "Not a git repo — moving to Done without commit",
        )
        .await;
        return None;
    }

    match run_git_status(repo_path).await {
        Ok(porcelain) => Some(porcelain),
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to check git status — staying in {} for retry: {}",
                phase.as_ref(),
                error_context,
            );
            None
        }
    }
}

/// Finalize a ticket given an already-obtained `git status --porcelain` output.
async fn finalize_ticket_with_git_status(
    ticket: Ticket,
    ws: Workspace,
    source: TicketPhase,
    porcelain: &str,
) {
    let repo_path = ws.as_path();

    if porcelain.trim().is_empty() {
        transition_ticket_to_done(
            &ticket,
            source,
            "Clean working tree — moving to Done without commit",
        )
        .await;
        return;
    }

    match crate::git::commands::run_git_commit(repo_path, &ticket.title).await {
        Ok(commit_info) => {
            // A pipeline auto-commit is a ref-only change the file watcher
            // never reports, so notify the GUI to refresh the footer promptly
            // instead of waiting for the periodic remote timer.
            crate::git::commands::notify_git_commit(repo_path);
            finalize_commit_and_transition(&ticket, commit_info, source).await;
        }
        Err(e) => {
            error!(
                ticket = %ticket.id,
                error = %e,
                "Commit failed — staying in {} for retry",
                source.as_ref(),
            );
        }
    }
}

/// After a successful `git commit`, persist the metadata and transition the
/// ticket to Done atomically.
async fn finalize_commit_and_transition(
    ticket: &Ticket,
    commit_info: crate::git::commands::CommitInfo,
    source: TicketPhase,
) {
    let comment = format_commit_summary(
        commit_info.short_hash(),
        commit_info.lines_added,
        commit_info.lines_removed,
    );

    let phase_label = source.as_ref();

    crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(&ticket.id);

    let notify_policy = determine_notify_policy(&ticket.workspace_name, &ticket.id).await;

    let log_label = format!(
        "finalize Done transition from {phase_label} ({})",
        commit_info.short_hash(),
    );

    if matches!(
        with_comment_and_transition(
            TransitionCtx::new(ticket, source, TicketPhase::Done, notify_policy, &log_label),
            async |tx| {
                BoardStore::set_commit_info_tx(
                    tx,
                    &ticket.id,
                    &commit_info.hash,
                    commit_info.lines_added,
                    commit_info.lines_removed,
                )
                .await?;
                BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, &comment).await?;
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

/// Format a commit summary line for the ticket comment history.
fn format_commit_summary(short_hash: &str, added: i64, removed: i64) -> String {
    match (added, removed) {
        (0, 0) => format!("Committed as `{short_hash}` (no changes)"),
        (a, 0) => format!("Committed as `{short_hash}` (+{a})"),
        (0, r) => format!("Committed as `{short_hash}` (-{r})"),
        (a, r) => format!("Committed as `{short_hash}` (+{a}/-{r})"),
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
            "Sanitation could not complete the round (the agent did not respond) \
             — resetting for a fresh attempt.",
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
                "Sanitation could not complete the round (the verdict could not be \
                 extracted) — resetting for a fresh attempt.",
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
            // commit straight to Done (no bounce budget consumed).
            let already_skipped = board()
                .has_comment_containing(
                    &ticket.id,
                    Role::Sanitation.as_str(),
                    "skipping sanitation agent",
                )
                .await
                .unwrap_or(false);
            if !already_skipped {
                let comment =
                    "🧹 No new or untracked files — skipping sanitation agent, committing to Done.";
                if let Err(e) = board()
                    .add_comment(&ticket.id, Role::Sanitation.as_str(), comment)
                    .await
                {
                    warn!(ticket = %ticket.id, error = %e, "Failed to record sanitation skip comment");
                }
            }
            clear_implementation_roster(&crate::session::store().conn, job_id, &ticket.id).await;
            let Some(porcelain) = ensure_git_or_done_and_get_status(
                &ticket,
                &ws,
                TicketPhase::InSanitation,
                "finalize",
            )
            .await
            else {
                return;
            };
            finalize_ticket_with_git_status(
                ticket.as_ref().clone(),
                ws.clone(),
                TicketPhase::InSanitation,
                &porcelain,
            )
            .await;
            return;
        }
        Ok(files) => files.join("\n"),
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to list untracked files — proceeding with empty list",
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
        clear_implementation_roster(&crate::session::store().conn, job_id, &ticket.id).await;
        let Some(porcelain) =
            ensure_git_or_done_and_get_status(ticket, ws, TicketPhase::InSanitation, "finalize")
                .await
        else {
            return;
        };
        finalize_ticket_with_git_status(
            ticket.clone(),
            ws.clone(),
            TicketPhase::InSanitation,
            &porcelain,
        )
        .await;
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
            /* drains_siblings */ true,
            Role::Sanitation.as_str(),
            &comment,
            job_id,
            ws,
        )
        .await;
    }
}
