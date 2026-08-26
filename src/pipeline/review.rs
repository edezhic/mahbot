//! InReview phase module — parallel reviewers verify the change.
//!
//! Owns the reviewer-specific git-state decisions the shared dispatch engine
//! routes to it: the skip-review (content-identical) check, the churn-calibrated
//! reviewer count, and reviewed-base recording after a passing review.

use std::path::Path;
use std::sync::Arc;

use crate::Workspace;
use crate::git::commands::{
    has_unstaged_changes, run_git_add_all, run_git_diff_stats, run_git_head, run_git_status,
    run_git_write_tree,
};
use crate::pipeline::board::Ticket;

use super::{
    ParallelVerdict, REVIEWER_VI, TicketPhase, debug, dispatch_verifiers, info, is_ticket_in_phase,
    warn,
};

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String, resumed: bool) {
    if !is_ticket_in_phase(&ticket.id, TicketPhase::InReview).await {
        let _ = crate::jobs::complete_ticket_job(&crate::session::store().conn, &job_id).await;
        return;
    }
    dispatch_verifiers(ticket, ws, REVIEWER_VI, job_id, resumed).await;
}

/// Decide whether the reviewer pass may be skipped for a ticket.
fn should_skip_review(
    reviewed_head: Option<&str>,
    reviewed_tree: Option<&str>,
    current_head: Option<&str>,
    current_tree: Option<&str>,
    porcelain: &str,
) -> bool {
    let (Some(base_head), Some(base_tree)) = (reviewed_head, reviewed_tree) else {
        return false;
    };
    let (Some(head), Some(tree)) = (current_head, current_tree) else {
        return false;
    };
    head == base_head && tree == base_tree && !has_unstaged_changes(porcelain)
}

/// Compute the skip-review decision for a ticket.
pub(crate) async fn compute_review_skip(ticket: &Ticket, repo_path: &Path) -> anyhow::Result<bool> {
    let porcelain = run_git_status(repo_path).await?;
    let head = run_git_head(repo_path).await.ok();
    let tree = run_git_write_tree(repo_path).await.ok();
    if (head.is_none() || tree.is_none()) && ticket.reviewed_head.is_some() {
        warn!(
            ticket = %ticket.id,
            head = head.is_some(),
            tree = tree.is_some(),
            "Could not compute full content identity — running full review",
        );
    }
    Ok(should_skip_review(
        ticket.reviewed_head.as_deref(),
        ticket.reviewed_tree.as_deref(),
        head.as_deref(),
        tree.as_deref(),
        &porcelain,
    ))
}

/// Gather the working-tree churn at review dispatch.
async fn working_tree_churn(repo_path: &Path) -> anyhow::Result<i64> {
    let (added, removed) = run_git_diff_stats(repo_path).await?;
    Ok(added + removed)
}

/// Compute the reviewer count for a review round.
pub(crate) async fn compute_reviewer_count(ticket: &Ticket, repo_path: &Path) -> usize {
    let tiny = crate::pipeline::verdict::DEFAULT_REVIEW_COUNT_TINY_CHURN;
    let low = crate::pipeline::verdict::DEFAULT_REVIEW_COUNT_LOW_CHURN;
    let high = crate::pipeline::verdict::DEFAULT_REVIEW_COUNT_HIGH_CHURN;

    match working_tree_churn(repo_path).await {
        Ok(total) => {
            let base = crate::pipeline::verdict::review_base_from_signals(total, tiny, low, high);
            info!(
                ticket = %ticket.id,
                total_churn = total,
                reviewer_base = base,
                "Reviewer count calibration: base {base} from total churn",
            );
            crate::pipeline::verdict::review_agent_count(base, ticket.priority)
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Could not compute working-tree churn — reviewer base defaults to 3",
            );
            3
        }
    }
}

/// Stage all changes and record the ticket's reviewed base (HEAD + index tree)
/// after a review round that produced verdicts and transitioned the ticket.
pub(crate) async fn record_reviewed_base_after_review(
    repo_path: &Path,
    ticket_id: &str,
    git_available: bool,
    transitioned: bool,
    results: &[ParallelVerdict],
    log_prefix: &str,
) {
    let reviewed = results
        .iter()
        .any(|r| matches!(r, ParallelVerdict::Verdict(_)));
    if git_available && transitioned && reviewed {
        if let Err(e) = run_git_add_all(repo_path).await {
            warn!(
                ticket = %ticket_id,
                error = %e,
                "{log_prefix}Failed to stage changes after review — reviewed base not recorded",
            );
        } else {
            let head = run_git_head(repo_path).await.ok();
            let tree = run_git_write_tree(repo_path).await.ok();
            if head.is_none() || tree.is_none() {
                warn!(
                    ticket = %ticket_id,
                    head = head.is_some(),
                    tree = tree.is_some(),
                    "{log_prefix}Could not compute content identity after review — reviewed base not recorded",
                );
            } else if let Err(e) = super::board()
                .set_reviewed_base(ticket_id, head.as_deref(), tree.as_deref())
                .await
            {
                warn!(
                    ticket = %ticket_id,
                    error = %e,
                    "{log_prefix}Failed to record reviewed base — later rounds will re-review",
                );
            } else {
                debug!(ticket = %ticket_id, "{log_prefix}Recorded reviewed base after review");
            }
        }
    }
}
