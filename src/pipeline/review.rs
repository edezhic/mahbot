//! InReview phase module — parallel reviewers verify the change.
//!
//! Owns the reviewer-specific git-state decisions the shared dispatch engine
//! routes to it: the skip-review (content-identical) check, the churn-calibrated
//! reviewer count, and reviewed-base recording after a passing review.

use std::path::Path;
use std::sync::Arc;

use crate::git::commands::{
    has_unstaged_changes, run_git_add_all, run_git_head, run_git_status, run_git_worktree_snapshot,
    run_git_write_tree,
};
use crate::pipeline::board::Ticket;
use crate::{Role, Workspace};

use super::{
    ParallelVerdict, REVIEWER_VI, SYSTEM_ROLE, TicketPhase, TransitionCtx, VerifierInfo,
    comment_and_transition, debug, dispatch_verifiers, info, warn,
};

pub(crate) async fn run(ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    dispatch_verifiers(ticket, ws, REVIEWER_VI, job_id).await;
}

/// Whether git is usable for a reviewer round (reviewer-only: QA never skips
/// review or records a reviewed base).
pub(crate) async fn git_available_for_review(ws: &Workspace, vi: VerifierInfo) -> bool {
    vi.role == Role::Reviewer
        && crate::git::commands::git_is_installed().await
        && crate::git::commands::is_git_repo(ws.as_path())
}

/// Handle the reviewer-only skip-review path (content identical to the recorded
/// base). Returns `true` when the skip was applied — the caller must return
/// immediately; `false` means the review should proceed normally.
pub(crate) async fn maybe_skip_review(
    ticket: &Ticket,
    ws: &Workspace,
    vi: VerifierInfo,
    job_id: &str,
) -> bool {
    if !git_available_for_review(ws, vi).await {
        return false;
    }
    let repo_path = ws.as_path();
    match compute_review_skip(ticket, repo_path).await {
        Ok(true) => {
            info!(
                ticket = %ticket.id,
                "Content identical to reviewed base — skipping reviewer dispatch",
            );
            let _ = comment_and_transition(
                TransitionCtx::buffered(
                    ticket,
                    vi.active_phase,
                    TicketPhase::InQa,
                    vi.log_label,
                    Role::Reviewer.as_str(),
                ),
                SYSTEM_ROLE,
                "Content is identical to the reviewed base recorded for this ticket \
                 (same HEAD commit and index tree, no working-tree changes). \
                 Skipping reviewer dispatch.",
            )
            .await;
            let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
            true
        }
        Ok(false) => false,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Git status check failed for skip-review — proceeding with normal review",
            );
            false
        }
    }
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
    let snapshot = run_git_worktree_snapshot(repo_path).await?;
    // An unborn HEAD is not a valid churn baseline: surface as Err so the
    // caller defaults the reviewer base instead of calibrating on a zero diff.
    if snapshot.unborn_head {
        anyhow::bail!("Repository has no commits — no churn baseline");
    }
    // Churn calibration uses only the exact line counts — the
    // huge/binary untracked file-count must NOT influence reviewer counts.
    Ok(snapshot.stats.added + snapshot.stats.removed)
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
) {
    let reviewed = results
        .iter()
        .any(|r| matches!(r, ParallelVerdict::Verdict(_)));
    if git_available && transitioned && reviewed {
        if let Err(e) = run_git_add_all(repo_path).await {
            warn!(
                ticket = %ticket_id,
                error = %e,
                "Failed to stage changes after review — reviewed base not recorded",
            );
        } else {
            let head = run_git_head(repo_path).await.ok();
            let tree = run_git_write_tree(repo_path).await.ok();
            if head.is_none() || tree.is_none() {
                warn!(
                    ticket = %ticket_id,
                    head = head.is_some(),
                    tree = tree.is_some(),
                    "Could not compute content identity after review — reviewed base not recorded",
                );
            } else if let Err(e) = super::board()
                .set_reviewed_base(ticket_id, head.as_deref(), tree.as_deref())
                .await
            {
                warn!(
                    ticket = %ticket_id,
                    error = %e,
                    "Failed to record reviewed base — later rounds will re-review",
                );
            } else {
                debug!(ticket = %ticket_id, "Recorded reviewed base after review");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::commands::MAX_UNTRACKED_SIZE;
    use crate::util::test::init_temp_repo;

    #[tokio::test]
    async fn working_tree_churn_ignores_huge_binary_file_count() {
        let (_dir, repo_path) = init_temp_repo();
        // A normal untracked file (3 lines) plus an oversized untracked file.
        std::fs::write(repo_path.join("a.rs"), b"fn foo() {\n    bar();\n}\n").unwrap();
        let size = usize::try_from(MAX_UNTRACKED_SIZE).unwrap() + 1;
        std::fs::write(repo_path.join("big.bin"), vec![b'a'; size]).unwrap();
        let churn = working_tree_churn(&repo_path).await.unwrap();
        // Churn is added+removed only; the oversized file contributes nothing.
        assert_eq!(churn, 3);
    }
}
