//! Board poller — picks up tickets from the board and dispatches agents.
//!
//! Poll phases — dispatches agents based on ticket phase:
//! - Backlog → spawn Analyst agents (`PARALLEL_AGENT_COUNT` parallel)
//! - ReadyForDevelopment → spawn Engineer agent
//! - InDiagnostics → dispatch diagnostics runner (shell commands)
//! - DiagnosticsDone → spawn Reviewer agents (`PARALLEL_AGENT_COUNT` parallel)
//! - Reviewed → spawn QA agents (`PARALLEL_AGENT_COUNT` parallel)
//!
//! Reviewer and QA phases share a single `PollPhase::VerifierCheck` variant
//! with per-phase configuration carried in `VerifierInfo` constants
//! (`REVIEWER_VI`, `QA_VI`).
//!
//! Additionally, QaPassed tickets are auto-committed and moved to Done outside
//! the claim loop (no agent required).

use std::fmt::Write;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use futures_util::future::join_all;

use crate::agent::run_agent;
use crate::board::{BOARD, BoardStore, Ticket, TicketComment, TicketPhase};
use crate::manager_queue::{JobKind, ManagerJob};
use crate::prompt::{load_prompt, substitute};
use crate::session::ticket_session_key;
use crate::ticket_buffer;
use crate::tools::shell::{ShellMode, ShellTool};
use crate::{Role, Tool, Workspace};

/// Number of parallel agents spawned per verification phase (Analyst, Reviewer, QA).
const PARALLEL_AGENT_COUNT: usize = 3;

/// Comments threshold — tickets accumulating more than this number of comments
/// are tripped by the circuit breaker (i.e., the trip point is > threshold),
/// transitioning to Failed and pausing the workspace.
const CIRCUIT_BREAKER_COMMENT_THRESHOLD: usize = 100;

/// Maximum number of cumulative diagnostics failures allowed before the circuit
/// breaker trips. The breaker trips when `count > DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD`
/// (i.e., at ≥5 failures), failing the ticket and pausing the workspace to
/// prevent thrashing.
const DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD: usize = 4;

/// Minimum acceptable verification score (0-10) for analysis phase.
const ANALYSIS_THRESHOLD: u8 = 7;

/// Minimum acceptable verification score (0-10) for review and QA phases.
const REVIEW_QA_THRESHOLD: u8 = 9;

/// Returns the global [`BoardStore`] singleton.
#[inline]
fn board() -> &'static BoardStore {
    crate::board::store()
}

/// Returns `true` if the ticket is in the expected phase (safe to proceed).
/// Returns `false` if the ticket was moved externally or an error occurred.
#[must_use]
async fn is_ticket_in_phase(ticket_id: &str, expected: TicketPhase) -> bool {
    match board().get_ticket_status(ticket_id).await {
        Ok(Some(status)) => {
            let ok = status == expected;
            if !ok {
                debug!(
                    ticket = %ticket_id,
                    expected = %expected,
                    actual = %status,
                    "Ticket moved externally — bailing out",
                );
            }
            ok
        }
        Ok(None) => {
            warn!(ticket = %ticket_id, "Ticket not found — may have been deleted");
            false
        }
        Err(e) => {
            warn!(ticket = %ticket_id, error = %e, "Failed to check ticket status");
            false
        }
    }
}

/// Shared pre-flight guard for dispatch functions that spawn agents.
/// Verifies the ticket is still in `expected` (avoids wasted DB writes
/// and LLM API costs if the ticket was moved externally) and checks the
/// circuit breaker (fails tickets with excessive comment accumulation and
/// pauses the workspace to prevent dispatch thrashing).
///
/// Returns `true` when it's safe for the caller to proceed. Returns `false`
/// when the ticket has moved, been failed, or the workspace was paused — the
/// caller should bail out immediately.
///
/// Used by all agent-spawning dispatch functions
/// ([`dispatch_backlog_analysts`], [`dispatch_engineer`], [`dispatch_verifiers`])
/// for structural consistency. Previously, verifiers intentionally omitted this
/// pre-agent check — churned tickets got one last review cycle before the
/// circuit breaker could trip. Adding the pre-agent guard saves LLM credits
/// by failing tickets with excessive churn before verifier agents run, at the
/// cost of removing that last-chance review cycle. Diagnostics uses a separate
/// circuit breaker (see [`trip_diagnostics_circuit_breaker_if_exceeded`]).
#[must_use]
async fn guard_phase_and_circuit_breaker(
    ticket: &Ticket,
    expected: TicketPhase,
    label: &str,
) -> bool {
    if !is_ticket_in_phase(&ticket.id, expected).await {
        return false;
    }
    if trip_circuit_breaker_if_exceeded(ticket, expected, label).await {
        return false;
    }
    true
}

/// Controls whether a ticket transition triggers an immediate notification
/// to the Manager (via [`notify_ticket`]) or is buffered for batched delivery.
enum NotifyPolicy {
    /// Immediately enqueue a Manager notification for this transition.
    Notify,
    /// Buffer the transition for batched delivery alongside the next
    /// notification. See [`ticket_buffer`] for details.
    Buffer,
}

/// Transition a ticket to `target` phase if it's still in `expected` phase.
/// Returns `Ok(())` if the transition was applied (ticket was still in expected
/// phase). Returns `Err(String)` with a descriptive message if the ticket was
/// moved externally or an error occurred.
///
/// If `notify` is [`NotifyPolicy::Notify`], calls [`notify_ticket`] on success;
/// errors from notification are logged and discarded (not propagated).
///
/// # Note: workspace auto-pause on failure
///
/// Auto-pausing the workspace on ticket failure (`Failed` status) is handled
/// inside [`notify_ticket`] — it only fires when
/// `notify` is [`NotifyPolicy::Notify`]. If adding a `Failed` transition with
/// [`NotifyPolicy::Buffer`], workspace pausing will NOT occur automatically;
/// ensure appropriate handling at the call site.
async fn transition_ticket(
    ticket: &Ticket,
    expected: TicketPhase,
    target: TicketPhase,
    notify: NotifyPolicy,
) -> Result<(), String> {
    match board()
        .transition_to(&ticket.id, Some(expected), target, None)
        .await
    {
        Ok(()) => {
            if matches!(notify, NotifyPolicy::Notify) {
                notify_ticket(ticket, target).await;
            } else if let Some(ws) =
                resolve_ticket_workspace(ticket, "cannot buffer transition").await
            {
                ticket_buffer::push(
                    &ws.name,
                    &ticket.id,
                    ticket.status.as_ref(),
                    target.as_ref(),
                );
            }

            // Auto-pause on failure is handled inside `notify_ticket`.
            // If a `Failed` transition with `NotifyPolicy::Buffer` is added here,
            // auto-pause will NOT occur — ensure workspace pausing is
            // handled explicitly if needed.

            Ok(())
        }
        Err(e) => {
            debug!(
                ticket = %ticket.id,
                expected = %expected,
                target = %target,
                error = %e,
                "Failed to update ticket status",
            );
            Err(e.to_string())
        }
    }
}

/// Resolve a workspace from a ticket's stored `workspace_name`.
///
/// Returns `None` and logs a warning if the workspace cannot be found. Both
/// `Ok(None)` (name not in DB) and `Err(...)` (DB error) result in `None`.
/// Callers that need fine-grained error handling should call
/// [`crate::workspace::get_by_name`] directly.
///
/// The `context` string is embedded in the log message to distinguish callers.
#[must_use]
async fn resolve_ticket_workspace(
    ticket: &Ticket,
    context: &'static str,
) -> Option<crate::Workspace> {
    match crate::workspace::get_by_name(&ticket.workspace_name).await {
        Ok(Some(ws)) => Some(ws),
        Ok(None) => {
            warn!(
                ticket = %ticket.id,
                workspace_name = %ticket.workspace_name,
                "Workspace not found for ticket — {context}",
            );
            None
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                workspace_name = %ticket.workspace_name,
                error = %e,
                "Failed to look up workspace for ticket — {context}",
            );
            None
        }
    }
}

/// Bounce a failed ticket back to [`TicketPhase::ReadyForDevelopment`] with a
/// pipeline reservation, ensuring it gets priority re-dispatch over fresh
/// tickets.
///
/// Used by both the diagnostics runner and the verifier (reviewer/QA)
/// processing — both paths had nearly identical duplicated bounce-back logic
/// that differed only in the source phase and log label, and the verifier
/// path was missing the defensive `assigned_to` cleanup on transition failure.
async fn bounce_back_to_development(ticket: &Ticket, source_phase: TicketPhase, log_label: &str) {
    match board()
        .transition_to(
            &ticket.id,
            Some(source_phase),
            TicketPhase::ReadyForDevelopment,
            Some(true),
        )
        .await
    {
        Ok(()) => {
            ticket_buffer::push(
                &ticket.workspace_name,
                &ticket.id,
                source_phase.as_ref(),
                TicketPhase::ReadyForDevelopment.as_ref(),
            );
            info!(
                ticket = %ticket.id,
                "{log_label} failed — pipeline reservation set for rework priority",
            );
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "{log_label} failed but transition to ReadyForDevelopment \
                 failed — ticket stuck in {phase}, clearing assigned_to for retry",
                phase = source_phase.as_ref(),
            );
            if let Err(e) = board().set_assigned_to(&ticket.id, None).await {
                warn!(
                    ticket = %ticket.id,
                    error = %e,
                    "Failed to clear assigned_to on transition failure in bounce_back_to_development",
                );
            }
        }
    }
}

/// Prepare and dispatch a notification for a ticket transition.
///
/// Renders the notification message and enqueues a Manager job via the
/// serialized [`crate::manager_queue::MANAGER_QUEUE`]. The consumer loop handles user lookup
/// and delivery (broadcasts to all users with this workspace active).
///
/// # Workspace auto-pause on failure
///
/// When `status` is [`TicketPhase::Failed`], this function also pauses
/// the ticket's workspace so the user can inspect before any further
/// automated work. This is a best-effort operation — if workspace
/// resolution fails or the DB call errors, the pause is silently skipped
/// and logged (errors never propagate to the caller).
///
/// The session key (`manager_{ws_name}`) is intentionally shared between
/// user-facing Manager chat (main.rs) and notification agents — the same Manager
/// must see both notification context and user conversation history in a unified
/// session. Do NOT change this key or add `manager_` to `TRANSIENT_SESSION_PREFIXES`
/// — it would either break context continuity or nuke user conversation history.
///
/// Never panics — errors are logged and discarded.
async fn notify_ticket(ticket: &Ticket, status: TicketPhase) {
    let Some(ws) = resolve_ticket_workspace(ticket, "skipping notification").await else {
        error!(
            ticket = %ticket.id,
            workspace_name = %ticket.workspace_name,
            "Workspace resolution failed — notification skipped"
        );
        return;
    };

    // Auto-pause the workspace when a ticket fails, so the user
    // can inspect before any further automated work.
    // Note: if workspace resolution succeeded just above but this
    // DB call fails, the pause is silently skipped — errors are
    // handled internally and the impact is negligible on an exceptional path.
    if status == TicketPhase::Failed && !ws.paused {
        if let Err(e) = crate::workspace::store().set_paused(&ws.name, true).await {
            warn!(
                ticket = %ticket.id,
                workspace = %ws.name,
                error = %e,
                "Failed to pause workspace after ticket failure",
            );
        } else {
            info!(
                ticket = %ticket.id,
                workspace = %ws.name,
                "Workspace paused due to ticket failure",
            );
        }
    }

    // Build a single-line transition log for this ticket.
    let transition_log = format!(
        "[{}] {}: {} → {}",
        ticket.reporter,
        ticket.id,
        ticket.status,
        status.as_ref()
    );

    // Drain buffered non-critical transitions before rendering the
    // notification template. The drained entries are injected via the
    // canonical {{ticket_updates}} placeholder, which evaluates to an
    // empty string (harmless) when there are no buffered transitions.
    // Data-loss guard: drain only after workspace lookup succeeds
    // (above) — if lookup had failed, the buffer entries remain for
    // the next delivery attempt.
    let drained = crate::ticket_buffer::drain(&ws.name);

    let message = substitute(
        &load_prompt("notification.md"),
        &[
            ("{{ticket_id}}", &ticket.id),
            ("{{ticket_title}}", &ticket.title),
            ("{{ticket_status}}", status.as_ref()),
            ("{{transition_log}}", &transition_log),
            ("{{ticket_updates}}", &drained),
        ],
    );

    // Enqueue to the serialized Manager queue instead of spawning a task.
    // Routing is handled by the consumer loop via DB lookup.
    crate::manager_queue::manager_queue().enqueue(ManagerJob {
        content: message,
        workspace_name: ws.name,
        kind: JobKind::TicketNotify,
    });
}

pub async fn run_management() {
    // Reset in-flight tickets from previous runs (crash/restart recovery)
    if let Some(board) = BOARD.get()
        && let Err(e) = board.reset_inflight_tickets().await
    {
        error!(error = %e, "Failed to reset in-flight tickets");
    }

    let interval = Duration::from_secs(1);
    loop {
        if !crate::shutdown::sleep_or_shutdown(interval).await {
            break;
        }
        if let Err(e) = poll_round().await {
            error!(error = %e, "Board poller round failed");
        }
    }
}

/// Shared dispatch helper: log the ticket+workspace, then spawn the phase
/// dispatcher in a background task.
///
/// This is a plain `fn` (not `async`) because both `info!()` and
/// `tokio::spawn()` are synchronous operations — no `.await` needed.
///
/// # Panic safety
///
/// The dispatch runs inside an inner [`tokio::spawn`] so panics are caught via
/// the [`JoinHandle`](tokio::task::JoinHandle).  On panic the ticket transitions
/// to [`TicketPhase::Failed`] with notification so the manager can investigate.
fn spawn_dispatch(phase: PollPhase, ticket: Ticket, ws: Workspace) {
    let phase_info = phase.info();
    let target_phase = phase_info.claim_target;

    info!(
        ticket = %ticket.id,
        title = %ticket.title,
        workspace = %ws.name,
        "Dispatching {} ticket",
        phase_info.role_label,
    );

    // Wrap in Arc so the panic-recovery clone is a cheap refcount bump
    // instead of a deep copy of the entire comments Vec.
    let ticket = Arc::new(ticket);
    let ticket_for_failure = Arc::clone(&ticket);

    tokio::spawn(async move {
        // Spawn an inner task so panics inside dispatch() are caught via
        // the JoinHandle rather than silently swallowed by the outer spawn.
        let handle = tokio::spawn(async move {
            phase.dispatch(ticket, ws).await;
        });

        match handle.await {
            Ok(()) => {
                // Happy path — dispatch completed successfully and handled
                // its own transitions.
            }
            Err(join_error) => {
                // Dispatch panicked — terminal failure.
                // JoinError's Display impl includes the panic payload for
                // panicked tasks, and "cancelled" for cancelled tasks.
                error!(
                    ticket = %ticket_for_failure.id,
                    panic = %join_error,
                    "Dispatch panicked — transitioning ticket to Failed",
                );
                // Best-effort transition: the ticket may have been moved
                // externally while the dispatch was running.
                if let Err(e) = board()
                    .add_comment(
                        &ticket_for_failure.id,
                        "system",
                        &format!("❌ Dispatch panicked: {join_error}"),
                    )
                    .await
                {
                    warn!(
                        ticket = %ticket_for_failure.id,
                        error = %e,
                        "Failed to record panic comment",
                    );
                }
                if let Err(e) = transition_ticket(
                    &ticket_for_failure,
                    target_phase,
                    TicketPhase::Failed,
                    NotifyPolicy::Notify,
                )
                .await
                {
                    warn!(
                        ticket = %ticket_for_failure.id,
                        error = %e,
                        "Failed to transition ticket to Failed after dispatch panic",
                    );
                }
            }
        }
    });
}

/// Verifier-specific metadata, embedded directly in the [`PollPhase::VerifierCheck`]
/// variant and used as the parameter to [`dispatch_verifiers`]. Carries all
/// information needed for dispatch (role, source phase, prompt paths, phase
/// lifecycle) so no round-trip through [`PollPhase::info()`] is required.
#[derive(Copy, Clone)]
struct VerifierInfo {
    role: Role,
    log_label: &'static str,
    /// The ticket phase that triggers this verifier dispatch
    /// (e.g. [`TicketPhase::DiagnosticsDone`] for reviewers,
    /// [`TicketPhase::Reviewed`] for QA).
    source: TicketPhase,
    success_phase: TicketPhase,
    /// The ticket phase during which this verifier is active — the phase
    /// a ticket must be in for the verifier to run (e.g.
    /// [`TicketPhase::InReview`] for reviewers, [`TicketPhase::InQa`] for QA).
    active_phase: TicketPhase,
    prompt_template: &'static str,
    extraction_prompt_path: &'static str,
}

const REVIEWER_VI: VerifierInfo = VerifierInfo {
    role: Role::Reviewer,
    log_label: "Reviewers",
    source: TicketPhase::DiagnosticsDone,
    success_phase: TicketPhase::Reviewed,
    active_phase: TicketPhase::InReview,
    prompt_template: "review.md",
    extraction_prompt_path: "extraction/reviewer.md",
};

const QA_VI: VerifierInfo = VerifierInfo {
    role: Role::Qa,
    log_label: "QA",
    source: TicketPhase::Reviewed,
    success_phase: TicketPhase::QaPassed,
    active_phase: TicketPhase::InQa,
    prompt_template: "qa.md",
    extraction_prompt_path: "extraction/qa.md",
};

/// Static metadata for a single poll phase.
///
/// All phase-specific data lives here, sourced from the single [`PollPhase::info()`]
/// match — adding any phase requires one row in that match.
#[derive(Copy, Clone)]
struct PollPhaseInfo {
    source: TicketPhase,
    claim_target: TicketPhase,
    /// Whether this phase requires a clear pipeline (only one ticket at a
    /// time through development → review → QA).
    require_clear_pipeline: bool,
    role_label: &'static str,
}

/// A single poll phase: maps a `from → to` ticket transition to the agent
/// that handles it.
///
/// Phase metadata lives in [`PollPhase::info()`] — a single match expression
/// that returns all phase-specific data. The `VerifierCheck` variant carries
/// its `VerifierInfo` inline, with the `source` phase encoded in the data
/// rather than the variant identity (so reviewer and QA phases share one variant).
#[derive(Copy, Clone)]
enum PollPhase {
    BacklogAnalysis,
    EngineerDevelopment,
    DiagnosticsCheck,
    VerifierCheck(VerifierInfo),
}

impl PollPhase {
    /// Return all static metadata for this phase.
    fn info(self) -> PollPhaseInfo {
        match self {
            Self::BacklogAnalysis => PollPhaseInfo {
                source: TicketPhase::Backlog,
                claim_target: TicketPhase::Analysis,
                require_clear_pipeline: false,
                role_label: Role::Analyst.as_str(),
            },
            Self::EngineerDevelopment => PollPhaseInfo {
                source: TicketPhase::ReadyForDevelopment,
                claim_target: TicketPhase::InDevelopment,
                require_clear_pipeline: true,
                role_label: Role::Engineer.as_str(),
            },
            Self::DiagnosticsCheck => PollPhaseInfo {
                source: TicketPhase::InDiagnostics,
                claim_target: TicketPhase::InDiagnostics,
                require_clear_pipeline: false,
                role_label: "diagnostics",
            },
            Self::VerifierCheck(vi) => PollPhaseInfo {
                source: vi.source,
                claim_target: vi.active_phase,
                require_clear_pipeline: false,
                role_label: vi.role.as_str(),
            },
        }
    }

    /// Dispatch the appropriate agent(s) for a claimed ticket.
    async fn dispatch(self, ticket: Arc<Ticket>, ws: Workspace) {
        match self {
            Self::BacklogAnalysis => dispatch_backlog_analysts(ticket, ws).await,
            Self::EngineerDevelopment => dispatch_engineer(ticket, ws).await,
            Self::DiagnosticsCheck => dispatch_diagnostics(ticket, ws).await,
            Self::VerifierCheck(vi) => {
                dispatch_verifiers(ticket, ws, vi).await;
            }
        }
    }
}

/// Pipeline phases that use atomic source→claim_target claim transitions.
///
/// DiagnosticsCheck is intentionally excluded — it keeps the ticket in
/// InDiagnostics while running and guards re-dispatch via `assigned_to`.
/// QaPassed→Done uses a separate list-based dispatch because the commit
/// must succeed before transitioning to Done, so there is no atomic claim
/// to perform.
const CLAIM_PHASES: &[PollPhase] = &[
    PollPhase::BacklogAnalysis,
    PollPhase::EngineerDevelopment,
    PollPhase::VerifierCheck(REVIEWER_VI),
    PollPhase::VerifierCheck(QA_VI),
];

/// Run the given action for each ticket in `phase` for the named workspace.
///
/// Lists tickets via [`BoardStore::list_tickets_in_phase`], iterates, and logs
/// a structured error on failure. Replaces the identical 14-line match blocks
/// that previously appeared for Diagnostics and QaPassed dispatch.
async fn for_tickets_in_phase(phase: TicketPhase, ws_name: &str, mut action: impl FnMut(Ticket)) {
    match board().list_tickets_in_phase(phase, ws_name).await {
        Ok(tickets) => {
            for ticket in tickets {
                action(ticket);
            }
        }
        Err(e) => error!(workspace = ws_name, phase = %phase, error = %e, "Phase listing failed"),
    }
}

/// Run one poll round: claim actionable tickets and dispatch agents.
///
/// Single pass over workspaces — for each, attempt claims across all pipeline
/// phases, then handle DiagnosticsCheck and QaPassed. Previously phase-major
/// (all workspaces claim Backlog, then all claim Engineer, …); now workspace-major
/// (workspace A claims all phases, then workspace B, …). Correctness is preserved
/// because claims are atomic per-workspace and `require_clear_pipeline` gates
/// are checked within each workspace independently.
async fn poll_round() -> anyhow::Result<()> {
    let board = crate::board::store();

    let workspaces = match crate::workspace::store().list().await {
        Ok(ws_list) => ws_list,
        Err(e) => {
            error!(error = %e, "Failed to list workspaces");
            return Ok(());
        }
    };

    for ws in &workspaces {
        // 1. Claim for each pipeline phase (skip when the workspace is
        //    individually paused).
        //    On claim error we `break` out of the phase loop — this skips all
        //    remaining CLAIM_PHASES for this workspace and falls through to
        //    Diagnostics/QaPassed (which handle their own errors independently).
        //    A DB-down workspace won't block other workspaces; a transient claim
        //    failure won't generate log noise for every remaining phase.
        if !ws.paused {
            for &phase in CLAIM_PHASES {
                let info = phase.info();
                let ticket = match board
                    .claim_ticket_in_workspace(
                        info.source,
                        info.claim_target,
                        &ws.name,
                        info.require_clear_pipeline,
                    )
                    .await
                {
                    Ok(Some(t)) => {
                        // Buffer the claim transition. The returned ticket already
                        // has status = info.claim_target (from SQL RETURNING), so record
                        // the transition from info.source.
                        ticket_buffer::push(
                            &ws.name,
                            &t.id,
                            info.source.as_ref(),
                            t.status.as_ref(),
                        );
                        t
                    }
                    Ok(None) => continue,
                    Err(e) => {
                        error!(
                            workspace = %ws.name,
                            phase = %info.role_label,
                            error = %e,
                            "Claim failed, skipping remaining phases for workspace",
                        );
                        break;
                    }
                };
                spawn_dispatch(phase, ticket, ws.clone());
            }
        }

        // 2. Dispatch unassigned InDiagnostics tickets.
        //
        // DiagnosticsCheck does NOT use the claim loop because claim
        // transitions source→target atomically, but diagnostics keeps the
        // ticket in InDiagnostics while running (only transitions on
        // completion). Instead, we list InDiagnostics tickets for this
        // workspace directly and guard against re-dispatch via
        // `assigned_to IS NULL`.
        for_tickets_in_phase(TicketPhase::InDiagnostics, &ws.name, |ticket| {
            if ticket.assigned_to.is_some() {
                return; // already dispatched, still running
            }
            spawn_dispatch(PollPhase::DiagnosticsCheck, ticket, ws.clone());
        })
        .await;

        // 3. Auto-commit QaPassed tickets.
        //
        // Spawned via tokio::spawn (not synchronous .await) to prevent
        // git operations from blocking the entire poll loop. Unlike the
        // claim-based phases, there is no atomic source→target transition
        // — the ticket stays in QaPassed until the commit succeeds, so
        // re-dispatch is harmless (git status short-circuits a no-op).
        for_tickets_in_phase(TicketPhase::QaPassed, &ws.name, |ticket| {
            let ws = ws.clone();
            tokio::spawn(async move {
                finalize_qa_passed(ticket, ws).await;
            });
        })
        .await;
    }

    Ok(())
}

/// Run an Engineer agent to implement the ticket.
///
/// Guards with [`guard_phase_and_circuit_breaker`] (phase check + comment-count
/// circuit breaker) before starting. Gathers feedback comments from all roles
/// since the last engineer run and includes them in the agent prompt. After the
/// agent finishes, performs a post-run phase check to catch race conditions,
/// then transitions:
/// - InDiagnostics (buffer) on successful completion
/// - Failed (notify) if the agent failed or returned no output
async fn dispatch_engineer(ticket: Arc<Ticket>, ws: Workspace) {
    let session_key = ticket_session_key(&ticket.id, Role::Engineer.as_str());

    if !guard_phase_and_circuit_breaker(&ticket, TicketPhase::InDevelopment, "Engineer").await {
        return;
    }

    // Gather all new comments since the last engineer run
    let last_eng_pos = ticket
        .comments
        .iter()
        .rposition(|c| c.role == Role::Engineer.as_str());
    let feedback: Vec<&str> = ticket
        .comments
        .iter()
        .skip(last_eng_pos.map_or(0, |i| i + 1))
        .map(|c| c.content.as_str())
        .collect();

    let mut message = "Implement the ticket described in the system prompt.".to_string();
    if !feedback.is_empty() {
        let _ = write!(
            message,
            "\n\n---\nNew feedback to address:\n{}",
            feedback.join("\n---\n")
        );
    }

    if let Err(e) = board()
        .set_assigned_to(&ticket.id, Some(&session_key))
        .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Failed to set engineer assignee — agent will run unassigned"
        );
    }

    let (_agent, response) =
        run_agent(session_key, Role::Engineer, &ws, Some(&ticket), &message).await;

    // Post-run check still needed for race conditions during agent execution.
    if !is_ticket_in_phase(&ticket.id, TicketPhase::InDevelopment).await {
        return;
    }

    // Diagnostics are dispatched by the poll loop as a separate
    // PollPhase::DiagnosticsCheck — see poll_round().
    let (comment_text, target_phase, notify) = if let Some(ref text) = response {
        (
            text.as_str(),
            TicketPhase::InDiagnostics,
            NotifyPolicy::Buffer,
        )
    } else {
        ("Agent failed", TicketPhase::Failed, NotifyPolicy::Notify)
    };

    if let Err(e) = board()
        .add_comment(&ticket.id, Role::Engineer.as_str(), comment_text)
        .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Failed to record engineer comment",
        );
    }
    if let Err(e) =
        transition_ticket(&ticket, TicketPhase::InDevelopment, target_phase, notify).await
    {
        let verb = match target_phase {
            TicketPhase::InDiagnostics => "completed",
            _ => "failed",
        };
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Engineer {verb} but transition to {phase} failed — ticket stuck in {stuck}",
            phase = target_phase.as_ref(),
            stuck = TicketPhase::InDevelopment.as_ref(),
        );
        if let Err(e) = board().set_assigned_to(&ticket.id, None).await {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to clear assigned_to on transition error",
            );
        }
    }
}

// ── QA to Done (auto-commit) ───────────────────────────────────────────
//
// NOTE: finalize_qa_passed takes `Ticket` by value. It is called from
// `poll_round` via `tokio::spawn` (not `spawn_dispatch`) because:
// - There is no claim transition — the ticket stays in QaPassed until
//   commit succeeds, so transient failures are harmless (re-dispatched
//   next poll cycle).
// - spawn_dispatch's panic-recovery moves tickets to Failed, but the
//   correct behavior here is to stay in QaPassed for retry.
// - No Arc wrapping is needed because `Ticket` is moved by value into
//   the spawned task.

/// Transition a QaPassed ticket to Done with a descriptive reason.
async fn transition_qa_to_done(ticket: &Ticket, reason: &str) {
    info!(ticket = %ticket.id, "{reason}");
    if let Err(e) = transition_ticket(
        ticket,
        TicketPhase::QaPassed,
        TicketPhase::Done,
        NotifyPolicy::Notify,
    )
    .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "QA passed but transition to Done failed",
        );
    }
}

/// Auto-commit changes after QA passes and move the ticket to Done.
///
/// Checks for a dirty working tree via `git status --porcelain`:
/// - **Clean tree:** skips commit, transitions directly to Done with notification.
/// - **Dirty tree:** runs `git commit -m "<ticket title>"` via [`crate::diff_parse::run_git_commit`].
/// - **Commit failure:** ticket stays in QaPassed, poller retries next cycle.
/// - **Not a git repo / no git installed:** transitions to Done without commit.
async fn finalize_qa_passed(ticket: Ticket, ws: Workspace) {
    let repo_path = ws.as_path();

    if !crate::diff_parse::git_is_installed().await {
        transition_qa_to_done(&ticket, "Git not installed — moving to Done without commit").await;
        return;
    }

    if !crate::diff_parse::is_git_repo(repo_path).await {
        transition_qa_to_done(&ticket, "Not a git repo — moving to Done without commit").await;
        return;
    }

    // Check for working tree changes
    let has_changes = match crate::diff_parse::run_git_status(repo_path).await {
        Ok(output) => !output.trim().is_empty(),
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to check git status — staying in QaPassed for retry"
            );
            return;
        }
    };

    if !has_changes {
        transition_qa_to_done(
            &ticket,
            "Clean working tree — moving to Done without commit",
        )
        .await;
        return;
    }

    match crate::diff_parse::run_git_commit(repo_path, &ticket.title).await {
        Ok(commit_info) => {
            // Persist commit info (hash + line stats) before transitioning.
            if let Err(e) = board()
                .set_commit_info(
                    &ticket.id,
                    &commit_info.hash,
                    commit_info.lines_added,
                    commit_info.lines_removed,
                )
                .await
            {
                warn!(ticket = %ticket.id, "Failed to store commit info: {e}");
            }

            // Add system comment with human-readable commit summary.
            let short_hash = commit_info.hash.get(..7).unwrap_or(&commit_info.hash);
            let comment = format_commit_summary(
                short_hash,
                commit_info.lines_added,
                commit_info.lines_removed,
            );
            if let Err(e) = board().add_comment(&ticket.id, "system", &comment).await {
                warn!(ticket = %ticket.id, "Failed to add commit comment: {e}");
            }

            // Transition to Done.
            transition_qa_to_done(&ticket, &format!("Committed {short_hash}, moving to Done"))
                .await;
        }
        Err(e) => {
            error!(
                ticket = %ticket.id,
                error = %e,
                "Commit failed — staying in QaPassed for retry"
            );
        }
    }
}

// ── Git helpers ────────────────────────────────────────────────────────

/// Format a commit summary line for the ticket comment history.
///
/// Covers all combinations: no changes, only additions, only deletions,
/// or both.
fn format_commit_summary(short_hash: &str, added: i64, removed: i64) -> String {
    match (added, removed) {
        (0, 0) => format!("Committed as `{short_hash}` (no changes)"),
        (a, 0) => format!("Committed as `{short_hash}` (+{a})"),
        (0, r) => format!("Committed as `{short_hash}` (-{r})"),
        (a, r) => format!("Committed as `{short_hash}` (+{a}/-{r})"),
    }
}

// ── Post-development diagnostics ───────────────────────────────────────

/// Run diagnostics commands after the engineer completes development.
///
/// Called by [`PollPhase::DiagnosticsCheck`] via [`spawn_dispatch`].
/// Uses [`BoardStore::claim_diagnostics`] (atomic claim+phase check) to
/// prevent double-dispatch. Unlike dispatch_engineer (which is dispatched
/// from the atomic claim loop and already owns the ticket by the time its
/// dispatch runs), diagnostics keeps the ticket in InDiagnostics while
/// executing, so a separate atomic guard is needed to close the TOCTOU window.
/// Loads discovered diagnostics commands for the workspace and runs them
/// sequentially. Stops at the first failure. After execution, transitions
/// the ticket to either `DiagnosticsDone` (all passed) or `ReadyForDevelopment`
/// (any failure), unless the circuit breaker trips (see
/// [`DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD`]).
#[allow(clippy::too_many_lines)]
async fn dispatch_diagnostics(ticket: Arc<Ticket>, ws: Workspace) {
    match board().claim_diagnostics(&ticket.id).await {
        Ok(true) => {} // Claim succeeded — proceed.
        Ok(false) => {
            warn!(
                ticket = %ticket.id,
                "Diagnostics claim failed — ticket already claimed or moved out of InDiagnostics"
            );
            return;
        }
        Err(e) => {
            error!(
                ticket = %ticket.id,
                error = %e,
                "Diagnostics claim error — bailing out",
            );
            return;
        }
    }

    // 1. Load diagnostics commands for this workspace.
    let diag = match crate::workspace::store().get_diagnostics(&ws.name).await {
        Ok(Some(cmds)) if !cmds.is_empty() => Some(cmds),
        Ok(Some(_) | None) => None,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to load diagnostics for workspace — transitioning to DiagnosticsDone"
            );
            None
        }
    };

    let Some(diag) = diag else {
        if let Err(e) = transition_ticket(
            &ticket,
            TicketPhase::InDiagnostics,
            TicketPhase::DiagnosticsDone,
            NotifyPolicy::Buffer,
        )
        .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "No diagnostics commands — failed to transition to DiagnosticsDone",
            );
        }
        return;
    };

    // Check circuit breaker before running diagnostics.
    if trip_diagnostics_circuit_breaker_if_exceeded(&ticket).await {
        return;
    }

    // 2. Run commands sequentially in the prescribed order.

    let mut comment = String::from("🔍 Auto-diagnostics");
    let mut all_passed = true;
    let mut failed_at: &str = "";

    for (label, cmd_opt) in diag.commands() {
        let Some(cmd) = cmd_opt else {
            continue;
        };

        let _ = write!(comment, "\n\n{label} ({cmd}):\n");

        match ShellTool::new(ShellMode::Full)
            .execute(&ws, serde_json::json!({"command": cmd}))
            .await
        {
            Ok(output) => {
                // ShellTool returns Ok(String) for non-zero exits (annotation
                // appended). Exit 0 produces no annotation, so any occurrence
                // of "[exit status: " means non-zero exit or signal termination.
                let failed = output.contains("[exit status: ");
                let display = if output.is_empty() {
                    "(no output)".to_string()
                } else {
                    output
                };
                comment.push_str(&display);

                if failed {
                    all_passed = false;
                    failed_at = label;
                    break;
                }
            }
            Err(e) => {
                // Timeout or process launch failure.
                comment.push_str(&e.to_string());
                all_passed = false;
                failed_at = label;
                break;
            }
        }
    }

    // 3. Final outcome.
    let target = if all_passed {
        comment.push_str("\n\n---\n✅ All diagnostics passed");
        TicketPhase::DiagnosticsDone
    } else {
        let _ = write!(comment, "\n\n---\n❌ Diagnostics failed at {failed_at}");
        TicketPhase::ReadyForDevelopment
    };
    if let Err(e) = board()
        .add_comment(&ticket.id, "diagnostics", &comment)
        .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Failed to record diagnostics comment",
        );
    }

    if target == TicketPhase::ReadyForDevelopment {
        bounce_back_to_development(&ticket, TicketPhase::InDiagnostics, "Diagnostics").await;
    } else if let Err(e) = transition_ticket(
        &ticket,
        TicketPhase::InDiagnostics,
        target,
        NotifyPolicy::Buffer,
    )
    .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Diagnostics completed but transition to DiagnosticsDone \
             failed — clearing assigned_to for retry",
        );
        if let Err(e) = board().set_assigned_to(&ticket.id, None).await {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to clear assigned_to on transition error",
            );
        }
    }
}

/// Diagnostics-specific circuit breaker: counts prior diagnostics system
/// comments that indicate failures, and fails the ticket (plus pauses the
/// workspace) if the count exceeds [`DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD`]
/// (i.e., trip at ≥5 failures).
///
/// Re-fetches comments because the ticket snapshot may be stale.
/// Excludes comments already containing "Circuit breaker" to prevent
/// self-counting cascade on re-dispatch.
///
/// Returns `true` if the circuit breaker tripped (caller should abort);
/// `false` otherwise, including on comment fetch errors (fail-open).
/// On transition failure, still returns `true` to abort dispatch.
#[must_use]
async fn trip_diagnostics_circuit_breaker_if_exceeded(ticket: &Ticket) -> bool {
    run_circuit_breaker(
        ticket,
        TicketPhase::InDiagnostics,
        DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD,
        |comments| {
            comments
                .iter()
                .filter(|c| {
                    c.role == "diagnostics"
                        && c.content.starts_with("🔍 Auto-diagnostics")
                        && c.content.contains('❌')
                        && !c.content.contains("Circuit breaker")
                })
                .count()
        },
        |count| {
            format!(
                "🔍 Auto-diagnostics\n\n❌ Circuit breaker: {count} prior diagnostic \
                 failures. Failing ticket and pausing workspace."
            )
        },
        "Diagnostics",
    )
    .await
}

// ── Parallel agent helpers (shared) ─────────────────────────────────────

/// Result from a single parallel verifier agent.
struct ParallelVerdict {
    response: String,
    verdict: Option<crate::Verdict>,
}

/// Extract structured verdicts from parallel agent results.
/// Agents with empty responses get `verdict: None`; others are parsed via
/// [`crate::extraction::retry_extract_structured::<Verdict>`] using the provided extraction prompt.
/// All non-empty extractions run concurrently via [`join_all`].
async fn extract_parallel_verdicts(
    results: Vec<(crate::Agent, String)>,
    extraction_prompt: &str,
) -> Vec<ParallelVerdict> {
    let retry_prompt = crate::prompt::load_prompt("extraction/retry.md");

    let futures: Vec<_> = results
        .into_iter()
        .map(|(agent, response)| {
            let extraction_prompt = extraction_prompt.to_string();
            let retry_prompt = retry_prompt.clone();
            async move {
                if response.is_empty() {
                    return ParallelVerdict {
                        response,
                        verdict: None,
                    };
                }
                // KV cache preservation: `agent.extract_structured` uses the
                // agent's own parameters (model, temperature, reasoning_effort,
                // tools, provider routing) so the extraction call is byte-identical
                // to the original verifier agent call — the provider can reuse the
                // cached prefix.
                let verdict = agent
                    .extract_structured::<crate::Verdict>(&extraction_prompt, &retry_prompt, 5)
                    .await
                    .ok();
                ParallelVerdict { response, verdict }
            }
        })
        .collect();

    join_all(futures).await
}

/// Run [`PARALLEL_AGENT_COUNT`] agents of the same role in parallel, then extract structured verdicts
/// using the provided extraction prompt.
/// Session keys are formatted as `ticket_{ticket.id}_{role}_{i}_{suffix}`
/// where `suffix` is a unique 6-char NanoID for retry-cycle disambiguation.
/// Each agent creates its own CancellationToken and auto-registers.
async fn run_parallel_with_extraction(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    role: Role,
    prompt: &str,
    extraction_prompt: &str,
) -> Vec<ParallelVerdict> {
    let suffix = crate::generate_suffix();
    let futures: Vec<_> = (0..PARALLEL_AGENT_COUNT)
        .map(move |i| {
            let ticket = Arc::clone(ticket);
            let prompt = prompt.to_string();
            let ws = ws.clone();
            let base = ticket_session_key(&ticket.id, role.as_str());
            let session_key = format!("{base}_{i}_{suffix}");
            async move {
                let (agent, response) =
                    run_agent(session_key, role, &ws, Some(&ticket), &prompt).await;
                (agent, response.unwrap_or_default())
            }
        })
        .collect();
    let results = join_all(futures).await;
    extract_parallel_verdicts(results, extraction_prompt).await
}

/// Check whether a review or QA verdict passes (score at or above
/// [`REVIEW_QA_THRESHOLD`]). Returns `false` when the verdict is missing
/// or the score is below threshold.
#[must_use]
fn verdict_passes(verdict: Option<&crate::Verdict>) -> bool {
    verdict.is_some_and(|v| v.score >= REVIEW_QA_THRESHOLD)
}

/// Format a Verdict's critique and issues into a comment body string
/// using bullet-list style: critique followed by "Issues:\n- item1\n- item2".
fn format_verdict_body(verdict: &crate::Verdict) -> String {
    let mut text = verdict.critique.as_deref().unwrap_or_default().to_string();
    if !verdict.issues_detected.is_empty() {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str("Issues:\n");
        for issue in &verdict.issues_detected {
            let _ = writeln!(text, "- {issue}");
        }
    }
    text
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerdictFilter {
    /// Record ALL verdicts — passing and failing alike (used by analysts).
    All,
    /// Record only FAILING verdicts — passing verdicts are silently skipped (used by reviewers/QA).
    FailingOnly,
}

/// Determine whether a parallel verifier result warrants a comment,
/// and format the comment string if so.
///
/// Behaviour depends on `filter`:
/// - [`VerdictFilter::FailingOnly`]: returns `None` for passing verdicts
///   (score ≥ [`REVIEW_QA_THRESHOLD`]). For failing verdicts with an empty body,
///   a fallback message with the score is returned so engineers still get feedback.
/// - [`VerdictFilter::All`]: returns a comment for ALL verdicts (passing and failing).
///   For verdicts with an empty body, the same score fallback message is returned.
///
/// `comment_role` is used in the empty-response, extraction-failure, and
/// empty-body fallback branches for human-readable role attribution.
fn format_verdict_comment(
    r: &ParallelVerdict,
    comment_role: &str,
    filter: VerdictFilter,
) -> Option<String> {
    if let Some(v) = &r.verdict {
        // Analysts want ALL verdicts recorded; verifiers only want failing ones.
        if filter == VerdictFilter::FailingOnly && verdict_passes(Some(v)) {
            return None; // passing verdict, verifier path only
        }
        let comment = format_verdict_body(v);
        if comment.is_empty() {
            return Some(format!(
                "{} agent scored {}/10 with no specific critique provided.",
                comment_role, v.score
            ));
        }
        return Some(comment);
    }
    if r.response.is_empty() {
        Some(format!(
            "{comment_role} agent failed to produce a response — counting as a failure."
        ))
    } else {
        Some(format!(
            "{comment_role} produced a response but verdict extraction failed — \
             treating as a failure."
        ))
    }
}

/// Outcome of [`record_verdict_comments`]: counts how many comments were
/// attempted (non-filtered) and how many actually succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerdictCommentResult {
    /// Number of comments that passed the filter and were attempted.
    attempted: usize,
    /// Number of comments that were successfully written.
    written: usize,
}

/// Record per-agent verdict comments on a ticket.
///
/// Analysts record ALL verdicts (passing + failing) so that every
/// verdict is visible in the ticket discussion — this differs from
/// verifiers (reviewers / QA), which only record failing comments.
///
/// Returns a [`VerdictCommentResult`] with the number of comments attempted
/// and successfully written. Write failures are logged as warnings but do
/// not propagate — the caller can inspect the result to detect total failure.
async fn record_verdict_comments(
    ticket_id: &str,
    results: &[ParallelVerdict],
    role_str: &str,
    filter: VerdictFilter,
) -> VerdictCommentResult {
    let mut attempted = 0usize;
    let mut written = 0usize;
    for (i, r) in results.iter().enumerate() {
        let role_label = format!("{role_str}_{}", i + 1);
        if let Some(comment) = format_verdict_comment(r, &role_label, filter) {
            attempted += 1;
            if let Err(e) = board().add_comment(ticket_id, &role_label, &comment).await {
                warn!(
                    ticket = ticket_id,
                    role = %role_label,
                    error = %e,
                    "Failed to record verdict comment",
                );
            } else {
                written += 1;
            }
        }
    }
    VerdictCommentResult { attempted, written }
}

// ── Backlog Analysis ──────────────────────────────────────────────────

/// Spawn 3 parallel analyst agents to research a backlog ticket.
/// All verdicts are recorded as comments, then the ticket transitions to:
/// - Planning (notify) when ALL analysts pass (≥ `ANALYSIS_THRESHOLD`/10)
/// - Paused (notify) when any analyst fails, with a comment listing the counts
///
/// Before spawning agents, [`guard_phase_and_circuit_breaker`] checks the phase and
/// trips the comment-count circuit breaker (which may transition the ticket to
/// Failed and pause the workspace). Returns `false` when the caller should abort.
async fn dispatch_backlog_analysts(ticket: Arc<Ticket>, ws: Workspace) {
    if !guard_phase_and_circuit_breaker(&ticket, TicketPhase::Analysis, "Analysts").await {
        return;
    }

    let message = load_prompt("analyze.md");
    let extraction_prompt = load_prompt("extraction/analyst.md");
    let parallel_results =
        run_parallel_with_extraction(&ticket, &ws, Role::Analyst, &message, &extraction_prompt)
            .await;

    // Post-run check still needed for race conditions during agent execution.
    if !is_ticket_in_phase(&ticket.id, TicketPhase::Analysis).await {
        return;
    }

    handle_analyst_verdicts(&ticket, &parallel_results).await;
}

/// Evaluate analyst verdicts and transition the ticket:
///
/// Records per-analyst comments (if verdict exists), counts responses and
/// extractions via post-loop iterators, then transitions:
/// - to Planning (notify) if ALL analysts passed (≥ `ANALYSIS_THRESHOLD`/10)
/// - to Paused (notify) if any analyst failed, with a comment listing the counts
async fn handle_analyst_verdicts(ticket: &Ticket, results: &[ParallelVerdict]) {
    // Record per-analyst comments.
    // Analysts record ALL verdicts (passing + failing) — see `record_verdict_comments`.
    let verdict_result = record_verdict_comments(
        &ticket.id,
        results,
        Role::Analyst.as_str(),
        VerdictFilter::All,
    )
    .await;

    let nonempty_count = results.iter().filter(|r| !r.response.is_empty()).count();
    let total = results.len();
    let mut lgtm = 0usize;
    let mut minor_issues = 0usize;
    let mut potential_blockers = 0usize;
    let mut missing_analysis = 0usize;

    for r in results {
        match &r.verdict {
            Some(v) if v.score >= ANALYSIS_THRESHOLD && v.issues_detected.is_empty() => lgtm += 1,
            Some(v) if v.score >= ANALYSIS_THRESHOLD => minor_issues += 1,
            Some(_) => potential_blockers += 1,
            None => missing_analysis += 1,
        }
    }

    let summary = build_analyst_summary(
        total,
        lgtm,
        minor_issues,
        potential_blockers,
        missing_analysis,
    );
    if let Err(e) = board().add_comment(&ticket.id, "system", &summary).await {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Failed to record analyst summary comment",
        );
    }

    let extracted_count = total - missing_analysis;
    let passing_count = lgtm + minor_issues;

    // Compare against PARALLEL_AGENT_COUNT (not extracted_count) intentionally:
    // a missing/empty verdict is treated as non-passing — all dispatched
    // analysts must produce passing verdicts for the ticket to proceed.
    let all_passed = passing_count == PARALLEL_AGENT_COUNT;

    let target = if all_passed {
        TicketPhase::Planning
    } else {
        TicketPhase::Paused
    };

    // Log if all verdict comments failed (DB issue), but don't pause —
    // analysts are advisory, not gating, so pausing would be excessive.
    if verdict_result.attempted > 0 && verdict_result.written == 0 {
        warn!(
            ticket = %ticket.id,
            "Analyst verdict comments: {attempted} expected, 0 written — DB may be unavailable",
            attempted = verdict_result.attempted,
        );
    }

    if let Err(e) =
        transition_ticket(ticket, TicketPhase::Analysis, target, NotifyPolicy::Notify).await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Analyst verdicts completed but transition to {phase} failed — ticket stuck in {stuck}",
            phase = target.as_ref(),
            stuck = TicketPhase::Analysis.as_ref(),
        );
        return;
    }
    if all_passed {
        info!(
            ticket = %ticket.id,
            nonempty_count,
            "Backlog analysis complete — all analysts passed (≥ {ANALYSIS_THRESHOLD}/10)",
        );
    } else {
        info!(
            ticket = %ticket.id,
            nonempty_count,
            extracted_count,
            passing_count,
            "Backlog analysis incomplete — paused ({nonempty_count}/{PARALLEL_AGENT_COUNT} responded, \
             {extracted_count} extracted, {passing_count} passed)",
        );
    }
}

/// Build a natural-language summary of analyst verdict categories.
///
/// Categorizes each analyst as LGTM, minor issues, potential blockers, or missing
/// analysis. Only categories with non-zero counts appear in the description.
///
/// Label strings must not start with a leading space — `format!` inserts one
/// between count and label automatically when using the "All {label}" form.
fn build_analyst_summary(
    total: usize,
    lgtm: usize,
    minor_issues: usize,
    potential_blockers: usize,
    missing_analysis: usize,
) -> String {
    let description = [
        (lgtm, "LGTM"),
        (minor_issues, "found minor issues"),
        (potential_blockers, "flagged potential blockers"),
        (missing_analysis, "provided no analysis"),
    ]
    .iter()
    .filter(|&&(count, _label)| count > 0)
    .map(|&(count, label)| {
        if count == total {
            format!("All {label}")
        } else {
            format!("{count} {label}")
        }
    })
    .collect::<Vec<_>>()
    .join(", ");

    format!("{total} analysts reviewed this ticket. {description}.")
}

// ── Shared Circuit Breaker ──────────────────────────────

/// Shared circuit breaker skeleton: fetch comments, count, compare to threshold,
/// add a system comment (first, for crash-safety), then transition to
/// [`TicketPhase::Failed`].
///
/// Both concrete breakers (diagnostics and general) delegate to this helper,
/// supplying their counting logic, threshold, comment format, and log label via
/// parameters and closures. This eliminates ~80% structural duplication while
/// preserving exact behavioral semantics.
///
/// The workspace is paused automatically via [`notify_ticket`], called from
/// [`transition_ticket`] with [`NotifyPolicy::Notify`].
///
/// # Self-counting prevention
///
/// The `count_fn` closure is responsible for filtering out comments that the
/// breaker itself produces — otherwise the breaker would count its own comments
/// on subsequent iterations and re-trip immediately. For the diagnostics breaker
/// specifically, `count_fn` excludes comments containing `"Circuit breaker"`,
/// and the corresponding `comment_text` closure MUST produce a comment that
/// includes that exact substring. A mismatch between these two closures creates
/// a self-counting cascade on re-dispatch.
///
/// # Return value
///
/// Returns `true` if the breaker tripped — the caller MUST abort dispatch.
/// Returns `true` even on transition failure (the caller should still abort
/// rather than dispatching an agent to a stale or unreachable ticket).
/// Returns `false` only when the count does not exceed `threshold`.
///
/// # Parameters
///
/// * `expected` — the phase the ticket must currently be in for the transition
///   to succeed (passed through to [`transition_ticket`]).
/// * `threshold` — the count at which the breaker trips (using `>` comparison).
/// * `count_fn` — extracts the count from the fetched comment list. Responsible
///   for its own filtering (including self-counting prevention).
/// * `comment_text` — formats the system comment body given the count.
#[must_use]
async fn run_circuit_breaker(
    ticket: &Ticket,
    expected: TicketPhase,
    threshold: usize,
    count_fn: impl Fn(&[TicketComment]) -> usize,
    comment_text: impl Fn(usize) -> String,
    log_label: &str,
) -> bool {
    let comments = match board().get_comments(&ticket.id).await {
        Ok(c) => c,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to fetch comments for circuit breaker — proceeding anyway"
            );
            return false;
        }
    };

    let count = count_fn(&comments);

    if count <= threshold {
        return false;
    }

    info!(
        ticket = %ticket.id,
        count,
        threshold,
        log_label,
        "Circuit breaker tripped at {count}/{threshold} ({log_label}) — failing ticket and pausing workspace"
    );

    if let Err(e) = board()
        .add_comment(&ticket.id, "system", &comment_text(count))
        .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Failed to record circuit breaker comment",
        );
    }

    if let Err(e) =
        transition_ticket(ticket, expected, TicketPhase::Failed, NotifyPolicy::Notify).await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Circuit breaker tripped but transition to Failed failed",
        );
        return true;
    }

    // Workspace auto-pause: `transition_ticket` (called with
    // `NotifyPolicy::Notify`) pauses the workspace automatically.

    true
}

/// Trip the circuit breaker if a ticket has accumulated excessive activity.
///
/// Counts ALL ticket comments (system, analyst, engineer, reviewer, QA alike)
/// without filtering by role or type. This is intentional — the circuit breaker
/// measures total ticket activity, not review/QA cycles specifically. Any ticket
/// that generates excessive churn of any kind gets failed.
///
/// If the comment count exceeds [`CIRCUIT_BREAKER_COMMENT_THRESHOLD`], fail the ticket
/// (via [`TicketPhase::Failed`]) and pause the workspace (via [`notify_ticket`]).
/// Return `true` (caller should early-return). On transition failure, still returns
/// `true` to abort dispatch.
#[must_use]
async fn trip_circuit_breaker_if_exceeded(
    ticket: &Ticket,
    expected: TicketPhase,
    log_label: &str,
) -> bool {
    run_circuit_breaker(
        ticket,
        expected,
        CIRCUIT_BREAKER_COMMENT_THRESHOLD,
        <[TicketComment]>::len,
        |count| {
            format!(
                "Failed after {count} comments — ticket has accumulated too many comments \
                 (circuit breaker, threshold: {CIRCUIT_BREAKER_COMMENT_THRESHOLD}). \
                 Workspace paused for human investigation."
            )
        },
        log_label,
    )
    .await
}

/// Process parallel verifier results: add failing comments, determine pass/fail,
/// and update ticket status accordingly.
///
/// Notification does NOT fire on pass — verifiers transition without notifying.
/// Instead, notification fires when the ticket reaches Done (after the
/// QaPassed commit succeeds in [`finalize_qa_passed`]).
///
/// If any verifier fails (score below `REVIEW_QA_THRESHOLD`) → `ReadyForDevelopment`
/// (the circuit breaker check is handled *before* dispatch by
/// [`guard_phase_and_circuit_breaker`], so only the bounce-back is needed here).
/// If all pass → the verifier's `success_phase`.
async fn process_verdict_results(
    ticket: &Ticket,
    results: &[ParallelVerdict],
    verifier: VerifierInfo,
) {
    // Record failing comments
    let verdict_result = record_verdict_comments(
        &ticket.id,
        results,
        verifier.role.as_str(),
        VerdictFilter::FailingOnly,
    )
    .await;

    // Determine outcome
    let any_failed = results.iter().any(|r| !verdict_passes(r.verdict.as_ref()));

    // Total write failure takes precedence over bounce-back: if comments
    // were expected (attempted > 0) but nothing was written, the DB may be
    // unavailable. Pausing the ticket prevents silent LLM-credit loop where
    // the engineer operates blind (no critique visible) and the circuit
    // breaker can't detect the churn.
    if verdict_result.attempted > 0 && verdict_result.written == 0 {
        warn!(
            ticket = %ticket.id,
            "{label} verdict comments: {attempted} expected, 0 written — DB may be unavailable",
            label = verifier.log_label,
            attempted = verdict_result.attempted,
        );

        // Best-effort system comment — the pause transition does not depend on it.
        if let Err(e) = board()
            .add_comment(
                &ticket.id,
                "system",
                &format!(
                    "All {n} {label} verdict comment writes failed — DB may be unavailable. \
                     Ticket paused.",
                    n = verdict_result.attempted,
                    label = verifier.log_label,
                ),
            )
            .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to write system comment about DB failure for {label}",
                label = verifier.log_label,
            );
        }

        if let Err(e) = transition_ticket(
            ticket,
            verifier.active_phase,
            TicketPhase::Paused,
            NotifyPolicy::Notify,
        )
        .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "{label} verdict DB failure: transition to Paused also failed — \
                 ticket stays in {phase} (safe, no re-dispatch)",
                label = verifier.log_label,
                phase = verifier.active_phase.as_ref(),
            );
        }
        return;
    }

    if any_failed {
        bounce_back_to_development(ticket, verifier.active_phase, verifier.log_label).await;
    } else if let Err(e) = transition_ticket(
        ticket,
        verifier.active_phase,
        verifier.success_phase,
        NotifyPolicy::Buffer,
    )
    .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "{role} verdicts completed but transition to {phase} failed — ticket stuck in {stuck}",
            role = verifier.log_label,
            phase = verifier.success_phase.as_ref(),
            stuck = verifier.active_phase.as_ref(),
        );
    } else {
        info!(
            ticket = %ticket.id,
            "{log_label}: all passed (≥ {REVIEW_QA_THRESHOLD}/10)",
            log_label = verifier.log_label,
        );
    }
}

/// Shared dispatch logic for parallel verifiers (reviewers and QA).
/// Fetches the engineer's last comment, builds a prompt from the template,
/// runs [`PARALLEL_AGENT_COUNT`] parallel verifiers of the given role, and processes the verdicts.
async fn dispatch_verifiers(ticket: Arc<Ticket>, ws: Workspace, vi: VerifierInfo) {
    // Pre-agent guard: check phase and trip circuit breaker early to
    // avoid wasting LLM API calls on tickets with excessive churn.
    if !guard_phase_and_circuit_breaker(&ticket, vi.active_phase, vi.log_label).await {
        return;
    }

    let engineer_response = ticket
        .comments
        .iter()
        .rev()
        .find(|c| c.role == Role::Engineer.as_str())
        .map(|c| &c.content)
        .map_or("(no output)", String::as_str);

    let prompt = substitute(
        &crate::prompt::load_prompt(vi.prompt_template),
        &[("{{agent_response}}", engineer_response)],
    );

    let extraction_prompt = crate::prompt::load_prompt(vi.extraction_prompt_path);
    let results =
        run_parallel_with_extraction(&ticket, &ws, vi.role, &prompt, &extraction_prompt).await;

    // Post-run check still needed for race conditions during agent execution.
    if !is_ticket_in_phase(&ticket.id, vi.active_phase).await {
        return;
    }

    // If every verifier agent failed to produce a verdict (all crashed, timed
    // out, or returned unparseable output), this is a terminal failure — retrying
    // would waste credits on a fundamentally broken dispatch.
    let all_failed = results.iter().all(|r| r.verdict.is_none());
    if all_failed {
        if let Err(e) = board()
            .add_comment(
                &ticket.id,
                "system",
                &format!(
                    "❌ All {label} agents failed to produce verdicts — \
                     ticket marked as Failed.",
                    label = vi.log_label,
                ),
            )
            .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                role = %vi.log_label,
                "Failed to record all-{label}-failed system comment",
                label = vi.log_label,
            );
        }
        if let Err(e) = transition_ticket(
            &ticket,
            vi.active_phase,
            TicketPhase::Failed,
            NotifyPolicy::Notify,
        )
        .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                role = %vi.log_label,
                "All {label} agents failed but transition to Failed also failed",
                label = vi.log_label,
            );
        }
        return;
    }

    process_verdict_results(&ticket, &results, vi).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::DEFAULT_TICKET_PHASE;
    use crate::util::test::init_test_stores;
    use crate::workspace::test_ws_named;

    /// Verify that `guard_phase_and_circuit_breaker` rejects a ticket
    /// whose phase does not match `expected`. This validates that the
    /// pre-agent guard works correctly for `dispatch_verifiers` (and all
    /// other agent-spawning dispatch functions).
    #[tokio::test]
    async fn guard_phase_mismatch_rejected() {
        init_test_stores().await;

        let ws = test_ws_named("/tmp/test", "test");
        let ticket_id = board()
            .create_ticket("Test", "desc", &ws, DEFAULT_TICKET_PHASE, &[], "test", None)
            .await
            .expect("create_ticket");

        // Transition to InDevelopment so we have a known phase
        board()
            .transition_to(
                &ticket_id,
                Some(TicketPhase::Backlog),
                TicketPhase::InDevelopment,
                None,
            )
            .await
            .expect("transition_to");

        let ticket = board()
            .get_ticket(&ticket_id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");

        // Call with wrong phase — guard should reject immediately
        assert!(
            !guard_phase_and_circuit_breaker(&ticket, TicketPhase::InReview, "test_label").await,
            "guard_phase_and_circuit_breaker must reject a phase mismatch"
        );

        // Call with correct phase and 0 comments (below threshold) — guard should pass
        assert!(
            guard_phase_and_circuit_breaker(&ticket, TicketPhase::InDevelopment, "test_label")
                .await,
            "guard_phase_and_circuit_breaker must pass when phase matches and comments are below threshold"
        );
    }

    /// Verify that `record_verdict_comments` correctly counts attempted and
    /// written comments based on verdict filter and write success.
    #[tokio::test]
    async fn record_verdict_comments_counts() {
        init_test_stores().await;

        let ws = test_ws_named("/tmp/test", "test");
        let ticket_id = board()
            .create_ticket("Test", "desc", &ws, DEFAULT_TICKET_PHASE, &[], "test", None)
            .await
            .expect("create_ticket");

        // ── FailingOnly with all-passing verdicts ──
        // Should produce attempted=0, written=0 (no comments to write).
        let passing_verdict = crate::Verdict {
            score: REVIEW_QA_THRESHOLD, // 9/10 — passes
            critique: None,
            issues_detected: vec![],
        };
        let results = vec![ParallelVerdict {
            response: "Looks good.".into(),
            verdict: Some(passing_verdict),
        }];
        let result = record_verdict_comments(
            &ticket_id,
            &results,
            Role::Reviewer.as_str(),
            VerdictFilter::FailingOnly,
        )
        .await;
        assert_eq!(
            result.attempted, 0,
            "passing verdicts with FailingOnly filter should produce 0 attempted"
        );
        assert_eq!(
            result.written, 0,
            "passing verdicts with FailingOnly filter should produce 0 written"
        );

        // ── FailingOnly with a failing verdict ──
        // Should produce attempted=1, written=1.
        let failing = crate::Verdict {
            score: 3, // below threshold
            critique: Some("Missing error handling.".into()),
            issues_detected: vec!["No timeout check".into()],
        };
        let results = vec![ParallelVerdict {
            response: "Has issues.".into(),
            verdict: Some(failing),
        }];
        let result = record_verdict_comments(
            &ticket_id,
            &results,
            Role::Reviewer.as_str(),
            VerdictFilter::FailingOnly,
        )
        .await;
        assert_eq!(result.attempted, 1, "failing verdict should be attempted");
        assert_eq!(result.written, 1, "failing verdict should be written");

        // Verify the comment was actually stored.
        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert_eq!(comments.len(), 1, "one comment should exist");
        assert_eq!(comments[0].role, "reviewer_1");

        // ── All filter (analyst path) ──
        // Should produce attempted=2, written=2 (both verdicts recorded).
        let pass = crate::Verdict {
            score: 10,
            critique: Some("Excellent analysis.".into()),
            issues_detected: vec![],
        };
        let fail = crate::Verdict {
            score: 4,
            critique: Some("Needs more research.".into()),
            issues_detected: vec!["Missing citations".into()],
        };
        let results = vec![
            ParallelVerdict {
                response: "Agent 1 response.".into(),
                verdict: Some(pass),
            },
            ParallelVerdict {
                response: "Agent 2 response.".into(),
                verdict: Some(fail),
            },
        ];
        let result = record_verdict_comments(
            &ticket_id,
            &results,
            Role::Analyst.as_str(),
            VerdictFilter::All,
        )
        .await;
        assert_eq!(
            result.attempted, 2,
            "All filter should attempt both verdicts"
        );
        assert_eq!(result.written, 2, "All filter should write both verdicts");

        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert_eq!(comments.len(), 3, "now three comments total");
    }

    /// Regression: verify `process_verdict_results` does not panic when
    /// total comment-write failure occurs (attempted > 0, written == 0).
    ///
    /// This exercises the pause-on-total-failure path that transitions the
    /// ticket to `Paused` when `record_verdict_comments` reports no successful
    /// writes despite non-zero attempts.
    ///
    /// The test uses a `Ticket` struct with a non-existent ID (ghost ticket).
    /// Because [`open_with_schema`](crate::turso::open_with_schema) sets
    /// `PRAGMA foreign_keys = ON`, `add_comment` fails with a FK violation
    /// when inserting into `ticket_comments` — the referenced `ticket_id`
    /// does not exist in `tickets`. This causes `record_verdict_comments`
    /// to return `attempted = 1, written = 0`, entering the pause branch.
    ///
    /// The dangling transaction from the aborted `add_comment` is self-healed
    /// by [`Connection::maybe_rollback_dangling_tx`] on the next write
    /// operation, so no global state leaks to sibling tests.
    #[tokio::test]
    async fn pause_branch_does_not_panic_on_total_failure() {
        init_test_stores().await;

        // Build a Ticket with a non-existent ID (ghost ticket).
        // FK enforcement is active (set by open_with_schema), so
        // add_comment will fail with a FK violation.
        let ticket = Ticket {
            id: crate::generate_id(),
            title: String::new(),
            description: String::new(),
            status: TicketPhase::InReview,
            assigned_to: None,
            workspace_name: "test".into(),
            created_at: String::new(),
            updated_at: String::new(),
            comments: vec![],
            prerequisites: vec![],
            supersedes: None,
            superseded_by: None,
            commit_hash: None,
            lines_added: None,
            lines_removed: None,
            reporter: "test".into(),
            is_archived: false,
            pipeline_reservation: false,
        };

        // Build a failing verdict — score below REVIEW_QA_THRESHOLD (9/10)
        // ensures format_verdict_comment returns Some(...).
        let failing = crate::Verdict {
            score: 3,
            critique: Some("Missing error handling.".into()),
            issues_detected: vec!["No timeout check".into()],
        };
        let results = vec![ParallelVerdict {
            response: "Has issues.".into(),
            verdict: Some(failing),
        }];

        let vi = VerifierInfo {
            role: Role::Reviewer,
            log_label: "Reviewers",
            source: TicketPhase::DiagnosticsDone,
            success_phase: TicketPhase::Reviewed,
            active_phase: TicketPhase::InReview,
            prompt_template: "review.md",
            extraction_prompt_path: "extraction/reviewer.md",
        };

        // This should not panic — all DB operation failures
        // (add_comment, transition_to) are caught and logged internally.
        process_verdict_results(&ticket, &results, vi).await;
    }
}
