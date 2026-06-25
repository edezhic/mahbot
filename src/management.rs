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
            let _ = board().set_assigned_to(&ticket.id, None).await;
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
                let _ = board()
                    .add_comment(
                        &ticket_for_failure.id,
                        "system",
                        &format!("❌ Dispatch panicked: {join_error}"),
                    )
                    .await;
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

    let _ = board()
        .set_assigned_to(&ticket.id, Some(&session_key))
        .await;

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

    let _ = board()
        .add_comment(&ticket.id, Role::Engineer.as_str(), comment_text)
        .await;
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
        let _ = board().set_assigned_to(&ticket.id, None).await;
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

    // Determine whether to notify immediately or buffer: if other active
    // tickets still exist in this workspace, buffer the Done transition so
    // the Manager only gets one notification when the last ticket finishes.
    // Active tickets = PIPELINE_BLOCKING_STATUSES + ReadyForDevelopment.
    //
    // Race condition: multiple QaPassed tickets in the same workspace are
    // finalized concurrently (tokio::spawn in poll_round). Both may see each
    // other as active and both buffer. In this scenario all tickets are already
    // Done in the database — the only consequence is delayed notifications
    // until the next UserMessage drains the buffer.
    let notify_policy = match board()
        .has_active_tickets_excluding(&ticket.workspace_name, &ticket.id)
        .await
    {
        Ok(true) => {
            debug!(
                ticket = %ticket.id,
                workspace = %ticket.workspace_name,
                "Other active tickets remain — buffering Done notification",
            );
            NotifyPolicy::Buffer
        }
        Ok(false) => {
            // This is the last active ticket in the workspace — notify
            // immediately. Draining the buffer also delivers any previously
            // buffered Done transitions from this workspace.
            NotifyPolicy::Notify
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                workspace = %ticket.workspace_name,
                error = %e,
                "Failed to check active tickets — notifying to be safe",
            );
            NotifyPolicy::Notify
        }
    };

    if let Err(e) = transition_ticket(
        ticket,
        TicketPhase::QaPassed,
        TicketPhase::Done,
        notify_policy,
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
            let _ = board()
                .set_commit_info(
                    &ticket.id,
                    &commit_info.hash,
                    commit_info.lines_added,
                    commit_info.lines_removed,
                )
                .await;

            // Add system comment with human-readable commit summary.
            let short_hash = commit_info.hash.get(..7).unwrap_or(&commit_info.hash);
            let comment = format_commit_summary(
                short_hash,
                commit_info.lines_added,
                commit_info.lines_removed,
            );
            let _ = board().add_comment(&ticket.id, "system", &comment).await;

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
    let _ = board()
        .add_comment(&ticket.id, "diagnostics", &comment)
        .await;

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
        let _ = board().set_assigned_to(&ticket.id, None).await;
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

/// Record per-agent verdict comments on a ticket.
///
/// Analysts record ALL verdicts (passing + failing) so that every
/// verdict is visible in the ticket discussion — this differs from
/// verifiers (reviewers / QA), which only record failing comments.
async fn record_verdict_comments(
    ticket_id: &str,
    results: &[ParallelVerdict],
    role_str: &str,
    filter: VerdictFilter,
) {
    for (i, r) in results.iter().enumerate() {
        let role_label = format!("{role_str}_{}", i + 1);
        if let Some(comment) = format_verdict_comment(r, &role_label, filter) {
            let _ = board().add_comment(ticket_id, &role_label, &comment).await;
        }
    }
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
    record_verdict_comments(
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
    let _ = board().add_comment(&ticket.id, "system", &summary).await;

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

    let _ = board()
        .add_comment(&ticket.id, "system", &comment_text(count))
        .await;

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

    // Pause all other ReadyForDevelopment tickets in the same workspace to
    // prevent them from auto-starting after the workspace is unpaused. The
    // human must manually move them back to ReadyForDevelopment after
    // investigating the root cause of the breaker trip.
    //
    // There is a small race window: between listing ReadyForDevelopment
    // tickets here and transitioning them individually, a concurrent poll
    // cycle could claim one. This is rare in practice (dispatch tasks spawn
    // after the claim loop completes) and the CAS guard in transition_to
    // handles the transition gracefully.
    let other_tickets = match board()
        .list_tickets_in_phase(TicketPhase::ReadyForDevelopment, &ticket.workspace_name)
        .await
    {
        Ok(tickets) => tickets,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                workspace = %ticket.workspace_name,
                error = %e,
                "Failed to list ReadyForDevelopment tickets for pausing \
                 — breaker trip proceeds without pausing siblings",
            );
            return true;
        }
    };

    let pause_comment = format!(
        "Paused due to circuit breaker trip on {}: {}. Unpause manually after investigation.",
        ticket.id, ticket.title,
    );

    // Filter out the tripped ticket: defense-in-depth. The tripped ticket
    // was already transitioned to Failed above and shouldn't appear in the
    // ReadyForDevelopment results, but the filter keeps us safe if a
    // concurrent race or future refactor changes the timing.
    for other in other_tickets.iter().filter(|t| t.id != ticket.id) {
        // If the transition fails (e.g., ticket was already claimed or moved
        // externally), skip it and continue with the remaining tickets.
        if let Err(e) = transition_ticket(
            other,
            TicketPhase::ReadyForDevelopment,
            TicketPhase::Paused,
            NotifyPolicy::Buffer,
        )
        .await
        {
            debug!(
                other_ticket = %other.id,
                error = %e,
                "Failed to pause other ReadyForDevelopment ticket — likely raced by external move",
            );
            continue;
        }

        let _ = board()
            .add_comment(&other.id, "system", &pause_comment)
            .await;
    }

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
    record_verdict_comments(
        &ticket.id,
        results,
        verifier.role.as_str(),
        VerdictFilter::FailingOnly,
    )
    .await;

    // Determine outcome
    let any_failed = results.iter().any(|r| !verdict_passes(r.verdict.as_ref()));

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
        let _ = board()
            .add_comment(
                &ticket.id,
                "system",
                &format!(
                    "❌ All {label} agents failed to produce verdicts — \
                     ticket marked as Failed.",
                    label = vi.log_label,
                ),
            )
            .await;
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

    /// Verify that when the circuit breaker trips on a ticket, all other
    /// ReadyForDevelopment tickets in the same workspace are paused with a
    /// system comment referencing the tripped ticket. Tickets in other
    /// workspaces must not be affected.
    #[tokio::test]
    async fn circuit_breaker_pauses_other_ready_for_development_tickets() {
        init_test_stores().await;
        // Initialize workspace store if not already done by a sibling test.
        // Required by transition_ticket for ticket buffer resolution.
        // The workspace does not need to exist in the store
        // (resolve_ticket_workspace returns None gracefully for
        // non-existent workspaces, merely skipping the buffer push).
        // Initialize workspace store — race-safe: if another test already
        // initialized it, init_global returns an error which we ignore.
        if crate::workspace::WORKSPACES.get().is_none() {
            let _ = crate::workspace::init_global().await;
        }

        let ws_a = test_ws_named("/ws_a", "ws_a");
        let ws_b = test_ws_named("/ws_b", "ws_b");

        // Create ticket A in workspace A — this will trip the circuit breaker.
        let ticket_a_id = board()
            .create_ticket(
                "Trip Ticket",
                "desc",
                &ws_a,
                TicketPhase::ReadyForDevelopment,
                &[],
                "test",
                None,
            )
            .await
            .expect("create_ticket A");

        // Create ticket B in workspace A — this should be paused when A trips.
        let ticket_b_id = board()
            .create_ticket(
                "Victim Ticket",
                "desc",
                &ws_a,
                TicketPhase::ReadyForDevelopment,
                &[],
                "test",
                None,
            )
            .await
            .expect("create_ticket B");

        // Create ticket C in workspace B — this must NOT be paused.
        let ticket_c_id = board()
            .create_ticket(
                "Other Workspace Ticket",
                "desc",
                &ws_b,
                TicketPhase::ReadyForDevelopment,
                &[],
                "test",
                None,
            )
            .await
            .expect("create_ticket C");

        // Add a comment to ticket A so the circuit breaker has something to count.
        board()
            .add_comment(&ticket_a_id, "system", "Some comment")
            .await
            .expect("add_comment to A");

        // Fetch ticket A and trip the circuit breaker with threshold 0.
        let ticket_a = board()
            .get_ticket(&ticket_a_id)
            .await
            .expect("get_ticket A")
            .expect("ticket A exists");

        let tripped = run_circuit_breaker(
            &ticket_a,
            TicketPhase::ReadyForDevelopment,
            0,                         // threshold = 0, so 1 comment > 0 trips
            |comments| comments.len(), // count all comments
            |count| format!("Breaker tripped at {count}"),
            "test",
        )
        .await;

        assert!(tripped, "circuit breaker should have tripped");

        // ── Verify ticket A is Failed ──
        {
            let ticket_a = board()
                .get_ticket(&ticket_a_id)
                .await
                .expect("get_ticket A")
                .expect("ticket A exists");
            assert_eq!(
                ticket_a.status,
                TicketPhase::Failed,
                "tripped ticket A should be Failed"
            );
        }

        // ── Verify ticket B (same workspace) is Paused ──
        {
            let ticket_b = board()
                .get_ticket(&ticket_b_id)
                .await
                .expect("get_ticket B")
                .expect("ticket B exists");
            assert_eq!(
                ticket_b.status,
                TicketPhase::Paused,
                "other ReadyForDevelopment ticket B in same workspace should be Paused"
            );
        }

        // ── Verify ticket C (different workspace) is still ReadyForDevelopment ──
        {
            let ticket_c = board()
                .get_ticket(&ticket_c_id)
                .await
                .expect("get_ticket C")
                .expect("ticket C exists");
            assert_eq!(
                ticket_c.status,
                TicketPhase::ReadyForDevelopment,
                "ticket C in different workspace must not be paused"
            );
        }

        // ── Verify ticket B has a system comment referencing ticket A ──
        {
            let comments = board()
                .get_comments(&ticket_b_id)
                .await
                .expect("get_comments for B");
            let pause_comment = comments
                .iter()
                .find(|c| c.role == "system")
                .expect("ticket B should have a system comment");

            assert!(
                pause_comment.content.contains(&ticket_a_id),
                "pause comment should contain the tripped ticket's ID"
            );
            assert!(
                pause_comment
                    .content
                    .contains("Paused due to circuit breaker trip on"),
                "pause comment should start with expected format"
            );
            assert!(
                pause_comment
                    .content
                    .contains("Unpause manually after investigation"),
                "pause comment should end with expected format"
            );
        }
    }

    /// Verify that `record_verdict_comments` correctly writes comments
    /// based on verdict filter.
    #[tokio::test]
    async fn record_verdict_comments_counts() {
        init_test_stores().await;

        let ws = test_ws_named("/tmp/test", "test");
        let ticket_id = board()
            .create_ticket("Test", "desc", &ws, DEFAULT_TICKET_PHASE, &[], "test", None)
            .await
            .expect("create_ticket");

        // ── FailingOnly with all-passing verdicts ──
        // Should produce 0 comments (nothing to write).
        let passing_verdict = crate::Verdict {
            score: REVIEW_QA_THRESHOLD, // 9/10 — passes
            critique: None,
            issues_detected: vec![],
        };
        let results = vec![ParallelVerdict {
            response: "Looks good.".into(),
            verdict: Some(passing_verdict),
        }];
        record_verdict_comments(
            &ticket_id,
            &results,
            Role::Reviewer.as_str(),
            VerdictFilter::FailingOnly,
        )
        .await;

        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert_eq!(
            comments.len(),
            0,
            "passing verdicts with FailingOnly filter should produce 0 comments"
        );

        // ── FailingOnly with a failing verdict ──
        // Should produce 1 comment.
        let failing = crate::Verdict {
            score: 3, // below threshold
            critique: Some("Missing error handling.".into()),
            issues_detected: vec!["No timeout check".into()],
        };
        let results = vec![ParallelVerdict {
            response: "Has issues.".into(),
            verdict: Some(failing),
        }];
        record_verdict_comments(
            &ticket_id,
            &results,
            Role::Reviewer.as_str(),
            VerdictFilter::FailingOnly,
        )
        .await;

        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert_eq!(
            comments.len(),
            1,
            "failing verdict should create one comment"
        );
        assert_eq!(comments[0].role, "reviewer_1");

        // ── All filter (analyst path) ──
        // Should produce 2 comments (both verdicts recorded).
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
        record_verdict_comments(
            &ticket_id,
            &results,
            Role::Analyst.as_str(),
            VerdictFilter::All,
        )
        .await;

        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert_eq!(
            comments.len(),
            3,
            "All filter should write both verdicts (total 3)"
        );
    }

    // ── transition_qa_to_done — conditional notification ─────────────

    /// Ensures the ticket buffer is initialized exactly once across all tests.
    /// Unlike `ticket_buffer::reset()`, this does not clear existing entries
    /// from other workspaces, avoiding flake when tests run concurrently.
    static TICKET_BUF_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    fn ensure_ticket_buffer_initialized() {
        TICKET_BUF_INIT.get_or_init(|| {
            // `init_global` panics if already set, but OnceLock ensures
            // this code runs exactly once.
            crate::ticket_buffer::init_global();
        });
    }

    /// Prepare a workspace for `transition_qa_to_done` tests.
    ///
    /// Ensures all global stores (board, workspace, manager_queue) are
    /// initialized (race-safe with concurrent tests), creates a unique
    /// workspace in the store, and returns a [`Workspace`] struct referencing
    /// it. Each test must pass a unique `suffix` to avoid UNIQUE constraint
    /// and cross-test pollution on the shared ticket buffer.
    async fn setup_transition_qa_to_done_test(suffix: &str) -> crate::Workspace {
        init_test_stores().await;
        ensure_ticket_buffer_initialized();
        // Workspace store — race-safe: if another test already initialized it,
        // init_global returns an error which we ignore.
        if crate::workspace::WORKSPACES.get().is_none() {
            let _ = crate::workspace::init_global().await;
        }
        // Manager queue — needed by the Notify path inside notify_ticket.
        // Race-safe: if another test already initialized it, init_global
        // returns an error which we ignore.
        if crate::manager_queue::MANAGER_QUEUE.get().is_none() {
            let _ = crate::manager_queue::init_global();
        }

        let ws_name = format!("ws_{suffix}");
        let ws_path = format!("/tmp/test_{suffix}");
        let now = crate::turso::now();
        crate::workspace::store()
            .conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at, paused) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                turso::params![ws_name.clone(), ws_path.clone(), now.clone(), now, 0],
            )
            .await
            .expect("insert test workspace");

        test_ws_named(&ws_path, &ws_name)
    }

    /// Verify that a single QaPassed ticket (the last active one) triggers an
    /// immediate Notify — no buffer entries should remain.
    #[tokio::test]
    async fn transition_qa_to_done_last_ticket_notifies() {
        let ws = setup_transition_qa_to_done_test("last_notifies").await;
        let ticket_id = board()
            .create_ticket(
                "Last Ticket",
                "desc",
                &ws,
                TicketPhase::QaPassed,
                &[],
                "test",
                None,
            )
            .await
            .expect("create_ticket");

        let ticket = board()
            .get_ticket(&ticket_id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");

        // No other active tickets — should Notify (not buffer)
        transition_qa_to_done(&ticket, "Test transition — last ticket").await;

        // Verify transitioned to Done
        let updated = board()
            .get_ticket(&ticket_id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");
        assert_eq!(
            updated.status,
            TicketPhase::Done,
            "Single ticket should transition to Done",
        );

        // Buffer should be empty (Notify path drains nothing; we had no buffer)
        let drained = crate::ticket_buffer::drain("ws_last_notifies");
        assert!(
            drained.is_empty(),
            "No buffered entries expected for last-ticket transition: got {drained:?}",
        );
    }

    /// Verify that a QaPassed ticket in a workspace with other active tickets
    /// buffers the Done notification instead of notifying immediately.
    #[tokio::test]
    async fn transition_qa_to_done_active_tickets_buffers() {
        let ws = setup_transition_qa_to_done_test("active_buffers").await;

        // Create one ticket in QaPassed (the one we'll transition)
        let qa_id = board()
            .create_ticket(
                "QA Ticket",
                "desc",
                &ws,
                TicketPhase::QaPassed,
                &[],
                "test",
                None,
            )
            .await
            .expect("create QA ticket");

        // Create another active ticket (ReadyForDevelopment — active without reservation)
        board()
            .create_ticket(
                "Active RFD",
                "desc",
                &ws,
                TicketPhase::ReadyForDevelopment,
                &[],
                "test",
                None,
            )
            .await
            .expect("create RFD ticket");

        let ticket = board()
            .get_ticket(&qa_id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");

        // Other active ticket exists — should Buffer
        transition_qa_to_done(&ticket, "Test transition — other active exists").await;

        // Verify transitioned to Done
        let updated = board()
            .get_ticket(&qa_id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");
        assert_eq!(
            updated.status,
            TicketPhase::Done,
            "Ticket should transition to Done even when buffered",
        );

        // Buffer should contain the Done transition
        let drained = crate::ticket_buffer::drain("ws_active_buffers");
        assert!(
            !drained.is_empty(),
            "Buffered entry expected when other active tickets exist",
        );
        assert!(
            drained.contains("qa_passed → done"),
            "Buffer entry should contain the status transition: {drained}",
        );
    }

    /// Verify that when multiple QaPassed tickets exist and the last one finishes,
    /// the notification includes previously buffered Done transitions (buffer drain).
    #[tokio::test]
    async fn transition_qa_to_done_last_ticket_drains_buffer() {
        let ws = setup_transition_qa_to_done_test("drains_buffer").await;

        // Two QaPassed tickets in the same workspace
        let ticket_a_id = board()
            .create_ticket(
                "Ticket A",
                "desc",
                &ws,
                TicketPhase::QaPassed,
                &[],
                "test",
                None,
            )
            .await
            .expect("create ticket A");

        let ticket_b_id = board()
            .create_ticket(
                "Ticket B",
                "desc",
                &ws,
                TicketPhase::QaPassed,
                &[],
                "test",
                None,
            )
            .await
            .expect("create ticket B");

        let ticket_a = board()
            .get_ticket(&ticket_a_id)
            .await
            .expect("get_ticket")
            .expect("ticket A exists");

        // Transition ticket A — ticket B is still QaPassed (active), so Buffer
        transition_qa_to_done(&ticket_a, "Test — ticket A done, B still active").await;

        // Transition ticket B — no more active tickets, should Notify
        let ticket_b = board()
            .get_ticket(&ticket_b_id)
            .await
            .expect("get_ticket")
            .expect("ticket B exists");
        transition_qa_to_done(&ticket_b, "Test — ticket B done, last ticket").await;

        // Verify both tickets are Done
        for (id, label) in [(&ticket_a_id, "A"), (&ticket_b_id, "B")] {
            let t = board()
                .get_ticket(id)
                .await
                .expect("get_ticket")
                .unwrap_or_else(|| panic!("ticket {label} exists"));
            assert_eq!(t.status, TicketPhase::Done, "Ticket {label} should be Done");
        }

        // The Notify path on ticket B should have drained the buffer,
        // consuming ticket A's buffered entry. No entries for this workspace
        // should remain.
        let drained = crate::ticket_buffer::drain("ws_drains_buffer");
        assert!(
            drained.is_empty(),
            "Buffer should be empty after last ticket's Notify drains it",
        );
    }
}
