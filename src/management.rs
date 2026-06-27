//! Board poller — picks up tickets from the board and dispatches agents.
//!
//! Poll phases — dispatches agents based on ticket phase:
//! - Backlog → spawn Analyst agents (`PARALLEL_AGENT_COUNT` parallel)
//! - ReadyForDevelopment → spawn Engineer agent
//! - InDiagnostics → dispatch diagnostics runner (shell commands)
//! - DiagnosticsDone → spawn Reviewer agents (`PARALLEL_AGENT_COUNT` parallel)
//! - Reviewed → spawn QA agents (`PARALLEL_AGENT_COUNT` parallel)
//! - QaPassed → check for untracked files; if found, claim to InSanitation and
//!   dispatch Sanitation agent, otherwise commit and transition to Done
//! - InSanitation → dispatch Sanitation agent (via `assigned_to` re-dispatch guard)
//! - SanitationPassed → auto-commit and transition to Done
//!
//! Reviewer and QA phases share a single `PollPhase::VerifierCheck` variant
//! with per-phase configuration carried in `VerifierInfo` constants
//! (`REVIEWER_VI`, `QA_VI`).
//!
//! The Sanitation phase (sanitation.md agent prompt, Role::Sanitation) inspects
//! new/untracked files before the auto-commit step. Garbage artifacts cause a
//! bounce back to ReadyForDevelopment; clean files proceed to Done via commit.

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
/// transitioning to Failed for Manager triage.
const CIRCUIT_BREAKER_COMMENT_THRESHOLD: usize = 50;

/// Maximum number of cumulative diagnostics failures allowed before the circuit
/// breaker trips. The breaker trips when `count > DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD`
/// (i.e., at ≥5 failures), failing the ticket to prevent thrashing.
const DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD: usize = 4;

/// Prefix for all auto-diagnostics comments on tickets.
const DIAGNOSTICS_COMMENT_PREFIX: &str = "🔍 Auto-diagnostics";
/// Marker appended when all diagnostics pass.
const DIAGNOSTICS_PASSED_MARKER: &str = "✅ All diagnostics passed";
/// Marker appended when diagnostics fail (includes the failed-at label after it).
const DIAGNOSTICS_FAILED_MARKER: &str = "❌ Diagnostics failed at";
/// Substring used both by the breaker trip comment and the self-counting guard
/// — must stay in sync between `comment_text` and `count_fn`.
const CIRCUIT_BREAKER_TRIP_MARKER: &str = "Circuit breaker";

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
/// circuit breaker (fails tickets with excessive comment accumulation
/// to prevent dispatch thrashing).
///
/// Returns `true` when it's safe for the caller to proceed. Returns `false`
/// when the ticket has moved, been failed — the caller should bail out immediately.
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
///
/// Returns `Ok(())` if the transition was applied (ticket was still in expected
/// phase). Returns `Err(String)` with a descriptive message if the ticket was
/// moved externally or an error occurred. On failure, clears `assigned_to` so
/// the poll loop can re-dispatch if needed.
///
/// The `pipeline_reservation` parameter is forwarded to
/// [`BoardStore::transition_to`]: pass `Some(true)` for bounce-back transitions
/// (back to [`TicketPhase::ReadyForDevelopment`]) to ensure the ticket gets
/// priority re-dispatch over fresh tickets, or `None` for all other transitions.
///
/// If `notify` is [`NotifyPolicy::Notify`], calls [`notify_ticket`] on success;
/// errors from notification are logged and discarded (not propagated).
async fn transition_ticket(
    ticket: &Ticket,
    expected: TicketPhase,
    target: TicketPhase,
    notify: NotifyPolicy,
    pipeline_reservation: Option<bool>,
) -> Result<(), String> {
    match board()
        .transition_to(&ticket.id, Some(expected), target, pipeline_reservation)
        .await
    {
        Ok(()) => {
            if matches!(notify, NotifyPolicy::Notify) {
                notify_ticket(ticket, target).await;
            } else if let Some(ws) =
                resolve_ticket_workspace(ticket, "cannot buffer transition").await
            {
                ticket_buffer::push(&ws.name, &ticket.id, ticket.status, target);
            }

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
            let _ = board().set_assigned_to(&ticket.id, None).await;
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

/// Prepare and dispatch a notification for a ticket transition.
///
/// Renders the notification message and enqueues a Manager job via the
/// serialized [`crate::manager_queue::MANAGER_QUEUE`]. The consumer loop handles user lookup
/// and delivery (broadcasts to all users with this workspace active).
///
/// This is a pure notification function — it does NOT pause the workspace.
/// The Manager handles failed tickets autonomously via the triage prompt.
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
        // Spawn an inner task so panics inside the dispatch are caught via
        // the JoinHandle rather than silently swallowed by the outer spawn.
        let handle = tokio::spawn(async move {
            match phase {
                PollPhase::BacklogAnalysis => dispatch_backlog_analysts(ticket, ws).await,
                PollPhase::EngineerDevelopment => dispatch_engineer(ticket, ws).await,
                PollPhase::SanitationCheck => dispatch_sanitation(ticket, ws).await,
                PollPhase::DiagnosticsCheck => dispatch_diagnostics(ticket, ws).await,
                PollPhase::VerifierCheck(vi) => dispatch_verifiers(ticket, ws, vi).await,
            }
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
                    None,
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
    SanitationCheck,
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
            Self::SanitationCheck => PollPhaseInfo {
                source: TicketPhase::QaPassed,
                claim_target: TicketPhase::InSanitation,
                // Note: claim_target is consumed by spawn_dispatch's
                // panic-recovery transition (target_phase → Failed); source is
                // metadata-only since SanitationCheck is excluded from CLAIM_PHASES
                // and the actual QaPassed→InSanitation transition happens via
                // raw SQL in handle_qa_passed.
                require_clear_pipeline: false,
                role_label: Role::Sanitation.as_str(),
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
}

/// Pipeline phases that use atomic source→claim_target claim transitions.
///
/// DiagnosticsCheck and SanitationCheck are intentionally excluded — they
/// keep the ticket in InDiagnostics/InSanitation while running and guard
/// re-dispatch via `assigned_to` and pre-condition checks respectively.
/// QaPassed→Done uses a separate list-based dispatch because the commit
/// must succeed before transitioning to Done, so there is no atomic claim
/// to perform.
///
/// [`TicketPhase::Planning`] is intentionally absent from this list.
/// Planning tickets require Manager judgment and are never picked up
/// automatically — the Manager (or user) must manually advance or cancel
/// them. This is by design, not an omission.
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
        // 1. Claim for each pipeline phase.
        //
        // When the workspace is paused, only block EngineerDevelopment
        // (ready_for_development → in_development). All other phases
        // (analysis, review, QA, …) proceed normally.
        //
        // On claim error we `break` out of the phase loop — this skips all
        // remaining CLAIM_PHASES for this workspace and falls through to
        // Diagnostics/QaPassed (which handle their own errors independently).
        // A DB-down workspace won't block other workspaces; a transient claim
        // failure won't generate log noise for every remaining phase.
        for &phase in CLAIM_PHASES {
            if ws.paused && matches!(phase, PollPhase::EngineerDevelopment) {
                continue;
            }
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
                    ticket_buffer::push(&ws.name, &t.id, info.source, t.status);
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

        // 3. SanitationPassed → Done (auto-commit).
        //
        // After the sanitation agent approves, the ticket reaches SanitationPassed.
        // We commit the changes and transition to Done, following the same pattern
        // as the QaPassed→Done commit flow.
        for_tickets_in_phase(TicketPhase::SanitationPassed, &ws.name, |ticket| {
            let ws = ws.clone();
            tokio::spawn(async move {
                finalize_sanitation_passed(ticket, ws).await;
            });
        })
        .await;

        // 4. Handle QaPassed tickets.
        //
        // For each QaPassed ticket, check whether the working tree has new/untracked
        // files. If it does, claim the ticket to InSanitation and dispatch a sanitation
        // agent. Otherwise, commit directly and transition to Done (existing behavior).
        //
        // Spawned via tokio::spawn to prevent git operations from blocking the poll loop.
        // The ticket stays in QaPassed until either the claim or the commit succeeds,
        // so re-dispatch is harmless.
        for_tickets_in_phase(TicketPhase::QaPassed, &ws.name, |ticket| {
            let ws = ws.clone();
            tokio::spawn(async move {
                handle_qa_passed(ticket, ws).await;
            });
        })
        .await;

        // 5. Dispatch unassigned InSanitation tickets.
        //
        // SanitationCheck does NOT use the claim loop — the claim (QaPassed→InSanitation)
        // happens inside handle_qa_passed, which also pushes the ticket_buffer entry.
        // We guard against re-dispatch via assigned_to IS NULL, matching the
        // DiagnosticsCheck pattern.
        for_tickets_in_phase(TicketPhase::InSanitation, &ws.name, |ticket| {
            if ticket.assigned_to.is_some() {
                return; // already dispatched, still running
            }
            spawn_dispatch(PollPhase::SanitationCheck, ticket, ws.clone());
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

    let mut message = String::new();
    if feedback.is_empty() {
        message = "Implement the ticket described in the system prompt.".to_string();
    } else {
        let _ = write!(
            message,
            "New feedback to address:\n{}",
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
    if let Err(e) = transition_ticket(
        &ticket,
        TicketPhase::InDevelopment,
        target_phase,
        notify,
        None,
    )
    .await
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

/// Determine whether to notify immediately or buffer the Done transition.
///
/// If other active tickets remain in the workspace, the notification is
/// buffered so the Manager only gets one notification when the last ticket
/// finishes. Active tickets = `PIPELINE_BLOCKING_STATUSES` + `ReadyForDevelopment`.
///
/// # Race condition
///
/// Multiple QaPassed tickets in the same workspace are finalized concurrently
/// (`tokio::spawn` in `poll_round`). Both may see each other as active and
/// both buffer. In this scenario all tickets are already Done in the database
/// — the only consequence is delayed notifications until the next
/// `UserMessage` drains the buffer.
async fn determine_notify_policy(workspace_name: &str, ticket_id: &str) -> NotifyPolicy {
    match board()
        .has_active_tickets_excluding(workspace_name, ticket_id)
        .await
    {
        Ok(true) => {
            debug!(
                ticket = %ticket_id,
                workspace = %workspace_name,
                "Other active tickets remain — buffering Done notification",
            );
            NotifyPolicy::Buffer
        }
        Ok(false) => NotifyPolicy::Notify,
        Err(e) => {
            warn!(
                ticket = %ticket_id,
                workspace = %workspace_name,
                error = %e,
                "Failed to check active tickets — notifying to be safe",
            );
            NotifyPolicy::Notify
        }
    }
}

/// Transition a ticket to Done with a descriptive reason from the given source phase.
async fn transition_ticket_to_done(ticket: &Ticket, source: TicketPhase, reason: &str) {
    info!(ticket = %ticket.id, "{reason}");

    let notify_policy = determine_notify_policy(&ticket.workspace_name, &ticket.id).await;

    if let Err(e) = transition_ticket(ticket, source, TicketPhase::Done, notify_policy, None).await
    {
        let phase_label = source.as_ref();
        warn!(
            ticket = %ticket.id,
            error = %e,
            "{phase_label} passed but transition to Done failed",
        );
    }
}

/// Auto-commit changes and move the ticket to Done.
///
/// Parameterized by source phase so both [`finalize_qa_passed`] and
/// [`finalize_sanitation_passed`] share the same implementation.
///
/// Checks for a dirty working tree via `git status --porcelain`:
/// - **Clean tree:** skips commit, transitions directly to Done with notification.
/// - **Dirty tree:** runs `git commit -m "<ticket title>"` via [`crate::diff_parse::run_git_commit`].
/// - **Commit failure:** ticket stays in `source`, poller retries next cycle.
/// - **Not a git repo / no git installed:** transitions to Done without commit.
async fn finalize_ticket_from_phase(ticket: Ticket, ws: Workspace, source: TicketPhase) {
    let repo_path = ws.as_path();
    let phase_label = source.as_ref();

    if !crate::diff_parse::git_is_installed().await {
        transition_ticket_to_done(
            &ticket,
            source,
            "Git not installed — moving to Done without commit",
        )
        .await;
        return;
    }

    if !crate::diff_parse::is_git_repo(repo_path) {
        transition_ticket_to_done(
            &ticket,
            source,
            "Not a git repo — moving to Done without commit",
        )
        .await;
        return;
    }

    // Check for working tree changes
    let has_changes = match crate::diff_parse::run_git_status(repo_path).await {
        Ok(output) => !output.trim().is_empty(),
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to check git status — staying in {phase_label} for retry"
            );
            return;
        }
    };

    if !has_changes {
        transition_ticket_to_done(
            &ticket,
            source,
            "Clean working tree — moving to Done without commit",
        )
        .await;
        return;
    }

    match crate::diff_parse::run_git_commit(repo_path, &ticket.title).await {
        Ok(commit_info) => {
            commit_and_transition_ticket_from(&ticket, commit_info, source).await;
        }
        Err(e) => {
            error!(
                ticket = %ticket.id,
                error = %e,
                "Commit failed — staying in {phase_label} for retry"
            );
        }
    }
}

/// Auto-commit changes after QA passes and move the ticket to Done.
async fn finalize_qa_passed(ticket: Ticket, ws: Workspace) {
    finalize_ticket_from_phase(ticket, ws, TicketPhase::QaPassed).await;
}

/// After a successful `git commit`, persist the metadata and transition the
/// ticket to Done atomically within a single DB transaction.
///
/// Parameterized by source phase so both the QaPassed→Done and
/// SanitationPassed→Done flows share the same implementation.
async fn commit_and_transition_ticket_from(
    ticket: &Ticket,
    commit_info: crate::diff_parse::CommitInfo,
    source: TicketPhase,
) {
    let short_hash = commit_info.hash.get(..7).unwrap_or(&commit_info.hash);
    let comment = format_commit_summary(
        short_hash,
        commit_info.lines_added,
        commit_info.lines_removed,
    );

    let phase_label = source.as_ref();

    let tx = match board().conn.begin_tx().await {
        Ok(tx) => tx,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to begin transaction — staying in {phase_label} for retry",
            );
            return;
        }
    };

    let outcome: anyhow::Result<()> = async {
        BoardStore::set_commit_info_tx(
            &tx,
            &ticket.id,
            &commit_info.hash,
            commit_info.lines_added,
            commit_info.lines_removed,
        )
        .await?;
        BoardStore::add_comment_tx(&tx, &ticket.id, "system", &comment).await?;
        BoardStore::transition_to_tx(&tx, &ticket.id, Some(source), TicketPhase::Done, None)
            .await?;
        Ok(())
    }
    .await;

    match outcome {
        Ok(()) => {
            if let Err(e) = tx.commit().await {
                error!(
                    ticket = %ticket.id,
                    error = %e,
                    "Commit succeeded ({short_hash}) but DB transaction commit failed — \
                     ticket stays in {phase_label} for retry, orphan commit in repo",
                );
                return;
            }

            // In-memory side-effects happen after the transaction commits.
            crate::registry::AGENT_REGISTRY.cancel_by_ticket_id(&ticket.id);

            info!(ticket = %ticket.id, "Committed {short_hash}, moving to Done");

            let notify_policy = determine_notify_policy(&ticket.workspace_name, &ticket.id).await;

            match notify_policy {
                NotifyPolicy::Notify => notify_ticket(ticket, TicketPhase::Done).await,
                NotifyPolicy::Buffer => {
                    if let Some(ws) =
                        resolve_ticket_workspace(ticket, "cannot buffer Done transition").await
                    {
                        ticket_buffer::push(&ws.name, &ticket.id, source, TicketPhase::Done);
                    }
                }
            }
        }
        Err(e) => {
            // tx is dropped → TxGuard::drop sets the dangling_tx flag,
            // triggering a rollback on the next write attempt.
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to finalize Done transition — transaction rolled back, \
                 ticket stays in {phase_label} for retry",
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

// ── Sanitation ─────────────────────────────────────────────────────────

/// Maximum number of consecutive sanitation failures allowed before the
/// sanitation circuit breaker trips. The breaker trips when a ticket's
/// sanitation failure count exceeds this threshold, failing the ticket.
///
/// This is a separate, lower threshold than the general comment-count
/// circuit breaker — sanitation failures are cheap to detect and should
/// not consume 50 comments before tripping.
const SANITATION_CIRCUIT_BREAKER_THRESHOLD: usize = 3;

/// Handle a QaPassed ticket: check for untracked/new files and either
/// transition to InSanitation for sanitation agent dispatch or commit
/// directly to Done.
///
/// Checks the working tree for untracked files (`git status --porcelain`
/// showing `??` or `A `). If untracked files exist, atomically transitions
/// the ticket to InSanitation with `assigned_to` set (no TOCTOU window
/// between transition and assignment), and dispatches the sanitation agent.
/// Otherwise, commits and transitions to Done (existing behavior).
async fn handle_qa_passed(ticket: Ticket, ws: Workspace) {
    let repo_path = ws.as_path();

    // Only check git if it's available and the repo exists.
    if !crate::diff_parse::git_is_installed().await || !crate::diff_parse::is_git_repo(repo_path) {
        finalize_qa_passed(ticket, ws).await;
        return;
    }

    // Check for untracked files — use list_untracked_files which returns the
    // actual file list, then check if non-empty (avoids a separate git call).
    let untracked = match list_untracked_files(repo_path).await {
        Ok(files) => files,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to check git status for untracked files — staying in QaPassed for retry"
            );
            return;
        }
    };

    if untracked.is_empty() {
        // No new/untracked files — commit directly (current behavior).
        finalize_qa_passed(ticket, ws).await;
        return;
    }

    // Untracked files exist — claim this specific ticket to InSanitation.
    //
    // We use a raw SQL UPDATE instead of `transition_to()` because we need to
    // atomically set both `status` AND `assigned_to` in a single statement.
    // `transition_to` always clears `assigned_to = NULL`, but we need to set
    // it to the session key to prevent the poll loop's InSanitation re-dispatch
    // guard (assigned_to IS NULL) from firing a second agent during the yield
    // between two separate calls.
    //
    // This is safe because `ticket` is owned (handle_qa_passed takes Ticket by
    // value) and we already checked `untracked.is_empty()` — no other handler
    // can race on this specific ticket id because the CAS on `status = 'qa_passed'`
    // ensures at most one handler wins the transition. There is no running agent
    // to cancel (the ticket is in QaPassed, a transitory handoff phase with no
    // agent mid-execution), so the execute_and_cancel wrapper that `transition_to`
    // uses is not needed here.
    //
    // WARNING: Do NOT copy this raw-SQL pattern for phases that DO have running
    // agents. The execute_and_cancel wrapper in `transition_to` cancels agents
    // for the claimed ticket before transitioning — without it, a stale agent
    // would continue running with a now-stale phase reference. QaPassed is safe
    // because it is a transitory handoff phase with no running agent.
    let session_key = ticket_session_key(&ticket.id, Role::Sanitation.as_str());
    // Check the sanitation pipeline: reject the claim if another ticket is
    // already in InSanitation or SanitationPassed in this workspace. This
    // enforces the manager's requirement that sanitation is serialized per
    // workspace — only one ticket at a time through the sanitation pipeline.
    // We check only InSanitation/SanitationPassed (not the full
    // PIPELINE_BLOCKING_STATUSES) because earlier pipeline phases like QaPassed
    // should not block a new ticket from entering sanitation.
    let rows = board()
        .conn
        .execute(
            "UPDATE tickets SET status = ?1, assigned_to = ?2, updated_at = ?3 \
             WHERE id = ?4 AND status = ?5 \
             AND NOT EXISTS (SELECT 1 FROM tickets t2 \
               WHERE t2.workspace_name = (SELECT workspace_name FROM tickets WHERE id = ?4) \
               AND t2.id != ?4 \
               AND t2.status IN ('in_sanitation','sanitation_passed'))",
            turso::params![
                TicketPhase::InSanitation.as_ref(),
                session_key.as_str(),
                crate::turso::now(),
                ticket.id.as_str(),
                TicketPhase::QaPassed.as_ref(),
            ],
        )
        .await;

    let rows = match rows {
        Ok(r) => r,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to transition QaPassed ticket to InSanitation"
            );
            return;
        }
    };

    if rows == 0 {
        debug!(
            ticket = %ticket.id,
            "QaPassed ticket moved externally — skipping sanitation dispatch",
        );
        return;
    }

    ticket_buffer::push(
        &ticket.workspace_name,
        &ticket.id,
        TicketPhase::QaPassed,
        TicketPhase::InSanitation,
    );

    // Re-fetch ticket with updated status/comments before dispatch.
    // By this point assigned_to is already set atomically in the same
    // UPDATE that changed the status, so the poll loop's InSanitation
    // re-dispatch guard (assigned_to IS NULL) will not fire for this ticket.
    let updated = match board().get_ticket(&ticket.id).await {
        Ok(Some(t)) => t,
        Ok(None) => {
            warn!(ticket = %ticket.id, "Ticket disappeared after transition — skipping");
            // Ticket was deleted between CAS and re-fetch. Clear assigned_to so
            // the poll loop doesn't think a running agent exists for a phantom ticket.
            let _ = board().set_assigned_to(&ticket.id, None).await;
            return;
        }
        Err(e) => {
            warn!(ticket = %ticket.id, error = %e, "Failed to re-fetch ticket after transition");
            // Re-fetch failed (DB error). The ticket is stuck in InSanitation with
            // assigned_to set but no agent dispatched. Clear assigned_to to allow the
            // poll loop to retry dispatching a sanitation agent on the next cycle.
            let _ = board().set_assigned_to(&ticket.id, None).await;
            return;
        }
    };

    spawn_dispatch(PollPhase::SanitationCheck, updated, ws);
}

/// Sanitation circuit breaker — trip if the ticket has accumulated too
/// many consecutive sanitation failures.
///
/// Delegates to [`run_circuit_breaker`] to ensure fresh comment fetching,
/// proper sibling ticket drainage (moving other ReadyForDevelopment tickets
/// to Planning), and consistent failure handling with the general breaker.
///
/// Counts system comments where role == "system" and content contains
/// "Sanitation failed" but NOT "Circuit breaker" (to prevent self-counting
/// cascade on re-dispatch after the breaker trips).
///
/// Separate from the general comment-count circuit breaker so that
/// garbage-thrashing tickets are caught early without consuming the full
/// 50-comment budget.
#[must_use]
async fn trip_sanitation_circuit_breaker_if_exceeded(
    ticket: &Ticket,
    expected: TicketPhase,
) -> bool {
    run_circuit_breaker(
        ticket,
        expected,
        SANITATION_CIRCUIT_BREAKER_THRESHOLD,
        |comments| {
            comments
                .iter()
                .filter(|c| {
                    c.role == "system"
                        && c.content.contains("Sanitation failed")
                        && !c.content.contains("Circuit breaker")
                })
                .count()
        },
        |count| {
            format!(
                "❌ Sanitation circuit breaker tripped after {count} consecutive failures. \
                 (threshold: {SANITATION_CIRCUIT_BREAKER_THRESHOLD})",
            )
        },
        "Sanitation",
    )
    .await
}

/// Run the sanitation agent to inspect new/untracked files in the workspace.
///
/// Called by [`PollPhase::SanitationCheck`] via [`spawn_dispatch`]. Runs a
/// single sanitation agent with tools to inspect files and determine whether
/// they are legitimate project files or intermediate garbage.
///
/// After the agent completes, extracts a structured [`SanitationVerdict`]:
/// - If **clean** (pass = true): transitions to [`TicketPhase::SanitationPassed`]
///   (transitory handoff before auto-commit).
/// - If **garbage detected** (pass = false): adds a comment listing the offending
///   files and transitions the ticket to [`TicketPhase::ReadyForDevelopment`] with a
///   pipeline reservation (via [`transition_ticket`]), matching the existing review/QA
///   failure pattern.
#[allow(clippy::too_many_lines)]
async fn dispatch_sanitation(ticket: Arc<Ticket>, ws: Workspace) {
    let session_key = ticket_session_key(&ticket.id, Role::Sanitation.as_str());

    // Phase check only (no general circuit breaker): verify the ticket is still
    // in InSanitation before starting the agent. The dedicated sanitation circuit
    // breaker below (threshold: 3) will always trip before the general comment-count
    // breaker, so running both would waste a DB round-trip fetching comments twice.
    if !is_ticket_in_phase(&ticket.id, TicketPhase::InSanitation).await {
        return;
    }

    // Check the sanitation-specific circuit breaker before running the agent.
    if trip_sanitation_circuit_breaker_if_exceeded(&ticket, TicketPhase::InSanitation).await {
        return;
    }

    //
    // Unlike handle_qa_passed (which fails closed on git errors — returning early
    // to stay in QaPassed for retry), dispatch_sanitation takes a fail-open approach:
    // if we can't list untracked files, we pass an empty list rather than failing the
    // ticket. The sanitation agent will see "(could not list untracked files)" and
    // proceed. This is intentional: by the time dispatch_sanitation runs, the ticket
    // has already been claimed to InSanitation with assigned_to set. Failing-closed
    // (returning early) would leave the ticket stuck in InSanitation with no agent
    // running, requiring the next poll cycle's re-dispatch guard to recover. Passing
    // an empty list is at-worst a no-op (the agent passes, ticket proceeds to commit);
    // at-best the agent may still detect garbage from known patterns.
    //
    // Note: this re-runs `git status --porcelain` even though `handle_qa_passed`
    // already collected the untracked file list. The re-run is unavoidable because
    // `dispatch_sanitation` runs in a separate async task (spawned via `spawn_dispatch`)
    // and the data from `handle_qa_passed` cannot be shared across that boundary.
    // The shell overhead of one `git status` call per sanitation cycle is negligible
    // relative to the LLM agent cost that follows.
    let untracked_files = match list_untracked_files(ws.as_path()).await {
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

    let (agent, response) =
        run_agent(session_key, Role::Sanitation, &ws, Some(&ticket), &prompt).await;

    // Post-run phase check — bail if ticket was moved externally.
    if !is_ticket_in_phase(&ticket.id, TicketPhase::InSanitation).await {
        return;
    }

    let Some(ref _text) = response else {
        // Agent failed or was cancelled — record failure and clear assigned_to
        // for re-dispatch retry. The system comment lets the sanitation circuit
        // breaker detect repeated failures.
        warn!(
            ticket = %ticket.id,
            "Sanitation agent returned no output — clearing assigned_to for retry"
        );
        let _ = board()
            .add_comment(
                &ticket.id,
                "system",
                "Sanitation failed — agent returned no output",
            )
            .await;
        let _ = board().set_assigned_to(&ticket.id, None).await;
        return;
    };

    let extraction_prompt = crate::prompt::load_prompt("extraction/sanitation.md");
    let retry_prompt = crate::prompt::load_prompt("extraction/retry.md");

    let verdict: crate::SanitationVerdict = match agent
        .extract_structured(&extraction_prompt, &retry_prompt, 5)
        .await
    {
        Ok(v) => v,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to extract sanitation verdict — clearing assigned_to for retry"
            );
            // Record the failure as a system comment so the circuit breaker
            // can detect repeated extraction failures.
            let _ = board()
                .add_comment(
                    &ticket.id,
                    "system",
                    &format!("Sanitation failed — verdict extraction error: {e}"),
                )
                .await;
            let _ = board().set_assigned_to(&ticket.id, None).await;
            return;
        }
    };

    if verdict.pass {
        // Clean — transition to SanitationPassed for auto-commit.
        info!(
            ticket = %ticket.id,
            "Sanitation passed — transitioning to SanitationPassed",
        );

        // Add a comment summarizing the sanitation check.
        let comment = if verdict.garbage_files.is_empty() {
            format!(
                "🧹 Sanitation passed: {rationale}",
                rationale = verdict.rationale
            )
        } else {
            format!(
                "🧹 Sanitation passed (files reviewed): {rationale}",
                rationale = verdict.rationale
            )
        };
        let _ = board()
            .add_comment(&ticket.id, Role::Sanitation.as_str(), &comment)
            .await;

        if let Err(e) = transition_ticket(
            &ticket,
            TicketPhase::InSanitation,
            TicketPhase::SanitationPassed,
            NotifyPolicy::Buffer,
            None,
        )
        .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Sanitation passed but transition to SanitationPassed failed — \
                 ticket stuck in InSanitation"
            );
        }
    } else {
        // Garbage detected — bounce back to development with details.
        let garbage_list = verdict.garbage_files.join("\n- ");
        let comment = format!(
            "🗑️ Sanitation failed — garbage files detected:\n- {garbage_list}\n\nRationale: {rationale}",
            rationale = verdict.rationale,
        );
        let _ = board()
            .add_comment(&ticket.id, Role::Sanitation.as_str(), &comment)
            .await;

        // Also add a system comment so the sanitation circuit breaker can detect it.
        let _ = board()
            .add_comment(
                &ticket.id,
                "system",
                &format!(
                    "Sanitation failed — garbage files: {count}",
                    count = verdict.garbage_files.len(),
                ),
            )
            .await;

        if let Err(e) = transition_ticket(
            &ticket,
            TicketPhase::InSanitation,
            TicketPhase::ReadyForDevelopment,
            NotifyPolicy::Buffer,
            Some(true),
        )
        .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Sanitation failed but transition to ReadyForDevelopment also failed",
            );
        } else {
            info!(
                ticket = %ticket.id,
                "Sanitation failed — pipeline reservation set for rework priority",
            );
        }
    }
}

/// List untracked/new files in the working tree by running `git status --porcelain`.
///
/// Delegates to [`parse_untracked_from_porcelain`] for the actual parsing logic.
/// Catches both `??` (untracked) and any entry starting with `A` (staged as new,
/// including `A ` clean staged and `AM` staged+modified).
async fn list_untracked_files(repo_path: &std::path::Path) -> anyhow::Result<Vec<String>> {
    use tokio::process::Command;

    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repo_path)
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git status --porcelain failed: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_untracked_from_porcelain(&stdout))
}

/// Parse untracked/new file paths from `git status --porcelain` output.
///
/// Returns file paths for entries where the index status indicates a new file:
/// - `??` — untracked file
/// - `A ` at position 0 — staged as new (first char is `A`)
///
/// This correctly catches `A ` (staged, clean) and `AM` (staged as new, then
/// modified in working tree) because both start with `A`. The porcelain format
/// is `<XY><space><path>` where X = index status, Y = working tree status.
/// Path always starts at index 3.
///
/// Excludes ` A` (not tracked, added only to working tree — this is a file
/// that exists but is not tracked by git; it falls under `??` instead).
#[must_use]
fn parse_untracked_from_porcelain(porcelain: &str) -> Vec<String> {
    porcelain
        .lines()
        .filter(|line| line.starts_with("?? ") || line.starts_with('A'))
        // Safety: porcelain lines are at minimum 4 chars (<XY><space><path>), but
        // we guard with `get()` to prevent panics on malformed input.
        .filter_map(|line| {
            let path = line.get(3..)?;
            if path.is_empty() {
                None
            } else {
                Some(path.to_string())
            }
        })
        .collect()
}

/// Finalize a SanitationPassed ticket: commit changes and transition to Done.
///
/// Delegates to the parameterized [`finalize_ticket_from_phase`].
async fn finalize_sanitation_passed(ticket: Ticket, ws: Workspace) {
    finalize_ticket_from_phase(ticket, ws, TicketPhase::SanitationPassed).await;
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
            None,
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

    let mut comment = String::from(DIAGNOSTICS_COMMENT_PREFIX);
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
        comment.push_str("\n\n---\n");
        comment.push_str(DIAGNOSTICS_PASSED_MARKER);
        TicketPhase::DiagnosticsDone
    } else {
        let _ = write!(comment, "\n\n---\n{DIAGNOSTICS_FAILED_MARKER} {failed_at}");
        TicketPhase::ReadyForDevelopment
    };
    let _ = board()
        .add_comment(&ticket.id, "diagnostics", &comment)
        .await;

    if target == TicketPhase::ReadyForDevelopment {
        if let Err(e) = transition_ticket(
            &ticket,
            TicketPhase::InDiagnostics,
            TicketPhase::ReadyForDevelopment,
            NotifyPolicy::Buffer,
            Some(true),
        )
        .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Diagnostics failed but transition to ReadyForDevelopment also failed",
            );
        } else {
            info!(
                ticket = %ticket.id,
                "Diagnostics failed — pipeline reservation set for rework priority",
            );
        }
    } else if let Err(e) = transition_ticket(
        &ticket,
        TicketPhase::InDiagnostics,
        target,
        NotifyPolicy::Buffer,
        None,
    )
    .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Diagnostics completed but transition to DiagnosticsDone \
             failed — ticket stuck in DiagnosticsDone",
        );
    }
}

/// Diagnostics-specific circuit breaker: counts prior diagnostics system
/// comments that indicate failures, and fails the ticket if the count
/// exceeds [`DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD`] (i.e., trip at ≥5 failures).
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
                        && c.content.starts_with(DIAGNOSTICS_COMMENT_PREFIX)
                        && c.content.contains(DIAGNOSTICS_FAILED_MARKER)
                        && !c.content.contains(CIRCUIT_BREAKER_TRIP_MARKER)
                })
                .count()
        },
        |count| {
            format!(
                "{DIAGNOSTICS_COMMENT_PREFIX}\n\n❌ {CIRCUIT_BREAKER_TRIP_MARKER}: {count} prior diagnostic \
                 failures. Failing ticket."
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
/// - Planning (notify) when any analyst fails, with a comment listing the counts
///
/// Before spawning agents, [`guard_phase_and_circuit_breaker`] checks the phase and
/// trips the comment-count circuit breaker (which may transition the ticket to
/// Failed for Manager triage). Returns `false` when the caller should abort.
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
/// - to Planning (notify) if any analyst failed, with a comment listing the counts
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

    let target = TicketPhase::Planning;

    if let Err(e) = transition_ticket(
        ticket,
        TicketPhase::Analysis,
        target,
        NotifyPolicy::Notify,
        None,
    )
    .await
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
            "Backlog analysis incomplete — moved to planning ({nonempty_count}/{PARALLEL_AGENT_COUNT} responded, \
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
/// The Manager is notified via [`transition_ticket`] when the ticket
/// transitions to [`TicketPhase::Failed`].
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
        "Circuit breaker tripped at {count}/{threshold} ({log_label}) — failing ticket"
    );

    let _ = board()
        .add_comment(&ticket.id, "system", &comment_text(count))
        .await;

    if let Err(e) = transition_ticket(
        ticket,
        expected,
        TicketPhase::Failed,
        NotifyPolicy::Notify,
        None,
    )
    .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Circuit breaker tripped but transition to Failed failed",
        );
        return true;
    }

    // Move all other ReadyForDevelopment tickets in the same workspace to
    // Planning to prevent them from auto-starting while the Manager triages
    // the failure.
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
                "Failed to list ReadyForDevelopment tickets for moving to planning \
                 — breaker trip proceeds without moving siblings",
            );
            return true;
        }
    };

    let planning_move_comment = format!(
        "Moved to planning due to circuit breaker trip on {}: {}. Re-advance to ReadyForDevelopment after Manager resolves the failure.",
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
            TicketPhase::Planning,
            NotifyPolicy::Buffer,
            None,
        )
        .await
        {
            debug!(
                other_ticket = %other.id,
                error = %e,
                "Failed to move other ReadyForDevelopment ticket to planning — likely raced by external move",
            );
            continue;
        }

        let _ = board()
            .add_comment(&other.id, "system", &planning_move_comment)
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
/// (via [`TicketPhase::Failed`]).
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
                 Ticket failed — Manager will triage."
            )
        },
        log_label,
    )
    .await
}

/// Process parallel verifier results: add failing comments, determine pass/fail,
/// and update ticket status accordingly.
///
/// Handles three outcomes in priority order:
///
/// 1. **All agents failed to produce a verdict** (every result has `verdict: None`
///    — crashed, timed out, or unparseable output) → transition to [`TicketPhase::Failed`]
///    with [`NotifyPolicy::Notify`]. This is a terminal failure; retrying would waste
///    credits on a fundamentally broken dispatch.
///
/// 2. **Any verifier failed** (score below [`REVIEW_QA_THRESHOLD`]) → transition back to
///    [`TicketPhase::ReadyForDevelopment`] with a pipeline reservation (via
///    [`transition_ticket`]). The circuit
///    breaker is checked *before* dispatch by [`guard_phase_and_circuit_breaker`], so only
///    the bounce-back is needed here.
///
/// 3. **All passed** (all at or above threshold) → transition to the verifier's
///    `success_phase` with [`NotifyPolicy::Buffer`]. No immediate notification fires —
///    it waits until the ticket reaches Done (after the QaPassed commit succeeds in
///    [`finalize_qa_passed`]).
async fn process_verdict_results(
    ticket: &Ticket,
    results: &[ParallelVerdict],
    verifier: VerifierInfo,
) {
    // Record per-agent comments for ALL outcomes (including the all-failed case).
    // Previously the all-failed case skipped this and only wrote a summary comment;
    // the per-agent comments provide more diagnostic information.
    record_verdict_comments(
        &ticket.id,
        results,
        verifier.role.as_str(),
        VerdictFilter::FailingOnly,
    )
    .await;

    // Priority 1: all agents failed to produce verdicts — terminal failure.
    let all_failed = results.iter().all(|r| r.verdict.is_none());
    if all_failed {
        let _ = board()
            .add_comment(
                &ticket.id,
                "system",
                &format!(
                    "❌ All {label} agents failed to produce verdicts — \
                     ticket marked as Failed.",
                    label = verifier.log_label,
                ),
            )
            .await;
        if let Err(e) = transition_ticket(
            ticket,
            verifier.active_phase,
            TicketPhase::Failed,
            NotifyPolicy::Notify,
            None,
        )
        .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                role = %verifier.log_label,
                "All {label} agents failed but transition to Failed also failed",
                label = verifier.log_label,
            );
        }
        return;
    }

    // Priority 2: any verifier failed — bounce back to development.
    let any_failed = results.iter().any(|r| !verdict_passes(r.verdict.as_ref()));
    if any_failed {
        if let Err(e) = transition_ticket(
            ticket,
            verifier.active_phase,
            TicketPhase::ReadyForDevelopment,
            NotifyPolicy::Buffer,
            Some(true),
        )
        .await
        {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "{log_label} failed but transition to ReadyForDevelopment also failed",
                log_label = verifier.log_label,
            );
        } else {
            info!(
                ticket = %ticket.id,
                "{log_label} failed — pipeline reservation set for rework priority",
                log_label = verifier.log_label,
            );
        }
        return;
    }

    // Priority 3: all passed — transition to the success phase (buffered).
    if let Err(e) = transition_ticket(
        ticket,
        verifier.active_phase,
        verifier.success_phase,
        NotifyPolicy::Buffer,
        None,
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

    process_verdict_results(&ticket, &results, vi).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::DEFAULT_TICKET_PHASE;
    use crate::board::TicketBuilder;
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
        let ticket_id = TicketBuilder::new(board(), ws)
            .title("Test")
            .create()
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
    /// ReadyForDevelopment tickets in the same workspace are moved to Planning
    /// with a system comment referencing the tripped ticket. Tickets in other
    /// workspaces must not be affected.
    #[tokio::test]
    async fn circuit_breaker_moves_other_ready_for_development_tickets_to_planning() {
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
        let trip_id = TicketBuilder::new(board(), ws_a.clone())
            .title("Trip Ticket")
            .phase(TicketPhase::ReadyForDevelopment)
            .create()
            .await
            .expect("create_ticket A");

        // Create ticket B in workspace A — this should be moved to Planning when A trips.
        let victim_id = TicketBuilder::new(board(), ws_a)
            .title("Victim Ticket")
            .phase(TicketPhase::ReadyForDevelopment)
            .create()
            .await
            .expect("create_ticket B");

        // Create ticket C in workspace B — this must NOT be moved.
        let other_ws_id = TicketBuilder::new(board(), ws_b)
            .title("Other Workspace Ticket")
            .phase(TicketPhase::ReadyForDevelopment)
            .create()
            .await
            .expect("create_ticket C");

        // Add a comment to ticket A so the circuit breaker has something to count.
        board()
            .add_comment(&trip_id, "system", "Some comment")
            .await
            .expect("add_comment to A");

        // Fetch ticket A and trip the circuit breaker with threshold 0.
        let ticket_a = board()
            .get_ticket(&trip_id)
            .await
            .expect("get_ticket A")
            .expect("ticket A exists");

        let tripped = run_circuit_breaker(
            &ticket_a,
            TicketPhase::ReadyForDevelopment,
            0,                      // threshold = 0, so 1 comment > 0 trips
            <[TicketComment]>::len, // count all comments
            |count| format!("Breaker tripped at {count}"),
            "test",
        )
        .await;

        assert!(tripped, "circuit breaker should have tripped");

        // ── Verify ticket A is Failed ──
        {
            let ticket_a = board()
                .get_ticket(&trip_id)
                .await
                .expect("get_ticket A")
                .expect("ticket A exists");
            assert_eq!(
                ticket_a.status,
                TicketPhase::Failed,
                "tripped ticket A should be Failed"
            );
        }

        // ── Verify ticket B (same workspace) is Planning ──
        {
            let ticket_b = board()
                .get_ticket(&victim_id)
                .await
                .expect("get_ticket B")
                .expect("ticket B exists");
            assert_eq!(
                ticket_b.status,
                TicketPhase::Planning,
                "other ReadyForDevelopment ticket B in same workspace should be Planning"
            );
        }

        // ── Verify ticket C (different workspace) is still ReadyForDevelopment ──
        {
            let ticket_c = board()
                .get_ticket(&other_ws_id)
                .await
                .expect("get_ticket C")
                .expect("ticket C exists");
            assert_eq!(
                ticket_c.status,
                TicketPhase::ReadyForDevelopment,
                "ticket C in different workspace must not be moved"
            );
        }

        // ── Verify ticket B has a system comment referencing ticket A ──
        {
            let comments = board()
                .get_comments(&victim_id)
                .await
                .expect("get_comments for B");
            let comment = comments
                .iter()
                .find(|c| c.role == "system")
                .expect("ticket B should have a system comment");

            assert!(
                comment.content.contains(&trip_id),
                "comment should contain the tripped ticket's ID"
            );
            assert!(
                comment
                    .content
                    .contains("Moved to planning due to circuit breaker trip on"),
                "comment should start with expected format"
            );
            assert!(
                comment.content.contains(
                    "Re-advance to ReadyForDevelopment after Manager resolves the failure"
                ),
                "comment should end with expected format"
            );
        }
    }

    /// Verify that `record_verdict_comments` correctly writes comments
    /// based on verdict filter.
    #[tokio::test]
    async fn record_verdict_comments_counts() {
        init_test_stores().await;

        let ws = test_ws_named("/tmp/test", "test");
        let ticket_id = TicketBuilder::new(board(), ws)
            .title("Test")
            .create()
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

    // ── transition_ticket_to_done — conditional notification ─────────

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

    /// Prepare a workspace for `transition_ticket_to_done` tests (QaPassed→Done path).
    ///
    /// Ensures all global stores (board, workspace, manager_queue) are
    /// initialized (race-safe with concurrent tests), creates a unique
    /// workspace in the store, and returns a [`Workspace`] struct referencing
    /// it. Each test must pass a unique `suffix` to avoid UNIQUE constraint
    /// and cross-test pollution on the shared ticket buffer.
    async fn setup_ticket_to_done_test(suffix: &str) -> crate::Workspace {
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

    /// Verify the Buffer → Notify + drain sequence across two QaPassed tickets
    /// via `transition_ticket_to_done`: the first one buffers, the last one
    /// notifies and drains the buffer.
    #[tokio::test]
    async fn transition_ticket_to_done_buffer_and_notify() {
        let ws = setup_ticket_to_done_test("drains_buffer").await;

        // Two QaPassed tickets in the same workspace
        let first_id = TicketBuilder::new(board(), ws.clone())
            .title("Ticket A")
            .phase(TicketPhase::QaPassed)
            .create()
            .await
            .expect("create ticket A");

        let second_id = TicketBuilder::new(board(), ws)
            .title("Ticket B")
            .phase(TicketPhase::QaPassed)
            .create()
            .await
            .expect("create ticket B");

        let ticket_a = board()
            .get_ticket(&first_id)
            .await
            .expect("get_ticket")
            .expect("ticket A exists");

        // Transition ticket A — ticket B is still QaPassed (active), so Buffer
        transition_ticket_to_done(
            &ticket_a,
            TicketPhase::QaPassed,
            "Test — ticket A done, B still active",
        )
        .await;

        // Intermediate assertion: verify the Buffer path was actually taken.
        // Without this, a bug where has_active_tickets_excluding incorrectly
        // returns false (causing Notify instead of Buffer) would only be caught
        // by the final empty-buffer check — which could still pass if the Notify
        // path also happened to drain the buffer cleanly (e.g., by sending an
        // empty notification). Draining here verifies entry was pushed.
        let intermediate = crate::ticket_buffer::drain("ws_drains_buffer");
        assert!(
            !intermediate.is_empty(),
            "After first QaPassed → Done with other active tickets: \
             should have buffered the notification (got empty buffer)",
        );

        // Transition ticket B — no more active tickets, should Notify and drain
        let ticket_b = board()
            .get_ticket(&second_id)
            .await
            .expect("get_ticket")
            .expect("ticket B exists");
        transition_ticket_to_done(
            &ticket_b,
            TicketPhase::QaPassed,
            "Test — ticket B done, last ticket",
        )
        .await;

        // Verify both tickets are Done
        for (id, label) in [(&first_id, "A"), (&second_id, "B")] {
            let t = board()
                .get_ticket(id)
                .await
                .expect("get_ticket")
                .unwrap_or_else(|| panic!("ticket {label} exists"));
            assert_eq!(t.status, TicketPhase::Done, "Ticket {label} should be Done");
        }

        // No entries should remain for this workspace (the Notify path on
        // ticket B calls drain() internally; we drained the intermediate
        // buffer above, so this check is for leftover / stale entries).
        let drained = crate::ticket_buffer::drain("ws_drains_buffer");
        assert!(
            drained.is_empty(),
            "Buffer should be empty after last ticket's Notify drains it",
        );
    }

    /// Verify that transitioning a ticket to [`TicketPhase::Failed`] does NOT
    /// pause the workspace. The old auto-pause behavior was removed — workspace
    /// pausing is no longer part of the Failed transition.
    #[tokio::test]
    async fn transition_to_failed_does_not_pause_workspace() {
        init_test_stores().await;
        if crate::workspace::WORKSPACES.get().is_none() {
            let _ = crate::workspace::init_global().await;
        }

        let ws_name = "ws_no_pause_on_fail_test";
        let ws_path = "/tmp/test_ws_no_pause_on_fail";
        let now = crate::turso::now();
        crate::workspace::store()
            .conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at, paused) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                turso::params![ws_name, ws_path, now.clone(), now, 0],
            )
            .await
            .expect("insert test workspace");

        let ws = test_ws_named(ws_path, ws_name);

        let ticket_id = TicketBuilder::new(board(), ws)
            .title("Test Ticket")
            .create()
            .await
            .expect("create_ticket");

        let ticket = board()
            .get_ticket(&ticket_id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");

        // Transition to Failed with Buffer policy — workspace should NOT pause
        transition_ticket(
            &ticket,
            DEFAULT_TICKET_PHASE,
            TicketPhase::Failed,
            NotifyPolicy::Buffer,
            None,
        )
        .await
        .expect("transition to Failed");

        // Verify workspace is NOT paused
        let ws = crate::workspace::get_by_name(ws_name)
            .await
            .expect("get_by_name")
            .expect("workspace exists");
        assert!(
            !ws.paused,
            "Workspace should NOT be paused after ticket failure (auto-pause was removed)"
        );
    }

    /// Verify that transitioning a ticket to a non-failure phase does NOT pause
    /// the workspace. Transitions should never pause the workspace.
    #[tokio::test]
    async fn transition_to_non_failed_does_not_pause() {
        init_test_stores().await;
        if crate::workspace::WORKSPACES.get().is_none() {
            let _ = crate::workspace::init_global().await;
        }
        if crate::manager_queue::MANAGER_QUEUE.get().is_none() {
            let _ = crate::manager_queue::init_global();
        }

        let ws_name = "ws_no_pause_test";
        let ws_path = "/tmp/test_ws_no_pause";
        let now = crate::turso::now();
        crate::workspace::store()
            .conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at, paused) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                turso::params![ws_name, ws_path, now.clone(), now, 0],
            )
            .await
            .expect("insert test workspace");

        let ws = test_ws_named(ws_path, ws_name);

        let ticket_id = TicketBuilder::new(board(), ws)
            .title("Test Ticket")
            .create()
            .await
            .expect("create_ticket");

        let ticket = board()
            .get_ticket(&ticket_id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");

        // Transition from Backlog to Analysis (non-failure) with Notify policy
        transition_ticket(
            &ticket,
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            NotifyPolicy::Notify,
            None,
        )
        .await
        .expect("transition to Analysis");

        // Verify workspace is NOT paused
        let ws = crate::workspace::get_by_name(ws_name)
            .await
            .expect("get_by_name")
            .expect("workspace exists");
        assert!(
            !ws.paused,
            "Workspace should NOT be paused after non-failure transition"
        );
    }

    // ── parse_untracked_from_porcelain — git porcelain parsing ──

    /// Verify that `parse_untracked_from_porcelain` correctly extracts untracked
    /// and staged-as-new file paths from git porcelain output.
    #[test]
    fn parse_untracked_from_porcelain_extracts_new_files() {
        let porcelain = "\
?? new_file.rs
M  modified.rs
?? another_new.py
A  staged_new.js
?? dir/untracked.txt
 M working_tree_only.txt
?? temp.log
AM staged_then_modified.js
 A working_tree_new.txt
";

        let files = parse_untracked_from_porcelain(porcelain);

        assert_eq!(files.len(), 6);
        assert!(files.contains(&"new_file.rs".to_string()));
        assert!(files.contains(&"another_new.py".to_string()));
        assert!(files.contains(&"staged_new.js".to_string()));
        assert!(files.contains(&"dir/untracked.txt".to_string()));
        assert!(files.contains(&"temp.log".to_string()));
        assert!(files.contains(&"staged_then_modified.js".to_string()));
        // These should be excluded:
        assert!(!files.contains(&"modified.rs".to_string()));
        assert!(!files.contains(&"working_tree_only.txt".to_string()));
        assert!(!files.contains(&"working_tree_new.txt".to_string()));
    }

    /// Verify that empty or no-new-files porcelain produces an empty list.
    #[test]
    fn parse_untracked_from_porcelain_empty_when_no_new_files() {
        let porcelain = "\
M  modified.rs
 M working_tree_only.txt
D  deleted.rs
 A working_tree_new.txt
";

        let files = parse_untracked_from_porcelain(porcelain);
        assert!(
            files.is_empty(),
            "Should be empty when no new/untracked files"
        );
    }

    /// Verify that the porcelain slicing at &line[3..] is correct for various
    /// git status --porcelain line formats involving new files.
    ///
    /// Tests via the actual `parse_untracked_from_porcelain` function rather than
    /// reimplementing the slicing logic inline, so a refactor of the parsing
    /// algorithm would break this test.
    #[test]
    fn parse_untracked_from_porcelain_slicing() {
        // fmt: <XY><space><path>
        let test_cases = [
            ("?? foo.rs", "foo.rs"),
            ("?? dir/bar.py", "dir/bar.py"),
            ("A  staged.txt", "staged.txt"),
            ("AM both.txt", "both.txt"), // A in index, M in working tree
        ];

        for &(line, expected_path) in &test_cases {
            // Build a multi-line porcelain string containing just this line.
            let input = format!("{line}\n");
            let files = parse_untracked_from_porcelain(&input);
            assert_eq!(
                files,
                vec![expected_path.to_string()],
                "Failed for line: {line:?}"
            );
        }
    }

    /// Verify that malformed/truncated porcelain lines don't cause panics.
    #[test]
    fn parse_untracked_from_porcelain_handles_malformed_input() {
        // Lines shorter than 4 chars that match the filter should not panic.
        let short_lines = ["A", "A ", "?? ", "??"];

        for &bad_line in &short_lines {
            let files = parse_untracked_from_porcelain(bad_line);
            // The line matches the filter but is too short for slicing.
            // get(3..) returns None, so the line is silently skipped.
            assert!(
                files.is_empty(),
                "Malformed line {bad_line:?} should produce empty result, got {files:?}"
            );
        }
    }

    // ── trip_sanitation_circuit_breaker_if_exceeded — counting ──

    /// Verify that the sanitation circuit breaker counting logic works correctly.
    #[tokio::test]
    async fn sanitation_breaker_counts_failures() {
        init_test_stores().await;
        let ws = test_ws_named("/tmp/test", "san_breaker_test");
        let ticket_id = TicketBuilder::new(board(), ws)
            .title("Sanitation Breaker Test")
            .phase(TicketPhase::InSanitation)
            .create()
            .await
            .expect("create ticket");

        // Add 2 sanitation failure comments (below threshold of 3).
        for _ in 0..2 {
            let _ = board()
                .add_comment(&ticket_id, "system", "Sanitation failed — garbage files: 1")
                .await;
        }

        let ticket = board()
            .get_ticket(&ticket_id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");

        // Should NOT trip (2 <= 3)
        assert!(
            !trip_sanitation_circuit_breaker_if_exceeded(&ticket, TicketPhase::InSanitation).await,
            "Should NOT trip with 2 failures (threshold: 3)"
        );

        // Add a 3rd failure comment (should still not trip — runs the count_fn
        // which counts by the "Sanitation failed" substring, using fresh comments
        // from DB, not the stale in-memory ticket.comments).
        let _ = board()
            .add_comment(&ticket_id, "system", "Sanitation failed — garbage files: 1")
            .await;

        // ... actually 3 <= 3 means the breaker does NOT trip yet.
        // The breaker trips when count > threshold, i.e., at 4 failures.
        // Add a 4th failure.
        let _ = board()
            .add_comment(&ticket_id, "system", "Sanitation failed — garbage files: 1")
            .await;

        // Now with 4 failures, should trip (4 > 3).
        // Re-fetch ticket with fresh comments (run_circuit_breaker fetches comments
        // from DB internally, so we just need the ticket id).
        let ticket = board()
            .get_ticket(&ticket_id)
            .await
            .expect("get_ticket")
            .expect("ticket exists");

        let tripped =
            trip_sanitation_circuit_breaker_if_exceeded(&ticket, TicketPhase::InSanitation).await;
        assert!(tripped, "Should trip with 4 failures (threshold: 3, 4 > 3)");

        // Verify the ticket is now Failed
        let status = board()
            .get_ticket_status(&ticket_id)
            .await
            .expect("get_ticket_status")
            .expect("ticket exists");
        assert_eq!(
            status,
            TicketPhase::Failed,
            "Circuit breaker should transition to Failed"
        );
    }
}
