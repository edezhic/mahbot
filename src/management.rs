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
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use futures_util::FutureExt;
use futures_util::future::join_all;

use crate::agent::run_agent;
use crate::board::{BOARD, BoardStore, PipelineCheck, Ticket, TicketComment, TicketPhase};
use crate::git_commands::{
    list_new_or_untracked_files, parse_new_files_from_porcelain, run_git_status,
};
use crate::manager_queue::{JobKind, ManagerJob};
use crate::prompt::{load_prompt, substitute};
use crate::role::{DIAGNOSTICS_ROLE, SYSTEM_ROLE};
use crate::session::ticket_session_key;
use crate::ticket_buffer;
use crate::tools::shell::{ShellMode, ShellTool};
use crate::turso::TxGuard;
use crate::util::panic_message;

use crate::{DiagnosticsCommands, Role, Workspace};

/// Number of parallel agents spawned per verification phase (Analyst, Reviewer, QA).
const PARALLEL_AGENT_COUNT: usize = 3;

/// Prefix for all auto-diagnostics comments on tickets.
const DIAGNOSTICS_COMMENT_PREFIX: &str = "🔍 Auto-diagnostics";
/// Comment-formatting constant — appended to the diagnostics comment body when
/// all checks pass. This is **not** a circuit-breaker marker; the circuit breaker
/// only checks for [`DIAGNOSTICS_FAILED_MARKER`] substring and [`DIAGNOSTICS_ROLE`].
const DIAGNOSTICS_PASSED_MARKER: &str = "✅ All diagnostics passed";
/// Marker appended when diagnostics fail (includes the failed-at label after it).
const DIAGNOSTICS_FAILED_MARKER: &str = "❌ Diagnostics failed at";

/// Marker for sanitation failure system comments — [`CircuitBreakerKind::Sanitation`]'s
/// [`trip_info`](CircuitBreakerKind::trip_info) depends on substring matching
/// this value, so it must not drift from comment text.
const SANITATION_FAILED_MARKER: &str = "Sanitation failed";

/// Minimum acceptable verification score (0-10) for analyst verdicts.
const ANALYST_PASS_THRESHOLD: u8 = 7;

/// Minimum acceptable verification score (0-10) for review and QA phases.
const REVIEW_QA_THRESHOLD: u8 = 9;

/// Returns the global [`BoardStore`] singleton.
#[inline]
fn board() -> &'static BoardStore {
    crate::board::store()
}

/// Best-effort clearing of `assigned_to` on early-return / error paths.
///
/// Prevents stuck tickets when a dispatch function must return without
/// transitioning the ticket. Errors are logged but not propagated —
/// callers are already on an error path and should not fail again here.
///
/// Uses [`BoardStore::clear_assigned_to_no_cancel`] — unlike
/// [`BoardStore::set_assigned_to`], this does NOT cancel any running agent.
/// All call sites are post-agent so there is no agent to cancel.
///
/// ## TOCTOU race
///
/// A concurrent claim may set a new assignee between a phase check and this
/// clear. That's very low probability and the same race is accepted in
/// [`record_sanitation_failure`].
async fn clear_assigned_to(ticket_id: &str, context: &str) {
    if let Err(e) = board().clear_assigned_to_no_cancel(ticket_id).await {
        warn!(
            ticket = %ticket_id,
            error = %e,
            "Failed to clear assigned_to: {context}",
        );
    }
}

// ── Circuit breaker kind ──────────────────────────────────────────────────────

/// Identifies which circuit breaker variant to use for phase-guard checks
/// and trip logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter)]
enum CircuitBreakerKind {
    /// General comment-count breaker: trips when the total number of comments
    /// exceeds 30.
    General,
    /// Sanitation-failure breaker: trips when cumulative sanitation failures
    /// exceed 3.
    Sanitation,
    /// Diagnostics-failure breaker: trips when cumulative diagnostics failures
    /// exceed 4.
    Diagnostics,
}

impl CircuitBreakerKind {
    /// Returns the trip-count threshold for this breaker variant.
    /// The breaker trips when [`trip_info`](CircuitBreakerKind::trip_info) returns
    /// `Some` (the count exceeds this threshold).
    const fn threshold(self) -> usize {
        match self {
            Self::General => 30,
            Self::Sanitation => 3,
            Self::Diagnostics => 4,
        }
    }

    /// Compute trip information for this breaker variant.
    ///
    /// Counts failures matching this variant's criteria from the ticket comments.
    /// If the count exceeds the variant's threshold, returns
    /// `Some((count, threshold, message))` where `message` is the formatted
    /// trip comment string to post on the ticket. Returns `None` if the breaker
    /// should not trip (count ≤ threshold).
    fn trip_info(self, comments: &[TicketComment]) -> Option<(usize, usize, String)> {
        let threshold = self.threshold();
        let count = match self {
            Self::General => comments.len(),
            Self::Sanitation => comments
                .iter()
                .filter(|c| c.role == SYSTEM_ROLE && c.content.contains(SANITATION_FAILED_MARKER))
                .count(),
            Self::Diagnostics => comments
                .iter()
                .filter(|c| {
                    c.role == DIAGNOSTICS_ROLE && c.content.contains(DIAGNOSTICS_FAILED_MARKER)
                })
                .count(),
        };

        if count <= threshold {
            return None;
        }

        let msg = match self {
            Self::General => format!(
                "Failed after {count} comments — ticket has accumulated too many comments \
                 (circuit breaker, threshold: {threshold}). \
                 Ticket failed — Manager will triage."
            ),
            Self::Sanitation => format!(
                "❌ Sanitation circuit breaker tripped after {count} cumulative failures. \
                 (threshold: {threshold})",
            ),
            Self::Diagnostics => format!(
                "{DIAGNOSTICS_COMMENT_PREFIX}\n\n❌ Circuit breaker: {count} prior diagnostic \
                 failures. Failing ticket."
            ),
        };

        Some((count, threshold, msg))
    }
}

/// Returns `true` if the ticket is in the expected phase (safe to proceed).
/// Returns `false` if the ticket was moved externally or an error occurred.
#[must_use]
async fn is_ticket_in_phase(ticket_id: &str, expected_phase: TicketPhase) -> bool {
    match board().get_ticket_phase(ticket_id).await {
        Ok(Some(phase)) => {
            let ok = phase == expected_phase;
            if !ok {
                debug!(
                    ticket = %ticket_id,
                    expected_phase = %expected_phase,
                    actual = %phase,
                    "Ticket moved externally — bailing out",
                );
            }
            ok
        }
        Ok(None) => {
            debug!(ticket = %ticket_id, "Ticket not found — row missing (violates architecture invariant)");
            false
        }
        Err(e) => {
            warn!(ticket = %ticket_id, error = %e, "Failed to check ticket phase");
            false
        }
    }
}

/// Controls whether a ticket transition triggers an immediate notification
/// to the Manager (via [`notify_ticket`]) or is buffered for batched delivery.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum NotifyPolicy {
    /// Immediately enqueue a Manager notification for this transition.
    Notify,
    /// Buffer the transition for batched delivery alongside the next
    /// notification. See [`ticket_buffer`] for details.
    Buffer,
}

/// Notification side-effect after a successful ticket transition. Parameters
/// follow the `(source, target)` convention; `source` feeds the buffer entry
/// when buffering.
async fn dispatch_notification(
    ticket: &Ticket,
    source: TicketPhase,
    target: TicketPhase,
    notify: NotifyPolicy,
) {
    match notify {
        NotifyPolicy::Notify => notify_ticket(ticket, target).await,
        NotifyPolicy::Buffer => {
            ticket_buffer::push(&ticket.workspace_name, &ticket.id, source, target);
        }
    }
}

/// The transition context for [`comment_and_transition`] and [`with_comment_and_transition`].
///
/// Encapsulates the ticket, source/target phases, notify policy, and log label
/// — everything needed for a comment+transition operation. Comment text is
/// passed as separate parameters to each function since
/// [`comment_and_transition`] always requires a comment (it is no longer
/// optional), and [`with_comment_and_transition`] delegates all comment
/// writing to a closure.
///
/// # ⚠ Argument ordering
///
/// `source` and `target` are both [`TicketPhase`], so swapping them is a
/// potential runtime bug (the database rejects the transition). Always use
/// named fields when constructing this struct.
#[derive(Debug)]
struct TransitionCtx<'a> {
    ticket: &'a Ticket,
    source: TicketPhase,
    target: TicketPhase,
    notify: NotifyPolicy,
    log_label: &'a str,
}

/// Unified helper for combining comment writes + phase transition + notification.
///
/// Wraps [`crate::turso::with_tx`] for the comment-writing closure and phase
/// transition, then dispatches a notification. The closure is responsible for
/// writing all per-agent/system comments to the database.
///
/// `pipeline_reservation` is automatically derived from the target phase:
/// `Some(true)` when transitioning to [`TicketPhase::ReadyForDevelopment`]
/// (bounce-back transitions get priority re-dispatch over fresh tickets),
/// `None` for all other transitions.
///
/// Returns `true` on success, `false` on failure (with a warning logged).
///
/// # Correctness
///
/// Uses [`BoardStore::transition_to_tx`] which does **not** cancel registered
/// agents (unlike [`BoardStore::transition_to`]).
/// This is correct because all call sites of `with_comment_and_transition` are
/// post-agent paths (verdict handling, diagnostics completion, etc.) — no
/// agents should be running on this ticket at any call site that reaches this
/// function. Do **not** call this on a path where an agent may still be
/// executing on the ticket.
#[must_use]
async fn with_comment_and_transition<F>(args: TransitionCtx<'_>, write_comments: F) -> bool
where
    F: AsyncFnOnce(&TxGuard<'_>) -> anyhow::Result<()>,
{
    let pipeline_reservation = (args.target == TicketPhase::ReadyForDevelopment).then_some(true);

    if let Err(e) = crate::turso::with_tx(
        &board().conn,
        &args.ticket.id,
        args.log_label,
        async move |tx| {
            write_comments(tx).await?;
            BoardStore::transition_to_tx(
                tx,
                &args.ticket.id,
                Some(args.source),
                args.target,
                pipeline_reservation,
            )
            .await?;
            Ok(())
        },
    )
    .await
    {
        // Use phase values directly (not strings) so phase names can't drift.
        warn!(
            ticket = %args.ticket.id,
            error = %e,
            "{}: transition to {} failed — ticket stuck in {}",
            args.log_label, args.target, args.source,
        );
        return false;
    }

    dispatch_notification(args.ticket, args.source, args.target, args.notify).await;
    true
}

/// Write a comment to a ticket, then transition it to a new phase.
///
/// Delegates to [`with_comment_and_transition`]; see that function for
/// transaction semantics, notification dispatch, and return-value conventions.
///
/// The `comment` argument is required — the compiler guarantees it is always
/// written (eliminating the previous class of bugs where `comment: None`
/// silently produced a no-comment transition).
#[must_use]
async fn comment_and_transition(ctx: TransitionCtx<'_>, comment: (&str, &str)) -> bool {
    let ticket = ctx.ticket;

    with_comment_and_transition(ctx, async |tx| {
        let (role, text) = comment;
        BoardStore::add_comment_tx(tx, &ticket.id, role, text).await?;
        Ok(())
    })
    .await
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
async fn resolve_ticket_workspace(ticket: &Ticket, log_label: &str) -> Option<crate::Workspace> {
    match crate::workspace::get_by_name(&ticket.workspace_name).await {
        Ok(Some(ws)) => Some(ws),
        Ok(None) => {
            warn!(
                ticket = %ticket.id,
                workspace_name = %ticket.workspace_name,
                "Workspace not found for ticket — {log_label}",
            );
            None
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                workspace_name = %ticket.workspace_name,
                error = %e,
                "Failed to look up workspace for ticket — {log_label}",
            );
            None
        }
    }
}

/// This is a pure notification function — it does NOT pause the workspace.
/// The Manager handles failed tickets autonomously via the triage prompt.
///
/// # Invariant: failure comment before Failed transition
///
/// When `target_phase == TicketPhase::Failed`, the failure details are read from
/// the database (last comment, any role) instead of being passed as a parameter.
/// The caller MUST ensure the failure comment has already been written to the DB
/// before calling this function (the transition closure runs first in
/// [`with_comment_and_transition`], so this invariant holds for all call paths).
/// The session key (`manager_{ws_name}`) is intentionally shared between
/// user-facing Manager chat (main.rs) and notification agents — the same Manager
/// must see both notification context and user conversation history in a unified
/// session. Do NOT change this key or add `manager_` to `TRANSIENT_SESSION_PREFIXES`
/// — it would either break context continuity or nuke user conversation history.
async fn notify_ticket(ticket: &Ticket, target_phase: TicketPhase) {
    let Some(ws) = resolve_ticket_workspace(ticket, "skipping notification").await else {
        error!(
            ticket = %ticket.id,
            workspace_name = %ticket.workspace_name,
            "Workspace resolution failed — notification skipped"
        );
        return;
    };

    let transition_log = format!(
        "[{}] {}: {} → {}",
        ticket.reporter,
        ticket.id,
        ticket.phase,
        target_phase.as_ref()
    );

    // Drain buffered non-critical transitions before rendering the
    // notification template. The drained entries are injected via the
    // canonical {{ticket_updates}} placeholder, which evaluates to an
    // empty string (harmless) when there are no buffered transitions.
    // Data-loss guard: drain only after workspace lookup succeeds
    // (above) — if lookup had failed, the buffer entries remain for
    // the next delivery attempt.
    let drained = crate::ticket_buffer::drain(&ws.name);

    let mut message = substitute(
        &load_prompt("notification.md"),
        &[
            ("{{ticket_id}}", &ticket.id),
            ("{{ticket_title}}", &ticket.title),
            ("{{ticket_phase}}", target_phase.as_ref()),
            ("{{transition_log}}", &transition_log),
            ("{{ticket_updates}}", &drained),
        ],
    );

    if target_phase == TicketPhase::Failed {
        // Fetch the last comment (any role) from the database — the failure
        // comment was already written by the closure before this notification
        // call. Falls back to a generic message if no comment exists.
        let failure_details: String = match board().get_comments(&ticket.id).await {
            Ok(comments) => comments.last().map_or_else(
                || "(unknown failure reason)".to_string(),
                |c| c.content.clone(),
            ),
            Err(_) => "(unknown failure reason)".to_string(),
        };

        let warning = substitute(
            &load_prompt("warning.md"),
            &[("{{failure_details}}", &failure_details)],
        );
        message.push_str("\n\n");
        message.push_str(&warning);
    }

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
/// The dispatch runs inside a single [`tokio::spawn`] and uses
/// [`FutureExt::catch_unwind`](futures_util::FutureExt::catch_unwind) to catch
/// panics.  On panic the ticket transitions to [`TicketPhase::Failed`] with
/// notification so the manager can investigate.
fn spawn_dispatch(phase: PollPhase, ticket: Ticket, ws: Workspace) {
    let phase_info = phase.info();
    let expected_phase = phase_info.expected_phase;
    let kind = phase_info.circuit_breaker_kind;
    let log_label = phase_info.log_label;

    info!(
        ticket = %ticket.id,
        title = %ticket.title,
        workspace = %ws.name,
        "Dispatching {} ticket",
        phase_info.log_label,
    );

    // Cancel any stale agents for this ticket before dispatching new ones.
    // This is a uniform pre-flight step that applies to all dispatch paths.
    crate::registry::AGENT_REGISTRY.cancel_by_ticket_id(&ticket.id);

    // Wrap in Arc so the panic-recovery clone is a cheap refcount bump
    // instead of a deep copy of the entire comments Vec.
    let ticket = Arc::new(ticket);
    let ticket_for_failure = Arc::clone(&ticket);

    tokio::spawn(async move {
        // ── Pre-flight guard checks ──
        //
        // Adding a new PollPhase variant requires adding a row in
        // PollPhase::info() which now also carries the circuit_breaker_kind
        // and log_label fields (enforced by the single match in info()).
        //
        // The post-agent is_ticket_in_phase check in each dispatch function is
        // a separate concern (race-condition guard) and is preserved there.
        if !is_ticket_in_phase(&ticket.id, expected_phase).await {
            return;
        }
        if try_trip_circuit_breaker(&ticket, expected_phase, kind, log_label).await {
            return;
        }

        // Correctness: AssertUnwindSafe is sound because:
        //   - `ticket` is Arc<Ticket> (atomic refcount, panic-safe); the inner
        //     Ticket data may be inconsistent after a panic, but it is consumed
        //     entirely within the unwound closure and never inspected afterwards.
        //   - `ticket_for_failure` is a separate Arc clone captured by the outer
        //     closure — it is not wrapped in AssertUnwindSafe, so panic recovery
        //     always has a valid reference for error reporting.
        //   - `ws` is moved in and consumed; no shared state remains.
        let result = std::panic::AssertUnwindSafe(async move {
            match phase {
                PollPhase::BacklogAnalysis => dispatch_backlog_analysts(ticket, ws).await,
                PollPhase::EngineerDevelopment => dispatch_engineer(ticket, ws).await,
                PollPhase::SanitationCheck => dispatch_sanitation(ticket, ws).await,
                PollPhase::DiagnosticsCheck => dispatch_diagnostics(ticket, ws).await,
                PollPhase::VerifierCheck(vi) => dispatch_verifiers(ticket, ws, vi).await,
            }
        })
        .catch_unwind()
        .await;

        if let Err(payload) = result {
            let msg = panic_message(&*payload);
            error!(
                ticket = %ticket_for_failure.id,
                panic = %msg,
                "Dispatch panicked — transitioning ticket to Failed",
            );
            // Best-effort transition: the ticket may have been moved
            // externally while the dispatch was running.
            let _ = comment_and_transition(
                TransitionCtx {
                    ticket: &ticket_for_failure,
                    source: expected_phase,
                    target: TicketPhase::Failed,
                    notify: NotifyPolicy::Notify,
                    log_label: "dispatch panic",
                },
                (SYSTEM_ROLE, &format!("❌ Dispatch panicked: {msg}")),
            )
            .await;
        }
    });
}

/// Verifier-specific metadata, embedded directly in the [`PollPhase::VerifierCheck`]
/// variant and used as the parameter to [`dispatch_verifiers`]. Carries all
/// information needed for dispatch (role, prompt paths, phase lifecycle) so
/// no round-trip through [`PollPhase::info()`] is required.
#[derive(Copy, Clone)]
struct VerifierInfo {
    role: Role,
    /// Human-readable label used in logs and circuit-breaker messages.
    ///
    /// Conventions:
    /// - Prefer Title Case: `"Sanitation"`, `"Diagnostics"`, `"Engineer"`
    /// - Keep abbreviations uppercase: `"QA"` (not `"Qa"`)
    /// - Keep lowercase-with-spaces only for natural-language phrases where
    ///   Title Case would hurt readability: `"dispatch panic"`
    /// - Plural is acceptable when the label refers to a dispatched group:
    ///   `"Reviewers"` (3 parallel agents)
    log_label: &'static str,
    success_phase: TicketPhase,
    /// The ticket phase that the verifier treats as its *source* — the phase a
    /// ticket must be in for the verifier to run (e.g. [`TicketPhase::InReview`]
    /// for reviewers, [`TicketPhase::InQa`] for QA).  This is the phase that
    /// transitions *from* when the verifier finishes (to [`success_phase`] on
    /// success, or to Failed/ReadyForDevelopment on failure).
    ///
    /// Contrast with [`PollPhaseInfo::expected_phase`] which serves as the
    /// *target* phase for claim transitions in the poll loop.
    source_phase: TicketPhase,
    prompt_template: &'static str,
    extraction_prompt_path: &'static str,
}

const REVIEWER_VI: VerifierInfo = VerifierInfo {
    role: Role::Reviewer,
    log_label: "Reviewers",
    success_phase: TicketPhase::Reviewed,
    source_phase: TicketPhase::InReview,
    prompt_template: "review.md",
    extraction_prompt_path: "extraction/reviewer.md",
};

const QA_VI: VerifierInfo = VerifierInfo {
    role: Role::Qa,
    log_label: "QA",
    success_phase: TicketPhase::QaPassed,
    source_phase: TicketPhase::InQa,
    prompt_template: "qa.md",
    extraction_prompt_path: "extraction/qa.md",
};

/// Static metadata for a single poll phase.
///
/// All phase-specific data lives here — including the circuit-breaker kind and
/// log label — sourced from the single [`PollPhase::info()`] match. Adding any
/// phase requires one row in that match.
#[derive(Copy, Clone)]
struct PollPhaseInfo {
    expected_phase: TicketPhase,
    /// How this phase checks pipeline occupancy. [`Enforce`](PipelineCheck::Enforce)
    /// blocks claims when another pipeline ticket is active in the workspace;
    /// [`Skip`](PipelineCheck::Skip) allows concurrent claims.
    pipeline_check: PipelineCheck,
    /// Which circuit breaker variant to use for phase-guard checks.
    circuit_breaker_kind: CircuitBreakerKind,
    /// Human-readable label used in logs and circuit-breaker messages.
    /// PascalCase for roles ("Engineer", "Analyst"), "QA" for the QA verifier.
    log_label: &'static str,
}

/// A single poll phase: maps a `from → to` ticket transition to the agent
/// that handles it.
///
/// Phase metadata lives in [`PollPhase::info()`] — a single match expression
/// that returns all phase-specific data. The `VerifierCheck` variant carries
/// its `VerifierInfo` inline (so reviewer and QA phases share one variant).
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
                expected_phase: TicketPhase::Analysis,
                pipeline_check: PipelineCheck::Skip,
                circuit_breaker_kind: CircuitBreakerKind::General,
                log_label: "Analyst",
            },
            Self::EngineerDevelopment => PollPhaseInfo {
                expected_phase: TicketPhase::InDevelopment,
                pipeline_check: PipelineCheck::Enforce,
                circuit_breaker_kind: CircuitBreakerKind::General,
                log_label: "Engineer",
            },
            Self::SanitationCheck => PollPhaseInfo {
                expected_phase: TicketPhase::InSanitation,
                // SanitationCheck is excluded from CLAIM_PHASES since the
                // actual QaPassed→InSanitation transition happens via
                // claim_sanitation in handle_qa_passed.
                pipeline_check: PipelineCheck::Skip,
                circuit_breaker_kind: CircuitBreakerKind::Sanitation,
                log_label: "Sanitation",
            },
            Self::DiagnosticsCheck => PollPhaseInfo {
                expected_phase: TicketPhase::InDiagnostics,
                pipeline_check: PipelineCheck::Skip,
                circuit_breaker_kind: CircuitBreakerKind::Diagnostics,
                log_label: "Diagnostics",
            },
            Self::VerifierCheck(vi) => PollPhaseInfo {
                expected_phase: vi.source_phase,
                pipeline_check: PipelineCheck::Skip,
                circuit_breaker_kind: CircuitBreakerKind::General,
                log_label: vi.log_label,
            },
        }
    }
}

/// Pipeline phases that use atomic source→expected_phase claim transitions.
///
/// Each tuple is `(source_phase, poll_phase)` — the `source_phase` is the
/// expected current phase of the ticket before claiming, and `poll_phase`
/// encodes the target phase and dispatch metadata. Encoding the source phase
/// in the tuple rather than inside [`PollPhaseInfo`] eliminates a field with
/// dual semantics (it was metadata-only for non-claim phases).
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
const CLAIM_PHASES: &[(TicketPhase, PollPhase)] = &[
    (TicketPhase::Backlog, PollPhase::BacklogAnalysis),
    (
        TicketPhase::ReadyForDevelopment,
        PollPhase::EngineerDevelopment,
    ),
    (
        TicketPhase::DiagnosticsDone,
        PollPhase::VerifierCheck(REVIEWER_VI),
    ),
    (TicketPhase::Reviewed, PollPhase::VerifierCheck(QA_VI)),
];

/// Run the given action for each ticket in `phase` for the named workspace.
///
/// Lists tickets via [`BoardStore::list_all_tickets`] with both filters set.
/// Does NOT load comments — lightweight enough for poll loops.
async fn for_tickets_in_phase(phase: TicketPhase, workspace_name: &str, action: impl Fn(Ticket)) {
    match board()
        .list_all_tickets(Some(workspace_name), Some(phase))
        .await
    {
        Ok(tickets) => {
            for ticket in tickets {
                action(ticket);
            }
        }
        Err(e) => {
            error!(workspace = workspace_name, phase = %phase, error = %e, "Phase listing failed");
        }
    }
}

/// Spawn background tasks for each ticket in the given phase.
///
/// Wraps [`for_tickets_in_phase`] with a `tokio::spawn` for each ticket, so
/// each ticket is processed concurrently and independently. The ticket stays
/// in its current phase until processing completes — transient failures cause
/// a re-dispatch on the next poll cycle rather than a transition to `Failed`.
///
/// Raw `tokio::spawn` is used here instead of `spawn_dispatch` because:
/// - There is no claim transition — the ticket stays in its phase until the
///   operation succeeds, so transient failures are harmless (re-dispatched
///   on the next poll cycle).
/// - `spawn_dispatch`'s panic-recovery moves tickets to `Failed`, but the
///   correct behavior here is to stay in the current phase for retry.
/// - No `Arc` wrapping is needed because `Ticket` is moved by value into
///   the spawned task.
async fn spawn_for_each_ticket_in_phase<F, Fut>(phase: TicketPhase, ws: &Workspace, f: F)
where
    F: Fn(Ticket, Workspace) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    for_tickets_in_phase(phase, &ws.name, |ticket| {
        let f = f.clone();
        let ws = ws.clone();
        tokio::spawn(async move {
            f(ticket, ws).await;
        });
    })
    .await;
}

/// Dispatch unassigned tickets in the given phase.
///
/// Both DiagnosticsCheck and SanitationCheck use this pattern because the
/// ticket stays in its current phase while the agent runs (rather than
/// transitioning via the claim loop). We list tickets for the phase directly
/// and guard against re-dispatch via \`assigned_to IS NULL\` — tickets that
/// already have an \`assigned_to\` value are mid-execution and should not be
/// re-dispatched.
async fn dispatch_unassigned_in_phase(
    phase: TicketPhase,
    dispatch_phase: PollPhase,
    ws: &Workspace,
) {
    for_tickets_in_phase(phase, &ws.name, |ticket| {
        if ticket.assigned_to.is_some() {
            return;
        }
        spawn_dispatch(dispatch_phase, ticket, ws.clone());
    })
    .await;
}

/// Run one poll round: claim actionable tickets and dispatch agents.
///
/// Single pass over workspaces — for each, attempt claims across all pipeline
/// phases, then handle DiagnosticsCheck and QaPassed. Previously phase-major
/// (all workspaces claim Backlog, then all claim Engineer, …); now workspace-major
/// (workspace A claims all phases, then workspace B, …). Correctness is preserved
/// because claims are atomic per-workspace and `PipelineCheck` gates
/// are checked within each workspace independently.
async fn poll_round() -> anyhow::Result<()> {
    let board = board();

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
        for &(source, phase) in CLAIM_PHASES {
            if ws.paused && matches!(phase, PollPhase::EngineerDevelopment) {
                continue;
            }
            let info = phase.info();
            let ticket = match board
                .claim_ticket_in_workspace(
                    source,
                    info.expected_phase,
                    &ws.name,
                    info.pipeline_check,
                )
                .await
            {
                Ok(Some(t)) => {
                    // Buffer the claim transition. The returned ticket already
                    // has status = info.expected_phase (from SQL RETURNING), so record
                    // the transition from source.
                    ticket_buffer::push(&ws.name, &t.id, source, t.phase);
                    t
                }
                Ok(None) => continue,
                Err(e) => {
                    error!(
                        workspace = %ws.name,
                        phase = %info.log_label,
                        error = %e,
                        "Claim failed, skipping remaining claim phases for workspace",
                    );
                    break;
                }
            };
            spawn_dispatch(phase, ticket, ws.clone());
        }

        // 2. DiagnosticsCheck — diagnostics keeps the ticket in InDiagnostics
        // while running, so the claim loop isn't applicable.
        dispatch_unassigned_in_phase(TicketPhase::InDiagnostics, PollPhase::DiagnosticsCheck, ws)
            .await;

        // 3. SanitationPassed → Done (auto-commit), following the same pattern
        // as the QaPassed→Done commit flow.
        spawn_for_each_ticket_in_phase(TicketPhase::SanitationPassed, ws, |ticket, ws| {
            finalize_ticket_from_phase(ticket, ws, TicketPhase::SanitationPassed)
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
        spawn_for_each_ticket_in_phase(TicketPhase::QaPassed, ws, |ticket, ws| {
            handle_qa_passed(ticket, ws)
        })
        .await;

        // 5. SanitationCheck — the claim (QaPassed→InSanitation) already happened
        // inside handle_qa_passed, so we only dispatch unassigned tickets.
        dispatch_unassigned_in_phase(TicketPhase::InSanitation, PollPhase::SanitationCheck, ws)
            .await;
    }

    Ok(())
}

/// Run an Engineer agent to implement the ticket.
///
/// Gathers feedback comments from all roles since the last engineer run and
/// includes them in the agent prompt. After the agent finishes, performs a
/// post-run phase check to catch race conditions, then transitions:
/// - InDiagnostics (buffer) on successful completion
/// - Failed (notify) if the agent failed or returned no output
async fn dispatch_engineer(ticket: Arc<Ticket>, ws: Workspace) {
    let session_key = ticket_session_key(&ticket.id, Role::Engineer.as_str());

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

    let message = if feedback.is_empty() {
        "Implement the ticket described in the system prompt.".to_string()
    } else {
        format!("New feedback to address:\n{}", feedback.join("\n---\n"))
    };

    if let Err(e) = board()
        .set_assigned_to(&ticket.id, Some(&session_key))
        .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Failed to set assigned_to for engineer — stale agent not cancelled",
        );
    }

    let (_agent, response) =
        run_agent(session_key, Role::Engineer, &ws, Some(&ticket), &message).await;

    // Post-run check still needed for race conditions during agent execution.
    if !is_ticket_in_phase(&ticket.id, TicketPhase::InDevelopment).await {
        clear_assigned_to(&ticket.id, "ticket left InDevelopment").await;
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

    if !comment_and_transition(
        TransitionCtx {
            ticket: &ticket,
            source: TicketPhase::InDevelopment,
            target: target_phase,
            notify,
            log_label: "Engineer",
        },
        (Role::Engineer.as_str(), comment_text),
    )
    .await
    {
        clear_assigned_to(&ticket.id, "transition failed (Engineer)").await;
        return;
    }

    info!(
        ticket = %ticket.id,
        target = %target_phase,
        "Engineer finished — transitioned ticket",
    );
}

/// Determine whether to notify immediately or buffer the Done transition.
///
/// If other active tickets remain in the workspace, the notification is
/// buffered so the Manager only gets one notification when the last ticket
/// finishes. Active tickets = `PIPELINE_BLOCKING_PHASES` + `ReadyForDevelopment`.
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
async fn transition_ticket_to_done(ticket: &Ticket, source: TicketPhase, comment: &str) {
    let notify_policy = determine_notify_policy(&ticket.workspace_name, &ticket.id).await;
    let log_label = source.as_ref();
    if comment_and_transition(
        TransitionCtx {
            ticket,
            source,
            target: TicketPhase::Done,
            notify: notify_policy,
            log_label,
        },
        (SYSTEM_ROLE, comment),
    )
    .await
    {
        info!(ticket = %ticket.id, "{comment}");
    }
}

/// Transition the ticket to Done if git is unavailable.
///
/// Returns `true` if the caller should return immediately (transition to Done
/// already performed), `false` if git is usable and normal operations should proceed.
#[must_use]
async fn transition_ticket_to_done_if_git_unavailable(
    ticket: &Ticket,
    repo_path: &Path,
    source: TicketPhase,
) -> bool {
    if !crate::git_commands::git_is_installed().await {
        transition_ticket_to_done(
            ticket,
            source,
            "Git not installed — moving to Done without commit",
        )
        .await;
        return true;
    }
    if !crate::git_commands::is_git_repo(repo_path) {
        transition_ticket_to_done(
            ticket,
            source,
            "Not a git repo — moving to Done without commit",
        )
        .await;
        return true;
    }
    false
}

/// Auto-commit changes and move the ticket to Done.
///
/// Parameterized by source phase so both the QaPassed→Done and
/// SanitationPassed→Done flows share the same implementation.
///
/// Always checks git availability via [`transition_ticket_to_done_if_git_unavailable`],
/// then runs `git status --porcelain` to determine whether the working tree is dirty.
/// - **Clean tree:** skips commit, transitions directly to Done with notification.
/// - **Dirty tree:** runs `git commit -m "<ticket title>"` via [`crate::git_commands::run_git_commit`].
/// - **Commit failure:** ticket stays in `source`, poller retries next cycle.
/// - **Git unavailable:** delegated to [`transition_ticket_to_done_if_git_unavailable`].
async fn finalize_ticket_from_phase(ticket: Ticket, ws: Workspace, source: TicketPhase) {
    let repo_path = ws.as_path();
    let phase_label = source.as_ref();

    if transition_ticket_to_done_if_git_unavailable(&ticket, repo_path, source).await {
        return;
    }

    let has_changes = match crate::git_commands::run_git_status(repo_path).await {
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

    match crate::git_commands::run_git_commit(repo_path, &ticket.title).await {
        Ok(commit_info) => {
            finalize_commit_and_transition(&ticket, commit_info, source).await;
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

/// After a successful `git commit`, persist the metadata and transition the
/// ticket to Done atomically within a single DB transaction.
///
/// Parameterized by source phase so both the QaPassed→Done and
/// SanitationPassed→Done flows share the same implementation.
async fn finalize_commit_and_transition(
    ticket: &Ticket,
    commit_info: crate::git_commands::CommitInfo,
    source: TicketPhase,
) {
    let comment = format_commit_summary(
        commit_info.short_hash(),
        commit_info.lines_added,
        commit_info.lines_removed,
    );

    let phase_label = source.as_ref();

    // Cancel agents BEFORE the transaction to avoid orphaned in-memory agents
    // if the process crashes after the commit succeeds but before cancellation
    // reaches the agent registry. If the transaction subsequently fails and the
    // ticket is re-dispatched on the next poll cycle, the cancelled agents are
    // simply re-registered — wasted work is preferable to orphaned agents on a
    // Done ticket (which crash-recovery cannot rescue).
    crate::registry::AGENT_REGISTRY.cancel_by_ticket_id(&ticket.id);

    if crate::turso::with_tx(
        &board().conn,
        &ticket.id,
        &format!(
            "finalize Done transition from {phase_label} ({})",
            commit_info.short_hash()
        ),
        async |tx| {
            BoardStore::finalize_done_tx(
                tx,
                &ticket.id,
                &commit_info.hash,
                commit_info.lines_added,
                commit_info.lines_removed,
                &comment,
                source,
            )
            .await
        },
    )
    .await
    .is_ok()
    {
        info!(ticket = %ticket.id, "Committed {}, moving to Done", commit_info.short_hash());

        let notify_policy = determine_notify_policy(&ticket.workspace_name, &ticket.id).await;
        dispatch_notification(ticket, source, TicketPhase::Done, notify_policy).await;
    } else {
        warn!(
            ticket = %ticket.id,
            short_hash = commit_info.short_hash(),
            "Commit was written to git but board transaction failed — \
             orphan commit in repo, will retry on next poll cycle",
        );
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

    // Git not available or not a git repo — transition to Done directly.
    if transition_ticket_to_done_if_git_unavailable(&ticket, repo_path, TicketPhase::QaPassed).await
    {
        return;
    }

    let porcelain = match run_git_status(repo_path).await {
        Ok(out) => out,
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to check git status for untracked files — staying in QaPassed for retry"
            );
            return;
        }
    };

    let untracked = parse_new_files_from_porcelain(&porcelain);

    if untracked.is_empty() {
        finalize_ticket_from_phase(ticket, ws, TicketPhase::QaPassed).await;
    } else {
        // Untracked files exist — claim this specific ticket to InSanitation
        // via the dedicated claim_sanitation method (see BoardStore docs).
        let session_key = ticket_session_key(&ticket.id, Role::Sanitation.as_str());
        let claimed = match board().claim_sanitation(&ticket.id, &session_key).await {
            Ok(c) => c,
            Err(e) => {
                warn!(
                    ticket = %ticket.id,
                    error = %e,
                    "Failed to transition QaPassed ticket to InSanitation"
                );
                return;
            }
        };

        if !claimed {
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

        spawn_dispatch(PollPhase::SanitationCheck, ticket, ws);
    }
}

/// Record a sanitation failure: add a system comment for the circuit breaker
/// and clear assigned_to so the ticket can be re-dispatched.
async fn record_sanitation_failure(ticket_id: &str, reason: impl std::fmt::Display) {
    let reason_str = format!("{SANITATION_FAILED_MARKER} — {reason}");
    if let Err(e) = crate::turso::with_tx(
        &board().conn,
        ticket_id,
        "record sanitation failure",
        async |tx| {
            BoardStore::add_comment_tx(tx, ticket_id, SYSTEM_ROLE, &reason_str).await?;
            BoardStore::set_assigned_to_tx(tx, ticket_id, None).await?;
            Ok(())
        },
    )
    .await
    {
        warn!(
            ticket = %ticket_id,
            error = %e,
            "Failed to record sanitation failure (circuit-breaker comment + assigned_to clear)",
        );
    }
}

/// Run the sanitation agent to inspect new/untracked files in the workspace.
///
/// Called by [`PollPhase::SanitationCheck`] via [`spawn_dispatch`]. Runs a
/// single sanitation agent with tools to inspect files and determine whether
/// they are legitimate project files or intermediate garbage.
///
/// After the agent completes, extracts a structured [`SanitationVerdict`] and
/// delegates to [`process_sanitation_verdict`] for pass/fail processing.
async fn dispatch_sanitation(ticket: Arc<Ticket>, ws: Workspace) {
    let session_key = ticket_session_key(&ticket.id, Role::Sanitation.as_str());

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
    let untracked_files = match list_new_or_untracked_files(ws.as_path()).await {
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
        clear_assigned_to(&ticket.id, "ticket left InSanitation").await;
        return;
    }

    if response.is_none() {
        // Agent failed or was cancelled — record failure and clear assigned_to
        // for re-dispatch retry. The system comment lets the sanitation circuit
        // breaker detect repeated failures.
        warn!(
            ticket = %ticket.id,
            "Sanitation agent returned no output — clearing assigned_to for retry"
        );
        record_sanitation_failure(&ticket.id, "agent returned no output").await;
        return;
    }

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
            record_sanitation_failure(&ticket.id, format!("verdict extraction error: {e}")).await;
            return;
        }
    };

    process_sanitation_verdict(&ticket, verdict).await;
}

/// Process the result of a sanitation agent inspection.
///
/// Called by [`dispatch_sanitation`] after the agent completes and the
/// [`SanitationVerdict`] has been extracted.
///
/// - If **clean** (pass = true): transitions to [`TicketPhase::SanitationPassed`]
///   (transitory handoff before auto-commit).
/// - If **garbage detected** (pass = false): adds a comment listing the offending
///   files and transitions the ticket to [`TicketPhase::ReadyForDevelopment`] with a
///   pipeline reservation (via [`comment_and_transition`]), matching the existing review/QA
///   failure pattern.
async fn process_sanitation_verdict(ticket: &Ticket, verdict: crate::SanitationVerdict) {
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
        if !comment_and_transition(
            TransitionCtx {
                ticket,
                source: TicketPhase::InSanitation,
                target: TicketPhase::SanitationPassed,
                notify: NotifyPolicy::Buffer,
                log_label: "Sanitation",
            },
            (Role::Sanitation.as_str(), &comment),
        )
        .await
        {
            clear_assigned_to(&ticket.id, "transition to SanitationPassed failed").await;
            return;
        }

        info!(
            ticket = %ticket.id,
            "Sanitation passed — transitioned to SanitationPassed",
        );
    } else {
        let garbage_list = verdict.garbage_files.join("\n- ");
        let comment = format!(
            "🗑️ Sanitation failed — garbage files detected:\n- {garbage_list}\n\nRationale: {rationale}\n\n\
             These files might have been accidentally generated by other agents in the workspace for testing purposes. Engineer needs to clean them up if they are not required in the scope of the ticket.",
            rationale = verdict.rationale,
            garbage_list = garbage_list,
        );
        // Pre-build the system comment so we can pass it into the transaction.
        let sys_comment = format!(
            "{SANITATION_FAILED_MARKER} — garbage files: {count}",
            count = verdict.garbage_files.len(),
        );

        // Write both comments and transition atomically via
        // [`with_comment_and_transition`], which wraps all writes in a single
        // transaction. This matches the pattern used by all other verdict paths.
        if !with_comment_and_transition(
            TransitionCtx {
                ticket,
                source: TicketPhase::InSanitation,
                target: TicketPhase::ReadyForDevelopment,
                notify: NotifyPolicy::Buffer,
                log_label: "Sanitation",
            },
            async |tx| {
                BoardStore::add_comment_tx(
                    tx,
                    &ticket.id,
                    Role::Sanitation.as_str(),
                    comment.as_str(),
                )
                .await?;
                BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, sys_comment.as_str())
                    .await?;
                Ok(())
            },
        )
        .await
        {
            clear_assigned_to(
                &ticket.id,
                "transition to ReadyForDevelopment failed (Sanitation)",
            )
            .await;
            return;
        }

        info!(
            ticket = %ticket.id,
            "Sanitation failed — bounced back to ReadyForDevelopment with pipeline reservation",
        );
    }
}

// ── Post-development diagnostics ───────────────────────────────────────

/// Run diagnostics commands sequentially, collecting output and pass/fail status.
///
/// Executes each non-`None` command from [`DiagnosticsCommands::commands`] via
/// [`ShellTool`], appending output (scrubbed of credentials) to an accumulating
/// comment string. Stops at the first failure (non-zero exit or execution error).
/// Appends a pass/fail marker before returning. Labels are string literals from
/// [`DiagnosticsCommands::commands`].
///
/// Returns `(comment_text, all_passed)` where `comment_text` includes the
/// [`DIAGNOSTICS_COMMENT_PREFIX`] header and the appropriate pass/fail marker.
async fn run_diagnostics_commands(diag: &DiagnosticsCommands, ws: &Workspace) -> (String, bool) {
    let mut comment = String::from(DIAGNOSTICS_COMMENT_PREFIX);
    let mut all_passed = true;
    let mut failed_at: &str = "";

    for (label, cmd_opt) in diag.commands() {
        let Some(cmd) = cmd_opt else {
            continue;
        };

        let _ = write!(comment, "\n\n{label} ({cmd}):\n");

        match ShellTool::new(ShellMode::Full)
            .execute_with_status(ws, serde_json::json!({"command": cmd}))
            .await
        {
            Ok((output, exit_code)) => {
                let display = if output.is_empty() {
                    "(no output)".to_string()
                } else {
                    output
                };
                // Output is already credential-scrubbed by ShellTool's output
                // pipeline at pipeline entry, so no further scrubbing needed.
                comment.push_str(&display);

                if exit_code != Some(0) {
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

    if all_passed {
        comment.push_str("\n\n---\n");
        comment.push_str(DIAGNOSTICS_PASSED_MARKER);
    } else {
        let _ = write!(comment, "\n\n---\n{DIAGNOSTICS_FAILED_MARKER} {failed_at}");
    }

    (comment, all_passed)
}

/// Run diagnostics commands after the engineer completes development.
///
/// Called by [`PollPhase::DiagnosticsCheck`] via [`spawn_dispatch`].
/// Checks the diagnostics-specific circuit breaker first (consistent with all
/// other dispatchers — [`dispatch_engineer`], [`dispatch_sanitation`],
/// [`dispatch_backlog_analysts`], [`dispatch_verifiers`]), then uses
/// [`BoardStore::claim_diagnostics`] to set `assigned_to` and prevent
/// double-dispatch. Unlike the pipeline-phase dispatchers (which are dispatched
/// from the atomic claim loop and already own the ticket by the time their
/// dispatch runs), diagnostics keeps the ticket in `InDiagnostics` while
/// executing, so a separate atomic claim is needed to close the TOCTOU window.
/// Loads discovered diagnostics commands for the workspace and runs them
/// sequentially via [`run_diagnostics_commands`]. Stops at the first failure.
/// After execution, transitions the ticket to either `DiagnosticsDone` (all
/// passed) or `ReadyForDevelopment` (any failure), unless the circuit breaker
/// trips (see [`CircuitBreakerKind::Diagnostics`]).
async fn dispatch_diagnostics(ticket: Arc<Ticket>, ws: Workspace) {
    // Circuit breaker check happens in spawn_dispatch before entering this
    // function — consistent with all other dispatchers.

    match board()
        .claim_diagnostics(&ticket.id, DIAGNOSTICS_ROLE)
        .await
    {
        Err(e) => {
            error!(
                ticket = %ticket.id,
                error = %e,
                "Diagnostics claim error — bailing out",
            );
            return;
        }
        Ok(false) => {
            warn!(
                ticket = %ticket.id,
                "Diagnostics claim failed — ticket already claimed or moved out of InDiagnostics"
            );
            return;
        }
        Ok(true) => {}
    }

    // Separate the decision (target phase + comment body) from the action
    // (single transition call), matching the dispatch_engineer precedent.
    let (target_phase, comment_body): (TicketPhase, String) =
        match crate::workspace::store().get_diagnostics(&ws.name).await {
            Ok(Some(cmds)) if !cmds.is_empty() => {
                // Run commands sequentially in the prescribed order.
                let (comment, all_passed) = run_diagnostics_commands(&cmds, &ws).await;

                // Post-run check: verify ticket hasn't been moved externally while
                // diagnostics commands ran. Consistent with dispatch_engineer and
                // dispatch_sanitation.
                if !is_ticket_in_phase(&ticket.id, TicketPhase::InDiagnostics).await {
                    clear_assigned_to(&ticket.id, "ticket left InDiagnostics").await;
                    return;
                }

                if all_passed {
                    // Path C1: All diagnostics passed — transition to DiagnosticsDone.
                    (TicketPhase::DiagnosticsDone, comment)
                } else {
                    // Path C2: Diagnostics failed — bounce back to development.
                    (TicketPhase::ReadyForDevelopment, comment)
                }
            }
            Ok(_) => {
                // Path B: No diagnostics commands configured (or empty list) — skip.
                (
                    TicketPhase::DiagnosticsDone,
                    "No diagnostics commands are configured for this workspace \
                     — diagnostics skipped."
                        .to_string(),
                )
            }
            Err(e) => {
                // Path A: DB error loading diagnostics — log and skip.
                warn!(
                    ticket = %ticket.id,
                    error = %e,
                    "Failed to load diagnostics for workspace — transitioning to DiagnosticsDone",
                );
                (
                    TicketPhase::DiagnosticsDone,
                    format!("Could not load diagnostics commands due to a database error: {e}"),
                )
            }
        };

    if !comment_and_transition(
        TransitionCtx {
            ticket: &ticket,
            source: TicketPhase::InDiagnostics,
            target: target_phase,
            notify: NotifyPolicy::Buffer,
            log_label: "Diagnostics",
        },
        (DIAGNOSTICS_ROLE, &comment_body),
    )
    .await
    {
        clear_assigned_to(&ticket.id, "transition failed (Diagnostics)").await;
        return;
    }

    info!(
        ticket = %ticket.id,
        target = %target_phase,
        "Diagnostics finished — transitioned ticket",
    );
}

// ── Parallel agent helpers (shared) ─────────────────────────────────────
//
// Why `process_analyst_verdicts` and `process_verifier_verdicts` are separate
// -----------------------------------------------------------------------
// Both follow the same skeleton (record comments -> classify -> transition)
// but differ in four ways that make a single unified function awkward:
//
//   * Classification — analysts use 4 categories
//     (lgtm/minor_issues/potential_blockers/missing_analysis) that feed
//     `format_analyst_summary`; reviewers/QA use a binary pass/fail via
//     `verdict_passes` against `REVIEW_QA_THRESHOLD`.
//   * Transition policy — analysts always advance to `Planning` regardless
//     of outcome (even failures proceed, just with a comment listing the
//     counts). Reviewers/QA have a 3-way outcome: all-failed -> Failed,
//     any-failed -> bounce back to development, all-pass -> success phase.
//   * Comment recording — analysts record all verdicts
//     (`VerdictFilter::All`); reviewers/QA record only failing verdicts
//     (`VerdictFilter::FailingOnly`).
//   * Signature — analysts need only `&Ticket` and `&[ParallelVerdict]`;
//     reviewers/QA need the `VerifierInfo` struct to drive the 3-way
//     transition (success phase, active phase, role label). This structural
//     difference alone prevents a shared function signature without closures.

/// Result from a single parallel verifier agent.
///
/// Three mutually-exclusive states — the type system guarantees
/// that "no response" and "parse failure" cannot be confused.
#[derive(Clone)]
enum ParallelVerdict {
    /// Agent failed to produce any response (crashed, timed out, empty output).
    NoResponse,
    /// Agent produced a response but structured verdict extraction failed.
    ParseFailed,
    /// Agent produced a successfully-parsed verdict.
    Verdict(crate::Verdict),
}

/// Run [`PARALLEL_AGENT_COUNT`] agents of the same role in parallel, then extract structured verdicts
/// from their responses.
///
/// Session keys are formatted as `ticket_{ticket.id}_{role}_{i}_{suffix}`
/// where `suffix` is a unique 6-char NanoID for retry-cycle disambiguation.
/// Each agent creates its own CancellationToken and auto-registers.
///
/// Agents with empty responses get [`ParallelVerdict::NoResponse`]; agents that
/// respond but fail to parse get [`ParallelVerdict::ParseFailed`]; successful
/// agents get [`ParallelVerdict::Verdict`]. All extraction attempts run
/// concurrently via [`join_all`].
async fn run_parallel_agents(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    role: Role,
    prompt: &str,
    extraction_prompt: &str,
) -> Vec<ParallelVerdict> {
    let suffix = crate::generate_suffix();
    let retry_prompt = load_prompt("extraction/retry.md");
    let futures: Vec<_> = (0..PARALLEL_AGENT_COUNT)
        .map(move |i| {
            let ticket = Arc::clone(ticket);
            let prompt = prompt.to_string();
            let ws = ws.clone();
            let base = ticket_session_key(&ticket.id, role.as_str());
            let session_key = format!("{base}_{i}_{suffix}");
            let extraction_prompt = extraction_prompt.to_string();
            let retry_prompt = retry_prompt.clone();
            async move {
                let (agent, response) =
                    run_agent(session_key, role, &ws, Some(&ticket), &prompt).await;
                let response = response.unwrap_or_default();
                if response.is_empty() {
                    return ParallelVerdict::NoResponse;
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
                match verdict {
                    Some(v) => ParallelVerdict::Verdict(v),
                    None => ParallelVerdict::ParseFailed,
                }
            }
        })
        .collect();
    join_all(futures).await
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
    let mut text = verdict.critique.clone().unwrap_or_default();
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
    match r {
        ParallelVerdict::Verdict(v) => {
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
            Some(comment)
        }
        ParallelVerdict::ParseFailed => Some(format!(
            "{comment_role} produced a response but verdict extraction failed — \
             treating as a failure."
        )),
        ParallelVerdict::NoResponse => Some(format!(
            "{comment_role} agent failed to produce a response — counting as a failure."
        )),
    }
}

/// Record per-agent verdict comments on a ticket (inside an existing transaction).
///
/// Analysts record ALL verdicts (passing + failing) so that every
/// verdict is visible in the ticket discussion — this differs from
/// verifiers (reviewers / QA), which only record failing comments.
async fn record_verdict_comments_tx(
    tx: &TxGuard<'_>,
    ticket_id: &str,
    results: &[ParallelVerdict],
    role_str: &str,
    filter: VerdictFilter,
) -> anyhow::Result<()> {
    for (i, r) in results.iter().enumerate() {
        let role_label = format!("{role_str}_{}", i + 1);
        if let Some(comment) = format_verdict_comment(r, &role_label, filter) {
            BoardStore::insert_comment_tx(tx, ticket_id, &role_label, &comment).await?;
        }
    }
    Ok(())
}

// ── Backlog Analysis ──────────────────────────────────────────────────

/// Spawn 3 parallel analyst agents to research a backlog ticket.
/// All verdicts are recorded as comments, then the ticket transitions to:
/// - Planning (notify) when ALL analysts pass (≥ `ANALYST_PASS_THRESHOLD`/10)
/// - Planning (notify) when any analyst fails, with a comment listing the counts
///
/// The circuit-breaker guard is handled centrally by [`spawn_dispatch`].
///
/// ## Note: no `clear_assigned_to` on post-run phase check
///
/// Unlike [`dispatch_engineer`], [`dispatch_diagnostics`], and [`dispatch_sanitation`],
/// this function does **not** call [`clear_assigned_to`] when the post-run phase check
/// fails (the ticket moved externally during analysis). This is intentional:
///
/// * **`assigned_to` is already `NULL`** — [`claim_ticket_in_workspace`] sets
///   `assigned_to = NULL` during the Backlog → InAnalysis claim
///   (see [board.rs:906-912]). There is no assigned user to clear.
/// * **Ephemeral session keys** — [`run_parallel_agents`] generates unique session
///   keys (`{base}_{i}_{suffix}`) that are never written to the ticket's `assigned_to`
///   field. The agent registry entries for these parallel agents have already finished
///   or been cancelled by the pre-flight [`AGENT_REGISTRY.cancel_by_ticket_id`] call
///   in [`spawn_dispatch`].
/// * **TOCTOU race** — calling [`clear_assigned_to`] would unnecessarily risk
///   overwriting an assignee that a concurrent claim set between the phase check and
///   the clear. Since `assigned_to` is already `NULL`, there is nothing to gain.
async fn dispatch_backlog_analysts(ticket: Arc<Ticket>, ws: Workspace) {
    let prompt_key = if ticket.reporter == Role::Maintainer.as_str() {
        "analyze/maintainer_ticket.md"
    } else {
        "analyze/manager_ticket.md"
    };
    let message = load_prompt(prompt_key);
    let extraction_prompt = load_prompt("extraction/analyst.md");
    let results =
        run_parallel_agents(&ticket, &ws, Role::Analyst, &message, &extraction_prompt).await;
    if !is_ticket_in_phase(&ticket.id, TicketPhase::Analysis).await {
        return;
    }

    process_analyst_verdicts(&ticket, &results).await;
}

/// Evaluate analyst verdicts and transition the ticket:
///
/// Records per-analyst comments (if verdict exists), counts responses and
/// extractions via post-loop iterators, then transitions:
/// - to Planning (notify) if ALL analysts passed (≥ `ANALYST_PASS_THRESHOLD`/10)
/// - to Planning (notify) if any analyst failed, with a comment listing the counts
///
/// See the "Parallel agent helpers (shared)" section for why this is separate
/// from [`process_verifier_verdicts`].
async fn process_analyst_verdicts(ticket: &Ticket, results: &[ParallelVerdict]) {
    let nonempty_count = results
        .iter()
        .filter(|r| !matches!(r, ParallelVerdict::NoResponse))
        .count();
    let total = results.len();
    let mut lgtm = 0usize;
    let mut minor_issues = 0usize;
    let mut potential_blockers = 0usize;
    let mut missing_analysis = 0usize;

    for r in results {
        match r {
            ParallelVerdict::Verdict(v)
                if v.score >= ANALYST_PASS_THRESHOLD && v.issues_detected.is_empty() =>
            {
                lgtm += 1;
            }
            ParallelVerdict::Verdict(v) if v.score >= ANALYST_PASS_THRESHOLD => minor_issues += 1,
            ParallelVerdict::Verdict(_) => potential_blockers += 1,
            ParallelVerdict::NoResponse | ParallelVerdict::ParseFailed => missing_analysis += 1,
        }
    }

    let summary = format_analyst_summary(
        total,
        lgtm,
        minor_issues,
        potential_blockers,
        missing_analysis,
    );
    let extracted_count = total - missing_analysis;
    let passing_count = lgtm + minor_issues;

    // Compare against PARALLEL_AGENT_COUNT (not extracted_count) intentionally:
    // a missing/empty verdict is treated as non-passing — all dispatched
    // analysts must produce passing verdicts for the ticket to proceed.
    let all_passed = passing_count == PARALLEL_AGENT_COUNT;

    if !with_comment_and_transition(
        TransitionCtx {
            ticket,
            source: TicketPhase::Analysis,
            target: TicketPhase::Planning,
            notify: NotifyPolicy::Notify,
            log_label: "Analyst",
        },
        async |tx| {
            record_verdict_comments_tx(
                tx,
                &ticket.id,
                results,
                Role::Analyst.as_str(),
                VerdictFilter::All,
            )
            .await?;

            BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, &summary).await?;
            Ok(())
        },
    )
    .await
    {
        return;
    }

    if all_passed {
        info!(
            ticket = %ticket.id,
            nonempty_count,
            "Backlog analysis complete — all analysts passed (≥ {ANALYST_PASS_THRESHOLD}/10)",
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

/// Format a natural-language summary of analyst verdict categories.
///
/// Categorizes each analyst as LGTM, minor issues, potential blockers, or missing
/// analysis. Only categories with non-zero counts appear in the description.
///
/// Label strings must not start with a leading space — `format!` inserts one
/// between count and label automatically when using the "All {label}" form.
fn format_analyst_summary(
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

/// After a ticket fails via circuit breaker, move all other ReadyForDevelopment
/// tickets in the same workspace to Planning so the Manager can triage the
/// failure without new tickets auto-starting.
///
/// `Planning` tickets are **not** auto-claimed by the poll loop — they require
/// Manager intervention to advance. This prevents new tickets from silently
/// proceeding while existing failures are investigated.
///
/// Does not push individual buffer entries for the moved tickets; the user is
/// already notified about the primary ticket's circuit breaker failure, so
/// per-sibling notifications are noise.
///
/// # Precondition
/// `ticket` must already be in the `Failed` phase before calling this function.
/// Callers must transition the ticket to `Failed` first.
async fn drain_ready_for_development_siblings(ticket: &Ticket) {
    match board()
        .drain_ready_for_development_to_planning(&ticket.workspace_name)
        .await
    {
        Ok(updated) if updated > 0 => {
            info!(
                tickets = updated,
                workspace = %ticket.workspace_name,
                "Moved {updated} ReadyForDevelopment ticket(s) to Planning after circuit breaker trip",
            );
        }
        Ok(_) => {
            debug!(
                workspace = %ticket.workspace_name,
                "No ReadyForDevelopment siblings to drain after circuit breaker trip",
            );
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                workspace = %ticket.workspace_name,
                error = %e,
                "Failed to move ReadyForDevelopment tickets to Planning \
                 — breaker trip proceeds without moving siblings",
            );
        }
    }
}

/// Shared circuit breaker skeleton: fetch comments, evaluate via
/// [`CircuitBreakerKind::trip_info`], add a system comment via the returned
/// message string, then transition to [`TicketPhase::Failed`].
///
/// All three concrete breakers (General, Sanitation, Diagnostics) delegate to this
/// helper, supplying their variant logic via the [`CircuitBreakerKind`] enum. This
/// eliminates ~80% structural duplication while preserving exact behavioral semantics.
///
/// The Manager is notified when the ticket transitions to [`TicketPhase::Failed`].
///
/// # Self-counting prevention
///
/// Each breaker variant naturally excludes its own trip comment from counting:
///
/// * **General breaker** — counts all comments via `comments.len()`; it prevents
///   re-dispatch by transitioning to the terminal `Failed` phase before the
///   breaker could re-read the same trip comment.
/// * **Sanitation breaker** — filters comments by role `"system"` and content
///   containing [`SANITATION_FAILED_MARKER`], but trip comments use different
///   text, so they are never counted.
/// * **Diagnostics breaker** — filters comments by role `"diagnostics"` and content
///   containing [`DIAGNOSTICS_FAILED_MARKER`];
///   trip comments always use role `SYSTEM_ROLE` (set by this function), so they
///   are never counted.
///
/// See each variant's [`CircuitBreakerKind::trip_info`] implementation for
/// the exact filtering logic.
///
/// # Return value
///
/// Returns `true` if the breaker tripped — the caller MUST abort dispatch.
/// Returns `true` even on transition failure (the caller should still abort
/// rather than dispatching an agent to a stale or unreachable ticket).
/// Returns `false` only when the count does not exceed the variant's threshold.
///
/// This transitions the ticket to `Failed` (and drains ReadyForDevelopment
/// siblings) when the breaker trips. The `try_trip` verb clearly communicates
/// this side effect.
///
/// # Parameters
///
/// * `ticket` — the ticket being evaluated for the circuit breaker.
/// * `source_phase` — the phase the ticket must currently be in for the transition
///   to succeed (passed as the `source` parameter to the inner `comment_and_transition` call).
/// * `kind` — identifies which circuit breaker variant to use. Determines
///   threshold, failure-counting logic, and trip-comment format.
/// * `log_label` — human-readable label used in log messages to identify the
///   circuit breaker caller (e.g., `"Engineer"`, `"Sanitation"`, `"Diagnostics"`).
#[must_use]
async fn try_trip_circuit_breaker(
    ticket: &Ticket,
    source_phase: TicketPhase,
    kind: CircuitBreakerKind,
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

    let Some((count, threshold, msg)) = kind.trip_info(&comments) else {
        return false;
    };

    info!(
        ticket = %ticket.id,
        count,
        threshold,
        log_label,
        "Circuit breaker tripped at {count}/{threshold} ({log_label}) — failing ticket"
    );

    if comment_and_transition(
        TransitionCtx {
            ticket,
            source: source_phase,
            target: TicketPhase::Failed,
            notify: NotifyPolicy::Notify,
            log_label: &format!("{log_label} circuit breaker"),
        },
        (SYSTEM_ROLE, &msg),
    )
    .await
    {
        drain_ready_for_development_siblings(ticket).await;
    }

    true
}

/// Process parallel verifier results: add failing comments, determine pass/fail,
/// and update ticket phase accordingly.
///
/// Handles three outcomes in priority order:
///
/// 1. **All agents failed to produce a verdict** (every result has `verdict: None`
///    — crashed, timed out, or unparseable output) → transition to [`TicketPhase::Failed`]
///    with [`NotifyPolicy::Notify`]. This is a terminal failure; retrying would waste
///    credits on a fundamentally broken dispatch.
///
/// 2. **Any verifier failed** (score below [`REVIEW_QA_THRESHOLD`]) → transition back to
///    [`TicketPhase::ReadyForDevelopment`] with a pipeline reservation (directly via
///    [`transition_to_tx`](BoardStore::transition_to_tx)). The circuit
///    breaker is already checked in [`spawn_dispatch`] before agents start, so only
///    the bounce-back is needed here.
///
/// 3. **All passed** (all at or above threshold) → transition to the verifier's
///    `success_phase` with [`NotifyPolicy::Buffer`]. No immediate notification fires —
///    it waits until the ticket reaches Done (after the QaPassed commit succeeds in
///    [`finalize_ticket_from_phase`]).
///
/// See the "Parallel agent helpers (shared)" section for why this is separate
/// from [`process_analyst_verdicts`].
async fn process_verifier_verdicts(
    ticket: &Ticket,
    results: &[ParallelVerdict],
    verifier: VerifierInfo,
) {
    let all_failed = results
        .iter()
        .all(|r| !matches!(r, ParallelVerdict::Verdict(_)));
    let any_failed = results.iter().any(|r| match r {
        ParallelVerdict::Verdict(v) => !verdict_passes(Some(v)),
        _ => true,
    });

    // Determine transition parameters based on the three-way branch:
    //   all-failed → Failed (notify, with failure comment)
    //   any-failed → ReadyForDevelopment (buffer, pipeline reservation)
    //   all-passed → verifier.success_phase (buffer)
    let (target, notify) = if all_failed {
        (TicketPhase::Failed, NotifyPolicy::Notify)
    } else if any_failed {
        (TicketPhase::ReadyForDevelopment, NotifyPolicy::Buffer)
    } else {
        (verifier.success_phase, NotifyPolicy::Buffer)
    };

    // Build the failure comment string (only used in the all-failed branch).
    let failure_comment = if all_failed {
        Some(format!(
            "❌ All {} agents failed to produce verdicts — \
             ticket marked as Failed.",
            verifier.log_label,
        ))
    } else {
        None
    };

    if !with_comment_and_transition(
        TransitionCtx {
            ticket,
            source: verifier.source_phase,
            target,
            notify,
            log_label: verifier.log_label,
        },
        async |tx| {
            record_verdict_comments_tx(
                tx,
                &ticket.id,
                results,
                verifier.role.as_str(),
                VerdictFilter::FailingOnly,
            )
            .await?;

            // For the all-failed case, also write the system failure comment
            // via with_comment_and_transition (matching the sanitation failure path).
            if let Some(ref fc) = failure_comment {
                BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, fc).await?;
            }

            Ok(())
        },
    )
    .await
    {
        return;
    }

    if !all_failed && !any_failed {
        info!(
            ticket = %ticket.id,
            "{log_label}: all passed (≥ {REVIEW_QA_THRESHOLD}/10)",
            log_label = verifier.log_label,
        );
    } else if any_failed {
        info!(
            ticket = %ticket.id,
            "{log_label} failed — pipeline reservation set for rework priority",
            log_label = verifier.log_label,
        );
    }
}

/// Shared dispatch logic for parallel verifiers (reviewers and QA).
/// Fetches the engineer's last comment, builds a prompt from the template,
/// runs [`PARALLEL_AGENT_COUNT`] parallel verifiers of the given role, and processes the verdicts.
///
/// ## Note: no `clear_assigned_to` on post-run phase check
///
/// See [`dispatch_backlog_analysts`] for the full rationale — the same
/// structural reasons apply here (parallel agents via [`run_parallel_agents`],
/// `assigned_to` set to `NULL` during the [`claim_ticket_in_workspace`] claim).
async fn dispatch_verifiers(ticket: Arc<Ticket>, ws: Workspace, vi: VerifierInfo) {
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
    let results = run_parallel_agents(&ticket, &ws, vi.role, &prompt, &extraction_prompt).await;
    if !is_ticket_in_phase(&ticket.id, vi.source_phase).await {
        return;
    }

    process_verifier_verdicts(&ticket, &results, vi).await;
}

#[cfg(test)]
#[path = "management_tests.rs"]
mod tests;
