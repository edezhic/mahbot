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

use futures_util::FutureExt;
use futures_util::future::join_all;

use crate::agent::run_agent;
use crate::board::{BOARD, BoardStore, Ticket, TicketComment, TicketPhase};
use crate::diff_parse::list_untracked_files;
use crate::manager_queue::{JobKind, ManagerJob};
use crate::prompt::{load_prompt, substitute};
use crate::role::{DIAGNOSTICS_ROLE, SYSTEM_ROLE};
use crate::session::ticket_session_key;
use crate::ticket_buffer;
use crate::tools::shell::{ShellMode, ShellTool};
use crate::util::panic_message;
use crate::util::scrub_credentials;
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

/// Maximum number of consecutive sanitation failures allowed before the
/// sanitation circuit breaker trips. The breaker trips when a ticket's
/// sanitation failure count exceeds this threshold, failing the ticket.
///
/// This is a separate, lower threshold than the general comment-count
/// circuit breaker — sanitation failures are cheap to detect and should
/// not consume 50 comments before tripping.
const SANITATION_CIRCUIT_BREAKER_THRESHOLD: usize = 3;

/// Compile-time invariant: the sanitation circuit breaker must always trip
/// before the general comment-count breaker, otherwise a ticket could
/// accumulate `CIRCUIT_BREAKER_COMMENT_THRESHOLD` comments during repeated
/// sanitation loops before tripping.
const _: () = assert!(SANITATION_CIRCUIT_BREAKER_THRESHOLD < CIRCUIT_BREAKER_COMMENT_THRESHOLD);

/// Compile-time invariant: the diagnostics circuit breaker must also always
/// trip before the general comment-count breaker, otherwise a ticket could
/// accumulate `CIRCUIT_BREAKER_COMMENT_THRESHOLD` comments during repeated
/// diagnostics loops before tripping. This is a conservative approximation
/// because the general breaker counts all comments (not just diagnostics), but
/// it guarantees that diagnostics-only chatter cannot bypass the general breaker.
const _: () = assert!(DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD < CIRCUIT_BREAKER_COMMENT_THRESHOLD);

/// Prefix for all auto-diagnostics comments on tickets.
const DIAGNOSTICS_COMMENT_PREFIX: &str = "🔍 Auto-diagnostics";
/// Comment-formatting constant — appended to the diagnostics comment body when
/// all checks pass. This is **not** a circuit-breaker marker; the circuit breaker
/// only checks for [`DIAGNOSTICS_FAILED_MARKER`] prefix and [`DIAGNOSTICS_ROLE`].
const DIAGNOSTICS_PASSED_MARKER: &str = "✅ All diagnostics passed";
/// Marker appended when diagnostics fail (includes the failed-at label after it).
const DIAGNOSTICS_FAILED_MARKER: &str = "❌ Diagnostics failed at";

/// Prefix for sanitation failure system comments — the circuit breaker's
/// `count_fn` depends on substring matching this value, so it must not drift
/// from comment text.
const SANITATION_FAILED_PREFIX: &str = "Sanitation failed";

/// Minimum acceptable verification score (0-10) for analysis phase.
const ANALYSIS_THRESHOLD: u8 = 7;

/// Minimum acceptable verification score (0-10) for review and QA phases.
const REVIEW_QA_THRESHOLD: u8 = 9;

/// Returns the global [`BoardStore`] singleton.
#[inline]
fn board() -> &'static BoardStore {
    crate::board::store()
}

// ── Circuit breaker helper functions ──────────────────────────────────────────

/// Count only sanitation-failure system comments (role == `SYSTEM_ROLE`, content
/// contains [`SANITATION_FAILED_PREFIX`]). Used by the sanitation circuit breaker.
fn count_sanitation_failures(comments: &[TicketComment]) -> usize {
    comments
        .iter()
        .filter(|c| c.role == SYSTEM_ROLE && c.content.contains(SANITATION_FAILED_PREFIX))
        .count()
}

fn general_breaker_comment(count: usize) -> String {
    format!(
        "Failed after {count} comments — ticket has accumulated too many comments \
         (circuit breaker, threshold: {CIRCUIT_BREAKER_COMMENT_THRESHOLD}). \
         Ticket failed — Manager will triage."
    )
}

fn sanitation_breaker_comment(count: usize) -> String {
    format!(
        "❌ Sanitation circuit breaker tripped after {count} consecutive failures. \
         (threshold: {SANITATION_CIRCUIT_BREAKER_THRESHOLD})",
    )
}

/// Count prior diagnostics failure comments (role == `DIAGNOSTICS_ROLE`, content
/// starts with [`DIAGNOSTICS_COMMENT_PREFIX`] and contains [`DIAGNOSTICS_FAILED_MARKER`]).
/// Used by the diagnostics circuit breaker.
fn count_diagnostics_failures(comments: &[TicketComment]) -> usize {
    comments
        .iter()
        .filter(|c| {
            c.role == DIAGNOSTICS_ROLE
                && c.content.starts_with(DIAGNOSTICS_COMMENT_PREFIX)
                && c.content.contains(DIAGNOSTICS_FAILED_MARKER)
        })
        .count()
}

/// Format the circuit-breaker trip comment for diagnostics failures.
fn diagnostics_breaker_comment(count: usize) -> String {
    format!(
        "{DIAGNOSTICS_COMMENT_PREFIX}\n\n❌ Circuit breaker: {count} prior diagnostic \
         failures. Failing ticket."
    )
}

/// Returns `true` if the ticket is in the expected phase (safe to proceed).
/// Returns `false` if the ticket was moved externally or an error occurred.
#[must_use]
async fn is_ticket_in_phase(ticket_id: &str, expected: TicketPhase) -> bool {
    match board().get_ticket_phase(ticket_id).await {
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
/// Used directly by [`dispatch_engineer`] and by [`dispatch_parallel_with_guard`]
/// (which wraps the guard + dispatch + post-check skeleton shared by
/// [`dispatch_backlog_analysts`] and [`dispatch_verifiers`]).
/// Diagnostics uses a separate circuit breaker (see [`run_circuit_breaker`]).
#[must_use]
async fn guard_phase_and_circuit_breaker(
    ticket: &Ticket,
    expected: TicketPhase,
    label: &str,
) -> bool {
    if !is_ticket_in_phase(&ticket.id, expected).await {
        return false;
    }
    if run_circuit_breaker(
        ticket,
        expected,
        CIRCUIT_BREAKER_COMMENT_THRESHOLD,
        <[TicketComment]>::len,
        general_breaker_comment,
        label,
    )
    .await
    {
        return false;
    }
    true
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

/// Dispatch a ticket transition notification, either immediately or buffered.
///
/// Called after a successful transition to handle the notification side-effect.
/// `source` is the phase the ticket transitioned *from* (used for the buffer
/// entry).
///
/// The `log_label` string is forwarded to [`resolve_ticket_workspace`] to
/// distinguish callers in log messages.
async fn dispatch_notification(
    ticket: &Ticket,
    target: TicketPhase,
    source: TicketPhase,
    notify: NotifyPolicy,
    log_label: &'static str,
) {
    match notify {
        NotifyPolicy::Notify => notify_ticket(ticket, target).await,
        NotifyPolicy::Buffer => {
            if let Some(ws) = resolve_ticket_workspace(ticket, log_label).await {
                ticket_buffer::push(&ws.name, &ticket.id, source, target);
            }
        }
    }
}

/// Transition a ticket to `target` phase if it's still in `source` phase.
///
/// Returns `Ok(())` if the transition was applied (ticket was still in source
/// phase). Returns `Err(anyhow::Error)` with a descriptive message if the ticket was
/// moved externally (phase mismatch) **or** a DB error occurred. On failure,
/// does **not** clear `assigned_to` — doing so would either steal the new
/// owner's assignment (phase mismatch) or orphan the ticket (DB error), and
/// [`BoardStore::transition_to`] already handles agent cancellation on the
/// success path.
///
/// The `pipeline_reservation` parameter is forwarded to
/// [`BoardStore::transition_to`]: pass `Some(true)` for bounce-back transitions
/// (back to [`TicketPhase::ReadyForDevelopment`]) to ensure the ticket gets
/// priority re-dispatch over fresh tickets, or `None` for all other transitions.
///
/// If `notify` is [`NotifyPolicy::Notify`], delegates to [`dispatch_notification`]
/// on success (which may enqueue a notification or buffer the transition);
/// errors from notification are logged and discarded (not propagated).
async fn transition_ticket(
    ticket: &Ticket,
    source: TicketPhase,
    target: TicketPhase,
    notify: NotifyPolicy,
    pipeline_reservation: Option<bool>,
) -> anyhow::Result<()> {
    match board()
        .transition_to(&ticket.id, Some(source), target, pipeline_reservation)
        .await
    {
        Ok(()) => {
            dispatch_notification(ticket, target, source, notify, "cannot buffer transition").await;

            Ok(())
        }
        Err(e) => {
            debug!(
                ticket = %ticket.id,
                source = %source,
                target = %target,
                error = %e,
                "Failed to update ticket status",
            );
            Err(e)
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
    log_label: &'static str,
) -> Option<crate::Workspace> {
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

/// Bounce a ticket back to ReadyForDevelopment with pipeline reservation,
/// logging success/failure. Used when diagnostics, sanitation, or verifier
/// review/QA fails and the ticket needs priority re-dispatch.
async fn bounce_back_to_development(ticket: &Ticket, source: TicketPhase, log_label: &str) {
    if let Err(e) = transition_ticket(
        ticket,
        source,
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
            log_label = log_label,
        );
    } else {
        info!(
            ticket = %ticket.id,
            "{log_label} failed — pipeline reservation set for rework priority",
            log_label = log_label,
        );
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
/// # Invariant: failure comment before Failed transition
///
/// When `status == TicketPhase::Failed`, the warning template loads
/// failure-specific details from the **latest comment** on the ticket (via
/// [`BoardStore::get_comments`]). Every code path that transitions a ticket to
/// `Failed` MUST write a relevant comment first — otherwise the warning will
/// surface unrelated or stale content (or "No failure details available."
/// if no comment exists). This invariant is upheld by all four
/// existing failure paths (dispatch panic, agent failure, circuit breaker, all
/// verifiers failed) but is not enforced at the type or API level.
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

    let transition_log = format!(
        "[{}] {}: {} → {}",
        ticket.reporter,
        ticket.id,
        ticket.phase,
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

    let mut message = substitute(
        &load_prompt("notification.md"),
        &[
            ("{{ticket_id}}", &ticket.id),
            ("{{ticket_title}}", &ticket.title),
            ("{{ticket_status}}", status.as_ref()),
            ("{{transition_log}}", &transition_log),
            ("{{ticket_updates}}", &drained),
        ],
    );

    if status == TicketPhase::Failed {
        let failure_details = match board().get_comments(&ticket.id).await {
            Ok(comments) => comments.last().map_or_else(
                || "No failure details available.".to_string(),
                |c| c.content.clone(),
            ),
            Err(e) => {
                warn!(
                    ticket = %ticket.id,
                    error = %e,
                    "Failed to load comments for failure notification",
                );
                "No failure details available.".to_string()
            }
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
    let active_phase = phase_info.active_phase;

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
        // Safety: AssertUnwindSafe is sound because:
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
            let _ = board()
                .add_comment(
                    &ticket_for_failure.id,
                    SYSTEM_ROLE,
                    &format!("❌ Dispatch panicked: {msg}"),
                )
                .await;
            if let Err(e) = transition_ticket(
                &ticket_for_failure,
                active_phase,
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
    });
}

/// Verifier-specific metadata, embedded directly in the [`PollPhase::VerifierCheck`]
/// variant and used as the parameter to [`dispatch_verifiers`]. Carries all
/// information needed for dispatch (role, prompt paths, phase lifecycle) so
/// no round-trip through [`PollPhase::info()`] is required.
#[derive(Copy, Clone)]
struct VerifierInfo {
    role: Role,
    log_label: &'static str,
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
    success_phase: TicketPhase::Reviewed,
    active_phase: TicketPhase::InReview,
    prompt_template: "review.md",
    extraction_prompt_path: "extraction/reviewer.md",
};

const QA_VI: VerifierInfo = VerifierInfo {
    role: Role::Qa,
    log_label: "QA",
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
    active_phase: TicketPhase,
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
                active_phase: TicketPhase::Analysis,
                require_clear_pipeline: false,
                role_label: Role::Analyst.as_str(),
            },
            Self::EngineerDevelopment => PollPhaseInfo {
                active_phase: TicketPhase::InDevelopment,
                require_clear_pipeline: true,
                role_label: Role::Engineer.as_str(),
            },
            Self::SanitationCheck => PollPhaseInfo {
                active_phase: TicketPhase::InSanitation,
                // Note: active_phase is consumed by spawn_dispatch's
                // panic-recovery transition (active_phase → Failed).
                // SanitationCheck is excluded from CLAIM_PHASES since the
                // actual QaPassed→InSanitation transition happens via
                // claim_sanitation in handle_qa_passed.
                require_clear_pipeline: false,
                role_label: Role::Sanitation.as_str(),
            },
            Self::DiagnosticsCheck => PollPhaseInfo {
                active_phase: TicketPhase::InDiagnostics,
                require_clear_pipeline: false,
                role_label: DIAGNOSTICS_ROLE,
            },
            Self::VerifierCheck(vi) => PollPhaseInfo {
                active_phase: vi.active_phase,
                require_clear_pipeline: false,
                role_label: vi.role.as_str(),
            },
        }
    }
}

/// Pipeline phases that use atomic source→active_phase claim transitions.
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
/// Lists tickets via [`BoardStore::list_tickets_in_phase`], iterates, and logs
/// a structured error on failure.
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
/// because claims are atomic per-workspace and `require_clear_pipeline` gates
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
                    info.active_phase,
                    &ws.name,
                    info.require_clear_pipeline,
                )
                .await
            {
                Ok(Some(t)) => {
                    // Buffer the claim transition. The returned ticket already
                    // has status = info.active_phase (from SQL RETURNING), so record
                    // the transition from source.
                    ticket_buffer::push(&ws.name, &t.id, source, t.phase);
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

        // 2. DiagnosticsCheck — diagnostics keeps the ticket in InDiagnostics
        // while running, so the claim loop isn't applicable.
        dispatch_unassigned_in_phase(TicketPhase::InDiagnostics, PollPhase::DiagnosticsCheck, ws)
            .await;

        // 3. SanitationPassed → Done (auto-commit).
        //
        // After the sanitation agent approves, the ticket reaches SanitationPassed.
        // We commit the changes and transition to Done, following the same pattern
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
/// Parameterized by source phase so both the QaPassed→Done and
/// SanitationPassed→Done flows share the same implementation.
///
/// Checks for a dirty working tree via `git status --porcelain`:
/// - **Clean tree:** skips commit, transitions directly to Done with notification.
/// - **Dirty tree:** runs `git commit -m "<ticket title>"` via [`crate::diff_parse::run_git_commit`].
/// - **Commit failure:** ticket stays in `source`, poller retries next cycle.
/// - **Not a git repo / no git installed:** transitions to Done without commit.
async fn finalize_ticket_from_phase(ticket: Ticket, ws: Workspace, source: TicketPhase) {
    let repo_path = ws.as_path();
    let phase_label = source.as_ref();

    let reason = if !crate::diff_parse::git_is_installed().await {
        Some("Git not installed — moving to Done without commit")
    } else if !crate::diff_parse::is_git_repo(repo_path) {
        Some("Not a git repo — moving to Done without commit")
    } else {
        None
    };
    if let Some(r) = reason {
        transition_ticket_to_done(&ticket, source, r).await;
        return;
    }

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

/// Begin a transaction, run `work`, and commit on success.
///
/// `action_label` accepts any `&str` including dynamic temporaries from
/// `format!` — it is intentionally not `&'static str` to allow callers to
/// include dynamic context (e.g. phase transitions, short hashes) in log
/// messages. Use a verb phrase for natural reading (e.g. "record sanitation
/// failure", "write verdict comments") rather than a bare noun.
async fn with_tx(
    ticket_id: &str,
    action_label: &str,
    work: impl AsyncFnOnce(&crate::turso::TxGuard<'_>) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    crate::turso::with_tx(&board().conn, ticket_id, action_label, work).await
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

    // Cancel agents BEFORE the transaction to avoid orphaned in-memory agents
    // if the process crashes after the commit succeeds but before cancellation
    // reaches the agent registry. If the transaction subsequently fails and the
    // ticket is re-dispatched on the next poll cycle, the cancelled agents are
    // simply re-registered — wasted work is preferable to orphaned agents on a
    // Done ticket (which crash-recovery cannot rescue).
    crate::registry::AGENT_REGISTRY.cancel_by_ticket_id(&ticket.id);

    if with_tx(
        &ticket.id,
        &format!("finalize Done transition from {phase_label} ({short_hash})"),
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
            BoardStore::transition_to_tx(tx, &ticket.id, Some(source), TicketPhase::Done, None)
                .await?;
            Ok(())
        },
    )
    .await
    .is_ok()
    {
        info!(ticket = %ticket.id, "Committed {short_hash}, moving to Done");

        let notify_policy = determine_notify_policy(&ticket.workspace_name, &ticket.id).await;
        dispatch_notification(
            ticket,
            TicketPhase::Done,
            source,
            notify_policy,
            "cannot buffer Done transition",
        )
        .await;
    } else {
        warn!(
            ticket = %ticket.id,
            short_hash,
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

    // Only check git if it's available and the repo exists.
    if !crate::diff_parse::git_is_installed().await || !crate::diff_parse::is_git_repo(repo_path) {
        finalize_ticket_from_phase(ticket, ws, TicketPhase::QaPassed).await;
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
        finalize_ticket_from_phase(ticket, ws, TicketPhase::QaPassed).await;
        return;
    }

    // Untracked files exist — claim this specific ticket to InSanitation
    // via the dedicated claim_sanitation method (see BoardStore docs).
    let claimed = match board().claim_sanitation(&ticket.id).await {
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

/// Record a sanitation failure: add a system comment for the circuit breaker
/// and clear assigned_to so the ticket can be re-dispatched.
async fn record_sanitation_failure(ticket_id: &str, reason: impl std::fmt::Display) {
    let reason_str = format!("{SANITATION_FAILED_PREFIX} — {reason}");
    let _ = with_tx(ticket_id, "record sanitation failure", async |tx| {
        BoardStore::add_comment_tx(tx, ticket_id, SYSTEM_ROLE, &reason_str).await?;
        BoardStore::set_assigned_to_tx(tx, ticket_id, None).await?;
        Ok(())
    })
    .await;
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
    //
    // Sanitation circuit breaker — trip if the ticket has accumulated too
    // many consecutive sanitation failures.
    //
    // Delegates to run_circuit_breaker which calls
    // drain_ready_for_development_siblings to move other ReadyForDevelopment
    // tickets to Planning, and consistent failure handling with the general breaker.
    //
    // Counts system comments where role == `SYSTEM_ROLE` and content contains
    // "Sanitation failed".
    //
    // Separate from the general comment-count circuit breaker so that
    // garbage-thrashing tickets are caught early without consuming the full
    // 50-comment budget.
    if run_circuit_breaker(
        &ticket,
        TicketPhase::InSanitation,
        SANITATION_CIRCUIT_BREAKER_THRESHOLD,
        count_sanitation_failures,
        sanitation_breaker_comment,
        "Sanitation",
    )
    .await
    {
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
        record_sanitation_failure(&ticket.id, "agent returned no output").await;
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
            record_sanitation_failure(&ticket.id, format!("verdict extraction error: {e}")).await;
            return;
        }
    };

    if verdict.pass {
        // Clean — transition to SanitationPassed for auto-commit.
        info!(
            ticket = %ticket.id,
            "Sanitation passed — transitioning to SanitationPassed",
        );

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
            "🗑️ Sanitation failed — garbage files detected:\n- {garbage_list}\n\nRationale: {rationale}
            
            These files might have been accidentally generated by other agents in the workspace for testing purposes. Engineer needs to clean them up if they are not required in the scope of the ticket.",
            rationale = verdict.rationale,
        );
        let _ = board()
            .add_comment(&ticket.id, Role::Sanitation.as_str(), &comment)
            .await;

        // Also add a system comment so the sanitation circuit breaker can detect it.
        let _ = board()
            .add_comment(
                &ticket.id,
                SYSTEM_ROLE,
                &format!(
                    "{SANITATION_FAILED_PREFIX} — garbage files: {count}",
                    count = verdict.garbage_files.len(),
                ),
            )
            .await;

        bounce_back_to_development(&ticket, TicketPhase::InSanitation, "Sanitation").await;
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
    // Counts prior diagnostics system comments that indicate failures, and
    // fails the ticket if the count exceeds DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD
    // (i.e., trip at ≥5 failures).
    if run_circuit_breaker(
        &ticket,
        TicketPhase::InDiagnostics,
        DIAGNOSTICS_CIRCUIT_BREAKER_THRESHOLD,
        count_diagnostics_failures,
        diagnostics_breaker_comment,
        "Diagnostics",
    )
    .await
    {
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
                comment.push_str(&scrub_credentials(&display));

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
    if all_passed {
        comment.push_str("\n\n---\n");
        comment.push_str(DIAGNOSTICS_PASSED_MARKER);
    } else {
        let _ = write!(comment, "\n\n---\n{DIAGNOSTICS_FAILED_MARKER} {failed_at}");
    }
    let _ = board()
        .add_comment(&ticket.id, DIAGNOSTICS_ROLE, &comment)
        .await;

    if all_passed {
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
                "Diagnostics completed but transition to DiagnosticsDone \
                 failed — ticket stuck in DiagnosticsDone",
            );
        }
    } else {
        bounce_back_to_development(&ticket, TicketPhase::InDiagnostics, "Diagnostics").await;
    }
}

// ── Parallel agent helpers (shared) ─────────────────────────────────────

/// Result from a single parallel verifier agent.
#[derive(Clone)]
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
    let _ = with_tx(ticket_id, "write verdict comments", async |tx| {
        for (i, r) in results.iter().enumerate() {
            let role_label = format!("{role_str}_{}", i + 1);
            if let Some(comment) = format_verdict_comment(r, &role_label, filter) {
                BoardStore::add_comment_tx(tx, ticket_id, &role_label, &comment).await?;
            }
        }
        Ok(())
    })
    .await;
}

/// Shared orchestration skeleton for dispatch functions that spawn parallel
/// agents and then process results via extraction.
///
/// 1. Runs [`guard_phase_and_circuit_breaker`] (phase check + circuit breaker)
/// 2. Spawns agents via [`run_parallel_with_extraction`]
/// 3. Runs a post-dispatch phase check (guard against race conditions)
///
/// Returns `None` if either guard failed — the caller should bail out.
/// Returns `Some(results)` when it's safe to process the verdicts.
///
/// Used by [`dispatch_backlog_analysts`] and [`dispatch_verifiers`].
async fn dispatch_parallel_with_guard(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    guard_phase: TicketPhase,
    guard_label: &str,
    role: Role,
    prompt: &str,
    extraction_prompt: &str,
) -> Option<Vec<ParallelVerdict>> {
    if !guard_phase_and_circuit_breaker(ticket, guard_phase, guard_label).await {
        return None;
    }
    let results = run_parallel_with_extraction(ticket, ws, role, prompt, extraction_prompt).await;
    if !is_ticket_in_phase(&ticket.id, guard_phase).await {
        return None;
    }
    Some(results)
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
    let message = load_prompt("analyze.md");
    let extraction_prompt = load_prompt("extraction/analyst.md");
    let Some(results) = dispatch_parallel_with_guard(
        &ticket,
        &ws,
        TicketPhase::Analysis,
        "Analysts",
        Role::Analyst,
        &message,
        &extraction_prompt,
    )
    .await
    else {
        return;
    };

    handle_analyst_verdicts(&ticket, &results).await;
}

/// Evaluate analyst verdicts and transition the ticket:
///
/// Records per-analyst comments (if verdict exists), counts responses and
/// extractions via post-loop iterators, then transitions:
/// - to Planning (notify) if ALL analysts passed (≥ `ANALYSIS_THRESHOLD`/10)
/// - to Planning (notify) if any analyst failed, with a comment listing the counts
///
/// # Design note: why this is separate from [`process_verdict_results`]
///
/// Both functions follow the same skeleton (record comments → classify → transition)
/// but differ in semantics that make a unified awkward:
///
/// * **Classification** — analysts use 4 categories (`lgtm`/`minor_issues`/`potential_blockers`/`missing_analysis`)
///   while reviewers/QA use a binary pass/fail. The 4-class output feeds
///   [`build_analyst_summary`], which has no reviewer/QA equivalent.
/// * **Transition policy** — analysts always transition to `Planning` regardless of
///   outcome (even failures proceed, just with a comment listing the counts).
///   Reviewers/QA have a 3-way outcome: all-failed→Failed, any-failed→bounce back
///   to development, all-pass→success phase.
/// * **Comment recording** — analysts record all verdicts (`VerdictFilter::All`);
///   reviewers/QA record only failing verdicts (`VerdictFilter::FailingOnly`).
/// * **Signature** — analysts only need a `&Ticket` and `&[ParallelVerdict]`; reviewers/QA
///   need the [`VerifierInfo`] struct to drive the 3-way transition (success phase,
///   active phase, role label). This structural difference alone prevents a shared
///   function signature without closures.
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
    let _ = board().add_comment(&ticket.id, SYSTEM_ROLE, &summary).await;

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

/// After a ticket fails via circuit breaker, move all other ReadyForDevelopment
/// tickets in the same workspace to Planning so the Manager can triage the
/// failure without new tickets auto-starting.
async fn drain_ready_for_development_siblings(ticket: &Ticket) {
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
            return;
        }
    };

    let planning_move_comment = format!(
        "Moved to planning due to circuit breaker trip on {}: {}. Re-advance to ReadyForDevelopment after Manager resolves the failure.",
        ticket.id, ticket.title,
    );

    // There is a small race window: between listing ReadyForDevelopment
    // tickets here and transitioning them individually, a concurrent poll
    // cycle could claim one. This is rare in practice (dispatch tasks spawn
    // after the claim loop completes) and the CAS guard in transition_to
    // handles the transition gracefully.
    //
    // Filter out the tripped ticket: defense-in-depth. The tripped ticket
    // was already transitioned to Failed by the caller before this function
    // runs and shouldn't appear in the ReadyForDevelopment results, but the
    // filter keeps us safe if a concurrent race or future refactor changes
    // the timing.
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
            .add_comment(&other.id, SYSTEM_ROLE, &planning_move_comment)
            .await;
    }
}

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
/// The `count_fn` closure must not count the breaker's own trip comment.
/// Each domain-specific breaker naturally excludes its trip comment:
///
/// * **Diagnostics breaker** — filters comments by role `"diagnostics"`,
///   but trip comments always use role `SYSTEM_ROLE` (set by this function).
/// * **Sanitation breaker** — filters comments by content containing
///   `"Sanitation failed"`, but trip comments use different text.
/// * **General breaker** — counts all comments (`len`); it prevents
///   re-dispatch by transitioning to the terminal `Failed` phase.
///
/// The `comment_text` closure should produce output consistent with
/// `count_fn`'s filters to avoid accidental self-counting.
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
/// * `ticket` — the ticket being evaluated for the circuit breaker.
/// * `expected` — the phase the ticket must currently be in for the transition
///   to succeed (passed through to [`transition_ticket`]).
/// * `threshold` — the count at which the breaker trips (using `>` comparison).
/// * `count_fn` — extracts the count from the fetched comment list. Responsible
///   for its own filtering (including self-counting prevention).
/// * `comment_text` — formats the system comment body given the count.
/// * `log_label` — human-readable label used in log messages to identify the
///   circuit breaker caller.
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
        .add_comment(&ticket.id, SYSTEM_ROLE, &comment_text(count))
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

    drain_ready_for_development_siblings(ticket).await;

    true
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
///    [`finalize_ticket_from_phase`]).
///
/// # Design note: why this is separate from [`handle_analyst_verdicts`]
///
/// Both functions follow the same skeleton (record comments → classify → transition)
/// but differ in semantics that make a unified handler awkward:
///
/// * **Classification** — verifiers use a binary pass/fail (via [`verdict_passes`])
///   against [`REVIEW_QA_THRESHOLD`]. Analysts use 4 categories
///   (`lgtm`/`minor_issues`/`potential_blockers`/`missing_analysis`) that feed a
///   natural-language summary with no reviewer/QA equivalent.
/// * **Transition policy** — verifiers have a 3-way outcome: all-failed→[`TicketPhase::Failed`],
///   any-failed→bounce back to development, all-pass→[`VerifierInfo::success_phase`].
///   Analysts always transition to `Planning` regardless of outcome.
/// * **Comment recording** — verifiers record only failing verdicts
///   ([`VerdictFilter::FailingOnly`]); analysts record all verdicts ([`VerdictFilter::All`]).
/// * **Signature** — verifiers require [`VerifierInfo`] to drive the 3-way transition
///   (carries `success_phase`, `active_phase`, role label). Analysts only need the
///   ticket and results. This structural difference alone prevents a shared signature
///   without closures.
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
                SYSTEM_ROLE,
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
    if results.iter().any(|r| !verdict_passes(r.verdict.as_ref())) {
        bounce_back_to_development(ticket, verifier.active_phase, verifier.log_label).await;
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
    let Some(results) = dispatch_parallel_with_guard(
        &ticket,
        &ws,
        vi.active_phase,
        vi.log_label,
        vi.role,
        &prompt,
        &extraction_prompt,
    )
    .await
    else {
        return;
    };

    process_verdict_results(&ticket, &results, vi).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::DEFAULT_TICKET_PHASE;
    use crate::util::test::make_ticket;
    use crate::util::test::{expect_ticket, expect_ticket_phase, init_test_stores};
    use crate::workspace::test_ws_named;

    /// Verify that `guard_phase_and_circuit_breaker` rejects a ticket
    /// whose phase does not match `expected`. This validates that the
    /// pre-agent guard works correctly for `dispatch_verifiers` (and all
    /// other agent-spawning dispatch functions).
    #[tokio::test]
    async fn guard_phase_mismatch_rejected() {
        init_test_stores().await;

        let ws = test_ws_named("/tmp/test", "test");
        let ticket_id = make_ticket(board(), ws, "Test", TicketPhase::Backlog).await;

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

        let ticket = expect_ticket(board(), &ticket_id).await;

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
    #[allow(clippy::too_many_lines)]
    #[tokio::test]
    async fn circuit_breaker_moves_other_ready_for_development_tickets_to_planning() {
        init_management_test_stores().await;

        let ws_a = test_ws_named("/ws_a", "ws_a");
        let ws_b = test_ws_named("/ws_b", "ws_b");

        // Create ticket A in workspace A — this will trip the circuit breaker.
        let trip_id = make_ticket(
            board(),
            ws_a.clone(),
            "Trip Ticket",
            TicketPhase::ReadyForDevelopment,
        )
        .await;

        // Create ticket B in workspace A — this should be moved to Planning when A trips.
        let victim_id = make_ticket(
            board(),
            ws_a,
            "Victim Ticket",
            TicketPhase::ReadyForDevelopment,
        )
        .await;

        // Create ticket C in workspace B — this must NOT be moved.
        let other_ws_id = make_ticket(
            board(),
            ws_b,
            "Other Workspace Ticket",
            TicketPhase::ReadyForDevelopment,
        )
        .await;

        // Add a comment to ticket A so the circuit breaker has something to count.
        board()
            .add_comment(&trip_id, SYSTEM_ROLE, "Some comment")
            .await
            .expect("add_comment to A");

        // Fetch ticket A and trip the circuit breaker with threshold 0.
        let ticket_a = expect_ticket(board(), &trip_id).await;

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
            let ticket_a = expect_ticket(board(), &trip_id).await;
            assert_eq!(
                ticket_a.phase,
                TicketPhase::Failed,
                "tripped ticket A should be Failed"
            );
        }

        // ── Verify ticket B (same workspace) is Planning ──
        {
            let ticket_b = expect_ticket(board(), &victim_id).await;
            assert_eq!(
                ticket_b.phase,
                TicketPhase::Planning,
                "other ReadyForDevelopment ticket B in same workspace should be Planning"
            );
        }

        // ── Verify ticket C (different workspace) is still ReadyForDevelopment ──
        {
            let ticket_c = expect_ticket(board(), &other_ws_id).await;
            assert_eq!(
                ticket_c.phase,
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
                .find(|c| c.role == SYSTEM_ROLE)
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
        let ticket_id = make_ticket(board(), ws, "Test", TicketPhase::Backlog).await;

        // ── FailingOnly with all-passing verdicts ──
        // Should produce 0 comments (nothing to write).
        let results = vec![pass_result("Looks good.")];
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
        let results = vec![fail_result("Has issues.")];
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
        let results = vec![
            analyst_verdict("Agent 1 response.", 10, "Excellent analysis.", &[]),
            analyst_verdict(
                "Agent 2 response.",
                4,
                "Needs more research.",
                &["Missing citations"],
            ),
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

    /// Initialize all stores needed by management tests that call
    /// [`transition_ticket`] or interact with the ticket buffer.
    ///
    /// Idempotent across concurrent tests — all globals use OnceCell/OnceLock
    /// guards, so duplicate initialization is a harmless no-op.
    async fn init_management_test_stores() {
        init_test_stores().await;

        let _ = crate::workspace::init_global().await;
        let _ = crate::manager_queue::init_global();
    }

    /// Create a test workspace — parameters are `(name, path)`;
    /// [`test_ws_named`] takes `(path, name)`, so the order is swapped internally.
    async fn create_test_workspace(name: &str, path: &str) -> crate::Workspace {
        let now = crate::turso::now();
        crate::workspace::store()
            .conn
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at, paused) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                turso::params![name, path, now.clone(), now, 0],
            )
            .await
            .expect("insert test workspace");
        test_ws_named(path, name)
    }

    /// Shorthand for [`init_management_test_stores`] + [`create_test_workspace`]
    /// with a generated `ws_{suffix}` / `/tmp/test_{suffix}` name/path.
    ///
    /// Creates a **DB-backed** workspace (inserted into the test DB), unlike
    /// [`setup_ticket`] which returns an in-memory workspace.
    ///
    /// Each test must pass a unique `suffix` to avoid UNIQUE constraint
    /// and cross-test pollution on the shared ticket buffer.
    async fn setup_db_workspace(suffix: &str) -> crate::Workspace {
        init_management_test_stores().await;

        let ws_name = format!("ws_{suffix}");
        let ws_path = format!("/tmp/test_{suffix}");
        create_test_workspace(&ws_name, &ws_path).await
    }

    /// Shorthand for [`init_management_test_stores`] + [`test_ws_named`] +
    /// [`TicketBuilder`].
    ///
    /// Creates an in-memory workspace (no DB insertion) with the given `path`
    /// and `name`, creates a ticket with `title` and starting `phase`, and
    /// returns `(workspace, ticket_id)`.
    async fn setup_ticket(
        ws_path: &str,
        ws_name: &str,
        title: &str,
        phase: TicketPhase,
    ) -> (crate::Workspace, String) {
        init_management_test_stores().await;
        let ws = test_ws_named(ws_path, ws_name);
        let ticket_id = make_ticket(board(), ws.clone(), title, phase).await;
        (ws, ticket_id)
    }

    /// Shorthand for [`setup_ticket`] when the workspace is not needed.
    ///
    /// Calls [`setup_ticket`] but discards the workspace, returning only the
    /// ticket ID. Use this when a test needs a ticket in a given phase but
    /// does not interact with the `Workspace` value.
    async fn setup_ticket_id(
        ws_path: &str,
        ws_name: &str,
        title: &str,
        phase: TicketPhase,
    ) -> String {
        let (_, id) = setup_ticket(ws_path, ws_name, title, phase).await;
        id
    }

    /// Verify the Buffer → Notify + drain sequence across two QaPassed tickets
    /// via `transition_ticket_to_done`: the first one buffers, the last one
    /// notifies and drains the buffer.
    #[tokio::test]
    async fn transition_ticket_to_done_buffer_and_notify() {
        let ws = setup_db_workspace("drains_buffer").await;

        // Two QaPassed tickets in the same workspace
        let first_id = make_ticket(board(), ws.clone(), "Ticket A", TicketPhase::QaPassed).await;
        let second_id = make_ticket(board(), ws, "Ticket B", TicketPhase::QaPassed).await;

        let ticket_a = expect_ticket(board(), &first_id).await;

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
        let ticket_b = expect_ticket(board(), &second_id).await;
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
            assert_eq!(t.phase, TicketPhase::Done, "Ticket {label} should be Done");
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

    // ── Transition does not pause workspace ─────────────────────────

    /// Verify that transitioning a ticket never pauses the workspace.
    /// The old auto-pause behavior was removed — workspace pausing is no
    /// longer part of any transition path.
    #[tokio::test]
    async fn transition_never_pauses_workspace() {
        struct Case {
            name: &'static str,
            ws_suffix: &'static str,
            ws_path: &'static str,
            source: TicketPhase,
            target: TicketPhase,
            policy: NotifyPolicy,
        }

        let cases = [
            Case {
                name: "Failed with Buffer",
                ws_suffix: "ws_no_pause_on_fail_test",
                ws_path: "/tmp/test_ws_no_pause_on_fail",
                source: DEFAULT_TICKET_PHASE,
                target: TicketPhase::Failed,
                policy: NotifyPolicy::Buffer,
            },
            Case {
                name: "non-failure with Notify",
                ws_suffix: "ws_no_pause_test",
                ws_path: "/tmp/test_ws_no_pause",
                source: TicketPhase::Backlog,
                target: TicketPhase::Analysis,
                policy: NotifyPolicy::Notify,
            },
        ];

        init_management_test_stores().await;
        for case in &cases {
            let ws = create_test_workspace(case.ws_suffix, case.ws_path).await;
            let ticket_id = make_ticket(board(), ws, "Test Ticket", TicketPhase::Backlog).await;
            let ticket = expect_ticket(board(), &ticket_id).await;

            transition_ticket(&ticket, case.source, case.target, case.policy, None)
                .await
                .expect("transition_ticket");

            let ws = crate::workspace::get_by_name(case.ws_suffix)
                .await
                .expect("get_by_name")
                .expect("workspace exists");
            assert!(
                !ws.paused,
                "case {}: workspace should NOT be paused after transition to {:?}",
                case.name, case.target,
            );
        }
    }

    // ── notify_ticket — smoke tests ──────────────────────────────────

    /// Smoke test: transitioning to Failed with Notify should not panic.
    /// Catches asset-loading or DB panics in the notification path.
    #[tokio::test]
    async fn notify_ticket_failed_transition_does_not_panic() {
        let ws = setup_db_workspace("failed_notify_test").await;

        let ticket_id = make_ticket(board(), ws, "Failed Notify Test", TicketPhase::Backlog).await;

        // Add a comment mimicking the actual failure path
        let _ = board()
            .add_comment(&ticket_id, SYSTEM_ROLE, "❌ Test failure detail")
            .await;

        let ticket = expect_ticket(board(), &ticket_id).await;

        // Transition to Failed with Notify — must not panic
        transition_ticket(
            &ticket,
            DEFAULT_TICKET_PHASE,
            TicketPhase::Failed,
            NotifyPolicy::Notify,
            None,
        )
        .await
        .expect("transition to Failed");
    }

    /// Smoke test: transitioning to a non-Failed phase with Notify should not
    /// panic. The warning template is only loaded for Failed transitions, so
    /// this path exercises that the conditional guard works.
    #[tokio::test]
    async fn notify_ticket_non_failed_transition_does_not_panic() {
        let ws = setup_db_workspace("non_failed_notify_test").await;

        let ticket_id =
            make_ticket(board(), ws, "Non-Failed Notify Test", TicketPhase::Backlog).await;

        let ticket = expect_ticket(board(), &ticket_id).await;

        // Transition from Backlog to Analysis with Notify — must not panic
        transition_ticket(
            &ticket,
            TicketPhase::Backlog,
            TicketPhase::Analysis,
            NotifyPolicy::Notify,
            None,
        )
        .await
        .expect("transition to Analysis");
    }

    // ── run_circuit_breaker — sanitation counting ──

    /// Verify that the sanitation circuit breaker counting logic works correctly.
    #[tokio::test]
    async fn sanitation_breaker_counts_failures() {
        let ticket_id = setup_ticket_id(
            "/tmp/test",
            "san_breaker_test",
            "Sanitation Breaker Test",
            TicketPhase::InSanitation,
        )
        .await;

        // Add 2 sanitation failure comments (below threshold of 3).
        for _ in 0..2 {
            let _ = board()
                .add_comment(
                    &ticket_id,
                    SYSTEM_ROLE,
                    &format!("{SANITATION_FAILED_PREFIX} — garbage files: 1"),
                )
                .await;
        }

        let ticket = expect_ticket(board(), &ticket_id).await;

        // Should NOT trip (2 <= 3)
        assert!(
            !run_circuit_breaker(
                &ticket,
                TicketPhase::InSanitation,
                SANITATION_CIRCUIT_BREAKER_THRESHOLD,
                count_sanitation_failures,
                sanitation_breaker_comment,
                "Sanitation",
            )
            .await,
            "Should NOT trip with 2 failures (threshold: 3)"
        );

        // Add a 3rd failure comment (should still not trip — runs the count_fn
        // which counts by the "Sanitation failed" substring, using fresh comments
        // from DB, not the stale in-memory ticket.comments).
        let _ = board()
            .add_comment(
                &ticket_id,
                SYSTEM_ROLE,
                &format!("{SANITATION_FAILED_PREFIX} — garbage files: 1"),
            )
            .await;

        // ... actually 3 <= 3 means the breaker does NOT trip yet.
        // The breaker trips when count > threshold, i.e., at 4 failures.
        // Add a 4th failure.
        let _ = board()
            .add_comment(
                &ticket_id,
                SYSTEM_ROLE,
                &format!("{SANITATION_FAILED_PREFIX} — garbage files: 1"),
            )
            .await;

        // Now with 4 failures, should trip (4 > 3).
        // Re-fetch ticket with fresh comments (run_circuit_breaker fetches comments
        // from DB internally, so we just need the ticket id).
        let ticket = expect_ticket(board(), &ticket_id).await;

        let tripped = run_circuit_breaker(
            &ticket,
            TicketPhase::InSanitation,
            SANITATION_CIRCUIT_BREAKER_THRESHOLD,
            count_sanitation_failures,
            sanitation_breaker_comment,
            "Sanitation",
        )
        .await;
        assert!(tripped, "Should trip with 4 failures (threshold: 3, 4 > 3)");

        // Verify the ticket is now Failed
        let status = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            status,
            TicketPhase::Failed,
            "Circuit breaker should transition to Failed"
        );
    }

    // ── Setup helpers ──────────────────────────────────────────────────────

    /// Shared helper: create a passing verdict (score >= REVIEW_QA_THRESHOLD).
    fn pass_verdict() -> crate::Verdict {
        crate::Verdict {
            score: REVIEW_QA_THRESHOLD,
            critique: Some("Good work.".into()),
            issues_detected: vec![],
        }
    }

    /// Shared helper: create a failing verdict (score < REVIEW_QA_THRESHOLD).
    fn fail_verdict() -> crate::Verdict {
        crate::Verdict {
            score: 3,
            critique: Some("Missing error handling.".into()),
            issues_detected: vec!["No timeout check".into()],
        }
    }

    /// Helper: a `ParallelVerdict` with no verdict (agent produced no response).
    fn no_verdict() -> ParallelVerdict {
        ParallelVerdict {
            response: String::new(),
            verdict: None,
        }
    }

    /// Helper: wrap a passing verdict with a response string (reviewer/QA flow).
    fn pass_result(response: &str) -> ParallelVerdict {
        ParallelVerdict {
            response: response.into(),
            verdict: Some(pass_verdict()),
        }
    }

    /// Helper: wrap a failing verdict with a response string (reviewer/QA flow).
    fn fail_result(response: &str) -> ParallelVerdict {
        ParallelVerdict {
            response: response.into(),
            verdict: Some(fail_verdict()),
        }
    }

    /// Helper: construct an analyst verdict with explicit score / critique / issues.
    fn analyst_verdict(
        response: &str,
        score: u8,
        critique: &str,
        issues: &[&str],
    ) -> ParallelVerdict {
        ParallelVerdict {
            response: response.into(),
            verdict: Some(crate::Verdict {
                score,
                critique: Some(critique.into()),
                issues_detected: issues.iter().map(|&s| s.into()).collect(),
            }),
        }
    }

    // ── process_verdict_results — verdict processing ─────────────────────

    /// Verify all verdict-processing outcomes:
    /// - All failed → Failed
    /// - Any failed → bounce-back to ReadyForDevelopment with pipeline reservation
    /// - All passed (Reviewer) → Reviewed
    /// - All passed (QA) → QaPassed
    #[tokio::test]
    async fn process_verdict_results_cases() {
        struct Case {
            name: &'static str,
            ws_suffix: &'static str,
            title: &'static str,
            phase: TicketPhase,
            results: Vec<ParallelVerdict>,
            vi: VerifierInfo,
            expected_status: TicketPhase,
            expected_pipeline_reservation: bool,
        }

        init_management_test_stores().await;

        let cases = vec![
            Case {
                name: "all failed -> Failed",
                ws_suffix: "vp_all_fail",
                title: "VP All Failed",
                phase: TicketPhase::InReview,
                results: vec![no_verdict(); 3],
                vi: REVIEWER_VI,
                expected_status: TicketPhase::Failed,
                expected_pipeline_reservation: false,
            },
            Case {
                name: "any failed -> bounce-back with pipeline reservation",
                ws_suffix: "vp_any_fail",
                title: "VP Any Failed",
                phase: TicketPhase::InReview,
                results: vec![
                    pass_result("Good."),
                    fail_result("Issues found."),
                    pass_result("Looks fine."),
                ],
                vi: REVIEWER_VI,
                expected_status: TicketPhase::ReadyForDevelopment,
                expected_pipeline_reservation: true,
            },
            Case {
                name: "all passed -> Reviewed",
                ws_suffix: "vp_all_pass",
                title: "VP All Pass",
                phase: TicketPhase::InReview,
                results: vec![
                    pass_result("Good."),
                    pass_result("Fine."),
                    pass_result("OK."),
                ],
                vi: REVIEWER_VI,
                expected_status: TicketPhase::Reviewed,
                expected_pipeline_reservation: false,
            },
            Case {
                name: "all passed (QA) -> QaPassed",
                ws_suffix: "vp_qa_pass",
                title: "VP QA Pass",
                phase: TicketPhase::InQa,
                results: vec![
                    pass_result("QA pass."),
                    pass_result("OK."),
                    pass_result("Good."),
                ],
                vi: QA_VI,
                expected_status: TicketPhase::QaPassed,
                expected_pipeline_reservation: false,
            },
        ];

        for case in &cases {
            let ws = test_ws_named("/tmp/test", case.ws_suffix);
            let ticket_id = make_ticket(board(), ws, case.title, case.phase).await;

            let ticket = expect_ticket(board(), &ticket_id).await;

            process_verdict_results(&ticket, &case.results, case.vi).await;

            let ticket = expect_ticket(board(), &ticket_id).await;
            assert_eq!(
                ticket.phase, case.expected_status,
                "case {}: expected status {:?}, got {:?}",
                case.name, case.expected_status, ticket.phase,
            );
            assert_eq!(
                ticket.pipeline_reservation, case.expected_pipeline_reservation,
                "case {}: expected pipeline_reservation={}, got {}",
                case.name, case.expected_pipeline_reservation, ticket.pipeline_reservation,
            );
        }
    }

    // ── run_circuit_breaker — general circuit breaker ────────

    /// Verify the circuit breaker trips at the threshold boundary:
    /// - `> CIRCUIT_BREAKER_COMMENT_THRESHOLD` comments → trips (ticket → Failed)
    /// - `= CIRCUIT_BREAKER_COMMENT_THRESHOLD` comments → does NOT trip
    #[tokio::test]
    async fn circuit_breaker_comment_boundary() {
        struct Case {
            name: &'static str,
            ws_suffix: &'static str,
            title: &'static str,
            comment_count: usize,
            expected_trip: bool,
            expected_status: TicketPhase,
        }

        init_management_test_stores().await;

        let cases = [
            Case {
                name: "> threshold trips",
                ws_suffix: "cb_thresh",
                title: "CB Threshold",
                comment_count: CIRCUIT_BREAKER_COMMENT_THRESHOLD + 1,
                expected_trip: true,
                expected_status: TicketPhase::Failed,
            },
            Case {
                name: "= threshold does not trip",
                ws_suffix: "cb_no_trip",
                title: "CB No Trip",
                comment_count: CIRCUIT_BREAKER_COMMENT_THRESHOLD,
                expected_trip: false,
                expected_status: TicketPhase::InReview,
            },
        ];

        for case in &cases {
            let ws = test_ws_named("/tmp/test", case.ws_suffix);
            let ticket_id = make_ticket(board(), ws, case.title, TicketPhase::InReview).await;

            for i in 0..case.comment_count {
                board()
                    .add_comment(&ticket_id, "user", &format!("Comment {i}"))
                    .await
                    .expect("add_comment");
            }

            let ticket = expect_ticket(board(), &ticket_id).await;

            let tripped = run_circuit_breaker(
                &ticket,
                TicketPhase::InReview,
                CIRCUIT_BREAKER_COMMENT_THRESHOLD,
                <[TicketComment]>::len,
                general_breaker_comment,
                "test",
            )
            .await;
            assert_eq!(
                tripped, case.expected_trip,
                "case {}: expected trip={}, got tripped={}",
                case.name, case.expected_trip, tripped,
            );

            let status = expect_ticket_phase(board(), &ticket_id).await;
            assert_eq!(
                status, case.expected_status,
                "case {}: expected status {:?}, got {:?}",
                case.name, case.expected_status, status,
            );
        }
    }

    /// The general circuit breaker's count_fn (`<[TicketComment]>::len`) does
    /// *not* filter out its own trip comment (unlike the diagnostics breaker
    /// which has explicit self-counting prevention). Cascade prevention relies
    /// on the phase guard: after tripping, the ticket transitions to Failed,
    /// and `guard_phase_and_circuit_breaker` rejects the next cycle via
    /// phase mismatch before the breaker is called.
    #[tokio::test]
    async fn circuit_breaker_guard_prevents_retrip() {
        let ticket_id =
            setup_ticket_id("/tmp/test", "cb_guard", "CB Guard", TicketPhase::InReview).await;

        for i in 0..=CIRCUIT_BREAKER_COMMENT_THRESHOLD {
            board()
                .add_comment(&ticket_id, "user", &format!("Comment {i}"))
                .await
                .expect("add_comment");
        }

        // Trip the breaker
        let ticket = expect_ticket(board(), &ticket_id).await;

        let tripped = run_circuit_breaker(
            &ticket,
            TicketPhase::InReview,
            CIRCUIT_BREAKER_COMMENT_THRESHOLD,
            <[TicketComment]>::len,
            general_breaker_comment,
            "test",
        )
        .await;
        assert!(tripped, "breaker should trip");

        let status = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            status,
            TicketPhase::Failed,
            "ticket should be Failed after trip"
        );

        // The trip comment contains the circuit breaker marker for consistency
        // with other circuit breaker trip messages.
        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        let has_marker = comments
            .iter()
            .any(|c| c.content.to_lowercase().contains("circuit breaker"));
        assert!(
            has_marker,
            "trip comment must contain circuit breaker marker"
        );

        // Phase guard prevents a second trip: the ticket is now Failed, not
        // InReview, so is_ticket_in_phase rejects it before the breaker runs.
        let ticket = expect_ticket(board(), &ticket_id).await;

        let guarded = guard_phase_and_circuit_breaker(&ticket, TicketPhase::InReview, "test").await;
        assert!(
            !guarded,
            "phase guard must reject re-trip (ticket is now Failed)"
        );

        let status = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(status, TicketPhase::Failed, "ticket must remain Failed");
    }

    // ── handle_analyst_verdicts — analyst scoring and transitions ─────────

    /// Verify handle_analyst_verdicts across all outcomes:
    /// - All analysts pass → Planning with "All LGTM" summary
    /// - Partial fail → Planning with "blockers" summary
    /// - No verdicts → Planning with "no analysis" summary
    #[tokio::test]
    async fn handle_analyst_verdicts_cases() {
        struct Case {
            name: &'static str,
            ws_suffix: &'static str,
            title: &'static str,
            results: Vec<ParallelVerdict>,
            expected_comment_substring: &'static str,
        }

        init_management_test_stores().await;

        let cases = vec![
            Case {
                name: "all pass -> Planning with LGTM",
                ws_suffix: "an_all_pass",
                title: "Analyst All Pass",
                results: vec![
                    analyst_verdict("Analysis A", 10, "Great analysis.", &[]),
                    analyst_verdict("Analysis B", 9, "Solid work.", &[]),
                    analyst_verdict("Analysis C", 8, "Good analysis.", &[]),
                ],
                expected_comment_substring: "All LGTM",
            },
            Case {
                name: "partial fail -> Planning with blockers",
                ws_suffix: "an_partial",
                title: "Analyst Partial Fail",
                results: vec![
                    analyst_verdict("Analysis A", 10, "Great.", &[]),
                    analyst_verdict("Analysis B", 3, "Poor analysis.", &["Missing data"]),
                    analyst_verdict("Analysis C", 8, "Decent.", &["Minor issue"]),
                ],
                expected_comment_substring: "blockers",
            },
            Case {
                name: "no verdicts -> Planning with no analysis",
                ws_suffix: "an_no_v",
                title: "Analyst No Verdicts",
                results: vec![no_verdict(); 3],
                expected_comment_substring: "no analysis",
            },
        ];

        for case in &cases {
            let ws = test_ws_named("/tmp/test", case.ws_suffix);
            let ticket_id = make_ticket(board(), ws, case.title, TicketPhase::Analysis).await;

            let ticket = expect_ticket(board(), &ticket_id).await;

            handle_analyst_verdicts(&ticket, &case.results).await;

            let status = expect_ticket_phase(board(), &ticket_id).await;
            assert_eq!(
                status,
                TicketPhase::Planning,
                "case {}: expected Planning, got {:?}",
                case.name,
                status,
            );

            // Verify comment structure: 3 per-analyst + 1 system summary = 4
            let comments = board()
                .get_comments(&ticket_id)
                .await
                .expect("get_comments");
            assert_eq!(
                comments.len(),
                4,
                "case {}: expected 4 comments (3 per-analyst + 1 system summary), got {}",
                case.name,
                comments.len(),
            );

            let system = comments.iter().find(|c| c.role == SYSTEM_ROLE);
            assert!(
                system.is_some(),
                "case {}: system summary comment should exist",
                case.name,
            );
            assert!(
                system
                    .unwrap()
                    .content
                    .contains(case.expected_comment_substring),
                "case {}: system comment should contain {:?}, got: {}",
                case.name,
                case.expected_comment_substring,
                system.unwrap().content,
            );
        }
    }

    // ── handle_qa_passed — QA → Done path ───────────────────────────────

    /// handle_qa_passed first checks whether git is available and whether the
    /// workspace path is a git repo. In test environments git may exist, but
    /// the workspace path is deliberately not a git repo, so the function
    /// falls through to finalize_ticket_from_phase → transition_ticket_to_done.
    /// This test validates the graceful non-git fallback path.
    #[tokio::test]
    async fn handle_qa_passed_no_git_to_done() {
        // Use a path that cannot be a git repo regardless of the test
        // environment's current working directory.
        let (ws, ticket_id) = setup_ticket(
            "/nonexistent/mahbot-test-qa-no-git",
            "qa_no_git",
            "QA No Git",
            TicketPhase::QaPassed,
        )
        .await;

        let ticket = expect_ticket(board(), &ticket_id).await;

        handle_qa_passed(ticket, ws).await;

        let status = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            status,
            TicketPhase::Done,
            "QA passed should eventually transition to Done"
        );
    }

    /// handle_qa_passed with untracked files present should claim the ticket
    /// to InSanitation and dispatch a sanitation agent. Creates a real git repo
    /// with an untracked file to exercise the full claim path.
    #[tokio::test]
    async fn handle_qa_passed_untracked_files_to_insanitation() {
        // Skip if git is not installed — the test cannot create a repo.
        if !crate::diff_parse::git_is_installed().await {
            eprintln!("git not installed — skipping git-dependent test");
            return;
        }

        // Create a temp directory and init a git repo
        let (_dir, repo_path) = crate::util::test::init_temp_repo();

        // Create an untracked file
        std::fs::write(repo_path.join("untracked.txt"), b"garbage").expect("write untracked file");

        let (ws, ticket_id) = setup_ticket(
            repo_path.to_str().unwrap(),
            "qa_untracked",
            "QA Untracked",
            TicketPhase::QaPassed,
        )
        .await;

        let ticket = expect_ticket(board(), &ticket_id).await;

        handle_qa_passed(ticket, ws).await;

        let status = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            status,
            TicketPhase::InSanitation,
            "QA passed with untracked files should transition to InSanitation"
        );

        // Verify assigned_to is set to the sanitation session key
        let ticket = expect_ticket(board(), &ticket_id).await;
        let expected_key =
            crate::session::ticket_session_key(&ticket_id, crate::Role::Sanitation.as_str());
        assert_eq!(
            ticket.assigned_to.as_deref(),
            Some(expected_key.as_str()),
            "assigned_to should be set to sanitation session key"
        );
    }
}
