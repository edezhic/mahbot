//! Board poller — picks up tickets from the board and dispatches agents.
//!
//! Poll phases — dispatches agents based on ticket phase:
//! - Backlog → spawn Analyst agents (3 parallel; escalates to 5 on unanimous blockers)
//! - ReadyForDevelopment → spawn Engineer agent
//! - InDiagnostics → dispatch diagnostics runner (shell commands)
//! - DiagnosticsDone → spawn Reviewer agents (calibrated dynamic count)
//! - Reviewed → spawn QA agent (single tester)
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

use chrono::Duration as ChronoDuration;
use futures_util::FutureExt;
use futures_util::future::join_all;

use crate::agent::{RETRY_EXHAUSTION_MARKER, run_agent};
use crate::board::{BOARD, BoardStore, PipelineCheck, Ticket, TicketComment, TicketPhase};
use crate::git_commands::{
    has_unstaged_changes, list_new_or_untracked_files, parse_new_files_from_porcelain,
    run_git_add_all, run_git_diff_stats, run_git_head, run_git_status, run_git_write_tree,
};
use crate::jobs::ResumableStage;
use crate::message_router;
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::role::{DIAGNOSTICS_ROLE, SANITATION_ROLE, SYSTEM_ROLE};
use crate::session::{manager_agent_id, ticket_agent_id};
use crate::ticket_buffer;
use crate::tools::shell::{ShellMode, ShellTool};
use crate::turso::TxGuard;
use crate::util::panic_message;
use crate::workspace::spawn_workspace_discovery;

use crate::{Agent, DiagnosticsCommands, Role, Workspace, WorkspaceStatus};

/// Default number of parallel analyst agents per round. Reviewers use a
/// calibrated dynamic count (see [`crate::joint_verdict::review_agent_count`]);
/// QA runs a single tester (see [`QA_PARALLEL_AGENT_COUNT`]).
pub(crate) const DEFAULT_PARALLEL_AGENT_COUNT: usize = 3;

/// QA runs exactly one tester per round — reviewers already verify the change
/// in depth. May be revisited if a single tester proves insufficient.
const QA_PARALLEL_AGENT_COUNT: usize = 1;

/// Minimum acceptable verification score (0-10) for analyst verdicts.
pub(crate) const ANALYST_PASS_THRESHOLD: u8 = 7;

/// Minimum acceptable verification score (0-10) for review and QA phases.
const REVIEW_QA_THRESHOLD: u8 = 9;
/// Neutral reason for a phase-gate bail (transients are not misattributed).
const PHASE_GATE_BAIL_REASON: &str = "ticket not in expected phase";

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
/// Uses [`BoardStore::set_assigned_to_no_cancel`] with `None` — the
/// non-cancelling assignment variant. All call sites are post-agent so there
/// is no agent to cancel.
///
/// ## TOCTOU race
///
/// A concurrent claim may set a new assignee between a phase check and this
/// clear. That's very low probability and the same race is accepted in
/// [`record_sanitation_failure`].
async fn clear_assigned_to_no_cancel(ticket_id: &str, context: &str) {
    if let Err(e) = board().set_assigned_to_no_cancel(ticket_id, None).await {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitBreakerKind {
    /// Sanitation-failure breaker: trips when cumulative sanitation failures
    /// exceed 3.
    Sanitation,
    /// Diagnostics-failure breaker: trips when cumulative diagnostics failures
    /// exceed 4.
    Diagnostics,
}

impl CircuitBreakerKind {
    /// Returns the maximum tolerated failure count for this breaker variant.
    /// The breaker trips when [`should_trip`](CircuitBreakerKind::should_trip) returns
    /// `Some` (the count exceeds this maximum).
    const fn max_count(self) -> usize {
        match self {
            Self::Sanitation => 3,
            Self::Diagnostics => 4,
        }
    }

    /// Determine whether this breaker variant should trip.
    ///
    /// Counts failures matching this variant's criteria. If the count exceeds
    /// the variant's `max_count`, returns `Some((count, max_count))`. Returns
    /// `None` if the breaker should not trip (count ≤ max_count).
    fn should_trip(self, comments: &[TicketComment]) -> Option<(usize, usize)> {
        let max_count = self.max_count();
        let count = match self {
            Self::Sanitation => count_matching_comments(
                comments,
                SANITATION_ROLE,
                &load_prompt("pipeline/sanitation_failed.md"),
            ),
            Self::Diagnostics => count_matching_comments(
                comments,
                DIAGNOSTICS_ROLE,
                &load_prompt("pipeline/diagnostics_failed.md"),
            ),
        };

        if count <= max_count {
            None
        } else {
            Some((count, max_count))
        }
    }

    /// Format the trip message for this breaker variant.
    fn trip_message(self, count: usize, max_count: usize) -> String {
        match self {
            Self::Sanitation => format!(
                "❌ Sanitation circuit breaker tripped after {count} cumulative failures. \
                 (max: {max_count})",
            ),
            Self::Diagnostics => {
                format!("❌ Circuit breaker: {count} prior diagnostic failures. Failing ticket.")
            }
        }
    }
}

/// Count ticket comments matching a specific role and marker substring.
fn count_matching_comments(comments: &[TicketComment], role: &str, marker: &str) -> usize {
    comments
        .iter()
        .filter(|c| c.role == role && c.content.contains(marker))
        .count()
}

/// Returns `true` if the ticket is in the expected phase (safe to proceed).
/// Returns `false` otherwise — the ticket may have been moved externally,
/// not found in the database, or a database error occurred.
/// The caller should abort its current work on this ticket.
/// Canonical fail-closed contract for the two wrappers below.
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

/// Bails when not in expected phase: completes stage job via [`complete_ticket_stage_job`],
/// does NOT touch `assigned_to` (parallel rounds keep `assigned_to=NULL` owned by
/// [`run_parallel_agents`]). See [`is_ticket_in_phase`] for the fail-closed contract
/// (moved externally / missing row / DB error → bail). Sibling [`phase_changed_and_clear_assignment`]
/// clears `assigned_to` instead for single-slot re-dispatch rounds.
///
/// Three guard sites: analysis round ([`maybe_escalate_analysis`], [`finalize_analysis_round`]), verifier round ([`dispatch_verifiers`]), resume path ([`resume_analysis_round`], [`resume_verifier_round`], [`resume_stage_round`]).
#[must_use]
async fn complete_job_and_bail_if_phase_moved(
    ticket_id: &str,
    expected: TicketPhase,
    job_id: &str,
) -> bool {
    if !is_ticket_in_phase(ticket_id, expected).await {
        complete_ticket_stage_job(job_id).await;
        return true;
    }
    false
}

/// Bails when not in expected phase: clears `assigned_to` via [`clear_assigned_to_no_cancel`]
/// to unblock re-dispatch for single-agent rounds. See [`is_ticket_in_phase`] for the
/// fail-closed contract (moved externally / missing row / DB error → bail). Sibling
/// [`complete_job_and_bail_if_phase_moved`] terminalizes the stage job instead and does
/// not touch `assigned_to`.
///
/// Call sites ([`finalize_engineer_round`], [`finalize_sanitation_round`], [`dispatch_diagnostics`]) have previously set `assigned_to` via [`claim_ticket_in_workspace`] or [`claim_diagnostics`], so clearing it here prevents a stale assignment from blocking re-dispatch.
#[must_use]
async fn phase_changed_and_clear_assignment(ticket_id: &str, expected: TicketPhase) -> bool {
    if !is_ticket_in_phase(ticket_id, expected).await {
        let label = format!("ticket left {expected:?}");
        clear_assigned_to_no_cancel(ticket_id, &label).await;
        return true;
    }
    false
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

/// The transition context for [`comment_and_transition`] and [`with_comment_and_transition`].
///
/// Encapsulates the ticket, source/target phases, notify policy, and log label
/// — everything needed for a comment+transition operation. Comment text is
/// passed as separate parameters to each function since
/// [`comment_and_transition`] always requires a comment (it is no longer
/// optional), and [`with_comment_and_transition`] delegates all comment
/// writing to a closure.
///
#[derive(Debug)]
struct TransitionCtx<'t, 'l> {
    ticket: &'t Ticket,
    source: TicketPhase,
    target: TicketPhase,
    notify: NotifyPolicy,
    log_label: &'l str,
    /// True when this Failed transition was a circuit-breaker drain — the
    /// failure notification then describes the drain instead of the
    /// auto-pause. Read only on Failed transitions; the readers are the
    /// circuit-breaker trip (`true`) and the technical-failure sites —
    /// dispatch panic, engineer failure, verifier all-failed (`false`).
    /// Every Failed site must set it explicitly so the notification renders
    /// the sentence matching the mechanism that ran.
    breaker_trip: bool,
}

impl<'t, 'l> TransitionCtx<'t, 'l> {
    fn new(
        ticket: &'t Ticket,
        source: TicketPhase,
        target: TicketPhase,
        notify: NotifyPolicy,
        log_label: &'l str,
    ) -> Self {
        Self {
            ticket,
            source,
            target,
            notify,
            log_label,
            breaker_trip: false,
        }
    }
    fn notifying(
        ticket: &'t Ticket,
        source: TicketPhase,
        target: TicketPhase,
        log_label: &'l str,
    ) -> Self {
        Self::new(ticket, source, target, NotifyPolicy::Notify, log_label)
    }
    fn buffered(
        ticket: &'t Ticket,
        source: TicketPhase,
        target: TicketPhase,
        log_label: &'l str,
    ) -> Self {
        Self::new(ticket, source, target, NotifyPolicy::Buffer, log_label)
    }
    fn with_breaker(mut self, breaker_trip: bool) -> Self {
        self.breaker_trip = breaker_trip;
        self
    }
}

/// Result of a stage-finalization comment+transition.
///
/// The phase-guard miss is a first-class, expected outcome: a ticket that was
/// moved externally (cancelled, superseded, or otherwise transitioned) while
/// the stage was finishing is a clean skip, not a failure. Only genuine write
/// failures keep the warning path.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FinalizeOutcome {
    /// Guard applied — comment + transition committed; notification dispatched.
    Applied,
    /// Guard missed — the ticket was moved externally while the stage was
    /// finishing. Nothing was written; a silent, expected skip (at most a
    /// low-level log entry).
    Moved,
    /// Genuine write failure — warning already logged, assignment cleared.
    Failed,
}

/// Unified helper for combining comment writes + phase transition + notification.
///
/// Wraps [`crate::turso::with_tx_outcome`] for the comment-writing closure and
/// phase transition, then dispatches a notification. The closure is
/// responsible for writing all per-agent/system comments to the database.
///
/// `pipeline_reservation` is automatically derived from the target phase:
/// `Some(true)` when transitioning to [`TicketPhase::ReadyForDevelopment`]
/// (bounce-back transitions get priority re-dispatch over fresh tickets),
/// `None` for all other transitions.
///
/// Returns a [`FinalizeOutcome`]:
/// - [`Applied`](FinalizeOutcome::Applied) — guard applied, finalization
///   committed, notification dispatched.
/// - [`Moved`](FinalizeOutcome::Moved) — the CAS phase guard missed: the
///   ticket was moved externally while the stage was finishing. The
///   transaction is rolled back silently (nothing written, no warning) and
///   `assigned_to` is **not** cleared — the external mover already handled
///   it.
/// - [`Failed`](FinalizeOutcome::Failed) — genuine write failure: a warning
///   is logged and `assigned_to` is cleared for re-dispatch.
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
async fn with_comment_and_transition<F>(
    ctx: TransitionCtx<'_, '_>,
    write_comments: F,
) -> FinalizeOutcome
where
    F: AsyncFnOnce(&TxGuard<'_>) -> anyhow::Result<()>,
{
    let pipeline_reservation = (ctx.target == TicketPhase::ReadyForDevelopment).then_some(true);

    let outcome = match crate::turso::with_tx_outcome(
        &board().conn,
        &ctx.ticket.id,
        ctx.log_label,
        async move |tx| {
            write_comments(tx).await?;
            // The transition's `Ok(false)` (guard miss) is the closure's
            // `Ok(false)`: the whole transaction is rolled back silently.
            BoardStore::transition_to_tx(
                tx,
                &ctx.ticket.id,
                Some(ctx.source),
                ctx.target,
                pipeline_reservation,
            )
            .await
        },
    )
    .await
    {
        Ok(true) => FinalizeOutcome::Applied,
        Ok(false) => {
            debug!(
                ticket = %ctx.ticket.id,
                "{}: ticket moved externally while in {} — finalization skipped (nothing written)",
                ctx.log_label, ctx.source,
            );
            FinalizeOutcome::Moved
        }
        Err(e) => {
            // Use phase values directly (not strings) so phase names can't drift.
            warn!(
                ticket = %ctx.ticket.id,
                error = %e,
                "{}: transition to {} failed — ticket stuck in {}",
                ctx.log_label, ctx.target, ctx.source,
            );
            // Clear assigned_to so the ticket can be re-dispatched on the next poll
            // cycle. All call sites set assigned_to before reaching this function, so
            // the field is always populated when this runs. Only on genuine write
            // failures — a guard-missed ticket was already handled by the mover.
            clear_assigned_to_no_cancel(&ctx.ticket.id, ctx.log_label).await;
            FinalizeOutcome::Failed
        }
    };

    // Notifications fire only when the guard applied — a moved ticket was
    // already announced by the external mover, and a genuine failure has no
    // transition to announce.
    if matches!(outcome, FinalizeOutcome::Applied) {
        match ctx.notify {
            NotifyPolicy::Notify => {
                notify_ticket(ctx.ticket, ctx.source, ctx.target, ctx.breaker_trip).await;
            }
            NotifyPolicy::Buffer => {
                ticket_buffer::push(
                    &ctx.ticket.workspace_name,
                    &ctx.ticket.id,
                    ctx.source,
                    ctx.target,
                    ticket_buffer::TransitionOrigin::Pipeline,
                );
            }
        }
    }
    outcome
}

/// Write a comment to a ticket, then transition it to a new phase.
///
/// Delegates to [`with_comment_and_transition`]; see that function for
/// transaction semantics, notification dispatch, and return-value conventions.
///
/// Both `role` and `text` are required — the compiler guarantees a comment
/// is always written.
#[must_use]
async fn comment_and_transition(
    ctx: TransitionCtx<'_, '_>,
    role: &str,
    text: &str,
) -> FinalizeOutcome {
    let ticket = ctx.ticket;

    with_comment_and_transition(ctx, async |tx| {
        BoardStore::add_comment_tx(tx, &ticket.id, role, text).await?;
        Ok(())
    })
    .await
}

/// Shared finalizer for post-agent comment-and-transition sites.
///
/// Runs [`comment_and_transition`] and, on [`Applied`](FinalizeOutcome::Applied),
/// logs `message` with the ticket ID and the `target` phase field from `ctx`.
/// On a guard miss ([`Moved`](FinalizeOutcome::Moved)) nothing is emitted —
/// the ticket was moved externally and the skip is expected. On genuine
/// failure ([`Failed`](FinalizeOutcome::Failed)) `comment_and_transition`
/// already logged the warning, so nothing more is emitted here. Must be the
/// caller's final step.
async fn comment_and_transition_or_bail(
    ctx: TransitionCtx<'_, '_>,
    role: &str,
    text: &str,
    message: &str,
) {
    let ticket_id = &ctx.ticket.id;
    let target = ctx.target;
    if !matches!(
        comment_and_transition(ctx, role, text).await,
        FinalizeOutcome::Applied
    ) {
        return;
    }
    info!(ticket = %ticket_id, target = %target, "{message}");
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

/// Wording shared by the failure-comment pause note and the Manager
/// notification: the pause gate blocks automatic Backlog→Analysis and
/// ReadyForDevelopment→InDevelopment claims (see [`run_claim_pipeline`]),
/// while later-phase work (review, QA, in-flight runs) continues.
/// Single-sourced so the two sites cannot drift.
fn paused_workspace_sentence() -> &'static str {
    "new analysis and development claims are blocked until the workspace is resumed"
}

/// Pause the workspace after a technical/agent failure so queued development
/// tickets are not claimed and don't cascade through the pipeline failing
/// identically one after another.
///
/// Returns a notice string to append to the ticket's failure comment, or an
/// empty string when the workspace was not paused (already paused, or the
/// service is shutting down — a shutdown-interrupted run must never pause).
///
/// # Orphaned-pause corner
///
/// Runs before the caller's Failed transition: if that transition fails (DB
/// error), the workspace stays paused with no failure comment or
/// notification. Rare and recoverable via the normal GUI unpause.
///
/// # Circuit-breaker exclusion
///
/// This is intentionally NOT called from [`try_trip_circuit_breaker`] — the
/// comment-based breaker trips keep their existing drain-to-Planning handling
/// and must not pause. The engineer hard-failure bounce trip is the one
/// exception: it pauses at the engineer failure site, before
/// the Failed transition, because the budget-exhausting failure is a
/// technical failure like any other engineer failure. Callers are the
/// technical-failure sites: dispatch panic, engineer agent failure (including
/// its bounce-budget trip), all verifier agents failing, and the GUI cancel
/// of an in-flight agent run.
///
/// # Scope
///
/// Sanitation/diagnostics agent failures never reach this helper: they retry,
/// then trip their circuit breaker (which must not pause). Analyst failures
/// never fail the ticket (analysts always advance to Planning), and mixed
/// verifier rounds (some failures + a sub-threshold verdict) bounce to
/// ReadyForDevelopment without pausing. Analyze-tool sub-agent failures (parallel
/// analysts under [`AnalyzeTool`](crate::tools::AnalyzeTool)) likewise never
/// reach this helper — they are tool calls inside a caller's run, not
/// ticket-level failures. The pause gate blocks automatic Backlog→Analysis
/// and ReadyForDevelopment→InDevelopment claims, so a pause stops further
/// pickup of queued Backlog/RFD tickets on the poll cycles that follow (a
/// claim already in flight within the current cycle may still land);
/// tickets already past analysis continue through their later phases.
pub(crate) async fn pause_workspace_on_failure(ticket: &Ticket, reason: &str) -> String {
    if crate::shutdown::aborting() {
        // Shutdown AND the graceful drain are excluded: a drain-cut round is
        // not a failure, so no auto-pause fires at exit time (the job stays
        // launched for boot resume).
        return String::new();
    }
    let Some(ws) = resolve_ticket_workspace(ticket, "auto-pause skipped").await else {
        return String::new();
    };
    if ws.paused {
        return String::new();
    }
    match crate::workspace::store().set_paused(&ws.name, true).await {
        Ok(()) => {
            info!(
                ticket = %ticket.id,
                workspace = %ws.name,
                reason,
                "Workspace auto-paused after failure"
            );
            format!(
                "\n\n⚠️ Workspace paused: {reason} — {}.",
                paused_workspace_sentence()
            )
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                workspace = %ws.name,
                reason,
                error = %e,
                "Failed to auto-pause workspace after technical failure",
            );
            String::new()
        }
    }
}

/// Fetch the last ticket comment (any role) as the failure details for a
/// Manager notification. The transition closure writes the failure comment
/// LAST (after any circuit-breaker trip comment), so it is the last comment
/// and carries the concrete error. Falls back to a generic message when no
/// comment exists or the read fails.
async fn last_comment_as_failure_details(ticket_id: &str) -> String {
    match board().get_comments(ticket_id).await {
        Ok(comments) => comments.last().map_or_else(
            || "(unknown failure reason)".to_string(),
            |c| c.content.clone(),
        ),
        Err(_) => "(unknown failure reason)".to_string(),
    }
}

/// Enqueue a notification for the Manager about a ticket transition.
///
/// Renders a template with the ticket ID, title, target phase, transition log,
/// and the workspace's buffered non-critical transitions (see "Side effects"
/// below), then routes the result through the message router.
///
/// This function does NOT pause the workspace — technical-failure pause
/// happens at the trigger sites (dispatch panic, engineer failure, verifier
/// all-failed, GUI cancel) before the transition that fires this notification.
/// The Manager handles failed tickets via the triage prompt.
///
/// # Side effects
///
/// Drains the [`ticket_buffer`] for this workspace before rendering the
/// notification template. The drained entries are injected via the
/// `{{ticket_updates}}` placeholder. See the inline comment at the drain call
/// site for the data-loss guard (drain happens only after workspace lookup
/// succeeds so that buffered entries survive a temporary lookup failure).
///
/// # Invariant: failure comment written before the notification
///
/// When `target_phase` is `Failed`, or the transition is the engineer
/// hard-failure bounce (source `InDevelopment` → `ReadyForDevelopment` — the
/// only Notify transition to RFD, enforced structurally at the gate below),
/// the failure details are read from the database (last comment, any role)
/// instead of being passed as a parameter. The caller MUST ensure the failure
/// comment has already been written to the DB before calling this function
/// (the transition closure runs first in [`with_comment_and_transition`], so
/// this invariant holds for all call paths).
/// The agent ID (`manager_{ws_name}`) is intentionally shared between
/// user-facing Manager chat (main.rs) and notification agents — the same Manager
/// must see both notification context and user conversation history in a unified
/// session. Do NOT change this ID or add `manager_` to `TRANSIENT_AGENT_ID_PREFIXES`
/// — it would either break context continuity or nuke user conversation history.
async fn notify_ticket(
    ticket: &Ticket,
    source: TicketPhase,
    target_phase: TicketPhase,
    breaker_trip: bool,
) {
    let Some(ws) = resolve_ticket_workspace(ticket, "skipping notification").await else {
        error!(
            ticket = %ticket.id,
            workspace_name = %ticket.workspace_name,
            "Workspace resolution failed — notification skipped"
        );
        return;
    };

    // Immediate notifications are only ever fired by pipeline code paths
    // (never by user actions — those are buffered), so the origin marker is
    // the constant Pipeline variant, rendered inline for symmetry with
    // drained buffer entries. The enum's Display impl is the single source
    // of truth for the user-visible wording.
    let transition_log = format!(
        "[{}] {}: {} → {} ({})",
        ticket.reporter,
        ticket.id,
        source.as_ref(),
        target_phase.as_ref(),
        ticket_buffer::TransitionOrigin::Pipeline
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
        &load_prompt("pipeline/notification.md"),
        &[
            ("{{ticket_id}}", &ticket.id),
            ("{{ticket_title}}", &ticket.title),
            ("{{ticket_phase}}", target_phase.as_ref()),
            ("{{transition_log}}", &transition_log),
            ("{{ticket_updates}}", &drained),
        ],
    );

    // The engineer hard-failure bounce is the ONLY Notify transition
    // InDevelopment → ReadyForDevelopment (verifier/sanitation bounces buffer,
    // manual moves are user actions), so gating on the source phase
    // structurally excludes any future RFD+Notify transition from silently
    // rendering the engineer-bounce template.
    let engineer_bounce =
        source == TicketPhase::InDevelopment && target_phase == TicketPhase::ReadyForDevelopment;
    if target_phase == TicketPhase::Failed || engineer_bounce {
        // The transition closure wrote the failure comment LAST (after any
        // circuit-breaker trip comment), so the last comment carries the
        // concrete error.
        let failure_details = last_comment_as_failure_details(&ticket.id).await;

        // The workspace_status sentence is chosen from the failure MECHANISM,
        // not from `ws.paused` alone: a circuit-breaker drain moved tickets
        // back to Planning (regardless of pause state), while a technical
        // failure auto-pauses when `ws.paused` is set. When a technical
        // failure could not pause (set_paused error), neither claim is made.
        let workspace_status = if breaker_trip {
            "Beware that all the other tickets have been moved back from Ready for Dev \
             to Planning."
                .to_string()
        } else if ws.paused {
            format!("The workspace is paused — {}.", paused_workspace_sentence())
        } else {
            "The workspace was not paused — remaining queued tickets may still be claimed."
                .to_string()
        };

        // The engineer hard-failure bounce gets a dedicated template: the
        // ticket was NOT failed, so the generic failed-ticket triage wording
        // (supersede/revert guidance) would misdirect the Manager.
        let template = if engineer_bounce {
            "pipeline/engineer_bounce_notification.md"
        } else {
            "pipeline/failure_notification.md"
        };
        let warning = substitute(
            &load_prompt(template),
            &[
                ("{{failure_details}}", &failure_details),
                ("{{workspace_status}}", &workspace_status),
            ],
        );
        message.push_str("\n\n");
        message.push_str(&warning);
    }

    // Route through the agent-ID message router.
    // The consumer loop resolves the workspace and runs the agent.
    let agent_id = manager_agent_id(&ws.name);
    message_router::route(
        &agent_id,
        message_router::AgentJob {
            content: message,
            workspace_name: ws.name,
            user_name: String::new(),
            channel: String::new(),
            kind: message_router::JobKind::TicketNotify,
            role: crate::Role::Manager,
            reply_target: None,
            pending_job_id: None,
        },
    );
}

#[expect(clippy::too_many_lines)]
pub async fn run_management() {
    // Boot recovery scan: first statement of run_management,
    // BEFORE reset_inflight_tickets. Replays pending envelopes, materializes
    // the resumed-ticket exclusion set, resets everything else, and returns
    // the ticket_stage jobs selected for resume.
    let resumable = match crate::jobs::recover_from_restart().await {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, "Boot recovery scan failed — proceeding with plain reset");
            if let Some(board) = BOARD.get() {
                let _ = board.reset_inflight_tickets(&[]).await;
            }
            Vec::new()
        }
    };

    // Boot recovery for workspaces: discovery leaves no job rows, so a
    // workspace still in 'analyzing' at startup means a crashed/panicked
    // mid-discovery run. Reclassify to 'pending' so the pickup step retries
    // it (or waits for the provider) — fixes the historical "stranded in
    // analyzing forever" behavior. Must precede the first poll cycle.
    if let Err(e) = crate::workspace::store()
        .reclassify_analyzing_to_pending()
        .await
    {
        warn!(error = %e, "Boot recovery: failed to reclassify stranded analyzing workspaces");
    }

    // Resume the selected rounds (silent background resume — no Manager
    // notifications; results deliver via normal paths).
    for stage in resumable {
        // Or-pattern: every variant carries job_id + workspace_name.
        let (job_id, workspace_name) = match &stage {
            ResumableStage::TicketStage {
                job_id,
                workspace_name,
                ..
            }
            | ResumableStage::Research {
                job_id,
                workspace_name,
                ..
            }
            | ResumableStage::Analyze {
                job_id,
                workspace_name,
                ..
            }
            | ResumableStage::ResearchCleanup {
                job_id,
                workspace_name,
            } => (job_id, workspace_name),
        };
        let Ok(Some(workspace)) = crate::workspace::store().get_by_name(workspace_name).await
        else {
            warn!(
                job = %job_id,
                workspace = %workspace_name,
                "Resume workspace unresolvable — deleting job row",
            );
            // Design: "Unresolvable workspace → delete job row" — a done row
            // would linger without a workspace to drive it (envelope kinds
            // have no ticket to reset; ticket_stage kinds are re-covered by
            // the next boot's reset once no job row protects them).
            // Any path that removes a research_cleanup row must release the
            // run folder in the same operation (the row is the folder-hold;
            // no sweep is a backstop anymore). A cleanup can never run
            // without its workspace, so the folder would be orphaned — release
            // it before terminalizing the row.
            if matches!(&stage, ResumableStage::ResearchCleanup { .. }) {
                crate::research_cleanup::release_run_folder(job_id).await;
            }
            let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
            continue;
        };
        match stage {
            // research/analyze jobs carry no ticket — re-dispatch the orchestrator.
            ResumableStage::Research {
                job_id,
                capped: false,
                ..
            } => {
                info!(job = %job_id, "Resuming research run at boot");
                let ws = workspace.clone();
                tokio::spawn(async move {
                    crate::tools::research::resume_research_run(&job_id, &ws).await;
                });
            }
            // Over-cap research: deliver the partial report from the last
            // checkpoint (the envelope is the caller's only result path).
            ResumableStage::Research {
                job_id,
                capped: true,
                ..
            } => {
                info!(
                    job = %job_id,
                    "Delivering research partial report (boot re-dispatch cap exceeded)",
                );
                let ws = workspace.clone();
                tokio::spawn(async move {
                    crate::tools::research::research_capped_partial_report(&job_id, &ws).await;
                });
            }
            ResumableStage::Analyze {
                job_id,
                capped: false,
                ..
            } => {
                info!(job = %job_id, "Resuming analyze round at boot");
                let ws = workspace.clone();
                tokio::spawn(async move {
                    crate::tools::analyze::resume_analyze_round(&job_id, &ws).await;
                });
            }
            // Over-cap analyze: deliver the failure envelope to the original
            // caller (the <analyze-tool-result> envelope is the async-analyze caller's
            // only result path — "failed = terminal … surface to user").
            ResumableStage::Analyze {
                job_id,
                capped: true,
                ..
            } => {
                info!(
                    job = %job_id,
                    "Delivering analyze failure envelope (boot re-dispatch cap exceeded)",
                );
                let ws = workspace.clone();
                tokio::spawn(async move {
                    crate::tools::analyze::analyze_capped_envelope(&job_id, &ws).await;
                });
            }
            ResumableStage::ResearchCleanup { job_id, .. } => {
                info!(job = %job_id, "Resuming research cleanup agent at boot");
                let ws = workspace.clone();
                tokio::spawn(async move {
                    crate::research_cleanup::resume_research_cleanup(&job_id, &ws).await;
                });
            }
            ResumableStage::TicketStage {
                job_id,
                ticket_id,
                stage,
                ..
            } => {
                if let Ok(Some(ticket)) = crate::board::store().get_ticket(&ticket_id).await {
                    info!(
                        job = %job_id,
                        ticket = %ticket_id,
                        stage = %stage,
                        "Resuming ticket stage round at boot",
                    );
                    tokio::spawn(resume_ticket_stage_round(stage, job_id, ticket, workspace));
                } else {
                    warn!(
                        job = %job_id,
                        ticket = %ticket_id,
                        "Resume ticket not found — deleting job row",
                    );
                    let _ = crate::jobs::complete_ticket_stage_job(
                        &crate::session::store().conn,
                        &job_id,
                    )
                    .await;
                }
            }
        }
    }

    let interval = Duration::from_secs(1);
    loop {
        // Drain-aware: break when the graceful drain begins (in-flight rounds
        // finish; no new work is spawned).
        if !crate::shutdown::sleep_or_shutdown_or_drain(interval).await {
            break;
        }
        poll_round().await;
    }
}

/// Shared dispatch helper: log the ticket+workspace, then spawn the phase
/// dispatcher in a background task.
///
/// This is a plain `fn` (not `async`) because both `info!()` and
/// `tokio::spawn()` are synchronous operations — no `.await` needed.
///
/// When the circuit breaker trips for a ticket, all sibling
/// [`TicketPhase::ReadyForDevelopment`] tickets in the same workspace are
/// drained to [`TicketPhase::Planning`] so the Manager can triage the failure
/// without new tickets auto-starting.
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
        // The bounce breaker is not pre-flight (it is enforced at the bounce
        // sites: review/QA in process_verifier_verdicts and engineer hard
        // failures in bounce_engineer_hard_failure) — phases without a
        // pre-flight breaker pass `None` and try_trip_circuit_breaker returns
        // false immediately.
        //
        // The post-agent phase_changed_and_clear_assignment check in each dispatch function is
        // a separate concern (race-condition guard) and is preserved there.
        if !is_ticket_in_phase(&ticket.id, expected_phase).await {
            return;
        }
        if try_trip_circuit_breaker(&ticket, expected_phase, kind, log_label).await {
            // Circuit breaker tripped — unconditionally drain the development
            // pipeline regardless of which pre-flight breaker type fired
            // (Sanitation or Diagnostics; the bounce breaker has no pre-flight
            // arm and is enforced at its bounce sites instead). This is a
            // deliberate invariant:
            //
            //   Any breaker trip signals a potentially dirty workspace. If the
            //   drain were narrowed to only some breaker types, the Engineer
            //   could pick up an unrelated ReadyForDevelopment ticket while the
            //   workspace still has unfinished changes from the failed ticket —
            //   leading to cascading failures, wasted API credits, and confused
            //   agents.
            //
            //   Draining ALL ReadyForDevelopment tickets forces the Manager to
            //   triage the failed ticket first, inspect workspace state, deal
            //   with any uncommitted changes, and ensure the tree is clean
            //   before unrelated work safely resumes.
            drain_ready_for_development_siblings(&ticket).await;
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
            // Panic payloads can embed credential-bearing content (paths, env,
            // tool output) — scrub before the log and the failure comment.
            let msg = crate::util::scrub_credentials(&panic_message(&*payload));
            error!(
                ticket = %ticket_for_failure.id,
                panic = %msg,
                "Dispatch panicked — transitioning ticket to Failed",
            );
            if crate::shutdown::aborting() {
                // Drain-cut: no Failed transition or workspace pause during the
                // drain — the job stays status='launched' for boot resume
                // (consistent with every other drain-cut path).
                warn!(
                    ticket = %ticket_for_failure.id,
                    "Dispatch panic during drain — leaving ticket for boot resume"
                );
                return;
            }
            // Pause before the Failed transition so the failure notification
            // reflects the paused workspace. Shutdown is excluded inside the helper.
            let pause_note =
                pause_workspace_on_failure(&ticket_for_failure, "dispatch panic").await;
            // Best-effort transition: the ticket may have been moved
            // externally while the dispatch was running.
            let panic_comment = format!("❌ Dispatch panicked: {msg}{pause_note}");
            let _ = comment_and_transition(
                TransitionCtx::notifying(
                    &ticket_for_failure,
                    expected_phase,
                    TicketPhase::Failed,
                    "dispatch panic",
                ),
                SYSTEM_ROLE,
                &panic_comment,
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
    /// The phase the verifier is *actively working in* — the ticket phase
    /// the ticket occupies while the verifier's agents run (e.g.
    /// [`TicketPhase::InReview`] for reviewers, [`TicketPhase::InQa`] for QA).
    /// This is the phase that transitions *from* when the verifier finishes
    /// (to [`success_phase`] on success, or to Failed/ReadyForDevelopment
    /// on failure).
    ///
    /// Contrast with [`PollPhaseInfo::expected_phase`] which serves as the
    /// *target* phase for claim transitions in the poll loop.
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
    /// `None` for phases without a pre-flight breaker — the bounce breaker
    /// has no pre-flight arm (it is enforced at the bounce sites:
    /// review/QA in `process_verifier_verdicts`, engineer hard failures in
    /// `bounce_engineer_hard_failure`).
    circuit_breaker_kind: Option<CircuitBreakerKind>,
    /// Fresh-ticket grace for claims: when `Some`, tickets created within
    /// this window are not claimed (only the BacklogAnalysis phase sets it —
    /// see [`BoardStore::BACKLOG_CLAIM_GRACE`]).
    claim_grace: Option<ChronoDuration>,
    /// Human-readable label used in logs and circuit-breaker messages.
    /// PascalCase for roles ("Engineer", "Analyst"), "QA" for the QA verifier.
    log_label: &'static str,
}

impl PollPhaseInfo {
    /// Create a new [`PollPhaseInfo`] with standard defaults:
    /// [`PipelineCheck::Skip`] and no pre-flight circuit breaker (`None`).
    const fn new(expected_phase: TicketPhase, log_label: &'static str) -> Self {
        Self {
            expected_phase,
            pipeline_check: PipelineCheck::Skip,
            circuit_breaker_kind: None,
            claim_grace: None,
            log_label,
        }
    }
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
            // Backlog tickets younger than BoardStore::BACKLOG_CLAIM_GRACE stay
            // in Backlog: the Manager usually moves a fresh ticket straight to
            // Planning/ReadyForDevelopment right after create_ticket.
            Self::BacklogAnalysis => PollPhaseInfo {
                claim_grace: Some(BoardStore::BACKLOG_CLAIM_GRACE),
                ..PollPhaseInfo::new(TicketPhase::Analysis, "Analyst")
            },
            Self::EngineerDevelopment => PollPhaseInfo {
                pipeline_check: PipelineCheck::Enforce,
                ..PollPhaseInfo::new(TicketPhase::InDevelopment, "Engineer")
            },
            Self::SanitationCheck => PollPhaseInfo {
                // SanitationCheck is excluded from CLAIM_PHASES since the
                // actual QaPassed→InSanitation transition happens via
                // claim_sanitation in handle_qa_passed.
                circuit_breaker_kind: Some(CircuitBreakerKind::Sanitation),
                ..PollPhaseInfo::new(TicketPhase::InSanitation, "Sanitation")
            },
            Self::DiagnosticsCheck => PollPhaseInfo {
                circuit_breaker_kind: Some(CircuitBreakerKind::Diagnostics),
                ..PollPhaseInfo::new(TicketPhase::InDiagnostics, "Diagnostics")
            },
            Self::VerifierCheck(vi) => PollPhaseInfo::new(vi.active_phase, vi.log_label),
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

/// Spawn a background task per ticket in `phase` for the workspace.
///
/// Lists tickets via [`BoardStore::list_all_tickets`] with both filters set;
/// does NOT load comments — lightweight enough for poll loops. On DB errors,
/// logs and returns; tickets stay in phase and are re-picked next poll cycle.
///
/// Raw `tokio::spawn` is used instead of `spawn_dispatch` because there is no
/// claim transition — the ticket stays in its phase until the operation
/// succeeds, so transient failures (and panics, which `spawn_dispatch` would
/// move to `Failed`) are harmless: the ticket is re-dispatched on the next
/// poll cycle. `Ticket` is moved by value, so no `Arc` wrapping is needed.
async fn spawn_for_each_ticket_in_phase<F, Fut>(phase: TicketPhase, ws: &Workspace, f: F)
where
    F: Fn(Ticket, Workspace) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    match board().list_all_tickets(Some(&ws.name), Some(phase)).await {
        Ok(tickets) => {
            for ticket in tickets {
                let f = f.clone();
                let ws = ws.clone();
                tokio::spawn(async move {
                    f(ticket, ws).await;
                });
            }
        }
        Err(e) => {
            error!(workspace = %ws.name, phase = %phase, error = %e, "Phase listing failed");
        }
    }
}

/// Run one poll round: claim actionable tickets and dispatch agents.
///
/// Workspaces are processed concurrently via [`tokio::spawn`] so that a
/// slow or panicking workspace does not delay others. The per-workspace
/// steps are issued in order; safety comes from atomic claim transitions
/// (per-ticket work runs in detached tasks, not serially).
///
/// The per-workspace steps (see [`process_single_workspace`]) are:
///
/// 0. **Pending pickup** — claim `Pending` workspaces into their first
///    discovery once the LLM provider is configured
///    ([`pickup_pending_workspace`]).
/// 1. **Pipeline claims** — atomic source→target phase transitions
///    (`run_claim_pipeline`).
///
///    Steps 2-5 run concurrently via `tokio::join!` in
///    [`process_single_workspace`] (see there for DB serialization and
///    snapshot-timing notes):
/// 2. **DiagnosticsCheck** — dispatch unassigned `InDiagnostics` tickets
///    so diagnostics commands (fmt, lint, build, test) continue running.
/// 3. **SanitationPassed → Done** — auto-commit tickets that have passed
///    sanitation.
/// 4. **QaPassed** — check the working tree for untracked files; claim to
///    `InSanitation` if dirty, otherwise commit directly to `Done`.
/// 5. **SanitationCheck** — dispatch unassigned `InSanitation` tickets.
///
/// # Architecture history
///
/// The iteration order evolved over three refactors:
/// 1. **Phase-major** — all workspaces claim Backlog, then all claim Engineer, …
/// 2. **Workspace-major (serial)** — workspace A claims all phases, then B, …
/// 3. **Workspace-major (parallel)** — all workspaces processed concurrently.
///
/// Step (3) isolates per-workspace panics (a crash in one workspace no longer
/// permanently kills the management loop) and saves the serial DB-query
/// overhead that accumulated across many workspaces (~10–15 ms per workspace
/// per 1‑s cycle). The heavy work (LLM calls, git operations) was already
/// parallel via internal `tokio::spawn` in earlier iterations.
///
/// # Concurrency safety
///
/// - The [`BoardStore`] uses a single Turso connection wrapped in
///   `Arc<tokio::sync::Mutex>>` — SQL operations serialize at the mutex,
///   so concurrent access is safe. All workspaces share the same `board.db`;
///   per-workspace isolation is via SQL `WHERE workspace_name = ?` filtering.
/// - The [`ticket_buffer`] and [`registry::AGENT_REGISTRY`] are
///   `Mutex`‑protected global singletons; contention is negligible because
///   phase transitions are infrequent per‑workspace per cycle.
async fn poll_round() {
    let workspaces = match crate::workspace::store().list().await {
        Ok(ws_list) => ws_list,
        Err(e) => {
            error!(error = %e, "Failed to list workspaces");
            return;
        }
    };

    let tasks: Vec<_> = workspaces
        .into_iter()
        .map(|ws| {
            tokio::spawn(async move {
                process_single_workspace(ws).await;
            })
        })
        .collect();

    let results = join_all(tasks).await;
    crate::util::log_join_failures(
        results,
        "Panic in workspace poll round — management loop continues",
        "Workspace poll task was cancelled — management loop continues",
    );
}

/// Process the poll steps for a single workspace.
///
/// Steps 0-1 are issued in order because claims must be ordered and the git tree
/// is shared; steps 2-5 are concurrent via `tokio::join!` (see inline comments
/// below for serialization and snapshot notes). Per-ticket work runs concurrently
/// in detached tasks. Across workspaces this function is called concurrently by
/// [`poll_round`].
async fn process_single_workspace(ws: Workspace) {
    // 0. Pending-workspace pickup — a pending workspace (added without a
    // provider key, or returned to pending after provider-class discovery
    // failures) is claimed into its first discovery when the provider is
    // configured. Runs before the claim pipeline: a freshly-claimed workspace
    // re-arms the analysis pause, so no Backlog/RFD claims race the
    // discovery. A successful claim returns the fresh post-claim copy; the
    // pipeline below then runs against the claimed analyzing+paused state
    // instead of the pre-claim poll-round copy.
    let ws = match pickup_pending_workspace(&ws).await {
        Some(claimed) => claimed,
        None => ws,
    };

    // 1. Pipeline claims — atomic source→target transitions
    run_claim_pipeline(&ws).await;

    // 2-5. Concurrent poll listings via `tokio::join!` — the four phase listings
    // are independent and each already handles its own errors per phase
    // (see `spawn_for_each_ticket_in_phase`). DB listings serialize on
    // `Arc<Mutex<turso::Connection>>`, so the benefit is async scheduling tail
    // latency removal on the 1s poll loop, not DB parallelism.
    //
    // Snapshot timing: `handle_qa_passed` claims run in detached `tokio::spawn`
    // tasks, so a concurrent `InSanitation` listing may snapshot before
    // just-claimed tickets appear and will pick them up next poll cycle (~1s) —
    // already tolerated by existing design (tickets re-dispatched next cycle).
    // This is true even sequentially; concurrent join only changes listing
    // snapshot interleaving, not claim completion.
    tokio::join!(
        // 2. DiagnosticsCheck — diagnostics keeps the ticket
        // in InDiagnostics while running, so the claim loop isn't applicable.
        // Tickets that already have assigned_to set are mid-execution and must not
        // be re-dispatched; transient DB listing errors are safe — tickets stay
        // in phase and are re-dispatched on the next poll cycle (~1s).
        spawn_for_each_ticket_in_phase(TicketPhase::InDiagnostics, &ws, |ticket, ws| async move {
            if ticket.assigned_to.is_some() {
                return;
            }
            spawn_dispatch(PollPhase::DiagnosticsCheck, ticket, ws);
        }),
        // 3. SanitationPassed → Done (auto-commit),
        // following the same pattern as the QaPassed→Done commit flow.
        spawn_for_each_ticket_in_phase(
            TicketPhase::SanitationPassed,
            &ws,
            |ticket, ws| async move {
                let Some(porcelain) = ensure_git_or_done_and_get_status(
                    &ticket,
                    &ws,
                    TicketPhase::SanitationPassed,
                    "finalize",
                )
                .await
                else {
                    return;
                };
                finalize_ticket_with_git_status(
                    ticket,
                    ws,
                    TicketPhase::SanitationPassed,
                    &porcelain,
                )
                .await;
            },
        ),
        // 4. Handle QaPassed tickets.
        //
        // For each QaPassed ticket, check whether the working tree has new/untracked
        // files. If it does, claim the ticket to InSanitation and dispatch a sanitation
        // agent. Otherwise, commit directly and transition to Done (existing behavior).
        //
        // Spawned via tokio::spawn (inside spawn_for_each_ticket_in_phase) to prevent
        // git operations from blocking the poll loop. The ticket stays in QaPassed
        // until either the claim or the commit succeeds, so re-dispatch is harmless.
        spawn_for_each_ticket_in_phase(TicketPhase::QaPassed, &ws, handle_qa_passed),
        // 5. SanitationCheck — handle_qa_passed (step 4) runs
        // in spawned tasks concurrent with this step. It may: (a) claim
        // QaPassed→InSanitation with assigned_to set, or (b) commit the ticket
        // directly to Done. Tickets that already have assigned_to set are
        // mid-execution and are skipped — neither path races with this step
        // because the ticket is either in a different phase or already assigned.
        spawn_for_each_ticket_in_phase(TicketPhase::InSanitation, &ws, |ticket, ws| async move {
            if ticket.assigned_to.is_some() {
                return;
            }
            spawn_dispatch(PollPhase::SanitationCheck, ticket, ws);
        }),
    );
}

/// Pickup step: claim a `Pending` workspace into its first discovery when the
/// LLM provider is configured and no provider-failure cooldown is armed.
///
/// The claim is an atomic conditional UPDATE (`status = 'pending'` →
/// `'analyzing'`, `paused = 1`) so concurrent pickups, GUI Re-analyze, or
/// delete cannot double-dispatch — the live `discovery_generation` is read in
/// the same statement (never hardcoded), so the spawned discovery's finalize
/// passes the generation guard. Diagnostics discovery is skipped when
/// diagnostics already exist: a crashed mid-rediscover workspace re-picked-up
/// at boot must not overwrite user-managed diagnostics.
///
/// Returns the fresh post-claim [`Workspace`] (status `analyzing`, paused) so
/// the caller can run the remaining poll steps against the claimed state
/// rather than the pre-claim poll-round copy; `None` when the workspace stays
/// pending.
async fn pickup_pending_workspace(ws: &Workspace) -> Option<Workspace> {
    // `?` propagates the claim's None residual (the payload type differs from
    // this function's return type, which is fine for Option).
    let (generation, discover_diagnostics) = pickup_claim(ws).await?;

    info!(
        workspace = %ws.name,
        generation,
        discover_diagnostics,
        "Pickup: pending workspace claimed into discovery"
    );
    spawn_workspace_discovery(ws, generation, discover_diagnostics);

    // Re-read so the poll steps that follow see the claimed state
    // (analyzing + paused), not the stale pre-claim copy. On a read failure
    // the caller falls back to the poll-round copy — the pipeline's status
    // gate ([`blocks_claim`]) still blocks new-work claims on it.
    crate::workspace::store()
        .get_by_name(&ws.name)
        .await
        .ok()
        .flatten()
}

/// Decide + claim half of [`pickup_pending_workspace`], split out so tests can
/// exercise the gating and the atomic claim without spawning real discovery
/// agents. Returns `Some((discovery_generation, discover_diagnostics))` when
/// the workspace was atomically claimed, `None` when it must stay pending
/// (not pending / provider unconfigured / cooldown armed / claim lost to a
/// concurrent claimer or delete).
///
/// # Pause-flag overloading
///
/// The claim deliberately does **not** gate on `ws.paused`. A `Pending`
/// workspace always carries `paused = 1` — the analysis pause written by
/// `add()` and the discovery finalizer — so gating on the paused column
/// would block every pending pickup and break the analysis-pause flow. The
/// user pause toggle therefore cannot stop a pending workspace's retry
/// cycle: it only gates the ticket pipeline (see [`blocks_claim`]).
/// A successful pickup discovery also unpauses via the finalizer (the
/// documented analysis-pause lifecycle), so a manual pause on a `Pending`
/// workspace is not preserved across a pickup — the escalating cooldown,
/// not the pause flag, bounds the retry rate. Do not "fix" this by adding a
/// `paused` gate here.
///
/// The reverse direction is covered by the pipeline instead: a manual GUI
/// unpause on a `Pending`/`Analyzing`/`Failed` workspace does **not**
/// re-enable new-work claims — [`blocks_claim`] requires `Ready` for
/// Backlog/RFD claims regardless of the pause flag, so a non-Ready workspace
/// cannot take development work even when unpaused.
async fn pickup_claim(ws: &Workspace) -> Option<(i64, bool)> {
    if ws.status != WorkspaceStatus::Pending {
        return None;
    }
    if !crate::config::provider_configured() {
        return None;
    }
    if crate::workspace::pending_pickup_cooldown_active(&ws.name) {
        return None;
    }

    let storage = crate::workspace::store();
    let generation = match storage.claim_pending_for_discovery(&ws.name).await {
        Ok(Some(generation)) => generation,
        // Concurrent claimer (GUI Re-analyze / another pickup) won the race —
        // the row is no longer pending; nothing to do.
        Ok(None) => return None,
        Err(e) => {
            // Persistent DB faults must not strand the workspace silently:
            // log and let the next poll cycle re-evaluate.
            warn!(
                workspace = %ws.name,
                error = %e,
                "Pickup: failed to claim pending workspace — retrying next poll cycle"
            );
            return None;
        }
    };

    // Fresh read for the diagnostics-preservation decision: the poll-round
    // copy may predate a concurrent set_diagnostics. On a read failure the
    // poll-round copy decides instead of defaulting to discovery — the
    // non-destructive direction for the crash-mid-rediscover case this flag
    // exists to protect. The diagnostics generation guard only covers a save
    // landing *while the discovery runs*; this fresh read additionally
    // preserves a save made between the poll-round list and the claim (a save
    // after this read is a narrow accepted residual race).
    let discover_diagnostics = match storage.get_by_name(&ws.name).await {
        Ok(Some(fresh)) => fresh.diagnostics.is_none(),
        Ok(None) | Err(_) => ws.diagnostics.is_none(),
    };

    Some((generation, discover_diagnostics))
}

/// Whether the automatic claim for `phase` is blocked for `ws`.
///
/// Two gates apply to *new-work* claims — BacklogAnalysis (backlog →
/// analysis) and EngineerDevelopment (ready_for_development →
/// in_development):
///
/// * **Pause gate** — a paused workspace stops the automatic pickup of new
///   work, so queued tickets don't cascade into a workspace that failed or
///   was manually paused.
/// * **Status gate** — the workspace must have completed discovery
///   (`Ready`). Pending/Analyzing/Failed workspaces have missing or stale
///   contexts; a manual unpause on them (or the same-round stale copy after
///   a pending-pickup claim) must not re-enable new-work claims.
///
/// Later-phase claims (DiagnosticsDone→Review, Reviewed→QA) proceed so
/// tickets already past analysis finish without getting stuck — pausing and
/// incomplete discovery gate new analysis/development, not in-progress work.
fn blocks_claim(ws: &Workspace, phase: PollPhase) -> bool {
    if !matches!(
        phase,
        PollPhase::BacklogAnalysis | PollPhase::EngineerDevelopment
    ) {
        return false;
    }
    ws.paused || ws.status != WorkspaceStatus::Ready
}

/// Claim for each pipeline phase in a workspace.
///
/// [`blocks_claim`] skips the automatic pickup of new work (backlog →
/// analysis and ready_for_development → in_development) when the workspace is
/// paused **or** has not completed discovery (`Ready`). Later-phase claims
/// (DiagnosticsDone→Review, Reviewed→QA) proceed normally so that tickets
/// already past analysis finish without getting stuck — pausing gates new
/// analysis/development, not in-progress work.
///
/// On claim error we `break` out of the phase loop — this skips all
/// remaining CLAIM_PHASES for this workspace and falls through to
/// Diagnostics/QaPassed (which handle their own errors independently).
/// A DB-down workspace won't block other workspaces; a transient claim
/// failure won't generate log noise for every remaining phase.
async fn run_claim_pipeline(ws: &Workspace) {
    let board = board();
    for &(source, phase) in CLAIM_PHASES {
        if blocks_claim(ws, phase) {
            continue;
        }
        let info = phase.info();
        let ticket = match board
            .claim_ticket_in_workspace(
                source,
                info.expected_phase,
                &ws.name,
                info.pipeline_check,
                info.claim_grace,
            )
            .await
        {
            Ok(Some(t)) => {
                // Buffer the claim transition. The returned ticket already
                // has phase = info.expected_phase (from SQL RETURNING), so record
                // the transition from source.
                ticket_buffer::push(
                    &ws.name,
                    &t.id,
                    source,
                    t.phase,
                    ticket_buffer::TransitionOrigin::Pipeline,
                );
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
}

/// Run a single pre-registered agent. Registration stays in the caller
/// (before prompt evaluation / assignment) so mid-work comment delivery is
/// unaffected; run_agent's exit guard unregisters on every path (incl. panic).
async fn run_single_agent(
    agent_id: String,
    role: Role,
    ws: &Workspace,
    ticket: &Ticket,
    message: &str,
    incoming_rx: tokio::sync::mpsc::UnboundedReceiver<crate::message_router::AgentJob>,
    resume: bool,
) -> (Agent, Option<String>) {
    run_agent(
        agent_id,
        role,
        ws,
        Some(ticket),
        message,
        String::new(),
        String::new(),
        Some(incoming_rx),
        resume,
        None,
        None,
        None,
    )
    .await
}

/// Build the failure comment for a failed Engineer run.
///
/// Persists the classified cause + underlying detail so distinct root causes
/// (LLM retry exhaustion, process shutdown, user cancellation, concrete agent
/// errors) stop looking identical in ticket history. The generic template is
/// retained only for genuinely-unknown causes. Error detail is credential-
/// scrubbed then sandwich-truncated to [`crate::util::FAILURE_DETAIL_CAP`].
///
/// Classification order matters: the global shutdown token fires on SIGTERM/
/// SIGINT and dashboard close, the per-agent token on /stop and dashboard
/// close — check the global token first so shutdown isn't mislabeled as a
/// user cancel. The generic template branch is a total-function fallback for
/// genuinely-unknown causes (currently unreachable via run_agent's contract,
/// which yields either a captured error or a cancelled agent) — retained per
/// the failure-reporting contract. The failure-comment write during service
/// shutdown is best-effort (the dispatch task is detached); acceptable —
/// state semantics are unchanged.
fn engineer_failure_comment(shutdown: bool, cancelled: bool, error: Option<&str>) -> String {
    if shutdown {
        return "Engineer failed: service shutting down — the run was interrupted \
                by process shutdown."
            .to_string();
    }
    if cancelled {
        return "Engineer failed: cancelled by user.".to_string();
    }
    let Some(detail) = error else {
        return load_prompt("pipeline/engineer_failed.md");
    };
    let detail = crate::util::scrub_credentials(detail);
    let detail = crate::util::truncate_sandwich(
        &detail,
        crate::util::FAILURE_DETAIL_CAP,
        "engineer failure",
    );
    // Matches the single-sourced agent-loop exhaustion marker
    // (crate::agent::RETRY_EXHAUSTION_MARKER) so retry exhaustion keeps its
    // dedicated classification.
    if detail.contains(RETRY_EXHAUSTION_MARKER) {
        format!("Engineer failed: LLM provider retry exhaustion.\n\n{detail}")
    } else {
        format!("Engineer failed.\n\n{detail}")
    }
}

/// Extract a concise structured summary of the engineer's work for the ticket
/// comment. Session-clean: the extraction runs on a local copy of the session
/// history ([`Agent::extract_verdict`] never writes to the session store), so
/// the session stays resumable without the extraction in it. The ticket comment
/// is the authoritative record; on extraction failure or empty results the raw
/// response is kept (fail-open — work is never silently dropped). Uses the
/// short comment-only retry budget — a fail-open operation must not stall the
/// pipeline behind the verdict-gate schedule. The fallback paths scrub without
/// truncating (fail-open content is preserved; only the success path applies
/// the 24 KB sandwich cap). The extraction widens the post-agent window
/// (assigned_to still set, agent unregistered) to at most ~90 s: comments
/// arriving then are persisted and picked up on the next engineer round,
/// never lost.
async fn engineer_comment_text(agent: &Agent, raw: &str) -> String {
    let ticket_id = agent.ticket.as_ref().map_or("?", |t| t.id.as_str());
    let policy = crate::retry::RetryPolicy::comment();
    let extraction_prompt = load_prompt("extraction/engineer.md");
    let summary = match agent
        .extract_verdict::<crate::EngineerSummary>(&extraction_prompt, None, Some(&policy))
        .await
    {
        Ok(summary) => summary,
        Err(e) => {
            warn!(
                ticket = %ticket_id,
                error = %e,
                "Engineer summary extraction failed — using raw response for ticket comment"
            );
            return crate::util::scrub_credentials(raw);
        }
    };
    let items: Vec<&str> = summary
        .items
        .iter()
        .map(String::as_str)
        .filter(|s| !s.trim().is_empty())
        .collect();
    if items.is_empty() {
        warn!(
            ticket = %ticket_id,
            "Engineer summary extraction returned no usable items — using raw response for ticket comment"
        );
        return crate::util::scrub_credentials(raw);
    }
    let mut out = String::from("Implemented / fixed / executed:");
    for item in items {
        let _ = write!(out, "\n- {}", item.replace('\n', " "));
    }
    let synopsis = summary.summary.as_deref().unwrap_or("").trim();
    if !synopsis.is_empty() {
        let _ = write!(out, "\n\n### Summary\n{synopsis}");
    }
    crate::util::truncate_sandwich(
        &crate::util::scrub_credentials(&out),
        crate::util::FAILURE_DETAIL_CAP,
        "engineer summary comment",
    )
}

/// Drain-cut guard for the single-agent stage-round finalizers
/// ([`finalize_engineer_round`], [`finalize_sanitation_round`]): when the
/// round produced no response and the app is aborting (drain), the job must
/// stay status='launched' for boot resume — no failure record, no
/// AGENT_FAILURE_EMOJI. Silent on resume (the fresh dispatch already logged
/// the drain). Returns `true` when the caller must abort the finalize tail
/// early, leaving the job 'launched'; `false` to continue processing. `label`
/// parameterizes the log line ("Engineer"/"Sanitation").
fn stage_round_drain_cut(
    ticket_id: &str,
    label: &str,
    response: Option<&str>,
    resumed: bool,
) -> bool {
    let drain_cut = response.is_none() && crate::shutdown::aborting();
    if drain_cut && !resumed {
        info!(
            ticket = %ticket_id,
            "{label} round cut short by drain — job stays launched for boot resume",
        );
    }
    drain_cut
}

/// Shared engineer post-run tail: phase/drain guards, failure handling, pause,
/// transition, and job terminalization. Diagnostics are dispatched by the poll
/// loop as a separate `PollPhase::DiagnosticsCheck` (see `poll_round`) — the
/// success path only transitions to InDiagnostics. The guards live here:
/// phase-moved → complete job and bail; drain-cut (see
/// [`stage_round_drain_cut`]) → leave the job 'launched' for boot resume
/// (fresh dispatches log, resumes stay silent); response `None` past the
/// guards is a real failure or user cancel, delegated to
/// [`handle_engineer_failure`].
///
/// `resumed` selects the observability log strings.
async fn finalize_engineer_round(
    ticket: &Ticket,
    agent: &Agent,
    response: Option<&str>,
    job_id: &str,
    resumed: bool,
) {
    if phase_changed_and_clear_assignment(&ticket.id, TicketPhase::InDevelopment).await {
        complete_ticket_stage_job(job_id).await;
        return;
    }
    if stage_round_drain_cut(&ticket.id, "Engineer", response, resumed) {
        return;
    }

    // Success path: engineer produced output — transition to InDiagnostics
    // (unchanged).
    if let Some(text) = response {
        let comment_text = engineer_comment_text(agent, text).await;
        comment_and_transition_or_bail(
            TransitionCtx::buffered(
                ticket,
                TicketPhase::InDevelopment,
                TicketPhase::InDiagnostics,
                "Engineer",
            ),
            Role::Engineer.as_str(),
            &comment_text,
            if resumed {
                "Resumed engineer finished — transitioned ticket"
            } else {
                "Engineer finished — transitioned ticket"
            },
        )
        .await;
        complete_ticket_stage_job(job_id).await;
        return;
    }

    // Past the guards above, response None here is a real failure or a user
    // cancel — classify, pause, and either fail or bounce. The helper returns
    // false when a shutdown/drain landed mid-tail: the job must then stay
    // status='launched' for boot resume (mirrors finalize_verifier_round's
    // early return before job completion).
    if !handle_engineer_failure(ticket, agent, resumed).await {
        return;
    }
    complete_ticket_stage_job(job_id).await;
}

/// Handle the engineer failure tail (response `None` past the guards).
///
/// - **User cancel** → ticket Failed + workspace pause (unchanged, never
///   auto-re-queued).
/// - **Hard failure** ("Agent failed") → the workspace pauses and the ticket
///   bounces back to ReadyForDevelopment for an automatic retry, consuming
///   the same bounce budget as review/QA bounces. When the budget is
///   exhausted the ticket moves to Failed via the circuit breaker — still
///   with the workspace paused and WITHOUT draining ReadyForDevelopment
///   siblings. The concrete error is recorded as a `SYSTEM_ROLE`
///   comment so it survives into the retry's rework feedback (dispatch_engineer
///   builds feedback from comments after the last engineer-role comment) and
///   into the Manager notification.
/// - **Shutdown/drain race** (token/drain landing after the first drain-cut
///   check and before the transition commit — re-checked after the
///   workspace-pause await) → no bounce, no Failed transition; the job stays
///   'launched' for boot resume. A pause may already be committed if the
///   drain flipped mid-await (recoverable via the normal unpause).
///
/// Returns `false` when a shutdown/drain cut the failure handling short — the
/// caller must then leave the job status='launched' for boot resume. Returns
/// `true` otherwise (failure handling ran to the transition attempt); the
/// caller terminalizes the stage job.
async fn handle_engineer_failure(ticket: &Ticket, agent: &Agent, resumed: bool) -> bool {
    let cancelled = agent.is_cancelled();

    // No drain re-check here: the caller's stage_round_drain_cut just ran
    // aborting() and no await separates that read from this point, so a
    // second read would be provably identical. The first real gap is the
    // pause await below, which is where the post-pause guard lives.

    // Pause before the transition so the failure notification reflects the
    // paused workspace. Shutdown is excluded inside the helper; a hard
    // failure pauses on every occurrence, including the budget-exhausting
    // bounce-trip one.
    let pause_reason = if cancelled {
        "user cancelled the agent run"
    } else {
        "engineer agent failure"
    };
    let pause_note = pause_workspace_on_failure(ticket, pause_reason).await;

    // Second drain-cut: the pause await is the first real gap after the
    // caller's stage_round_drain_cut — a drain landing there must not commit
    // a bounce or a Failed transition (the job stays 'launched' for boot
    // resume). The pause may already be committed if the drain flipped
    // mid-await; recoverable via the normal unpause and strictly better than
    // committing the transition during the drain.
    if crate::shutdown::aborting() {
        info!(
            ticket = %ticket.id,
            "Engineer failure cut short by shutdown/drain after the pause — job stays launched for boot resume",
        );
        return false;
    }

    // The guards above ensure the global token is not cancelled when this
    // read normally runs, but it CAN fire in the window between the last
    // guard and this read — passing the live value keeps such a shutdown
    // classified as shutdown instead of being misread as a user cancel or a
    // hard failure.
    let failure_comment = engineer_failure_comment(
        crate::shutdown::shutdown_token().is_cancelled(),
        cancelled,
        agent.failure.as_deref(),
    );
    let comment_text = format!("{failure_comment}{pause_note}");

    if cancelled {
        // User-initiated cancellation keeps today's semantics: ticket Failed
        // + workspace pause, never auto-re-queued.
        comment_and_transition_or_bail(
            TransitionCtx::notifying(
                ticket,
                TicketPhase::InDevelopment,
                TicketPhase::Failed,
                "Engineer",
            ),
            SYSTEM_ROLE,
            &comment_text,
            if resumed {
                "Resumed engineer cancelled — transitioned ticket"
            } else {
                "Engineer cancelled — transitioned ticket"
            },
        )
        .await;
    } else {
        bounce_engineer_hard_failure(ticket, &comment_text, resumed).await;
    }
    true
}

/// Bounce a hard-failed engineer ticket back to ReadyForDevelopment for an
/// automatic retry, sharing the review/QA bounce budget. The counter is
/// bumped atomically with the transition (drift-free invariant). When the
/// budget is exhausted the ticket fails via the circuit breaker — workspace
/// still paused, ReadyForDevelopment siblings NOT drained (unlike the
/// verifier trip). The concrete-error comment is written LAST so the Manager
/// notification surfaces it (`notify_ticket` renders the last comment as the
/// failure details). `resumed` selects the observability log prefix.
async fn bounce_engineer_hard_failure(ticket: &Ticket, comment_text: &str, resumed: bool) {
    let bounce_trip = bounce_exhausted(ticket.bounce_count);
    let trip_comment = bounce_trip.then(bounce_breaker_trip_comment);
    let target = if bounce_trip {
        TicketPhase::Failed
    } else {
        TicketPhase::ReadyForDevelopment
    };

    // Not a circuit-breaker drain: siblings are NOT moved to Planning, so the Failed notification must render the paused-workspace sentence, not the drain sentence.
    let outcome = with_comment_and_transition(
        TransitionCtx::notifying(ticket, TicketPhase::InDevelopment, target, "Engineer"),
        async |tx| {
            // Trip comment first, failure comment last — notify_ticket
            // renders comments.last() as the failure details, so the
            // Manager sees the concrete error, not just the trip message.
            if let Some(comment) = &trip_comment {
                BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, comment).await?;
            }
            BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, comment_text).await?;
            // The failing bounce is not counted (stays at the max).
            if !bounce_trip {
                BoardStore::increment_bounce_count_tx(tx, &ticket.id).await?;
            }
            Ok(())
        },
    )
    .await;

    if matches!(outcome, FinalizeOutcome::Applied) {
        let resumed_prefix = if resumed { "Resumed " } else { "" };
        if bounce_trip {
            info!(
                ticket = %ticket.id,
                "{resumed_prefix}Engineer hard failure exhausted the bounce budget — ticket failed",
            );
        } else {
            info!(
                ticket = %ticket.id,
                "{resumed_prefix}Engineer hard failure — ticket bounced to ReadyForDevelopment for retry",
            );
        }
    }
}

/// Run an Engineer agent to implement the ticket.
///
/// Gathers feedback comments from all roles since the last engineer run and
/// includes them in the agent prompt. After the agent finishes, performs a
/// post-run phase check to catch race conditions, then transitions:
/// - InDiagnostics (buffer) on successful completion
/// - ReadyForDevelopment (notify) on a hard "Agent failed" outcome — the
///   ticket retries automatically, sharing the review/QA bounce budget
/// - Failed (notify) on user cancel, or when the bounce budget is exhausted
async fn dispatch_engineer(ticket: Arc<Ticket>, ws: Workspace) {
    let agent_id = ticket_agent_id(&ticket.id, Role::Engineer.as_str());

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
        load_prompt("implement.md")
    } else {
        substitute(
            &load_prompt("pipeline/bounce_feedback.md"),
            &[("{{feedback}}", &feedback.join("\n---\n"))],
        )
    };

    // Spawn the durable engineer round (single-slot roster) BEFORE the round
    // tail — the job is the resume handle across crashes AND graceful drains.
    let job_id = crate::generate_id();
    let Ok(_slot) = spawn_single_slot_round(
        &job_id,
        &ticket,
        &ws,
        "engineer",
        TicketPhase::InDevelopment,
        Role::Engineer,
        &message,
        &agent_id,
    )
    .await
    else {
        error!(
            ticket = %ticket.id,
            "Failed to spawn engineer job — aborting dispatch",
        );
        return;
    };
    run_stage_agent_round(
        &ticket,
        &ws,
        &job_id,
        &message,
        false,
        StageRoundKind::Engineer,
    )
    .await;
}

/// The single-agent stage rounds sharing one run tail: engineer (NULL-seat
/// anchor, [`finalize_engineer_round`]) and sanitation (job-derived agent ID,
/// [`finalize_sanitation_round`]).
#[derive(Clone, Copy)]
enum StageRoundKind {
    Engineer,
    Sanitation,
}

/// Run the stage-agent round tail shared by fresh dispatch and boot resume.
///
/// The NULL-seat anchor (permanently-NULL `job_id`) is upserted FIRST so the
/// accumulated session `ticket_{id}_engineer` survives round-job deletion and
/// the 8h purge. The upsert is idempotent (partial unique index — single row
/// per agent_id, job_id NULL): a crash between `spawn_single_slot_round` and
/// the fresh-dispatch upsert (two adjacent awaits) would otherwise leave this
/// round WITHOUT an anchor — after CASCADE the session loses TTL protection
/// and the next bounce round loses S5 context. (Engineer stage only.)
///
/// The sanitation agent ID is job-derived (`ticket_{job_id}_sanitation`):
/// derived FIRST as a pure function of job_id, then registered (router +
/// `assigned_to`), then the session-non-emptiness discriminator is read — the
/// session-store read and router registration are independent (benign
/// ordering). The register overwrites the claim-time placeholder
/// (`ticket_{ticket_id}_sanitation`) with the job-derived run ID so mid-run
/// comments route to the actual agent (and covers the re-dispatch path after
/// [`record_sanitation_failure`] clears `assigned_to`). (Sanitation stage only.)
///
/// Resume dispatch rule — session-non-emptiness discriminator: the check is
/// "any session content", not "current round's task present" — existing
/// session → empty message (no duplicate task-prompt append); missing session
/// → stored task re-dispatched. A crash between the stage's durable write
/// (engineer: anchor upsert; sanitation: spawn row) and this round's
/// session.init leaves the round's feedback undelivered (a non-empty prior
/// round then dispatches an empty message). Narrow window, quality-only —
/// accepted. Fresh dispatch always uses the rendered prompt.
async fn run_stage_agent_round(
    ticket: &Ticket,
    ws: &Workspace,
    job_id: &str,
    task: &str,
    resumed: bool,
    kind: StageRoundKind,
) {
    if let StageRoundKind::Engineer = kind
        && let Err(e) = crate::jobs::upsert_engineer_anchor(
            &crate::session::store().conn,
            &ticket.id,
            task,
            crate::jobs::RowStatus::Launched,
        )
        .await
    {
        warn!(
            ticket = %ticket.id,
            job = %job_id,
            error = %e,
            "Failed to upsert engineer anchor — session continuity across bounces degraded",
        );
    }

    let agent_id = match kind {
        StageRoundKind::Engineer => crate::jobs::engineer_anchor_id(&ticket.id),
        StageRoundKind::Sanitation => format!("ticket_{job_id}_sanitation"),
    };
    let incoming_rx = register_agent_and_assign(
        &ticket.id,
        &agent_id,
        match kind {
            StageRoundKind::Engineer if resumed => {
                "Failed to persist assigned_to for resumed engineer — comments may not route"
            }
            StageRoundKind::Engineer => {
                "Failed to persist assigned_to — stale agent already cancelled at dispatch, proceeding without DB assignment"
            }
            StageRoundKind::Sanitation => {
                "Failed to persist assigned_to for sanitation agent — mid-run comments may not route"
            }
        },
    )
    .await;

    let has_session = resumed && crate::session::store().has_content(&agent_id).await;
    let message = if has_session {
        String::new()
    } else {
        task.to_string()
    };
    let (agent, response) = run_single_agent(
        agent_id,
        match kind {
            StageRoundKind::Engineer => Role::Engineer,
            StageRoundKind::Sanitation => Role::Sanitation,
        },
        ws,
        ticket,
        &message,
        incoming_rx,
        resumed,
    )
    .await;

    match kind {
        StageRoundKind::Engineer => {
            finalize_engineer_round(ticket, &agent, response.as_deref(), job_id, resumed).await;
        }
        StageRoundKind::Sanitation => {
            finalize_sanitation_round(ticket, &agent, response.as_deref(), job_id, resumed).await;
        }
    }
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
    }
}

/// Ensure git is available and run `git status --porcelain`.
///
/// # Side effects
///
/// **May transition the ticket to Done.** If git is unavailable (`git` not
/// installed or the workspace is not a git repo), this helper immediately
/// transitions the ticket to Done and returns `None`. This
/// is intentional — the ticket has already reached a terminal pipeline phase
/// and should not block on infrastructure issues.
///
/// # Returns
///
/// - `Some(porcelain)` — git is available and `git status --porcelain`
///   succeeded. The caller should use the output to decide how to finalize.
/// - `None` — git is unavailable (ticket was already moved to Done) or the
///   status query failed (caller should return; the poller will retry).
async fn ensure_git_or_done_and_get_status(
    ticket: &Ticket,
    ws: &Workspace,
    phase: TicketPhase,
    error_context: &'static str,
) -> Option<String> {
    let repo_path = ws.as_path();

    if !crate::git_commands::git_is_installed().await {
        transition_ticket_to_done(
            ticket,
            phase,
            "Git not installed — moving to Done without commit",
        )
        .await;
        return None;
    }
    if !crate::git_commands::is_git_repo(repo_path) {
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
///
/// Callers **must** have already verified git availability and obtained a porcelain
/// string via [`run_git_status`] before calling this function.
///
/// - **Clean tree** (empty porcelain): transitions directly to Done.
/// - **Dirty tree**: commits the changes via [`crate::git_commands::run_git_commit`].
/// - **Commit failure**: ticket stays in `source` phase; the poller retries.
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

    match crate::git_commands::run_git_commit(repo_path, &ticket.title).await {
        Ok(commit_info) => {
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
/// ticket to Done atomically within a single DB transaction via
/// [`with_comment_and_transition`].
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
    let Some(porcelain) = ensure_git_or_done_and_get_status(
        &ticket,
        &ws,
        TicketPhase::QaPassed,
        "untracked files check",
    )
    .await
    else {
        return;
    };

    let untracked = parse_new_files_from_porcelain(&porcelain);

    if untracked.is_empty() {
        // Git status and availability already checked above — delegate directly
        // to the helper that commits dirty changes or transitions to Done.
        finalize_ticket_with_git_status(ticket, ws, TicketPhase::QaPassed, &porcelain).await;
    } else {
        // Untracked files exist — claim this specific ticket to InSanitation
        // via the dedicated claim_sanitation method (see BoardStore docs).
        let agent_id = ticket_agent_id(&ticket.id, Role::Sanitation.as_str());
        let claimed = match board().claim_sanitation(&ticket.id, &agent_id).await {
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
            ticket_buffer::TransitionOrigin::Pipeline,
        );

        spawn_dispatch(PollPhase::SanitationCheck, ticket, ws);
    }
}

/// Record a sanitation failure: add a [`SANITATION_ROLE`] comment for the circuit breaker
/// and clear assigned_to so the ticket can be re-dispatched.
///
/// `raw_dump`: when the failure is a verdict
/// extraction failure, the last-attempt raw response is dumped into the
/// comment the same way the parallel verdict path does.
async fn record_sanitation_failure(
    ticket_id: &str,
    reason: impl std::fmt::Display,
    raw_dump: Option<&crate::retry::RetryExhausted>,
) {
    let reason_str = match raw_dump {
        Some(failure) => format!(
            "{} — {reason}\n\n{}",
            load_prompt("pipeline/sanitation_failed.md"),
            raw_response_dump_section(failure)
        ),
        None => format!(
            "{} — {reason}",
            load_prompt("pipeline/sanitation_failed.md")
        ),
    };
    if let Err(e) = crate::turso::with_tx(
        &board().conn,
        ticket_id,
        "record sanitation failure",
        async |tx| {
            BoardStore::add_comment_tx(tx, ticket_id, SANITATION_ROLE, &reason_str).await?;
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

/// Register an agent in the message router and persist the SAME ID in the
/// ticket's `assigned_to` so mid-run comments route to it.
///
/// Returns the agent's inbox receiver, ready for [`run_agent`].
///
/// # Ordering invariant (register-before-set)
///
/// The router registration MUST happen before `assigned_to` is updated —
/// routing looks the stored ID up in the router, so persisting an ID that is
/// not yet registered would create a drop window.
///
/// Uses [`BoardStore::set_assigned_to_no_cancel`] (the non-cancelling
/// variant): the cancel-variant side-effect `AGENT_REGISTRY.cancel_by_ticket_id`
/// would kill a concurrently running agent for this ticket. That cancel is a
/// no-op at this point anyway —
/// no agent for this ticket is in `AGENT_REGISTRY` yet (agents enter it only
/// inside [`run_agent`], after assignment) and stale agents were already
/// cancelled pre-flight by [`spawn_dispatch`].
///
/// The warn message is a parameter because call sites describe the failure
/// differently.
async fn register_agent_and_assign(
    ticket_id: &str,
    agent_id: &str,
    warn_message: &str,
) -> tokio::sync::mpsc::UnboundedReceiver<crate::message_router::AgentJob> {
    // Register in the message router so comments can be delivered mid-work.
    let incoming_rx = message_router::register_agent(agent_id);

    if let Err(e) = board()
        .set_assigned_to_no_cancel(ticket_id, Some(agent_id))
        .await
    {
        warn!(
            ticket = ticket_id,
            error = %e,
            "{warn_message}",
        );
    }

    incoming_rx
}

/// Absorb the post-run tail shared by dispatch and resume: the phase/drain
/// guards, the response-None failure block, verdict extraction with error
/// handling, and the job terminalization. Guards: phase-moved → complete job
/// and bail; drain-cut (see [`stage_round_drain_cut`]) → leave the job
/// 'launched' for boot resume (fresh dispatches log, resumes stay silent);
/// response-None past the guards is a real failure.
async fn finalize_sanitation_round(
    ticket: &Ticket,
    agent: &Agent,
    response: Option<&str>,
    job_id: &str,
    resumed: bool,
) {
    if phase_changed_and_clear_assignment(&ticket.id, TicketPhase::InSanitation).await {
        complete_ticket_stage_job(job_id).await;
        return;
    }
    if stage_round_drain_cut(&ticket.id, "Sanitation", response, resumed) {
        return;
    }
    let resumed_suffix = if resumed { " (resumed)" } else { "" };
    if response.is_none() {
        // Agent failed or was cancelled — record failure and clear assigned_to
        // for re-dispatch retry. The marker comment lets the sanitation circuit
        // breaker detect repeated failures. (Past the guard above, response
        // None here is a real failure.)
        warn!(
            ticket = %ticket.id,
            "Sanitation agent returned no output{resumed_suffix} — clearing assigned_to for retry"
        );
        record_sanitation_failure(
            &ticket.id,
            format!("agent returned no output{resumed_suffix}"),
            None,
        )
        .await;
        complete_ticket_stage_job(job_id).await;
        return;
    }

    let extraction_prompt = crate::prompt::load_prompt("extraction/sanitation.md");
    match agent
        .extract_verdict::<crate::SanitationVerdict>(&extraction_prompt, None, None)
        .await
    {
        Ok(verdict) => process_sanitation_verdict(ticket, verdict).await,
        Err(failure) => {
            warn!(
                ticket = %ticket.id,
                error = %failure,
                "Failed to extract sanitation verdict{resumed_suffix} — clearing assigned_to for retry"
            );
            record_sanitation_failure(
                &ticket.id,
                format!("verdict extraction error{resumed_suffix}: {failure}"),
                Some(&failure),
            )
            .await;
        }
    }

    complete_ticket_stage_job(job_id).await;
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

    // Spawn the durable sanitation round BEFORE the session write — the job
    // is the resume handle. The agent ID is derived per-run from the job id
    // (`ticket_{job_id}_sanitation`); the roster row carries the
    // ACTUAL run id so the purge live-session protection and resume paths
    // match the real session.
    let job_id = crate::generate_id();
    let Ok(_slot) = spawn_single_slot_round(
        &job_id,
        &ticket,
        &ws,
        "sanitation",
        TicketPhase::InSanitation,
        Role::Sanitation,
        &prompt,
        &format!("ticket_{job_id}_sanitation"),
    )
    .await
    else {
        error!(
            ticket = %ticket.id,
            "Failed to spawn sanitation job — aborting dispatch",
        );
        return;
    };
    run_stage_agent_round(
        &ticket,
        &ws,
        &job_id,
        &prompt,
        false,
        StageRoundKind::Sanitation,
    )
    .await;
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
        comment_and_transition_or_bail(
            TransitionCtx::buffered(
                ticket,
                TicketPhase::InSanitation,
                TicketPhase::SanitationPassed,
                "Sanitation",
            ),
            Role::Sanitation.as_str(),
            &comment,
            "Sanitation passed — transitioned to SanitationPassed",
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
        // Pre-build the system comment so we can pass it into the transaction.
        let count_str = verdict.garbage_files.len().to_string();
        let sys_comment = substitute(
            &load_prompt("pipeline/sanitation_circuit_breaker_comment.md"),
            &[
                (
                    "{{sanitation_failed_marker}}",
                    &load_prompt("pipeline/sanitation_failed.md"),
                ),
                ("{{count}}", &count_str),
            ],
        );

        // Write both comments and transition atomically via
        // [`with_comment_and_transition`], which wraps all writes in a single
        // transaction. This matches the pattern used by all other verdict paths.
        if !matches!(
            with_comment_and_transition(
                TransitionCtx::buffered(
                    ticket,
                    TicketPhase::InSanitation,
                    TicketPhase::ReadyForDevelopment,
                    "Sanitation"
                ),
                async |tx| {
                    BoardStore::add_comment_tx(
                        tx,
                        &ticket.id,
                        Role::Sanitation.as_str(),
                        comment.as_str(),
                    )
                    .await?;
                    BoardStore::add_comment_tx(
                        tx,
                        &ticket.id,
                        SANITATION_ROLE,
                        sys_comment.as_str(),
                    )
                    .await?;
                    Ok(())
                },
            )
            .await,
            FinalizeOutcome::Applied
        ) {
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
/// Returns `(comment_text, all_passed)` where `comment_text` carries the
/// per-command lines and the appropriate pass/fail marker.
async fn run_diagnostics_commands(diag: &DiagnosticsCommands, ws: &Workspace) -> (String, bool) {
    let mut comment = String::new();
    let mut all_passed = true;
    let mut failed_at: &str = "";

    for (label, cmd_opt) in diag.commands() {
        let Some(cmd) = cmd_opt else {
            continue;
        };

        let started = std::time::Instant::now();
        match ShellTool::new(ShellMode::Full)
            .execute_with_status(ws, serde_json::json!({"command": cmd}))
            .await
        {
            Ok((_output, Some(0))) => {
                // Successful commands carry a compact one-line summary — the
                // full output is noise in the ticket.
                let _ = write!(
                    comment,
                    "\n\n{label} ({cmd}): PASSED in {:.1}s",
                    started.elapsed().as_secs_f64(),
                );
            }
            Ok((output, _exit_code)) => {
                // Failed command: keep the whole output exactly as before
                // (including stderr and exit details).
                let _ = write!(comment, "\n\n{label} ({cmd}):\n");
                let display = if output.is_empty() {
                    "(no output)".to_string()
                } else {
                    output
                };
                // Output is already credential-scrubbed by ShellTool's output
                // pipeline at pipeline entry, so no further scrubbing needed.
                comment.push_str(&display);
                all_passed = false;
                failed_at = label;
                break;
            }
            Err(e) => {
                // Timeout or process launch failure.
                let _ = write!(comment, "\n\n{label} ({cmd}):\n");
                comment.push_str(&e.to_string());
                all_passed = false;
                failed_at = label;
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

    // The removed "Auto-diagnostics" header would leave the first command's
    // separator as leading blank lines — strip them.
    // Owner-deletes-at-end: the diagnostics runner is a NON-AGENT consumer of
    // ShellTool, so its spill files are recorded under the non-agent owner key —
    // clean them up here (equivalent to the agent-run-end hook).
    crate::tools::shell::cleanup_agent_spills(crate::tools::shell::NON_AGENT_SPILL_OWNER);
    (comment.trim_start_matches('\n').to_string(), all_passed)
}

/// Run diagnostics commands after the engineer completes development.
///
/// Called by [`PollPhase::DiagnosticsCheck`] via [`spawn_dispatch`].
/// The circuit-breaker guard is handled centrally there, consistent with all
/// other dispatchers. Uses [`BoardStore::claim_diagnostics`] to set
/// `assigned_to` and prevent
/// double-dispatch. Unlike the pipeline-phase dispatchers (which are dispatched
/// from the atomic claim loop and already own the ticket by the time their
/// dispatch runs), diagnostics keeps the ticket in `InDiagnostics` while
/// executing, so a separate atomic claim is needed to close the TOCTOU window.
/// Loads discovered diagnostics commands for the workspace and runs them
/// sequentially via [`run_diagnostics_commands`]. Stops at the first failure.
/// After execution, transitions the ticket to either `DiagnosticsDone` (all
/// passed) or `ReadyForDevelopment` (any failure).
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

    // Separate the decision (target phase + comment body + outcome log) from
    // the action (single transition call), matching the dispatch_engineer
    // precedent.
    let (target_phase, comment_body, outcome_log): (TicketPhase, String, &str) =
        match crate::workspace::store().get_diagnostics(&ws.name).await {
            Ok(Some(cmds)) if !cmds.is_empty() => {
                // Run commands sequentially in the prescribed order.
                let (comment, all_passed) = run_diagnostics_commands(&cmds, &ws).await;

                // Post-run check: verify ticket hasn't been moved externally while
                // diagnostics commands ran.
                if phase_changed_and_clear_assignment(&ticket.id, TicketPhase::InDiagnostics).await
                {
                    return;
                }

                if all_passed {
                    // Path C1: All diagnostics passed — transition to DiagnosticsDone.
                    (
                        TicketPhase::DiagnosticsDone,
                        comment,
                        "Diagnostics finished — transitioned ticket",
                    )
                } else {
                    // Path C2: Diagnostics failed — bounce back to development.
                    (
                        TicketPhase::ReadyForDevelopment,
                        comment,
                        "Diagnostics failed — transitioned ticket",
                    )
                }
            }
            Ok(_) => {
                // Path B: No diagnostics commands configured (or empty list) — skip.
                (
                    TicketPhase::DiagnosticsDone,
                    "No diagnostics commands are configured for this workspace \
                     — diagnostics skipped."
                        .to_string(),
                    "Diagnostics skipped — transitioned ticket",
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
                    "Diagnostics failed — transitioned ticket",
                )
            }
        };

    comment_and_transition_or_bail(
        TransitionCtx::buffered(
            &ticket,
            TicketPhase::InDiagnostics,
            target_phase,
            "Diagnostics",
        ),
        DIAGNOSTICS_ROLE,
        &comment_body,
        outcome_log,
    )
    .await;
}

// ── Parallel agent helpers (shared) ─────────────────────────────────────
//
// Why `process_analyst_verdicts` and `process_verifier_verdicts` are separate
// Both follow the same skeleton (joint comment -> classify -> transition)
// but differ in three ways that make a single unified function awkward:
//
//   * Classification — analysts use 4 categories
//     (lgtm/minor_issues/potential_blockers/missing_analysis) that feed
//     `format_analyst_summary`; reviewers/QA use a binary pass/fail via
//     `verdict_passes` against `REVIEW_QA_THRESHOLD`.
//   * Transition policy — analysts always advance to `Planning` regardless
//     of outcome (fail-open; the joint comment gives the Manager depth).
//     Reviewers/QA have a 3-way outcome: all-failed -> Failed,
//     any-failed -> bounce back to development (bounce counter bumped,
//     11th bounce fails), all-pass -> success phase.
//   * Signature — analysts need only `&Ticket` and `&[ParallelVerdict]`;
//     reviewers/QA need the `VerifierInfo` struct to drive the 3-way
//     transition (success phase, active phase, role label). This structural
//     difference alone prevents a shared function signature without closures.

/// Result from a single parallel verifier agent.
///
/// Three mutually-exclusive states — the type system guarantees
/// that "no response" and "parse failure" cannot be confused.
#[derive(Clone)]
pub(crate) enum ParallelVerdict {
    /// Agent failed to produce any response (crashed, timed out, empty output).
    /// Carries the collapsed failure reason (scrubbed) so the cause survives
    /// through to the ticket failure record and the log.
    NoResponse(String),
    /// Agent produced a response but structured verdict extraction failed
    /// after exhausting the hardened retry loop. Carries the
    /// [`crate::retry::RetryExhausted`] so the raw last-attempt response can be
    /// dumped into the ticket comment.
    ParseFailed(crate::retry::RetryExhausted),
    /// Agent produced a successfully-parsed verdict.
    Verdict(crate::Verdict),
}

/// Reject verdict scores outside [0,10] — a garbage score must never pass any
/// gate. Runs fail-closed inside the extraction retry
/// loop: rejection is a parse failure and triggers the re-prompt.
fn validate_verdict_score(v: &crate::Verdict) -> Result<(), String> {
    if v.score <= 10 {
        Ok(())
    } else {
        Err(format!("verdict score {} out of range [0,10]", v.score))
    }
}

/// Human-readable stage name for a parallel-verdict role (used in the joint
/// comment title and the comment role).
#[must_use]
fn stage_name(role: Role) -> &'static str {
    match role {
        Role::Analyst => "Analysis",
        Role::Reviewer => "Review",
        Role::Qa => "QA",
        // Only the three parallel-verdict roles reach this function (all call
        // sites pass Analyst/Reviewer/Qa).
        _ => unreachable!("stage_name called with a non-verdict role"),
    }
}

/// Inverse of [`stage_name`]: resolve a stage-name comment role back to the
/// verdict role (used by the GUI to color joint-comment badges).
#[must_use]
pub(crate) fn stage_role(name: &str) -> Option<Role> {
    match name {
        "Analysis" => Some(Role::Analyst),
        "Review" => Some(Role::Reviewer),
        "QA" => Some(Role::Qa),
        _ => None,
    }
}

/// Build the joint comment for a round: deterministic merge + single LLM
/// synthesis pass (stage role's own model) + rendering. Runs entirely before
/// any board transaction — the synthesis must never hold the board write lock.
///
/// `ticket_title` is threaded into the synthesis call's Running Agents group
/// header label (purely presentational — the ticket group keeps its name even
/// when only the synthesis call remains after the round's agents deregistered).
#[expect(clippy::too_many_arguments)]
async fn build_round_joint_comment(
    stage: &str,
    results: &[ParallelVerdict],
    threshold: u8,
    role: Role,
    header: &str,
    ws: &Workspace,
    ticket_id: &str,
    ticket_title: &str,
) -> String {
    let mut verdicts: Vec<crate::joint_verdict::JointVerdict<'_>> = Vec::new();
    let mut failures: Vec<crate::joint_verdict::JointFailure> = Vec::new();
    for (i, r) in results.iter().enumerate() {
        match r {
            ParallelVerdict::Verdict(v) => verdicts.push(crate::joint_verdict::JointVerdict {
                agent_index: i,
                verdict: v,
            }),
            ParallelVerdict::NoResponse(reason) => {
                failures.push(crate::joint_verdict::JointFailure {
                    agent_index: i,
                    dump: reason.clone(),
                });
            }
            ParallelVerdict::ParseFailed(f) => failures.push(crate::joint_verdict::JointFailure {
                agent_index: i,
                dump: crate::util::scrub_credentials(&raw_response_dump_section(f)),
            }),
        }
    }
    let round = crate::joint_verdict::JointRound {
        stage,
        dispatched: results.len(),
        verdicts,
        failures,
        header: header.to_string(),
        threshold,
    };
    if round
        .verdicts
        .iter()
        .all(|v| v.verdict.issues_detected.is_empty())
    {
        // No issues to merge: either all agents failed (no response / parse
        // failure) or every valid verdict reported an empty issues list.
        // A synthesis pass over zero issues would waste a provider call and
        // could produce a misleading "no issues" summary on a round that
        // actually bounced. Render the deterministic no-issues form instead.
        crate::joint_verdict::render_joint_comment(
            &round,
            &crate::consensus::RepairOutcome::Fallback,
            &crate::consensus::ItemTable::new(&crate::joint_verdict::issues_by_agent(&round)),
        )
    } else {
        crate::joint_verdict::build_joint_comment(&round, role, ws, ticket_id, ticket_title).await
    }
}

/// Load per-agent angle supplements for a verifier role (review_angles.md,
/// qa_angles.md). Missing or malformed assets degrade to no supplements —
/// dispatch then uses today's identical shared prompt.
fn load_verifier_angles(role: Role) -> Vec<String> {
    match role {
        Role::Reviewer => load_prompt_sections("review_angles.md"),
        Role::Qa => load_prompt_sections("qa_angles.md"),
        _ => Vec::new(),
    }
}

/// Run `count` agents of the same role in parallel, then extract structured verdicts
/// from their responses.
///
/// Roster slots are pre-built by the caller ([`spawn_ticket_stage_round`] for
/// fresh dispatch, [`append_ticket_stage_slots`] for analysis escalation). The
/// agent-id format and the angle-cycling rule are canonicalized in
/// [`ticket_stage_agent_id`] and [`ticket_stage_slot_task`] respectively —
/// both call sites must go through them so the two rules cannot drift. Each
/// agent creates its own CancellationToken and auto-registers. KV-cache
/// discipline: the variation is limited to the user message — model,
/// reasoning effort, and tools stay identical across agents.
///
/// Agents with empty responses get [`ParallelVerdict::NoResponse`]; agents that
/// respond but fail to parse get [`ParallelVerdict::ParseFailed`]; successful
/// agents get [`ParallelVerdict::Verdict`]. All extraction attempts run
/// concurrently (leader-staggered: the first member starts immediately, the
/// rest after its first LLM call so they hit its cached prefix; skipped on
/// boot resume).
#[expect(clippy::too_many_lines)]
async fn run_parallel_agents(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    role: Role,
    extraction_prompt: &str,
    job_id: &str,
    slots: &[TicketStageSlot],
    resume: bool,
) -> Vec<ParallelVerdict> {
    // ── Register the not-yet-done agents in the message router BEFORE
    // spawning ── This allows the board's comment-routing to deliver
    // mid-work comments to any of these agents. Done slots (resumed rounds)
    // are never re-invoked — their outcomes are read from the roster.
    let launched: Vec<&TicketStageSlot> = slots
        .iter()
        .filter(|s| s.status != crate::jobs::RowStatus::Done)
        .collect();
    let receivers: Vec<_> = launched
        .iter()
        .map(|s| message_router::register_agent(&s.agent_id))
        .collect();

    // Set assigned_to to the launched agent IDs (no cancellation — agents are
    // already registered and running would be cancelled).
    let assigned_to_str = launched
        .iter()
        .map(|s| s.agent_id.as_str())
        .collect::<Vec<_>>()
        .join(",");
    if !assigned_to_str.is_empty()
        && let Err(e) = board()
            .set_assigned_to_no_cancel(&ticket.id, Some(&assigned_to_str))
            .await
    {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Failed to set assigned_to for parallel agents",
        );
    }

    // ── Spawn and run all launched agents (leader-staggered) ───────────
    // Members are spawned (not joined inline): a panicked/cancelled member
    // resolves to ParallelVerdict::NoResponse below — the round continues
    // fail-open and the member checkpoints Failed, matching the analyze/research
    // round semantics.
    let mut results: Vec<ParallelVerdict> = Vec::with_capacity(slots.len());
    {
        let members: Vec<_> = launched
            .iter()
            .zip(receivers)
            .map(|(slot, rx)| {
                let ticket = Arc::clone(ticket);
                let ws = ws.clone();
                let extraction_prompt = extraction_prompt.to_string();
                let agent_id = slot.agent_id.clone();
                let task = slot.task.clone();
                move |round: crate::agent::RoundOpts| async move {
                    // Mid-stagger-wait cancellation window: followers start
                    // up to the stagger bound after the leader, so a phase
                    // move (Manager interrupt, bounce, transition) during the
                    // wait cancels only the already-registered leader. Gate
                    // members on the live phase so a moved ticket does not
                    // burn a full round (the post-round phase gate would
                    // discard the verdicts anyway). is_ticket_in_phase also
                    // fails closed on DB errors / missing rows — the reason
                    // string stays neutral so transients are not misattributed.
                    // A move-away-and-back between the leader's gate failure
                    // and a follower's check lets that follower run a full
                    // round with N-1 members — rare, fail-open, accepted.
                    if !is_ticket_in_phase(&ticket.id, ticket.phase).await {
                        // Release the stagger wait immediately: the ticket
                        // moved, so every follower bails at its own gate too —
                        // don't make them sit out the bound. Then drop the
                        // router entry (run_agent's exit guard never runs on
                        // this path).
                        if let Some(notify) = &round.first_call_notify {
                            notify.notify_one();
                        }
                        message_router::unregister_agent(&agent_id);
                        return ParallelVerdict::NoResponse(PHASE_GATE_BAIL_REASON.to_string());
                    }
                    // Session-non-emptiness discriminator (resume dispatch
                    // rule): a resumed slot whose session already contains the
                    // task continues with an empty message (no duplicate
                    // task-prompt append); a missing/empty session dispatches
                    // fresh with the stored task (covers a crash between
                    // roster-write and the first session write).
                    let has_session =
                        resume && crate::session::store().has_content(&agent_id).await;
                    let (agent, response) = run_agent(
                        agent_id.clone(),
                        role,
                        &ws,
                        Some(&ticket),
                        if has_session { "" } else { &task },
                        String::new(),
                        String::new(),
                        Some(rx),
                        resume,
                        Some(round),
                        None,
                        None,
                    )
                    .await;

                    // run_agent's exit guard unregisters from the message
                    // router on every path (incl. panic) — the caller
                    // registered this member via Some(rx); the phase-gate
                    // bail above unregisters explicitly since run_agent
                    // never runs there.
                    let response = response.unwrap_or_default();
                    if response.is_empty() {
                        // Preserve the failure reason (full chain, scrubbed) so
                        // the all-failed record and log carry the real cause
                        // instead of a generic "no response".
                        let reason = agent.failure_reason("agent produced no response");
                        ParallelVerdict::NoResponse(crate::util::scrub_credentials(&reason))
                    } else {
                        // Prefix-cache preservation: `agent.extract_verdict`
                        // uses the agent's own parameters (model,
                        // reasoning_effort, tools, provider routing) with no
                        // response_format override — the extraction request
                        // shares the agent-loop prefix (system + history), so
                        // only the appended extraction prompt misses the cache.
                        //
                        // The hardened outer retry loop (13 attempts,
                        // backoff 5/10/20/40/60/60 s, 720 s wall cap) enforces
                        // fail-closed score validation. On terminal
                        // failure the RetryExhausted (carrying the last-attempt raw
                        // text) flows into ParallelVerdict::ParseFailed for the ticket
                        // comment.
                        let verdict = agent
                            .extract_verdict::<crate::Verdict>(
                                &extraction_prompt,
                                Some(&validate_verdict_score),
                                None,
                            )
                            .await;
                        match verdict {
                            Ok(v) => ParallelVerdict::Verdict(v),
                            Err(e) => ParallelVerdict::ParseFailed(e),
                        }
                    }
                }
            })
            .collect();
        let handles = crate::agent::spawn_staggered_round(members, resume).await;
        let mut run_results: Vec<ParallelVerdict> = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(v) => run_results.push(v),
                Err(e) => run_results.push(round_member_failed(e)),
            }
        }

        // ── Checkpoint: write per-agent outcomes BEFORE ParallelVerdict
        // construction completes (no read-modify-write race). ──
        let conn = &crate::session::store().conn;
        let mut by_agent: std::collections::HashMap<&str, &ParallelVerdict> =
            std::collections::HashMap::new();
        for (slot, result) in launched.iter().zip(&run_results) {
            let outcome = serialize_verdict_outcome(result);
            let status = if matches!(result, ParallelVerdict::NoResponse(_)) {
                crate::jobs::RowStatus::Failed
            } else {
                crate::jobs::RowStatus::Done
            };
            if let Err(e) = crate::jobs::write_agent_outcome(
                conn,
                job_id,
                &slot.agent_id,
                status,
                Some(&outcome),
            )
            .await
            {
                warn!(
                    job = %job_id,
                    agent = %slot.agent_id,
                    error = %e,
                    "Failed to checkpoint agent outcome",
                );
            }
            by_agent.insert(slot.agent_id.as_str(), result);
        }

        // ── Assemble results in dispatch order: done slots replay their
        // stored outcome; launched slots use the fresh run. ──
        for slot in slots {
            if slot.status == crate::jobs::RowStatus::Done {
                let outcome = slot.outcome.clone().unwrap_or_default();
                results.push(deserialize_verdict_outcome(&outcome));
            } else if let Some(r) = by_agent.get(slot.agent_id.as_str()) {
                results.push((*r).clone());
            } else {
                unreachable!("every non-Done slot is launched and recorded 1:1 in by_agent");
            }
        }
    }

    // ── Cleanup: clear assigned_to (no cancel, agents already done) ────
    if let Err(e) = board().set_assigned_to_no_cancel(&ticket.id, None).await {
        warn!(
            ticket = %ticket.id,
            error = %e,
            "Failed to clear assigned_to after parallel agents",
        );
    }

    results
}

/// Map a panicked round member's [`tokio::task::JoinError`] to a contained
/// [`ParallelVerdict::NoResponse`] (round continues fail-open) carrying the
/// scrubbed panic message as the reason. Cancelled handles are impossible —
/// nothing aborts these tasks; `into_panic` panics loudly on that unreachable
/// case (surfaced via the dispatch catch_unwind; resume paths log-and-die).
/// If a deadline-abort is ever added (the analyze/research `await_round_members`
/// precedent), restore an `is_cancelled()` mapping before `into_panic`.
fn round_member_failed(e: tokio::task::JoinError) -> ParallelVerdict {
    let reason = crate::util::scrub_credentials(&crate::util::panic_message(&*e.into_panic()));
    tracing::warn!(%reason, "round member task failed");
    ParallelVerdict::NoResponse(reason)
}

/// Check whether a review or QA verdict passes (score at or above
/// [`REVIEW_QA_THRESHOLD`]).
#[must_use]
fn verdict_passes(verdict: &crate::Verdict) -> bool {
    verdict.score >= REVIEW_QA_THRESHOLD
}
/// Build the raw-response dump section for a verdict-extraction failure
/// comment.
///
/// - `Some(non-empty)` → sandwich-truncated raw text in a code fence.
/// - `Some(empty)` → explicit "final attempt was a tool call" note.
/// - `None` → transport/truncation final failure: per-attempt trail +
///   classification (no in-loop text exists to dump).
///
/// No scrubbing/markdown-escaping is applied — analyst reports are already
/// written unescaped on the comment path (pre-existing condition, out of
/// scope).
fn raw_response_dump_section(failure: &crate::retry::RetryExhausted) -> String {
    match failure.last_raw.as_deref() {
        Some(text) if !text.trim().is_empty() => format!(
            "Raw agent response (last attempt):\n```\n{}\n```",
            crate::util::truncate_sandwich(
                text,
                crate::util::FAILURE_DETAIL_CAP,
                "verdict response"
            )
        ),
        Some(_) => "Final attempt was a tool call — no text response produced.".to_string(),
        None => format!(
            "Extraction failed after {} attempt(s) — final failure: {} ({})",
            failure.failures.len(),
            failure.final_class.label(),
            failure.detail,
        ),
    }
}

// ── Ticket-stage job roster (durable round record) ─────────────────────

/// One roster slot of a ticket_stage round. Roster rows ARE the round record:
/// dispatched_count = roster size; escalation appends slots 3,4 (re-evaluated
/// only when roster size == 3); replay re-runs missing agents FIRST, then
/// re-processes when all outcomes present.
struct TicketStageSlot {
    /// Dispatch slot index (0-based; escalation continues at 3, 4).
    idx: i64,
    agent_id: String,
    /// FINAL per-agent rendered prompt (angle appended) — makes replay exact
    /// regardless of the angle formula or slot numbering.
    task: String,
    status: crate::jobs::RowStatus,
    /// Stored agents.outcome (tagged JSON) — set on replay of a done slot.
    outcome: Option<String>,
}

/// Serialize a [`ParallelVerdict`] into the agents.outcome column. Tagged JSON
/// keeps the three variants distinct so replay reconstruction is exact.
fn serialize_verdict_outcome(result: &ParallelVerdict) -> String {
    match result {
        ParallelVerdict::Verdict(v) => serde_json::json!({ "verdict": v }).to_string(),
        ParallelVerdict::NoResponse(reason) => {
            serde_json::json!({ "no_response": reason }).to_string()
        }
        ParallelVerdict::ParseFailed(f) => {
            serde_json::json!({ "parse_failed": raw_response_dump_section(f) }).to_string()
        }
    }
}

/// Reconstruct a [`ParallelVerdict`] from the agents.outcome column.
/// Round-trip lossless for Verdict; ParseFailed degrades to NoResponse
/// carrying the raw dump section (cosmetic — both are non-passing in the
/// verdict processors).
fn deserialize_verdict_outcome(outcome: &str) -> ParallelVerdict {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(outcome) else {
        return ParallelVerdict::NoResponse("unreadable stored outcome".to_string());
    };
    if let Some(verdict) = v.get("verdict") {
        match serde_json::from_value(verdict.clone()) {
            Ok(v) => ParallelVerdict::Verdict(v),
            Err(_) => ParallelVerdict::NoResponse("unreadable stored verdict".to_string()),
        }
    } else if let Some(r) = v.get("no_response").and_then(serde_json::Value::as_str) {
        ParallelVerdict::NoResponse(r.to_string())
    } else if let Some(p) = v.get("parse_failed").and_then(serde_json::Value::as_str) {
        ParallelVerdict::NoResponse(p.to_string())
    } else {
        ParallelVerdict::NoResponse("unrecognized stored outcome".to_string())
    }
}

/// Map a verdict role to the agents.kind vocabulary.
const fn agent_kind_for_role(role: Role) -> crate::jobs::AgentKind {
    match role {
        Role::Reviewer | Role::Qa => crate::jobs::AgentKind::Verifier,
        Role::Engineer => crate::jobs::AgentKind::Engineer,
        Role::Sanitation => crate::jobs::AgentKind::Sanitation,
        _ => crate::jobs::AgentKind::Analyst,
    }
}

/// Next round number for a (ticket_id, stage) pair — MAX(round) + 1.
async fn next_ticket_stage_round(
    conn: &crate::turso::Connection,
    ticket_id: &str,
    stage: &str,
) -> i64 {
    conn.query_row(
        "SELECT COALESCE(MAX(round), 0) + 1 FROM ticket_stage_jobs WHERE ticket_id = ?1 AND stage = ?2",
        crate::turso::params![ticket_id, stage],
        |row| row.get::<i64>(0),
    )
    .await
    .unwrap_or(1)
}

/// Canonical agent-id format for ticket_stage roster slots
/// (`ticket_{ticket_id}_{idx}_{suffix}_{role}`).
///
/// `idx` is the slot's GLOBAL index within the job (base round: the 0-based
/// per-slot index; escalation: the continuation index 3, 4). `suffix` is a
/// unique 6-char NanoID generated per dispatch batch for retry-cycle
/// disambiguation — a fresh suffix per batch guarantees escalation slots can
/// never collide with base-round ids, even across crash/resume cycles.
///
/// This is the single home for the id contract shared by
/// [`spawn_ticket_stage_round`] (fresh dispatch) and
/// [`append_ticket_stage_slots`] (analysis escalation) — the two must never
/// drift. Resume paths read the stored agent ids straight from the roster and
/// only match the `ticket_` prefix, so the exact shape is convention, not a
/// parse contract.
#[must_use]
fn ticket_stage_agent_id(ticket_id: &str, idx: i64, suffix: &str, role: Role) -> String {
    format!("ticket_{}_{}_{}_{}", ticket_id, idx, suffix, role.as_str())
}

/// Render the FINAL per-agent task for a ticket_stage roster slot — the
/// canonical angle-cycling rule shared by [`spawn_ticket_stage_round`] and
/// [`append_ticket_stage_slots`].
///
/// `angles` are the per-agent angle supplements from [`load_verifier_angles`]
/// (empty for non-verifier roles → the shared prompt is used untouched).
/// `slot_count` is the TOTAL number of slots in the job; `global_idx` is the
/// slot's index within the job (base round: `i`; escalation: `roster_len + k`).
///
/// - no angles → bare `prompt`;
/// - `slot_count == 1` (single-agent round, e.g. QA's lone tester) →
///   concatenate ALL angle sections;
/// - otherwise → `prompt` + the angle section selected by
///   `global_idx % angles.len()` (defensive cycling; reviewers max at 4).
///
/// KV-cache discipline: the variation is limited to the user message — model,
/// reasoning effort, and tools stay identical across agents.
#[must_use]
fn ticket_stage_slot_task(
    prompt: &str,
    angles: &[String],
    slot_count: usize,
    global_idx: usize,
) -> String {
    if angles.is_empty() {
        prompt.to_string()
    } else if slot_count == 1 {
        format!("{prompt}\n\n{}", angles.join("\n\n"))
    } else {
        format!("{prompt}\n\n{}", angles[global_idx % angles.len()])
    }
}

/// Spawn a ticket_stage job + roster in ONE transaction (before any agent
/// session write). The job row carries the rendered prompt template; each
/// roster row carries the FINAL per-agent prompt. Returns the job id + slots.
async fn spawn_ticket_stage_round(
    ticket: &Ticket,
    ws: &Workspace,
    stage: &'static str,
    phase: TicketPhase,
    role: Role,
    prompt: &str,
    count: usize,
) -> anyhow::Result<(String, Vec<TicketStageSlot>)> {
    let job_id = crate::generate_id();
    let suffix = crate::generate_suffix();
    let angles = load_verifier_angles(role);
    let mut slots = Vec::with_capacity(count);
    for i in 0..count {
        let idx = i64::try_from(i).unwrap_or(i64::MAX);
        let agent_id = ticket_stage_agent_id(&ticket.id, idx, &suffix, role);
        let task = ticket_stage_slot_task(prompt, &angles, count, i);
        slots.push(TicketStageSlot {
            idx,
            agent_id,
            task,
            status: crate::jobs::RowStatus::Launched,
            outcome: None,
        });
    }
    let agents: Vec<crate::jobs::NewAgent> = slots
        .iter()
        .map(|s| crate::jobs::NewAgent {
            agent_id: s.agent_id.clone(),
            kind: agent_kind_for_role(role),
            idx: Some(s.idx),
            task: s.task.clone(),
        })
        .collect();
    // Round + child row computed BEFORE the spawn tx: the ticket_stage_jobs
    // child row is inserted in the SAME tx as the job (all kinds use the
    // shared in-tx child pattern — a missing child row is impossible for a
    // committed job).
    let round = next_ticket_stage_round(&crate::session::store().conn, &ticket.id, stage).await;
    crate::jobs::spawn_job(
        &crate::session::store().conn,
        &job_id,
        prompt,
        &ws.name,
        "",
        "",
        role,
        &agents,
        &crate::jobs::SpawnChild::TicketStage {
            ticket_id: ticket.id.clone(),
            stage: stage.to_string(),
            phase: phase.as_ref().to_string(),
            round,
        },
    )
    .await?;
    Ok((job_id, slots))
}

/// Spawn a single-slot ticket_stage job whose roster row carries the EXACT
/// run agent_id (engineer: the NULL-seat anchor id; sanitation: the
/// job-derived `ticket_{job_id}_sanitation`). Roster rows must reference the
/// ACTUAL session — phantom generated ids would never be checkpointed, would
/// be ignored by resume paths, and would let the purge's live-session
/// protection clause miss an active round. The anchor's NULL-seat row and this
/// round's roster row coexist under the composite PK (job_id, agent_id).
#[expect(clippy::too_many_arguments)]
async fn spawn_single_slot_round(
    job_id: &str,
    ticket: &Ticket,
    ws: &Workspace,
    stage: &'static str,
    phase: TicketPhase,
    role: Role,
    prompt: &str,
    agent_id: &str,
) -> anyhow::Result<TicketStageSlot> {
    let slot = TicketStageSlot {
        idx: 0,
        agent_id: agent_id.to_string(),
        task: prompt.to_string(),
        status: crate::jobs::RowStatus::Launched,
        outcome: None,
    };
    crate::jobs::spawn_job(
        &crate::session::store().conn,
        job_id,
        prompt,
        &ws.name,
        "",
        "",
        role,
        &[crate::jobs::NewAgent {
            agent_id: agent_id.to_string(),
            kind: agent_kind_for_role(role),
            idx: Some(0),
            task: prompt.to_string(),
        }],
        &crate::jobs::SpawnChild::TicketStage {
            ticket_id: ticket.id.clone(),
            stage: stage.to_string(),
            phase: phase.as_ref().to_string(),
            round: next_ticket_stage_round(&crate::session::store().conn, &ticket.id, stage).await,
        },
    )
    .await?;
    Ok(slot)
}

/// Append escalation slots (3, 4) to an existing ticket_stage job.
async fn append_ticket_stage_slots(
    ticket: &Ticket,
    job_id: &str,
    role: Role,
    prompt: &str,
    count: usize,
) -> anyhow::Result<Vec<TicketStageSlot>> {
    let roster = crate::jobs::list_agents_for_job(&crate::session::store().conn, job_id).await?;
    let roster_len = roster.len();
    let next_idx = i64::try_from(roster_len).unwrap_or(i64::MAX);
    let suffix = crate::generate_suffix();
    let angles = load_verifier_angles(role);
    // The angle rule sees the job as a whole: slot_count is the TOTAL roster
    // size (base + escalation) and the global index (roster_len + k) keeps
    // angle selection continuous with the base-round slots.
    let slot_count = roster_len + count;
    let mut slots = Vec::with_capacity(count);
    for (k, i) in (next_idx..next_idx + i64::try_from(count).unwrap_or(i64::MAX)).enumerate() {
        let agent_id = ticket_stage_agent_id(&ticket.id, i, &suffix, role);
        let task = ticket_stage_slot_task(prompt, &angles, slot_count, roster_len + k);
        crate::session::store()
            .conn
            .execute(
                crate::jobs::AGENT_INSERT_SQL,
                crate::jobs::agent_params(
                    job_id,
                    &agent_id,
                    agent_kind_for_role(role),
                    Some(i),
                    &task,
                ),
            )
            .await?;
        slots.push(TicketStageSlot {
            idx: i,
            agent_id,
            task,
            status: crate::jobs::RowStatus::Launched,
            outcome: None,
        });
    }
    Ok(slots)
}

/// Terminalize a ticket_stage job AFTER the board transition+comment ran
/// (ordering contract: jobs-first would strand tickets — job done + ticket
/// still in a blocking phase → excluded from boot reset, never rolled back).
async fn complete_ticket_stage_job(job_id: &str) {
    if let Err(e) =
        crate::jobs::complete_ticket_stage_job(&crate::session::store().conn, job_id).await
    {
        warn!(job = %job_id, error = %e, "Failed to terminalize ticket_stage job");
    }
}

/// Load the roster of a ticket_stage job from the agents table (boot resume).
async fn load_ticket_stage_slots(job_id: &str) -> anyhow::Result<Vec<TicketStageSlot>> {
    let rows = crate::jobs::list_agents_for_job(&crate::session::store().conn, job_id).await?;
    Ok(rows
        .into_iter()
        .map(|r| TicketStageSlot {
            idx: r.idx.unwrap_or(0),
            agent_id: r.agent_id,
            task: r.task,
            status: r.status.parse().unwrap_or(crate::jobs::RowStatus::Launched),
            outcome: r.outcome,
        })
        .collect())
}

// ── Backlog Analysis ──────────────────────────────────────────────────

/// Escalate a unanimous-blocker analysis round with 2 extra analysts on the
/// SAME job, extending `results`. Returns false when the ticket left the
/// Analysis phase — the job was already completed; the caller must return.
///
/// Roster gate: escalation is re-evaluated only at base roster size. One
/// verdict per slot by construction of [`run_parallel_agents`], so
/// `results.len() == slots.len()`; the base roster is
/// `DEFAULT_PARALLEL_AGENT_COUNT` by construction of
/// [`spawn_ticket_stage_round`]. On resume this guards against re-escalation
/// after a crash between the base round and the escalation batch. Skipped
/// during the graceful drain — no new jobs are spawned while draining.
#[must_use]
async fn maybe_escalate_analysis(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    job_id: &str,
    extraction_prompt: &str,
    task: &str,
    resume: bool,
    results: &mut Vec<ParallelVerdict>,
) -> bool {
    if results.len() == DEFAULT_PARALLEL_AGENT_COUNT
        && crate::joint_verdict::analysis_escalation_needed(results, DEFAULT_PARALLEL_AGENT_COUNT)
        && !crate::shutdown::aborting()
    {
        // Re-check the phase before spending the second batch — the ticket
        // may have been moved externally while the first batch ran.
        if complete_job_and_bail_if_phase_moved(&ticket.id, TicketPhase::Analysis, job_id).await {
            return false;
        }
        if resume {
            info!(ticket = %ticket.id, job = %job_id, "Resume: escalating with 2 additional analysts");
        } else {
            info!(ticket = %ticket.id, "All analysts flagged blockers — escalating with 2 additional analysts");
        }
        let extra_slots = match append_ticket_stage_slots(ticket, job_id, Role::Analyst, task, 2)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                if resume {
                    warn!(job = %job_id, error = %e, "Resume: escalation slot append failed — proceeding");
                } else {
                    warn!(ticket = %ticket.id, error = %e, "Failed to append escalation slots — proceeding with base round");
                }
                Vec::new()
            }
        };
        let extra = if extra_slots.is_empty() {
            Vec::new()
        } else {
            run_parallel_agents(
                ticket,
                ws,
                Role::Analyst,
                extraction_prompt,
                job_id,
                &extra_slots,
                resume,
            )
            .await
        };
        results.extend(extra);
    }
    true
}

/// Spawn 3 parallel analyst agents (base) to research a backlog ticket,
/// escalating to 5 when every dispatched analyst flags blockers. One joint
/// comment (role = stage name "Analysis") replaces the per-agent comments and
/// the system summary; the ticket then transitions to Planning:
/// - Planning (notify) when ALL analysts pass (≥ `ANALYST_PASS_THRESHOLD`/10)
/// - Planning (notify) when any analyst fails, with the joint comment giving
///   the Manager the depth (analysis is fail-open)
///
/// The circuit-breaker guard is handled centrally by [`spawn_dispatch`].
///
/// ## Note: no `clear_assigned_to_no_cancel` on post-run phase check
///
/// Unlike [`dispatch_engineer`], [`dispatch_diagnostics`], and [`dispatch_sanitation`],
/// this function does **not** call [`clear_assigned_to_no_cancel`] when the post-run phase check
/// fails (the ticket moved externally during analysis). This is intentional:
///
/// * **`assigned_to` is already managed** — [`run_parallel_agents`] sets `assigned_to`
///   to the comma-separated agent IDs for mid-work comment routing, and clears it
///   after all agents finish. If the phase check fails (ticket moved externally),
///   the parallel agents are already done and unregistered — `assigned_to` was already
///   cleared or is about to be overwritten by the external transition.
/// * **Ephemeral agent IDs** — [`run_parallel_agents`] generates unique agent
///   IDs (`{base}_{i}_{suffix}`) that are written to the ticket's `assigned_to`
///   field for comment routing and then cleared after the agents finish.
///   The agent registry entries are already cleaned up.
/// * **TOCTOU race** — calling [`clear_assigned_to_no_cancel`] would unnecessarily risk
///   overwriting an assignee that a concurrent claim set between the phase check and
///   the clear. Since [`run_parallel_agents`] already clears `assigned_to` internally,
///   there is nothing to gain.
async fn dispatch_backlog_analysts(ticket: Arc<Ticket>, ws: Workspace) {
    let prompt_key = if ticket.reporter == Role::Maintainer.as_str() {
        "analyze/maintainer_ticket.md"
    } else {
        "analyze/manager_ticket.md"
    };
    let message = load_prompt(prompt_key);

    // Spawn the durable analysis round (jobs + roster, one tx) BEFORE the
    // agents' first session writes — the roster is the round record.
    let Ok((job_id, slots)) = spawn_ticket_stage_round(
        &ticket,
        &ws,
        "analysis",
        TicketPhase::Analysis,
        Role::Analyst,
        &message,
        DEFAULT_PARALLEL_AGENT_COUNT,
    )
    .await
    else {
        error!(
            ticket = %ticket.id,
            "Failed to spawn analysis job — aborting dispatch",
        );
        return;
    };
    run_analysis_round(&ticket, &ws, &job_id, &slots, &message, false).await;
}

/// Run an analysis round: parallel analysts, unanimous-blocker escalation,
/// then verdict finalization. Shared by fresh dispatch and boot resume so the
/// post-round tail stays in lockstep.
///
/// `task` is the message the agents were dispatched with — on fresh dispatch
/// the in-memory prompt, on resume re-read from `jobs.task`. Equal by
/// construction of [`spawn_ticket_stage_round`] (it stores the message as
/// `jobs.task`); if that ever changes, the resume path silently diverges.
///
/// Analysis stays fail-open — the ticket always advances to Planning, the
/// Manager decides. Escalation is re-evaluated ONLY at base roster size (see
/// [`maybe_escalate_analysis`]): on resume this guards against re-escalation
/// after a crash between the base round and the escalation batch. The drain
/// guard mirrors the fresh dispatch path.
///
/// Early-returns (without finalizing) when the ticket left the Analysis phase
/// — both callers end with this call, so nothing may follow it.
async fn run_analysis_round(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    job_id: &str,
    slots: &[TicketStageSlot],
    task: &str,
    resumed: bool,
) {
    let extraction_prompt = load_prompt("extraction/analyst.md");
    let mut results = run_parallel_agents(
        ticket,
        ws,
        Role::Analyst,
        &extraction_prompt,
        job_id,
        slots,
        resumed,
    )
    .await;
    if !maybe_escalate_analysis(
        ticket,
        ws,
        job_id,
        &extraction_prompt,
        task,
        resumed,
        &mut results,
    )
    .await
    {
        return;
    }
    finalize_analysis_round(ws, ticket, &results, job_id).await;
}

/// Evaluate analyst verdicts and transition the ticket:
///
/// One joint comment (role = "Analysis", the stage name) replaces the
/// per-agent comments and the system summary — written after every round,
/// including all-pass rounds (the audit trail is uniform). Analysis is
/// fail-open: the ticket always advances to Planning and the Manager decides;
/// the joint comment gives the Manager the depth.
///
/// See the "Parallel agent helpers (shared)" section for why this is separate
/// from [`process_verifier_verdicts`].
async fn process_analyst_verdicts(ws: &Workspace, ticket: &Ticket, results: &[ParallelVerdict]) {
    let nonempty_count = results
        .iter()
        .filter(|r| !matches!(r, ParallelVerdict::NoResponse(_)))
        .count();
    let dispatched = results.len();
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
            ParallelVerdict::NoResponse(_) | ParallelVerdict::ParseFailed(_) => {
                missing_analysis += 1;
            }
        }
    }

    let summary = format_analyst_summary(
        dispatched,
        lgtm,
        minor_issues,
        potential_blockers,
        missing_analysis,
    );
    let extracted_count = dispatched - missing_analysis;
    let passing_count = lgtm + minor_issues;

    // Gate against the actually-dispatched count (N_gate), never a hard-coded
    // 3: a missing/empty verdict is treated as non-passing — all dispatched
    // analysts must produce passing verdicts for the ticket to proceed.
    let all_passed = passing_count == dispatched;

    // No exit-time transition during the graceful drain: the
    // outcomes are checkpointed on the roster; the job stays status='launched'
    // and boot resume re-processes them. A drained analysis round must NOT
    // write a misleading joint comment or Notify the Manager.
    if crate::shutdown::aborting() {
        return;
    }

    // Synthesis runs BEFORE the transition and never holds the board write
    // lock (the comment+transition transaction serializes all board writes).
    // Always written — the audit trail needs a per-round stage comment even
    // on fully clean rounds.
    let joint_comment = build_round_joint_comment(
        stage_name(Role::Analyst),
        results,
        ANALYST_PASS_THRESHOLD,
        Role::Analyst,
        &summary,
        ws,
        &ticket.id,
        &ticket.title,
    )
    .await;

    if !matches!(
        with_comment_and_transition(
            TransitionCtx::notifying(
                ticket,
                TicketPhase::Analysis,
                TicketPhase::Planning,
                "Analyst"
            ),
            async |tx| {
                BoardStore::add_comment_tx(
                    tx,
                    &ticket.id,
                    stage_name(Role::Analyst),
                    &joint_comment,
                )
                .await?;
                Ok(())
            },
        )
        .await,
        FinalizeOutcome::Applied
    ) {
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
            "Backlog analysis incomplete — moved to planning ({nonempty_count}/{dispatched} responded, \
             {extracted_count} extracted, {passing_count} passed)",
        );
    }
}

/// Finalize an analysis round — shared by fresh dispatch and boot resume so
/// the paths stay in lockstep. Drain-cut runs both BEFORE and AFTER
/// [`process_analyst_verdicts`]: a pre-existing drain skips the verdict
/// processing (no new work during drain); any drain leaves the job
/// launched — mid-process for boot resume (no transition), after the
/// transition with the ticket in Planning (completed at boot).
async fn finalize_analysis_round(
    ws: &Workspace,
    ticket: &Ticket,
    results: &[ParallelVerdict],
    job_id: &str,
) {
    if complete_job_and_bail_if_phase_moved(&ticket.id, TicketPhase::Analysis, job_id).await {
        return;
    }
    if crate::shutdown::aborting() {
        return;
    }
    process_analyst_verdicts(ws, ticket, results).await;
    if crate::shutdown::aborting() {
        return;
    }
    complete_ticket_stage_job(job_id).await;
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
/// # Critical invariant: all breaker types trigger the drain
///
/// This function is called unconditionally after **any** circuit breaker trip
/// that reaches the pre-flight breaker path (Sanitation, Diagnostics) or the
/// review/QA bounce trip (the mid-round bounce breaker). This is intentional and
/// conservative: any breaker trip signals that the workspace may be in a dirty
/// or contaminated state. If the drain were narrowed to only some breaker
/// types, an unrelated `ReadyForDevelopment` ticket could auto-start while the
/// workspace still has unfinished changes from the failed ticket — leading to
/// cascading failures, wasted API credits, and confused agents.
///
/// The engineer hard-failure bounce trip is the deliberate exception: it does
/// NOT call this function — it pauses the workspace instead, which blocks new
/// ReadyForDevelopment claims while the pause is active (see
/// [`pause_workspace_on_failure`] and [`blocks_claim`]).
///
/// Draining every `ReadyForDevelopment` ticket forces the Manager to, in order:
///
/// 1. Triage the failed ticket first.
/// 2. Inspect the workspace state and deal with any uncommitted changes.
/// 3. Ensure the workspace tree is clean before unrelated ticket work resumes.
///
/// This invariant must not be weakened without a corresponding mechanism to
/// guarantee workspace cleanliness before starting new development work.
///
/// Does not push individual buffer entries for the moved tickets; the user is
/// already notified about the primary ticket's circuit breaker failure, so
/// per-sibling notifications are noise.
///
/// # Precondition
/// The circuit breaker must have tripped for `ticket` (i.e.,
/// [`try_trip_circuit_breaker`] returned `true` and the transition to
/// `Failed` has been attempted). The drain operates on the workspace
/// identified by `ticket.workspace_name` and is safe to call even if the
/// transition failed — the important invariant is that a breaker tripped,
/// so the pipeline should pause.
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

/// Shared circuit breaker skeleton: obtain comments, evaluate via
/// [`CircuitBreakerKind::should_trip`], format the trip message via
/// [`CircuitBreakerKind::trip_message`], add a system comment, then
/// transition to [`TicketPhase::Failed`].
///
/// The two comment-based breakers (Sanitation, Diagnostics) delegate to this
/// helper, supplying their variant logic via the [`CircuitBreakerKind`] enum.
/// This eliminates ~80% structural duplication while preserving exact
/// behavioral semantics. Phases without a pre-flight breaker pass `None` and
/// this returns `false` immediately — the bounce breaker is not pre-flight:
/// it is enforced at the bounce sites, which fail the ticket atomically with
/// the bounce-back when the counter is at
/// [`MAX_BOUNCES`](crate::joint_verdict::MAX_BOUNCES): review/QA bounces in
/// [`process_verifier_verdicts`] and engineer hard-failure bounces in
/// [`bounce_engineer_hard_failure`].
///
/// The Manager is notified when the ticket transitions to [`TicketPhase::Failed`].
///
/// # Self-counting prevention
///
/// Each breaker variant naturally excludes its own trip comment from counting:
///
/// * **Sanitation breaker** — filters comments by role `"sanitation_admin"` and content
///   containing the value of the `sanitation_failed.md` prompt;
///   trip comments always use role `SYSTEM_ROLE` (set by this function), so they
///   are never counted.
/// * **Diagnostics breaker** — filters comments by role `"diagnostics"` and content
///   containing the value of the `diagnostics_failed.md` prompt;
///   trip comments always use role `SYSTEM_ROLE` (set by this function), so they
///   are never counted.
///
/// See each variant's [`CircuitBreakerKind::should_trip`] implementation for
/// the exact filtering logic.
///
/// # Return value
///
/// Returns `true` if the breaker tripped — the caller MUST abort dispatch.
/// Returns `true` even on transition failure (the caller should still abort
/// rather than dispatching an agent to a stale or unreachable ticket).
#[must_use]
async fn try_trip_circuit_breaker(
    ticket: &Ticket,
    source_phase: TicketPhase,
    kind: Option<CircuitBreakerKind>,
    log_label: &str,
) -> bool {
    let Some(kind) = kind else {
        return false;
    };
    let comments = if ticket.comments.is_empty() {
        // Comments are only pre-loaded for claim-pipeline tickets
        // (LoadComments::Yes via claim_ticket_in_workspace). Tickets listed by
        // the poll loop's in-phase dispatchers (LoadComments::No) have an empty
        // vec — fetch from DB.
        match board().get_comments(&ticket.id).await {
            Ok(c) => std::borrow::Cow::Owned(c),
            Err(e) => {
                warn!(
                    ticket = %ticket.id,
                    error = %e,
                    "Failed to fetch comments for circuit breaker — proceeding anyway"
                );
                return false;
            }
        }
    } else {
        std::borrow::Cow::Borrowed(ticket.comments.as_slice())
    };

    let Some((count, max_count)) = kind.should_trip(&comments) else {
        return false;
    };

    let msg = kind.trip_message(count, max_count);

    info!(
        ticket = %ticket.id,
        count,
        max_count,
        log_label,
        "Circuit breaker tripped at {count}/{max_count} ({log_label}) — failing ticket"
    );

    let breaker_label = format!("{log_label} circuit breaker");
    match comment_and_transition(
        TransitionCtx::notifying(ticket, source_phase, TicketPhase::Failed, &breaker_label)
            .with_breaker(true),
        SYSTEM_ROLE,
        &msg,
    )
    .await
    {
        // Guard applied, or the ticket was already moved externally
        // (cancelled, superseded, ...) while the stage finished — nothing to
        // fail, no loop risk.
        FinalizeOutcome::Applied | FinalizeOutcome::Moved => {}
        // Genuine write failure: the breaker tripped but the Failed transition
        // did not land — the ticket is still claimable in the source phase and
        // will be re-dispatched, so the loop is real.
        FinalizeOutcome::Failed => {
            error!(
                ticket = %ticket.id,
                source_phase = %source_phase,
                log_label = %breaker_label,
                "Circuit breaker transition to Failed failed — ticket may loop indefinitely",
            );
        }
    }

    true
}

/// Aggregate per-agent failure reasons for a failed verifier round.
///
/// `NoResponse` entries carry the collapsed agent-failure reason; `ParseFailed`
/// entries carry their [`RetryExhausted`] detail INCLUDING the raw last-attempt
/// response dump (the primary extraction-failure diagnostic). Each reason is
/// capped at a per-agent share of [`crate::util::FAILURE_DETAIL_CAP`] so the
/// shares are fair — the caller's outer `truncate_sandwich` still re-caps the
/// aggregate, so in multi-agent all-ParseFailed rounds the middle dumps may
/// still be trimmed. Only called from the all-failed branch, where every
/// result is `NoResponse` or `ParseFailed`.
fn verifier_failure_reasons(results: &[ParallelVerdict]) -> String {
    let per_agent_cap = crate::util::FAILURE_DETAIL_CAP / results.len().max(1);
    let reasons: Vec<String> = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| match r {
            ParallelVerdict::NoResponse(reason) => Some(format!("{}. {reason}", i + 1)),
            ParallelVerdict::ParseFailed(f) => Some(format!(
                "{}. verdict extraction failed: {}",
                i + 1,
                crate::util::truncate_sandwich(
                    &raw_response_dump_section(f),
                    per_agent_cap,
                    "agent failure",
                )
            )),
            // Unreachable at the only call site (all_failed) — required for
            // match exhaustiveness.
            ParallelVerdict::Verdict(_) => None,
        })
        .collect();
    format!("\n\nPer-agent failures:\n{}", reasons.join("\n"))
}

/// The bounce circuit-breaker trip comment: the ticket was bounced (review/QA
/// bounce or engineer hard failure) too many times. Shared by the verifier
/// bounce trip and the engineer hard-failure trip so the wording cannot drift
/// (engineer hard failures share the review/QA bounce budget).
/// Must retain the "circuit breaker" substring (asserted by
/// `eleventh_bounce_fails_ticket`).
#[must_use]
fn bounce_breaker_trip_comment() -> String {
    let max = crate::joint_verdict::MAX_BOUNCES;
    format!(
        "Failed after {max} bounces — ticket bounced back too many times \
         (circuit breaker, max: {max}). Ticket failed — Manager will triage."
    )
}

/// Returns `true` when the bounce budget is exhausted and the ticket must fail
/// instead of bouncing back to [`TicketPhase::ReadyForDevelopment`].
///
/// Uses `usize::try_from(bounce_count).unwrap_or(usize::MAX) >= MAX_BOUNCES`
/// so a negative `bounce_count` (corrupted row) or a value that does not fit
/// in `usize` maps to `usize::MAX` and therefore to exhausted — fail-closed
/// rather than bouncing forever. A plain `i64 >= MAX_BOUNCES as i64`
/// comparison would treat negatives as "not exhausted" and would not clamp
/// overflow on 32-bit targets where `usize::MAX < i64::MAX`.
#[must_use]
fn bounce_exhausted(bounce_count: i64) -> bool {
    usize::try_from(bounce_count).unwrap_or(usize::MAX) >= crate::joint_verdict::MAX_BOUNCES
}

/// Pure 3-way verifier bounce target table.
///
/// - `all_failed == true`  → [`TicketPhase::Failed`] (terminal, even if
///   `exhausted` is true).
/// - `any_failed == true`  → [`TicketPhase::Failed`] when `exhausted`,
///   otherwise [`TicketPhase::ReadyForDevelopment`] (bounce back with
///   incremented counter).
/// - otherwise             → `success_phase` (all verifiers passed).
///
/// This helper is the verifier 3-way only. The engineer hard-failure path is
/// intentionally 2-way (`Failed` vs `ReadyForDevelopment`) and should call
/// only [`bounce_exhausted`] directly without this helper to avoid threading a
/// dummy `success_phase` argument through a path that never uses it.
#[must_use]
fn decide_verifier_bounce_target(
    all_failed: bool,
    any_failed: bool,
    exhausted: bool,
    success_phase: TicketPhase,
) -> TicketPhase {
    if all_failed {
        TicketPhase::Failed
    } else if any_failed {
        if exhausted {
            TicketPhase::Failed
        } else {
            TicketPhase::ReadyForDevelopment
        }
    } else {
        success_phase
    }
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
///    [`handle_qa_passed`]).
///
/// Returns `true` when the comment+transition completed, `false` on failure
/// (the ticket stays in `verifier.active_phase`).
///
/// See the "Parallel agent helpers (shared)" section for why this is separate
/// from [`process_analyst_verdicts`].
#[expect(clippy::too_many_lines)]
async fn process_verifier_verdicts(
    ws: &Workspace,
    ticket: &Ticket,
    results: &[ParallelVerdict],
    verifier: VerifierInfo,
) -> bool {
    let all_failed = results
        .iter()
        .all(|r| !matches!(r, ParallelVerdict::Verdict(_)));
    // Gate against the actually-dispatched count (N_gate): a no-response or
    // parse failure counts as a failed agent, so a partial round bounces.
    let any_failed = results.iter().any(|r| match r {
        ParallelVerdict::Verdict(v) => !verdict_passes(v),
        _ => true,
    });

    // Bounce-based circuit breaker: the 11th bounce fails the ticket. The
    // counter is incremented atomically with the bounce-back transition.
    // The ticket's counter is authoritative here: it was loaded at claim
    // time and only the bounce-back transition (which happens AFTER this
    // check) increments it, so no re-fetch can differ from it.
    let exhausted = bounce_exhausted(ticket.bounce_count);
    // Determine transition parameters based on the three-way branch:
    //   all-failed → Failed (notify, with failure comment)
    //   any-failed → ReadyForDevelopment (buffer, pipeline reservation),
    //                unless the bounce breaker trips → Failed
    //   all-passed → verifier.success_phase (buffer)
    let target =
        decide_verifier_bounce_target(all_failed, any_failed, exhausted, verifier.success_phase);
    let notify = if all_failed {
        NotifyPolicy::Notify
    } else if any_failed {
        if exhausted {
            NotifyPolicy::Notify
        } else {
            NotifyPolicy::Buffer
        }
    } else {
        NotifyPolicy::Buffer
    };
    let bounce_trip = any_failed && exhausted && !all_failed;
    // No exit-time ticket rollback: during the graceful drain a
    // failed round must NOT drive the ticket to Failed, bounce it, or pause
    // the workspace — the job stays status='launched' for boot resume, which
    // re-processes the stored outcomes. Suppresses ALL transitions (consistent
    // with process_analyst_verdicts): a partial round whose drain-cut members
    // produced NoResponse would otherwise hit the any_failed bounce path and
    // bounce the ticket with a joint comment during the drain, discarding the
    // checkpointed outcomes.
    if crate::shutdown::aborting() {
        info!(
            ticket = %ticket.id,
            stage = %verifier.log_label,
            "Verifier round cut short by drain — job stays launched for boot resume",
        );
        return false;
    }

    // Preserve the per-agent failure reasons (previously collapsed into a
    // generic "no response") and auto-pause the workspace BEFORE the Failed
    // transition so the failure notification reflects the paused workspace.
    // Shutdown is excluded inside the pause helper. The bounce breaker is
    // excluded from pausing, like all other circuit-breaker trips.
    let (failure_comment, reasons) = if all_failed {
        let pause_note =
            pause_workspace_on_failure(ticket, "all verifier agents failed to produce verdicts")
                .await;
        let reasons = verifier_failure_reasons(results);
        let header = substitute(
            &load_prompt("pipeline/verifiers_all_failed.md"),
            &[("{{agent_type}}", verifier.log_label)],
        );
        let body = format!("{reasons}{pause_note}");
        let failure_comment = crate::util::truncate_sandwich(
            &crate::util::scrub_credentials(&format!("{header}\n{body}")),
            crate::util::FAILURE_DETAIL_CAP,
            "verifier failure",
        );
        (failure_comment, reasons)
    } else {
        (String::new(), String::new())
    };

    // One joint comment replaces the per-agent failing comments — written
    // after every round EXCEPT all-failed rounds (those keep their dedicated
    // SYSTEM_ROLE failure comment; a joint comment would duplicate it).
    // Synthesis runs BEFORE the transition (never holding the board write
    // lock); the comment+transition transaction stays short.
    let joint_comment = if all_failed {
        None
    } else {
        Some(
            build_round_joint_comment(
                stage_name(verifier.role),
                results,
                REVIEW_QA_THRESHOLD,
                verifier.role,
                "",
                ws,
                &ticket.id,
                &ticket.title,
            )
            .await,
        )
    };

    let bounce_breaker_comment = bounce_trip.then(bounce_breaker_trip_comment);

    if !matches!(
        with_comment_and_transition(
            TransitionCtx::new(
                ticket,
                verifier.active_phase,
                target,
                notify,
                verifier.log_label
            )
            .with_breaker(bounce_trip),
            async |tx| {
                if let Some(comment) = &joint_comment {
                    BoardStore::add_comment_tx(tx, &ticket.id, stage_name(verifier.role), comment)
                        .await?;
                }
                if let Some(comment) = &bounce_breaker_comment {
                    BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, comment).await?;
                }
                if all_failed {
                    BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, &failure_comment)
                        .await?;
                }
                if target == TicketPhase::ReadyForDevelopment {
                    BoardStore::increment_bounce_count_tx(tx, &ticket.id).await?;
                }
                Ok(())
            },
        )
        .await,
        FinalizeOutcome::Applied
    ) {
        return false;
    }

    // A bounce-breaker trip fails the ticket with a dirty workspace — drain
    // ReadyForDevelopment siblings so the Engineer cannot pick up unrelated
    // work on top of the failed tree (mirrors the pre-flight breaker drain).
    if bounce_trip {
        drain_ready_for_development_siblings(ticket).await;
    }

    if all_failed {
        info!(
            ticket = %ticket.id,
            reasons = %crate::util::truncate_sandwich(
                &crate::util::scrub_credentials(&reasons),
                crate::util::FAILURE_DETAIL_CAP,
                "verifier failure",
            ),
            "{log_label}: all verifier agents failed to produce verdicts — ticket moved to Failed",
            log_label = verifier.log_label,
        );
    } else if bounce_trip {
        info!(
            ticket = %ticket.id,
            "Bounce circuit breaker tripped ({MAX_BOUNCES} bounces) — ticket failed",
            MAX_BOUNCES = crate::joint_verdict::MAX_BOUNCES,
        );
    } else if any_failed {
        info!(
            ticket = %ticket.id,
            "{log_label} failed — pipeline reservation set for rework priority",
            log_label = verifier.log_label,
        );
    } else {
        info!(
            ticket = %ticket.id,
            "{log_label}: all passed (≥ {REVIEW_QA_THRESHOLD}/10)",
            log_label = verifier.log_label,
        );
    }
    true
}

/// Decide whether the reviewer pass may be skipped for a ticket.
///
/// Skipping is allowed ONLY when the ticket has a recorded reviewed base
/// (from a prior completed review round on this ticket) and the current
/// content is identical to that base: same HEAD commit, same index tree,
/// and a clean porcelain (no unstaged or untracked changes).
///
/// Conservative by design — may over-review, never under-review: any missing
/// input (no recorded base, uncomputable HEAD/tree) yields `false` (review).
#[must_use]
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

/// Compute the skip-review decision for a ticket: fetch the current content
/// identity (porcelain, HEAD, index tree) and compare it against the ticket's
/// recorded reviewed base. Porcelain errors propagate to the caller (full
/// review fallback); HEAD/tree failures yield `None` — never a skip.
async fn compute_review_skip(ticket: &Ticket, repo_path: &Path) -> anyhow::Result<bool> {
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

/// Gather the working-tree churn at review dispatch: total added + deleted
/// lines vs HEAD, including lines from new files (staged new files appear in
/// the diff; untracked files are enumerated and counted too). The diff is
/// computed against HEAD — no commit for this ticket exists yet (DB line
/// stats are populated only at final done).
async fn working_tree_churn(repo_path: &Path) -> anyhow::Result<i64> {
    let (added, removed) = run_git_diff_stats(repo_path).await?;
    Ok(added + removed)
}

/// Compute the reviewer count for a review round.
///
/// The base is recomputed from the LIVE working-tree diff at every review
/// round dispatch — rework-grown diffs can escalate reviewer counts across
/// rounds (2 → 3 → 4), which is intended. Bounces do not change the count.
/// QA never passes through here (fixed 1).
///
/// Shadow instrumentation: the counterfactual signals are logged at info
/// level so the formula can be validated against a cohort after launch.
async fn compute_reviewer_count(ticket: &Ticket, repo_path: &Path) -> usize {
    let low =
        i64::try_from(crate::joint_verdict::DEFAULT_REVIEW_COUNT_LOW_CHURN).unwrap_or(i64::MAX);
    let high =
        i64::try_from(crate::joint_verdict::DEFAULT_REVIEW_COUNT_HIGH_CHURN).unwrap_or(i64::MAX);

    match working_tree_churn(repo_path).await {
        Ok(total) => {
            let base = crate::joint_verdict::review_base_from_signals(total, low, high);
            info!(
                ticket = %ticket.id,
                total_churn = total,
                reviewer_base = base,
                "Reviewer count calibration: base {base} from total churn",
            );
            crate::joint_verdict::review_agent_count(base, ticket.priority)
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

/// Resume a ticket_stage round at boot: re-run missing roster
/// slots FIRST, then re-process verdicts from stored outcomes when all are
/// present. Silent background resume — no Manager notifications; results
/// deliver via normal paths (board comment/transition).
async fn resume_ticket_stage_round(stage: String, job_id: String, ticket: Ticket, ws: Workspace) {
    match stage.as_str() {
        "analysis" => resume_analysis_round(&job_id, ticket, ws).await,
        "review" => resume_verifier_round(&job_id, ticket, ws, REVIEWER_VI).await,
        "qa" => resume_verifier_round(&job_id, ticket, ws, QA_VI).await,
        "engineer" => resume_stage_round(&job_id, ticket, ws, StageRoundKind::Engineer).await,
        "sanitation" => resume_stage_round(&job_id, ticket, ws, StageRoundKind::Sanitation).await,
        other => {
            warn!(stage = %other, job = %job_id, "Unknown ticket_stage on resume — completing job");
            complete_ticket_stage_job(&job_id).await;
        }
    }
}

/// Resume an analysis round: re-run not-done roster slots, re-evaluate
/// escalation only when roster size == 3 (matches analysis_escalation_needed),
/// then re-process verdicts through the existing process_analyst_verdicts.
async fn resume_analysis_round(job_id: &str, ticket: Ticket, ws: Workspace) {
    if complete_job_and_bail_if_phase_moved(&ticket.id, TicketPhase::Analysis, job_id).await {
        return;
    }
    let Ok(slots) = load_ticket_stage_slots(job_id).await else {
        complete_ticket_stage_job(job_id).await;
        return;
    };
    let ticket_arc = Arc::new(ticket);
    let task = job_task(job_id).await;
    run_analysis_round(&ticket_arc, &ws, job_id, &slots, &task, true).await;
}

/// Stage all changes and record the ticket's reviewed base (HEAD + index tree)
/// after a review round that produced verdicts and transitioned the ticket —
/// a skipped base on a failed transition or an all-failed round (content never
/// reviewed) would let a later round skip content nobody saw. Shared by live
/// dispatch and boot resume; the base gates the skip-review check, so the paths
/// must stay in lockstep. A resumed round replays verdicts from an older
/// content state but records the CURRENT working-tree state.
async fn record_reviewed_base_after_review(
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
            } else if let Err(e) = board()
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

/// Resume a verifier round (review/QA): re-run missing slots, then re-process
/// verdicts. REPLAY INCLUDES THE DISPATCH TAIL — the
/// record_reviewed_base_after_review helper — so the skip-review gate
/// (tickets.reviewed_head/reviewed_tree) still fires on later rounds. The
/// reviewer count is never frozen: it is recomputed from the live working-tree
/// diff at every dispatch, so a resumed round re-derives it like any other.
async fn resume_verifier_round(job_id: &str, ticket: Ticket, ws: Workspace, vi: VerifierInfo) {
    if complete_job_and_bail_if_phase_moved(&ticket.id, vi.active_phase, job_id).await {
        return;
    }
    let Ok(slots) = load_ticket_stage_slots(job_id).await else {
        complete_ticket_stage_job(job_id).await;
        return;
    };
    let extraction_prompt = load_prompt(vi.extraction_prompt_path);
    let ticket_arc = Arc::new(ticket);
    let results = run_parallel_agents(
        &ticket_arc,
        &ws,
        vi.role,
        &extraction_prompt,
        job_id,
        &slots,
        true,
    )
    .await;
    if complete_job_and_bail_if_phase_moved(&ticket_arc.id, vi.active_phase, job_id).await {
        return;
    }
    if crate::shutdown::aborting() {
        // Drain-cut: outcomes checkpointed, job stays launched for boot resume.
        return;
    }

    let (_is_reviewer, _repo_path, git_available) = verifier_git_state(&ws, vi).await;

    finalize_verifier_round(&ws, &ticket_arc, vi, &results, job_id, true, git_available).await;
}

/// Resume a single-agent stage round (engineer/sanitation) at boot: phase-
/// guard, then re-run the shared stage-agent tail (see
/// [`run_stage_agent_round`] for the resume dispatch rule).
async fn resume_stage_round(job_id: &str, ticket: Ticket, ws: Workspace, kind: StageRoundKind) {
    let phase = match kind {
        StageRoundKind::Engineer => TicketPhase::InDevelopment,
        StageRoundKind::Sanitation => TicketPhase::InSanitation,
    };
    if complete_job_and_bail_if_phase_moved(&ticket.id, phase, job_id).await {
        return;
    }
    let task = job_task(job_id).await;
    run_stage_agent_round(&ticket, &ws, job_id, &task, true, kind).await;
}

/// Read a job's stored task (the FINAL rendered prompt template).
async fn job_task(job_id: &str) -> String {
    crate::session::store()
        .conn
        .query_optional(
            "SELECT task FROM jobs WHERE id = ?1",
            crate::turso::params![job_id],
            |row| row.get::<String>(0),
        )
        .await
        .ok()
        .flatten()
        .unwrap_or_default()
}

/// Finalize a verifier round (review/QA) — shared by fresh dispatch and boot
/// resume so the post-verdict tail stays in lockstep. `resumed` selects the
/// observability strings (drain-cut log gains a "Resumed " prefix,
/// [`record_reviewed_base_after_review`] a "Resume: " one — the verifier path
/// logs on resume, unlike the silent engineer/sanitation wrappers; preserved).
/// The drain guard inside [`process_verifier_verdicts`] already logged the
/// cut-short (the outer re-log is the pre-existing double-log, kept as-is);
/// the job stays status='launched' for boot resume.
async fn finalize_verifier_round(
    ws: &Workspace,
    ticket: &Ticket,
    vi: VerifierInfo,
    results: &[ParallelVerdict],
    job_id: &str,
    resumed: bool,
    git_available: bool,
) {
    let transitioned = process_verifier_verdicts(ws, ticket, results, vi).await;
    if !transitioned && crate::shutdown::aborting() {
        let cut_short = if resumed {
            "Resumed verifier round cut short by drain — job stays launched for boot resume"
        } else {
            "Verifier round cut short by drain — job stays launched for boot resume"
        };
        info!(ticket = %ticket.id, "{cut_short}");
        return;
    }

    record_reviewed_base_after_review(
        ws.as_path(),
        &ticket.id,
        git_available,
        transitioned,
        results,
        if resumed { "Resume: " } else { "" },
    )
    .await;
    complete_ticket_stage_job(job_id).await;
}

/// Git availability + reviewer identity shared by the verifier pair:
/// `(is_reviewer, repo_path, git_available)`.
async fn verifier_git_state(ws: &Workspace, vi: VerifierInfo) -> (bool, &Path, bool) {
    let is_reviewer = vi.role == Role::Reviewer;
    let repo_path = ws.as_path();
    let git_available = is_reviewer
        && crate::git_commands::git_is_installed().await
        && crate::git_commands::is_git_repo(repo_path);
    (is_reviewer, repo_path, git_available)
}

/// Shared dispatch logic for parallel verifiers (reviewers and QA).
/// Fetches the engineer's last comment, builds a prompt from the template,
/// runs verifiers of the given role (reviewers use the calibrated dynamic
/// count; QA runs a single agent), and processes the verdicts.
///
/// ## Note: no `clear_assigned_to_no_cancel` on post-run phase check
///
/// See [`dispatch_backlog_analysts`] for the full rationale — the same
/// structural reasons apply here (parallel agents via [`run_parallel_agents`],
/// `assigned_to` set to `NULL` during the [`claim_ticket_in_workspace`] claim).
async fn dispatch_verifiers(ticket: Arc<Ticket>, ws: Workspace, vi: VerifierInfo) {
    // ── Skip-review check for Reviewers ──────────────────────────────
    // Skip ONLY when the current content is identical to the reviewed base
    // recorded on this ticket by a prior completed review round (same HEAD,
    // same index tree, clean porcelain). A ticket with no recorded base —
    // first review round, brand-new commit — must always run the full pass.
    let (is_reviewer, repo_path, git_available) = verifier_git_state(&ws, vi).await;

    if git_available {
        match compute_review_skip(&ticket, repo_path).await {
            Ok(true) => {
                info!(
                    ticket = %ticket.id,
                    "Content identical to reviewed base — skipping reviewer dispatch",
                );
                let _ = comment_and_transition(
                    TransitionCtx::buffered(
                        &ticket,
                        vi.active_phase,
                        TicketPhase::Reviewed,
                        vi.log_label,
                    ),
                    SYSTEM_ROLE,
                    "Content is identical to the reviewed base recorded for this ticket \
                     (same HEAD commit and index tree, no working-tree changes). \
                     Skipping reviewer dispatch.",
                )
                .await;
                return;
            }
            Ok(false) => {}
            Err(e) => {
                warn!(
                    ticket = %ticket.id,
                    error = %e,
                    "Git status check failed for skip-review — proceeding with normal review",
                );
            }
        }
    }

    // ── Normal verifier dispatch ─────────────────────────────────────
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
    let count = if is_reviewer {
        compute_reviewer_count(&ticket, repo_path).await
    } else {
        // QA calibration is infeasible from history — fixed at 1.
        QA_PARALLEL_AGENT_COUNT
    };
    let verifier_label = if count == 1 { "verifier" } else { "verifiers" };
    info!(
        ticket = %ticket.id,
        role = %vi.role.as_str(),
        count,
        verifier_label,
        "Dispatching {count} parallel {verifier_label}",
    );

    // Spawn the durable verifier round (jobs + roster, one tx) BEFORE the
    // agents' first session writes — the roster is the round record.
    let stage = if is_reviewer { "review" } else { "qa" };
    let Ok((job_id, slots)) = spawn_ticket_stage_round(
        &ticket,
        &ws,
        stage,
        vi.active_phase,
        vi.role,
        &prompt,
        count,
    )
    .await
    else {
        error!(
            ticket = %ticket.id,
            "Failed to spawn verifier job — aborting dispatch",
        );
        return;
    };
    let results = run_parallel_agents(
        &ticket,
        &ws,
        vi.role,
        &extraction_prompt,
        &job_id,
        &slots,
        false,
    )
    .await;
    if complete_job_and_bail_if_phase_moved(&ticket.id, vi.active_phase, &job_id).await {
        return;
    }

    finalize_verifier_round(&ws, &ticket, vi, &results, &job_id, false, git_available).await;
}

#[cfg(test)]
#[path = "management_tests.rs"]
mod tests;
