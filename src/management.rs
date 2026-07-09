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
/// only checks for [`DIAGNOSTICS_FAILED_MARKER`] prefix and [`DIAGNOSTICS_ROLE`].
const DIAGNOSTICS_PASSED_MARKER: &str = "✅ All diagnostics passed";
/// Marker appended when diagnostics fail (includes the failed-at label after it).
const DIAGNOSTICS_FAILED_MARKER: &str = "❌ Diagnostics failed at";

/// Prefix for sanitation failure system comments — [`CircuitBreakerKind::Sanitation`]'s
/// [`trip_count`](CircuitBreakerKind::trip_count) depends on substring matching
/// this value, so it must not drift from comment text.
const SANITATION_FAILED_PREFIX: &str = "Sanitation failed";

/// Minimum acceptable verification score (0-10) for analyst verdicts.
const ANALYST_PASS_THRESHOLD: u8 = 7;

/// Minimum acceptable verification score (0-10) for review and QA phases.
const REVIEW_QA_THRESHOLD: u8 = 9;

/// Returns the global [`BoardStore`] singleton.
#[inline]
fn board() -> &'static BoardStore {
    crate::board::store()
}

// ── Circuit breaker kind ──────────────────────────────────────────────────────

/// Identifies which circuit breaker variant to use for phase-guard checks
/// and trip logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter)]
enum CircuitBreakerKind {
    /// General comment-count breaker: trips when the total number of comments
    /// exceeds 30.
    General,
    /// Sanitation-failure breaker: trips when consecutive sanitation failures
    /// exceed 3.
    Sanitation,
    /// Diagnostics-failure breaker: trips when cumulative diagnostics failures
    /// exceed 4.
    Diagnostics,
}

impl CircuitBreakerKind {
    /// Returns the trip-count threshold for this breaker variant.
    /// The breaker trips when `trip_count > threshold()`.
    const fn threshold(self) -> usize {
        match self {
            Self::General => 30,
            Self::Sanitation => 3,
            Self::Diagnostics => 4,
        }
    }

    /// Extract the relevant trip count from the ticket's comments for
    /// this breaker variant.
    fn trip_count(self, comments: &[TicketComment]) -> usize {
        match self {
            Self::General => comments.len(),
            Self::Sanitation => comments
                .iter()
                .filter(|c| c.role == SYSTEM_ROLE && c.content.contains(SANITATION_FAILED_PREFIX))
                .count(),
            Self::Diagnostics => comments
                .iter()
                .filter(|c| {
                    c.role == DIAGNOSTICS_ROLE
                        && c.content.starts_with(DIAGNOSTICS_COMMENT_PREFIX)
                        && c.content.contains(DIAGNOSTICS_FAILED_MARKER)
                })
                .count(),
        }
    }

    /// Format the circuit-breaker trip comment body for this breaker variant.
    fn comment(self, count: usize) -> String {
        match self {
            Self::General => {
                let threshold = self.threshold();
                format!(
                    "Failed after {count} comments — ticket has accumulated too many comments \
                     (circuit breaker, threshold: {threshold}). \
                     Ticket failed — Manager will triage."
                )
            }
            Self::Sanitation => {
                let threshold = self.threshold();
                format!(
                    "❌ Sanitation circuit breaker tripped after {count} consecutive failures. \
                     (threshold: {threshold})",
                )
            }
            Self::Diagnostics => format!(
                "{DIAGNOSTICS_COMMENT_PREFIX}\n\n❌ Circuit breaker: {count} prior diagnostic \
                 failures. Failing ticket."
            ),
        }
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
            warn!(ticket = %ticket_id, error = %e, "Failed to check ticket status");
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
/// when buffering, and `log_label` distinguishes callers in logs.
///
/// `failure_comment` is meaningful only when `target == TicketPhase::Failed` and
/// `notify == NotifyPolicy::Notify` — it carries the failure detail text that
/// was already written to the DB atomically with the transition, avoiding a
/// redundant DB round-trip in [`notify_ticket`]. Callers transitioning to
/// non-Failed phases should pass `None`.
async fn dispatch_notification(
    ticket: &Ticket,
    source: TicketPhase,
    target: TicketPhase,
    notify: NotifyPolicy,
    log_label: &str,
    failure_comment: Option<&str>,
) {
    match notify {
        NotifyPolicy::Notify => notify_ticket(ticket, target, failure_comment).await,
        NotifyPolicy::Buffer => {
            if let Some(ws) = resolve_ticket_workspace(ticket, log_label).await {
                ticket_buffer::push(&ws.name, &ticket.id, source, target);
            }
        }
    }
}

/// Log a warning when a ticket transition fails, using `source` and `target`
/// phases directly so hardcoded phase-name strings can't drift.
///
/// `log_label` is a human-readable name for the dispatch phase (e.g.
/// `"Engineer"`, `"Sanitation"`, `"Analyst"`). `verb` is the past-tense action
/// the phase performed — typically `"completed"`, but may be `"failed"` or
/// `"passed"` depending on context.
fn warn_transition_failed(
    ticket: &Ticket,
    source: TicketPhase,
    target: TicketPhase,
    log_label: &str,
    verb: &str,
    error: &anyhow::Error,
) {
    warn!(
        ticket = %ticket.id,
        error = %error,
        "{log_label} {verb} but transition to {target} failed — ticket stuck in {source}",
    );
}

/// A comment to add to a ticket, with the role that authored it.
///
/// Using a named struct instead of a `(role, content)` tuple eliminates the
/// risk of accidentally swapping the two fields — the same bug class that
/// [`TransitionParams`] already prevents for `source`/`target` via named
/// fields.
#[derive(Debug, Copy, Clone)]
struct CommentParam<'a> {
    role: &'a str,
    content: &'a str,
}

/// Bundled parameters for [`comment_and_transition`].
///
/// Encapsulates all arguments needed to add a comment (and optional extra
/// comment) to a ticket and atomically transition it between phases. Using
/// named fields prevents argument-ordering bugs (particularly
/// [`source`](TransitionParams::source)/[`target`](TransitionParams::target)
/// swaps) and makes call sites self-documenting.
///
/// # ⚠ Argument ordering
///
/// `source` and `target` are both [`TicketPhase`], so swapping them is a
/// potential runtime bug (the database rejects the transition). Always use
/// named fields when constructing this struct.
#[derive(Debug)]
struct TransitionParams<'a> {
    ticket: &'a Ticket,
    comment: CommentParam<'a>,
    extra: Option<CommentParam<'a>>,
    source: TicketPhase,
    target: TicketPhase,
    notify: NotifyPolicy,
    log_label: &'a str,
    verb: &'a str,
}

/// Write one or more comments to a ticket, then transition it to a new phase.
///
/// All comments and the phase transition are wrapped in a single transaction
/// via [`crate::turso::with_tx`]. If any operation fails, the entire
/// transaction is rolled back — neither the comments nor the transition takes
/// effect. Returns `true` on success, `false` on failure.
///
/// On success, dispatches a notification after the transaction commits (via
/// [`dispatch_notification`]).
///
/// This is atomic — there is no crash window between comments and transition
/// where orphaned comments could persist.
///
/// # Correctness
///
/// The caller must provide exactly one required `comment` and an optional
/// second `extra` comment. All comments and the phase transition are written
/// in a single database transaction. The type signature enforces that at
/// least one comment is always written — a missing comment is a
/// compile-time error, not a runtime panic.
///
/// Uses [`BoardStore::transition_to_tx`] which does **not** cancel registered
/// agents (unlike [`BoardStore::transition_to`]).
/// This is correct because `comment_and_transition` is only called from post-agent
/// paths (verdict handling, diagnostics completion, sanitation verdict, etc.) —
/// no agents should be running on this ticket at any call site that reaches
/// this function.
///
/// `pipeline_reservation` is automatically derived from the target phase:
/// `Some(true)` when transitioning to [`TicketPhase::ReadyForDevelopment`]
/// (bounce-back transitions get priority re-dispatch over fresh tickets),
/// `None` for all other transitions.
#[must_use]
async fn comment_and_transition(params: TransitionParams<'_>) -> bool {
    // Extract failure_comment before the with_tx closure, which captures
    // params by move. Used exclusively for Failed-target transitions to avoid a
    // redundant DB round-trip in notify_ticket.
    let failure_comment = (params.target == TicketPhase::Failed).then_some(params.comment.content);
    let pipeline_reservation = (params.target == TicketPhase::ReadyForDevelopment).then_some(true);

    if let Err(e) = crate::turso::with_tx(
        &board().conn,
        &params.ticket.id,
        &format!("{} {}", params.log_label, params.verb),
        async |tx| {
            BoardStore::add_comment_tx(
                tx,
                &params.ticket.id,
                params.comment.role,
                params.comment.content,
            )
            .await?;
            if let Some(extra) = params.extra {
                BoardStore::add_comment_tx(tx, &params.ticket.id, extra.role, extra.content)
                    .await?;
            }
            BoardStore::transition_to_tx(
                tx,
                &params.ticket.id,
                Some(params.source),
                params.target,
                pipeline_reservation,
            )
            .await?;
            Ok(())
        },
    )
    .await
    {
        warn_transition_failed(
            params.ticket,
            params.source,
            params.target,
            params.log_label,
            params.verb,
            &e,
        );
        return false;
    }

    // Notification fires after the transaction commits so the user always
    // sees the new phase in the database.
    dispatch_notification(
        params.ticket,
        params.source,
        params.target,
        params.notify,
        params.log_label,
        failure_comment,
    )
    .await;

    true
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
/// When `target_phase == TicketPhase::Failed`, the failure details are passed
/// directly via `failure_comment` instead of re-reading from the database.
/// The caller MUST ensure `failure_comment` is `Some` when `target == Failed`.
/// The session key (`manager_{ws_name}`) is intentionally shared between
/// user-facing Manager chat (main.rs) and notification agents — the same Manager
/// must see both notification context and user conversation history in a unified
/// session. Do NOT change this key or add `manager_` to `TRANSIENT_SESSION_PREFIXES`
/// — it would either break context continuity or nuke user conversation history.
///
/// # Panics
///
/// Panics if `target == TicketPhase::Failed` and `failure_comment` is `None`,
/// indicating a programming error in a caller that failed to provide the
/// required failure detail text.
async fn notify_ticket(ticket: &Ticket, target_phase: TicketPhase, failure_comment: Option<&str>) {
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
            ("{{ticket_status}}", target_phase.as_ref()),
            ("{{transition_log}}", &transition_log),
            ("{{ticket_updates}}", &drained),
        ],
    );

    if target_phase == TicketPhase::Failed {
        // failure_comment must be Some when target is Failed — all call paths
        // that transition to Failed write the comment atomically with the
        // transition and pass it through the call chain (via
        // comment_and_transition → dispatch_notification → notify_ticket).
        // A None here indicates a programming error in a new or modified caller.
        let failure_details =
            failure_comment.expect("failure_comment must be Some when target is Failed");

        let warning = substitute(
            &load_prompt("warning.md"),
            &[("{{failure_details}}", failure_details)],
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
        // ── Pre-flight guard checks ──
        //
        // Adding a new PollPhase variant requires deciding its
        // circuit_breaker_kind() (enforced by the match in
        // circuit_breaker_kind()).
        //
        // The post-agent is_ticket_in_phase check in each dispatch function is
        // a separate concern (race-condition guard) and is preserved there.
        let (kind, log_label) = phase.circuit_breaker_kind();
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
            let _ = transition_ticket_to_failed(
                &ticket_for_failure,
                expected_phase,
                &format!("❌ Dispatch panicked: {msg}"),
                "dispatch panic",
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
/// All phase-specific data lives here, sourced from the single [`PollPhase::info()`]
/// match — adding any phase requires one row in that match.
#[derive(Copy, Clone)]
struct PollPhaseInfo {
    expected_phase: TicketPhase,
    /// How this phase checks pipeline occupancy. [`Enforce`](PipelineCheck::Enforce)
    /// blocks claims when another pipeline ticket is active in the workspace;
    /// [`Skip`](PipelineCheck::Skip) allows concurrent claims.
    pipeline_check: PipelineCheck,
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
                expected_phase: TicketPhase::Analysis,
                pipeline_check: PipelineCheck::Skip,
                role_label: Role::Analyst.as_str(),
            },
            Self::EngineerDevelopment => PollPhaseInfo {
                expected_phase: TicketPhase::InDevelopment,
                pipeline_check: PipelineCheck::Enforce,
                role_label: Role::Engineer.as_str(),
            },
            Self::SanitationCheck => PollPhaseInfo {
                expected_phase: TicketPhase::InSanitation,
                // Note: expected_phase is consumed by spawn_dispatch's
                // panic-recovery transition (expected_phase → Failed).
                // SanitationCheck is excluded from CLAIM_PHASES since the
                // actual QaPassed→InSanitation transition happens via
                // claim_sanitation in handle_qa_passed.
                pipeline_check: PipelineCheck::Skip,
                role_label: Role::Sanitation.as_str(),
            },
            Self::DiagnosticsCheck => PollPhaseInfo {
                expected_phase: TicketPhase::InDiagnostics,
                pipeline_check: PipelineCheck::Skip,
                role_label: DIAGNOSTICS_ROLE,
            },
            Self::VerifierCheck(vi) => PollPhaseInfo {
                expected_phase: vi.source_phase,
                pipeline_check: PipelineCheck::Skip,
                role_label: vi.role.as_str(),
            },
        }
    }

    /// Returns the circuit breaker kind and log label for this phase.
    ///
    /// The log label matches the current convention (PascalCase for roles, "QA"
    /// for the QA verifier) to preserve log output consistency.
    fn circuit_breaker_kind(self) -> (CircuitBreakerKind, &'static str) {
        match self {
            Self::BacklogAnalysis => (CircuitBreakerKind::General, "Analyst"),
            Self::EngineerDevelopment => (CircuitBreakerKind::General, "Engineer"),
            Self::SanitationCheck => (CircuitBreakerKind::Sanitation, "Sanitation"),
            Self::DiagnosticsCheck => (CircuitBreakerKind::Diagnostics, "Diagnostics"),
            Self::VerifierCheck(vi) => (CircuitBreakerKind::General, vi.log_label),
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

        // 3. SanitationPassed → Done (auto-commit), following the same pattern
        // as the QaPassed→Done commit flow.
        spawn_for_each_ticket_in_phase(TicketPhase::SanitationPassed, ws, |ticket, ws| {
            finalize_ticket_from_phase(ticket, ws, TicketPhase::SanitationPassed, None)
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

    let verb = match target_phase {
        TicketPhase::InDiagnostics => "completed",
        _ => "failed",
    };

    if !comment_and_transition(TransitionParams {
        ticket: &ticket,
        comment: CommentParam {
            role: Role::Engineer.as_str(),
            content: comment_text,
        },
        extra: None,
        source: TicketPhase::InDevelopment,
        target: target_phase,
        notify,
        log_label: "Engineer",
        verb,
    })
    .await
    {
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
    let log_label = match source {
        TicketPhase::QaPassed => "QA",
        TicketPhase::SanitationPassed => "Sanitation",
        _ => {
            warn!(
                ticket = %ticket.id,
                phase = %source.as_ref(),
                "transition_ticket_to_done called from unexpected source phase",
            );
            "Unexpected"
        }
    };
    if comment_and_transition(TransitionParams {
        ticket,
        comment: CommentParam {
            role: SYSTEM_ROLE,
            content: comment,
        },
        extra: None,
        source,
        target: TicketPhase::Done,
        notify: notify_policy,
        log_label,
        verb: "passed",
    })
    .await
    {
        info!(ticket = %ticket.id, "{comment}");
    }
}

#[must_use]
async fn transition_ticket_to_failed(
    ticket: &Ticket,
    source: TicketPhase,
    comment: &str,
    log_label: &str,
) -> bool {
    comment_and_transition(TransitionParams {
        ticket,
        comment: CommentParam {
            role: SYSTEM_ROLE,
            content: comment,
        },
        extra: None,
        source,
        target: TicketPhase::Failed,
        notify: NotifyPolicy::Notify,
        log_label,
        verb: "failed",
    })
    .await
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
/// When `precomputed_porcelain` is `Some`, the caller has already verified git
/// availability and run `git status --porcelain`. The function skips redundant
/// git checks and uses the provided output directly. When `None`, it performs
/// the full git availability check and status call.
///
/// Checks for a dirty working tree via `git status --porcelain`:
/// - **Clean tree:** skips commit, transitions directly to Done with notification.
/// - **Dirty tree:** runs `git commit -m "<ticket title>"` via [`crate::git_commands::run_git_commit`].
/// - **Commit failure:** ticket stays in `source`, poller retries next cycle.
/// - **Git unavailable:** delegated to [`transition_ticket_to_done_if_git_unavailable`].
async fn finalize_ticket_from_phase(
    ticket: Ticket,
    ws: Workspace,
    source: TicketPhase,
    precomputed_porcelain: Option<&str>,
) {
    let repo_path = ws.as_path();
    let phase_label = source.as_ref();

    // If no precomputed output was provided, check git availability first.
    if precomputed_porcelain.is_none()
        && transition_ticket_to_done_if_git_unavailable(&ticket, repo_path, source).await
    {
        return;
    }

    let has_changes = match precomputed_porcelain {
        Some(out) => !out.trim().is_empty(),
        None => match crate::git_commands::run_git_status(repo_path).await {
            Ok(output) => !output.trim().is_empty(),
            Err(e) => {
                warn!(
                    ticket = %ticket.id,
                    error = %e,
                    "Failed to check git status — staying in {phase_label} for retry"
                );
                return;
            }
        },
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
        dispatch_notification(
            ticket,
            source,
            TicketPhase::Done,
            notify_policy,
            "finalize_commit_and_transition",
            None,
        )
        .await;
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
///
/// Runs `git status --porcelain` exactly once — the output is shared with
/// [`finalize_ticket_from_phase`] to avoid a redundant ~50–500ms subprocess
/// call and a redundant git-availability check.
async fn handle_qa_passed(ticket: Ticket, ws: Workspace) {
    let repo_path = ws.as_path();

    // Git not available or not a git repo — transition to Done directly.
    if transition_ticket_to_done_if_git_unavailable(&ticket, repo_path, TicketPhase::QaPassed).await
    {
        return;
    }

    // Run git status once. The porcelain output is used for both untracked-file
    // detection (to decide whether to dispatch sanitation) and commit logic
    // (when passing to finalize_ticket_from_phase). Sharing the output avoids a
    // redundant ~50–500ms subprocess call.
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
        // No untracked/new files — finalize to Done, reusing the porcelain
        // output to skip the redundant git availability check and git status
        // call inside finalize_ticket_from_phase.
        finalize_ticket_from_phase(ticket, ws, TicketPhase::QaPassed, Some(&porcelain)).await;
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
    let reason_str = format!("{SANITATION_FAILED_PREFIX} — {reason}");
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
        if !comment_and_transition(TransitionParams {
            ticket,
            comment: CommentParam {
                role: Role::Sanitation.as_str(),
                content: &comment,
            },
            extra: None,
            source: TicketPhase::InSanitation,
            target: TicketPhase::SanitationPassed,
            notify: NotifyPolicy::Buffer,
            log_label: "Sanitation",
            verb: "passed",
        })
        .await
        {
            return;
        }

        info!(
            ticket = %ticket.id,
            "Sanitation passed — transitioned to SanitationPassed",
        );
    } else {
        let garbage_list = verdict.garbage_files.join("\n- ");
        let comment = format!(
            "🗑️ Sanitation failed — garbage files detected:\n- {garbage_list}\n\nRationale: {rationale}
            
            These files might have been accidentally generated by other agents in the workspace for testing purposes. Engineer needs to clean them up if they are not required in the scope of the ticket.",
            rationale = verdict.rationale,
        );
        // Pre-build the system comment so we can pass it into the transaction.
        let sys_comment = format!(
            "{SANITATION_FAILED_PREFIX} — garbage files: {count}",
            count = verdict.garbage_files.len(),
        );

        // Write both comments and transition atomically via
        // [`comment_and_transition`], which wraps all writes in a single
        // transaction. This matches the pattern used by all other verdict paths.
        if !comment_and_transition(TransitionParams {
            ticket,
            comment: CommentParam {
                role: Role::Sanitation.as_str(),
                content: comment.as_str(),
            },
            extra: Some(CommentParam {
                role: SYSTEM_ROLE,
                content: sys_comment.as_str(),
            }),
            source: TicketPhase::InSanitation,
            target: TicketPhase::ReadyForDevelopment,
            notify: NotifyPolicy::Buffer,
            log_label: "Sanitation",
            verb: "failed",
        })
        .await
        {
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
                // Output is already credential-scrubbed by ShellTool's
                // finish_shell_output, so no further scrubbing needed.
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

/// Transition from `InDiagnostics` to `DiagnosticsDone` with a comment
/// explaining the outcome (completed, skipped, or errored).
///
/// This is a thin wrapper around [`comment_and_transition`] that fills in
/// the shared parameters (`DIAGNOSTICS_ROLE`, source/target phases,
/// notify policy, and log label), leaving only the caller-specific
/// `comment` text and `verb` string.
///
/// Returns `true` if the transition succeeded, `false` otherwise.
///
/// This was previously named `finish_diagnostics` — renamed to match the
/// `transition_ticket_to_*` naming convention used by the sibling wrappers
/// [`transition_ticket_to_done`], [`transition_ticket_to_failed`], etc.
#[must_use]
async fn transition_ticket_to_diagnostics_done(ticket: &Ticket, comment: &str, verb: &str) -> bool {
    comment_and_transition(TransitionParams {
        ticket,
        comment: CommentParam {
            role: DIAGNOSTICS_ROLE,
            content: comment,
        },
        extra: None,
        source: TicketPhase::InDiagnostics,
        target: TicketPhase::DiagnosticsDone,
        notify: NotifyPolicy::Buffer,
        log_label: "Diagnostics",
        verb,
    })
    .await
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
#[allow(clippy::too_many_lines)]
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

    // 1. Load diagnostics commands for this workspace.
    match crate::workspace::store().get_diagnostics(&ws.name).await {
        Ok(Some(cmds)) if !cmds.is_empty() => {
            // 2. Run commands sequentially in the prescribed order.
            let (comment, all_passed) = run_diagnostics_commands(&cmds, &ws).await;

            // Post-run check: verify ticket hasn't been moved externally while
            // diagnostics commands ran. Consistent with dispatch_engineer and
            // dispatch_sanitation.
            if !is_ticket_in_phase(&ticket.id, TicketPhase::InDiagnostics).await {
                return;
            }

            if all_passed {
                // Path C1: All diagnostics passed — transition to DiagnosticsDone.
                let _ = transition_ticket_to_diagnostics_done(&ticket, &comment, "completed").await;
            } else {
                // Path C2: Diagnostics failed — write the failure comment and
                // bounce back to development for rework.
                //
                // Uses `comment_and_transition` (which wraps both writes in a
                // single transaction). `pipeline_reservation` is automatically
                // derived as `Some(true)` when bouncing back to
                // ReadyForDevelopment, ensuring the ticket gets priority
                // re-dispatch over fresh tickets.
                // If the process crashes before the transaction commits, neither
                // the comment nor the transition is persisted — the ticket stays
                // in InDiagnostics and reset_inflight_tickets on the next startup
                // handles recovery without inflating the circuit breaker counter.
                if !comment_and_transition(TransitionParams {
                    ticket: &ticket,
                    comment: CommentParam {
                        role: DIAGNOSTICS_ROLE,
                        content: &comment,
                    },
                    extra: None,
                    source: TicketPhase::InDiagnostics,
                    target: TicketPhase::ReadyForDevelopment,
                    notify: NotifyPolicy::Buffer,
                    log_label: "Diagnostics",
                    verb: "failed",
                })
                .await
                {
                    return;
                }

                info!(
                    ticket = %ticket.id,
                    "Diagnostics failed — pipeline reservation set for rework priority",
                );
            }
        }
        Ok(_) => {
            // Path B: No diagnostics commands configured (or empty list) — skip.
            let _ = transition_ticket_to_diagnostics_done(
                &ticket,
                "No diagnostics commands are configured for this workspace — diagnostics skipped.",
                "skipped (no commands configured)",
            )
            .await;
        }
        Err(e) => {
            // Path A: DB error loading diagnostics — log and skip.
            warn!(
                ticket = %ticket.id,
                error = %e,
                "Failed to load diagnostics for workspace — transitioning to DiagnosticsDone",
            );
            let _ = transition_ticket_to_diagnostics_done(
                &ticket,
                &format!("Could not load diagnostics commands due to a database error: {e}"),
                "skipped (DB error)",
            )
            .await;
        }
    }
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
            BoardStore::add_comment_tx(tx, ticket_id, &role_label, &comment).await?;
        }
    }
    Ok(())
}

/// Shared orchestration skeleton for dispatch functions that spawn parallel
/// agents and then process results via extraction.
///
/// 1. Spawns agents via [`run_parallel_agents`]
/// 2. Runs a post-dispatch phase check (guard against race conditions)
///
/// The pre-dispatch circuit-breaker guard is handled centrally by
/// [`spawn_dispatch`] — this function only needs the post-check.
///
/// Returns `None` if the post-check failed (ticket moved externally).
/// Returns `Some(results)` when it's safe to process the verdicts.
///
/// Used by [`dispatch_backlog_analysts`] and [`dispatch_verifiers`].
async fn dispatch_parallel_agents(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    expected_phase: TicketPhase,
    role: Role,
    prompt: &str,
    extraction_prompt: &str,
) -> Option<Vec<ParallelVerdict>> {
    let results = run_parallel_agents(ticket, ws, role, prompt, extraction_prompt).await;
    if !is_ticket_in_phase(&ticket.id, expected_phase).await {
        return None;
    }
    Some(results)
}

// ── Backlog Analysis ──────────────────────────────────────────────────

/// Spawn 3 parallel analyst agents to research a backlog ticket.
/// All verdicts are recorded as comments, then the ticket transitions to:
/// - Planning (notify) when ALL analysts pass (≥ `ANALYST_PASS_THRESHOLD`/10)
/// - Planning (notify) when any analyst fails, with a comment listing the counts
///
/// The circuit-breaker guard is handled centrally by [`spawn_dispatch`].
async fn dispatch_backlog_analysts(ticket: Arc<Ticket>, ws: Workspace) {
    let prompt_key = if ticket.reporter == Role::Maintainer.as_str() {
        "analyze/maintainer_ticket.md"
    } else {
        "analyze/manager_ticket.md"
    };
    let message = load_prompt(prompt_key);
    let extraction_prompt = load_prompt("extraction/analyst.md");
    let Some(results) = dispatch_parallel_agents(
        &ticket,
        &ws,
        TicketPhase::Analysis,
        Role::Analyst,
        &message,
        &extraction_prompt,
    )
    .await
    else {
        return;
    };

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

    // Write verdict comments, summary comment, and transition in a single
    // transaction so a crash between verdict comments and the transition
    // cannot leave orphan comments.
    //
    // We use [`BoardStore::transition_to_tx`] (no agent cancellation) rather
    // than [`BoardStore::transition_to`] (which cancels agents). This is safe
    // because analyst agents have already completed — no agents are running
    // on this ticket at this point.
    if let Err(e) = crate::turso::with_tx(
        &board().conn,
        &ticket.id,
        "analyst verdict processing",
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
            BoardStore::transition_to_tx(
                tx,
                &ticket.id,
                Some(TicketPhase::Analysis),
                TicketPhase::Planning,
                None,
            )
            .await?;
            Ok(())
        },
    )
    .await
    {
        warn_transition_failed(
            ticket,
            TicketPhase::Analysis,
            TicketPhase::Planning,
            "Analyst",
            "completed",
            &e,
        );
        return;
    }

    // Notification fires after the transaction commits so the user
    // always sees the new phase in the database.
    dispatch_notification(
        ticket,
        TicketPhase::Analysis,
        TicketPhase::Planning,
        NotifyPolicy::Notify,
        "Analyst",
        None,
    )
    .await;

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

/// Shared circuit breaker skeleton: fetch comments, count via
/// [`CircuitBreakerKind::trip_count`], compare to [`CircuitBreakerKind::threshold`],
/// add a system comment via [`CircuitBreakerKind::comment`], then transition to
/// [`TicketPhase::Failed`].
///
/// All three concrete breakers (General, Sanitation, Diagnostics) delegate to this
/// helper, supplying their variant logic via the [`CircuitBreakerKind`] enum. This
/// eliminates ~80% structural duplication while preserving exact behavioral semantics.
///
/// The Manager is notified via [`transition_ticket_to_failed`] when the ticket
/// transitions to [`TicketPhase::Failed`].
///
/// # Self-counting prevention
///
/// Each breaker variant naturally excludes its own trip comment from counting:
///
/// * **General breaker** — counts all comments via `comments.len()`; it prevents
///   re-dispatch by transitioning to the terminal `Failed` phase before the
///   breaker could re-read the same trip comment.
/// * **Sanitation breaker** — filters comments by role `"system"` and content
///   containing [`SANITATION_FAILED_PREFIX`], but trip comments use different
///   text, so they are never counted.
/// * **Diagnostics breaker** — filters comments by role `"diagnostics"` and content
///   starting with [`DIAGNOSTICS_COMMENT_PREFIX`] and containing [`DIAGNOSTICS_FAILED_MARKER`];
///   trip comments always use role `SYSTEM_ROLE` (set by this function), so they
///   are never counted.
///
/// See each variant's [`CircuitBreakerKind::trip_count`] implementation for
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
///   to succeed (passed to [`transition_ticket_to_failed`] as the `source` parameter).
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

    let count = kind.trip_count(&comments);
    let threshold = kind.threshold();

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

    if transition_ticket_to_failed(
        ticket,
        source_phase,
        &kind.comment(count),
        &format!("{log_label} circuit breaker"),
    )
    .await
    {
        drain_ready_for_development_siblings(ticket).await;
    }

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
    let (target, notify, verb) = if all_failed {
        (TicketPhase::Failed, NotifyPolicy::Notify, "failed")
    } else if any_failed {
        (
            TicketPhase::ReadyForDevelopment,
            NotifyPolicy::Buffer,
            "failed",
        )
    } else {
        (verifier.success_phase, NotifyPolicy::Buffer, "completed")
    };

    let pipeline_reservation = (target == TicketPhase::ReadyForDevelopment).then_some(true);

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

    // Write per-agent verdict comments AND the transition in a single
    // transaction so a crash between the two cannot leave orphan comments.
    //
    // Note: we use [`BoardStore::transition_to_tx`] (which does NOT cancel
    // registered agents) rather than [`BoardStore::transition_to`] (which
    // does via `execute_and_cancel`). This is safe because the verifier
    // agents have already completed and produced verdicts — no agents
    // should be running on this ticket at this point. The same reasoning
    // applies to [`comment_and_transition`] per its documentation.
    if let Err(e) = crate::turso::with_tx(
        &board().conn,
        &ticket.id,
        "verifier verdict processing",
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
            // (matching what transition_ticket_to_failed did via comment_and_transition).
            if let Some(ref fc) = failure_comment {
                BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, fc).await?;
            }

            BoardStore::transition_to_tx(
                tx,
                &ticket.id,
                Some(verifier.source_phase),
                target,
                pipeline_reservation,
            )
            .await?;
            Ok(())
        },
    )
    .await
    {
        warn_transition_failed(
            ticket,
            verifier.source_phase,
            target,
            verifier.log_label,
            verb,
            &e,
        );
        return;
    }

    // Notification fires after the transaction commits so the user
    // always sees the new phase in the database.
    dispatch_notification(
        ticket,
        verifier.source_phase,
        target,
        notify,
        verifier.log_label,
        failure_comment.as_deref(),
    )
    .await;

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
    let Some(results) = dispatch_parallel_agents(
        &ticket,
        &ws,
        vi.source_phase,
        vi.role,
        &prompt,
        &extraction_prompt,
    )
    .await
    else {
        return;
    };

    process_verifier_verdicts(&ticket, &results, vi).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::test::make_ticket;
    use crate::util::test::{
        create_test_workspace, expect_ticket, expect_ticket_phase, init_management_test_stores,
        init_test_stores,
    };
    use crate::workspace::test_ws_named;
    use strum::IntoEnumIterator;

    /// All non-General circuit breaker variants must have a threshold strictly
    /// less than [`CircuitBreakerKind::General`]'s threshold.
    ///
    /// ## Rationale
    ///
    /// - **Sanitation breaker** (`threshold = 3`): must trip before the general
    ///   breaker (`threshold = 30`), otherwise a ticket could accumulate 30+
    ///   comments during repeated sanitation loops without tripping.
    /// - **Diagnostics breaker** (`threshold = 4`): must also trip before the
    ///   general breaker. This is a conservative approximation — the general
    ///   breaker counts *all* comments (not just diagnostics), but guaranteeing
    ///   that diagnostics-only chatter cannot bypass the general breaker prevents
    ///   pathological ticket growth from repeated diagnostic cycles.
    ///
    /// > **Note:** This test replaces previous compile-time `const _: () = assert!(...)`
    /// > declarations that provided slightly earlier feedback (at `cargo build` vs
    /// > `cargo test`). The test is strictly more robust: it uses `IntoEnumIterator`
    /// > to exhaustively check all non-General variants, so newly added variants
    /// > are automatically covered without manual line additions.
    #[test]
    fn all_non_general_circuit_breakers_trip_before_general() {
        let general = CircuitBreakerKind::General.threshold();
        for kind in CircuitBreakerKind::iter() {
            if kind == CircuitBreakerKind::General {
                continue;
            }
            assert!(
                kind.threshold() < general,
                "{kind:?}.threshold() ({}) must be less than General.threshold() ({general})",
                kind.threshold(),
            );
        }
    }

    /// Verify that when the circuit breaker trips on a ticket, all other
    /// ReadyForDevelopment tickets in the same workspace are moved to Planning.
    /// Tickets in other workspaces must not be affected.
    #[tokio::test]
    async fn circuit_breaker_moves_other_ready_for_development_tickets_to_planning() {
        init_management_test_stores().await;

        let ws_a = test_ws_named("/ws_a", "ws_a");
        let ws_b = test_ws_named("/ws_b", "ws_b");

        // Create ticket A in workspace A — this will trip the circuit breaker.
        let trip_id = make_ticket(
            board(),
            &ws_a,
            "Trip Ticket",
            TicketPhase::ReadyForDevelopment,
        )
        .await;

        // Create ticket B in workspace A — this should be moved to Planning when A trips.
        let victim_id = make_ticket(
            board(),
            &ws_a,
            "Victim Ticket",
            TicketPhase::ReadyForDevelopment,
        )
        .await;

        // Create ticket C in workspace B — this must NOT be moved.
        let other_ws_id = make_ticket(
            board(),
            &ws_b,
            "Other Workspace Ticket",
            TicketPhase::ReadyForDevelopment,
        )
        .await;

        // Add comments to ticket A so the circuit breaker has something to count
        // (CircuitBreakerKind::General.threshold() + 1 = 31 comments, enough to trip).
        for i in 0..=CircuitBreakerKind::General.threshold() {
            board()
                .add_comment(&trip_id, SYSTEM_ROLE, &format!("Comment {i}"))
                .await
                .expect("add_comment to A");
        }

        // Fetch ticket A and trip the circuit breaker.
        let ticket_a = expect_ticket(board(), &trip_id).await;

        let tripped = try_trip_circuit_breaker(
            &ticket_a,
            TicketPhase::ReadyForDevelopment,
            CircuitBreakerKind::General,
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
    }

    /// Verify that `record_verdict_comments_tx` correctly writes comments
    /// based on verdict filter.
    #[tokio::test]
    async fn record_verdict_comments_filtering() {
        init_test_stores().await;

        let ticket_id = make_ticket(
            board(),
            &test_ws_named("/tmp/test", "test"),
            "Test",
            TicketPhase::Backlog,
        )
        .await;

        // ── FailingOnly with all-passing verdicts ──
        // Should produce 0 comments (nothing to write).
        let results = vec![pass_result()];
        crate::turso::with_tx(
            &board().conn,
            &ticket_id,
            "test verdict comments",
            async |tx| {
                record_verdict_comments_tx(
                    tx,
                    &ticket_id,
                    &results,
                    Role::Reviewer.as_str(),
                    VerdictFilter::FailingOnly,
                )
                .await
            },
        )
        .await
        .expect("record_verdict_comments_tx should succeed");

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
        let results = vec![fail_result()];
        crate::turso::with_tx(
            &board().conn,
            &ticket_id,
            "test verdict comments",
            async |tx| {
                record_verdict_comments_tx(
                    tx,
                    &ticket_id,
                    &results,
                    Role::Reviewer.as_str(),
                    VerdictFilter::FailingOnly,
                )
                .await
            },
        )
        .await
        .expect("record_verdict_comments_tx should succeed");

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
            analyst_verdict(10, "Excellent analysis.", &[]),
            analyst_verdict(4, "Needs more research.", &["Missing citations"]),
        ];
        crate::turso::with_tx(
            &board().conn,
            &ticket_id,
            "test verdict comments",
            async |tx| {
                record_verdict_comments_tx(
                    tx,
                    &ticket_id,
                    &results,
                    Role::Analyst.as_str(),
                    VerdictFilter::All,
                )
                .await
            },
        )
        .await
        .expect("record_verdict_comments_tx should succeed");

        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert_eq!(
            comments.len(),
            3,
            "All filter should write both verdicts (total 3)"
        );
        assert_eq!(comments[1].role, "analyst_1");
        assert_eq!(comments[2].role, "analyst_2");
    }

    // ── transition_ticket_to_done — conditional notification ─────────

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
        create_test_workspace(&ws_path, &ws_name).await
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
        let ticket_id = make_ticket(board(), &ws, title, phase).await;
        (ws, ticket_id)
    }

    /// Verify the Buffer → Notify + drain sequence across two QaPassed tickets
    /// via `transition_ticket_to_done`: the first one buffers, the last one
    /// notifies and drains the buffer.
    #[tokio::test]
    async fn transition_ticket_to_done_buffer_and_notify() {
        let ws = setup_db_workspace("drains_buffer").await;

        // Two QaPassed tickets in the same workspace
        let first_id = make_ticket(board(), &ws, "Ticket A", TicketPhase::QaPassed).await;
        let second_id = make_ticket(board(), &ws, "Ticket B", TicketPhase::QaPassed).await;

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

        // Verify both tickets are Done and have SYSTEM_ROLE comments
        for (id, label) in [(&first_id, "A"), (&second_id, "B")] {
            let t = board()
                .get_ticket(id)
                .await
                .expect("get_ticket")
                .unwrap_or_else(|| panic!("ticket {label} exists"));
            assert_eq!(t.phase, TicketPhase::Done, "Ticket {label} should be Done");

            // Each Done transition should have written a SYSTEM_ROLE comment
            let comments = board().get_comments(id).await.expect("get_comments");
            assert!(
                comments.iter().any(|c| c.role == SYSTEM_ROLE),
                "Ticket {label}: expected SYSTEM_ROLE comment from transition_ticket_to_done"
            );
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

    // ── try_trip_circuit_breaker — sanitation counting ──

    /// Verify that the sanitation circuit breaker counting logic works correctly.
    #[tokio::test]
    async fn sanitation_breaker_counts_failures() {
        init_management_test_stores().await;
        let ticket_id = make_ticket(
            board(),
            &test_ws_named("/tmp/test", "san_breaker_test"),
            "Sanitation Breaker Test",
            TicketPhase::InSanitation,
        )
        .await;

        // Add 2 sanitation failure comments (below threshold of 3).
        for _ in 0..2 {
            add_sanitation_failure(&ticket_id).await;
        }

        let ticket = expect_ticket(board(), &ticket_id).await;

        assert!(
            !try_trip_circuit_breaker(
                &ticket,
                TicketPhase::InSanitation,
                CircuitBreakerKind::Sanitation,
                "Sanitation",
            )
            .await,
            "Should NOT trip with 2 failures (threshold: 3)"
        );

        // Add a 3rd failure comment (should still not trip — reads fresh comments
        // from DB, not the stale in-memory ticket.comments).
        add_sanitation_failure(&ticket_id).await;

        // ... actually 3 <= 3 means the breaker does NOT trip yet.
        // The breaker trips when count > threshold, i.e., at 4 failures.
        // Add a 4th failure.
        add_sanitation_failure(&ticket_id).await;

        // Now with 4 failures, should trip (4 > 3).
        let tripped = try_trip_circuit_breaker(
            &ticket,
            TicketPhase::InSanitation,
            CircuitBreakerKind::Sanitation,
            "Sanitation",
        )
        .await;
        assert!(tripped, "Should trip with 4 failures (threshold: 3, 4 > 3)");

        // Verify the ticket is now Failed
        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase,
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

    /// Helper: a `ParallelVerdict` with no response.
    fn no_verdict() -> ParallelVerdict {
        ParallelVerdict::NoResponse
    }

    /// Add a sanitation failure comment for circuit breaker testing.
    async fn add_sanitation_failure(ticket_id: &str) {
        let _ = board()
            .add_comment(
                ticket_id,
                SYSTEM_ROLE,
                &format!("{SANITATION_FAILED_PREFIX} — garbage files: 1"),
            )
            .await;
    }

    /// Helper: wrap a passing verdict (reviewer/QA flow).
    fn pass_result() -> ParallelVerdict {
        ParallelVerdict::Verdict(pass_verdict())
    }

    /// Helper: wrap a failing verdict (reviewer/QA flow).
    fn fail_result() -> ParallelVerdict {
        ParallelVerdict::Verdict(fail_verdict())
    }

    /// Helper: construct an analyst verdict with explicit score / critique / issues.
    fn analyst_verdict(score: u8, critique: &str, issues: &[&str]) -> ParallelVerdict {
        ParallelVerdict::Verdict(crate::Verdict {
            score,
            critique: Some(critique.into()),
            issues_detected: issues.iter().map(|&s| s.into()).collect(),
        })
    }

    // ── process_verifier_verdicts — verdict processing ─────────────────────

    /// Verify all verdict-processing outcomes:
    /// - All failed → Failed
    /// - Any failed → bounce-back to ReadyForDevelopment with pipeline reservation
    /// - All passed (Reviewer) → Reviewed
    /// - All passed (QA) → QaPassed
    #[tokio::test]
    async fn process_verifier_verdicts_cases() {
        struct Case {
            name: &'static str,
            ws_suffix: &'static str,
            title: &'static str,
            phase: TicketPhase,
            results: Vec<ParallelVerdict>,
            vi: VerifierInfo,
            expected_phase: TicketPhase,
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
                expected_phase: TicketPhase::Failed,
                expected_pipeline_reservation: false,
            },
            Case {
                name: "any failed -> bounce-back with pipeline reservation",
                ws_suffix: "vp_any_fail",
                title: "VP Any Failed",
                phase: TicketPhase::InReview,
                results: vec![pass_result(), fail_result(), pass_result()],
                vi: REVIEWER_VI,
                expected_phase: TicketPhase::ReadyForDevelopment,
                expected_pipeline_reservation: true,
            },
            Case {
                name: "all passed -> Reviewed",
                ws_suffix: "vp_all_pass",
                title: "VP All Pass",
                phase: TicketPhase::InReview,
                results: vec![pass_result(), pass_result(), pass_result()],
                vi: REVIEWER_VI,
                expected_phase: TicketPhase::Reviewed,
                expected_pipeline_reservation: false,
            },
            Case {
                name: "all passed (QA) -> QaPassed",
                ws_suffix: "vp_qa_pass",
                title: "VP QA Pass",
                phase: TicketPhase::InQa,
                results: vec![pass_result(), pass_result(), pass_result()],
                vi: QA_VI,
                expected_phase: TicketPhase::QaPassed,
                expected_pipeline_reservation: false,
            },
        ];

        for case in &cases {
            let ticket_id = make_ticket(
                board(),
                &test_ws_named("/tmp/test", case.ws_suffix),
                case.title,
                case.phase,
            )
            .await;

            let ticket = expect_ticket(board(), &ticket_id).await;

            process_verifier_verdicts(&ticket, &case.results, case.vi).await;

            let ticket = expect_ticket(board(), &ticket_id).await;
            assert_eq!(
                ticket.phase, case.expected_phase,
                "case {}: expected phase {:?}, got {:?}",
                case.name, case.expected_phase, ticket.phase,
            );
            assert_eq!(
                ticket.pipeline_reservation, case.expected_pipeline_reservation,
                "case {}: expected pipeline_reservation={}, got {}",
                case.name, case.expected_pipeline_reservation, ticket.pipeline_reservation,
            );
        }
    }

    // ── try_trip_circuit_breaker — general circuit breaker ────────

    /// Verify the circuit breaker trips at the threshold boundary:
    /// - `> CircuitBreakerKind::General.threshold()` comments → trips (ticket → Failed)
    /// - `= CircuitBreakerKind::General.threshold()` comments → does NOT trip
    ///
    /// When the breaker trips, also verifies the trip comment contains the
    /// "circuit breaker" marker as produced by [`CircuitBreakerKind::comment`].
    #[tokio::test]
    async fn circuit_breaker_comment_boundary() {
        struct Case {
            name: &'static str,
            ws_suffix: &'static str,
            title: &'static str,
            comment_count: usize,
            expected_trip: bool,
            expected_phase: TicketPhase,
        }

        init_management_test_stores().await;

        let cases = [
            Case {
                name: "> threshold trips",
                ws_suffix: "cb_thresh",
                title: "CB Threshold",
                comment_count: CircuitBreakerKind::General.threshold() + 1,
                expected_trip: true,
                expected_phase: TicketPhase::Failed,
            },
            Case {
                name: "= threshold does not trip",
                ws_suffix: "cb_no_trip",
                title: "CB No Trip",
                comment_count: CircuitBreakerKind::General.threshold(),
                expected_trip: false,
                expected_phase: TicketPhase::InReview,
            },
        ];

        for case in &cases {
            let ticket_id = make_ticket(
                board(),
                &test_ws_named("/tmp/test", case.ws_suffix),
                case.title,
                TicketPhase::InReview,
            )
            .await;

            for i in 0..case.comment_count {
                board()
                    .add_comment(&ticket_id, "user", &format!("Comment {i}"))
                    .await
                    .expect("add_comment");
            }

            let ticket = expect_ticket(board(), &ticket_id).await;

            let tripped = try_trip_circuit_breaker(
                &ticket,
                TicketPhase::InReview,
                CircuitBreakerKind::General,
                "test",
            )
            .await;
            assert_eq!(
                tripped, case.expected_trip,
                "case {}: expected trip={}, got tripped={}",
                case.name, case.expected_trip, tripped,
            );

            let phase = expect_ticket_phase(board(), &ticket_id).await;
            assert_eq!(
                phase, case.expected_phase,
                "case {}: expected phase {:?}, got {:?}",
                case.name, case.expected_phase, phase,
            );

            // When the breaker trips, verify the trip comment contains the
            // circuit breaker marker as produced by `CircuitBreakerKind::comment`.
            if tripped {
                let comments = board()
                    .get_comments(&ticket_id)
                    .await
                    .expect("get_comments");
                let has_marker = comments
                    .iter()
                    .any(|c| c.content.to_lowercase().contains("circuit breaker"));
                assert!(
                    has_marker,
                    "case {}: trip comment must contain circuit breaker marker",
                    case.name,
                );
            }
        }
    }

    // ── process_analyst_verdicts — analyst scoring and transitions ─────────

    /// Verify process_analyst_verdicts across all outcomes:
    /// - All analysts pass → Planning with "All LGTM" summary
    /// - Partial fail → Planning with "blockers" summary
    /// - No verdicts → Planning with "no analysis" summary
    #[tokio::test]
    async fn process_analyst_verdicts_cases() {
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
                    analyst_verdict(10, "Great analysis.", &[]),
                    analyst_verdict(9, "Solid work.", &[]),
                    analyst_verdict(8, "Good analysis.", &[]),
                ],
                expected_comment_substring: "All LGTM",
            },
            Case {
                name: "partial fail -> Planning with blockers",
                ws_suffix: "an_partial",
                title: "Analyst Partial Fail",
                results: vec![
                    analyst_verdict(10, "Great.", &[]),
                    analyst_verdict(3, "Poor analysis.", &["Missing data"]),
                    analyst_verdict(8, "Decent.", &["Minor issue"]),
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
            let ticket_id = make_ticket(
                board(),
                &test_ws_named("/tmp/test", case.ws_suffix),
                case.title,
                TicketPhase::Analysis,
            )
            .await;

            let ticket = expect_ticket(board(), &ticket_id).await;

            process_analyst_verdicts(&ticket, &case.results).await;

            let phase = expect_ticket_phase(board(), &ticket_id).await;
            assert_eq!(
                phase,
                TicketPhase::Planning,
                "case {}: expected Planning, got {:?}",
                case.name,
                phase,
            );

            let comments = board()
                .get_comments(&ticket_id)
                .await
                .expect("get_comments");

            let system = comments
                .iter()
                .find(|c| c.role == SYSTEM_ROLE)
                .unwrap_or_else(|| {
                    panic!("case {}: system summary comment should exist", case.name)
                });
            assert!(
                system.content.contains(case.expected_comment_substring),
                "case {}: system comment should contain {:?}, got: {}",
                case.name,
                case.expected_comment_substring,
                system.content,
            );
        }
    }

    // ── handle_qa_passed — QA → Done path ───────────────────────────────

    /// handle_qa_passed first checks whether git is available and whether the
    /// workspace path is a git repo. In test environments git may exist, but
    /// the workspace path is deliberately not a git repo, so the function
    /// transitions directly to Done without committing.
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

        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase,
            TicketPhase::Done,
            "QA passed should eventually transition to Done"
        );

        // Verify a SYSTEM_ROLE comment was written capturing the reason.
        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert!(
            comments
                .iter()
                .any(|c| c.role == SYSTEM_ROLE && c.content.contains("without commit")),
            "Expected a SYSTEM_ROLE comment explaining why no commit was made"
        );
    }

    /// handle_qa_passed with untracked files present should claim the ticket
    /// to InSanitation and dispatch a sanitation agent. Creates a real git repo
    /// with an untracked file to exercise the full claim path.
    #[tokio::test]
    async fn handle_qa_passed_untracked_files_to_insanitation() {
        // Skip if git is not installed — the test cannot create a repo.
        if !crate::git_commands::git_is_installed().await {
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

        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase,
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

    // ── process_sanitation_verdict — verdict processing ──────────────────

    /// Verify both branches of `process_sanitation_verdict`:
    /// - pass=true → SanitationPassed with a sanitation comment
    /// - pass=false → ReadyForDevelopment with pipeline reservation,
    ///   a sanitation comment, and a system circuit-breaker comment
    #[tokio::test]
    async fn process_sanitation_verdict_cases() {
        init_management_test_stores().await;

        // Case 1: pass=true → SanitationPassed
        let ws_pass = test_ws_named("/tmp/test", "sv_pass");
        let pass_id = make_ticket(board(), &ws_pass, "SV Pass", TicketPhase::InSanitation).await;
        let ticket_pass = expect_ticket(board(), &pass_id).await;

        let pass_verdict = crate::SanitationVerdict {
            pass: true,
            garbage_files: vec![],
            rationale: "All files are legitimate project files.".into(),
        };

        process_sanitation_verdict(&ticket_pass, pass_verdict).await;

        let phase = expect_ticket_phase(board(), &pass_id).await;
        assert_eq!(
            phase,
            TicketPhase::SanitationPassed,
            "pass=true should transition to SanitationPassed, got {phase:?}",
        );

        // Verify a sanitation comment was added
        let comments = board().get_comments(&pass_id).await.expect("get_comments");
        let has_sanitation_comment = comments.iter().any(|c| c.role == Role::Sanitation.as_str());
        assert!(
            has_sanitation_comment,
            "pass=true should add a sanitation comment",
        );

        // No system comment (only added in fail path)
        let has_system_comment = comments.iter().any(|c| c.role == SYSTEM_ROLE);
        assert!(
            !has_system_comment,
            "pass=true should not add a system comment",
        );

        // Case 2: pass=false → ReadyForDevelopment with pipeline reservation
        let ws_fail = test_ws_named("/tmp/test", "sv_fail");
        let fail_id = make_ticket(board(), &ws_fail, "SV Fail", TicketPhase::InSanitation).await;
        let ticket_fail = expect_ticket(board(), &fail_id).await;

        let fail_verdict = crate::SanitationVerdict {
            pass: false,
            garbage_files: vec!["node_modules/".into(), "tmp/scratch.js".into()],
            rationale: "These are intermediate build artifacts.".into(),
        };

        process_sanitation_verdict(&ticket_fail, fail_verdict).await;

        let ticket = expect_ticket(board(), &fail_id).await;
        assert_eq!(
            ticket.phase,
            TicketPhase::ReadyForDevelopment,
            "pass=false should bounce back to ReadyForDevelopment, got {:?}",
            ticket.phase,
        );
        assert!(
            ticket.pipeline_reservation,
            "pass=false should set pipeline_reservation=true",
        );

        // Verify a sanitation comment was added about the garbage files
        let comments = board().get_comments(&fail_id).await.expect("get_comments");
        let has_garbage_comment = comments
            .iter()
            .any(|c| c.role == Role::Sanitation.as_str() && c.content.contains("node_modules/"));
        assert!(
            has_garbage_comment,
            "pass=false should have a sanitation comment mentioning garbage files",
        );

        // Verify a system comment with SANITATION_FAILED_PREFIX was added
        let has_system_breaker = comments
            .iter()
            .any(|c| c.role == SYSTEM_ROLE && c.content.contains(SANITATION_FAILED_PREFIX));
        assert!(
            has_system_breaker,
            "pass=false should have a system comment with the circuit breaker prefix",
        );
    }

    /// Verify that when a workspace has no diagnostics commands configured,
    /// `dispatch_diagnostics` writes a no-commands comment and transitions
    /// the ticket to `DiagnosticsDone`.
    #[tokio::test]
    async fn no_diagnostics_commands_skips_to_diagnostics_done() {
        init_management_test_stores().await;

        let ws = create_test_workspace("/tmp/test_no_diag", "test_no_diag").await;

        let ticket_id = make_ticket(
            board(),
            &ws,
            "No Diagnostics Commands",
            TicketPhase::InDiagnostics,
        )
        .await;

        // NOTE: Do NOT claim the ticket beforehand — dispatch_diagnostics
        // calls claim_diagnostics internally as its first step.
        let ticket = expect_ticket(board(), &ticket_id).await;

        dispatch_diagnostics(Arc::new(ticket), ws).await;

        // Verify transition to DiagnosticsDone.
        let phase = expect_ticket_phase(board(), &ticket_id).await;
        assert_eq!(
            phase,
            TicketPhase::DiagnosticsDone,
            "ticket should be DiagnosticsDone when no diagnostics commands are configured",
        );

        // Verify a diagnostics-role comment was written explaining the skip.
        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert!(
            !comments.is_empty(),
            "should have written at least one comment",
        );
        let comment = &comments[0];
        assert_eq!(comment.role, DIAGNOSTICS_ROLE);
        assert!(
            comment
                .content
                .contains("No diagnostics commands are configured"),
            "comment should explain that no diagnostics commands are configured: {}",
            comment.content,
        );
    }

    /// Verify that when diagnostics commands fail, `dispatch_diagnostics` writes a
    /// failure comment and bounces the ticket back to `ReadyForDevelopment` with
    /// `pipeline_reservation = true`. This exercises Path C2 through the complete
    /// transaction (comment + transition), complementing the Path B test above
    /// and verifying the with_tx crash-consistency fix.
    #[tokio::test]
    async fn diagnostics_failure_bounces_to_ready_for_development() {
        init_management_test_stores().await;

        let dir = tempfile::tempdir().expect("create temp dir");
        let ws_path = dir.path().to_string_lossy().to_string();
        let ws_name = "test_diag_fail";
        let ws = create_test_workspace(&ws_path, ws_name).await;

        // Set a diagnostics command that will always fail (exit non-zero).
        let cmds = DiagnosticsCommands {
            format: Some("false".to_string()),
            format_check: None,
            lint_fix: None,
            lint: None,
            type_check: None,
            build: None,
            unit_test: None,
        };
        crate::workspace::store()
            .set_diagnostics(ws_name, &cmds, &crate::turso::now())
            .await
            .expect("set diagnostics");

        let ticket_id = make_ticket(
            board(),
            &ws,
            "Diagnostics Failure Test",
            TicketPhase::InDiagnostics,
        )
        .await;

        // NOTE: Do NOT claim the ticket beforehand — dispatch_diagnostics
        // calls claim_diagnostics internally as its first step.
        let ticket = expect_ticket(board(), &ticket_id).await;

        dispatch_diagnostics(Arc::new(ticket), ws).await;

        // Verify transition to ReadyForDevelopment with pipeline_reservation.
        let ticket = expect_ticket(board(), &ticket_id).await;
        assert_eq!(
            ticket.phase,
            TicketPhase::ReadyForDevelopment,
            "diagnostics failure should bounce to ReadyForDevelopment",
        );
        assert!(
            ticket.pipeline_reservation,
            "bounced ticket should have pipeline_reservation = true",
        );

        // Verify a DIAGNOSTICS_ROLE comment was written with the failure marker.
        let comments = board()
            .get_comments(&ticket_id)
            .await
            .expect("get_comments");
        assert!(
            !comments.is_empty(),
            "should have written at least one comment",
        );
        let has_diag_failure = comments.iter().any(|c| {
            c.role == DIAGNOSTICS_ROLE
                && c.content.contains(DIAGNOSTICS_COMMENT_PREFIX)
                && c.content.contains(DIAGNOSTICS_FAILED_MARKER)
        });
        assert!(
            has_diag_failure,
            "should have a DIAGNOSTICS_ROLE comment with the failure marker",
        );

        // Keep dir alive until after the test completes.
        drop(dir);
    }
}
