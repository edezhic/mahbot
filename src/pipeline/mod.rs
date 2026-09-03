//! Pipeline orchestration — the single-puller dispatch driver over `tickets.phase`.
//!
//! The poll loop ([`run_management`]) is the ONLY way a stage's work starts.
//! Each pipeline phase owns a short-lived `jobs` row whose `kind` equals the
//! ticket's current phase (`analysis`, `in_development`, `in_diagnostics`,
//! `in_review`, `in_qa`, `in_sanitation`); the ticket's `phase` is the sole
//! durable running truth. The puller:
//!
//! 1. claims `Backlog -> Analysis` and `Queued -> InDevelopment`,
//! 2. creates a fresh phase job for any ticket in a working phase that has none
//!    (atomic via the unique `(kind, ticket_id)` index),
//! 3. spawns the phase body that runs the stage to completion.
//!
//! On phase completion the phase body transitions the ticket to the next phase
//! and deletes its own job; the puller then creates the next phase's job on a
//! later tick. There is no phase-transition finalizer dispatch and no in-memory
//! dispatch latch — the only in-memory state is the running-body registry
//! ([`PHASE_BODIES_RUNNING`]), a runtime-transport guard that stops a second
//! phase body from being spawned on the same job during the pre-roster window.
//! A paused workspace skips claims and job creation; current work finishes
//! normally and the unpause re-drives it.

pub mod analysis;
pub mod board;
pub mod chronicle;
pub mod development;
pub mod diagnostics;
pub mod qa;
pub mod review;
pub mod sanitation;
#[cfg(test)]
mod tests;
pub(crate) mod verdict;

use std::collections::HashSet;
use std::fmt::Write;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures_util::future::join_all;
use tracing::{debug, error, info, warn};

use crate::agent::message_router;
use crate::agent::role::{DIAGNOSTICS_ROLE, SYSTEM_ROLE};
use crate::agent::{RETRY_EXHAUSTION_MARKER, run_agent};
use crate::db::TxGuard;
use crate::git::commands::{list_new_or_untracked_files, run_git_status};
use crate::pipeline::board::{BoardStore, Ticket, TicketPhase};
use crate::prompt::{load_prompt, load_prompt_sections, substitute};
use crate::session::manager_agent_id;
use crate::util::UnwrapPoison;
use crate::{Role, Workspace, WorkspaceStatus};

pub(crate) use verdict::stage_name;
pub(crate) use verdict::{
    AgentSlot, ExtractionMode, JointRound, ParallelVerdict, build_round_grouping,
    deserialize_verdict_outcome, issue_grade, process_verifier_verdicts, render_joint_comment,
    round_member_failed, serialize_verdict_outcome, validate_blocker_verification,
    validate_verdict_score,
};
pub(crate) use verdict::{QA_VI, REVIEWER_VI, VerifierInfo};

use development::finalize_engineer_stage;
use sanitation::finalize_sanitation_stage;

/// Returns the global [`BoardStore`] singleton.
#[inline]
pub(crate) fn board() -> &'static BoardStore {
    crate::pipeline::board::store()
}

/// In-memory registry of phase-body job ids that currently have a live spawned
/// body. Runtime transport only (like [`crate::agent::registry`]) — NOT durable
/// running-truth; `tickets.phase` plus the job/roster remain the authority.
/// It closes the sub-second window between a phase job's creation and its
/// body's first roster write: the unique `(kind, ticket_id)` index prevents
/// duplicate *jobs*, this registry prevents duplicate *bodies* on one job.
static PHASE_BODIES_RUNNING: OnceLock<std::sync::Mutex<HashSet<String>>> = OnceLock::new();

fn phase_bodies_running() -> &'static std::sync::Mutex<HashSet<String>> {
    PHASE_BODIES_RUNNING.get_or_init(|| std::sync::Mutex::new(HashSet::new()))
}

/// Claim the spawn slot for a phase body. Returns `false` when another poll
/// tick already owns it (a body is live on this job) — the caller must not
/// re-spawn.
fn claim_phase_body(job_id: &str) -> bool {
    phase_bodies_running()
        .lock()
        .unwrap_poison()
        .insert(job_id.to_string())
}

/// Release a phase body's spawn slot.
fn release_phase_body(job_id: &str) {
    phase_bodies_running().lock().unwrap_poison().remove(job_id);
}

/// RAII guard: releases the phase-body slot when the spawned body task exits —
/// on normal completion, panic, or cancellation.
struct PhaseBodyGuard(String);

impl Drop for PhaseBodyGuard {
    fn drop(&mut self) {
        release_phase_body(&self.0);
    }
}

/// The pipeline boot + poll driver. Runs boot recovery (replays pending
/// envelopes, resumes the non-ticket resumable kinds), reclassifies stranded
/// analyzing workspaces, then loops the single-puller poll every 100 milliseconds.
pub async fn run_management() {
    // Boot recovery scan: first statement. Replays pending envelopes and
    // returns the non-ticket jobs selected for resume (research/analyze/
    // research_cleanup). Ticket phase jobs are NOT resumed here — the puller
    // re-drives a ticket in a working phase.
    let resumable = match crate::jobs::recover_from_restart().await {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, "Boot recovery scan failed — proceeding without resume");
            Vec::new()
        }
    };

    // A workspace still 'analyzing' at startup is a crashed mid-discovery run —
    // reclassify to 'pending' so the pickup retries it. Must precede the poll.
    if let Err(e) = crate::workspace::store()
        .reclassify_analyzing_to_pending()
        .await
    {
        warn!(error = %e, "Boot recovery: failed to reclassify stranded analyzing workspaces");
    }

    // Resume selected rounds (silent background resume); all remaining variants
    // are non-ticket envelope kinds carrying job_id + workspace_name.
    for stage in resumable {
        let (job_id, workspace_name) = match &stage {
            crate::jobs::ResumableJob::Research {
                job_id,
                workspace_name,
            }
            | crate::jobs::ResumableJob::Analyze {
                job_id,
                workspace_name,
            }
            | crate::jobs::ResumableJob::Implement {
                job_id,
                workspace_name,
            }
            | crate::jobs::ResumableJob::ResearchCleanup {
                job_id,
                workspace_name,
            } => (job_id, workspace_name),
        };
        let Ok(Some(workspace)) = crate::users::resolve_workspace(workspace_name).await else {
            warn!(
                job = %job_id,
                workspace = %workspace_name,
                "Resume workspace unresolvable — job left in place (deleted only by explicit abandon)",
            );
            continue;
        };
        match stage {
            crate::jobs::ResumableJob::Research { job_id, .. } => {
                info!(job = %job_id, "Resuming research run at boot");
                tokio::spawn(async move {
                    crate::tools::research::resume_research_run(&job_id, &workspace).await;
                });
            }
            crate::jobs::ResumableJob::Analyze { job_id, .. } => {
                info!(job = %job_id, "Resuming analyze round at boot");
                tokio::spawn(async move {
                    crate::tools::analyze::resume_analyze_round(&job_id, &workspace).await;
                });
            }
            crate::jobs::ResumableJob::Implement { job_id, .. } => {
                info!(job = %job_id, "Resuming implement round at boot");
                tokio::spawn(async move {
                    crate::tools::implement::resume_implement_round(&job_id, &workspace).await;
                });
            }
            crate::jobs::ResumableJob::ResearchCleanup { job_id, .. } => {
                info!(job = %job_id, "Resuming research cleanup agent at boot");
                tokio::spawn(async move {
                    crate::research_cleanup::resume_research_cleanup(&job_id, &workspace).await;
                });
            }
        }
    }

    // 100ms cadence: claim and phase-job creation react ~10x faster than a 1s
    // tick. Per-round work stays cheap; if DB load surfaces, cheapen the round
    // rather than raise this interval.
    let interval = Duration::from_millis(100);
    loop {
        // Drain-aware: break when the graceful drain begins (in-flight rounds
        // finish; no new work is spawned).
        if !crate::shutdown::sleep_or_shutdown_or_drain(interval).await {
            break;
        }
        poll_round().await;
    }
}

/// The poll loop: run the single-puller pass over every workspace every 100 milliseconds.
pub(crate) async fn poll_round() {
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

/// Process the puller steps for a single workspace.
async fn process_single_workspace(ws: Workspace) {
    // 0. Pending-workspace pickup — a pending workspace (added without a
    // provider key, or returned to pending after provider-class discovery
    // failures) is claimed into its first discovery when the provider is
    // configured. Runs before the claim pipeline: a freshly-claimed workspace
    // re-arms the analysis pause, so no Backlog/Queued claims race the
    // discovery. A successful claim returns the fresh post-claim copy; the
    // pipeline below then runs against the claimed analyzing+paused state
    // instead of the pre-claim poll-round copy.
    let ws = match pickup_pending_workspace(&ws).await {
        Some(claimed) => claimed,
        None => ws,
    };

    // Re-read the LIVE pause state; a genuinely paused workspace must not claim
    // new work or create jobs in this round.
    if let Ok(Some(live)) = crate::workspace::store().get_by_name(&ws.name).await
        && live.paused
        && live.status == WorkspaceStatus::Ready
    {
        tracing::debug!(workspace = %ws.name, "Workspace paused mid-poll — skipping claim/dispatch for this round");
        return;
    }

    // 1. Pipeline claims — atomic source→target transitions.
    run_claim_pipeline(&ws).await;

    // 2. Puller job-creation — every ticket in a working phase gets a phase
    // job (created once, unique indexed); the phase body runs the stage.
    dispatch_working_phases(&ws).await;
}

/// Claim a pending workspace into its first discovery, returning the fresh
/// post-claim [`Workspace`] (status `analyzing`, paused) so the remaining
/// poll steps run against the claimed state; `None` when the workspace stays
/// pending (not pending / provider unconfigured / cooldown armed / claim lost).
async fn pickup_pending_workspace(ws: &Workspace) -> Option<Workspace> {
    let (generation, discover_diagnostics) = pickup_claim(ws).await?;

    info!(
        workspace = %ws.name,
        generation,
        discover_diagnostics,
        "Pickup: pending workspace claimed into discovery"
    );
    crate::workspace::spawn_workspace_discovery(ws, generation, discover_diagnostics);

    // Re-read so the following poll steps see the claimed state (analyzing +
    // paused), not the stale pre-claim copy. On a read failure the caller
    // falls back to the poll-round copy — the pipeline's pause gate still
    // blocks new-work claims on it.
    crate::workspace::store()
        .get_by_name(&ws.name)
        .await
        .ok()
        .flatten()
}

/// Decide + claim half of [`pickup_pending_workspace`], split out so tests can
/// exercise the gating and the atomic claim without spawning real discovery
/// agents. Returns `Some((discovery_generation, discover_diagnostics))` when
/// the workspace was atomically claimed, `None` when it must stay pending.
///
/// The claim deliberately does **not** gate on `ws.paused`: a `Pending`
/// workspace always carries `paused = 1` (the analysis pause written by
/// `add()` and the discovery finalizer), so gating on the paused column would
/// block every pending pickup and break the analysis-pause flow.
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
    // exists to protect.
    let discover_diagnostics = match storage.get_by_name(&ws.name).await {
        Ok(Some(fresh)) => fresh.diagnostics.is_none(),
        Ok(None) | Err(_) => ws.diagnostics.is_none(),
    };

    Some((generation, discover_diagnostics))
}

/// Create a phase job (and spawn its body) for every ticket in a working phase
/// that has no launched phase job. The unique `(kind, ticket_id)` index makes
/// the create-once claim atomic.
///
/// Uses a single snapshot of the working-phase tickets
/// ([`board::list_working_tickets`]) per poll tick, then iterates them in
/// phase-major dispatch order.
async fn dispatch_working_phases(ws: &Workspace) {
    let conn = &crate::session::store().conn;
    let tickets = match board().list_working_tickets(&ws.name).await {
        Ok(t) => t,
        Err(e) => {
            warn!(workspace = %ws.name, error = %e, "Failed to list working-phase tickets");
            return;
        }
    };
    for (phase, ticket) in tickets {
        match crate::jobs::find_phase_job(conn, &ticket.id, phase).await {
            Ok(Some(job)) => {
                // Re-drive a phase job whose body is NOT running: a crash
                // or a pause-freeze leaves the job in place with no launched
                // agents and no live body. The registry claim is atomic, so
                // only one poll tick respawns the body.
                let has_agents = crate::jobs::job_has_launched_agents(conn, &job.id)
                    .await
                    .unwrap_or(true);
                if has_agents {
                    continue;
                }
                if !claim_phase_body(&job.id) {
                    continue;
                }
                info!(ticket = %ticket.id, phase = %phase, job = %job.id, "Re-driving idle phase job");
                spawn_phase_body(phase, Arc::new(ticket), ws.clone(), job.id);
                continue;
            }
            Ok(None) => {}
            Err(e) => {
                warn!(ticket = %ticket.id, phase = %phase, error = %e, "Failed to check phase job");
                continue;
            }
        }
        let (task, role) = phase_task(&ticket, phase);
        let job_id = crate::generate_id();
        // Claim the spawn slot before inserting the job so a concurrent poll
        // tick cannot re-drive this job_id in the window before the body
        // writes its first roster row.
        claim_phase_body(&job_id);
        if let Err(e) = crate::jobs::spawn_job(
            conn,
            &job_id,
            &task,
            &ws.name,
            "",
            "",
            role,
            &[],
            &crate::jobs::SpawnChild::Phase {
                phase,
                ticket_id: ticket.id.clone(),
            },
            None,
        )
        .await
        {
            release_phase_body(&job_id);
            warn!(ticket = %ticket.id, phase = %phase, error = %e, "Failed to create phase job (concurrent creation is a no-op)");
            continue;
        }
        info!(ticket = %ticket.id, phase = %phase, job = %job_id, "Created phase job — dispatching phase body");
        spawn_phase_body(phase, Arc::new(ticket), ws.clone(), job_id);
    }
}

/// The prompt `task` and dispatch `role` for a phase job, re-derived fresh
/// from the live ticket each time the phase job is created.
fn phase_task(ticket: &Ticket, phase: TicketPhase) -> (String, Role) {
    match phase {
        TicketPhase::Analysis => (
            crate::prompt::load_prompt(if ticket.reporter == Role::Maintainer.as_str() {
                "analyze/maintainer_ticket.md"
            } else {
                "analyze/manager_ticket.md"
            }),
            Role::Analyst,
        ),
        _ => (format!("Implement ticket {}", ticket.title), Role::Engineer),
    }
}

/// Re-fetch the ticket with comments so a phase body's work message and
/// `<current-ticket>` context render the live comment thread — including
/// bounce feedback added after the last stage round. The poll loop lists
/// tickets with `LoadComments::No` to stay cheap, so the body must refresh
/// before it builds its prompt. Falls back to the poll copy on a read failure
/// or concurrent delete so a phase body is never dropped.
async fn ticket_with_comments(ticket: Arc<Ticket>) -> Arc<Ticket> {
    match board().get_ticket(&ticket.id).await {
        Ok(Some(fresh)) => Arc::new(fresh),
        Ok(None) | Err(_) => ticket,
    }
}

/// Spawn the phase body for a phase job in a detached, panic-safe task. The
/// caller has already claimed the spawn slot in [`PHASE_BODIES_RUNNING`]; the
/// RAII guard releases it on every exit path (completion, panic, cancellation).
fn spawn_phase_body(phase: TicketPhase, ticket: Arc<Ticket>, ws: Workspace, job_id: String) {
    let log_job_id = job_id.clone();
    let guard_job = job_id.clone();
    tokio::spawn(async move {
        let _guard = PhaseBodyGuard(guard_job);
        // Refresh with comments inside the task so the body's work message and
        // `<current-ticket>` block see the real comment thread (bounce feedback
        // is a ticket comment). The panic clone comes from the refreshed ticket
        // too, so crash-failure cleanup has the same freshest identity.
        let ticket = ticket_with_comments(ticket).await;
        let panic_ticket = Arc::clone(&ticket);
        let run: futures_util::future::BoxFuture<'static, ()> = match phase {
            TicketPhase::Analysis => Box::pin(analysis::run(ticket, ws, job_id)),
            TicketPhase::InDevelopment => Box::pin(development::run(ticket, ws, job_id)),
            TicketPhase::InDiagnostics => Box::pin(diagnostics::run(ticket, ws, job_id)),
            TicketPhase::InReview => Box::pin(review::run(ticket, ws, job_id)),
            TicketPhase::InQa => Box::pin(qa::run(ticket, ws, job_id)),
            TicketPhase::InSanitation => Box::pin(sanitation::run(ticket, ws, job_id)),
            _ => {
                error!(phase = %phase, "spawn_phase_body called for a non-working phase");
                return;
            }
        };
        if futures_util::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(run))
            .await
            .is_ok()
        {
            return;
        }
        error!(phase = %phase, job = %log_job_id, "Phase body panicked — resetting for a fresh attempt");
        let comment = format!("Pipeline phase {phase} crashed: the phase body panicked.");
        reset_phase_attempt(
            &panic_ticket,
            phase,
            &log_job_id,
            "pipeline phase panic",
            &comment,
        )
        .await;
    });
}

/// The single prompt-driven claim pipeline: Backlog→Analysis (5s grace) and
/// Queued→InDevelopment (pipeline-occupied enforced).
///
/// Each claim is gated on a read-only existence probe over the SAME predicate
/// as the claim (shared `claim_candidate_where` builder, shared
/// now/cutoff — no one-tick false negative). The probe is a best-effort cost
/// gate: the claim UPDATE stays the atomic arbiter, so a false-positive probe
/// just costs a failed UPDATE and a candidate that appears between probe and
/// claim waits at most one 100ms tick. Probe errors MUST fail open
/// (`unwrap_or(true)`), because a false negative would skip a real claim and
/// stall the pipeline.
async fn run_claim_pipeline(ws: &Workspace) {
    // Compute the claim-time clock once per round and share it with the probes
    // AND the claims, so a probe and its claim can never evaluate eligibility
    // against a different instant.
    let now = crate::db::now();
    let grace_cutoff = (chrono::Utc::now() - BoardStore::BACKLOG_CLAIM_GRACE).to_rfc3339();

    if board()
        .backlog_claim_candidate_exists(&ws.name, &grace_cutoff)
        .await
        .unwrap_or(true)
        && let Ok(Some(ticket)) = board()
            .claim_backlog_for_analysis(&ws.name, now.clone(), grace_cutoff)
            .await
    {
        info!(ticket = %ticket.id, workspace = %ws.name, "Claimed Backlog → Analysis");
    }
    if board()
        .queued_claim_candidate_exists(&ws.name)
        .await
        .unwrap_or(true)
        && let Ok(Some(ticket)) = board().claim_queued_for_development(&ws.name, now).await
    {
        info!(ticket = %ticket.id, workspace = %ws.name, "Claimed Queued → InDevelopment");
        let _ = crate::jobs::upsert_session_pin(
            &crate::session::store().conn,
            &ticket.id,
            &ticket.title,
            crate::jobs::RowStatus::Launched,
            Role::Engineer,
        )
        .await;
    }
}

// ── Shared phase-module constants ───────────────────────────────────────

/// Neutral reason for a phase-gate bail (transients are not misattributed).
const PHASE_GATE_BAIL_REASON: &str = "ticket not in expected phase";

/// Maximum tolerated validation-phase bounces before the ticket fails (the
/// 11th bounce fails). Enforced by the unified validation-failure bounce path.
pub const MAX_BOUNCES: usize = 10;

/// GUI bounce-badge escalation threshold: at or below this many bounces the
/// badge stays muted (the ticket id color); above it, the badge uses the
/// prereq-indicator warning yellow. Halfway to the [`MAX_BOUNCES`] breaker cap.
pub const BOUNCE_BADGE_WARNING_THRESHOLD: usize = 5;

// ── Transition + notification helpers (shared by the phase modules) ─────

/// Returns `true` if the ticket is in the expected phase (safe to proceed).
/// Returns `false` otherwise — the ticket may have been moved externally,
/// not found in the database, or a database error occurred. Fail-closed.
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

/// Bail when the ticket is no longer in the expected phase: complete the phase
/// job and return `true` so the caller stops. Guards both a phase `run` entry
/// prologue (the puller re-drove this job after a transition) and a mid-round
/// body that a fresh attempt has superseded. Returns `false` to continue.
#[must_use]
async fn guard_job_phase(ticket_id: &str, expected: TicketPhase, job_id: &str) -> bool {
    if !is_ticket_in_phase(ticket_id, expected).await {
        if let Err(e) = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await {
            warn!(job = %job_id, error = %e, "Failed to complete job after phase-moved");
        }
        return true;
    }
    false
}

/// Controls whether a ticket transition triggers an immediate Manager
/// notification or is buffered for batched delivery.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum NotifyPolicy {
    Notify,
    Buffer,
}

/// The transition context for comment+transition operations.
#[derive(Debug)]
pub(crate) struct TransitionCtx<'t, 'l> {
    ticket: &'t Ticket,
    source: TicketPhase,
    target: TicketPhase,
    notify: NotifyPolicy,
    log_label: &'l str,
    /// True when this Failed transition was a bounce-breaker drain.
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
    pub(crate) fn buffered(
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
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum FinalizeOutcome {
    Applied,
    Moved,
    Failed,
}

/// Unified helper for combining comment writes + phase transition + notification.
#[must_use]
async fn with_comment_and_transition<F>(
    ctx: TransitionCtx<'_, '_>,
    write_comments: F,
) -> FinalizeOutcome
where
    F: AsyncFnOnce(&TxGuard<'_>) -> anyhow::Result<()>,
{
    let outcome = match crate::db::with_tx_outcome(
        &board().conn,
        &ctx.ticket.id,
        ctx.log_label,
        async move |tx| {
            write_comments(tx).await?;
            BoardStore::transition_to_tx(tx, &ctx.ticket.id, Some(ctx.source), ctx.target).await
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
            warn!(
                ticket = %ctx.ticket.id,
                error = %e,
                "{}: transition to {} failed — ticket stuck in {}",
                ctx.log_label, ctx.target, ctx.source,
            );
            FinalizeOutcome::Failed
        }
    };

    if matches!(outcome, FinalizeOutcome::Applied) {
        match ctx.notify {
            NotifyPolicy::Notify => {
                notify_ticket(ctx.ticket, ctx.source, ctx.target, ctx.breaker_trip).await;
            }
            // Buffer: the transition is materialized by the CDC-driven chronicle
            // subscriber and delivered on the next drain.
            NotifyPolicy::Buffer => {}
        }
    }
    outcome
}

/// Write a comment to a ticket, then transition it to a new phase.
#[must_use]
pub(crate) async fn comment_and_transition(
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
#[must_use]
async fn resolve_ticket_workspace(ticket: &Ticket, log_label: &str) -> Option<Workspace> {
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

/// Leave the phase job in place for the unpause re-drive after a cooperative
/// pause-freeze, and mark any still-`launched` roster rows `failed` (via
/// [`crate::jobs::interrupt_phase_job_roster`]) so the puller can re-dispatch:
/// those rows carry the interrupted round's stored per-slot tasks, so they are
/// preserved for a slot-resume rather than discarded. The `paused` signal is
/// typed and captured immutably at the agent's bail, so this handles even the
/// narrow window where the workspace is resumed between the bail and this
/// finalizer.
async fn pause_freezing(ticket: &Ticket, job_id: &str) {
    info!(
        ticket = %ticket.id,
        "Phase round paused — leaving job in place for the unpause re-drive"
    );
    if let Err(e) =
        crate::jobs::interrupt_phase_job_roster(&crate::session::store().conn, job_id).await
    {
        warn!(ticket = %ticket.id, job = %job_id, error = %e, "Failed to mark paused agents on pause-freeze");
    }
}

/// Wording for the Manager notification about a paused workspace.
fn paused_workspace_sentence() -> &'static str {
    "all in-flight work stops and no pipeline stage advances until the workspace is resumed"
}

/// Pause the workspace after a technical/agent failure so queued development
/// tickets are not claimed and don't cascade through the pipeline failing
/// identically one after another.
///
/// Returns `true` when this call paused the workspace, `false` when it was
/// already paused, could not be resolved, or the service is shutting down (a
/// shutdown-interrupted run must never pause).
pub(crate) async fn pause_workspace_on_failure(ticket: &Ticket, reason: &str) -> bool {
    if crate::shutdown::aborting() {
        return false;
    }
    let Some(ws) = resolve_ticket_workspace(ticket, "auto-pause skipped").await else {
        return false;
    };
    if ws.paused {
        return false;
    }
    match crate::workspace::store().set_paused(&ws.name, true).await {
        Ok(()) => {
            info!(
                ticket = %ticket.id,
                workspace = %ws.name,
                reason,
                "Workspace auto-paused after failure"
            );
            true
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                workspace = %ws.name,
                reason,
                error = %e,
                "Failed to auto-pause workspace after technical failure",
            );
            false
        }
    }
}

/// Hard technical failure at a phase: destroy the CURRENT attempt (its phase
/// job) so the puller creates a FRESH one from scratch. Cancels any orphaned
/// ticket agents, leaves an explanatory comment, pauses the workspace only for
/// the implementation phases, and deletes the phase job. Consumes no bounce
/// budget.
pub(crate) async fn reset_phase_attempt(
    ticket: &Ticket,
    phase: TicketPhase,
    job_id: &str,
    reason: &str,
    comment: &str,
) {
    crate::agent::registry::AGENT_REGISTRY.cancel_by_ticket_id(&ticket.id);
    // The workspace pause still happens for implementation phases, but it is
    // silent in ticket history (the caller's substantive reason is the comment).
    if phase.is_pipeline_occupied() {
        pause_workspace_on_failure(ticket, reason).await;
    }
    if let Err(e) = board().add_comment(&ticket.id, SYSTEM_ROLE, comment).await {
        warn!(ticket = %ticket.id, error = %e, "Failed to comment phase reset");
    }
    let _ = crate::jobs::terminalize_job(&crate::session::store().conn, job_id).await;
}

/// Fetch the last ticket comment (any role) as failure details for a Manager
/// notification.
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

    let transition_log = format!(
        "[{}] {}: {} → {}",
        ticket.reporter,
        ticket.id,
        source.as_ref(),
        target_phase.as_ref(),
    );

    // The chronicle row for the just-committed transition is materialized by the
    // CDC drainer (synchronous materializer). Flush it now so the transition that
    // triggered this notification is present in the <ticket-updates> block rather
    // than deferred to a later drain.
    if let Err(e) = crate::db::cdc::drain_once(&board().conn).await {
        tracing::warn!(error = %e, "CDC flush before Manager notification failed");
    }

    let drained = crate::pipeline::chronicle::drain(&ws.name).await;

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

    if target_phase == TicketPhase::Failed {
        let failure_details = last_comment_as_failure_details(&ticket.id).await;
        let workspace_status = if breaker_trip {
            "Beware that all the other tickets have been moved back from Queued \
             to Planning."
                .to_string()
        } else if ws.paused {
            format!("The workspace is paused — {}.", paused_workspace_sentence())
        } else {
            "The workspace was not paused — remaining queued tickets may still be claimed."
                .to_string()
        };

        let warning = substitute(
            &load_prompt("pipeline/failure_notification.md"),
            &[
                ("{{failure_details}}", &failure_details),
                ("{{workspace_status}}", &workspace_status),
            ],
        );
        message.push_str("\n\n");
        message.push_str(&warning);
    }

    let agent_id = manager_agent_id(&ws.name);
    message_router::route(
        &agent_id,
        message_router::AgentJob {
            content: message,
            workspace_name: ws.name,
            user_name: String::new(),
            channel: String::new(),
            kind: message_router::MessageKind::TicketNotify,
            role: crate::Role::Manager,
            reply_target: None,
            pending_job_id: None,
        },
    );
}

/// Register an agent in the message router and write a job-bound launched
/// roster row so mid-run comments route to it and the re-dispatch guard blocks.
async fn register_running_agent(
    job_id: &str,
    agent_id: &str,
    kind: crate::jobs::AgentKind,
    warn_message: &str,
) -> tokio::sync::mpsc::UnboundedReceiver<message_router::AgentJob> {
    let incoming_rx = message_router::register_agent(agent_id);

    if let Err(e) = crate::jobs::upsert_job_agent(
        &crate::session::store().conn,
        job_id,
        agent_id,
        kind,
        crate::jobs::RowStatus::Launched,
    )
    .await
    {
        warn!(
            agent = agent_id,
            job = job_id,
            error = %e,
            "{warn_message}",
        );
    }

    incoming_rx
}

// ── Parallel-agent machinery (analysis / review / QA) ───────────────────

/// Canonical agent-id format for ticket phase roster slots
/// (`ticket_{ticket_id}_{idx}_{suffix}_{role}`).
#[must_use]
fn agent_id(ticket_id: &str, idx: i64, suffix: &str, role: Role) -> String {
    format!("ticket_{}_{}_{}_{}", ticket_id, idx, suffix, role.as_str())
}

/// Render the FINAL per-agent task for a ticket phase roster slot.
#[must_use]
fn agent_slot_task(prompt: &str, angles: &[String], count: usize, global_idx: usize) -> String {
    if angles.is_empty() {
        prompt.to_string()
    } else if count == 1 {
        format!("{prompt}\n\n{}", angles.join("\n\n"))
    } else {
        format!("{prompt}\n\n{}", angles[global_idx % angles.len()])
    }
}

/// Build a batch of [`AgentSlot`]s using the canonical agent-id format and
/// angle-cycling rule.
#[must_use]
fn build_agent_slots(
    ticket_id: &str,
    role: Role,
    prompt: &str,
    angles: &[String],
    start_idx: i64,
    count: usize,
) -> Vec<AgentSlot> {
    let suffix = crate::generate_suffix();
    let mut slots = Vec::with_capacity(count);
    let global_start = usize::try_from(start_idx).unwrap_or(usize::MAX);
    for k in 0..count {
        let idx = start_idx + i64::try_from(k).unwrap_or(i64::MAX);
        let agent_id = agent_id(ticket_id, idx, &suffix, role);
        let task = agent_slot_task(prompt, angles, count, global_start + k);
        slots.push(AgentSlot {
            idx,
            agent_id,
            task,
            status: crate::jobs::RowStatus::Launched,
            outcome: None,
        });
    }
    slots
}

/// Build an [`AgentSlot`] from a stored roster row — the slot-resume
/// counterpart to [`build_agent_slots`]. Reuses the durable agent id, idx, task,
/// status and outcome so an interrupted round continues in place instead of
/// re-deriving a fresh roster with a new suffix.
#[must_use]
fn agent_slot_from_roster_row(r: &crate::jobs::AgentRow) -> AgentSlot {
    AgentSlot {
        idx: r.idx.unwrap_or(0),
        agent_id: r.agent_id.clone(),
        task: r.task.clone(),
        status: r
            .status
            .parse::<crate::jobs::RowStatus>()
            .unwrap_or(crate::jobs::RowStatus::Failed),
        outcome: r.outcome.clone(),
    }
}

/// Write a batch of round slots as launched roster rows (with their idx) so
/// the re-dispatch guard blocks and mid-run comments route to the agents.
async fn insert_round_slots(job_id: &str, slots: &[AgentSlot], kind: crate::jobs::AgentKind) {
    let conn = &crate::session::store().conn;
    for slot in slots {
        if let Err(e) = conn
            .execute(
                crate::jobs::AGENT_INSERT_SQL,
                crate::jobs::agent_params(job_id, &slot.agent_id, kind, Some(slot.idx), &slot.task),
            )
            .await
        {
            warn!(
                agent = %slot.agent_id,
                job = %job_id,
                error = %e,
                "Failed to write round roster row",
            );
        }
    }
}

async fn checkpoint_parallel_outcomes(
    job_id: &str,
    launched: &[&AgentSlot],
    run_results: &[ParallelVerdict],
) -> std::collections::HashMap<String, ParallelVerdict> {
    let conn = &crate::session::store().conn;
    let mut by_agent = std::collections::HashMap::with_capacity(launched.len());
    for (slot, result) in launched.iter().zip(run_results) {
        let outcome = serialize_verdict_outcome(result);
        // A ParseFailed member must be treated as not-Done (re-run on a resume)
        // — it did produce a response but the extraction failed, so its stored
        // outcome is only a diagnostic dump, not a completed verdict.
        let status = if matches!(
            result,
            ParallelVerdict::NoResponse(_) | ParallelVerdict::ParseFailed(_)
        ) {
            crate::jobs::RowStatus::Failed
        } else {
            crate::jobs::RowStatus::Done
        };
        if let Err(e) =
            crate::jobs::write_agent_outcome(conn, job_id, &slot.agent_id, status, Some(&outcome))
                .await
        {
            warn!(
                job = %job_id,
                agent = %slot.agent_id,
                error = %e,
                "Failed to checkpoint agent outcome",
            );
        }
        by_agent.insert(slot.agent_id.clone(), result.clone());
    }
    by_agent
}

fn assemble_parallel_results(
    slots: &[AgentSlot],
    by_agent: &std::collections::HashMap<String, ParallelVerdict>,
) -> Vec<ParallelVerdict> {
    let mut results = Vec::with_capacity(slots.len());
    for slot in slots {
        if slot.status == crate::jobs::RowStatus::Done {
            results.push(deserialize_verdict_outcome(
                slot.outcome.as_deref().unwrap_or(""),
            ));
        } else if let Some(r) = by_agent.get(slot.agent_id.as_str()) {
            results.push(r.clone());
        } else {
            unreachable!("every non-Done slot is launched and recorded 1:1 in by_agent");
        }
    }
    results
}

/// Run `count` agents of the same role in parallel, then extract structured
/// verdicts from their responses. Returns `(results, paused)` — `paused` is a
/// typed signal that any member stopped at its pause boundary.
///
/// `resume` marks a slot-resume round: not-Done slots are re-run with
/// session-continuation semantics (their stored tasks were already seeded), so
/// the message is suppressed when the persisted session already holds the task,
/// and the leader-stagger is skipped (the sessions have already diverged).
#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
async fn run_parallel_agents(
    ticket: &Arc<Ticket>,
    ws: &Workspace,
    role: Role,
    extraction_prompt: &str,
    extract_mode: ExtractionMode,
    job_id: &str,
    slots: &[AgentSlot],
    expected_phase: TicketPhase,
    resume: bool,
) -> (Vec<ParallelVerdict>, bool) {
    let launched: Vec<&AgentSlot> = slots
        .iter()
        .filter(|s| s.status != crate::jobs::RowStatus::Done)
        .collect();
    let receivers: Vec<_> = launched
        .iter()
        .map(|s| message_router::register_agent(&s.agent_id))
        .collect();

    {
        let members: Vec<_> = launched
            .iter()
            .zip(receivers)
            .map(|(slot, rx)| {
                let ticket = Arc::clone(ticket);
                let ws = ws.clone();
                let extraction_prompt = extraction_prompt.to_string();
                let extract_mode = extract_mode.clone();
                let agent_id = slot.agent_id.clone();
                let task = slot.task.clone();
                move |round: crate::agent::RoundOpts| async move {
                    if !is_ticket_in_phase(&ticket.id, expected_phase).await {
                        if let Some(notify) = &round.first_call_notify {
                            notify.notify_one();
                        }
                        message_router::unregister_agent(&agent_id);
                        return (
                            ParallelVerdict::NoResponse(PHASE_GATE_BAIL_REASON.to_string()),
                            false,
                        );
                    }
                    // Session-non-emptiness discriminator: a resumed slot whose
                    // session already holds the task continues with an empty
                    // message (no duplicate task-prompt append); a missing/empty
                    // session dispatches fresh with the stored task.
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
                        false,
                        Some(rx),
                        resume,
                        Some(round),
                        None,
                        None,
                    )
                    .await;

                    let response = response.unwrap_or_default();
                    if response.is_empty() {
                        // A cooperative pause-freeze is a TYPED signal captured
                        // immutably at the agent's LLM-boundary bail (the
                        // `paused_frozen` snapshot is never cleared by an
                        // unpause, unlike the live `pause_stop` flag): the
                        // agent stopped rather than failing. It is not a
                        // technical failure.
                        if agent.is_paused_frozen() {
                            (
                                ParallelVerdict::NoResponse("agent paused".to_string()),
                                true,
                            )
                        } else {
                            let reason = agent.failure_reason("agent produced no response");
                            (
                                ParallelVerdict::NoResponse(crate::util::scrub_credentials(
                                    &reason,
                                )),
                                false,
                            )
                        }
                    } else {
                        let verdict = match &extract_mode {
                            ExtractionMode::ScoreVerdict => agent
                                .extract_verdict::<crate::Verdict>(
                                    &extraction_prompt,
                                    Some(&validate_verdict_score),
                                    None,
                                )
                                .await
                                .map(ParallelVerdict::Verdict),
                            ExtractionMode::ScorelessVerdict => agent
                                .extract_verdict::<crate::AnalysisVerdict>(
                                    &extraction_prompt,
                                    None,
                                    None,
                                )
                                .await
                                .map(ParallelVerdict::Analysis),
                            ExtractionMode::BlockerVerification { blockers } => {
                                let blockers = std::sync::Arc::clone(blockers);
                                let validator = move |v: &crate::BlockerVerificationVerdict| {
                                    validate_blocker_verification(v, &blockers)
                                };
                                agent
                                    .extract_verdict::<crate::BlockerVerificationVerdict>(
                                        &extraction_prompt,
                                        Some(&validator),
                                        None,
                                    )
                                    .await
                                    .map(ParallelVerdict::BlockerVerification)
                            }
                        };
                        match verdict {
                            Ok(v) => (v, false),
                            Err(e) => (ParallelVerdict::ParseFailed(e), false),
                        }
                    }
                }
            })
            .collect();
        let handles = crate::agent::spawn_staggered_round(members, resume).await;
        let mut run_results: Vec<ParallelVerdict> = Vec::with_capacity(handles.len());
        let mut paused = false;
        for handle in handles {
            match handle.await {
                Ok((v, p)) => {
                    paused |= p;
                    run_results.push(v);
                }
                Err(e) => run_results.push(round_member_failed(e)),
            }
        }

        let by_agent = checkpoint_parallel_outcomes(job_id, &launched, &run_results).await;
        (assemble_parallel_results(slots, &by_agent), paused)
    }
}

/// Build the raw-response dump section for a verdict-extraction failure comment.
pub(crate) fn raw_response_dump_section(failure: &crate::retry::RetryExhausted) -> String {
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

/// Load per-agent angle supplements for a verifier role.
fn load_verifier_angles(role: Role) -> Vec<String> {
    match role {
        Role::Reviewer => load_prompt_sections("review_angles.md"),
        Role::Qa => load_prompt_sections("qa_angles.md"),
        _ => Vec::new(),
    }
}

// ── Single-agent stage rounds (engineer / sanitation) ───────────────────

/// The single-agent stage rounds sharing one run tail (each driven by a
/// permanently-NULL-seat session pin): engineer and sanitation.
#[derive(Clone, Copy)]
enum StageRunKind {
    Engineer,
    Sanitation,
}

impl StageRunKind {
    /// The role dispatched for this stage round.
    fn role(self) -> Role {
        match self {
            Self::Engineer => Role::Engineer,
            Self::Sanitation => Role::Sanitation,
        }
    }

    /// The `agents.kind` roster label for this stage round.
    fn agent_kind(self) -> crate::jobs::AgentKind {
        match self {
            Self::Engineer => crate::jobs::AgentKind::Engineer,
            Self::Sanitation => crate::jobs::AgentKind::Sanitation,
        }
    }
}

async fn run_single_agent(
    agent_id: String,
    role: Role,
    ws: &Workspace,
    ticket: &Ticket,
    message: &str,
    incoming_rx: tokio::sync::mpsc::UnboundedReceiver<message_router::AgentJob>,
    resume: bool,
) -> (crate::Agent, Option<String>) {
    run_agent(
        agent_id,
        role,
        ws,
        Some(ticket),
        message,
        String::new(),
        String::new(),
        false,
        Some(incoming_rx),
        resume,
        None,
        None,
        None,
    )
    .await
}

/// Drain-cut guard for the single-agent stage-round finalizers.
fn stage_drain_cut(ticket_id: &str, label: &str, response: Option<&str>) -> bool {
    let drain_cut = response.is_none() && crate::shutdown::aborting();
    if drain_cut {
        info!(
            ticket = %ticket_id,
            "{label} round cut short by drain — job stays launched for boot resume",
        );
    }
    drain_cut
}

/// Shared post-run guard prologue for the single-agent stage-round finalizers.
async fn guard_stage(
    ticket_id: &str,
    phase: TicketPhase,
    label: &str,
    response: Option<&str>,
    job_id: &str,
) -> bool {
    if guard_job_phase(ticket_id, phase, job_id).await {
        return true;
    }
    if stage_drain_cut(ticket_id, label, response) {
        return true;
    }
    false
}

/// Run the stage-agent round tail shared by fresh dispatch. The NULL-seat
/// pin preserves the accumulated stage session across round-job deletion.
async fn run_stage_agent(
    ticket: &Ticket,
    ws: &Workspace,
    job_id: &str,
    task: &str,
    kind: StageRunKind,
) {
    let role = kind.role();
    if let Err(e) = crate::jobs::upsert_session_pin(
        &crate::session::store().conn,
        &ticket.id,
        task,
        crate::jobs::RowStatus::Launched,
        role,
    )
    .await
    {
        warn!(
            ticket = %ticket.id,
            job = %job_id,
            error = %e,
            "Failed to upsert {} session pin — session continuity across bounces/resets degraded",
            role,
        );
    }

    let agent_id = crate::jobs::session_pin_id(&ticket.id, role);
    let incoming_rx = register_running_agent(
        job_id,
        &agent_id,
        kind.agent_kind(),
        match kind {
            StageRunKind::Engineer => {
                "Failed to register running engineer — stale agent already cancelled at dispatch, proceeding without roster registration"
            }
            StageRunKind::Sanitation => {
                "Failed to register running sanitation agent — mid-run comments may not route"
            }
        },
    )
    .await;

    let message = task.to_string();
    let (agent, response) = run_single_agent(
        agent_id,
        kind.role(),
        ws,
        ticket,
        &message,
        incoming_rx,
        false,
    )
    .await;

    // The typed pause-freeze is an IMMUTABLE snapshot captured at the agent's
    // LLM-boundary bail (`is_paused_frozen`), so a workspace unpause racing in
    // after the bail cannot clear it — a frozen run is never misrouted to the
    // hard-failure/reset path.
    let paused = agent.is_paused_frozen();

    match kind {
        StageRunKind::Engineer => {
            finalize_engineer_stage(ticket, &agent, response.as_deref(), job_id, ws, paused).await;
        }
        StageRunKind::Sanitation => {
            finalize_sanitation_stage(ticket, &agent, response.as_deref(), job_id, ws, paused)
                .await;
        }
    }
}

// ── Stage-handoff roster helpers ────────────────────────────────────────

/// Persist the phase job's dispatch task at dispatch time. Phase dispatch
/// re-derives its prompt from live state, so this stored value is not read for
/// re-dispatch. Best-effort.
async fn sync_phase_job_task(conn: &crate::db::Connection, job_id: &str, task: &str) {
    if let Err(e) = crate::jobs::update_phase_job_task(conn, job_id, task).await {
        warn!(job = %job_id, error = %e, "Failed to sync phase job task");
    }
}

/// Clear the prior stage's running-agent roster rows at a stage-completion
/// handoff, unblocking re-dispatch. Best-effort.
async fn clear_implementation_roster(conn: &crate::db::Connection, job_id: &str, ticket_id: &str) {
    if let Err(e) = crate::jobs::clear_launched_agents_for_job(conn, job_id).await {
        warn!(ticket = %ticket_id, job = %job_id, error = %e, "Failed to clear running agents on stage handoff");
    }
}

// ── Bounce breaker (unified validation non-success) ─────────────────────

/// Aggregate per-agent failure reasons for a failed verifier round.
fn bounce_breaker_trip_comment() -> String {
    let max = MAX_BOUNCES;
    format!(
        "Failed after {max} bounces — ticket bounced back too many times \
         (circuit breaker, max: {max}). Ticket failed — Manager will triage."
    )
}

/// Returns true when bounce budget is exhausted (`bounce_count >= MAX_BOUNCES`).
#[must_use]
fn bounce_exhausted(bounce_count: i64) -> bool {
    usize::try_from(bounce_count).unwrap_or(usize::MAX) >= MAX_BOUNCES
}

/// Move all other Queued tickets in the workspace to Planning.
async fn drain_queued_siblings(ticket: &Ticket) {
    match board()
        .drain_queued_to_planning(&ticket.workspace_name)
        .await
    {
        Ok(updated) if updated > 0 => {
            info!(
                tickets = updated,
                workspace = %ticket.workspace_name,
                "Moved {updated} Queued ticket(s) to Planning after bounce breaker trip",
            );
        }
        Ok(_) => {
            debug!(
                workspace = %ticket.workspace_name,
                "No Queued siblings to drain after bounce breaker trip",
            );
        }
        Err(e) => {
            warn!(
                ticket = %ticket.id,
                workspace = %ticket.workspace_name,
                error = %e,
                "Failed to move Queued tickets to Planning \
                 — breaker trip proceeds without moving siblings",
            );
        }
    }
}

/// Unified validation-failure bounce. Increments the shared bounce budget for
/// EVERY validation-phase non-success outcome. On exhaustion the ticket trips
/// to Failed (terminal); the phase job is deleted and the puller re-drives.
/// On a non-exhausting bounce the ticket goes back to InDevelopment and the
/// phase job is deleted so the puller creates a fresh engineer attempt.
pub(crate) async fn bounce_to_development(
    ticket: &Ticket,
    source: TicketPhase,
    log_label: &str,
    drains_siblings: bool,
    failure_role: &str,
    failure_comment: &str,
    job_id: &str,
) -> FinalizeOutcome {
    let trip = bounce_exhausted(ticket.bounce_count);
    let target = if trip {
        TicketPhase::Failed
    } else {
        TicketPhase::InDevelopment
    };
    let notify = if trip {
        NotifyPolicy::Notify
    } else {
        NotifyPolicy::Buffer
    };
    let trip_comment = trip.then(bounce_breaker_trip_comment);
    let ctx = TransitionCtx::new(ticket, source, target, notify, log_label)
        .with_breaker(trip && drains_siblings);

    let outcome = with_comment_and_transition(ctx, async |tx| {
        if let Some(comment) = &trip_comment {
            BoardStore::add_comment_tx(tx, &ticket.id, SYSTEM_ROLE, comment).await?;
        }
        if !failure_comment.is_empty() {
            BoardStore::add_comment_tx(tx, &ticket.id, failure_role, failure_comment).await?;
        }
        if !trip {
            BoardStore::increment_bounce_count_tx(tx, &ticket.id).await?;
        }
        Ok(())
    })
    .await;

    if matches!(outcome, FinalizeOutcome::Applied) {
        let conn = &crate::session::store().conn;
        if trip {
            if drains_siblings {
                drain_queued_siblings(ticket).await;
            }
            let _ = crate::jobs::terminalize_job(conn, job_id).await;
            info!(
                ticket = %ticket.id,
                "Bounce circuit breaker tripped ({} bounces) — ticket failed",
                MAX_BOUNCES,
            );
        } else {
            // Non-exhausting bounce: delete the phase job; the puller creates a
            // fresh InDevelopment attempt on the next tick.
            let _ = crate::jobs::terminalize_job(conn, job_id).await;
            info!(
                ticket = %ticket.id,
                target = %target,
                "{log_label} failed — engineer re-dispatched via the puller on a fresh attempt",
            );
        }
    }
    outcome
}

// ── Verifier rounds (review / QA) ───────────────────────────────────────
//
// Review and QA genuinely share one parallel-verifier engine (same prompt/sub,
// slot, extraction, and verdict-finalize tail — only the role/phase/count and
// the reviewer-only git-state decisions differ). The verdict-processing core
// (`process_verifier_verdicts` / `apply_clean_verifier_round`, the `VerifierInfo`
// metadata, and the review/QA thresholds) lives in [`crate::pipeline::verdict`];
// the phase-entry dispatch lives in `dispatch_verifiers` here. The
// per-phase entry points (`review::run`/`qa::run`) and the reviewer-only git
// logic (`review::compute_review_skip`/`compute_reviewer_count`/
// `record_reviewed_base_after_review`) and the QA count
// (`qa::QA_PARALLEL_AGENT_COUNT`) live in their phase modules. The shared tail
// is DRY, and each phase module owns its own decisions.

/// Finalize a verifier round (review/QA).
async fn finalize_verifier_round(
    ws: &Workspace,
    ticket: &Ticket,
    vi: VerifierInfo,
    results: &[ParallelVerdict],
    job_id: &str,
    is_reviewer: bool,
) {
    let transitioned = process_verifier_verdicts(ws, ticket, results, vi, job_id).await;
    if !transitioned && crate::shutdown::aborting() {
        info!(
            ticket = %ticket.id,
            "Verifier round cut short by drain — job stays launched for boot resume",
        );
        return;
    }

    // Only a passing review records the reviewed base (QA never has git
    // available, and the skip-review base is a reviewer-only concept).
    if is_reviewer {
        review::record_reviewed_base_after_review(
            ws.as_path(),
            &ticket.id,
            review::git_available_for_review(ws, vi).await,
            transitioned,
            results,
        )
        .await;
    }
}

/// Shared dispatch logic for parallel verifiers (reviewers and QA): guards the
/// job phase (derived from the VerifierInfo), then dispatches the round.
async fn dispatch_verifiers(ticket: Arc<Ticket>, ws: Workspace, vi: VerifierInfo, job_id: String) {
    if guard_job_phase(&ticket.id, vi.active_phase, &job_id).await {
        return;
    }
    let is_reviewer = vi.role == Role::Reviewer;
    if review::maybe_skip_review(&ticket, &ws, vi, &job_id).await {
        return;
    }

    let extraction_prompt = crate::prompt::load_prompt(vi.extraction_prompt_path);
    let conn = &crate::session::store().conn;

    // Slot-resume detection: a non-empty roster marks an interrupted round —
    // reconstruct Done slots from stored outcomes and re-run the not-Done ones
    // with their stored tasks. An empty roster (body crashed before its first
    // roster write) degrades to a fresh dispatch re-derived from live ticket
    // state.
    let roster = match crate::jobs::list_agents_for_job(conn, &job_id).await {
        Ok(roster) => roster,
        Err(e) => {
            // A read failure must NOT degrade to a fresh dispatch: doing so would
            // re-derive a new-suffix roster and orphan the interrupted round's
            // rows. Bail and let the poller re-drive (the job stays occupied,
            // with no running agents, so re-dispatch is safe).
            warn!(ticket = %ticket.id, job = %job_id, error = %e, "Failed to read phase roster — bailing to preserve interrupted round");
            return;
        }
    };

    if roster.is_empty() {
        fresh_dispatch_verifiers(ticket, ws, vi, job_id, is_reviewer).await;
        return;
    }

    // Resume: reuse the stored roster. The stored per-slot task is the round's
    // prompt (no re-derivation from live ticket state); the count is the roster
    // length (recomputed churn could differ across the interruption).
    let slots: Vec<AgentSlot> = roster.iter().map(agent_slot_from_roster_row).collect();
    let split = crate::jobs::split_slot_resume(&roster);
    let count = slots.len();
    let verifier_label = if count == 1 { "verifier" } else { "verifiers" };
    info!(
        ticket = %ticket.id,
        role = %vi.role.as_str(),
        count,
        verifier_label,
        "Resuming {count} parallel {verifier_label} from stored roster",
    );
    let not_done: Vec<String> = split.not_done.iter().map(|r| r.agent_id.clone()).collect();
    if let Err(e) = crate::jobs::rearm_roster_launched(conn, &job_id, &not_done).await {
        warn!(ticket = %ticket.id, job = %job_id, error = %e, "Failed to re-arm resumed roster slots");
    }
    let (results, paused) = run_parallel_agents(
        &ticket,
        &ws,
        vi.role,
        &extraction_prompt,
        ExtractionMode::ScoreVerdict,
        &job_id,
        &slots,
        vi.active_phase,
        true,
    )
    .await;
    finish_verifier_dispatch(ticket, ws, vi, job_id, is_reviewer, results, paused).await;
}

/// Fresh verifier dispatch (empty roster): build a new roster, sync the phase
/// job task, and run the round.
async fn fresh_dispatch_verifiers(
    ticket: Arc<Ticket>,
    ws: Workspace,
    vi: VerifierInfo,
    job_id: String,
    is_reviewer: bool,
) {
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
    let repo_path = ws.as_path();
    let count = if is_reviewer {
        review::compute_reviewer_count(&ticket, repo_path).await
    } else {
        qa::QA_PARALLEL_AGENT_COUNT
    };
    let verifier_label = if count == 1 { "verifier" } else { "verifiers" };
    info!(
        ticket = %ticket.id,
        role = %vi.role.as_str(),
        count,
        verifier_label,
        "Dispatching {count} parallel {verifier_label}",
    );

    let conn = &crate::session::store().conn;
    sync_phase_job_task(conn, &job_id, &prompt).await;

    let slots = build_agent_slots(
        &ticket.id,
        vi.role,
        &prompt,
        &load_verifier_angles(vi.role),
        0,
        count,
    );
    insert_round_slots(&job_id, &slots, crate::jobs::AgentKind::Verifier).await;
    let (results, paused) = run_parallel_agents(
        &ticket,
        &ws,
        vi.role,
        &extraction_prompt,
        ExtractionMode::ScoreVerdict,
        &job_id,
        &slots,
        vi.active_phase,
        false,
    )
    .await;
    finish_verifier_dispatch(ticket, ws, vi, job_id, is_reviewer, results, paused).await;
}

/// Shared tail of verifier dispatch (fresh and resumed) after the parallel
/// agents finish: guards the job phase, honours pause-freeze, then finalizes
/// the round's verdicts.
async fn finish_verifier_dispatch(
    ticket: Arc<Ticket>,
    ws: Workspace,
    vi: VerifierInfo,
    job_id: String,
    is_reviewer: bool,
    results: Vec<ParallelVerdict>,
    paused: bool,
) {
    if guard_job_phase(&ticket.id, vi.active_phase, &job_id).await {
        return;
    }

    // A pause-freeze is NOT a technical failure: leave the job in place for the
    // unpause re-drive (the typed `paused` signal was captured at bail time, so
    // it survives a workspace-unpause race that a live re-read of paused state
    // would miss).
    if paused {
        pause_freezing(&ticket, &job_id).await;
        return;
    }

    finalize_verifier_round(&ws, &ticket, vi, &results, &job_id, is_reviewer).await;
}

/// Determine whether to notify immediately or buffer the Done transition.
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
