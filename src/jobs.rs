//! Durable jobs layer: jobs/agents/pending_jobs lifecycle, boot recovery
//! scan, and the phase-only stale purge orchestrator.
//!
//! All rows live in the consolidated domain database (`core.db`) behind the
//! one shared connection — the jobs/session/board tables now share a single
//! transaction domain, so cross-store ordering and crash-safety can be
//! expressed in a single transaction (see the purge section in the design).
//!
//! Table ownership: the DDL lives in the append-only schema catalog
//! ([`crate::db::migrations`]); this module owns the row model, the lifecycle
//! helpers, the boot scan, and the purge orchestrator. Purge scope is
//! phase-only: ticket-phase jobs are time-deleted (the puller re-creates them
//! from `tickets.phase`); non-phase launched jobs are never purged by time
//! alone — explicit abandon only.

use crate::Role;
use crate::agent::message_router::{AgentJob, MessageKind};
use crate::db::{self, Connection, Row, TxGuard, Value, params};
use anyhow::{Context, Result};
use std::time::Duration;
use tracing::{debug, info, warn};

// ── Job-kind vocabulary (jobs.kind) ─────────────────────────────────────
// Values of `jobs.kind` are fully determined by the child row (one kind per
// child — `SpawnChild::kind_str`); the [`MessageKind`] enum travels inside the
// serialized pending_jobs envelope and is unrelated to that vocabulary.

/// Status vocabulary shared by `jobs.status` and `agents.status` — both
/// tables are schema-locked to the same launched/done/failed dictionary
/// ('failed' is written in prod: failed agent outcomes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowStatus {
    Launched,
    Done,
    Failed,
}

impl RowStatus {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Launched => "launched",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }
}

impl std::str::FromStr for RowStatus {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Case-sensitive to mirror the schema dictionary exactly.
        match s {
            "launched" => Ok(Self::Launched),
            "done" => Ok(Self::Done),
            "failed" => Ok(Self::Failed),
            _ => Err(anyhow::anyhow!(
                "Invalid row status '{s}'. Valid statuses: launched, done, failed"
            )),
        }
    }
}

/// Explicit dispatch mode of a job — replaces the NULL-sentinel overload of
/// `caller_agent_id`. `Sync` jobs are owned by the caller session pin
/// (`caller_agent_id`); `Async` jobs (research/research_cleanup/temp_cleanup,
/// ticket phases, and pin-less analyze/implement dispatches) resume at boot.
///
/// The string values are intrinsic Rust/SQL coupling: `as_str()`, the SQL
/// literals in [`find_owned_launched_jobs`] / [`abandon_session_jobs`], and
/// migration delta `30`'s backfill CASE must stay in sync on a rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobMode {
    Sync,
    Async,
}

impl JobMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sync => "sync",
            Self::Async => "async",
        }
    }
}

impl std::str::FromStr for JobMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Case-sensitive to mirror the schema dictionary exactly.
        match s {
            "sync" => Ok(Self::Sync),
            "async" => Ok(Self::Async),
            _ => Err(anyhow::anyhow!(
                "Invalid job mode '{s}'. Valid modes: sync, async"
            )),
        }
    }
}

/// Values of `agents.kind` — dispatch-slot kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AgentKind {
    Analyst,
    Verifier,
    Engineer,
    /// Single coder sub-agent for the durable async ImplementTool dispatch.
    Coder,
    Sanitation,
    /// Synthetic in-flight marker for the shell-command diagnostics stage (not
    /// an LLM agent — the roster row only drives the re-dispatch guard).
    Diagnostics,
}

impl AgentKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Analyst => "analyst",
            Self::Verifier => "verifier",
            Self::Engineer => "engineer",
            Self::Coder => "coder",
            Self::Sanitation => "sanitation",
            Self::Diagnostics => "diagnostics",
        }
    }
}

/// Graceful-shutdown drain cap: in-flight work completes within this window,
/// then stragglers are force-cancelled.
pub(crate) const DRAIN_CAP_SECS: u64 = 10 * 60;

/// Stale-purge cutoff (hours): only ticket-phase `jobs` rows older than this
/// are purged (they are recoverable from `tickets.phase`). Non-phase launched
/// jobs (researches, cleanups, sync/async envelope kinds) are never
/// time-deleted — explicit abandon only. `pending_jobs` envelopes are never
/// purged — at-least-once keeps unconfirmed rows alive.
pub const PURGE_CUTOFF_HOURS: i64 = 8;

// ── Row model ───────────────────────────────────────────────────────────
// Only the columns actually read by the scan paths are copied into memory;
// caller identity (task/role/user_name/channel) is re-queried on demand by
// `job_caller` instead of being carried on the row structs. The DB schema
// columns stay as-is — a future path re-adds a field when it reads one.

/// A row of the `jobs` table.
#[derive(Debug, Clone)]
pub(crate) struct JobRow {
    pub id: String,
    pub kind: String,
    pub workspace_name: String,
    pub retry_count: i64,
    pub ticket_id: Option<String>,
    /// The caller agent's stable session pin for sync analyze/implement jobs;
    /// NULL for async/boot-resumed jobs (see the sync-resume design).
    pub caller_agent_id: Option<String>,
    /// Explicit dispatch mode (see [`JobMode`]) — the discriminator that
    /// replaces the NULL-sentinel overload of `caller_agent_id`.
    pub mode: JobMode,
}

/// A row of the `agents` table. Carries the slot `idx` (in addition to the
/// status/outcome/task) so the slot-resume skeleton can reconstruct a round in
/// place without re-deriving it from live ticket state.
#[derive(Debug, Clone)]
pub(crate) struct AgentRow {
    pub idx: Option<i64>,
    pub agent_id: String,
    pub status: String,
    pub outcome: Option<String>,
    pub task: String,
}

/// A row of the `pending_jobs` table.
#[derive(Debug, Clone)]
pub(crate) struct PendingJobRow {
    pub id: String,
    pub target_agent_id: String,
    pub envelope: String,
    pub created_at: String,
}

crate::columns! {
    JOB_COLUMNS [JOB] {
        ID              => "id",
        KIND            => "kind",
        WORKSPACE_NAME  => "workspace_name",
        RETRY_COUNT     => "retry_count",
        TICKET_ID       => "ticket_id",
        CALLER_AGENT_ID => "caller_agent_id",
        MODE            => "mode",
    }
}

fn job_row_from(row: &Row) -> anyhow::Result<JobRow> {
    Ok(JobRow {
        id: row.get(COL_JOB_ID)?,
        kind: row.get(COL_JOB_KIND)?,
        workspace_name: row.get(COL_JOB_WORKSPACE_NAME)?,
        retry_count: row.get(COL_JOB_RETRY_COUNT)?,
        ticket_id: row.get(COL_JOB_TICKET_ID)?,
        caller_agent_id: row.get(COL_JOB_CALLER_AGENT_ID)?,
        // Defensive fallback: the migration backfills every row, so a NULL or
        // bogus value here is a legacy edge — treat it as async rather than
        // dropping the job at boot.
        mode: match row.get::<Option<String>>(COL_JOB_MODE)? {
            Some(s) => s.parse().unwrap_or_else(|e| {
                warn!(mode = %s, error = %e, "Invalid jobs.mode — falling back to async");
                JobMode::Async
            }),
            None => JobMode::Async,
        },
    })
}

fn agent_row_from(row: &Row) -> anyhow::Result<AgentRow> {
    Ok(AgentRow {
        idx: row.get(0)?,
        agent_id: row.get(1)?,
        status: row.get(2)?,
        outcome: row.get(3)?,
        task: row.get(4)?,
    })
}

fn pending_row_from(row: &Row) -> anyhow::Result<PendingJobRow> {
    Ok(PendingJobRow {
        id: row.get(0)?,
        target_agent_id: row.get(1)?,
        envelope: row.get(2)?,
        created_at: row.get(3)?,
    })
}

// ── Shared INSERT forms (one definition each) ───────────────────────────
// Each durable INSERT form has two producers — one lock-acquiring (conn),
// one lock-held (TxGuard). The SQL const + params builder is the single
// source of truth for the column set; call sites keep their own executor
// (never delegate a TxGuard call to conn.execute — the guard already holds
// the mutex) and their own error contexts.

/// pending_jobs envelope INSERT — producers: `message_router::persist_pending`
/// and `complete_job_with_envelope`.
pub(crate) const PENDING_JOB_INSERT_SQL: &str = "INSERT INTO pending_jobs \
     (id, target_agent_id, envelope, created_at) \
     VALUES (?1, ?2, ?3, ?4)";

/// Build the pending_jobs INSERT params from a durable envelope — the target
/// agent id is derived from the envelope (caller identity lives in the
/// serialized envelope, not the row).
pub(crate) fn pending_job_params(id: &str, envelope: &AgentJob, now: &str) -> Result<[Value; 4]> {
    let envelope_json = serde_json::to_string(envelope)?;
    Ok([
        id.into(),
        envelope_target(envelope).into(),
        envelope_json.into(),
        now.into(),
    ])
}

/// agents roster INSERT — producers: `spawn_job` roster loop and the pipeline's
/// `insert_round_slots`. Status is the literal 'launched'.
pub(crate) const AGENT_INSERT_SQL: &str = "INSERT INTO agents \
     (job_id, agent_id, kind, idx, status, task) \
     VALUES (?1, ?2, ?3, ?4, 'launched', ?5)";

/// Build the agents roster INSERT params (status is fixed by the SQL).
pub(crate) fn agent_params(
    job_id: &str,
    agent_id: &str,
    kind: AgentKind,
    idx: Option<i64>,
    task: &str,
) -> [Value; 5] {
    [
        job_id.into(),
        agent_id.into(),
        kind.as_str().into(),
        idx.into(),
        task.into(),
    ]
}

// ── Spawn (one tx) ──────────────────────────────────────────────────────

/// Optional kind-specific child row inserted in the SAME tx as the job row
/// (jobs + agents + child are one atomic unit). Mirrors research's in-tx
/// insert — a crash between a jobs+agents commit and a later child insert
/// would otherwise leave a resumable job whose child row is missing. The
/// shared helper takes the child explicitly (never branching on `kind`), so
/// every caller owns its child-row shape.
#[derive(Debug, Clone)]
pub(crate) enum SpawnChild {
    /// Pure kind marker — no child row (resume reads the jobs row alone).
    Analyze,
    /// Pure kind marker — no child row (resume reads the jobs row alone). The
    /// single coder's roster row holds the dispatched task; the job row holds
    /// the caller identity for resume delivery.
    Implement,
    /// research_jobs (id, state).
    Research,
    /// A research-run cleanup Sanitation agent. No child row: the jobs row's
    /// `task` holds the cleanup prompt, and the row id == the run id (folder
    /// name) so the row holds the run folder until the cleanup completes —
    /// the folder is released by the cleanup tail (folder first, row last).
    ResearchCleanup,
    /// A periodic OS temp-dir cleaner Sanitation agent (fire-and-forget, see
    /// `crate::temp`). No child row: the jobs row's `task` holds the
    /// cleanup prompt. Leftover rows are terminalized at boot (never
    /// resumed) — the cleaner's ephemeral workspace is never registered.
    TempCleanup,
    /// A per-phase ticket job: `jobs.kind` = `phase.as_ref()` (the single
    /// state-machine authority), `jobs.ticket_id` = the ticket. One short-lived
    /// job per pipeline phase; the ticket's `phase` is the only durable running
    /// truth. Roster grows during the phase (analysts/reviewers), and the job
    /// is deleted when the phase completes or is reset.
    Phase {
        phase: crate::pipeline::board::TicketPhase,
        ticket_id: String,
    },
}

impl SpawnChild {
    /// The `jobs.kind` value — one kind per child row; Analyze is the exception
    /// (pure kind marker, no child row). Deriving kind here closes the drift
    /// window of an inconsistent (kind, child) pair.
    #[must_use]
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Analyze => "analyze",
            Self::Implement => "implement",
            Self::Research => "research",
            Self::ResearchCleanup => "research_cleanup",
            Self::TempCleanup => "temp_cleanup",
            Self::Phase { phase, .. } => match phase {
                crate::pipeline::board::TicketPhase::Analysis => "analysis",
                crate::pipeline::board::TicketPhase::InDevelopment => "in_development",
                crate::pipeline::board::TicketPhase::InDiagnostics => "in_diagnostics",
                crate::pipeline::board::TicketPhase::InReview => "in_review",
                crate::pipeline::board::TicketPhase::InQa => "in_qa",
                crate::pipeline::board::TicketPhase::InSanitation => "in_sanitation",
                _ => unreachable!("non-working phase as a job kind"),
            },
        }
    }
}

/// `jobs.ticket_id` for a spawn child — `Some` only for ticket phase jobs.
#[must_use]
pub(crate) fn child_ticket_id(child: &SpawnChild) -> Option<&str> {
    match child {
        SpawnChild::Phase { ticket_id, .. } => Some(ticket_id.as_str()),
        _ => None,
    }
}

/// Spawn a job with its pre-generated agent roster in ONE transaction.
/// MUST commit before the agent's first session write. The kind is derived
/// from the child via [`SpawnChild::kind_str`] — closing the drift window of
/// an inconsistent (kind, child) pair.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn spawn_job(
    conn: &Connection,
    id: &str,
    task: &str,
    workspace_name: &str,
    user_name: &str,
    channel: &str,
    role: Role,
    agents: &[NewAgent],
    child: &SpawnChild,
    caller_agent_id: Option<&str>,
) -> Result<()> {
    let tx = conn.begin_tx().await?;
    insert_job_tx(
        &tx,
        id,
        task,
        workspace_name,
        user_name,
        channel,
        role,
        agents,
        child,
        caller_agent_id,
    )
    .await?;
    tx.commit().await?;
    Ok(())
}

/// The jobs-row INSERT + agent-roster loop + child-row match, run inside a
/// live transaction (`tx`). Extracted from [`spawn_job`] so callers that
/// already hold a handoff transaction (e.g. `transition_research_to_cleanup`)
/// can reuse the exact same row model. Does NOT commit — the caller owns the
/// transaction.
#[expect(clippy::too_many_arguments)]
async fn insert_job_tx(
    tx: &TxGuard<'_>,
    id: &str,
    task: &str,
    workspace_name: &str,
    user_name: &str,
    channel: &str,
    role: Role,
    agents: &[NewAgent],
    child: &SpawnChild,
    caller_agent_id: Option<&str>,
) -> Result<()> {
    let kind = child.kind_str();
    let ticket_id = child_ticket_id(child);
    // The mode is derived from the caller pin (sync ⇔ pin present). A sync
    // dispatch executed without a pin resolves to mode=async, preserving the
    // NULL-sentinel behavior the discriminator replaces.
    let mode = if caller_agent_id.is_some() {
        JobMode::Sync
    } else {
        JobMode::Async
    };
    let now = db::now();
    tx.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, user_name, channel, role, \
         ticket_id, retry_count, created_at, updated_at, caller_agent_id, mode) \
         VALUES (?1, ?2, 'launched', ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?9, ?10, ?11)",
        params![
            id,
            kind,
            task,
            workspace_name,
            user_name,
            channel,
            role.as_str(),
            ticket_id,
            now.clone(),
            caller_agent_id,
            mode.as_str(),
        ],
    )
    .await
    .with_context(|| format!("failed to spawn job {id}"))?;
    for a in agents {
        tx.execute(
            AGENT_INSERT_SQL,
            agent_params(id, &a.agent_id, a.kind, a.idx, &a.task),
        )
        .await
        .with_context(|| format!("failed to insert agent roster for job {id}"))?;
    }
    match child {
        // Analyze / Implement / ResearchCleanup / TempCleanup / ticket Phase are
        // pure kind markers — the job row alone drives resume (the cleanup
        // prompt lives in jobs.task, the ticket id on jobs.ticket_id).
        SpawnChild::Analyze
        | SpawnChild::Implement
        | SpawnChild::ResearchCleanup
        | SpawnChild::TempCleanup
        | SpawnChild::Phase { .. } => {}
        SpawnChild::Research => {
            tx.execute(
                "INSERT INTO research_jobs (id, state) VALUES (?1, '{}')",
                params![id],
            )
            .await
            .with_context(|| format!("failed to insert research_jobs row for job {id}"))?;
        }
    }
    Ok(())
}

/// One agent slot in a spawn roster.
pub(crate) struct NewAgent {
    pub agent_id: String,
    pub kind: AgentKind,
    pub idx: Option<i64>,
    pub task: String,
}

// ── Checkpoint ──────────────────────────────────────────────────────────

/// Checkpoint a job row: bump retry_count and touch `updated_at` (a plain
/// recency marker — the phase-only purge keys off it).
pub(crate) async fn checkpoint_job(conn: &Connection, id: &str, retry_count: i64) -> Result<()> {
    // Boot resumes re-arm status to Launched (failed → launched re-activation).
    let status = RowStatus::Launched;
    let now = db::now();
    conn.execute(
        "UPDATE jobs SET status = ?1, retry_count = ?2, updated_at = ?3 WHERE id = ?4",
        params![status.as_str(), retry_count, now, id],
    )
    .await
    .with_context(|| format!("failed to checkpoint job {id}"))?;
    Ok(())
}

/// Write one agent's terminal outcome (agents.outcome per-agent at round
/// collection — no read-modify-write race).
pub(crate) async fn write_agent_outcome(
    conn: &Connection,
    job_id: &str,
    agent_id: &str,
    status: RowStatus,
    outcome: Option<&str>,
) -> Result<()> {
    conn.execute(
        "UPDATE agents SET status = ?1, outcome = ?2 WHERE job_id = ?3 AND agent_id = ?4",
        params![status.as_str(), outcome, job_id, agent_id],
    )
    .await
    .with_context(|| format!("failed to write agent outcome for {agent_id}"))?;
    Ok(())
}

// ── Completion (the exactly-once boundary) ──────────────────────────────

/// Complete a job whose result is a durable envelope: ONE tx — INSERT
/// pending_jobs (id = source jobs.id) + DELETE the jobs row. Exactly-once
/// persistence boundary for analyze/research results.
///
/// INSERT-failure note: the tx rolls back, so the job row SURVIVES and the
/// caller routes the envelope best-effort (never a silent drop). The
/// surviving launched row is then resumed at the next boot and delivers a
/// SECOND envelope — an accepted duplicate-delivery window on the DB-error
/// path (bounded: one extra envelope, deduped by the consumer only in the
/// common at-most-once sense; the design's insert-failure policy trades the
/// rare duplicate for the never-silent-drop guarantee).
pub(crate) async fn complete_job_with_envelope(
    conn: &Connection,
    job_id: &str,
    envelope: &AgentJob,
) -> Result<()> {
    let now = db::now();
    let tx = conn.begin_tx().await?;
    tx.execute(
        PENDING_JOB_INSERT_SQL,
        pending_job_params(job_id, envelope, &now).context("serialize envelope")?,
    )
    .await
    .with_context(|| format!("failed to persist envelope for job {job_id}"))?;
    delete_job_tx(&tx, job_id).await?;
    // Manual-cancel gate INSIDE the tx (after the INSERT, before the
    // commit): the tx holds the single-writer lock, so a cancel that
    // fired after the caller's pre-completion gate check is still
    // observed here and rolls the tx back (the default DropBehavior) —
    // the pending row is never committed and the job row survives for
    // the cancel sweep. The sweep's own DELETE then serializes behind
    // this tx (single-writer), so no cancelled run can leave a
    // deliverable pending row. Only research runs ever register a cancel
    // signal (analyze job ids are never registered) — no behavior change
    // for other job kinds.
    if crate::research_cancel::is_cancelled(job_id) {
        tracing::info!(job = %job_id, "Job completion rolled back — run manually cancelled");
        return Ok(());
    }
    tx.commit().await?;
    Ok(())
}

/// COMPLETE a durable analyze/research job: one tx — INSERT pending_jobs
/// (envelope, id = job id) + DELETE jobs row (exactly-once boundary; the tx
/// detail lives on [`complete_job_with_envelope`]). Returns the envelope with
/// `pending_job_id` set when the row persisted, cleared on INSERT failure so
/// the caller falls back to a best-effort route: the result still reaches the
/// caller (never a silent drop — the envelope is the caller's only result
/// path), the pending row is simply absent.
pub(crate) async fn complete_durable_job(
    job_id: &str,
    content: String,
    kind: MessageKind,
    caller_role: Role,
    user_name: &str,
    channel: &str,
    workspace_name: &str,
) -> AgentJob {
    let mut envelope = AgentJob {
        content,
        workspace_name: workspace_name.to_string(),
        user_name: user_name.to_string(),
        channel: channel.to_string(),
        kind,
        role: caller_role,
        reply_target: None,
        pending_job_id: Some(job_id.to_string()),
    };
    // INSERT-failure policy: fall back to a non-durable best-effort route.
    if complete_job_with_envelope(&crate::session::store().conn, job_id, &envelope)
        .await
        .is_err()
    {
        envelope.pending_job_id = None;
    }
    envelope
}

/// Terminalize a job: DELETE the row entirely — CASCADE removes the agent
/// roster and child rows; a no-op when the row is already absent. Safe for any
/// jobs.kind. Ticket-phase call sites use it both on success (the phase body
/// transitions the ticket and deletes its own short-lived job; the puller
/// re-creates a fresh attempt on the next tick) and on hard-failure cleanup.
/// Envelope kinds (analyze/research) are terminal only when the row is gone —
/// the pending row IS the durable record.
pub(crate) async fn terminalize_job(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute("DELETE FROM jobs WHERE id = ?1", params![job_id])
        .await
        .context("terminalize job")?;
    Ok(())
}

// ── Explicit abandon (the only deletion path for unfinished non-phase jobs) ──
//
// Sync caller-owned jobs are never purged by time — only the caller's /clear or
// the deletion of the owning workspace abandons them. This is that path.

/// User-initiated /clear or session-row delete abandons the session's
/// unfinished sync jobs before the session is deleted — no silent orphaning.
/// CASCADE removes the roster rows; sync analyze/implement jobs carry no run
/// folders and never create pending_jobs rows.
#[expect(clippy::cast_possible_truncation)] // u64 row count fits usize on all supported targets
pub(crate) async fn abandon_session_jobs(caller_agent_id: &str) -> Result<usize> {
    let conn = &crate::session::store().conn;
    let deleted = conn
        .execute(
            "DELETE FROM jobs WHERE mode = 'sync' AND caller_agent_id = ?1",
            params![caller_agent_id],
        )
        .await
        .context("abandon session sync jobs")?;
    info!(
        caller_agent_id,
        count = deleted,
        "abandoned session sync jobs"
    );
    Ok(deleted as usize)
}

/// Atomic cancel-handoff row transition for a research run being cancelled:
/// DELETE the research job row (and its pending envelope) and INSERT the
/// `research_cleanup` job row reusing id == run_id — all in ONE tx. This is
/// the crash-safe handoff: the folder can never be orphaned with no cleanup
/// row, because after the tx a `research_cleanup` row exists for the run and
/// boot resume picks cleanup up even if the process dies right after. The
/// reverse order (insert before delete) is impossible — `jobs.id` is a PK, so
/// the cleanup row cannot be inserted while the research row with the same id
/// still exists — which is why delete-then-insert inside one tx is the atomic
/// unit. The cleanup row's existence is the folder-hold + boot-resume marker.
///
/// No `research_jobs` child row is created (the child row is research-only; the
/// cleanup prompt lives in `jobs.task`, exactly like a fresh cleanup dispatch).
pub(crate) async fn transition_research_to_cleanup(
    conn: &Connection,
    run_id: &str,
    cleanup_task: &str,
    workspace_name: &str,
) -> Result<()> {
    let tx = conn.begin_tx().await?;
    tx.execute("DELETE FROM pending_jobs WHERE id = ?1", params![run_id])
        .await
        .with_context(|| format!("failed to delete pending envelope for run {run_id}"))?;
    tx.execute("DELETE FROM jobs WHERE id = ?1", params![run_id])
        .await
        .with_context(|| format!("failed to delete research job row for run {run_id}"))?;
    insert_job_tx(
        &tx,
        run_id,
        cleanup_task,
        workspace_name,
        "",
        "",
        Role::Sanitation,
        &[NewAgent {
            agent_id: crate::research_cleanup::cleanup_agent_id(run_id),
            kind: AgentKind::Sanitation,
            idx: None,
            task: cleanup_task.to_string(),
        }],
        &SpawnChild::ResearchCleanup,
        None,
    )
    .await?;
    tx.commit().await?;
    Ok(())
}

/// Cancel the workspace's research/research_cleanup runs before its
/// workspace-row deletion: fires each run's cancel signal and cancels its
/// agents, then performs the cancelled-run cleanup handoff. A run with NO live
/// orchestrator is swept directly by [`crate::research_cancel::cancel_research_run`]
/// (dump-guarded folder release); a live orchestrator performs its own
/// cancelled-exit handoff (cleanup agent then folder release) after this
/// returns. Returns the cancelled-run count — every run whose cancel was fired,
/// whether swept directly (no live orchestrator) or handed off to a live
/// orchestrator's cancelled-exit handoff. The remaining job rows (ticket
/// phases, sync jobs) are deleted by the CALLER's workspace-row transaction,
/// so the jobs/workspace deletion commits atomically.
///
/// Known race: for a registered (live-orchestrator) run the handoff runs
/// asynchronously AFTER this function returns, while the caller deletes the
/// workspace row. The cleanup agent then runs against the orchestrator's
/// in-memory [`Workspace`] handle — it never re-resolves the workspace row — so
/// the folder is still released by the cleanup tail; the end state is correct.
/// The caller's blanket job/workspace DELETE and the handoff's
/// `research_cleanup` row creation are not in one tx, but nothing durable is
/// orphaned: the row created by the handoff is inserted after the caller's
/// DELETE and survives until the cleanup tail terminalizes it.
pub(crate) async fn abandon_workspace_research_runs(workspace_name: &str) -> Result<usize> {
    let conn = &crate::session::store().conn;
    let research_rows = conn
        .query(
            "SELECT id FROM jobs WHERE workspace_name = ?1 AND kind IN ('research','research_cleanup')",
            params![workspace_name],
        )
        .await
        .context("list research jobs for workspace abandon")?;
    let mut cancelled = 0usize;
    for row in &research_rows {
        let id: String = row.get(0)?;
        crate::research_cancel::cancel_research_run(&id).await;
        cancelled += 1;
    }
    Ok(cancelled)
}

// ── Sync caller-owned resume primitives ────────────────────────────────
//
// A sync analyze/implement job is owned by the caller session that launched it
// (`jobs.caller_agent_id` = the caller's stable session pin). The JOBS ROW is
// the source of truth for ownership. The caller's universal resume-completion
// step locates every launched caller-owned job by this pin and slot-resumes it.

/// Launched sync analyze/implement jobs owned by `caller_agent_id` (the
/// caller's stable session pin), newest first, awaiting deterministic resume
/// by the caller's universal resume-completion step. There is no freshness
/// gate: unfinished jobs are never deleted except by explicit abandon (the
/// /clear abandon path terminalizes a cleared session's jobs, which is what
/// previously made the fresh-cycle gate necessary).
pub(crate) async fn find_owned_launched_jobs(
    conn: &Connection,
    caller_agent_id: &str,
) -> Result<Vec<JobRow>> {
    let rows = conn
        .query(
            &format!(
                "SELECT {JOB_COLUMNS} FROM jobs j \
                 WHERE j.caller_agent_id = ?1 AND j.kind IN ('analyze','implement') AND j.status = 'launched' \
                   AND j.mode = 'sync' \
                 ORDER BY j.created_at DESC"
            ),
            params![caller_agent_id],
        )
        .await
        .context("find owned launched jobs")?;
    rows.iter().map(job_row_from).collect()
}

// ── Ticket phase-job substrate ─────────────────────────────────────────
//
// A ticket phase job is a short-lived `jobs` row whose `kind` equals the
// ticket's current phase (`analysis`, `in_development`, `in_diagnostics`,
// `in_review`, `in_qa`, `in_sanitation`) and whose `ticket_id` — stored
// directly on the job — links it to the ticket. `tickets.phase` is the sole
// durable running truth; the job is created by the single puller when a
// ticket in a working phase has no job, and deleted when the phase completes
// or is reset. The active agent(s) for a running job are held in the `agents`
// roster (status='launched'), which drives comment routing.

/// A phase job id (kind = phase, ticket_id set).
#[derive(Debug, Clone)]
pub(crate) struct TicketJobRow {
    pub id: String,
}

fn ticket_job_row_from(row: &Row) -> anyhow::Result<TicketJobRow> {
    Ok(TicketJobRow { id: row.get(0)? })
}

/// Load the launched phase job for a ticket, if one exists.
pub(crate) async fn find_phase_job(
    conn: &Connection,
    ticket_id: &str,
    phase: crate::pipeline::board::TicketPhase,
) -> Result<Option<TicketJobRow>> {
    conn.query_optional_cached(
        "SELECT id FROM jobs \
         WHERE ticket_id = ?1 AND kind = ?2 AND status = 'launched' LIMIT 1",
        params![ticket_id, phase.as_ref()],
        ticket_job_row_from,
    )
    .await
    .context("find ticket phase job")
}

/// Terminalize every launched phase job for a ticket (idempotent; no-op when
/// the ticket has none). Used when a ticket reaches a terminal phase so no
/// launched phase-job row lingers until the stale purge.
pub(crate) async fn complete_ticket_phase_jobs(conn: &Connection, ticket_id: &str) -> Result<()> {
    let rows = conn
        .query(
            "SELECT id FROM jobs WHERE ticket_id = ?1 AND status = 'launched'",
            params![ticket_id],
        )
        .await
        .context("find ticket phase jobs for completion")?;
    for row in rows {
        let id: String = row.get(0)?;
        terminalize_job(conn, &id).await?;
    }
    Ok(())
}

/// Update the phase job's stored `task`. Phase dispatch re-derives its prompt
/// from live state, so the stored value is not read for re-dispatch.
pub(crate) async fn update_phase_job_task(
    conn: &Connection,
    job_id: &str,
    task: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET task = ?2, updated_at = ?3 WHERE id = ?1",
        params![job_id, task, db::now()],
    )
    .await
    .context("update ticket phase job task")?;
    Ok(())
}

// ── Active-agent mapping via the agents roster (replaces tickets.assigned_to) ──

/// Upsert a job-bound roster row for a currently-running stage agent — the
/// active-agent marker that replaces `tickets.assigned_to`. `status='launched'`
/// marks the agent as running; a later call with `done`/`failed` clears it. The
/// agent roster is keyed on (job_id, agent_id) so the NULL-seat engineer anchor
/// (job_id=NULL) remains a distinct row for session-TTL continuity.
pub(crate) async fn upsert_job_agent(
    conn: &Connection,
    job_id: &str,
    agent_id: &str,
    kind: AgentKind,
    status: RowStatus,
) -> Result<()> {
    conn.execute(
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
         VALUES (?1, ?2, ?3, NULL, ?4, '') \
         ON CONFLICT(job_id, agent_id) DO UPDATE SET status = ?4",
        params![job_id, agent_id, kind.as_str(), status.as_str()],
    )
    .await
    .with_context(|| format!("failed to upsert job agent {agent_id} for job {job_id}"))?;
    Ok(())
}

/// List the currently-running agent ids for a ticket (`status='launched'`
/// roster rows across the ticket's phase jobs). Drives comment
/// routing to running agents.
pub(crate) async fn list_running_agents_for_ticket(
    conn: &Connection,
    ticket_id: &str,
) -> Result<Vec<String>> {
    let rows = conn
        .query(
            "SELECT a.agent_id FROM agents a \
             JOIN jobs j ON j.id = a.job_id \
             WHERE j.ticket_id = ?1 AND a.status = 'launched'",
            params![ticket_id],
        )
        .await
        .context("list running agents for ticket")?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        out.push(row.get::<String>(0)?);
    }
    Ok(out)
}

/// True when a job has any currently-running agent (`status='launched'` roster
/// row). Drives the implementation-iteration re-dispatch guard.
pub(crate) async fn job_has_launched_agents(conn: &Connection, job_id: &str) -> Result<bool> {
    Ok(conn
        .query_optional_cached(
            "SELECT 1 FROM agents WHERE job_id = ?1 AND status = 'launched' LIMIT 1",
            params![job_id],
            |_| Ok::<_, anyhow::Error>(1_i64),
        )
        .await?
        .is_some())
}

// ── Slot-resume skeleton (shared by the pipeline parallel phases + AnalyzeTool) ──
//
// An interrupted phase/analyze round is resumable from its roster: already-Done
// slots are reconstructed from their stored `outcome` (per-consumer hook),
// not-Done slots are re-run with their stored per-slot `task`. This module owns
// the shared partition/discriminator helpers so the two consumers never
// re-implement the split.

/// The slot-resume partition of a job's roster: Done slots (reconstructable
/// from their stored outcome) vs not-Done slots (re-run with their stored
/// task). Refers into the input [`AgentRow`]s — the caller owns the lifetime.
#[derive(Debug)]
pub(crate) struct SlotResume<'a> {
    pub done: Vec<&'a AgentRow>,
    pub not_done: Vec<&'a AgentRow>,
}

/// Partition a roster into Done and not-Done slots for a slot-resume.
/// A slot is Done iff its stored status is `done`; every other status
/// (`launched`/`failed`) is not-Done and must be re-run, with its stored
/// task, on resume. The per-consumer restorer turns a Done slot's stored
/// `outcome` back into the consumer's artifact (pipeline: deserialize the
/// verdict; AnalyzeTool: re-extract from the raw response).
#[must_use]
pub(crate) fn split_slot_resume(roster: &[AgentRow]) -> SlotResume<'_> {
    let mut done = Vec::new();
    let mut not_done = Vec::new();
    for row in roster {
        if row.status == RowStatus::Done.as_str() {
            done.push(row);
        } else {
            not_done.push(row);
        }
    }
    SlotResume { done, not_done }
}

/// Mark a phase job's stale `launched` roster rows as `failed` so the
/// re-dispatch guard sees no running agents while the rows — and their stored
/// per-slot tasks — survive for a slot-resume. Unlike
/// [`clear_launched_agents_for_job`] this preserves the not-Done slots instead
/// of discarding them. Used on pause-freeze and boot recovery for phase jobs.
pub(crate) async fn interrupt_phase_job_roster(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute(
        "UPDATE agents SET status = 'failed' WHERE job_id = ?1 AND status = 'launched'",
        params![job_id],
    )
    .await
    .context("mark interrupted phase roster rows failed")?;
    Ok(())
}

/// Re-arm a resumed round's not-Done roster slots as `launched` so the
/// `job_has_launched_agents` running-signal blocks a concurrent re-drive and
/// the Running Agents view lists the in-flight members. Also touches the job's
/// `updated_at` — phase jobs are still purge-eligible on stale updated_at (the
/// phase-only purge keys off `jobs.updated_at`), so the touch keeps a
/// long-running resumed round from being swept. The stored `idx`/`task` are
/// preserved.
pub(crate) async fn rearm_roster_launched(
    conn: &Connection,
    job_id: &str,
    agent_ids: &[String],
) -> Result<()> {
    if agent_ids.is_empty() {
        return Ok(());
    }
    let placeholders = db::sql_in_placeholders(agent_ids.len());
    let mut params: Vec<Value> = vec![job_id.into()];
    params.extend(agent_ids.iter().map(|id| Value::from(id.as_str())));
    conn.execute(
        &format!(
            "UPDATE agents SET status = 'launched' WHERE job_id = ?1 AND agent_id IN ({placeholders})"
        ),
        params,
    )
    .await
    .context("re-arm resumed roster slots as launched")?;
    conn.execute(
        "UPDATE jobs SET updated_at = ?2 WHERE id = ?1",
        params![job_id, db::now()],
    )
    .await
    .context("touch job updated_at on slot resume")?;
    Ok(())
}

/// Clear a job's stale 'launched' roster rows after a stage-completion handoff
/// (the previous phase's in-flight agents are no longer running). The job row
/// survives. For an interrupted PARALLEL phase round the task-preserving
/// [`interrupt_phase_job_roster`] is used instead so the not-Done slots'
/// stored tasks survive for a slot-resume.
pub(crate) async fn clear_launched_agents_for_job(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM agents WHERE job_id = ?1 AND status = 'launched'",
        params![job_id],
    )
    .await
    .context("clear launched agents for job")?;
    Ok(())
}

/// Delete the jobs row (CASCADE destroys research_jobs child rows)
/// inside an existing transaction. The agents roster is cascade-deleted with
/// the job, so no separate active-agent cleanup is needed.
async fn delete_job_tx(tx: &TxGuard<'_>, job_id: &str) -> Result<()> {
    tx.execute("DELETE FROM jobs WHERE id = ?1", params![job_id])
        .await
        .with_context(|| format!("failed to delete job {job_id}"))?;
    Ok(())
}

/// Resolve the consumer-loop agent ID for an envelope (Manager role → the
/// workspace Manager session; otherwise the caller's direct session).
/// `session::resolve_agent_id` already branches on `role == "manager"`
/// internally, so this is the single canonical form — analyze/research delivery
/// paths call it instead of reimplementing the branch.
pub(crate) fn envelope_target(job: &AgentJob) -> String {
    crate::session::resolve_agent_id(&job.user_name, job.role.as_str(), &job.workspace_name)
}

/// Caller identity + task of a job, read from the `jobs` row ALONE — child
/// rows are never required for resume delivery, so a job whose child
/// row is missing (e.g. a crash between the spawn tx and a child insert)
/// still resumes and delivers to the original caller instead of being
/// stranded.
pub(crate) struct JobCaller {
    pub task: String,
    pub role: String,
    pub user_name: String,
    pub channel: String,
}

/// Discriminated resume preamble so sync callers never mistake a gone job
/// for a drain-cut, or a transient DB read error for an abandon.
pub(crate) enum ResumePreamble {
    /// Caller identity + parsed role — the resume may proceed.
    Proceed(JobCaller, Role),
    /// Drain/shutdown active — quiet abort, the job row is untouched.
    DrainCut,
    /// The jobs row is absent — only possible via an explicit abandon (session
    /// /clear or workspace deletion) racing this resume; no terminalization
    /// happens (the row is already gone). The old terminalize-on-missing
    /// compensation existed only because jobs could be purged.
    Gone,
    /// The job row could not be READ (transient DB error) — NOT an abandon.
    /// The row is untouched; the error is carried for infra-failure reporting.
    Unreadable(anyhow::Error),
}

/// Terminal outcome of a resumed sync (analyze/implement) round.
pub(crate) enum SyncResumeOutcome {
    /// The round reached a terminal result (Ok text or Err) — the caller owns
    /// delivery and terminalization.
    Terminal(Role, JobCaller, anyhow::Result<String>),
    /// Drain-cut mid-resume — outcomes checkpointed, the job stays launched.
    DrainCut,
    /// The job row was explicitly abandoned between the ownership lookup and
    /// the resume.
    Gone,
}

/// Load the task + caller identity of a job from the `jobs` row alone.
pub(crate) async fn job_caller(conn: &Connection, job_id: &str) -> Result<Option<JobCaller>> {
    conn.query_optional(
        "SELECT task, role, user_name, channel FROM jobs WHERE id = ?1",
        params![job_id],
        |r| {
            Ok::<_, anyhow::Error>(JobCaller {
                task: r.get::<String>(0)?,
                role: r.get::<String>(1)?,
                user_name: r.get::<String>(2)?,
                channel: r.get::<String>(3)?,
            })
        },
    )
    .await
    .context("load job caller")
}

/// Boot-resume preamble shared by the analyze/research resume paths. Order is
/// load-bearing: aborting-guard FIRST (quiet return — the job row stays for the
/// next boot), then job_caller, then the gone-missing note. Returns the caller
/// and parsed role, or None when the caller should quiet-return. The
/// `abort_site`/`missing_site` params inject each call site's label into its
/// log line so the two callers' texts stay distinguishable.
pub(crate) async fn resume_job_preamble(
    conn: &Connection,
    job_id: &str,
    abort_site: &str,
    missing_site: &str,
) -> Option<(JobCaller, Role)> {
    match resume_job_preamble_discrete(conn, job_id, abort_site, missing_site).await {
        ResumePreamble::Proceed(caller, caller_role) => Some((caller, caller_role)),
        // Quiet abort for boot-resume paths: the row is untouched (drain),
        // already gone (abandon), or unreadable (transient DB error — retried
        // at the next boot).
        ResumePreamble::DrainCut | ResumePreamble::Gone | ResumePreamble::Unreadable(_) => None,
    }
}

/// Discriminating core of [`resume_job_preamble`]: same order (aborting guard,
/// then job_caller, then the gone-missing note) and same log texts, but the
/// caller gets an explicit [`ResumePreamble`] so a sync caller can tell a gone
/// job apart from a drain-cut.
pub(crate) async fn resume_job_preamble_discrete(
    conn: &Connection,
    job_id: &str,
    abort_site: &str,
    missing_site: &str,
) -> ResumePreamble {
    if crate::shutdown::aborting() {
        tracing::info!(job = %job_id, "{abort_site} aborted — drain/shutdown in progress");
        return ResumePreamble::DrainCut;
    }
    match job_caller(conn, job_id).await {
        Ok(Some(caller)) => {
            let caller_role = std::str::FromStr::from_str(&caller.role).unwrap_or(Role::Manager);
            ResumePreamble::Proceed(caller, caller_role)
        }
        // The row is only ever absent because an explicit abandon raced this
        // resume — report it as such.
        Ok(None) => {
            tracing::info!(job = %job_id, "{missing_site}: job row gone (explicitly abandoned) — skipping resume");
            ResumePreamble::Gone
        }
        // A transient DB read error is NOT an abandon — warn, and surface it
        // as an infra failure for the sync caller (fail-safe: no deletion,
        // the row is untouched either way).
        Err(e) => {
            tracing::warn!(job = %job_id, error = %e, "{missing_site}: failed to read job row — skipping resume");
            ResumePreamble::Unreadable(e)
        }
    }
}

/// Read a job's boot-resume retry count (0 = never boot-resumed). Used by the
/// research report's best-effort-telemetry gate (the real resume signal).
pub(crate) async fn job_retry_count(conn: &Connection, job_id: &str) -> i64 {
    match conn
        .query_optional(
            "SELECT retry_count FROM jobs WHERE id = ?1",
            params![job_id],
            |r| r.get::<i64>(0),
        )
        .await
    {
        Ok(Some(n)) => n,
        Ok(None) => 0,
        Err(e) => {
            warn!(job = %job_id, error = %e, "Failed to read job retry_count — assuming 0");
            0
        }
    }
}

// ── Queries ─────────────────────────────────────────────────────────────

/// Load all non-terminal jobs (launched|failed) for the boot scan.
pub(crate) async fn list_active_jobs(conn: &Connection) -> Result<Vec<JobRow>> {
    let rows = conn
        .query(
            &format!(
                "SELECT {JOB_COLUMNS} FROM jobs \
                 WHERE status != 'done' ORDER BY created_at"
            ),
            (),
        )
        .await
        .context("list active jobs")?;
    rows.iter().map(job_row_from).collect()
}

/// Load the agent roster for a job.
pub(crate) async fn list_agents_for_job(conn: &Connection, job_id: &str) -> Result<Vec<AgentRow>> {
    let rows = conn
        .query(
            "SELECT idx, agent_id, status, outcome, task \
             FROM agents WHERE job_id = ?1 ORDER BY idx",
            params![job_id],
        )
        .await
        .context("list agents for job")?;
    rows.iter().map(agent_row_from).collect()
}

/// Graceful-drain completion watcher.
///
/// Waits for the drain flag, then polls the agent registry AND the non-agent
/// call registry. Clean exit: no agent registered AND no orchestrator-only
/// LLM call in flight → every in-flight round has unwound (each running agent
/// is registered until its finalize_session completes; orchestrator calls —
/// analyze consolidation, research synthesis, joint-verdict grouping — are
/// tracked in [`crate::agent::registry::NON_AGENT_CALLS`]; the research
/// orchestrator additionally holds a whole-run guard, so the token is never
/// fired into an inter-phase window between analyst deregistration and the
/// next orchestrator call) → fires the global token, which the GUI
/// subscription turns into window exit. Cap expiry (10 min): force-cancel
/// stragglers — in-flight ops with >10 min remaining budget are
/// guaranteed-aborted and boot-resume via status='launched'.
///
/// Residual millisecond windows (accepted, bounded): a dispatch task's
/// untracked orchestration tail (board transition + completion tx after the
/// last agent deregisters / guard releases) and the registration start of a
/// phase body's joint-comment synthesis can both see both registries empty
/// for a few ms — the token may fire just before those DB writes land
/// (bounded by tx rollback + boot self-heal) or into a just-started call
/// (bounded: the 2s poll makes it practically unreachable, and a cut call
/// recovers via the job's checkpointed state).
///
/// The jobs table is NOT polled for completion: drain-cut rounds intentionally
/// leave their jobs status='launched' for boot resume, so
/// "no launched jobs remain" is unreachable in the common case — a count-based
/// wait would hold the window open for the full cap even after the work
/// unwound in seconds. The registries are the authoritative in-flight signal;
/// a just-spawned round that races the drain start (job committed, agents not
/// yet registered) is cut at its first LLM call and boot-resumes — bounded
/// and self-healing.
///
/// The GUI window stays open for the whole drain: iced::exit is deferred until
/// the token fires here (the drain only works while the iced
/// runtime lives).
pub async fn run_drain_watch() {
    use crate::shutdown::{force_cancel, shutdown, shutdown_token};
    use std::time::Instant;

    // Wait for the drain to begin (or an already-fired token).
    let token = shutdown_token();
    tokio::select! {
        () = crate::shutdown::drain_wait() => {}
        () = token.cancelled() => return,
    }
    if shutdown_token().is_cancelled() {
        return;
    }

    let start = Instant::now();
    loop {
        // Clean-exit requires BOTH the agent registry AND the non-agent call
        // registry to be empty: orchestrator-only LLM calls (analyze consolidation,
        // research synthesis, joint-verdict grouping) run with no registered
        // agents — firing the token the instant the registry empties would
        // abort an in-flight call via the provider token race the design
        // forbids. Drain-cut leftovers are covered: cut rounds leave their
        // agents unregistered AND their jobs status='launched' (never
        // re-registered), so the registries drain without them.
        if crate::agent::registry::AGENT_REGISTRY.list().is_empty()
            && crate::agent::registry::NON_AGENT_CALLS.list().is_empty()
        {
            info!("Drain complete — no in-flight agents or orchestrator calls; exiting");
            shutdown();
            return;
        }
        if start.elapsed() >= Duration::from_secs(DRAIN_CAP_SECS) {
            warn!("Drain cap ({DRAIN_CAP_SECS}s) reached — force-cancelling in-flight work");
            crate::agent::registry::AGENT_REGISTRY.shutdown_all();
            force_cancel();
            return;
        }
        if shutdown_token().is_cancelled() {
            return;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

/// Load all outstanding pending_jobs rows (boot replay + drain-recovery).
pub(crate) async fn list_pending_jobs(conn: &Connection) -> Result<Vec<PendingJobRow>> {
    let rows = conn
        .query(
            "SELECT id, target_agent_id, envelope, created_at \
             FROM pending_jobs ORDER BY created_at",
            (),
        )
        .await
        .context("list pending jobs")?;
    rows.iter().map(pending_row_from).collect()
}

/// Delete a delivered pending_jobs row (consumer-confirmed).
pub(crate) async fn delete_pending_job(conn: &Connection, id: &str) -> Result<()> {
    conn.execute("DELETE FROM pending_jobs WHERE id = ?1", params![id])
        .await
        .context("delete pending job")?;
    Ok(())
}

// ── Boot recovery scan ─────────────────────────────────────────────────

/// Outcome of a boot recovery scan: jobs selected for resume.
///
/// Ticket phase jobs are NOT resumed at boot — the single puller re-drives a
/// ticket in a working phase (its phase job row is left in place with stale
/// roster rows cleared, and the puller re-dispatches). Only the non-ticket
/// envelope kinds carry their own resume semantics.
pub(crate) enum ResumableJob {
    Research {
        job_id: String,
        workspace_name: String,
    },
    Analyze {
        job_id: String,
        workspace_name: String,
    },
    Implement {
        job_id: String,
        workspace_name: String,
    },
    /// A research-run cleanup Sanitation agent interrupted by a crash. The
    /// jobs row id == the run id, so the row is the cleanup's resume marker
    /// and holds the folder until the cleanup completes.
    ResearchCleanup {
        job_id: String,
        workspace_name: String,
    },
}

/// Strip the `<timestamp>…</timestamp>` wrapper from a persisted user message,
/// recognizing both the legacy prefix format
/// (`<timestamp>…</timestamp>\n\n{content}`) and the current suffix format
/// (`{content}\n\n<timestamp>…</timestamp>`). Returns the input unchanged when
/// neither wrapper is present. The timestamp body is validated against the
/// [`crate::session::render_timestamp`] shape so content that merely mimics
/// the wrapper is never mis-stripped. Residual ambiguity (inherent to the
/// format): a suffix message whose content begins with a full legacy-shaped
/// block is byte-identical to the legacy format and strips from the front —
/// unreachable in practice, and fails toward duplicate delivery (never loss).
fn strip_timestamp_wrapper(content: &str) -> &str {
    // Suffix format — only for non-legacy-prefixed messages: a legacy message
    // whose content happens to end with a timestamp-shaped block must not be
    // stripped from the tail.
    if !content.starts_with("<timestamp>")
        && let Some(body) = content.strip_suffix("</timestamp>")
        && let Some(start) = body.rfind("\n\n<timestamp>")
        && is_timestamp_body(&body[start + "\n\n<timestamp>".len()..])
    {
        return &body[..start];
    }
    // Legacy prefix format — requires the exact `\n\n` separator so a
    // suffix-format message whose content merely STARTS with a timestamp
    // block is never mis-stripped.
    if let Some(body) = content.strip_prefix("<timestamp>")
        && let Some(end) = body.find("</timestamp>")
        && is_timestamp_body(&body[..end])
        && let Some(after) = body[end..].strip_prefix("</timestamp>\n\n")
    {
        return after;
    }
    content
}

/// Whether `s` matches the `%Y-%m-%d %H:%M:%S (%Z)` shape produced by
/// [`crate::session::render_timestamp`]. The tz must be non-empty — chrono
/// Local's %Z always renders on macOS; an empty one would fail validation and
/// boot-replay dedup would degrade to duplicate delivery (never loss).
fn is_timestamp_body(s: &str) -> bool {
    let Some((datetime, tz)) = s.strip_suffix(')').and_then(|rest| rest.rsplit_once(" (")) else {
        return false;
    };
    !tz.is_empty() && chrono::NaiveDateTime::parse_from_str(datetime, "%Y-%m-%d %H:%M:%S").is_ok()
}

/// Dedup check: was the pending envelope already appended to the target
/// session? Every user message persists with a `<timestamp>…</timestamp>`
/// block (legacy prefix / current suffix, and Manager jobs get the
/// ticket-buffer drain prefix), so exact equality would always be false — a
/// SUFFIX match after stripping the timestamp wrapper is required. Plus a
/// created_at tiebreaker: skip only if pending.created_at <= the last user
/// message's created_at (closes the silent at-most-once hole for identical
/// consecutive messages). Both timestamps are compared via
/// [`crate::db::parse_utc_timestamp`] (chrono parsing, not raw lexical
/// string comparison — AutoSi trailing-zero trimming can theoretically
/// misorder sub-microsecond-different timestamps).
async fn pending_already_appended(conn: &Connection, row: &PendingJobRow) -> bool {
    let Ok(Some((last_content, last_created))) = conn
        .query_optional(
            "SELECT content, created_at FROM sessions \
             WHERE agent_id = ?1 AND role = 'user' ORDER BY id DESC LIMIT 1",
            params![row.target_agent_id.clone()],
            |r| Ok::<_, anyhow::Error>((r.get::<String>(0)?, r.get::<String>(1)?)),
        )
        .await
    else {
        return false;
    };
    let Ok(envelope) = serde_json::from_str::<AgentJob>(&row.envelope) else {
        return false;
    };
    // chrono parse beats raw lexical comparison (AutoSi trailing-zero
    // trimming can theoretically misorder sub-microsecond-different
    // timestamps); lexical order remains the fallback if either string is
    // unparseable (unreachable in practice — both come from db::now()).
    let appended_before = match (
        crate::db::parse_utc_timestamp(&row.created_at).ok(),
        crate::db::parse_utc_timestamp(&last_created).ok(),
    ) {
        (Some(row_ts), Some(last_ts)) => row_ts <= last_ts,
        _ => row.created_at <= last_created,
    };
    strip_timestamp_wrapper(&last_content).ends_with(&envelope.content) && appended_before
}

/// (0) Replay outstanding pending_jobs — the boot reclaim path for durable
/// envelopes (at-least-once delivery). Runs before anything that could purge.
///
/// Dedup semantics: the suffix + created_at tiebreaker closes the
/// listener-interleaving race — the scan and channel listeners may run
/// concurrently, the consumer serializes per agent.
async fn replay_pending_jobs(conn: &Connection) -> Result<usize> {
    let rows = list_pending_jobs(conn).await?;
    let mut replayed = 0usize;
    for row in &rows {
        let Ok(mut job) = serde_json::from_str::<AgentJob>(&row.envelope) else {
            warn!(pending_job = %row.id, "Pending job envelope unreadable — skipping");
            continue;
        };
        job.pending_job_id = Some(row.id.clone());
        if pending_already_appended(conn, row).await {
            // The message already reached the session — dedup prevents a
            // duplicate append. The row is reclaimed here (delivery complete;
            // response recovery is the dead-session poller's domain).
            debug!(pending_job = %row.id, "Pending job already appended — skipping");
            if let Err(e) = delete_pending_job(conn, &row.id).await {
                warn!(pending_job = %row.id, error = %e, "Failed to delete deduped pending job");
            }
            continue;
        }
        // Crash-window cleanup dispatch: a genuinely-undelivered
        // research-completion envelope whose run has NO `research_cleanup`
        // jobs row means the daemon died between complete_durable_job and
        // dispatch_research_cleanup — the cleanup agent never ran. Create the
        // durable row NOW (the dedup marker + folder-hold); the
        // `research_cleanup` boot-scan arm below — which reads
        // list_active_jobs AFTER this replay — is the SOLE dispatcher for the
        // agent (spawning here would double-run the same `cleanup_{run_id}`
        // id: the scan arm would register it via AgentRegistry, replacing +
        // cancelling this dispatch). Runs whose cleanup was already dispatched
        // have the row and are skipped here (their cleanup resumes via the
        // `research_cleanup` boot-scan arm).
        // `dispatch_cleanup_for_pending_envelope` uses RUN-FOLDER EXISTENCE as
        // its replay discriminator (crash-window classifier only, not a
        // dispatch condition): a folder still present means the cleanup never
        // ran; a folder already gone means the cleanup COMPLETED in a previous
        // lifetime while the envelope stayed undelivered (dead target session)
        // — no re-dispatch, so no per-boot duplicate cleanup LLM round.
        // Deliberately after the appended-dedup check: an already-appended
        // envelope proves dispatch ran in the previous lifetime, so a missing
        // row there means the cleanup COMPLETED — re-dispatching would be a
        // duplicate LLM round.
        if job.kind == MessageKind::ResearchResult
            && !crate::research_cleanup::research_cleanup_row_exists(conn, &row.id)
                .await
                .unwrap_or(true)
        {
            crate::research_cleanup::dispatch_cleanup_for_pending_envelope(&row.id, &job).await;
        }
        // The dedup above is the authoritative "in session" check: a row whose
        // content is NOT in the session was never appended (interrupted between
        // route and the consumer's append) — route the FULL content. Replaying
        // empty here would silently lose the envelope forever (at-most-once
        // violation).
        crate::agent::message_router::route(&row.target_agent_id, job);
        replayed += 1;
    }
    Ok(replayed)
}

/// Stale-job purge.
///
/// Cutoff = 8h. Deletes only stale TICKET-PHASE jobs (kinds matching
/// [`is_ticket_phase_kind`]) whose updated_at predates the cutoff AND whose
/// roster agents' sessions are all stale too (live sessions referenced by
/// unfinished jobs are NEVER purged — the agents table IS the marker).
/// A stale ticket phase job is deleted here; the puller re-creates it for the
/// ticket (which stays in its phase) on the next tick — `tickets.phase` is the
/// sole authority, so no board rollback is needed. Paused workspaces' phase
/// jobs are purge-immune (they hold the running slot while frozen).
/// Unfinished non-phase jobs are never time-deleted (explicit abandon only).
pub async fn purge_stale_jobs(cutoff: &str) -> Result<u64> {
    let conn = &crate::session::store().conn;

    // A paused (frozen) phase job is purge-immune: it holds the workspace's
    // running slot while frozen. The kind filter mirrors `is_ticket_phase_kind`.
    let phase_kinds = TICKET_PHASE_KINDS
        .iter()
        .map(|kind| format!("'{kind}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let rows = conn
        .query(
            &format!(
                "SELECT j.id, j.kind, j.workspace_name, j.ticket_id \
                 FROM jobs j \
                 WHERE j.updated_at < ?1 \
                   AND j.kind IN ({phase_kinds}) \
                   AND NOT EXISTS ( \
                     SELECT 1 FROM agents a \
                     JOIN session_metadata sm ON sm.agent_id = a.agent_id \
                     WHERE a.job_id = j.id AND sm.last_activity >= ?1)"
            ),
            params![cutoff],
        )
        .await
        .context("select stale jobs for purge")?;
    if rows.is_empty() {
        return Ok(0);
    }
    // Paused workspace names: a frozen phase job must survive the stale purge.
    // Best-effort — a `workspaces` table read failure falls back to purging.
    let paused_ws: std::collections::HashSet<String> = match crate::workspace::store().list().await
    {
        Ok(list) => list
            .into_iter()
            .filter(|w| w.paused)
            .map(|w| w.name)
            .collect(),
        Err(_) => std::collections::HashSet::new(),
    };
    let mut purge_ids: Vec<String> = Vec::with_capacity(rows.len());
    for row in &rows {
        let id: String = row.get(0)?;
        let workspace_name: String = row.get(2)?;
        // A paused workspace's phase job is frozen — purge-immune.
        if paused_ws.contains(&workspace_name) {
            continue;
        }
        purge_ids.push(id);
    }
    if purge_ids.is_empty() {
        return Ok(0);
    }

    // DELETE FROM jobs (CASCADE removes roster rows; the engineer anchor
    // survives — NULL FK child).
    let tx = conn.begin_tx().await?;
    let mut deleted = 0usize;
    for id in &purge_ids {
        delete_job_tx(&tx, id).await?;
        deleted += 1;
    }
    tx.commit().await?;
    if deleted > 0 {
        tracing::debug!(deleted, "Purged stale jobs");
    }
    Ok(deleted as u64)
}

/// `jobs.kind` values of the ticket working-phase kinds — the single source
/// shared by [`is_ticket_phase_kind`] and the purge's SQL filter.
pub(crate) const TICKET_PHASE_KINDS: &[&str] = &[
    "analysis",
    "in_development",
    "in_diagnostics",
    "in_review",
    "in_qa",
    "in_sanitation",
];

/// Is this `jobs.kind` one of the ticket working-phase kinds?
#[must_use]
pub(crate) fn is_ticket_phase_kind(kind: &str) -> bool {
    TICKET_PHASE_KINDS.contains(&kind)
}

/// Sync jobs (mode='sync') are settled by the owner's live resume-completion
/// step ([`find_owned_launched_jobs`]) — boot resume must never race it.
/// Research jobs always spawn with mode='async', so within the research/analyze
/// branch only the analyze half can fire.
fn skip_sync_job(job: &JobRow) -> bool {
    if job.mode == JobMode::Sync {
        info!(
            job = %job.id,
            caller = ?job.caller_agent_id,
            "sync job — caller resumes it; skipping boot resume",
        );
        true
    } else {
        false
    }
}

/// Binding 3: a resume-eligible job whose workspace can no longer be resolved
/// is left in place, not terminalized — the row is deleted only by explicit
/// abandon, so the boot loop skips it without bumping retry_count or touching
/// the roster. Returns `true` when the job should be skipped.
async fn workspace_unresolvable(job: &JobRow) -> bool {
    if crate::users::resolve_workspace(&job.workspace_name)
        .await
        .is_ok_and(|ws| ws.is_some())
    {
        return false;
    }
    warn!(
        job = %job.id,
        workspace = %job.workspace_name,
        "workspace unresolvable — job stays launched, skipped (no boot resume, no checkpoint)",
    );
    true
}

/// Boot recovery scan: first statement of run_management. Order: (0) replay
/// pending_jobs; (1) one scan over the active jobs — ticket phase jobs only get
/// their stale launched roster cleared (the puller re-drives them), the
/// research/analyze/implement/research_cleanup kinds resume, and temp_cleanup
/// rows are terminalized. A resume-eligible job whose workspace is unresolvable
/// is skipped in place (binding 3), never terminalized.
///
/// Every resumed job gets updated_at = now (the boot bump — the ONLY
/// protection for the pre-first-commit window; Session::init with an empty
/// message skips the last_activity upsert) and retry_count + 1 (every bump
/// here is one boot resume; in-job retries are not counted).
#[expect(clippy::cast_possible_truncation)]
pub(crate) async fn recover_from_restart() -> Result<Vec<ResumableJob>> {
    let start = std::time::Instant::now();
    let conn = &crate::session::store().conn;

    // (0) Deliverable replay — before anything that could purge.
    let replayed = replay_pending_jobs(conn).await?;

    // (1) Scan over the active jobs. Ticket phase jobs (kind = the ticket's
    // current phase) are NOT kind-discriminated resume targets — the single
    // puller re-drives them. We only clear their stale launched roster rows
    // (the previous process's agents are dead) so the puller re-dispatches.
    // The research/analyze/implement/research_cleanup/temp_cleanup kinds
    // resume or terminalize in line. The kind only selects the `ResumableJob`
    // variant.
    let jobs = list_active_jobs(conn).await?;
    let mut resumable: Vec<ResumableJob> = Vec::new();
    let mut resumed_other = 0usize;
    for job in &jobs {
        if is_ticket_phase_kind(&job.kind) {
            // A ticket phase job interrupted by a crash: any 'launched'
            // roster rows from the previous process are stale. They are
            // marked 'failed' (NOT deleted) so the puller re-dispatches a
            // slot-resume that reuses the stored per-slot tasks and
            // already-Done outcomes rather than re-deriving a fresh round.
            if let Err(e) = interrupt_phase_job_roster(conn, &job.id).await {
                warn!(
                    job = %job.id,
                    ticket = %job.ticket_id.as_deref().unwrap_or("?"),
                    error = %e,
                    "Failed to mark stale launched agents on ticket phase boot scan",
                );
            }
        } else if job.kind == "research" || job.kind == "analyze" {
            // Resume at the roster/state level (dispatch re-enters the
            // orchestrator with the stored task). Always bump retry_count.
            if skip_sync_job(job) || workspace_unresolvable(job).await {
                continue;
            }
            let kind = job.kind.as_str();
            let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
            resumable.push(if kind == "research" {
                ResumableJob::Research {
                    job_id: job.id.clone(),
                    workspace_name: job.workspace_name.clone(),
                }
            } else {
                ResumableJob::Analyze {
                    job_id: job.id.clone(),
                    workspace_name: job.workspace_name.clone(),
                }
            });
            resumed_other += 1;
        } else if job.kind == "implement" {
            // A single-coder implement round interrupted by a crash: resume it
            // like any other durable envelope kind. The coder roster row holds
            // the terminal outcome; the job row holds the caller identity.
            if skip_sync_job(job) || workspace_unresolvable(job).await {
                continue;
            }
            let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
            resumable.push(ResumableJob::Implement {
                job_id: job.id.clone(),
                workspace_name: job.workspace_name.clone(),
            });
            resumed_other += 1;
        } else if job.kind == "research_cleanup" {
            // A research-run cleanup Sanitation agent interrupted by a
            // crash. Resume it like any other durable job. The folder stays
            // held until the resumed tail releases it; a row removed by the
            // dump-guarded cancel sweep deliberately leaves a dump-holding
            // folder for the cleanup tail / OS sweep.
            if workspace_unresolvable(job).await {
                continue;
            }
            let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
            resumable.push(ResumableJob::ResearchCleanup {
                job_id: job.id.clone(),
                workspace_name: job.workspace_name.clone(),
            });
            resumed_other += 1;
        } else if job.kind == "temp_cleanup" {
            // A periodic OS temp-dir cleaner interrupted by a crash.
            // Fire-and-forget: the cleanup is best-effort and the next
            // scheduled pass re-runs it, so a leftover row is
            // terminalized (never resumed). The cleaner's workspace is a
            // synthetic ephemeral name that is never registered in
            // the `workspaces` table — resuming would hit run_management's
            // unresolvable-workspace path; terminalizing here keeps that
            // explicit and skips the catch-all warning.
            info!(
                job = %job.id,
                "Temp-dir cleanup row left over from a previous lifetime — terminalizing (fire-and-forget, no resume)",
            );
            let _ = terminalize_job(conn, &job.id).await;
        } else {
            warn!(job = %job.id, kind = %job.kind, "Unknown job kind — skipping");
        }
    }

    let elapsed = start.elapsed();
    info!(
        duration_ms = elapsed.as_millis() as u64,
        resumed_other,
        replayed_pending = replayed,
        "Boot recovery scan complete",
    );

    // Boot sweep for session pins (S5 terminal deletion): pins for
    // tickets already in a terminal phase are removed so the TTL guard stops
    // protecting their accumulated sessions (the 5-min archive loop re-runs
    // this — boot just catches up after downtime).
    let _ = purge_terminal_session_pins().await;

    Ok(resumable)
}

// ── Session pins (S5 — permanently-NULL seats) ──────────────────────────

/// Deterministic session-pin agent ID: the per-ticket accumulated session
/// `ticket_{ticket_id}_{role}`, permanently job_id = NULL.
#[must_use]
pub(crate) fn session_pin_id(ticket_id: &str, role: Role) -> String {
    crate::session::ticket_agent_id(ticket_id, role.as_str())
}

/// Upsert the session pin for `role` (engineer/sanitation). NEVER sets job_id
/// (setting it removes the row from the partial-index scope → later NULL
/// insert no longer conflicts → duplicate anchor). The DDL and this UPSERT use
/// the IDENTICAL syntactic WHERE form (`job_id IS NULL` on both sides).
pub(crate) async fn upsert_session_pin(
    conn: &Connection,
    ticket_id: &str,
    task: &str,
    status: RowStatus,
    role: Role,
) -> Result<()> {
    let pin_id = session_pin_id(ticket_id, role);
    conn.execute(
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, outcome, task) \
         VALUES (NULL, ?1, ?4, NULL, ?2, NULL, ?3) \
         ON CONFLICT(agent_id) WHERE job_id IS NULL \
         DO UPDATE SET status = ?2, task = ?3, outcome = NULL",
        params![pin_id, status.as_str(), task, role.as_str()],
    )
    .await
    .with_context(|| format!("failed to upsert {role} session pin for ticket {ticket_id}"))?;
    Ok(())
}

/// Parse `ticket_{ticket_id}_{role}` — the ticket id is everything before the
/// final `_{role}` (workspace names allow underscores, so ticket ids like
/// `my_ws-42` are valid — only the final suffix is stripped).
#[must_use]
pub(crate) fn session_pin_ticket_id(agent_id: &str, role: Role) -> Option<String> {
    agent_id
        .strip_prefix("ticket_")
        .and_then(|rest| {
            rest.strip_suffix(role.as_str())
                .and_then(|r| r.strip_suffix('_'))
        })
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

/// Terminal deletion of session-pins (S5 — permanently-NULL seats): pins
/// protect the accumulated engineer/sanitation session across bounce rounds and
/// re-attempts. Once the ticket reaches a terminal phase (Done/Cancelled/Failed
/// — no future bounces/resets), the pin is removed so the TTL guard stops
/// protecting the session (cleaned ≤8h later). Centralized here (5-min archive
/// loop + boot sweep). Idempotent: parse ticket_id from the pin agent_id,
/// phase-check on the board side (shared consolidated connection), delete.
pub(crate) async fn purge_terminal_session_pins() -> usize {
    let conn = &crate::session::store().conn;
    let Ok(rows) = conn
        .query(
            "SELECT agent_id, kind FROM agents WHERE job_id IS NULL \
             AND kind IN ('engineer', 'sanitation')",
            (),
        )
        .await
    else {
        return 0;
    };
    let mut deleted = 0usize;
    // Guarded access — the purge may run before BOARD is initialized (the
    // archive loop checks later in the same tick; boot sweep runs after init).
    let board_ready = crate::pipeline::board::BOARD.get().is_some();
    for row in &rows {
        let Ok(agent_id) = row.get::<String>(0) else {
            continue;
        };
        let Ok(kind) = row.get::<String>(1) else {
            continue;
        };
        let role = match kind.as_str() {
            "engineer" => Role::Engineer,
            _ => Role::Sanitation,
        };
        let ticket_id = session_pin_ticket_id(&agent_id, role);
        let Some(ticket_id) = ticket_id else {
            continue;
        };
        let terminal = board_ready
            && crate::pipeline::board::store()
                .get_ticket_phase(&ticket_id)
                .await
                .is_ok_and(|p| p.is_some_and(|ph| ph.is_terminal()));
        if !terminal {
            continue;
        }
        match conn
            .execute(
                "DELETE FROM agents WHERE agent_id = ?1 AND job_id IS NULL",
                params![agent_id.clone()],
            )
            .await
        {
            Ok(_) => deleted += 1,
            Err(e) => {
                warn!(agent = %agent_id, error = %e, "Failed to delete terminal session pin");
            }
        }
    }
    if deleted > 0 {
        info!(deleted, "Removed session pins for terminal tickets");
    }
    deleted
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Insert a minimal ticket row so a [`SpawnChild::Phase`] job's `ticket_id`
    /// FK resolves (the phase job row links to tickets).
    async fn seed_ticket(conn: &crate::db::Connection, ticket_id: &str, ws_name: &str) {
        let now = crate::db::now();
        conn.execute(
            "INSERT INTO tickets (id, title, description, workspace_name, created_at, updated_at) \
             VALUES (?1, 'title', 'desc', ?2, ?3, ?3)",
            crate::db::params![ticket_id, ws_name, now],
        )
        .await
        .unwrap();
    }

    async fn status_retry(conn: &crate::db::Connection, job_id: &str) -> (String, i64) {
        let rows = conn
            .query(
                "SELECT status, retry_count FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 1, "job row exists for {job_id}");
        (
            rows[0].get::<String>(0).unwrap(),
            rows[0].get::<i64>(1).unwrap(),
        )
    }

    /// (f) Boot recovery resumes only async analyze/implement/research jobs.
    /// Sync caller-owned jobs (mode='sync' from a Some(caller) spawn) are
    /// skipped — their caller session re-drives them — so they stay `launched`
    /// with `retry_count` untouched; the async job is check-pointed
    /// (retry_count bumped, status re-armed to launched). The mode discriminator
    /// replaces the NULL-sentinel overload of `caller_agent_id`.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams across the whole test
    async fn boot_recovery_skips_caller_owned_jobs() {
        let _guard = crate::util::test::retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let ws_name = "boot_recovery_ws";
        // Binding 3: boot recovery skips (not terminalizes) a resume-eligible
        // job whose workspace is unresolvable — so the spawns need a registered
        // workspace for the async analyze to be check-pointed.
        crate::util::test::create_test_workspace("/tmp/boot_recovery_ws", ws_name).await;

        for (job_id, kind, child, caller) in [
            (
                "owned_analyze",
                AgentKind::Analyst,
                SpawnChild::Analyze,
                Some("pin_a"),
            ),
            (
                "owned_implement",
                AgentKind::Coder,
                SpawnChild::Implement,
                Some("pin_b"),
            ),
            (
                "null_analyze",
                AgentKind::Analyst,
                SpawnChild::Analyze,
                None,
            ),
        ] {
            spawn_job(
                conn,
                job_id,
                "task",
                ws_name,
                "caller",
                "telegram",
                crate::Role::Engineer,
                &[NewAgent {
                    agent_id: format!("{job_id}_agent"),
                    kind,
                    idx: Some(0),
                    task: "task".to_string(),
                }],
                &child,
                caller,
            )
            .await
            .unwrap();
        }

        // Spawn derives mode from the caller pin (sync ⇔ pin present).
        let modes: Vec<String> = conn
            .query(
                "SELECT mode FROM jobs WHERE id IN ('owned_analyze','owned_implement','null_analyze') ORDER BY id",
                (),
            )
            .await
            .unwrap()
            .iter()
            .map(|r| r.get::<String>(0).unwrap())
            .collect();
        assert_eq!(
            modes,
            ["async".to_string(), "sync".to_string(), "sync".to_string()],
            "spawn derives mode from the caller pin (sync ⇔ pin present)"
        );

        let resumable = recover_from_restart().await.unwrap();

        assert!(
            resumable.iter().any(
                |r| matches!(r, ResumableJob::Analyze { job_id, .. } if job_id == "null_analyze")
            ),
            "the async analyze is selected for boot resume"
        );
        assert!(
            !resumable.iter().any(
                |r| matches!(r, ResumableJob::Analyze { job_id, .. } if job_id == "owned_analyze")
            ),
            "the sync caller-owned analyze is not boot-resumed"
        );
        assert!(
            !resumable.iter().any(
                |r| matches!(r, ResumableJob::Implement { job_id, .. } if job_id == "owned_implement")
            ),
            "the sync caller-owned implement is not boot-resumed"
        );

        // Sync jobs stay launched with retry_count unchanged (checkpoint NOT
        // applied); the async job is check-pointed.
        for job_id in ["owned_analyze", "owned_implement"] {
            assert_eq!(
                status_retry(conn, job_id).await,
                ("launched".to_string(), 0),
                "{job_id} stays launched with retry_count unchanged"
            );
        }
        assert_eq!(
            status_retry(conn, "null_analyze").await,
            ("launched".to_string(), 1),
            "the async analyze is check-pointed (retry_count bumped)"
        );
    }

    /// Binding 3 on the boot scan: a `research_cleanup` row whose workspace is
    /// unresolvable (a bogus non-personal name with no `workspaces` row —
    /// `resolve_workspace` returns `Ok(None)`) is SKIPPED in place, not
    /// resumed: it stays `launched` with `retry_count` 0 (no checkpoint bump),
    /// and it is absent from the resumable list.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn boot_scan_leaves_unresolvable_workspace_cleanup_row_in_place() {
        let _guard = crate::util::test::retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let job_id = "cleanup_missing_ws_1";
        let ws_name = "ws_boot_scan_missing_1";
        // NOT personal-prefixed, no `workspaces` row — the binding-3 skip path.
        crate::util::test::JobRowBuilder::new(
            conn,
            job_id,
            "research_cleanup",
            "sanitation",
            ws_name,
        )
        .task("cleanup prompt")
        .timestamps(crate::db::now())
        .insert()
        .await
        .unwrap();

        let resumable = recover_from_restart().await.unwrap();

        // Filter by job id: other tests' rows may legitimately appear in the
        // shared resumable list.
        let hit = resumable
            .iter()
            .any(|r| matches!(r, ResumableJob::ResearchCleanup { job_id: j, .. } if j == job_id));
        assert!(
            !hit,
            "unresolvable-workspace cleanup row is not resumed (binding-3 skip)"
        );
        assert_eq!(
            status_retry(conn, job_id).await,
            ("launched".to_string(), 0),
            "binding-3 skip leaves the cleanup row untouched (no checkpoint bump)"
        );
    }

    /// The /clear abandon path deletes ONLY the cleared session's sync jobs —
    /// async and ticket-phase jobs for the same workspace survive, and the sync
    /// job's roster cascades away with it.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn clear_abandons_session_sync_jobs() {
        let _guard = crate::util::test::retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let ws_name = "clear_abandon_ws";
        seed_ticket(conn, "t_clear", ws_name).await;

        for (job_id, child, caller) in [
            ("clear_sync", SpawnChild::Analyze, Some("pin_clear")),
            ("clear_async", SpawnChild::Analyze, None),
            (
                "clear_phase",
                SpawnChild::Phase {
                    phase: crate::pipeline::board::TicketPhase::Analysis,
                    ticket_id: "t_clear".to_string(),
                },
                None,
            ),
        ] {
            spawn_job(
                conn,
                job_id,
                "task",
                ws_name,
                "caller",
                "telegram",
                crate::Role::Engineer,
                &[NewAgent {
                    agent_id: format!("{job_id}_agent"),
                    kind: AgentKind::Analyst,
                    idx: Some(0),
                    task: "task".to_string(),
                }],
                &child,
                caller,
            )
            .await
            .unwrap();
        }

        let abandoned = abandon_session_jobs("pin_clear").await.unwrap();
        assert_eq!(abandoned, 1, "only the caller-owned sync job is abandoned");

        let remaining: Vec<String> = conn
            .query(
                "SELECT id FROM jobs WHERE id IN ('clear_sync','clear_async','clear_phase') ORDER BY id",
                (),
            )
            .await
            .unwrap()
            .iter()
            .map(|r| r.get::<String>(0).unwrap())
            .collect();
        assert_eq!(
            remaining,
            ["clear_async", "clear_phase"],
            "async + phase jobs survive the session abandon"
        );

        let roster: Vec<String> = conn
            .query(
                "SELECT agent_id FROM agents WHERE job_id = 'clear_sync'",
                (),
            )
            .await
            .unwrap()
            .iter()
            .map(|r| r.get::<String>(0).unwrap())
            .collect();
        assert!(roster.is_empty(), "sync job roster cascaded away");
    }

    /// Workspace deletion abandons every job for the workspace uniformly: sync,
    /// phase, research, and research_cleanup rows all go, research run folders
    /// are released via the run cancel, and the rosters cascade away.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn workspace_delete_abandons_all_jobs() {
        let _guard = crate::util::test::retry_tests_lock();
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let ws_name = "ws_abandon_test";
        seed_ticket(conn, "t_abandon", ws_name).await;

        for (job_id, child, caller) in [
            ("abandon_sync", SpawnChild::Analyze, Some("pin_abandon")),
            (
                "abandon_phase",
                SpawnChild::Phase {
                    phase: crate::pipeline::board::TicketPhase::Analysis,
                    ticket_id: "t_abandon".to_string(),
                },
                None,
            ),
            ("abandon_research", SpawnChild::Research, None),
            ("abandon_cleanup", SpawnChild::ResearchCleanup, None),
        ] {
            spawn_job(
                conn,
                job_id,
                "task",
                ws_name,
                "caller",
                "telegram",
                crate::Role::Engineer,
                &[NewAgent {
                    agent_id: format!("{job_id}_agent"),
                    kind: AgentKind::Analyst,
                    idx: Some(0),
                    task: "task".to_string(),
                }],
                &child,
                caller,
            )
            .await
            .unwrap();
        }

        // Full integration: the WorkspaceStore built over the shared test conn
        // (production shares one connection across domain stores) runs the
        // research-run sweep fail-closed, then deletes the remaining job rows
        // atomically with the workspace row.
        let ws_store = crate::workspace::WorkspaceStore { conn: conn.clone() };
        ws_store.delete(ws_name).await.unwrap();

        let jobs = conn
            .query(
                "SELECT id FROM jobs WHERE workspace_name = ?1",
                crate::db::params![ws_name],
            )
            .await
            .unwrap();
        assert!(jobs.is_empty(), "every job row for the workspace is gone");

        let research = conn
            .query(
                "SELECT id FROM research_jobs WHERE id IN ('abandon_research','abandon_cleanup')",
                (),
            )
            .await
            .unwrap();
        assert!(
            research.is_empty(),
            "research_jobs child rows cascaded away"
        );

        let roster = conn
            .query(
                "SELECT agent_id FROM agents WHERE job_id IN ('abandon_sync','abandon_phase','abandon_research','abandon_cleanup')",
                (),
            )
            .await
            .unwrap();
        assert!(roster.is_empty(), "every seeded agent roster cascaded away");
    }
}
