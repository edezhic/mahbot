//! Durable jobs layer: jobs/agents/pending_jobs lifecycle, boot recovery
//! scan, and stale purge orchestration.
//!
//! All rows live in the consolidated domain database (`core.db`) behind the
//! one shared connection — the jobs/session/board tables now share a single
//! transaction domain, so cross-store ordering and crash-safety can be
//! expressed in a single transaction (see the purge section in the design).
//!
//! Table ownership: the DDL is appended to the session SCHEMA const
//! (see `session/mod.rs`); this module owns the row model, the lifecycle
//! helpers, the boot scan, and the purge orchestrator.

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
pub enum RowStatus {
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

/// Values of `agents.kind` — dispatch-slot kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentKind {
    Analyst,
    Verifier,
    Engineer,
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
            Self::Sanitation => "sanitation",
            Self::Diagnostics => "diagnostics",
        }
    }
}

/// Graceful-shutdown drain cap: in-flight work completes within this window,
/// then stragglers are force-cancelled.
pub const DRAIN_CAP_SECS: u64 = 10 * 60;

/// Stale-purge cutoff (hours): `jobs` rows older than this are purged
/// (stranded blocking-phase tickets rolled back in place). `pending_jobs`
/// envelopes are never purged — at-least-once keeps unconfirmed rows alive.
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
}

/// A row of the `agents` table.
#[derive(Debug, Clone)]
pub(crate) struct AgentRow {
    pub agent_id: String,
    pub idx: Option<i64>,
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

fn job_row_from(row: &Row) -> anyhow::Result<JobRow> {
    Ok(JobRow {
        id: row.get(0)?,
        kind: row.get(1)?,
        workspace_name: row.get(2)?,
        retry_count: row.get(3)?,
    })
}

fn agent_row_from(row: &Row) -> anyhow::Result<AgentRow> {
    Ok(AgentRow {
        agent_id: row.get(0)?,
        idx: row.get(1)?,
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

/// agents roster INSERT — producers: `spawn_job` roster loop and
/// `management::append_ticket_analysis_slots`. Status is the literal 'launched'.
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
    /// ticket_jobs (id, ticket_id) — a per-run `ticket_analysis` analysis batch
    /// over the shared substrate. Analysis spawns its full Analyst roster at
    /// creation; round-ness (base vs escalation) is derived at the
    /// dispatch/resume boundary, never stored on the job. Phase lives only on
    /// the ticket.
    TicketAnalysis { ticket_id: String },
    /// ticket_jobs (id, ticket_id) — a single-owner `ticket_implementation`
    /// pipeline job over the shared substrate. Created with an empty roster;
    /// each stage's agent registers a launched row during dispatch.
    TicketImplementation { ticket_id: String },
}

impl SpawnChild {
    /// The `jobs.kind` value — one kind per child row; Analyze is the exception
    /// (pure kind marker, no child row). Deriving kind here closes the drift
    /// window of an inconsistent (kind, child) pair.
    #[must_use]
    pub const fn kind_str(&self) -> &'static str {
        match self {
            Self::Analyze => "analyze",
            Self::Research => "research",
            Self::ResearchCleanup => "research_cleanup",
            Self::TempCleanup => "temp_cleanup",
            Self::TicketAnalysis { .. } => "ticket_analysis",
            Self::TicketImplementation { .. } => "ticket_implementation",
        }
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
) -> Result<()> {
    let kind = child.kind_str();
    let now = db::now();
    let tx = conn.begin_tx().await?;
    tx.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, user_name, channel, role, \
         retry_count, created_at, updated_at) \
         VALUES (?1, ?2, 'launched', ?3, ?4, ?5, ?6, ?7, 0, ?8, ?8)",
        params![
            id,
            kind,
            task,
            workspace_name,
            user_name,
            channel,
            role.as_str(),
            now.clone(),
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
        // Analyze / ResearchCleanup / TempCleanup are pure kind markers — the
        // job row alone drives resume (the cleanup prompt lives in jobs.task).
        SpawnChild::Analyze | SpawnChild::ResearchCleanup | SpawnChild::TempCleanup => {}
        SpawnChild::Research => {
            tx.execute(
                "INSERT INTO research_jobs (id, state) VALUES (?1, '{}')",
                params![id],
            )
            .await
            .with_context(|| format!("failed to insert research_jobs row for job {id}"))?;
        }
        SpawnChild::TicketAnalysis { ticket_id }
        | SpawnChild::TicketImplementation { ticket_id } => {
            tx.execute(
                "INSERT INTO ticket_jobs (id, ticket_id) \
                 VALUES (?1, ?2)",
                params![id, ticket_id.clone()],
            )
            .await
            .with_context(|| format!("failed to insert ticket_jobs row for job {id}"))?;
        }
    }
    tx.commit().await?;
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

/// Checkpoint a job row: bump retry_count and touch `updated_at`
/// (every jobs write sets updated_at = now — the 8h purge keys off it).
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

/// Terminate a ticket job (status='done'), clearing its agent roster. The
/// single unified completion for both `ticket_analysis` and
/// `ticket_implementation` ticket jobs over the shared `ticket_jobs`
/// substrate. Ordering contract: the board transition+comment runs FIRST; this
/// is the jobs/session completion tx.
pub(crate) async fn complete_ticket_job(conn: &Connection, job_id: &str) -> Result<()> {
    let tx = conn.begin_tx().await?;
    tx.execute(
        "UPDATE jobs SET status = 'done', updated_at = ?1 WHERE id = ?2",
        params![db::now(), job_id],
    )
    .await?;
    // Belt-and-braces explicit delete alongside FK CASCADE.
    tx.execute("DELETE FROM agents WHERE job_id = ?1", params![job_id])
        .await?;
    tx.commit().await?;
    Ok(())
}

/// Kind-neutral terminalization for jobs whose result cannot be delivered
/// (missing child row / unresolvable caller identity): DELETE the row
/// entirely — CASCADE removes roster + child rows. Envelope kinds
/// (analyze/research) are terminal only when the row is gone (the pending row IS
/// the durable record); no-ops when the row is already absent. Safe for any
/// jobs.kind — unlike [`complete_ticket_job`], which is the
/// ticket-specific form (status='done' keeps the row for the
/// phase-check/reset logic).
pub(crate) async fn terminalize_job(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute("DELETE FROM jobs WHERE id = ?1", params![job_id])
        .await
        .context("terminalize job")?;
    Ok(())
}

// ── Ticket-job substrate (ticket_jobs) ─────────────────────────────────
//
// The `ticket_jobs` child table is the single shared substrate for the two
// ticket-job kinds: `jobs.kind='ticket_analysis'` (per-run analysis batch) and
// `jobs.kind='ticket_implementation'` (single-owner pipeline job). Both read
// their phase from `tickets.phase`; the job side carries no phase mirror. The
// active agent(s) for a running job are held in the `agents` roster
// (status='launched'), which drives comment routing and the mid-execution
// re-dispatch guard.

/// A row of the `ticket_jobs` child table (id, ticket_id) — the shared
/// row model for both the per-run `ticket_analysis` analysis job and the
/// single-owner `ticket_implementation` pipeline job.
#[derive(Debug, Clone)]
pub(crate) struct TicketJobRow {
    pub id: String,
    pub ticket_id: String,
    pub paused_frozen: bool,
}

fn ticket_job_row_from(row: &Row) -> anyhow::Result<TicketJobRow> {
    Ok(TicketJobRow {
        id: row.get(0)?,
        ticket_id: row.get(1)?,
        paused_frozen: row.get::<bool>(2)?,
    })
}

/// Load the implementation job for a ticket, if one exists.
pub(crate) async fn find_implementation_job(
    conn: &Connection,
    ticket_id: &str,
) -> Result<Option<TicketJobRow>> {
    conn.query_optional(
        "SELECT sj.id, sj.ticket_id, j.paused_frozen \
         FROM ticket_jobs sj \
         JOIN jobs j ON j.id = sj.id \
         WHERE sj.ticket_id = ?1 AND j.kind = 'ticket_implementation'",
        params![ticket_id],
        ticket_job_row_from,
    )
    .await
    .context("find ticket implementation")
}

/// Load every implementation job for a workspace (jobs kind='ticket_implementation').
pub(crate) async fn list_workspace_implementation_jobs(
    conn: &Connection,
    ws_name: &str,
) -> Result<Vec<TicketJobRow>> {
    let rows = conn
        .query(
            "SELECT sj.id, sj.ticket_id, j.paused_frozen \
             FROM ticket_jobs sj \
             JOIN jobs j ON j.id = sj.id \
             WHERE j.kind = 'ticket_implementation' AND j.workspace_name = ?1",
            params![ws_name],
        )
        .await
        .context("list workspace implementations")?;
    rows.iter().map(ticket_job_row_from).collect()
}

/// Load every analysis job for a workspace (jobs kind='ticket_analysis').
pub(crate) async fn list_workspace_analysis_jobs(
    conn: &Connection,
    ws_name: &str,
) -> Result<Vec<TicketJobRow>> {
    let rows = conn
        .query(
            "SELECT sj.id, sj.ticket_id, j.paused_frozen \
             FROM ticket_jobs sj \
             JOIN jobs j ON j.id = sj.id \
             WHERE j.kind = 'ticket_analysis' AND j.workspace_name = ?1",
            params![ws_name],
        )
        .await
        .context("list workspace analysis jobs")?;
    rows.iter().map(ticket_job_row_from).collect()
}

/// Update the implementation job's stored `task` so boot resume re-dispatches the
/// current stage with the right prompt (the implementation job is the resume handle).
pub(crate) async fn update_implementation_job_task(
    conn: &Connection,
    job_id: &str,
    task: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET task = ?2, updated_at = ?3 WHERE id = ?1",
        params![job_id, task, db::now()],
    )
    .await
    .context("update ticket implementation task")?;
    Ok(())
}

/// Terminalize a ticket's implementation (status='done'), idempotently finding the
/// implementation by ticket id first. No-op when the ticket has no implementation row. Used
/// on the normal Done transition path so implementation rows do not linger as
/// status='launched' until the 8h stale purge or the next boot recovery.
pub(crate) async fn complete_implementation_job_for_ticket(
    conn: &Connection,
    ticket_id: &str,
) -> Result<()> {
    let Some(implementation) = find_implementation_job(conn, ticket_id).await? else {
        return Ok(());
    };
    complete_ticket_job(conn, &implementation.id).await
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
    task: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
         VALUES (?1, ?2, ?3, NULL, ?4, ?5) \
         ON CONFLICT(job_id, agent_id) DO UPDATE SET status = ?4",
        params![job_id, agent_id, kind.as_str(), status.as_str(), task],
    )
    .await
    .with_context(|| format!("failed to upsert job agent {agent_id} for job {job_id}"))?;
    Ok(())
}

/// List the currently-running agent ids for a ticket (`status='launched'`
/// roster rows across the ticket's implementation and analysis jobs). Drives comment
/// routing to running agents.
pub(crate) async fn list_running_agents_for_ticket(
    conn: &Connection,
    ticket_id: &str,
) -> Result<Vec<String>> {
    let rows = conn
        .query(
            "SELECT a.agent_id FROM agents a \
             JOIN ticket_jobs ts ON ts.id = a.job_id \
             WHERE ts.ticket_id = ?1 AND a.status = 'launched'",
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

/// True when a ticket has any currently-running agent (`status='launched'`
/// roster row for the ticket's implementation/analysis jobs). Drives the re-dispatch
/// guard.
pub(crate) async fn ticket_has_active_agents(conn: &Connection, ticket_id: &str) -> Result<bool> {
    Ok(!list_running_agents_for_ticket(conn, ticket_id)
        .await?
        .is_empty())
}

/// True when a job has any currently-running agent (`status='launched'` roster
/// row). Drives the implementation-iteration re-dispatch guard.
pub(crate) async fn job_has_launched_agents(conn: &Connection, job_id: &str) -> Result<bool> {
    Ok(conn
        .query_optional(
            "SELECT 1 FROM agents WHERE job_id = ?1 AND status = 'launched' LIMIT 1",
            params![job_id],
            |_| Ok::<_, anyhow::Error>(1_i64),
        )
        .await?
        .is_some())
}

/// Clear a job's stale 'launched' roster rows after a crash (the previous
/// process's agents are not running). The job row survives; the roster is
/// rebuilt on resume dispatch. Used at boot for implementation jobs so a frozen
/// implementation's stale mid-execution marker does not block the poll's re-dispatch
/// gate on a later unpause.
pub(crate) async fn clear_launched_agents_for_job(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM agents WHERE job_id = ?1 AND status = 'launched'",
        params![job_id],
    )
    .await
    .context("clear launched agents for job")?;
    Ok(())
}

/// Mark a `ticket_analysis` job as frozen by a workspace pause. The freeze
/// marker is what allows [`crate::pipeline::management::re_drive_analysis_rounds`]
/// to tell a genuinely frozen round from a normally-finalizing one (which is
/// never marked). Idempotent; a no-op for non-analysis kinds.
pub(crate) async fn mark_analysis_frozen(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET paused_frozen = 1 WHERE id = ?1 AND kind = 'ticket_analysis'",
        params![job_id],
    )
    .await
    .context("mark analysis job frozen")?;
    Ok(())
}

/// Atomically claim a frozen `ticket_analysis` round for resume: clears the
/// freeze marker only while it is still set. Returns `true` when THIS caller
/// won the claim (the round was frozen) and must re-drive it; `false` when it
/// was already clear (resumed by another path, or never frozen). The
/// conditional UPDATE closes the double-resume race between the poll re-drive
/// and the boot-resume path.
pub(crate) async fn claim_analysis_resume(conn: &Connection, job_id: &str) -> Result<bool> {
    let n = conn
        .execute(
            "UPDATE jobs SET paused_frozen = 0 WHERE id = ?1 AND paused_frozen = 1",
            params![job_id],
        )
        .await
        .context("claim analysis resume")?;
    Ok(n > 0)
}

/// Delete the jobs row (CASCADE destroys ticket_jobs/research child rows)
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
/// then job_caller, then terminalize-on-missing — the guard fires on Err (DB
/// read failure) too, not just Ok(None). Returns the caller + parsed role, or
/// None when the caller should quiet-return. `abort_site`/`missing_site` keep
/// each site's log texts byte-identical (warn-vs-info distinction preserved).
pub(crate) async fn resume_job_preamble(
    conn: &Connection,
    job_id: &str,
    abort_site: &str,
    missing_site: &str,
) -> Option<(JobCaller, Role)> {
    if crate::shutdown::aborting() {
        tracing::info!(job = %job_id, "{abort_site} aborted — drain/shutdown in progress");
        return None;
    }
    let Ok(Some(caller)) = job_caller(conn, job_id).await else {
        tracing::warn!(job = %job_id, "{missing_site}: missing job row — terminalizing job");
        let _ = terminalize_job(conn, job_id).await;
        return None;
    };
    let caller_role = std::str::FromStr::from_str(&caller.role).unwrap_or(Role::Manager);
    Some((caller, caller_role))
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
            "SELECT id, kind, workspace_name, retry_count FROM jobs \
             WHERE status != 'done' ORDER BY created_at",
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
            "SELECT agent_id, idx, status, outcome, task \
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
/// last agent deregisters / guard releases) and the registration start of
/// ticket_analysis joint-comment synthesis can both see both registries empty
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
    use crate::shutdown::{aborting, force_cancel, shutdown, shutdown_token};
    use std::time::Instant;

    // Wait for the drain to begin (or an already-fired token).
    loop {
        if aborting() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
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
/// The ticket-job kinds — `jobs.kind='ticket_analysis'` (per-run analysis) and
/// `jobs.kind='ticket_implementation'` (single-owner pipeline implementation) —
/// are modeled as distinct resume variants. The boot-scan loop selects the
/// variant from `jobs.kind`, and runtime dispatch derives the stage from the
/// ticket's phase.
pub(crate) enum ResumableJob {
    /// A per-run `ticket_analysis` analysis job interrupted by a crash.
    /// Resumed in place: the ticket is still in `Analysis`.
    TicketAnalysis {
        job_id: String,
        ticket_id: String,
        workspace_name: String,
    },
    /// A single-owner `ticket_implementation` pipeline job interrupted by a
    /// crash. Resumed in place: the ticket is still in a pipeline-occupied
    /// phase.
    TicketImplementation {
        job_id: String,
        ticket_id: String,
        workspace_name: String,
    },
    Research {
        job_id: String,
        workspace_name: String,
    },
    Analyze {
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

/// Stale-job purge with in-place ticket rollback.
///
/// Cutoff = 8h. Deletes jobs whose updated_at predates the cutoff AND whose
/// roster agents' sessions are all stale too (live sessions referenced by
/// unfinished jobs are NEVER purged — the agents table IS the marker).
/// `ticket_analysis` jobs stranded in a blocking phase are rolled back IN PLACE
/// at purge time — two sequential transactions on the single consolidated
/// domain connection (ticket phase rollback, then jobs delete), no cross-file
/// tx (board/sessions now share one file).
///
/// Crash-safe ordering:
/// 1. SELECT the purge set (ticket_id → literal 'analysis' phase) BEFORE deleting —
///    CASCADE destroys ticket_jobs rows at delete time.
/// 2. ONE transaction (the shared consolidated connection) that rolls back each
///    stale analysis ticket's phase with a per-ticket CAS (phase CAS as the
///    last-line race guard; the rollback is unconditional per stale analysis
///    row — one-round analysis has no round guard).
/// 3. A SECOND transaction on the same connection: DELETE FROM jobs.
///
/// Crash after (2) → CAS fails next tick, boot phase-check deletes the stale
/// row; crash after (1) → nothing changed. Jobs-delete-first would strand a
/// ticket in a blocking phase with no job row — strictly worse.
///
/// A boot does NOT rescue a stranded ticket via resume: `recover_from_restart`
/// bumps every resumed row's updated_at (checkpoint_job), which makes the row
/// purge-immune on later ticks, and a purge-deleted row has nothing to resume
/// at all — recovery for the stranded ticket is specifically the next-boot
/// reset (`reset_analysis_tickets`). The in-place rollback here is the only
/// runtime rescue.
pub async fn purge_stale_jobs(cutoff: &str) -> Result<u64> {
    let conn = &crate::session::store().conn;

    // (1) SELECT the purge set.
    //
    // A paused (frozen) implementation job is purge-immune: it holds the workspace's
    // running slot while frozen, and must survive an arbitrarily long pause.
    // The freeze authority is the workspace `paused` flag (the implementation no
    // longer mirrors a pause), so paused-workspace implementation rows are excluded
    // below. Non-ticket jobs (researches, temp cleanups) have no
    // `ticket_jobs` child row and are handled by their own lifecycle.
    let rows = conn
        .query(
            "SELECT j.id, j.kind, j.workspace_name, ts.ticket_id \
             FROM jobs j \
             LEFT JOIN ticket_jobs ts ON ts.id = j.id \
             WHERE j.updated_at < ?1 \
               AND NOT EXISTS ( \
                 SELECT 1 FROM agents a \
                 JOIN session_metadata sm ON sm.agent_id = a.agent_id \
                 WHERE a.job_id = j.id AND sm.last_activity >= ?1)",
            params![cutoff],
        )
        .await
        .context("select stale jobs for purge")?;
    if rows.is_empty() {
        return Ok(0);
    }
    // Paused workspace names: a frozen implementation must survive the stale purge.
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
    // ticket_analysis job ids whose board rollback must land before the
    // sessions DELETE (a failed rollback keeps those rows for the next tick's
    // retry).
    let mut ticket_analysis_ids: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(rows.len());
    // (ticket_id, phase) pairs for the board rollback. One-round analysis has
    // no round guard — every stale analysis row rolls its ticket back
    // unconditionally (the board rollback reads the stored jobs rows from
    // here rather than opening a board-side view of ticket_jobs, and the
    // phase is the literal 'analysis' — the ticket is reset to Backlog).
    let mut rollbacks: Vec<(String, String)> = Vec::new();
    for row in &rows {
        let id: String = row.get(0)?;
        let kind: String = row.get(1)?;
        let workspace_name: String = row.get(2)?;
        // A paused workspace's implementation is frozen — purge-immune.
        if kind == "ticket_implementation" && paused_ws.contains(&workspace_name) {
            continue;
        }
        purge_ids.push(id.clone());
        if kind == "ticket_analysis" {
            ticket_analysis_ids.insert(id);
            let ticket_id: Option<String> = row.get(3).ok();
            if let Some(t) = ticket_id {
                rollbacks.push((t, "analysis".to_string()));
            }
        }
    }

    // (2) ONE transaction (shared consolidated connection) that rolls back each
    // stale analysis ticket's phase with a per-ticket CAS. A missing board
    // (uninitialized) or a failed tx means the rollback did NOT land — keep
    // the ticket_analysis rows so the next tick retries the CAS (aligned with
    // the tx-failure path; deleting the job would strand the ticket in a
    // blocking phase with no runtime recovery until the next boot reset).
    let mut rollback_ok = true;
    if !rollbacks.is_empty() {
        if crate::pipeline::board::BOARD.get().is_some() {
            rollback_ok = rollback_stranded_tickets(&rollbacks).await;
        } else {
            rollback_ok = false;
        }
    }

    // (3) A SECOND transaction on the same consolidated connection:
    // DELETE FROM jobs (CASCADE removes roster +
    // child rows; the engineer anchor survives — NULL FK child). A failed
    // board rollback keeps ticket_analysis rows in place so the next purge tick
    // retries the CAS — deleting the job first would strand the ticket in a
    // blocking phase with no runtime recovery until the next boot reset.
    //
    // Structural safety for research_cleanup rows (kind-agnostic DELETE here):
    // a research_cleanup row can only reach this purge with its run folder
    // ALREADY released — `recover_from_restart` handles every research_cleanup
    // row synchronously at boot, before this periodic loop's first tick, by
    // resuming it (the cleanup tail releases the folder before it
    // terminalizes the row, so the row never ages into the purge). No folder
    // is orphaned by the purge.
    let tx = conn.begin_tx().await?;
    let mut deleted = 0usize;
    for id in &purge_ids {
        if !rollback_ok && ticket_analysis_ids.contains(id) {
            continue;
        }
        delete_job_tx(&tx, id).await?;
        deleted += 1;
    }
    tx.commit().await?;
    if deleted > 0 {
        tracing::debug!(deleted, "Purged stale jobs");
    }
    Ok(deleted as u64)
}

/// Only the Analysis → Backlog reset transition applies to ticket_analysis job
/// rollback (the implementation-protected phases no longer produce ticket_analysis
/// rows). Transitory handoffs need no rollback — the phase CAS no-ops.
fn rollback_transition(phase: &str) -> Option<String> {
    let from = phase.parse::<crate::pipeline::board::TicketPhase>().ok()?;
    crate::pipeline::board::BoardStore::reset_analysis_transition(from).map(|to| to.to_string())
}

/// Roll back stranded tickets in place: phase + updated_at. Guarded by phase
/// CAS (the ticket's current phase must equal the job's dispatched-for phase).
/// Returns whether the board tx committed (false → purge keeps ticket_analysis
/// job rows so the next tick retries the CAS).
async fn rollback_stranded_tickets(rollbacks: &[(String, String)]) -> bool {
    let board = crate::pipeline::board::BOARD
        .get()
        .expect("BOARD initialized");
    let tx = match board.conn.begin_tx().await {
        Ok(tx) => tx,
        Err(e) => {
            warn!(error = %e, "Purge rollback: failed to begin board tx");
            return false;
        }
    };
    let now = crate::db::now();
    let mut rolled_back = 0usize;
    for (ticket_id, phase) in rollbacks {
        let Some(to) = rollback_transition(phase) else {
            continue;
        };
        // Phase CAS: the ticket's current phase must equal the dispatched-for
        // phase — a moved ticket is not rolled back.
        let sql = format!(
            "UPDATE tickets SET {} WHERE id = ?3 AND phase = ?4",
            crate::pipeline::board::BoardStore::RESET_TICKET_SET_CLAUSE
        );
        let updated = tx
            .execute(
                &sql,
                crate::db::params![to, now.clone(), ticket_id.clone(), phase.clone(),],
            )
            .await;
        match updated {
            Ok(n) if n > 0 => rolled_back += 1,
            Ok(_) => {
                debug!(ticket = %ticket_id, "Purge rollback CAS no-op (phase moved)");
            }
            Err(e) => {
                warn!(ticket = %ticket_id, error = %e, "Purge rollback failed");
            }
        }
    }
    match tx.commit().await {
        Ok(()) => {
            if rolled_back > 0 {
                info!(rolled_back, "Purge: rolled back stranded tickets in place");
            }
            true
        }
        Err(e) => {
            warn!(error = %e, "Purge rollback commit failed");
            false
        }
    }
}

/// Boot recovery scan: first statement of run_management, before
/// reset_analysis_tickets. Order: (0) replay pending_jobs; (1) ticket_analysis
/// scan → resumed-ticket exclusion set; (2) reworked reset_analysis_tickets
/// with NOT IN exclusion; (3) one combined loop over the remaining kinds
/// (research/analyze resumed, research_cleanup resumed, temp_cleanup
/// terminalized).
///
/// Every resumed job gets updated_at = now (the boot bump — the ONLY
/// protection for the pre-first-commit window; Session::init with an empty
/// message skips the last_activity upsert) and retry_count + 1 (every bump
/// here is one boot resume; in-job retries are not counted).
#[expect(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub(crate) async fn recover_from_restart() -> Result<Vec<ResumableJob>> {
    let start = std::time::Instant::now();
    let conn = &crate::session::store().conn;

    // (0) Deliverable replay — before anything that could purge.
    let replayed = replay_pending_jobs(conn).await?;

    // (1) One kind-discriminated scan over the active jobs → exclusion set,
    // phase-left completions, and the resume list. Ticket jobs
    // (`ticket_analysis`/`ticket_implementation`) share the single
    // `ticket_jobs` substrate and the same phase-check/resume-or-complete
    // dance; the
    // research/analyze/research_cleanup/temp_cleanup kinds resume or
    // terminalize in line. The kind only selects the `ResumableJob` variant.
    let jobs = list_active_jobs(conn).await?;
    let mut exclusion: Vec<String> = Vec::new();
    let mut to_complete: Vec<String> = Vec::new();
    let mut resumable: Vec<ResumableJob> = Vec::new();
    let mut resumed_other = 0usize;
    for job in &jobs {
        match job.kind.as_str() {
            "ticket_analysis" | "ticket_implementation" => {
                let is_implementation = job.kind == "ticket_implementation";
                // Phase-check against the ticket: in expected phase → resume
                // (add to exclusion); moved/cancelled/done → stale → mark done
                // at boot. The expected phase derives from `tickets.phase` —
                // implementation resumes iff the phase is pipeline-occupied,
                // analysis iff it is `Analysis`.
                let rows = conn
                    .query_map_strict(
                        "SELECT tj.id, tj.ticket_id, j.paused_frozen \
                         FROM ticket_jobs tj \
                         JOIN jobs j ON j.id = tj.id \
                         WHERE tj.id = ?1",
                        params![job.id.clone()],
                        ticket_job_row_from,
                    )
                    .await
                    .unwrap_or_default();
                let Some(row) = rows.into_iter().next() else {
                    // child row missing (e.g. partially-spawned) — leave for
                    // purge; the ticket resets normally.
                    continue;
                };
                // Every launched ticket job is resumable at boot (no re-dispatch
                // cap — the job never gets marked done here; retry_count still
                // feeds escalation). Bump updated_at (the boot bump) AND
                // increment retry_count (every bump here is one boot resume).
                let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
                let phase = crate::pipeline::board::store()
                    .get_ticket_phase(&row.ticket_id)
                    .await
                    .ok()
                    .flatten();
                let in_phase = if is_implementation {
                    phase.is_some_and(|p| p.is_pipeline_occupied())
                } else {
                    phase == Some(crate::pipeline::board::TicketPhase::Analysis)
                };
                if in_phase {
                    if is_implementation {
                        // A implementation interrupted by a crash: any 'launched' roster
                        // rows from the previous process are stale and must be
                        // cleared (a frozen implementation left behind a mid-execution
                        // marker that would otherwise block the poll's
                        // re-dispatch gate on a later unpause).
                        if let Err(e) = clear_launched_agents_for_job(conn, &job.id).await {
                            warn!(
                                job = %job.id,
                                ticket = %row.ticket_id,
                                error = %e,
                                "Failed to clear stale launched agents on implementation boot resume",
                            );
                        }
                    }
                    exclusion.push(row.ticket_id.clone());
                    if is_implementation {
                        resumable.push(ResumableJob::TicketImplementation {
                            job_id: job.id.clone(),
                            ticket_id: row.ticket_id,
                            workspace_name: job.workspace_name.clone(),
                        });
                    } else {
                        resumable.push(ResumableJob::TicketAnalysis {
                            job_id: job.id.clone(),
                            ticket_id: row.ticket_id,
                            workspace_name: job.workspace_name.clone(),
                        });
                    }
                } else {
                    to_complete.push(job.id.clone());
                }
            }
            "research" | "analyze" => {
                // Resume at the roster/state level (dispatch re-enters the
                // orchestrator with the stored task). Always bump retry_count.
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
            }
            "research_cleanup" => {
                // A research-run cleanup Sanitation agent interrupted by a
                // crash. Resume it like any other durable job. Any path that
                // removes a research_cleanup row must release the run folder in
                // the same operation.
                let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
                resumable.push(ResumableJob::ResearchCleanup {
                    job_id: job.id.clone(),
                    workspace_name: job.workspace_name.clone(),
                });
                resumed_other += 1;
            }
            "temp_cleanup" => {
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
            }
            _ => {
                warn!(job = %job.id, kind = %job.kind, "Unknown job kind — skipping");
            }
        }
    }

    // (2) reset_analysis_tickets with the exclusion (empty exclusion omits
    // the NOT IN clause).
    if let Some(board) = crate::pipeline::board::BOARD.get()
        && let Err(e) = board.reset_analysis_tickets(&exclusion).await
    {
        warn!(error = %e, "Failed to reset in-flight tickets");
    }

    // Mark phase-left jobs done (one UPDATE each — closes the "left launched
    // until purge" gap; reset handles the ticket normally).
    for id in &to_complete {
        if let Err(e) = complete_ticket_job(conn, id).await {
            warn!(job = %id, error = %e, "Failed to mark stale implementation/stage job done");
        }
    }

    let elapsed = start.elapsed();
    info!(
        duration_ms = elapsed.as_millis() as u64,
        resumed_tickets = resumable.len() - resumed_other,
        resumed_other,
        replayed_pending = replayed,
        "Boot recovery scan complete",
    );

    // Boot sweep for engineer anchors (S5 terminal deletion): anchors for
    // tickets already in a terminal phase are removed so the TTL guard stops
    // protecting their accumulated sessions (the 5-min archive loop re-runs
    // this — boot just catches up after downtime).
    let _ = purge_terminal_engineer_session_pins().await;

    Ok(resumable)
}

// ── Engineer anchor (S5 — permanently-NULL seat) ────────────────────────

/// Deterministic engineer anchor agent ID: the per-ticket accumulated
/// session `ticket_{ticket_id}_engineer`, permanently job_id = NULL.
#[must_use]
pub fn engineer_session_pin_id(ticket_id: &str) -> String {
    crate::session::ticket_agent_id(ticket_id, crate::Role::Engineer.as_str())
}

/// Upsert the engineer anchor. NEVER sets job_id (setting it removes the row
/// from the partial-index scope → later NULL insert no longer conflicts →
/// duplicate anchor). The DDL and this UPSERT use the IDENTICAL syntactic
/// WHERE form (`job_id IS NULL` on both sides).
pub(crate) async fn upsert_engineer_session_pin(
    conn: &Connection,
    ticket_id: &str,
    task: &str,
    status: RowStatus,
) -> Result<()> {
    let anchor_id = engineer_session_pin_id(ticket_id);
    conn.execute(
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, outcome, task) \
         VALUES (NULL, ?1, 'engineer', NULL, ?2, NULL, ?3) \
         ON CONFLICT(agent_id) WHERE job_id IS NULL \
         DO UPDATE SET status = ?2, task = ?3, outcome = NULL",
        params![anchor_id, status.as_str(), task],
    )
    .await
    .with_context(|| format!("failed to upsert engineer anchor for ticket {ticket_id}"))?;
    Ok(())
}

/// Parse `ticket_{ticket_id}_engineer` — the ticket id is everything before
/// the final `_engineer` (workspace names allow underscores, so ticket ids
/// like `my_ws-42` are valid — only the final suffix is stripped).
#[must_use]
pub fn engineer_session_pin_ticket_id(agent_id: &str) -> Option<String> {
    agent_id
        .strip_prefix("ticket_")
        .and_then(|rest| rest.strip_suffix("_engineer"))
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

/// Terminal deletion of engineer session-pins (S5): pins are permanently-NULL
/// seats protecting the accumulated engineer session across bounce rounds.
/// Once the ticket reaches a terminal phase (Done/Cancelled/Failed — no
/// future bounces), the pin is removed so the TTL guard stops protecting
/// the session (cleaned ≤8h later). Centralized here (5-min archive loop +
/// boot sweep) because the terminal paths are scattered (engineer failure,
/// verifier all-failed, sanitation failure, dispatch panic, supersede, user
/// cancel). Idempotent: parse ticket_id from the pin agent_id,
/// phase-check on the board side (shared consolidated connection), delete.
pub(crate) async fn purge_terminal_engineer_session_pins() -> usize {
    let conn = &crate::session::store().conn;
    let Ok(rows) = conn
        .query(
            "SELECT agent_id FROM agents WHERE job_id IS NULL AND kind = 'engineer'",
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
        let Some(ticket_id) = engineer_session_pin_ticket_id(&agent_id) else {
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
                warn!(agent = %agent_id, error = %e, "Failed to delete terminal engineer anchor");
            }
        }
    }
    if deleted > 0 {
        info!(deleted, "Removed engineer anchors for terminal tickets");
    }
    deleted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::params;

    // ── S5 anchor spike: turso 0.7.2 must accept the
    // partial-index UPSERT with identical syntactic WHERE on DDL + UPSERT.
    #[tokio::test]
    async fn anchor_upsert_partial_index_semantics() {
        let (store, _tmp) = crate::open_test_store!(crate::session::SessionStore, "session");
        let conn = &store.conn;

        let anchor_id = engineer_session_pin_id("t-1400");

        // First dispatch: anchor insert (NULL job_id).
        upsert_engineer_session_pin(conn, "t-1400", "task-1", RowStatus::Launched)
            .await
            .expect("anchor upsert (first)");
        let rows = conn
            .query(
                "SELECT agent_id, job_id, status, task FROM agents WHERE agent_id = ?1",
                params![anchor_id.clone()],
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 1, "exactly one anchor row after first upsert");
        assert!(rows[0].get::<Option<String>>(1).unwrap().is_none());
        assert_eq!(rows[0].get::<String>(2).unwrap(), "launched");
        assert_eq!(rows[0].get::<String>(3).unwrap(), "task-1");

        // Re-dispatch: UPSERT must mutate status/task, keep job_id NULL, keep
        // exactly one row.
        upsert_engineer_session_pin(conn, "t-1400", "task-2", RowStatus::Launched)
            .await
            .expect("anchor upsert (second)");
        let rows = conn
            .query(
                "SELECT agent_id, job_id, status, task FROM agents WHERE agent_id = ?1",
                params![anchor_id.clone()],
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 1, "anchor must stay a single row per agent_id");
        assert!(rows[0].get::<Option<String>>(1).unwrap().is_none());
        assert_eq!(rows[0].get::<String>(3).unwrap(), "task-2");

        // A roster row (non-NULL job_id) for the SAME agent_id must coexist
        // under the composite PK (job row first — FK constraint).
        crate::util::test::JobRowBuilder::new(conn, "j1", "ticket_analysis", "engineer", "ws")
            .timestamps(db::now())
            .insert()
            .await
            .expect("insert job for roster row");
        conn.execute(
            "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
             VALUES ('j1', ?1, 'engineer', NULL, 'launched', 'roster-task')",
            params![anchor_id.clone()],
        )
        .await
        .expect("roster row with same agent_id must coexist");
        let rows = conn
            .query(
                "SELECT job_id, task FROM agents WHERE agent_id = ?1 ORDER BY job_id IS NULL",
                params![anchor_id],
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 2, "anchor + roster row coexist");

        // A second anchor insert (NULL job_id) must conflict → single row.
        let err = conn
            .execute(
                "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
                 VALUES (NULL, ?1, 'engineer', NULL, 'launched', 'dup')",
                params![engineer_session_pin_id("t-1400")],
            )
            .await
            .expect_err("duplicate NULL-seat anchor must violate the partial unique index");
        debug!("anchor duplicate conflict: {err:?}");

        // FK CASCADE: deleting the roster's job removes only the roster row.
        conn.execute("DELETE FROM jobs WHERE id = 'j1'", ())
            .await
            .unwrap();
        let rows = conn
            .query(
                "SELECT job_id FROM agents WHERE agent_id = ?1",
                params![engineer_session_pin_id("t-1400")],
            )
            .await
            .unwrap();
        assert_eq!(
            rows.len(),
            1,
            "anchor survives job deletion (NULL FK child)"
        );
        assert!(rows[0].get::<Option<String>>(0).unwrap().is_none());
    }

    // ── Completion-tx atomicity ──────────────────────────────────────

    /// The completion tx is the exactly-once boundary: INSERT pending_jobs
    /// (envelope, id = job id) + DELETE jobs (CASCADE removes roster rows) in
    /// ONE transaction. A crash mid-tx leaves either both or neither.
    #[tokio::test]
    async fn complete_job_with_envelope_atomicity() {
        let (store, _tmp) = crate::open_test_store!(crate::session::SessionStore, "session");
        let conn = &store.conn;
        spawn_job(
            conn,
            "j1",
            "q",
            "ws",
            "",
            "",
            crate::Role::Manager,
            &[NewAgent {
                agent_id: "analyze_a1".to_string(),
                kind: AgentKind::Analyst,
                idx: Some(0),
                task: "t1".to_string(),
            }],
            &SpawnChild::Analyze,
        )
        .await
        .unwrap();
        let envelope = AgentJob {
            content: "<analyze-tool-result>\n\nok</analyze-tool-result>".to_string(),
            workspace_name: "ws".to_string(),
            user_name: String::new(),
            channel: String::new(),
            kind: MessageKind::AnalyzeToolResult,
            role: crate::Role::Manager,
            reply_target: None,
            pending_job_id: Some("j1".to_string()),
        };
        complete_job_with_envelope(conn, "j1", &envelope)
            .await
            .unwrap();
        // Pending row exists with the envelope; jobs row gone; roster cascaded.
        let pending = list_pending_jobs(conn).await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, "j1");
        let jobs = conn.query("SELECT COUNT(*) FROM jobs", ()).await.unwrap();
        assert_eq!(jobs[0].get::<i64>(0).unwrap(), 0);
        let agents = conn.query("SELECT COUNT(*) FROM agents", ()).await.unwrap();
        assert_eq!(agents[0].get::<i64>(0).unwrap(), 0);
    }

    // ── Resume preamble ──────────────────────────────────────────────

    /// The shared boot-resume preamble's two quiet-return paths: drain-abort
    /// must leave the job row untouched (it is the durable resume state for
    /// the next boot), and terminalize-on-missing (row gone — the let-else
    /// guard fires on Err of job_caller the same way) must quiet-return
    /// without resurrecting anything.
    #[tokio::test]
    #[expect(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
    async fn resume_job_preamble_abort_and_missing_quiet_return() {
        let _lock = crate::util::test::retry_tests_lock();
        let (store, _tmp) = crate::open_test_store!(crate::session::SessionStore, "session");
        let conn = &store.conn;
        spawn_job(
            conn,
            "j-preamble",
            "question?",
            "ws",
            "caller-user",
            "telegram",
            Role::Manager,
            &[],
            &SpawnChild::Analyze,
        )
        .await
        .unwrap();

        // Drain-abort: quiet return, the row stays for the next boot.
        crate::shutdown::drain_begin();
        let r = resume_job_preamble(conn, "j-preamble", "Analyze resume", "Analyze resume").await;
        crate::shutdown::drain_clear();
        assert!(r.is_none(), "drain-abort must quiet-return");
        let rows = conn
            .query("SELECT COUNT(*) FROM jobs WHERE id = 'j-preamble'", ())
            .await
            .unwrap();
        assert_eq!(
            rows[0].get::<i64>(0).unwrap(),
            1,
            "drain-abort must not terminalize the job"
        );

        // Terminalize-on-missing: the row is gone → job_caller Ok(None) →
        // terminalize (no-op DELETE) + quiet return; nothing is resurrected.
        conn.execute("DELETE FROM jobs WHERE id = 'j-preamble'", ())
            .await
            .unwrap();
        let r = resume_job_preamble(conn, "j-preamble", "Analyze resume", "Analyze resume").await;
        assert!(r.is_none(), "missing job row must quiet-return");
        let rows = conn
            .query("SELECT COUNT(*) FROM jobs WHERE id = 'j-preamble'", ())
            .await
            .unwrap();
        assert_eq!(
            rows[0].get::<i64>(0).unwrap(),
            0,
            "terminalize must not recreate the row"
        );
    }

    // ── Pending replay dedup ─────────────────────────────────────────

    /// Boot replay skips a pending row whose envelope was already appended to
    /// the target session (suffix match + created_at tiebreaker) and reclaims
    /// the row.
    #[tokio::test]
    async fn pending_replay_dedup_skips_appended() {
        let (store, _tmp) = crate::open_test_store!(crate::session::SessionStore, "session");
        let conn = &store.conn;
        let agent_id = "manager_ws";
        let content = "hello manager";
        // Persist the message exactly as Session::init would (suffix timestamp).
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) \
             VALUES (?1, 'user', ?2, ?3)",
            params![
                agent_id,
                format!("{content}\n\n<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>"),
                db::now()
            ],
        )
        .await
        .unwrap();
        // Pending row created BEFORE the message (tiebreaker must skip).
        conn.execute(
            "INSERT INTO pending_jobs (id, target_agent_id, envelope, created_at) \
             VALUES ('p1', ?1, ?2, ?3)",
            params![
                agent_id,
                serde_json::to_string(&AgentJob {
                    content: content.to_string(),
                    workspace_name: "ws".to_string(),
                    user_name: "u".to_string(),
                    channel: "telegram".to_string(),
                    kind: MessageKind::UserMessage,
                    role: crate::Role::Manager,
                    reply_target: None,
                    pending_job_id: None,
                })
                .unwrap(),
                // created_at BEFORE the session message → already appended.
                "2025-12-31T00:00:00Z",
            ],
        )
        .await
        .unwrap();
        let rows = list_pending_jobs(conn).await.unwrap();
        assert!(pending_already_appended(conn, &rows[0]).await);
        let replayed = replay_pending_jobs(conn).await.unwrap();
        assert_eq!(replayed, 0, "deduped row must not be re-routed");
        assert_eq!(
            list_pending_jobs(conn).await.unwrap().len(),
            0,
            "deduped row reclaimed"
        );
    }

    /// The dedup strips the timestamp wrapper in BOTH formats: legacy
    /// prefix (`<timestamp>…</timestamp>\n\n{content}`) and current suffix
    /// (`{content}\n\n<timestamp>…</timestamp>`), plus the Manager
    /// ticket-buffer drain prefix shape.
    #[test]
    fn strip_timestamp_wrapper_both_formats() {
        let body = "task text";
        let suffix = format!("{body}\n\n<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>");
        assert_eq!(strip_timestamp_wrapper(&suffix), body);
        let legacy = format!("<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>\n\n{body}");
        assert_eq!(strip_timestamp_wrapper(&legacy), body);
        assert_eq!(
            strip_timestamp_wrapper(body),
            body,
            "no wrapper — unchanged"
        );
        let drained =
            format!("drained\n{body}\n\n<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>");
        assert!(
            strip_timestamp_wrapper(&drained).ends_with(body),
            "drain prefix survives stripping"
        );
    }

    /// Content that mimics the timestamp wrapper must not be mis-stripped: a
    /// legacy message whose content ends with a timestamp-shaped block, a raw
    /// message that merely starts with `<timestamp>`, and a suffix-format
    /// message whose content begins with a timestamp block (the legacy branch
    /// requires the exact `</timestamp>\n\n` separator, so content merely
    /// STARTING with `<timestamp>…</timestamp> more` is left intact).
    #[test]
    fn strip_timestamp_wrapper_ignores_lookalikes() {
        let body = "report\n\n<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>";
        let legacy = format!("<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>\n\n{body}");
        assert_eq!(strip_timestamp_wrapper(&legacy), body);
        let raw = "<timestamp>config options</timestamp>\n\nrest of the message";
        assert_eq!(strip_timestamp_wrapper(raw), raw);
        let leading = "<timestamp>2026-01-01 00:00:00 (UTC)</timestamp> more\n\n\
                       <timestamp>2026-01-01 00:00:00 (UTC)</timestamp>";
        assert_eq!(
            strip_timestamp_wrapper(leading),
            leading,
            "suffix-format message whose content starts with a timestamp block stays intact"
        );
    }

    /// A session persisted in the LEGACY prefix format (pre-rollout rows) is
    /// still recognized by the boot-replay dedup — legacy sessions must keep
    /// deduplicating after the suffix-format rollout.
    #[tokio::test]
    async fn pending_replay_dedup_recognizes_legacy_prefix_format() {
        let (store, _tmp) = crate::open_test_store!(crate::session::SessionStore, "session");
        let conn = &store.conn;
        let agent_id = "manager_legacy";
        let content = "legacy hello";
        conn.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) \
             VALUES (?1, 'user', ?2, ?3)",
            params![
                agent_id,
                format!("<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>\n\n{content}"),
                db::now()
            ],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO pending_jobs (id, target_agent_id, envelope, created_at) \
             VALUES ('p-legacy', ?1, ?2, ?3)",
            params![
                agent_id,
                serde_json::to_string(&AgentJob {
                    content: content.to_string(),
                    workspace_name: "ws".to_string(),
                    user_name: "u".to_string(),
                    channel: "telegram".to_string(),
                    kind: MessageKind::UserMessage,
                    role: crate::Role::Manager,
                    reply_target: None,
                    pending_job_id: None,
                })
                .unwrap(),
                "2025-12-31T00:00:00Z",
            ],
        )
        .await
        .unwrap();
        let rows = list_pending_jobs(conn).await.unwrap();
        assert!(
            pending_already_appended(conn, &rows[0]).await,
            "legacy prefix-format message must still dedup"
        );
    }

    /// F1 at-least-once fix: a row whose content is NOT in the session
    /// (delivery interrupted between persist and the consumer's append) routes
    /// the FULL content — never empty. The dedup (suffix + created_at) is the
    /// authoritative in-session signal.
    #[tokio::test]
    async fn pending_replay_not_in_session_routes_full_content() {
        let _ = crate::agent::message_router::init_global();
        let (store, _tmp) = crate::open_test_store!(crate::session::SessionStore, "session");
        let conn = &store.conn;
        let agent_id = "f1_replay_target";
        let mut rx = crate::agent::message_router::register_agent(agent_id);
        let content = "FULL CONTENT";
        conn.execute(
            "INSERT INTO pending_jobs (id, target_agent_id, envelope, created_at) \
             VALUES ('p2', ?1, ?2, ?3)",
            params![
                agent_id,
                serde_json::to_string(&AgentJob {
                    content: content.to_string(),
                    workspace_name: "ws".to_string(),
                    user_name: String::new(),
                    channel: String::new(),
                    kind: MessageKind::AnalyzeToolResult,
                    role: crate::Role::Manager,
                    reply_target: None,
                    pending_job_id: None,
                })
                .unwrap(),
                db::now(),
            ],
        )
        .await
        .unwrap();
        let replayed = replay_pending_jobs(conn).await.unwrap();
        assert_eq!(replayed, 1, "row not in session must be re-routed");
        let routed = rx
            .try_recv()
            .expect("routed job must carry the full content");
        assert_eq!(
            routed.content, content,
            "not-in-session routes the FULL content, never empty"
        );
        crate::agent::message_router::unregister_agent(agent_id);
    }

    // ── TTL guard ────────────────────────────────────────────────────

    /// The agents table IS the marker: a transient session referenced by an
    /// agents row is NEVER purged by cleanup_old_transient_sessions.
    ///
    /// Serialized with the reset_inflight group: this test and those purge tests
    /// share the one consolidated connection (single transaction domain), and a
    /// purge test holds a raw tx briefly — running concurrently would break its
    /// begin_tx with "cannot start a transaction within a transaction".
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn ttl_guard_protects_job_tracked_sessions() {
        crate::util::test::init_test_stores().await;
        let store = crate::session::store();
        let stale = (chrono::Utc::now() - chrono::Duration::hours(20)).to_rfc3339();
        store
            .batch_append_with_context(
                "ticket_t1_0_suf_engineer",
                &[crate::ChatMessage::user("u")],
                "gui",
                "u",
                "ws",
                "engineer",
            )
            .await
            .unwrap();
        // Age the session past the cutoff so ONLY the TTL guard protects it.
        store
            .conn
            .execute(
                "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = 'ticket_t1_0_suf_engineer'",
                params![stale.clone()],
            )
            .await
            .unwrap();
        // Job-tracked: an agents row references the session.
        let conn = &store.conn;
        crate::util::test::JobRowBuilder::new(conn, "jttl", "ticket_analysis", "engineer", "ws")
            .timestamps(stale.clone())
            .insert()
            .await
            .unwrap();
        conn.execute(
            "INSERT INTO agents (job_id, agent_id, kind, status, task) \
             VALUES ('jttl', 'ticket_t1_0_suf_engineer', 'engineer', 'launched', '')",
            (),
        )
        .await
        .unwrap();
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::hours(PURGE_CUTOFF_HOURS)).to_rfc3339();
        crate::session::cleanup_old_transient_sessions(&cutoff)
            .await
            .unwrap();
        let msgs = store.load("ticket_t1_0_suf_engineer").await;
        assert_eq!(
            msgs.len(),
            1,
            "job-tracked session must survive the TTL guard"
        );
        // After the job goes terminal, the next tick cleans it up.
        conn.execute("DELETE FROM jobs WHERE id = 'jttl'", ())
            .await
            .unwrap();
        crate::session::cleanup_old_transient_sessions(&cutoff)
            .await
            .unwrap();
        let msgs = store.load("ticket_t1_0_suf_engineer").await;
        assert_eq!(
            msgs.len(),
            0,
            "unprotected session is cleaned up after cascade"
        );
    }

    // ── Stale purge + in-place rollback ──────────────────────────────

    /// Purge rolls a stranded ticket back IN PLACE (board-first ordering) and
    /// removes the stale job rows; a live (recently-active) session protects
    /// its job from the purge.
    ///
    /// Serialized with the reset_analysis_tickets tests (shared global board —
    /// a concurrent boot reset would clobber the fixture phases).

    /// A stale analysis job rolls its ticket back unconditionally — one-round
    /// analysis has no round guard, so every stale `ticket_analysis` row in the
    /// reset-transition phase (Analysis) rolls the ticket back to Backlog.
    ///
    /// Serialized with the reset_analysis_tickets tests (shared global board —
    /// a concurrent boot reset would clobber the fixture phases).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn purge_rolls_back_stale_analysis_job_unconditionally() {
        crate::util::test::init_test_stores().await;
        let conn = &crate::session::store().conn;
        let board = crate::pipeline::board::store();
        let ws = crate::workspace::test_ws_named("/tmp/purge_ws2", "purge_ws2");
        let ticket_id = crate::util::test::make_ticket(
            board,
            &ws,
            "Unconditional rollback",
            crate::pipeline::board::TicketPhase::Analysis,
        )
        .await;
        let stale = (chrono::Utc::now() - chrono::Duration::hours(20)).to_rfc3339();
        crate::util::test::JobRowBuilder::new(
            conn,
            "jr1",
            "ticket_analysis",
            "analyst",
            "purge_ws2",
        )
        .timestamps(stale.clone())
        .insert()
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO ticket_jobs (id, ticket_id) \
             VALUES (?1, ?2)",
            params!["jr1", ticket_id.clone()],
        )
        .await
        .unwrap();
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::hours(PURGE_CUTOFF_HOURS)).to_rfc3339();
        purge_stale_jobs(&cutoff).await.unwrap();
        let t = board.get_ticket(&ticket_id).await.unwrap().unwrap();
        assert_eq!(
            t.phase,
            crate::pipeline::board::TicketPhase::Backlog,
            "stale analysis job rolls the ticket back unconditionally (no round guard)"
        );
    }

    /// A frozen (paused) implementation job is purge-immune: it holds the workspace's
    /// running slot while frozen, so both the `ticket_jobs` row and the
    /// `jobs` row must survive the stale purge even when `updated_at` is past
    /// the cutoff (the purge's workspace-pause exclusion, not a fresh
    /// `updated_at`, protects it).
    ///
    /// Serialized with the reset_analysis_tickets tests (shared global board —
    /// a concurrent boot reset would clobber the fixture phases).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn purge_keeps_frozen_implementation_job() {
        crate::util::test::init_test_stores().await;
        let conn = &crate::session::store().conn;
        let ws = crate::util::test::create_test_workspace(
            "/tmp/purge_implementation_ws",
            "purge_implementation_ws",
        )
        .await;
        let ticket_id = crate::util::test::make_ticket(
            crate::pipeline::board::store(),
            &ws,
            "Frozen Implementation",
            crate::pipeline::board::TicketPhase::InDevelopment,
        )
        .await;
        // Real implementation job + ticket_jobs child row, then freeze it.
        spawn_job(
            conn,
            "pj_keep",
            "task",
            &ws.name,
            "",
            "",
            crate::Role::Engineer,
            &[],
            &SpawnChild::TicketImplementation {
                ticket_id: ticket_id.clone(),
            },
        )
        .await
        .unwrap();
        // Pause the workspace — the freeze authority. The implementation job's
        // updated_at is forced stale so the ONLY thing protecting the row is
        // the workspace pause (the purge's stale-window exclusion, not a
        // fresh `updated_at`).
        crate::workspace::store()
            .set_paused(&ws.name, true)
            .await
            .unwrap();
        let stale = (chrono::Utc::now() - chrono::Duration::hours(20)).to_rfc3339();
        conn.execute(
            "UPDATE jobs SET updated_at = ?1 WHERE id = 'pj_keep'",
            params![stale.clone()],
        )
        .await
        .unwrap();

        let cutoff =
            (chrono::Utc::now() - chrono::Duration::hours(PURGE_CUTOFF_HOURS)).to_rfc3339();
        purge_stale_jobs(&cutoff).await.unwrap();

        let job: Option<i64> = conn
            .query_optional("SELECT 1 FROM jobs WHERE id = 'pj_keep'", (), |row| {
                row.get(0)
            })
            .await
            .unwrap();
        assert!(
            job.is_some(),
            "a frozen implementation's jobs row must survive the stale purge"
        );
        let implementation: Option<i64> = conn
            .query_optional(
                "SELECT 1 FROM ticket_jobs WHERE id = 'pj_keep'",
                (),
                |row| row.get(0),
            )
            .await
            .unwrap();
        assert!(
            implementation.is_some(),
            "a frozen implementation's ticket_jobs row must survive the stale purge"
        );
    }

    /// The rollback-failure path: when the board rollback cannot land, the
    /// ticket_analysis job rows are KEPT (next tick retries the CAS). A held
    /// raw transaction makes the rollback's begin_tx fail deterministically
    /// ("cannot start a transaction within a transaction"). Because all stores
    /// now share ONE consolidated connection (single transaction domain), the
    /// same raw tx also blocks the whole sessions purge — so the test verifies
    /// the rollback-path guard directly ([`rollback_stranded_tickets`]) and
    /// then confirms a stale non-ticket_analysis row is purged once the raw tx
    /// is released.
    ///
    /// Serialized with the other reset_analysis_tickets tests (shared global
    /// board — the raw tx would clobber concurrent fixtures).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn purge_keeps_ticket_analysis_rows_when_rollback_fails() {
        crate::util::test::init_test_stores().await;
        let conn = &crate::session::store().conn;
        let board = crate::pipeline::board::store();
        let ws = crate::workspace::test_ws_named("/tmp/purge_ws3", "purge_ws3");
        let ticket_id = crate::util::test::make_ticket(
            board,
            &ws,
            "Rollback failure",
            crate::pipeline::board::TicketPhase::Analysis,
        )
        .await;
        let stale = (chrono::Utc::now() - chrono::Duration::hours(20)).to_rfc3339();
        // Stale ticket_analysis job (must be retained) + stale analyze job (purged).
        crate::util::test::JobRowBuilder::new(
            conn,
            "jfail_ts",
            "ticket_analysis",
            "analyst",
            "purge_ws3",
        )
        .timestamps(stale.clone())
        .insert()
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO ticket_jobs (id, ticket_id) \
             VALUES (?1, ?2)",
            params!["jfail_ts", ticket_id.clone()],
        )
        .await
        .unwrap();
        crate::util::test::JobRowBuilder::new(
            conn,
            "jfail_analyze",
            "analyze",
            "assistant",
            "purge_ws3",
        )
        .timestamps(stale.clone())
        .insert()
        .await
        .unwrap();
        // Raw BEGIN on the shared consolidated connection (bypasses the
        // wrapper's tx tracking). Under the consolidated connection every store
        // shares one transaction domain, so this raw tx blocks BOTH the board
        // rollback AND the sessions purge — it makes the board rollback's
        // begin_tx fail deterministically ("cannot start a transaction within a
        // transaction") and the whole purge is blocked.
        crate::pipeline::board::store()
            .conn
            .execute("BEGIN", ())
            .await
            .expect("raw board BEGIN");
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::hours(PURGE_CUTOFF_HOURS)).to_rfc3339();
        // The board rollback path cannot land while the raw tx is held. Under
        // the shared connection the same raw tx also blocks the sessions purge
        // (its own begin_tx fails), so the ticket_analysis row is KEPT.
        let rollback_ok = rollback_stranded_tickets(&[(
            ticket_id.clone(),
            crate::pipeline::board::TicketPhase::Analysis.to_string(),
        )])
        .await;
        // Restore the shared connection — ALWAYS, before any assertion, so the
        // raw tx never leaks into a later begin_tx across all stores even when
        // the rollback assertion below panics.
        let _ = crate::pipeline::board::store()
            .conn
            .execute("ROLLBACK", ())
            .await;
        assert!(
            !rollback_ok,
            "board rollback must fail to land while a raw tx is held"
        );
        // The ticket_analysis row survives (nothing was purged — the whole
        // purge was blocked by the open raw tx until just now).
        let ts_left = conn
            .query("SELECT COUNT(*) FROM jobs WHERE id = 'jfail_ts'", ())
            .await
            .unwrap();
        assert_eq!(
            ts_left[0].get::<i64>(0).unwrap(),
            1,
            "ticket_analysis row must survive a failed rollback (next tick retries)"
        );
        // Once the raw tx is released the full purge runs (the rollback now
        // lands) and stale non-ticket_analysis rows are purged.
        purge_stale_jobs(&cutoff).await.unwrap();
        let analyze_left = conn
            .query("SELECT COUNT(*) FROM jobs WHERE id = 'jfail_analyze'", ())
            .await
            .unwrap();
        assert_eq!(
            analyze_left[0].get::<i64>(0).unwrap(),
            0,
            "non-ticket_analysis stale rows are purged once the tx is released"
        );
    }

    // ── Boot classification / reset exclusion ────────────────────────

    /// reset_analysis_tickets skips excluded (resumed) tickets; everything
    /// else resets as before. Serialized with the other reset_analysis_tickets
    /// tests (shared global board — a concurrent reset would clobber the
    /// fixtures).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn reset_analysis_tickets_exclusion() {
        crate::util::test::init_test_stores().await;
        let board = crate::pipeline::board::store();
        let ws = crate::workspace::test_ws_named("/tmp/ws_x", "ws_x");
        let resumed = crate::util::test::make_ticket(
            board,
            &ws,
            "resume me",
            crate::pipeline::board::TicketPhase::Analysis,
        )
        .await;
        let reset = crate::util::test::make_ticket(
            board,
            &ws,
            "reset me",
            crate::pipeline::board::TicketPhase::Analysis,
        )
        .await;
        board
            .reset_analysis_tickets(std::slice::from_ref(&resumed))
            .await
            .unwrap();
        let t_resumed = board.get_ticket(&resumed).await.unwrap().unwrap();
        let t_reset = board.get_ticket(&reset).await.unwrap().unwrap();
        assert_eq!(
            t_resumed.phase,
            crate::pipeline::board::TicketPhase::Analysis
        );
        assert_eq!(t_reset.phase, crate::pipeline::board::TicketPhase::Backlog);
    }

    /// The boot scan adds an in-phase ticket_analysis job to the resumed-ticket
    /// exclusion set (its ticket must NOT reset — re-claiming while the
    /// resumed agent runs would duplicate work) and bumps retry_count (every
    /// boot resume). Tickets without a resumable job reset normally. Serialized
    /// with the other reset_analysis_tickets tests (shared global board).

    /// A research_cleanup row — even one with a high retry_count from many
    /// prior boot resumes — must always be resumed: the boot resume re-enters
    /// the interrupted Sanitation agent from its folder/row state. The
    /// checkpoint bump refreshes updated_at (the only protection for the
    /// pre-first-commit window against the 8h purge) and increments
    /// retry_count.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // recover_from_restart resets the shared global board
    async fn recover_from_restart_resumes_research_cleanup_always() {
        // Router init: recover_from_restart replays pending_jobs, which can
        // route envelopes (panic if ROUTER is uninitialized).
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        crate::util::test::JobRowBuilder::new(
            conn,
            "jclean",
            "research_cleanup",
            "sanitation",
            "ws_scan",
        )
        .retry_count(3)
        .timestamps(now.clone())
        .insert()
        .await
        .unwrap();
        // Run-folder fixture: what a crash-left cleanup run leaves behind.
        let run_folder = crate::research_cleanup::run_root_path("jclean");
        tokio::fs::create_dir_all(&run_folder).await.unwrap();
        let resumable = recover_from_restart().await.unwrap();
        assert!(
            resumable.iter().any(|r| matches!(
                r,
                ResumableJob::ResearchCleanup { job_id, .. } if job_id.as_str() == "jclean"
            )),
            "the research_cleanup job must always be selected for resume"
        );
        assert!(
            run_folder.exists(),
            "the run folder must NOT be released — the job is resumed, not abandoned"
        );
        let rc: i64 = conn
            .query_row(
                "SELECT retry_count FROM jobs WHERE id = 'jclean'",
                (),
                |r| r.get::<i64>(0),
            )
            .await
            .unwrap();
        assert_eq!(rc, 4, "retry_count must be bumped per boot resume");
        let updated: String = conn
            .query_row("SELECT updated_at FROM jobs WHERE id = 'jclean'", (), |r| {
                r.get::<String>(0)
            })
            .await
            .unwrap();
        assert!(
            crate::db::parse_utc_timestamp(&updated).unwrap()
                > crate::db::parse_utc_timestamp(&now).unwrap(),
            "the checkpoint bump must refresh updated_at"
        );
    }

    /// A `temp_cleanup` row left over from a previous lifetime (crash
    /// mid-cleaner) must be TERMINALIZED at boot, never resumed: the cleaner
    /// is fire-and-forget, its workspace is a synthetic ephemeral name that
    /// is never registered in the `workspaces` table, and the next scheduled pass
    /// re-runs the cleanup cleanly.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // recover_from_restart resets the shared global board
    async fn recover_from_restart_terminalizes_temp_cleanup_rows() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let now = crate::db::now();
        crate::util::test::JobRowBuilder::new(
            conn,
            "jtmpclean",
            "temp_cleanup",
            "sanitation",
            "tmp",
        )
        .timestamps(now.clone())
        .insert()
        .await
        .unwrap();
        let resumable = recover_from_restart().await.unwrap();
        // Terminalized: the row is GONE and no resumable stage references it.
        let remaining = conn
            .query(
                "SELECT COUNT(*) FROM jobs WHERE id = 'jtmpclean' AND kind = 'temp_cleanup'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            remaining[0].get::<i64>(0).unwrap(),
            0,
            "temp_cleanup row must be terminalized at boot (fire-and-forget, no resume)"
        );
        assert!(
            !resumable.iter().any(|r| matches!(
                r,
                ResumableJob::ResearchCleanup { job_id, .. } if job_id.as_str() == "jtmpclean"
            )),
            "no resumable entry may refer to the temp_cleanup row"
        );
    }

    /// Crash-window replay scan: a pending research-completion envelope whose
    /// run already HAS a `research_cleanup` jobs row (crash mid-cleanup) must
    /// not be re-dispatched — the existing row is the dedup marker and the
    /// `research_cleanup` boot-scan arm resumes it. The envelope itself still
    /// replays. This exercises the scan's workspace resolution + dedup path
    /// without launching a live cleanup agent.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shared process-lifetime store rows
    async fn replay_skips_cleanup_dispatch_when_cleanup_row_exists() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        crate::util::test::create_test_workspace("/tmp/test_ws_replay_cleanup", "ws_replay").await;
        let now = crate::db::now();
        crate::util::test::JobRowBuilder::new(
            conn,
            "rcln",
            "research_cleanup",
            "sanitation",
            "ws_replay",
        )
        .timestamps(now.clone())
        .insert()
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO pending_jobs (id, target_agent_id, envelope, created_at) \
             VALUES ('rcln', 'manager_ws_replay', ?1, ?2)",
            params![
                serde_json::to_string(&AgentJob {
                    content: "<research-result>done</research-result>".to_string(),
                    workspace_name: "ws_replay".to_string(),
                    user_name: "u".to_string(),
                    channel: "telegram".to_string(),
                    kind: MessageKind::ResearchResult,
                    role: crate::Role::Manager,
                    reply_target: None,
                    pending_job_id: None,
                })
                .unwrap(),
                now
            ],
        )
        .await
        .unwrap();
        let replayed = replay_pending_jobs(conn).await.unwrap();
        assert_eq!(replayed, 1, "the envelope itself is replayed");
        let rows = conn
            .query(
                "SELECT COUNT(*) FROM jobs WHERE id = 'rcln' AND kind = 'research_cleanup'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            rows[0].get::<i64>(0).unwrap(),
            1,
            "no duplicate cleanup dispatch — the existing row is the dedup marker"
        );
        let sessions = conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE agent_id = 'cleanup_rcln'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            sessions[0].get::<i64>(0).unwrap(),
            0,
            "the scan must not spawn a second cleanup agent"
        );
        // Clean up the rows this test created (shared process-lifetime store —
        // leftover rows would inflate other tests' replay counts).
        conn.execute("DELETE FROM pending_jobs WHERE id = 'rcln'", ())
            .await
            .unwrap();
        conn.execute("DELETE FROM jobs WHERE id = 'rcln'", ())
            .await
            .unwrap();
    }

    /// Crash-window replay → boot-scan resume, end to end: a pending
    /// research-completion envelope with NO cleanup row (daemon died between
    /// complete_durable_job and dispatch_research_cleanup). The replay must
    /// create the row ONLY — the `research_cleanup` boot-scan arm (which runs
    /// after the replay in recover_from_restart) must be the SOLE dispatcher,
    /// selecting the just-created row for resume. Spawning the agent during
    /// the replay would double-run the same `cleanup_{run_id}` id on this boot
    /// (the scan arm would register it via AgentRegistry, replacing + canceling
    /// the replay-spawned agent).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // recover_from_restart resets the shared global board
    async fn replay_creates_cleanup_row_then_scan_resumes() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        crate::util::test::create_test_workspace("/tmp/test_ws_replay_resume", "ws_replay2").await;
        let now = crate::db::now();
        conn.execute(
            "INSERT INTO pending_jobs (id, target_agent_id, envelope, created_at) \
             VALUES ('rcln2', 'manager_ws_replay2', ?1, ?2)",
            params![
                serde_json::to_string(&AgentJob {
                    content: "<research-result>done</research-result>".to_string(),
                    workspace_name: "ws_replay2".to_string(),
                    user_name: "u".to_string(),
                    channel: "telegram".to_string(),
                    kind: MessageKind::ResearchResult,
                    role: crate::Role::Manager,
                    reply_target: None,
                    pending_job_id: None,
                })
                .unwrap(),
                now
            ],
        )
        .await
        .unwrap();
        // Run-folder fixture: the terminalized run left its folder behind, so
        // the replay's folder-existence discriminator classifies this as the
        // crash window (cleanup never ran) and recreates the row.
        let run_folder = crate::research_cleanup::run_root_path("rcln2");
        tokio::fs::create_dir_all(&run_folder).await.unwrap();
        let resumable = recover_from_restart().await.unwrap();
        // (a) The replay created the row (dedup marker + folder-hold).
        assert!(
            crate::research_cleanup::research_cleanup_row_exists(conn, "rcln2")
                .await
                .unwrap(),
            "the crash-window replay must create the cleanup row"
        );
        // (b) The boot-scan arm selected the row for resume (sole dispatcher).
        assert!(
            resumable.iter().any(|r| matches!(
                r,
                ResumableJob::ResearchCleanup { job_id, .. } if job_id.as_str() == "rcln2"
            )),
            "the boot-scan arm must resume the replay-created cleanup row"
        );
        // (c) No cleanup agent was spawned by the replay itself (the resume
        // path in management.rs runs the agent — a spawn here would be the
        // double-dispatch the design forbids).
        let sessions = conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE agent_id = 'cleanup_rcln2'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            sessions[0].get::<i64>(0).unwrap(),
            0,
            "the replay must not spawn the cleanup agent — the boot-scan arm resumes it"
        );
        // Clean up the rows the test created (shared process-lifetime store —
        // leftover rows would inflate other tests' replay counts).
        conn.execute("DELETE FROM pending_jobs WHERE id = 'rcln2'", ())
            .await
            .unwrap();
        conn.execute("DELETE FROM jobs WHERE id = 'rcln2'", ())
            .await
            .unwrap();
        let _ = tokio::fs::remove_dir_all(&run_folder).await;
    }

    /// Cleanup-completed-but-envelope-undelivered dedup: a pending
    /// research-completion envelope whose run folder is ALREADY GONE (the
    /// cleanup ran to completion in a previous lifetime — the tail released
    /// the folder and terminalized the row) must NOT re-dispatch — folder
    /// absence is the completed marker on the replay path, so re-dispatching
    /// would cost one LLM round per boot until the row ages into the 8h purge
    /// ([`PURGE_CUTOFF_HOURS`]). The envelope itself still replays.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shared process-lifetime store rows
    async fn replay_skips_cleanup_dispatch_when_run_folder_gone() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        crate::util::test::create_test_workspace("/tmp/test_ws_replay_archived", "ws_replay3")
            .await;
        let now = crate::db::now();
        // Simulate a cleanup that completed in a previous lifetime: the run
        // folder was released by the cleanup tail (folder removed, row
        // terminalized) while the envelope stayed undelivered.
        let run_folder = crate::research_cleanup::run_root_path("rcln3");
        tokio::fs::create_dir_all(&run_folder).await.unwrap();
        crate::research_cleanup::release_run_folder("rcln3").await;
        assert!(!run_folder.exists(), "fixture: run folder released");
        conn.execute(
            "INSERT INTO pending_jobs (id, target_agent_id, envelope, created_at) \
             VALUES ('rcln3', 'manager_ws_replay3', ?1, ?2)",
            params![
                serde_json::to_string(&AgentJob {
                    content: "<research-result>done</research-result>".to_string(),
                    workspace_name: "ws_replay3".to_string(),
                    user_name: "u".to_string(),
                    channel: "telegram".to_string(),
                    kind: MessageKind::ResearchResult,
                    role: crate::Role::Manager,
                    reply_target: None,
                    pending_job_id: None,
                })
                .unwrap(),
                now
            ],
        )
        .await
        .unwrap();
        let replayed = replay_pending_jobs(conn).await.unwrap();
        assert_eq!(replayed, 1, "the envelope itself is replayed");
        assert!(
            !crate::research_cleanup::research_cleanup_row_exists(conn, "rcln3")
                .await
                .unwrap(),
            "a released run folder means the cleanup completed — no re-dispatch"
        );
        let sessions = conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE agent_id = 'cleanup_rcln3'",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            sessions[0].get::<i64>(0).unwrap(),
            0,
            "no cleanup agent for a completed cleanup"
        );
        // Clean up the rows the test created (shared process-lifetime store —
        // leftover rows would inflate other tests' replay counts).
        conn.execute("DELETE FROM pending_jobs WHERE id = 'rcln3'", ())
            .await
            .unwrap();
    }

    /// The kind-aware boot-resume predicate: a `ticket_implementation` job
    /// resumes iff the ticket is in a pipeline-occupied phase, a
    /// `ticket_analysis` job resumes iff the ticket is in `Analysis`, and any
    /// other phase completes the job as stale. This locks in the behavior
    /// change that replaced the old strict (kind, stage)-derived phase-equality
    /// guard — which could complete a live job whose stale stage mirror drifted
    /// from the ticket phase, stranding the ticket in an occupied phase.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // recover_from_restart resets the shared global board
    async fn recover_boot_predicate_is_kind_aware_on_ticket_phase() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let ws = crate::util::test::create_test_workspace(
            "/tmp/test_ws_boot_predicate",
            "boot_predicate_ws",
        )
        .await;
        let board = crate::pipeline::board::store();
        async fn job_status(conn: &crate::db::Connection, id: &str) -> String {
            conn.query_row("SELECT status FROM jobs WHERE id = ?1", params![id], |r| {
                r.get::<String>(0)
            })
            .await
            .unwrap()
        }

        // (a) ticket_analysis + ticket in Analysis → resume.
        let analysis_ticket = crate::util::test::make_ticket(
            board,
            &ws,
            "Analysis resume",
            crate::pipeline::board::TicketPhase::Analysis,
        )
        .await;
        spawn_job(
            conn,
            "jb_analysis",
            "task",
            &ws.name,
            "",
            "",
            crate::Role::Analyst,
            &[],
            &SpawnChild::TicketAnalysis {
                ticket_id: analysis_ticket.clone(),
            },
        )
        .await
        .unwrap();

        // (b) ticket_analysis + ticket moved out of Analysis (stale) → complete.
        let stale_analysis_ticket = crate::util::test::make_ticket(
            board,
            &ws,
            "Stale analysis",
            crate::pipeline::board::TicketPhase::Backlog,
        )
        .await;
        spawn_job(
            conn,
            "jb_analysis_stale",
            "task",
            &ws.name,
            "",
            "",
            crate::Role::Analyst,
            &[],
            &SpawnChild::TicketAnalysis {
                ticket_id: stale_analysis_ticket.clone(),
            },
        )
        .await
        .unwrap();

        // (c) ticket_implementation + ticket in a pipeline-occupied phase → resume.
        let impl_ticket = crate::util::test::make_ticket(
            board,
            &ws,
            "Implementation resume",
            crate::pipeline::board::TicketPhase::InDevelopment,
        )
        .await;
        spawn_job(
            conn,
            "jb_impl",
            "task",
            &ws.name,
            "",
            "",
            crate::Role::Engineer,
            &[],
            &SpawnChild::TicketImplementation {
                ticket_id: impl_ticket.clone(),
            },
        )
        .await
        .unwrap();

        // (d) ticket_implementation + ticket moved out of the pipeline (stale) → complete.
        let stale_impl_ticket = crate::util::test::make_ticket(
            board,
            &ws,
            "Stale implementation",
            crate::pipeline::board::TicketPhase::Backlog,
        )
        .await;
        spawn_job(
            conn,
            "jb_impl_stale",
            "task",
            &ws.name,
            "",
            "",
            crate::Role::Engineer,
            &[],
            &SpawnChild::TicketImplementation {
                ticket_id: stale_impl_ticket.clone(),
            },
        )
        .await
        .unwrap();

        let resumable = recover_from_restart().await.unwrap();

        // Analysis + Analysis phase → resumed.
        assert!(
            resumable.iter().any(|r| matches!(
                r,
                ResumableJob::TicketAnalysis { job_id, .. }
                    if job_id.as_str() == "jb_analysis"
            )),
            "a ticket_analysis job with the ticket in Analysis must resume",
        );
        assert_eq!(job_status(conn, "jb_analysis").await, "launched");

        // Analysis + stale phase → completed (not resumed).
        assert!(
            !resumable.iter().any(|r| matches!(
                r,
                ResumableJob::TicketAnalysis { job_id, .. } if job_id.as_str() == "jb_analysis_stale"
            )),
            "a ticket_analysis job with a moved ticket must complete, not resume",
        );
        assert_eq!(job_status(conn, "jb_analysis_stale").await, "done");

        // Implementation + occupied phase → resumed.
        assert!(
            resumable.iter().any(|r| matches!(
                r,
                ResumableJob::TicketImplementation { job_id, .. }
                    if job_id.as_str() == "jb_impl"
            )),
            "a ticket_implementation job with the ticket in a pipeline-occupied phase must resume",
        );
        assert_eq!(job_status(conn, "jb_impl").await, "launched");

        // Implementation + stale phase → completed (not resumed).
        assert!(
            !resumable.iter().any(|r| matches!(
                r,
                ResumableJob::TicketImplementation { job_id, .. } if job_id.as_str() == "jb_impl_stale"
            )),
            "a ticket_implementation job with a moved ticket must complete, not resume",
        );
        assert_eq!(job_status(conn, "jb_impl_stale").await, "done");
    }

    /// The workspace-pause freeze marker round-trips through the CAS claim:
    /// only a job marked by [`mark_analysis_frozen`] is claimable, and exactly
    /// one caller wins the claim (`paused_frozen = 1 → 0`). This is what
    /// lets [`crate::pipeline::management::re_drive_analysis_rounds`] tell a
    /// genuinely frozen round (re-driven) from a normally-finalizing one
    /// (never marked, never re-driven).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn analysis_freeze_marker_roundtrips_through_cas_claim() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let ws =
            crate::util::test::create_test_workspace("/tmp/ws_freeze_marker", "ws_freeze_marker")
                .await;
        let board = crate::pipeline::board::store();
        let ticket = crate::util::test::make_ticket(
            board,
            &ws,
            "Freeze marker",
            crate::pipeline::board::TicketPhase::Analysis,
        )
        .await;
        spawn_job(
            conn,
            "jb_freeze_marker",
            "task",
            &ws.name,
            "",
            "",
            crate::Role::Analyst,
            &[],
            &SpawnChild::TicketAnalysis {
                ticket_id: ticket.clone(),
            },
        )
        .await
        .unwrap();

        async fn marker(conn: &crate::db::Connection, job_id: &str) -> bool {
            conn.query_row(
                "SELECT paused_frozen FROM jobs WHERE id = ?1",
                params![job_id],
                |r| r.get::<bool>(0),
            )
            .await
            .unwrap()
        }

        // Freshly spawned: not frozen.
        assert!(!marker(conn, "jb_freeze_marker").await);

        // Mark frozen → re-drive sees it as frozen.
        mark_analysis_frozen(conn, "jb_freeze_marker")
            .await
            .unwrap();
        assert!(marker(conn, "jb_freeze_marker").await);

        // First claim wins; the marker clears.
        assert!(
            claim_analysis_resume(conn, "jb_freeze_marker")
                .await
                .unwrap()
        );
        assert!(!marker(conn, "jb_freeze_marker").await);

        // A second claim on the now-clear marker loses (already resumed).
        assert!(
            !claim_analysis_resume(conn, "jb_freeze_marker")
                .await
                .unwrap()
        );

        // An unmarked job is never claimable (never frozen).
        mark_analysis_frozen(conn, "jb_freeze_marker")
            .await
            .unwrap();
        assert!(
            claim_analysis_resume(conn, "jb_freeze_marker")
                .await
                .unwrap()
        );
        assert!(
            !claim_analysis_resume(conn, "jb_freeze_marker")
                .await
                .unwrap()
        );

        // Non-analysis kinds are never marked (guard in the helper).
        spawn_job(
            conn,
            "jb_freeze_impl",
            "task",
            &ws.name,
            "",
            "",
            crate::Role::Engineer,
            &[],
            &SpawnChild::TicketImplementation {
                ticket_id: ticket.clone(),
            },
        )
        .await
        .unwrap();
        mark_analysis_frozen(conn, "jb_freeze_impl").await.unwrap();
        assert!(!marker(conn, "jb_freeze_impl").await);
        assert!(!claim_analysis_resume(conn, "jb_freeze_impl").await.unwrap());
    }
}
