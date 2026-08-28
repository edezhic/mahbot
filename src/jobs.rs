//! Durable jobs layer: jobs/agents/pending_jobs lifecycle, boot recovery
//! scan, and stale purge orchestration.
//!
//! All rows live in the consolidated domain database (`core.db`) behind the
//! one shared connection — the jobs/session/board tables now share a single
//! transaction domain, so cross-store ordering and crash-safety can be
//! expressed in a single transaction (see the purge section in the design).
//!
//! Table ownership: the DDL lives in the append-only schema catalog
//! ([`crate::db::migrations`]); this module owns the row model, the lifecycle
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
    pub ticket_id: Option<String>,
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

fn job_row_from(row: &Row) -> anyhow::Result<JobRow> {
    Ok(JobRow {
        id: row.get(0)?,
        kind: row.get(1)?,
        workspace_name: row.get(2)?,
        retry_count: row.get(3)?,
        ticket_id: row.get(4)?,
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
) -> Result<()> {
    let kind = child.kind_str();
    let ticket_id = child_ticket_id(child);
    let now = db::now();
    let tx = conn.begin_tx().await?;
    tx.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, user_name, channel, role, \
         ticket_id, retry_count, created_at, updated_at) \
         VALUES (?1, ?2, 'launched', ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?9)",
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

/// Terminate a ticket's phase job (DELETE the row; the cascade clears the
/// agent roster). A phase job is short-lived per phase — on success the phase
/// body transitions the ticket and deletes its own job; on hard-failure
/// cleanup it is also destroyed (the puller re-creates a fresh attempt).
pub(crate) async fn complete_ticket_job(conn: &Connection, job_id: &str) -> Result<()> {
    terminalize_job(conn, job_id).await
}

/// Kind-neutral terminalization for jobs whose result cannot be delivered
/// (missing child row / unresolvable caller identity): DELETE the row
/// entirely — CASCADE removes roster + child rows. Envelope kinds
/// (analyze/research) are terminal only when the row is gone (the pending row IS
/// the durable record); no-ops when the row is already absent. Safe for any
/// jobs.kind — [`complete_ticket_job`] is the same per-phase DELETE (the phase
/// job is transient by design; the puller recreates it on the next tick).
pub(crate) async fn terminalize_job(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute("DELETE FROM jobs WHERE id = ?1", params![job_id])
        .await
        .context("terminalize job")?;
    Ok(())
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
    conn.query_optional(
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
        .query_optional(
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
/// `updated_at` so a long-running resumed round is not sweep-purged (the 8h
/// purge keys off `jobs.updated_at`). The stored `idx`/`task` are preserved.
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
            "SELECT id, kind, workspace_name, retry_count, ticket_id FROM jobs \
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
/// Cutoff = 8h. Deletes jobs whose updated_at predates the cutoff AND whose
/// roster agents' sessions are all stale too (live sessions referenced by
/// unfinished jobs are NEVER purged — the agents table IS the marker).
/// A stale ticket phase job is deleted here; the puller re-creates it for the
/// ticket (which stays in its phase) on the next tick — `tickets.phase` is the
/// sole authority, so no board rollback is needed. Paused workspaces' phase
/// jobs are purge-immune (they hold the running slot while frozen).
pub async fn purge_stale_jobs(cutoff: &str) -> Result<u64> {
    let conn = &crate::session::store().conn;

    // A paused (frozen) phase job is purge-immune: it holds the workspace's
    // running slot while frozen, and must survive an arbitrarily long pause.
    // The freeze authority is the workspace `paused` flag. Non-ticket jobs
    // (researches, temp cleanups) are handled by their own lifecycle.
    //
    // A stale ticket phase job is deleted here; the puller re-creates it for
    // the ticket (which stays in its phase) on the next tick — no rollback is
    // needed because `tickets.phase` is the sole authority and the puller is
    // the sole dispatch driver.
    let rows = conn
        .query(
            "SELECT j.id, j.kind, j.workspace_name, j.ticket_id \
             FROM jobs j \
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
        let kind: String = row.get(1)?;
        let workspace_name: String = row.get(2)?;
        // A paused workspace's phase job is frozen — purge-immune.
        if is_ticket_phase_kind(&kind) && paused_ws.contains(&workspace_name) {
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

/// Is this `jobs.kind` one of the ticket working-phase kinds?
pub(crate) fn is_ticket_phase_kind(kind: &str) -> bool {
    matches!(
        kind,
        "analysis" | "in_development" | "in_diagnostics" | "in_review" | "in_qa" | "in_sanitation"
    )
}

/// Boot recovery scan: first statement of run_management. Order: (0) replay
/// pending_jobs; (1) one scan over the active jobs — ticket phase jobs only get
/// their stale launched roster cleared (the puller re-drives them), the
/// research/analyze/research_cleanup kinds resume, and temp_cleanup rows are
/// terminalized.
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
            let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
            resumable.push(ResumableJob::Implement {
                job_id: job.id.clone(),
                workspace_name: job.workspace_name.clone(),
            });
            resumed_other += 1;
        } else if job.kind == "research_cleanup" {
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

    // Boot sweep for engineer anchors (S5 terminal deletion): anchors for
    // tickets already in a terminal phase are removed so the TTL guard stops
    // protecting their accumulated sessions (the 5-min archive loop re-runs
    // this — boot just catches up after downtime).
    let _ = purge_terminal_session_pins().await;

    Ok(resumable)
}

// ── Engineer anchor (S5 — permanently-NULL seat) ────────────────────────

/// Deterministic engineer anchor agent ID: the per-ticket accumulated
/// session `ticket_{ticket_id}_engineer`, permanently job_id = NULL.
#[must_use]
pub fn engineer_session_pin_id(ticket_id: &str) -> String {
    crate::session::ticket_agent_id(ticket_id, crate::Role::Engineer.as_str())
}

/// Deterministic sanitation anchor agent ID: the per-ticket accumulated
/// session `ticket_{ticket_id}_sanitation`, permanently job_id = NULL.
#[must_use]
pub fn sanitation_session_pin_id(ticket_id: &str) -> String {
    crate::session::ticket_agent_id(ticket_id, crate::Role::Sanitation.as_str())
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

/// Upsert the sanitation anchor (permanently-NULL seat) — mirrors the engineer
/// pin so the sanitation session survives re-attempts (resets/new phase jobs).
pub(crate) async fn upsert_sanitation_session_pin(
    conn: &Connection,
    ticket_id: &str,
    task: &str,
    status: RowStatus,
) -> Result<()> {
    let anchor_id = sanitation_session_pin_id(ticket_id);
    conn.execute(
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, outcome, task) \
         VALUES (NULL, ?1, 'sanitation', NULL, ?2, NULL, ?3) \
         ON CONFLICT(agent_id) WHERE job_id IS NULL \
         DO UPDATE SET status = ?2, task = ?3, outcome = NULL",
        params![anchor_id, status.as_str(), task],
    )
    .await
    .with_context(|| format!("failed to upsert sanitation anchor for ticket {ticket_id}"))?;
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

/// Parse `ticket_{ticket_id}_sanitation` — the ticket id is everything before
/// the final `_sanitation` (workspace names allow underscores, so ticket ids
/// like `my_ws-42` are valid — only the final suffix is stripped).
#[must_use]
pub fn sanitation_session_pin_ticket_id(agent_id: &str) -> Option<String> {
    agent_id
        .strip_prefix("ticket_")
        .and_then(|rest| rest.strip_suffix("_sanitation"))
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
        let ticket_id = if kind == "engineer" {
            engineer_session_pin_ticket_id(&agent_id)
        } else {
            sanitation_session_pin_ticket_id(&agent_id)
        };
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
                warn!(agent = %agent_id, error = %e, "Failed to delete terminal engineer anchor");
            }
        }
    }
    if deleted > 0 {
        info!(deleted, "Removed engineer anchors for terminal tickets");
    }
    deleted
}
