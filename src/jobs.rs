//! Durable job substrate: jobs/agents/pending_jobs lifecycle, boot recovery
//! scan, and stale purge orchestration.
//!
//! All rows live in sessions.db — the only DB sharing a transaction domain
//! with session appends (no cross-DB transactions; ordering and
//! crash-safety per the purge section in the design).
//!
//! Table ownership: the DDL is appended to the session SCHEMA const
//! (see `session/mod.rs`); this module owns the row model, the lifecycle
//! helpers, the boot scan, and the purge orchestrator.

use crate::Role;
use crate::message_router::{AgentJob, JobKind};
use crate::turso::{self, Connection, Row, TxGuard, Value, params};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

// ── Job-kind vocabulary (jobs.kind) ─────────────────────────────────────
// Values of `jobs.kind` are fully determined by the child row (one kind per
// child — `SpawnChild::kind_str`); the [`JobKind`] enum travels inside the
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    Analyst,
    Verifier,
    Engineer,
    Sanitation,
}

impl AgentKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Analyst => "analyst",
            Self::Verifier => "verifier",
            Self::Engineer => "engineer",
            Self::Sanitation => "sanitation",
        }
    }
}

/// Boot re-dispatch cap for a single job (retry_count increments per
/// boot-resume attempt; exceeding the cap → partial report from checkpoint
/// (research) or failed (other kinds)). In-job retries are not counted.
pub const MAX_BOOT_REDISPATCH: i64 = 3;

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
/// `management::append_ticket_stage_slots`. Status is the literal 'launched'.
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
    /// `crate::temp_cleanup`). No child row: the jobs row's `task` holds the
    /// cleanup prompt. Leftover rows are terminalized at boot (never
    /// resumed) — the cleaner's ephemeral workspace is never registered.
    TempCleanup,
    /// ticket_stage_jobs (id, ticket_id, stage, phase, round).
    TicketStage {
        ticket_id: String,
        stage: String,
        phase: String,
        round: i64,
    },
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
            Self::TicketStage { .. } => "ticket_stage",
        }
    }
}

/// Spawn a job with its pre-generated agent roster in ONE transaction.
/// MUST commit before the agent's first session write. The kind is derived
/// from the child via [`SpawnChild::kind_str`] — closing the drift window of
/// an inconsistent (kind, child) pair.
#[allow(clippy::too_many_arguments)]
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
    let now = turso::now();
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
        SpawnChild::TicketStage {
            ticket_id,
            stage,
            phase,
            round,
        } => {
            tx.execute(
                "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![id, ticket_id.clone(), stage.clone(), phase.clone(), round],
            )
            .await
            .with_context(|| format!("failed to insert ticket_stage_jobs row for job {id}"))?;
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
    let now = turso::now();
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
    let now = turso::now();
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
    kind: JobKind,
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

/// Terminalize a ticket_stage job (no pending row — TicketNotify flows via
/// the board comment/transition, best-effort). Ordering contract: the board
/// transition+comment runs FIRST; this is the sessions.db completion tx.
pub(crate) async fn complete_ticket_stage_job(conn: &Connection, job_id: &str) -> Result<()> {
    let tx = conn.begin_tx().await?;
    tx.execute(
        "UPDATE jobs SET status = 'done', updated_at = ?1 WHERE id = ?2",
        params![turso::now(), job_id],
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
/// jobs.kind — unlike [`complete_ticket_stage_job`], which is the
/// ticket_stage-specific form (status='done' keeps the row for the
/// phase-check/reset logic).
pub(crate) async fn terminalize_job(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute("DELETE FROM jobs WHERE id = ?1", params![job_id])
        .await
        .context("terminalize job")?;
    Ok(())
}

/// Delete the jobs row (CASCADE destroys ticket_stage/analyze/research child
/// rows) inside an existing transaction.
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
    crate::session::resolve_agent_id(
        &job.channel,
        &job.user_name,
        job.role.as_str(),
        &job.workspace_name,
    )
}

/// Caller identity + task of a job, read from the `jobs` row ALONE — child
/// rows are never required for resume/capped delivery, so a job whose child
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

/// Boot-resume preamble shared by the analyze/research resume and capped-report
/// paths (the 4 preambles were word-for-word identical). Order is load-bearing:
/// aborting-guard FIRST (quiet return — the job row stays for the next boot),
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
/// tracked in [`crate::call_registry::NON_AGENT_CALLS`]; the research
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
/// ticket_stage joint-comment synthesis can both see both registries empty
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
        if crate::registry::AGENT_REGISTRY.list().is_empty()
            && crate::call_registry::NON_AGENT_CALLS.list().is_empty()
        {
            info!("Drain complete — no in-flight agents or orchestrator calls; exiting");
            shutdown();
            return;
        }
        if start.elapsed() >= Duration::from_secs(DRAIN_CAP_SECS) {
            warn!("Drain cap ({DRAIN_CAP_SECS}s) reached — force-cancelling in-flight work");
            crate::registry::AGENT_REGISTRY.shutdown_all();
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
pub(crate) enum ResumableStage {
    TicketStage {
        job_id: String,
        ticket_id: String,
        stage: String,
        workspace_name: String,
    },
    Research {
        job_id: String,
        workspace_name: String,
        capped: bool,
    },
    Analyze {
        job_id: String,
        workspace_name: String,
        capped: bool,
    },
    /// A research-run cleanup Sanitation agent interrupted by a crash. The
    /// jobs row id == the run id, so the row is the cleanup's resume marker
    /// and holds the folder until the cleanup completes.
    ResearchCleanup {
        job_id: String,
        workspace_name: String,
    },
}

/// One in-phase ticket_stage candidate for the boot-scan round dedup.
struct StageCandidate {
    job_id: String,
    ticket_id: String,
    stage: String,
    workspace_name: String,
    round: i64,
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
/// [`crate::turso::parse_utc_timestamp`] (chrono parsing, not raw lexical
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
    // unparseable (unreachable in practice — both come from turso::now()).
    let appended_before = match (
        crate::turso::parse_utc_timestamp(&row.created_at).ok(),
        crate::turso::parse_utc_timestamp(&last_created).ok(),
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
        if job.kind == JobKind::ResearchResult
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
        crate::message_router::route(&row.target_agent_id, job);
        replayed += 1;
    }
    Ok(replayed)
}

/// Stale-job purge with in-place ticket rollback.
///
/// Cutoff = 8h. Deletes jobs whose updated_at predates the cutoff AND whose
/// roster agents' sessions are all stale too (live sessions referenced by
/// unfinished jobs are NEVER purged — the agents table IS the marker).
/// `ticket_stage` jobs stranded in a blocking phase are rolled back IN PLACE
/// at purge time — cross-DB (jobs delete in sessions.db + ticket phase
/// rollback in board.db), no cross-DB tx.
///
/// Crash-safe ordering:
/// 1. SELECT the purge set (ticket_id, phase, stage, round) BEFORE deleting —
///    CASCADE destroys ticket_stage rows at delete time.
/// 2. ONE board.db tx with per-ticket CAS rollback (phase CAS as the
///    last-line race guard; round guard: only the row's LATEST round for its
///    own (ticket_id, stage) — rounds are scoped per stage, never across
///    stages, so an older round of a different stage cannot suppress a stale
///    row's rollback).
/// 3. ONE sessions.db tx DELETE FROM jobs.
///
/// Crash after (2) → CAS fails next tick, boot phase-check deletes the stale
/// row; crash after (1) → nothing changed. Jobs-delete-first would strand a
/// ticket in a blocking phase with no job row — strictly worse.
///
/// A boot does NOT rescue a stranded ticket via resume: `recover_from_restart`
/// bumps every resumed row's updated_at (checkpoint_job), which makes the row
/// purge-immune on later ticks, and a purge-deleted row has nothing to resume
/// at all — recovery for the stranded ticket is specifically the next-boot
/// reset (`reset_inflight_tickets`). The in-place rollback here is the only
/// runtime rescue.
pub async fn purge_stale_jobs(cutoff: &str) -> Result<u64> {
    let conn = &crate::session::store().conn;

    // (1) SELECT the purge set.
    let rows = conn
        .query(
            "SELECT j.id, j.kind, ts.ticket_id, ts.phase, ts.round, ts.stage FROM jobs j \
             LEFT JOIN ticket_stage_jobs ts ON ts.id = j.id \
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
    let mut purge_ids: Vec<String> = Vec::with_capacity(rows.len());
    // ticket_stage job ids whose board rollback must land before the sessions
    // DELETE (a failed rollback keeps those rows for the next tick's retry).
    let mut ticket_stage_ids: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(rows.len());
    // (ticket_id, phase, is_latest_round) pairs for the board rollback, with
    // the row's LATEST round for its own (ticket_id, stage) pre-computed on
    // the sessions conn (the board tx cannot see ticket_stage_jobs — it
    // lives in sessions.db).
    let mut rollbacks: Vec<(String, String, bool)> = Vec::new();
    for row in &rows {
        let id: String = row.get(0)?;
        let kind: String = row.get(1)?;
        purge_ids.push(id.clone());
        if kind == "ticket_stage" {
            ticket_stage_ids.insert(id);
            let ticket_id: Option<String> = row.get(2).ok();
            let phase: Option<String> = row.get(3).ok();
            let round: Option<i64> = row.get(4).ok();
            let stage: Option<String> = row.get(5).ok();
            if let (Some(t), Some(p), Some(r), Some(s)) = (ticket_id, phase, round, stage) {
                // Round numbers are scoped per (ticket_id, stage) — an older
                // round of a DIFFERENT stage must not suppress this row's
                // rollback (a completed analysis round-2 row would otherwise
                // block the rescue of a stale review round-1 row, stranding
                // the ticket in a blocking phase).
                let latest: Option<i64> = conn
                    .query_row(
                        "SELECT MAX(round) FROM ticket_stage_jobs \
                         WHERE ticket_id = ?1 AND stage = ?2",
                        params![t.clone(), s.clone()],
                        |row| row.get::<i64>(0),
                    )
                    .await
                    .ok();
                rollbacks.push((t, p, latest == Some(r)));
            }
        }
    }

    // (2) ONE board.db tx with per-ticket CAS rollback. A missing board
    // (uninitialized) or a failed tx means the rollback did NOT land — keep
    // the ticket_stage rows so the next tick retries the CAS (aligned with the
    // tx-failure path; deleting the job would strand the ticket in a blocking
    // phase with no runtime recovery until the next boot reset).
    let mut rollback_ok = true;
    if !rollbacks.is_empty() {
        if crate::board::BOARD.get().is_some() {
            rollback_ok = rollback_stranded_tickets(&rollbacks).await;
        } else {
            rollback_ok = false;
        }
    }

    // (3) ONE sessions.db tx DELETE FROM jobs (CASCADE removes roster +
    // child rows; the engineer anchor survives — NULL FK child). A failed
    // board rollback keeps ticket_stage rows in place so the next purge tick
    // retries the CAS — deleting the job first would strand the ticket in a
    // blocking phase with no runtime recovery until the next boot reset.
    //
    // Structural safety for research_cleanup rows (kind-agnostic DELETE here):
    // a research_cleanup row can only reach this purge with its run folder
    // ALREADY released — `recover_from_restart` handles every research_cleanup
    // row synchronously at boot, before this periodic loop's first tick, by
    // either resuming it (the cleanup tail releases the folder before it
    // terminalizes the row, so the row never ages into the purge) or capping
    // it (the cap branch releases the folder in-process and leaves the row to
    // age out HERE, folder already gone). No folder is orphaned by the purge.
    let tx = conn.begin_tx().await?;
    let mut deleted = 0usize;
    for id in &purge_ids {
        if !rollback_ok && ticket_stage_ids.contains(id) {
            continue;
        }
        delete_job_tx(&tx, id).await?;
        deleted += 1;
    }
    tx.commit().await?;
    if deleted > 0 {
        info!(deleted, "Purged stale jobs");
    }
    Ok(deleted as u64)
}

/// 5 of the 6 RESET_TRANSITIONS: InDiagnostics has no job row — its tickets
/// are never stranded by the purge. Transitory handoffs need no rollback —
/// the phase CAS no-ops.
fn rollback_transition(phase: &str) -> Option<(String, bool)> {
    let from = phase.parse::<crate::board::TicketPhase>().ok()?;
    if from == crate::board::TicketPhase::InDiagnostics {
        return None;
    }
    crate::board::BoardStore::reset_transition(from)
        .map(|(to, pipeline_reservation)| (to.to_string(), pipeline_reservation))
}

/// Roll back stranded tickets in place: phase + assigned_to = NULL +
/// updated_at + pipeline_reservation. Guarded by round (only the LATEST
/// round for the row's (ticket_id, stage) rolls back — pre-computed on the
/// sessions conn) AND phase CAS (the ticket's current phase must equal the
/// job's dispatched-for phase). Returns whether the board tx committed (false
/// → purge keeps ticket_stage job rows so the next tick retries the CAS).
async fn rollback_stranded_tickets(rollbacks: &[(String, String, bool)]) -> bool {
    let board = crate::board::BOARD.get().expect("BOARD initialized");
    let tx = match board.conn.begin_tx().await {
        Ok(tx) => tx,
        Err(e) => {
            warn!(error = %e, "Purge rollback: failed to begin board tx");
            return false;
        }
    };
    let now = crate::turso::now();
    let mut rolled_back = 0usize;
    for (ticket_id, phase, is_latest) in rollbacks {
        let Some((to, pipeline_reservation)) = rollback_transition(phase) else {
            continue;
        };
        // Round guard: only the LATEST round for the row's (ticket_id, stage)
        // rolls back.
        if !is_latest {
            debug!(ticket = %ticket_id, "Purge rollback skipped (not latest round)");
            continue;
        }
        // Phase CAS: the ticket's current phase must equal the dispatched-for
        // phase — a moved ticket is not rolled back.
        let updated = tx
            .execute(
                "UPDATE tickets SET phase = ?1, assigned_to = NULL, updated_at = ?2, \
                 pipeline_reservation = ?4 \
                 WHERE id = ?3 AND phase = ?5",
                crate::turso::params![
                    to,
                    now.clone(),
                    ticket_id.clone(),
                    i64::from(pipeline_reservation),
                    phase.clone(),
                ],
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
/// reset_inflight_tickets. Order: (0) replay pending_jobs; (1) ticket_stage
/// scan → resumed-ticket exclusion set; (2) reworked reset_inflight_tickets
/// with NOT IN exclusion; (3) research; (4) analyze.
///
/// Every resumed job gets updated_at = now (the boot bump — the ONLY
/// protection for the pre-first-commit window; Session::init with an empty
/// message skips the last_activity upsert) and retry_count + 1 (the
/// MAX_BOOT_REDISPATCH cap counts boot-resume attempts; in-job retries are
/// not counted).
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub(crate) async fn recover_from_restart() -> Result<Vec<ResumableStage>> {
    let start = std::time::Instant::now();
    let conn = &crate::session::store().conn;

    // (0) Deliverable replay — before anything that could purge.
    let replayed = replay_pending_jobs(conn).await?;

    // (1) ticket_stage scan → exclusion set + phase-left completions.
    let jobs = list_active_jobs(conn).await?;
    let mut exclusion: Vec<String> = Vec::new();
    let mut to_complete: Vec<String> = Vec::new();
    let mut resumable: Vec<ResumableStage> = Vec::new();
    // In-phase candidates with their round, deduped below per (ticket_id,
    // stage) keeping the NEWEST round: a crash → boot resume → manual phase
    // move → re-claim can leave two launched rounds for one stage, and both
    // would phase-match at the next boot (double dispatch) without this.
    let mut candidates: Vec<StageCandidate> = Vec::new();
    for job in &jobs {
        if job.kind != "ticket_stage" {
            continue;
        }
        // Phase-check against the ticket: in expected phase → resume (add to
        // exclusion); moved/cancelled/done → stale → mark done at boot.
        let stage_rows = conn
            .query(
                "SELECT ticket_id, stage, phase, round FROM ticket_stage_jobs WHERE id = ?1",
                params![job.id.clone()],
            )
            .await
            .unwrap_or_default();
        let Some(row) = stage_rows.first() else {
            // ticket_stage_jobs row missing (e.g. partially-spawned) — leave
            // for purge; the ticket resets normally.
            continue;
        };
        let ticket_id: String = row.get(0).expect("ticket_stage_jobs.ticket_id is NOT NULL");
        let stage: String = row.get(1).expect("ticket_stage_jobs.stage is NOT NULL");
        let phase: String = row.get(2).expect("ticket_stage_jobs.phase is NOT NULL");
        let round: i64 = row.get(3).expect("ticket_stage_jobs.round is NOT NULL");
        // Boot re-dispatch cap (design pin: retry_count feeding escalation).
        // Exceeding the cap marks the job DONE — NOT failed: a failed row
        // stays in list_active_jobs (status != 'done'), gets its updated_at
        // refreshed at every boot, and is therefore never purge-eligible,
        // while its roster rows keep TTL-protecting the round's stale
        // sessions forever. done + roster delete (complete_ticket_stage_job)
        // makes the row purge-eligible and releases the sessions; the ticket
        // then resets normally via reset_inflight_tickets (it was never
        // excluded) and re-claims via the poll loop with fresh escalation.
        if job.retry_count >= MAX_BOOT_REDISPATCH {
            warn!(
                job = %job.id,
                ticket = %ticket_id,
                retry_count = job.retry_count,
                "Ticket-stage job exceeded boot re-dispatch cap — marking done",
            );
            let _ = complete_ticket_stage_job(conn, &job.id).await;
            continue;
        }
        // Bump updated_at for every resumed job (the boot bump) AND increment
        // retry_count (the cap counts boot-resume attempts, not in-job
        // retries — every bump here is one resume).
        let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
        let in_phase = crate::board::store()
            .get_ticket_phase(&ticket_id)
            .await
            .is_ok_and(|p| p.is_some_and(|ph| ph.as_ref() == phase.as_str()));
        if in_phase {
            candidates.push(StageCandidate {
                job_id: job.id.clone(),
                ticket_id,
                stage,
                workspace_name: job.workspace_name.clone(),
                round,
            });
        } else {
            to_complete.push(job.id.clone());
        }
    }

    // Dedupe in-phase candidates per (ticket_id, stage) keeping the NEWEST
    // round — superseded older rounds are marked done (their tickets are
    // covered by the kept round's resume). Equal-round ties keep the
    // first-encountered (older-created) candidate; both are the same round,
    // either is safe to supersede.
    let mut best: std::collections::HashMap<(String, String), (i64, usize)> =
        std::collections::HashMap::new();
    for (idx, cand) in candidates.iter().enumerate() {
        let key = (cand.ticket_id.clone(), cand.stage.clone());
        match best.get(&key) {
            Some(&(best_round, _)) if best_round >= cand.round => {
                to_complete.push(cand.job_id.clone());
            }
            _ => {
                if let Some(&(_, prev_idx)) = best.get(&key) {
                    to_complete.push(candidates[prev_idx].job_id.clone());
                }
                best.insert(key, (cand.round, idx));
            }
        }
    }
    for (_round, idx) in best.values() {
        let cand = &candidates[*idx];
        exclusion.push(cand.ticket_id.clone());
        resumable.push(ResumableStage::TicketStage {
            job_id: cand.job_id.clone(),
            ticket_id: cand.ticket_id.clone(),
            stage: cand.stage.clone(),
            workspace_name: cand.workspace_name.clone(),
        });
    }

    // (2) reset_inflight_tickets with the exclusion (empty exclusion omits
    // the NOT IN clause).
    if let Some(board) = crate::board::BOARD.get()
        && let Err(e) = board.reset_inflight_tickets(&exclusion).await
    {
        warn!(error = %e, "Failed to reset in-flight tickets");
    }

    // Mark phase-left jobs done (one UPDATE each — closes the "left launched
    // until purge" gap; reset handles the ticket normally).
    for id in &to_complete {
        if let Err(e) = complete_ticket_stage_job(conn, id).await {
            warn!(job = %id, error = %e, "Failed to mark stale ticket_stage job done");
        }
    }
    // (3)/(4) research/analyze scans.
    let mut resumed_other = 0usize;
    for job in &jobs {
        match job.kind.as_str() {
            "research" | "analyze" => {
                // Resume at the roster/state level (dispatch re-enters the
                // orchestrator with the stored task; retry_count caps at
                // MAX_BOOT_REDISPATCH).
                let kind = job.kind.as_str();
                let capped = job.retry_count >= MAX_BOOT_REDISPATCH;
                if capped {
                    // The envelope is the caller's ONLY result path — never
                    // strand it: deliver a partial report (research) or
                    // failure envelope (analyze) from the checkpointed state.
                    let msg = if kind == "research" {
                        "Research job exceeded boot re-dispatch cap — delivering partial report"
                    } else {
                        "Analyze job exceeded boot re-dispatch cap — delivering failure envelope"
                    };
                    warn!(job = %job.id, kind = %kind, "{msg}");
                }
                // Capped resumes keep retry_count (the cap already fired);
                // normal resumes bump it. The variant mirrors the DB kind
                // (capped = old "{kind}_capped" suffix) — the match arm above
                // already discriminated research vs analyze.
                let _ = checkpoint_job(
                    conn,
                    &job.id,
                    if capped {
                        job.retry_count
                    } else {
                        job.retry_count + 1
                    },
                )
                .await;
                resumable.push(if kind == "research" {
                    ResumableStage::Research {
                        job_id: job.id.clone(),
                        workspace_name: job.workspace_name.clone(),
                        capped,
                    }
                } else {
                    ResumableStage::Analyze {
                        job_id: job.id.clone(),
                        workspace_name: job.workspace_name.clone(),
                        capped,
                    }
                });
                resumed_other += 1;
            }
            "research_cleanup" => {
                // A research-run cleanup Sanitation agent interrupted by a
                // crash. Resume it like any other durable job; on cap the row
                // is NOT resumed — the run folder is released IN-PROCESS
                // (recover_from_restart runs before any agent spawns, so
                // nothing is reading the folder) and the row is left to age
                // into the purge (8h from its last checkpoint), which deletes
                // the row. Any path that removes a research_cleanup row must
                // release the run folder in the same operation. Deliberately
                // no checkpoint bump on cap: that would refresh updated_at and
                // keep the row purge-immune forever (a capped cleanup is not
                // worth an LLM round every boot).
                if job.retry_count >= MAX_BOOT_REDISPATCH {
                    warn!(
                        job = %job.id,
                        "Research cleanup job exceeded boot re-dispatch cap — run folder released in-process, row left for the purge",
                    );
                    crate::research_cleanup::release_run_folder(&job.id).await;
                    continue;
                }
                let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
                resumable.push(ResumableStage::ResearchCleanup {
                    job_id: job.id.clone(),
                    workspace_name: job.workspace_name.clone(),
                });
                resumed_other += 1;
            }
            "ticket_stage" => {
                // Owned by scan (1) above: resume + exclusion set + boot bump
                // already happened there. Silently accept the canonical kind
                // here so it never trips the catch-all below (the first loop
                // resumes exactly the same job rows this loop iterates).
            }
            "temp_cleanup" => {
                // A periodic OS temp-dir cleaner interrupted by a crash.
                // Fire-and-forget: the cleanup is best-effort and the next
                // scheduled pass re-runs it, so a leftover row is
                // terminalized (never resumed). The cleaner's workspace is a
                // synthetic ephemeral name that is never registered in
                // workspaces.db — resuming would hit run_management's
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

    let elapsed = start.elapsed();
    info!(
        duration_ms = elapsed.as_millis() as u64,
        resumed_tickets = resumable.len(),
        resumed_other,
        replayed_pending = replayed,
        "Boot recovery scan complete",
    );

    // Boot sweep for engineer anchors (S5 terminal deletion): anchors for
    // tickets already in a terminal phase are removed so the TTL guard stops
    // protecting their accumulated sessions (the 5-min archive loop re-runs
    // this — boot just catches up after downtime).
    let _ = purge_terminal_engineer_anchors().await;

    Ok(resumable)
}

// ── Engineer anchor (S5 — permanently-NULL seat) ────────────────────────

/// Deterministic engineer anchor agent ID: the per-ticket accumulated
/// session `ticket_{ticket_id}_engineer`, permanently job_id = NULL.
#[must_use]
pub fn engineer_anchor_id(ticket_id: &str) -> String {
    crate::session::ticket_agent_id(ticket_id, crate::Role::Engineer.as_str())
}

/// Upsert the engineer anchor. NEVER sets job_id (setting it removes the row
/// from the partial-index scope → later NULL insert no longer conflicts →
/// duplicate anchor). The DDL and this UPSERT use the IDENTICAL syntactic
/// WHERE form (`job_id IS NULL` on both sides).
pub(crate) async fn upsert_engineer_anchor(
    conn: &Connection,
    ticket_id: &str,
    task: &str,
    status: RowStatus,
) -> Result<()> {
    let anchor_id = engineer_anchor_id(ticket_id);
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
pub fn engineer_anchor_ticket_id(agent_id: &str) -> Option<String> {
    agent_id
        .strip_prefix("ticket_")
        .and_then(|rest| rest.strip_suffix("_engineer"))
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

/// Terminal deletion of engineer anchors (S5): anchors are permanently-NULL
/// seats protecting the accumulated engineer session across bounce rounds.
/// Once the ticket reaches a terminal phase (Done/Cancelled/Failed — no
/// future bounces), the anchor is removed so the TTL guard stops protecting
/// the session (cleaned ≤8h later). Centralized here (5-min archive loop +
/// boot sweep) because the terminal paths are scattered (engineer failure,
/// verifier all-failed, sanitation failure, dispatch panic, supersede, user
/// cancel). Idempotent: parse ticket_id from the anchor agent_id,
/// phase-check on the board side (cross-DB — no join possible), delete.
pub(crate) async fn purge_terminal_engineer_anchors() -> usize {
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
    let board_ready = crate::board::BOARD.get().is_some();
    for row in &rows {
        let Ok(agent_id) = row.get::<String>(0) else {
            continue;
        };
        let Some(ticket_id) = engineer_anchor_ticket_id(&agent_id) else {
            continue;
        };
        let terminal = board_ready
            && crate::board::store()
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
    use crate::turso::params;

    // ── S5 anchor spike: turso 0.7.2 must accept the
    // partial-index UPSERT with identical syntactic WHERE on DDL + UPSERT.
    #[tokio::test]
    async fn anchor_upsert_partial_index_semantics() {
        let (store, _tmp) = crate::open_test_store!(crate::session::SessionStore, "session");
        let conn = &store.conn;

        let anchor_id = engineer_anchor_id("t-1400");

        // First dispatch: anchor insert (NULL job_id).
        upsert_engineer_anchor(conn, "t-1400", "task-1", RowStatus::Launched)
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
        upsert_engineer_anchor(conn, "t-1400", "task-2", RowStatus::Launched)
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
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, created_at, updated_at) \
             VALUES ('j1', 'ticket_stage', 'launched', '', 'ws', 'engineer', ?1, ?1)",
            params![turso::now()],
        )
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
                params![engineer_anchor_id("t-1400")],
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
                params![engineer_anchor_id("t-1400")],
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

    #[test]
    fn engineer_anchor_id_roundtrip() {
        assert_eq!(
            engineer_anchor_ticket_id(&engineer_anchor_id("mahbot-1400")).as_deref(),
            Some("mahbot-1400")
        );
        // Underscores in the ticket id are valid (workspace names allow them) —
        // only the final `_engineer` suffix is stripped.
        assert_eq!(
            engineer_anchor_ticket_id("ticket_my_ws-42_engineer").as_deref(),
            Some("my_ws-42")
        );
        // Wrong role suffix / wrong prefix → None.
        assert!(engineer_anchor_ticket_id("ticket_1400_reviewer").is_none());
        assert!(engineer_anchor_ticket_id("engineer_1400").is_none());
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
            kind: JobKind::AnalyzeToolResult,
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
    #[allow(clippy::await_holding_lock)] // deliberate: retry_tests_lock() serializes process-global test seams
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
                turso::now()
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
                    kind: JobKind::UserMessage,
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
                turso::now()
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
                    kind: JobKind::UserMessage,
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
        let _ = crate::message_router::init_global();
        let (store, _tmp) = crate::open_test_store!(crate::session::SessionStore, "session");
        let conn = &store.conn;
        let agent_id = "f1_replay_target";
        let mut rx = crate::message_router::register_agent(agent_id);
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
                    kind: JobKind::AnalyzeToolResult,
                    role: crate::Role::Manager,
                    reply_target: None,
                    pending_job_id: None,
                })
                .unwrap(),
                turso::now(),
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
        crate::message_router::unregister_agent(agent_id);
    }

    // ── TTL guard ────────────────────────────────────────────────────

    /// The agents table IS the marker: a transient session referenced by an
    /// agents row is NEVER purged by cleanup_old_transient_sessions.
    #[tokio::test]
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
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, created_at, updated_at) \
             VALUES ('jttl', 'ticket_stage', 'launched', '', 'ws', 'engineer', ?1, ?1)",
            params![stale.clone()],
        )
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
    /// Serialized with the reset_inflight_tickets tests (shared global board —
    /// a concurrent boot reset would clobber the fixture phases).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn purge_rolls_back_stranded_ticket() {
        crate::util::test::init_test_stores().await;
        let store = crate::session::store();
        let conn = &store.conn;
        let board = crate::board::store();
        // Ticket in InDevelopment (dispatched-for phase of the stale job).
        let ws = crate::workspace::test_ws_named("/tmp/purge_ws", "purge_ws");
        let ticket_id = crate::util::test::make_ticket(
            board,
            &ws,
            "Purge me",
            crate::board::TicketPhase::InDevelopment,
        )
        .await;
        // Stale job (updated_at long ago) with a stale session.
        let stale = (chrono::Utc::now() - chrono::Duration::hours(20)).to_rfc3339();
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, created_at, updated_at) \
             VALUES ('jstale', 'ticket_stage', 'launched', '', 'purge_ws', 'engineer', ?1, ?1)",
            params![stale.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
             VALUES ('jstale', ?1, 'engineer', 'in_development', 1)",
            params![ticket_id.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO agents (job_id, agent_id, kind, status, task) \
             VALUES ('jstale', 'ticket_purge-t1_engineer', 'engineer', 'launched', '')",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO session_metadata (agent_id, last_activity) \
             VALUES ('ticket_purge-t1_engineer', ?1)",
            params![stale],
        )
        .await
        .unwrap();
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::hours(PURGE_CUTOFF_HOURS)).to_rfc3339();
        let purged = purge_stale_jobs(&cutoff).await.unwrap();
        assert!(purged >= 1, "the stale job must be purged");
        // Ticket rolled back to ReadyForDevelopment with pipeline reservation.
        let t = board.get_ticket(&ticket_id).await.unwrap().unwrap();
        assert_eq!(t.phase, crate::board::TicketPhase::ReadyForDevelopment);
        assert!(t.assigned_to.is_none());
        // Job row gone (scoped to the purged id — the shared test DB holds
        // other tests' rows).
        let n = conn
            .query("SELECT COUNT(*) FROM jobs WHERE id = 'jstale'", ())
            .await
            .unwrap();
        assert_eq!(n[0].get::<i64>(0).unwrap(), 0);
        // The workspace var keeps the temp dir alive for the ticket's path.
        let _ = &ws;
    }

    /// rollback_transition mirrors the 5 job-bearing RESET_TRANSITIONS entries
    /// and rejects InDiagnostics (job-less), unmapped, and invalid phases.
    #[test]
    fn rollback_transition_covers_reset_table() {
        let cases: &[(&str, &str, bool)] = &[
            ("analysis", "backlog", false),
            ("in_development", "ready_for_development", true),
            ("in_review", "diagnostics_done", false),
            ("in_qa", "reviewed", false),
            ("in_sanitation", "qa_passed", true),
        ];
        for (phase, to, reservation) in cases {
            assert_eq!(
                rollback_transition(phase),
                Some((to.to_string(), *reservation))
            );
        }
        for phase in ["in_diagnostics", "ready_for_development", "bogus"] {
            assert_eq!(rollback_transition(phase), None);
        }
    }

    /// The round guard: only the LATEST round for the row's (ticket_id,
    /// stage) rolls back. A stale earlier-round job of the SAME stage's
    /// ticket stays untouched (per-stage scoping — an older round of a
    /// DIFFERENT stage must not suppress the rollback; that case is covered
    /// by [`purge_round_guard_scopes_per_stage`]).
    ///
    /// Serialized with the reset_inflight_tickets tests (shared global board —
    /// a concurrent boot reset would clobber the fixture phases).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn purge_round_guard_skips_older_round() {
        crate::util::test::init_test_stores().await;
        let conn = &crate::session::store().conn;
        let board = crate::board::store();
        let ws = crate::workspace::test_ws_named("/tmp/purge_ws2", "purge_ws2");
        let ticket_id = crate::util::test::make_ticket(
            board,
            &ws,
            "Round guard",
            crate::board::TicketPhase::InReview,
        )
        .await;
        let stale = (chrono::Utc::now() - chrono::Duration::hours(20)).to_rfc3339();
        // Round 1 (stale) + round 2 (fresh — not purged).
        for (id, round) in [("jr1", 1i64), ("jr2", 2i64)] {
            let ts = if round == 1 {
                stale.clone()
            } else {
                turso::now()
            };
            conn.execute(
                "INSERT INTO jobs (id, kind, status, task, workspace_name, role, created_at, updated_at) \
                 VALUES (?1, 'ticket_stage', 'launched', '', 'purge_ws2', 'reviewer', ?2, ?2)",
                params![id, ts.clone()],
            )
            .await
            .unwrap();
            conn.execute(
                "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
                 VALUES (?1, ?2, 'review', 'in_review', ?3)",
                params![id, ticket_id.clone(), round],
            )
            .await
            .unwrap();
        }
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::hours(PURGE_CUTOFF_HOURS)).to_rfc3339();
        purge_stale_jobs(&cutoff).await.unwrap();
        let t = board.get_ticket(&ticket_id).await.unwrap().unwrap();
        assert_eq!(
            t.phase,
            crate::board::TicketPhase::InReview,
            "older round's purge must not roll back the ticket (round guard)"
        );
    }

    /// The round guard scopes per (ticket_id, stage): a stale review round-1
    /// row must NOT be suppressed by the ticket's older COMPLETED analysis
    /// round-2 row — the review row rolls the ticket back out of the blocking
    /// phase (the bug: per-ticket MAX = 2 made the review row look non-latest,
    /// the row was deleted anyway, and the ticket stranded in in_review).
    ///
    /// The purge selects BOTH rows (a done row with old updated_at is
    /// selected too): the analysis row's own rollback attempt no-ops via the
    /// phase CAS mismatch (the ticket is in in_review, not analysis), so only
    /// the review row's rollback lands.
    ///
    /// Serialized with the reset_inflight_tickets tests (shared global board —
    /// a concurrent boot reset would clobber the fixture phases).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn purge_round_guard_scopes_per_stage() {
        crate::util::test::init_test_stores().await;
        let conn = &crate::session::store().conn;
        let board = crate::board::store();
        let ws = crate::workspace::test_ws_named("/tmp/purge_ws4", "purge_ws4");
        let ticket_id = crate::util::test::make_ticket(
            board,
            &ws,
            "Per-stage round guard",
            crate::board::TicketPhase::InReview,
        )
        .await;
        let stale = (chrono::Utc::now() - chrono::Duration::hours(20)).to_rfc3339();
        // Analysis round 2 (done, old updated_at — selected by the purge too)
        // + review round 1 (stale — the row that must trigger the rollback).
        for (id, stage, phase, role, round) in [
            ("jsa2", "analysis", "analysis", "analyst", 2i64),
            ("jsr1", "review", "in_review", "reviewer", 1i64),
        ] {
            conn.execute(
                "INSERT INTO jobs (id, kind, status, task, workspace_name, role, created_at, updated_at) \
                 VALUES (?1, 'ticket_stage', 'launched', '', 'purge_ws4', ?2, ?3, ?3)",
                params![id, role, stale.clone()],
            )
            .await
            .unwrap();
            conn.execute(
                "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![id, ticket_id.clone(), stage, phase, round],
            )
            .await
            .unwrap();
        }
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::hours(PURGE_CUTOFF_HOURS)).to_rfc3339();
        purge_stale_jobs(&cutoff).await.unwrap();
        // The review round-1 row is the ticket's latest review round → the
        // ticket rolls back out of the blocking phase.
        let t = board.get_ticket(&ticket_id).await.unwrap().unwrap();
        assert_eq!(
            t.phase,
            crate::board::TicketPhase::DiagnosticsDone,
            "stale review round must roll the ticket back out of in_review"
        );
        // Both stale rows were purged (done rows with old updated_at are
        // selected too — the analysis row's rollback attempt no-ops via the
        // phase CAS mismatch, but the row is still deleted).
        for id in ["jsa2", "jsr1"] {
            let n = conn
                .query("SELECT COUNT(*) FROM jobs WHERE id = ?1", params![id])
                .await
                .unwrap();
            assert_eq!(n[0].get::<i64>(0).unwrap(), 0, "{id} must be purged");
        }
    }

    /// The rollback-failure path: when the board rollback cannot land, the
    /// ticket_stage job rows are KEPT (next tick retries the CAS) while
    /// non-ticket_stage stale rows are still purged. A held raw board
    /// transaction makes the rollback's begin_tx fail deterministically
    /// ("cannot start a transaction within a transaction").
    ///
    /// Serialized with the other reset_inflight_tickets tests (shared global
    /// board — the raw tx would clobber concurrent fixtures).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn purge_keeps_ticket_stage_rows_when_rollback_fails() {
        crate::util::test::init_test_stores().await;
        let conn = &crate::session::store().conn;
        let stale = (chrono::Utc::now() - chrono::Duration::hours(20)).to_rfc3339();
        // Stale ticket_stage job (must be retained) + stale analyze job (purged).
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, created_at, updated_at) \
             VALUES ('jfail_ts', 'ticket_stage', 'launched', '', 'purge_ws3', 'engineer', ?1, ?1)",
            params![stale.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
             VALUES ('jfail_ts', 'ticket-missing', 'engineer', 'in_development', 1)",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, created_at, updated_at) \
             VALUES ('jfail_analyze', 'analyze', 'launched', '', 'purge_ws3', 'assistant', ?1, ?1)",
            params![stale.clone()],
        )
        .await
        .unwrap();
        // Raw BEGIN on the board connection (bypasses the wrapper's tx
        // tracking) → the rollback's begin_tx fails → rollback_ok = false.
        crate::board::store()
            .conn
            .execute("BEGIN", ())
            .await
            .expect("raw board BEGIN");
        let cutoff =
            (chrono::Utc::now() - chrono::Duration::hours(PURGE_CUTOFF_HOURS)).to_rfc3339();
        purge_stale_jobs(&cutoff).await.unwrap();
        // Restore the board connection for the next serial test.
        crate::board::store()
            .conn
            .execute("ROLLBACK", ())
            .await
            .expect("restore raw board tx");
        let ts_left = conn
            .query("SELECT COUNT(*) FROM jobs WHERE id = 'jfail_ts'", ())
            .await
            .unwrap();
        assert_eq!(
            ts_left[0].get::<i64>(0).unwrap(),
            1,
            "ticket_stage row must survive a failed rollback (next tick retries)"
        );
        let analyze_left = conn
            .query("SELECT COUNT(*) FROM jobs WHERE id = 'jfail_analyze'", ())
            .await
            .unwrap();
        assert_eq!(
            analyze_left[0].get::<i64>(0).unwrap(),
            0,
            "non-ticket_stage stale rows are purged regardless of the rollback"
        );
    }

    // ── Boot classification / reset exclusion ────────────────────────

    /// reset_inflight_tickets skips excluded (resumed) tickets; everything
    /// else resets as before. Serialized with the other reset_inflight_tickets
    /// tests (shared global board — a concurrent reset would clobber the
    /// fixtures).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn reset_inflight_tickets_exclusion() {
        crate::util::test::init_test_stores().await;
        let board = crate::board::store();
        let ws = crate::workspace::test_ws_named("/tmp/ws_x", "ws_x");
        let resumed = crate::util::test::make_ticket(
            board,
            &ws,
            "resume me",
            crate::board::TicketPhase::InDevelopment,
        )
        .await;
        let reset = crate::util::test::make_ticket(
            board,
            &ws,
            "reset me",
            crate::board::TicketPhase::InDevelopment,
        )
        .await;
        board
            .reset_inflight_tickets(&[resumed.clone()])
            .await
            .unwrap();
        let t_resumed = board.get_ticket(&resumed).await.unwrap().unwrap();
        let t_reset = board.get_ticket(&reset).await.unwrap().unwrap();
        assert_eq!(t_resumed.phase, crate::board::TicketPhase::InDevelopment);
        assert_eq!(
            t_reset.phase,
            crate::board::TicketPhase::ReadyForDevelopment
        );
    }

    /// The boot scan adds an in-phase ticket_stage job to the resumed-ticket
    /// exclusion set (its ticket must NOT reset — re-claiming while the
    /// resumed agent runs would duplicate work) and bumps retry_count (the
    /// MAX_BOOT_REDISPATCH cap counts boot resumes). Tickets without a
    /// resumable job reset normally. Serialized with the other
    /// reset_inflight_tickets tests (shared global board).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn recover_from_restart_excludes_resumed_tickets() {
        // Router init: recover_from_restart replays pending_jobs, which can
        // route envelopes (panic if ROUTER is uninitialized).
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let board = crate::board::store();
        let ws = crate::workspace::test_ws_named("/tmp/ws_scan", "ws_scan");
        let resumed_ticket = crate::util::test::make_ticket(
            board,
            &ws,
            "resumed",
            crate::board::TicketPhase::InDevelopment,
        )
        .await;
        let reset_ticket = crate::util::test::make_ticket(
            board,
            &ws,
            "reset",
            crate::board::TicketPhase::InDevelopment,
        )
        .await;
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
             created_at, updated_at) \
             VALUES ('jscan', 'ticket_stage', 'launched', '', 'ws_scan', 'engineer', 0, ?1, ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
             VALUES ('jscan', ?1, 'engineer', 'in_development', 1)",
            params![resumed_ticket.clone()],
        )
        .await
        .unwrap();

        let resumable = recover_from_restart().await.unwrap();
        assert!(
            resumable.iter().any(|r| matches!(
                r,
                ResumableStage::TicketStage { job_id, .. } if job_id.as_str() == "jscan"
            )),
            "the in-phase ticket_stage job must be selected for resume"
        );
        assert!(resumable.iter().any(|r| matches!(
            r,
            ResumableStage::TicketStage { ticket_id, .. } if ticket_id == &resumed_ticket
        )));
        let t1 = board.get_ticket(&resumed_ticket).await.unwrap().unwrap();
        assert_eq!(
            t1.phase,
            crate::board::TicketPhase::InDevelopment,
            "excluded ticket must NOT reset at boot"
        );
        let t2 = board.get_ticket(&reset_ticket).await.unwrap().unwrap();
        assert_eq!(
            t2.phase,
            crate::board::TicketPhase::ReadyForDevelopment,
            "unexcluded ticket resets normally"
        );
        // The boot bump incremented retry_count (the cap counts boot resumes).
        let rc = conn
            .query_row("SELECT retry_count FROM jobs WHERE id = 'jscan'", (), |r| {
                r.get::<i64>(0)
            })
            .await
            .unwrap();
        assert_eq!(rc, 1, "retry_count must be bumped per boot resume");
        // The boot bump also refreshed updated_at (the only protection for the
        // pre-first-commit window against the 8h purge).
        let bumped = conn
            .query_row("SELECT updated_at FROM jobs WHERE id = 'jscan'", (), |r| {
                r.get::<String>(0)
            })
            .await
            .unwrap();
        let before = crate::turso::parse_utc_timestamp(&now).unwrap();
        let after = crate::turso::parse_utc_timestamp(&bumped).unwrap();
        assert!(after > before, "boot bump must refresh updated_at");
    }

    /// A research_cleanup row that already hit the boot re-dispatch cap must
    /// NOT be resumed: the resume would spend one LLM round per boot, and the
    /// checkpoint bump would keep the row purge-immune forever (updated_at
    /// refreshed each boot), holding the run folder indefinitely. On cap the
    /// run folder is released IN-PROCESS (no sweep exists anymore) and the
    /// row is left to age into the 8h purge.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // recover_from_restart resets the shared global board
    async fn recover_from_restart_caps_research_cleanup_releases_folder() {
        // Router init: recover_from_restart replays pending_jobs, which can
        // route envelopes (panic if ROUTER is uninitialized).
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
             created_at, updated_at) \
             VALUES ('jclean', 'research_cleanup', 'launched', '', 'ws_scan', 'sanitation', ?1, ?2, ?2)",
            params![MAX_BOOT_REDISPATCH, now.clone()],
        )
        .await
        .unwrap();
        // Run-folder fixture: what a crash-left cleanup run leaves behind.
        let run_folder = crate::research_cleanup::run_root_path("jclean");
        tokio::fs::create_dir_all(&run_folder).await.unwrap();
        let resumable = recover_from_restart().await.unwrap();
        assert!(
            !resumable.iter().any(|r| matches!(
                r,
                ResumableStage::ResearchCleanup { job_id, .. } if job_id.as_str() == "jclean"
            )),
            "capped cleanup must NOT be selected for resume (left for the purge)"
        );
        assert!(
            !run_folder.exists(),
            "the cap branch releases the run folder in-process (no sweep exists)"
        );
        let updated: String = conn
            .query_row("SELECT updated_at FROM jobs WHERE id = 'jclean'", (), |r| {
                r.get::<String>(0)
            })
            .await
            .unwrap();
        assert_eq!(
            updated, now,
            "capped cleanup must not be checkpointed — updated_at unchanged so the 8h purge can reclaim the row"
        );
    }

    /// A `temp_cleanup` row left over from a previous lifetime (crash
    /// mid-cleaner) must be TERMINALIZED at boot, never resumed: the cleaner
    /// is fire-and-forget, its workspace is a synthetic ephemeral name that
    /// is never registered in workspaces.db, and the next scheduled pass
    /// re-runs the cleanup cleanly.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // recover_from_restart resets the shared global board
    async fn recover_from_restart_terminalizes_temp_cleanup_rows() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
             created_at, updated_at) \
             VALUES ('jtmpclean', 'temp_cleanup', 'launched', '', 'tmp', 'sanitation', 0, ?1, ?1)",
            params![now.clone()],
        )
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
                ResumableStage::ResearchCleanup { job_id, .. } if job_id.as_str() == "jtmpclean"
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
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
             created_at, updated_at) \
             VALUES ('rcln', 'research_cleanup', 'launched', '', 'ws_replay', 'sanitation', 0, ?1, ?1)",
            params![now.clone()],
        )
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
                    kind: JobKind::ResearchResult,
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
        let now = crate::turso::now();
        conn.execute(
            "INSERT INTO pending_jobs (id, target_agent_id, envelope, created_at) \
             VALUES ('rcln2', 'manager_ws_replay2', ?1, ?2)",
            params![
                serde_json::to_string(&AgentJob {
                    content: "<research-result>done</research-result>".to_string(),
                    workspace_name: "ws_replay2".to_string(),
                    user_name: "u".to_string(),
                    channel: "telegram".to_string(),
                    kind: JobKind::ResearchResult,
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
                ResumableStage::ResearchCleanup { job_id, .. } if job_id.as_str() == "rcln2"
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
    /// absence is the completed marker on the replay path, and re-dispatching
    /// would cost one LLM round per boot until the 7-day pending cap. The
    /// envelope itself still replays.
    #[tokio::test]
    #[serial_test::serial(reset_inflight)] // shared process-lifetime store rows
    async fn replay_skips_cleanup_dispatch_when_run_folder_gone() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        crate::util::test::create_test_workspace("/tmp/test_ws_replay_archived", "ws_replay3")
            .await;
        let now = crate::turso::now();
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
                    kind: JobKind::ResearchResult,
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

    /// Boot classification: launched jobs whose ticket left the
    /// expected phase are marked done at boot (closing the "left launched
    /// until purge" gap — reset handles the ticket); done jobs are filtered
    /// out entirely (never re-selected); concurrent launched rounds for the
    /// same (ticket, stage) dedupe to the NEWEST round (the superseded older
    /// round is marked done — both would phase-match otherwise → double
    /// dispatch). Serialized with the other reset_inflight_tickets tests
    /// (shared global board).
    #[tokio::test]
    #[serial_test::serial(reset_inflight)]
    async fn recover_from_restart_classifies_done_phase_left_dedupes_rounds() {
        // Router init: recover_from_restart replays pending_jobs, which can
        // route envelopes (panic if ROUTER is uninitialized).
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let board = crate::board::store();
        let ws = crate::workspace::test_ws_named("/tmp/ws_cls", "ws_cls");
        let now = crate::turso::now();

        // (1) Two launched rounds for one ticket+stage, both phase-matching:
        //     only the NEWEST (round 2) resumes; round 1 is marked done.
        let dup_ticket = crate::util::test::make_ticket(
            board,
            &ws,
            "dup",
            crate::board::TicketPhase::InDevelopment,
        )
        .await;
        for (id, round) in [("jdup1", 1i64), ("jdup2", 2i64)] {
            conn.execute(
                "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
                 created_at, updated_at) \
                 VALUES (?1, 'ticket_stage', 'launched', '', 'ws_cls', 'engineer', 0, ?2, ?2)",
                params![id, now.clone()],
            )
            .await
            .unwrap();
            conn.execute(
                "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
                 VALUES (?1, ?2, 'engineer', 'in_development', ?3)",
                params![id, dup_ticket.clone(), round],
            )
            .await
            .unwrap();
        }

        // (2) Launched but the ticket left the expected phase (job dispatched
        //     for in_review; ticket is in InQa) → stale: marked done at boot.
        let phase_left = crate::util::test::make_ticket(
            board,
            &ws,
            "phase_left",
            crate::board::TicketPhase::InQa,
        )
        .await;
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
             created_at, updated_at) \
             VALUES ('jcls2', 'ticket_stage', 'launched', '', 'ws_cls', 'reviewer', 0, ?1, ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
             VALUES ('jcls2', ?1, 'review', 'in_review', 1)",
            params![phase_left.clone()],
        )
        .await
        .unwrap();

        // (3) Done job → filtered out entirely (never resumed).
        let done_ticket = crate::util::test::make_ticket(
            board,
            &ws,
            "done",
            crate::board::TicketPhase::InDevelopment,
        )
        .await;
        conn.execute(
            "INSERT INTO jobs (id, kind, status, task, workspace_name, role, retry_count, \
             created_at, updated_at) \
             VALUES ('jcls3', 'ticket_stage', 'done', '', 'ws_cls', 'engineer', 0, ?1, ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
             VALUES ('jcls3', ?1, 'engineer', 'in_development', 1)",
            params![done_ticket.clone()],
        )
        .await
        .unwrap();

        let resumable = recover_from_restart().await.unwrap();
        // Only the NEWEST of the concurrent rounds resumes; the superseded
        // older round is marked done at boot.
        assert!(resumable.iter().any(|r| matches!(
            r,
            ResumableStage::TicketStage { job_id, .. } if job_id.as_str() == "jdup2"
        )));
        assert!(!resumable.iter().any(|r| matches!(
            r,
            ResumableStage::TicketStage { job_id, .. } if job_id.as_str() == "jdup1"
        )));
        let status1 = conn
            .query_row("SELECT status FROM jobs WHERE id = 'jdup1'", (), |r| {
                r.get::<String>(0)
            })
            .await
            .unwrap();
        assert_eq!(
            status1, "done",
            "superseded older round must be marked done"
        );
        // Phase-left + done → never resumed; phase-left marked done at boot.
        assert!(!resumable.iter().any(|r| matches!(
            r,
            ResumableStage::TicketStage { job_id, .. }
                if job_id.as_str() == "jcls2" || job_id.as_str() == "jcls3"
        )));
        let status2 = conn
            .query_row("SELECT status FROM jobs WHERE id = 'jcls2'", (), |r| {
                r.get::<String>(0)
            })
            .await
            .unwrap();
        assert_eq!(status2, "done");
        // Tickets: the deduped ticket stays in-phase (excluded); the others
        // reset normally.
        let t_dup = board.get_ticket(&dup_ticket).await.unwrap().unwrap();
        assert_eq!(t_dup.phase, crate::board::TicketPhase::InDevelopment);
        let t2 = board.get_ticket(&phase_left).await.unwrap().unwrap();
        assert_eq!(t2.phase, crate::board::TicketPhase::Reviewed);
        let t3 = board.get_ticket(&done_ticket).await.unwrap().unwrap();
        assert_eq!(t3.phase, crate::board::TicketPhase::ReadyForDevelopment);
    }
}
