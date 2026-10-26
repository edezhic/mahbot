//! Durable jobs layer: jobs/agents/pending_jobs lifecycle, boot recovery
//! scan, and the phase-only stale purge orchestrator.
//!
//! All rows live in the consolidated domain database (`core.db`) behind the
//! one writing connection — the jobs/session/board tables now share a single
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
use crate::pipeline::board::{TicketPhase, WORKING_PHASES};
use anyhow::{Context, Result};
use std::time::Duration;
use tracing::{debug, error, info, warn};

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
/// (`caller_agent_id`); `Async` jobs (research/analyze, research_cleanup,
/// ticket phases, and pin-less implement dispatches) resume at boot, while
/// `temp_cleanup` rows are terminalized instead.
///
/// The string values are intrinsic Rust/SQL coupling: `as_str()`, the SQL
/// literals in [`find_owned_launched_jobs`] / [`abandon_session_jobs`], and
/// migration delta `30`'s backfill CASE must stay in sync on a rename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobMode {
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
///
/// A round whose slots form more than one cohort carries a label per cohort
/// (`Reviewer`/`Tester` in the verification round, `Analyst`/`BlockerVerifier`
/// in analysis) because the resume path reads the cohorts back from the stored
/// roster: the label is the cohort identity, never the slot's position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AgentKind {
    Analyst,
    /// The analysis escalation cohort's blocker-verification slots.
    BlockerVerifier,
    Reviewer,
    Tester,
    Engineer,
    /// Single worker sub-agent for the durable async worker round (`send`'s
    /// `worker` addressee).
    Worker,
    Sanitation,
}

impl AgentKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Analyst => "analyst",
            Self::BlockerVerifier => "blocker_verifier",
            Self::Reviewer => "reviewer",
            Self::Tester => "tester",
            Self::Engineer => "engineer",
            Self::Worker => "worker",
            Self::Sanitation => "sanitation",
        }
    }
}

/// Graceful-shutdown drain cap: in-flight work completes within this window,
/// then stragglers are force-cancelled. Read from outside too: the exit bound a
/// quit arms has to outlast it (`shutdown::macos_quit`).
pub(crate) const DRAIN_CAP_SECS: u64 = 10 * 60;

/// Stale-purge cutoff (hours): only ticket-phase `jobs` rows older than this
/// are purged (they are recoverable from `tickets.phase`). Non-phase launched
/// jobs (researches, cleanups, sync/async envelope kinds) are never
/// time-deleted — explicit abandon only. `pending_jobs` envelopes are never
/// time-purged — at-least-once keeps unconfirmed rows alive (a cancel removing
/// one by id is a deliberate, caller-requested discard, not a purge).
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
    mode: JobMode,
    /// The absolute folder a delegated round runs in; `None` = the caller's own
    /// workspace (`workspace_name`).
    pub exec_dir: Option<String>,
    /// The row's `created_at` — the anchor the caller's resume-completion step
    /// compares against its own frame row to tell a round THIS frame dispatched
    /// from the leftover of an earlier one.
    pub created_at: String,
}

/// A row of the `agents` table. Carries the slot `idx` (in addition to the
/// status/outcome/task) so the slot-resume skeleton can reconstruct a round in
/// place without re-deriving it from live ticket state.
#[derive(Debug, Clone)]
pub(crate) struct AgentRow {
    pub idx: Option<i64>,
    pub agent_id: String,
    /// The row's `agents.kind` label — the roster's cohort identity.
    pub kind: String,
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
        EXEC_DIR        => "exec_dir",
        CREATED_AT      => "created_at",
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
        exec_dir: row.get(COL_JOB_EXEC_DIR)?,
        created_at: row.get(COL_JOB_CREATED_AT)?,
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

crate::columns! {
    AGENT_COLUMNS [AGENT] {
        IDX      => "idx",
        AGENT_ID => "agent_id",
        KIND     => "kind",
        STATUS   => "status",
        OUTCOME  => "outcome",
        TASK     => "task",
    }
}

fn agent_row_from(row: &Row) -> anyhow::Result<AgentRow> {
    Ok(AgentRow {
        idx: row.get(COL_AGENT_IDX)?,
        agent_id: row.get(COL_AGENT_AGENT_ID)?,
        kind: row.get(COL_AGENT_KIND)?,
        status: row.get(COL_AGENT_STATUS)?,
        outcome: row.get(COL_AGENT_OUTCOME)?,
        task: row.get(COL_AGENT_TASK)?,
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
    /// single worker's roster row holds the dispatched task; the job row holds
    /// the caller identity for resume delivery.
    Implement,
    /// research_jobs (id, state).
    Research,
    /// A research-run cleanup Sanitation agent. No child row: the jobs row's
    /// `task` holds the cleanup prompt, and the row id == the run id (folder
    /// name) so the row holds the run folder until the cleanup completes —
    /// the folder is released by the cleanup tail (folder first, row last).
    ResearchCleanup,
    /// A periodic temp cleaner Sanitation agent (see `crate::temp`).
    /// No child row: the jobs row's `task` holds the cleanup prompt. Leftover
    /// rows are terminalized at boot (never resumed) — the cleaner's
    /// ephemeral workspace is never a managed workspace.
    TempCleanup,
    /// A per-phase ticket job: `jobs.kind` = `phase.as_ref()` (the single
    /// state-machine authority), `jobs.ticket_id` = the ticket. One short-lived
    /// job per pipeline phase; the ticket's `phase` is the only durable running
    /// truth. Roster grows during the phase (analysts/reviewers), and the job
    /// is deleted when the phase completes or is reset.
    Phase {
        phase: TicketPhase,
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
            // A phase job is named after its phase (`jobs.kind` mirrors
            // `tickets.phase`, the sole running truth); only the working phases
            // are spawnable, so anything else is a caller bug worth a refusal.
            Self::Phase { phase, .. } => {
                assert!(
                    crate::pipeline::board::WORKING_PHASES.contains(phase),
                    "non-working phase as a job kind: {}",
                    phase.as_ref(),
                );
                <&'static str>::from(*phase)
            }
        }
    }
}

/// `jobs.ticket_id` for a spawn child — `Some` only for ticket phase jobs.
#[must_use]
fn child_ticket_id(child: &SpawnChild) -> Option<&str> {
    match child {
        SpawnChild::Phase { ticket_id, .. } => Some(ticket_id.as_str()),
        _ => None,
    }
}

/// Spawn a job with its pre-generated agent roster in ONE transaction.
/// MUST commit before the agent's first session write. The kind is derived
/// from the child via [`SpawnChild::kind_str`] — closing the drift window of
/// an inconsistent (kind, child) pair. A row for a cancelled analyze round is
/// never committed (see the gate below).
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
    exec_dir: Option<&str>,
    custom_launch: Option<&crate::Launch>,
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
        exec_dir,
        custom_launch,
    )
    .await?;
    // Manual-cancel gate INSIDE the tx (after the INSERT, before the commit),
    // mirroring [`complete_job_with_envelope`] and scoped to the child a cancel
    // action can name: the tx holds the single-writer lock, so a dispatch whose
    // row write races the cancel either observes the already-fired signal here —
    // the row is never committed — or commits first, and the cancel sweep's
    // DELETE then serializes behind this tx and removes the row it finds. A
    // cancellation and the analyze row this dispatch wrote can therefore never
    // both stand, so a cancelled round leaves no row for a start to resume unless
    // the sweep itself failed — and that case was answered to the caller as a
    // cancellation that did not happen. Every other child keeps its dispatch
    // unbounded by cancel signals — a research run's id, for one, is reused by the
    // cleanup row its cancelled-run handoff still has to write.
    if matches!(child, SpawnChild::Analyze) && crate::round_cancel::is_cancelled(id) {
        // Rolled back HERE rather than left to the guard's drop: a cancellation
        // that abandons its own write leaves no complaint about itself behind (the
        // deferred cleanup logs a WARN) and no dangling transaction to retry.
        tracing::info!(job = %id, "Job dispatch rolled back — round cancelled");
        tx.rollback().await?;
        return Ok(());
    }
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
    exec_dir: Option<&str>,
    custom_launch: Option<&crate::Launch>,
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
    // JSON for the custom-role launch the round was dispatched by (`NULL` for
    // every built-in caller) — read back by `job_caller` on boot resume.
    let custom_launch_json = custom_launch
        .map(serde_json::to_string)
        .transpose()
        .context("serialize jobs.custom_launch")?;
    let now = db::now();
    tx.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, user_name, channel, role, \
         ticket_id, retry_count, created_at, updated_at, caller_agent_id, mode, exec_dir, \
         custom_launch) \
         VALUES (?1, ?2, 'launched', ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?9, ?10, ?11, ?12, ?13)",
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
            exec_dir,
            custom_launch_json,
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
async fn checkpoint_job(conn: &Connection, id: &str, retry_count: i64) -> Result<()> {
    // Boot resume re-arms the row unconditionally — status is reset to
    // Launched regardless of its prior value (list_active_jobs takes the
    // non-terminal set).
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
    // observed here and rolls the tx back — the pending row is never
    // committed and the job row survives for the cancel sweep. The
    // sweep's own DELETE then serializes behind this tx (single-writer),
    // so no cancelled run can leave a deliverable pending row. Any round
    // that registered a cancel signal is covered by the gate — job kinds
    // that never register one (a worker round, for instance) keep their
    // unbounded completion. Rolled back HERE rather than by the guard's
    // drop, so a cancelled completion leaves no deferred-cleanup WARN
    // behind either.
    if crate::round_cancel::is_cancelled(job_id) {
        tracing::info!(job = %job_id, "Job completion rolled back — run manually cancelled");
        tx.rollback().await?;
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
/// path), the pending row is simply absent. A cancelled round's in-tx gate rolls
/// the write back and still returns Ok, so the id stays set (the rollback itself
/// failing is the one error); both call sites consult their cancel check before
/// routing and never hand that envelope on.
///
/// `custom`, when present, is the calling custom role's launch — written into
/// the envelope so [`envelope_target`] resolves the custom branch instead of a
/// fabricated built-in address.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn complete_durable_job(
    job_id: &str,
    content: String,
    kind: MessageKind,
    caller_role: Role,
    user_name: &str,
    channel: &str,
    workspace_name: &str,
    custom: Option<crate::Launch>,
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
        originating_workspace: None,
        custom_role: custom,
        reply_to_custom: None,
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

/// Whether a queued `pending_jobs` row still exists — the boot replay's re-check
/// that the envelope it is about to deliver was not discarded since the list was
/// read. A read failure answers `true`: in doubt, delivering is the safe side
/// (the row is replayed again next boot if it was in fact gone).
async fn pending_row_exists(conn: &Connection, id: &str) -> bool {
    conn.query_optional(
        "SELECT 1 FROM pending_jobs WHERE id = ?1",
        params![id],
        |_| Ok::<(), anyhow::Error>(()),
    )
    .await
    .map_or(true, |found| found.is_some())
}

/// The envelope of the result queued for `id`, when one waits — the
/// `pending_jobs` counterpart of the `jobs`-row readers, so no feature module
/// hand-writes this table's shape. An unparsable envelope answers `None`: it
/// addresses nothing a reader can honour.
pub(crate) async fn pending_envelope(conn: &Connection, id: &str) -> Result<Option<AgentJob>> {
    let raw = conn
        .query_optional(
            "SELECT envelope FROM pending_jobs WHERE id = ?1",
            params![id],
            |r| r.get::<String>(0),
        )
        .await
        .context("read a queued result")?;
    let Some(raw) = raw else {
        return Ok(None);
    };
    match serde_json::from_str(&raw) {
        Ok(job) => Ok(Some(job)),
        Err(e) => {
            tracing::debug!(
                pending_job = %id,
                error = %e,
                "Queued envelope unreadable — nothing to honour",
            );
            Ok(None)
        }
    }
}

/// Remove a gone round's own durable state in ONE tx: its queued result row
/// first (the boot replay would otherwise deliver it), then its `jobs` row (a
/// surviving row would be resumed at the next start, and its deletion cascades
/// the roster). Every statement is a no-op on missing rows, so this is
/// idempotent; a failure rolls the whole thing back.
pub(crate) async fn delete_round_rows(conn: &Connection, job_id: &str) -> Result<()> {
    let tx = conn
        .begin_tx()
        .await
        .context("begin the gone round's removal tx")?;
    tx.execute("DELETE FROM pending_jobs WHERE id = ?1", params![job_id])
        .await
        .context("delete the gone round's queued result")?;
    tx.execute("DELETE FROM jobs WHERE id = ?1", params![job_id])
        .await
        .context("delete the gone round's job row")?;
    tx.commit()
        .await
        .context("commit the gone round's removal")?;
    Ok(())
}

// ── Explicit abandon (deletion at the caller's request) ──
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
        None,
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
/// gate on the pool itself: a job is never dropped for being old, only by
/// explicit abandon (the /clear abandon path terminalizes a cleared session's
/// jobs, which is what previously made the fresh-cycle gate necessary). The
/// caller scopes its BINDING to its own dangling frame ([`JobRow::created_at`]
/// against the frame row), so a pool entry that frame did not dispatch is
/// abandoned rather than bound.
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
// ticket's current phase (`analysis`, `in_development`, `verification`,
// `in_sanitation`) and whose `ticket_id` — stored
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
    phase: TicketPhase,
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
/// active-agent marker that replaces `tickets.assigned_to`. `status` is the
/// roster's running signal (`launched` from every caller; a finished agent is
/// recorded by [`write_agent_outcome`] instead). The agent roster is keyed on
/// (job_id, agent_id) so the NULL-seat engineer anchor (job_id=NULL) remains a
/// distinct row for session-TTL continuity.
///
/// The update half is the re-registration of an existing `(job_id, agent_id)`:
/// the agent id is derived from the ticket and role, so the puller re-driving a
/// job it left in place (`dispatch_working_phases`, after `pause_freezing`
/// marked its rows `failed`) registers the same pair again and updates the row
/// instead of inserting it.
pub(crate) async fn upsert_job_agent(
    conn: &Connection,
    job_id: &str,
    agent_id: &str,
    kind: AgentKind,
    status: RowStatus,
) -> Result<()> {
    conn.upsert_row(
        "UPDATE agents SET status = ?1 WHERE job_id = ?2 AND agent_id = ?3",
        || params![status.as_str(), job_id, agent_id],
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
         VALUES (?1, ?2, ?3, NULL, ?4, '') \
         ON CONFLICT(job_id, agent_id) DO NOTHING",
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

// ── Slot-resume skeleton (the pipeline's parallel phase rounds) ──
//
// An interrupted parallel round is resumable from its roster: not-Done slots
// are re-run with their stored per-slot `task`, while the phase body
// reconstructs the Done half from each slot's stored `outcome`.

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

/// The delivery address a round's result is routed to, from the round's own
/// identity parts — the ONE derivation shared by [`envelope_target`] (a
/// completed envelope) and the analyze cancel action's ownership check (a live
/// `jobs` row), so what a cancel treats as the caller's can never drift from
/// where the result would be delivered.
pub(crate) fn target_agent_id(
    custom: Option<&crate::Launch>,
    user_name: &str,
    role: &str,
    workspace_name: &str,
) -> String {
    if let Some(custom) = custom {
        return custom.agent_id.clone();
    }
    crate::session::resolve_agent_id(user_name, role, workspace_name)
}

/// Resolve the consumer-loop agent ID for an envelope (Manager role → the
/// workspace Manager session; pinned Assistant in a personal workspace → that
/// user's assistant session). `session::resolve_agent_id` already branches on
/// `role == "manager"` internally, so this is the single canonical form —
/// analyze/research delivery paths call it instead of reimplementing the branch.
///
/// A custom-role launch's address is not derivable from (user, role, workspace):
/// it is carried explicitly in the envelope's `custom_role`, so it is returned
/// directly when present. Boot replay uses the STORED `target_agent_id` written
/// from this same derivation, so both stay in agreement.
pub(crate) fn envelope_target(job: &AgentJob) -> String {
    target_agent_id(
        job.custom_role.as_ref(),
        &job.user_name,
        job.role.as_str(),
        &job.workspace_name,
    )
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
    /// The custom-role launch the job was dispatched by (`jobs.custom_launch`),
    /// `None` for every built-in caller and for a NULL/unreadable value. A
    /// resume uses it to address the custom branch instead of a fabricated
    /// built-in address. Boxed so the caller payload stays small enough for the
    /// resume-outcome enums.
    pub custom_launch: Option<Box<crate::Launch>>,
}

/// Discriminated resume preamble so sync callers never mistake a gone job
/// for a drain-cut, or a transient DB read error for an abandon.
pub(crate) enum ResumePreamble {
    /// Caller identity + parsed role — the resume may proceed.
    Proceed(JobCaller, Role),
    /// Drain/shutdown active — quiet abort, the job row is untouched.
    DrainCut,
    /// The jobs row is absent — only possible via an explicit abandon (session
    /// /clear or workspace deletion) or a cancel whose own row deletion raced
    /// this resume; no terminalization happens (the row is already gone). The
    /// old terminalize-on-missing compensation existed only because jobs could
    /// be purged.
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
    /// The job row was gone between the ownership lookup and the resume — an
    /// explicit abandon, or a cancel that removed the row while this resume ran.
    Gone,
}

/// Parse a `jobs.custom_launch` value — the ONE policy for every reader of that
/// column: NULL and a corrupted value both mean "no launch", the corruption
/// warned and the row left to the reader's own refusal of that shape. A resume
/// passes it on and is refused downstream, in the consumer loop's custom-role
/// funnel; an ownership check derives the fabricated `…_{role}` address
/// [`envelope_target`] derives for a missing launch, which no live custom/worker
/// session carries (`Launch::agent_id` is always a `role_…`/`rolerun_…`/`worker_…`
/// address), so it refuses the row too.
pub(crate) fn parse_custom_launch(raw: Option<String>) -> Option<crate::Launch> {
    match raw {
        None => None,
        Some(raw) => match serde_json::from_str(&raw) {
            Ok(launch) => Some(launch),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "jobs.custom_launch is unreadable — treating as absent",
                );
                None
            }
        },
    }
}

/// Load the task + caller identity of a job from the `jobs` row alone.
async fn job_caller(conn: &Connection, job_id: &str) -> Result<Option<JobCaller>> {
    conn.query_optional(
        "SELECT task, role, user_name, channel, custom_launch FROM jobs WHERE id = ?1",
        params![job_id],
        |r| {
            Ok::<_, anyhow::Error>(JobCaller {
                task: r.get::<String>(0)?,
                role: r.get::<String>(1)?,
                user_name: r.get::<String>(2)?,
                channel: r.get::<String>(3)?,
                custom_launch: parse_custom_launch(r.get::<Option<String>>(4)?).map(Box::new),
            })
        },
    )
    .await
    .context("load job caller")
}

/// A live analyze round of the caller's: the cancel action needs the row's
/// ownership AND, while the row is still there, the execution directory a
/// cancelled round's synthetic search engine is named after
/// ([`crate::tools::delegation::release_placed_engine`]) — both from one read,
/// because neither may be looked up after the rows are gone.
pub(crate) struct CallerAnalyzeRound {
    /// The round's `place`, when it was dispatched with one.
    pub(crate) exec_dir: Option<String>,
}

/// The analyze round named by `job_id` when it is `caller`'s to stop: its `jobs`
/// row must be an analyze round whose own delivery address ([`target_agent_id`])
/// is the caller's session — the ONE derivation, so what a cancel treats as the
/// caller's can never drift from where the result would be delivered. The
/// `kind = 'analyze'` filter is load-bearing: it refuses the `deep` depth's
/// `research` job and every worker round, however the id was named. An absent row
/// answers `None`, and a row whose `custom_launch` cannot be read is read as
/// launch-less ([`parse_custom_launch`]) — a fabricated address for the
/// custom/worker roles that carry a launch, so another branch's round is never
/// stopped either.
pub(crate) async fn analyze_round_owned_by(
    conn: &Connection,
    job_id: &str,
    caller: &str,
) -> Result<Option<CallerAnalyzeRound>> {
    let row = conn
        .query_optional(
            "SELECT role, user_name, workspace_name, custom_launch, exec_dir FROM jobs \
             WHERE id = ?1 AND kind = 'analyze'",
            params![job_id],
            |r| {
                Ok::<_, anyhow::Error>((
                    target_agent_id(
                        parse_custom_launch(r.get::<Option<String>>(3)?).as_ref(),
                        &r.get::<String>(1)?,
                        &r.get::<String>(0)?,
                        &r.get::<String>(2)?,
                    ),
                    r.get::<Option<String>>(4)?,
                ))
            },
        )
        .await
        .context("read the round's own row")?;
    Ok(row.and_then(|(target, exec_dir)| {
        (target == caller).then_some(CallerAnalyzeRound { exec_dir })
    }))
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
            let mut caller_role =
                std::str::FromStr::from_str(&caller.role).unwrap_or(Role::Manager);
            // A row that carries a launch resumes as its custom branch: the
            // persisted launch is the address authority, never the tool's own
            // construction role.
            caller_role = caller_role.for_launch(caller.custom_launch.as_deref());
            ResumePreamble::Proceed(caller, caller_role)
        }
        // The row is gone: an explicit abandon, or a cancel, raced this resume —
        // report it as such.
        Ok(None) => {
            tracing::info!(job = %job_id, "{missing_site}: job row gone (abandoned or cancelled) — skipping resume");
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

/// `RowStatus::Done` as this module's SQL spells it — `as_str` is a `const fn`, so the terminal
/// status has one spelling here and cannot drift from the enum.
const DONE_STATUS: &str = RowStatus::Done.as_str();

/// Load all non-terminal jobs (launched|failed) for the boot scan.
async fn list_active_jobs(conn: &Connection) -> Result<Vec<JobRow>> {
    let rows = conn
        .query(
            &format!(
                "SELECT {JOB_COLUMNS} FROM jobs \
                 WHERE status != '{DONE_STATUS}' ORDER BY created_at"
            ),
            (),
        )
        .await
        .context("list active jobs")?;
    rows.iter().map(job_row_from).collect()
}

/// Load the agent roster for a job (with each row's cohort label).
pub(crate) async fn list_agents_for_job(conn: &Connection, job_id: &str) -> Result<Vec<AgentRow>> {
    let rows = conn
        .query(
            &format!("SELECT {AGENT_COLUMNS} FROM agents WHERE job_id = ?1 ORDER BY idx"),
            params![job_id],
        )
        .await
        .context("list agents for job")?;
    rows.iter().map(agent_row_from).collect()
}

/// The roster agent ids of every job that has not been terminalized, minus the slots that already
/// finished: the ids a cut round resumes IN PLACE from (the slot-resume here re-dispatches the
/// stored roster's NOT-Done slots — [`RowStatus::Done`] ones are reconstructed from their stored
/// outcomes instead), so the only durable names a stored round can come back under. A caller uses
/// this to tell a run that may still return from one that is over; a purged or terminalized job
/// drops out — the same non-terminal set [`list_active_jobs`] takes — and a done slot's id leaves
/// with its slot, since nothing will be re-dispatched under it again.
pub(crate) async fn resumable_roster_agent_ids(conn: &Connection) -> Result<Vec<String>> {
    let rows = conn
        .query(
            &format!(
                "SELECT DISTINCT a.agent_id FROM agents a \
                 WHERE a.job_id IN (SELECT id FROM jobs WHERE status != '{DONE_STATUS}') \
                 AND a.status != '{DONE_STATUS}'"
            ),
            (),
        )
        .await
        .context("list the roster agent ids of non-terminal jobs")?;
    Ok(rows
        .iter()
        .filter_map(|row| row.get::<String>(0).ok())
        .collect())
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

/// Delete every deferred envelope addressed to `target_agent_id`, returning how
/// many rows were dropped. The deliberate-stop path is the ONE place that drops
/// durable envelopes by address, and it deliberately waives the at-least-once
/// guarantee (a stopped launch must not be replayed at boot); every other
/// deletion stays by row id.
#[expect(clippy::cast_possible_truncation)] // u64 row count fits usize on all supported targets
pub(crate) async fn delete_pending_jobs_for(target_agent_id: &str) -> Result<usize> {
    let conn = &crate::session::store().conn;
    let deleted = conn
        .execute(
            "DELETE FROM pending_jobs WHERE target_agent_id = ?1",
            params![target_agent_id],
        )
        .await
        .context("delete pending jobs for agent")?;
    info!(
        target_agent_id,
        count = deleted,
        "dropped deferred envelopes for a stopped launch"
    );
    Ok(deleted as usize)
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
        /// The recorded target directory for the round; `None` = the caller's
        /// own workspace.
        exec_dir: Option<String>,
    },
    Implement {
        job_id: String,
        workspace_name: String,
        /// The recorded target directory for the round; `None` = the caller's
        /// own workspace.
        exec_dir: Option<String>,
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
        // Execution-time invariant: re-pin pinned roles (Assistant)
        // to the user's personal workspace; a pinned role with an empty user is
        // refused and the poisoned envelope deleted so it is not replayed forever.
        let Some(workspace_name) =
            crate::users::enforce_personal_pinning(job.role, &job.workspace_name, &job.user_name)
        else {
            error!(
                pending_job = %row.id,
                role = %job.role.as_str(),
                workspace = %job.workspace_name,
                "Pending job replay: pinned role with empty user — dropping poisoned envelope"
            );
            if let Err(e) = delete_pending_job(conn, &row.id).await {
                warn!(pending_job = %row.id, error = %e, "Failed to delete refused pending job");
            }
            continue;
        };
        job.workspace_name = workspace_name;
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
        if !pending_row_exists(conn, &row.id).await {
            debug!(
                pending_job = %row.id,
                "Pending job discarded by a cancel — not replayed",
            );
            continue;
        }
        // The last look this path can take: a discarded round's queued row is its
        // only trace (its invocation has long exited, so there is no signal to
        // consult), and a removal committing between the read above and the route
        // below is the one window this replay cannot close.
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
/// `is_ticket_phase_kind`) whose updated_at predates the cutoff AND whose
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
    // running slot while frozen. The kind filter mirrors `is_ticket_phase_kind`:
    // the working phases' own tokens.
    let phase_kinds = WORKING_PHASES
        .iter()
        .map(|phase| format!("'{}'", phase.as_ref()))
        .collect::<Vec<_>>()
        .join(", ");
    let rows = conn
        .query(
            &format!(
                "SELECT j.id, j.workspace_name \
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
        let workspace_name: String = row.get(1)?;
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

/// Is this `jobs.kind` one of the ticket working-phase kinds? A phase job is
/// named after its phase ([`SpawnChild::kind_str`]), so this is a check against
/// the working phase set itself rather than a second vocabulary list.
#[must_use]
fn is_ticket_phase_kind(kind: &str) -> bool {
    WORKING_PHASES.iter().any(|phase| phase.as_ref() == kind)
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
///
/// Deliberately NOT reported as a failure: the row is parked, not lost, and a
/// workspace that is merely unmounted resumes on a later boot. A custom role's
/// own launch never reaches here — the durable envelope is replayed through the
/// consumer loop, which reports it as a failed launch when its folder is gone
/// (`message_router::report_unrunnable_custom_role`).
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
                    exec_dir: job.exec_dir.clone(),
                }
            });
            resumed_other += 1;
        } else if job.kind == "implement" {
            // A single-worker implement round interrupted by a crash: resume it
            // like any other durable envelope kind. The worker roster row holds
            // the terminal outcome; the job row holds the caller identity.
            if skip_sync_job(job) || workspace_unresolvable(job).await {
                continue;
            }
            let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
            resumable.push(ResumableJob::Implement {
                job_id: job.id.clone(),
                workspace_name: job.workspace_name.clone(),
                exec_dir: job.exec_dir.clone(),
            });
            resumed_other += 1;
        } else if job.kind == "research_cleanup" {
            // A research-run cleanup Sanitation agent interrupted by a
            // crash. Resume it like any other durable job. The folder stays
            // held until the resumed tail releases it; a row removed by the
            // dump-guarded cancel sweep deliberately leaves a dump-holding
            // folder for the cleanup tail / periodic temp cleaner.
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
            // A periodic temp cleaner interrupted by a crash. The
            // cleanup is best-effort and the next due pass re-runs it, so a
            // leftover row is terminalized (never resumed). The cleaner's
            // workspace is a synthetic ephemeral name that has no `workspaces`
            // row — resuming would hit run_management's unresolvable-workspace
            // path; terminalizing here keeps that
            // explicit and skips the catch-all warning.
            info!(
                job = %job.id,
                "Temp cleaner row left over from a previous lifetime — terminalizing (no resume)",
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
/// insert no longer conflicts → duplicate anchor). The DDL and this write use
/// the IDENTICAL syntactic WHERE form (`job_id IS NULL` on both sides); the
/// update branch repeats the guard, so it can never touch the job-bound roster
/// row that carries the same agent id.
///
/// The update half is the normal path, not an edge case: the claim-time write
/// (`run_claim_pipeline`, Queued→InDevelopment) creates the engineer pin, and
/// every later `run_stage_agent` for the same ticket and role re-upserts it —
/// including each bounce/reset round, since the NULL-seat row survives the
/// round job's deletion.
pub(crate) async fn upsert_session_pin(
    conn: &Connection,
    ticket_id: &str,
    task: &str,
    status: RowStatus,
    role: Role,
) -> Result<()> {
    let pin_id = session_pin_id(ticket_id, role);
    conn.upsert_row(
        "UPDATE agents SET status = ?1, task = ?2, outcome = NULL \
         WHERE agent_id = ?3 AND job_id IS NULL",
        || params![status.as_str(), task, pin_id.as_str()],
        "INSERT INTO agents (job_id, agent_id, kind, idx, status, outcome, task) \
         VALUES (NULL, ?1, ?4, NULL, ?2, NULL, ?3) \
         ON CONFLICT(agent_id) WHERE job_id IS NULL DO NOTHING",
        params![pin_id.as_str(), status.as_str(), task, role.as_str()],
    )
    .await
    .with_context(|| format!("failed to upsert {role} session pin for ticket {ticket_id}"))?;
    Ok(())
}

/// Parse `ticket_{ticket_id}_{role}` — the ticket id is everything before the
/// final `_{role}` (workspace names allow underscores, so ticket ids like
/// `my_ws-42` are valid — only the final suffix is stripped).
#[must_use]
fn session_pin_ticket_id(agent_id: &str, role: Role) -> Option<String> {
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
/// protecting the session, which the transient-session TTL then reclaims.
/// Centralized here (5-min archive loop + boot sweep). Idempotent: parse
/// ticket_id from the pin agent_id, phase-check on the board side (shared
/// consolidated connection), delete.
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
                warn!(
                    ticket = %ticket_id,
                    agent = %agent_id,
                    error = %e,
                    "Failed to delete terminal session pin",
                );
            }
        }
    }
    if deleted > 0 {
        // Debug: a routine count of pins reclaimed — a failed delete keeps its
        // warn record above.
        debug!(deleted, "Removed session pins for terminal tickets");
    }
    deleted
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dispatch whose id is already cancelled commits no row: the cancel gate
    /// inside `spawn_job`'s transaction refuses it, roster included, so a round
    /// the caller was answered as stopped has nothing for a later start to resume
    /// — whatever becomes of the cancel's own sweep.
    #[serial_test::serial(recover_from_restart)]
    #[tokio::test]
    async fn spawn_job_refuses_a_cancelled_rounds_row() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let job_id = "spawn_gate_cancelled_round";
        let _cancel_guard = crate::round_cancel::register(job_id, None);
        crate::round_cancel::cancel(job_id);

        spawn_job(
            conn,
            job_id,
            "task",
            "spawn_gate_ws",
            "caller",
            "telegram",
            crate::Role::Engineer,
            &[NewAgent {
                agent_id: "spawn_gate_agent".to_string(),
                kind: AgentKind::Analyst,
                idx: Some(0),
                task: "task".to_string(),
            }],
            &SpawnChild::Analyze,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let row = conn
            .query(
                "SELECT 1 FROM jobs WHERE id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(
            row.is_empty(),
            "a cancelled round must not commit its jobs row"
        );
        let roster = conn
            .query(
                "SELECT 1 FROM agents WHERE job_id = ?1",
                crate::db::params![job_id],
            )
            .await
            .unwrap();
        assert!(
            roster.is_empty(),
            "the roster is written in the same transaction, so it goes with it"
        );

        // The gate is scoped to the analyze child: a research run's id is reused
        // by the cleanup row its cancelled-run handoff still has to write, so
        // that write is not refused by the run's fired signal.
        let cleanup_id = "spawn_gate_cleanup_round";
        let _cleanup_guard = crate::round_cancel::register(cleanup_id, None);
        crate::round_cancel::cancel(cleanup_id);
        spawn_job(
            conn,
            cleanup_id,
            "cleanup task",
            "spawn_gate_ws",
            "",
            "",
            crate::Role::Sanitation,
            &[],
            &SpawnChild::ResearchCleanup,
            None,
            None,
            None,
        )
        .await
        .unwrap();
        let cleanup_row = conn
            .query(
                "SELECT 1 FROM jobs WHERE id = ?1",
                crate::db::params![cleanup_id],
            )
            .await
            .unwrap();
        assert_eq!(
            cleanup_row.len(),
            1,
            "a cancelled run's cleanup row is still written"
        );
        conn.execute(
            "DELETE FROM jobs WHERE id IN (?1, ?2)",
            crate::db::params![job_id, cleanup_id],
        )
        .await
        .unwrap();
    }

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
    ///
    /// Serial with the other `recover_from_restart` test: a concurrent boot
    /// scan rescans the shared `jobs` table and check-points this test's async
    /// analyze a second time (retry_count 2), breaking the exact-count assert.
    #[serial_test::serial(recover_from_restart)]
    #[tokio::test]
    async fn boot_recovery_skips_caller_owned_jobs() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let ws_name = "boot_recovery_ws";
        // Binding 3: boot recovery skips (not terminalizes) a resume-eligible
        // job whose workspace is unresolvable — so the spawns need a managed
        // workspace for the async analyze to be check-pointed.
        crate::util::test::create_test_workspace("/tmp/boot_recovery_ws", ws_name).await;

        for (job_id, kind, child, caller, exec_dir) in [
            (
                "owned_analyze",
                AgentKind::Analyst,
                SpawnChild::Analyze,
                Some("pin_a"),
                None,
            ),
            (
                "owned_implement",
                AgentKind::Worker,
                SpawnChild::Implement,
                Some("pin_b"),
                None,
            ),
            (
                "null_analyze",
                AgentKind::Analyst,
                SpawnChild::Analyze,
                None,
                // A targeted round: the recorded directory must survive the boot
                // scan so the resume runs in the same directory.
                Some("/tmp/boot_recovery_target"),
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
                exec_dir,
                None,
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
            resumable.iter().any(|r| matches!(
                r,
                ResumableJob::Analyze {
                    job_id,
                    exec_dir: Some(dir),
                    ..
                } if job_id == "null_analyze" && dir == "/tmp/boot_recovery_target"
            )),
            "the targeted round's recorded directory survives the boot scan"
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
    #[serial_test::serial(recover_from_restart)] // a concurrent boot scan would rescan the shared jobs table — see the sibling test above
    #[tokio::test]
    async fn boot_scan_leaves_unresolvable_workspace_cleanup_row_in_place() {
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
    async fn clear_abandons_session_sync_jobs() {
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
                None,
                None,
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
    async fn workspace_delete_abandons_all_jobs() {
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
                None,
                None,
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
