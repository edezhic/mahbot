//! Durable jobs layer: jobs/agents/pending_jobs lifecycle, boot recovery scan and
//! the phase-only stale purge orchestrator.
//!
//! All rows live in the consolidated domain database (`core.db`) behind the one
//! writing connection, so jobs/session/board ordering and crash-safety are expressed
//! in a single transaction. This module owns the row model, the lifecycle helpers and
//! the boot scan; the DDL lives in the append-only catalog
//! ([`crate::db::migrations`]). Purge scope is phase-only: ticket-phase jobs are
//! time-deleted (the puller re-creates them from `tickets.phase`); non-phase launched
//! jobs and pending envelopes are never purged by time — explicit abandon only. The one
//! exception is a record row ([`RECORD_KINDS_SQL`]), kept [`RECORD_RETENTION_HOURS`]
//! and then purged by [`purge_expired_records`].

use crate::Role;
use crate::agent::message_router::{AgentJob, MessageKind};
use crate::db::{self, Connection, IntoParams, Row, TxGuard, Value, params};
use crate::outcome::Outcome;
use crate::pipeline::board::{TicketPhase, WORKING_PHASES};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::time::Duration;
use tracing::{debug, error, info, warn};

// Values of `jobs.kind` are fully determined by the child row (one kind per
// child — `SpawnChild::kind_str`); the [`MessageKind`] enum travels inside the
// serialized pending_jobs envelope and is unrelated to that vocabulary.

/// Explicit dispatch mode of a job — replaces the NULL-sentinel overload of `caller_agent_id`.
/// `Sync` jobs are owned by the caller session pin; `Async` jobs (research/analyze,
/// research_cleanup, ticket phases, pin-less implement dispatches) resume at boot, while
/// `temp_cleanup` rows are terminalized instead.
///
/// The string values are intrinsic Rust/SQL coupling: `as_str()`, the SQL literals in
/// [`find_owned_launched_jobs`] / [`abandon_session_jobs`] must stay in sync on a rename.
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
/// (`Reviewer`/`Tester`, `Analyst`/`BlockerVerifier`) because the resume path reads
/// the cohorts back from the stored roster: the label is the cohort identity.
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
    /// A command-line handover's single answering turn: the roster row that names the conversation
    /// which holds the handover's answer. It is never resumed as a slot.
    Turn,
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
            Self::Turn => "turn",
        }
    }
}

/// Graceful-shutdown drain cap: in-flight work completes within this window, then
/// stragglers are force-cancelled. Read from outside too: a quit-armed exit bound
/// has to outlast it (`shutdown::macos_quit`).
pub(crate) const DRAIN_CAP_SECS: u64 = 10 * 60;

/// Stale-purge cutoff (hours): only ticket-phase `jobs` rows older than this are
/// purged (they are recoverable from `tickets.phase`). Non-phase launched jobs are
/// never time-deleted (explicit abandon only), and `pending_jobs` envelopes are never
/// time-purged — at-least-once keeps unconfirmed rows alive.
pub const PURGE_CUTOFF_HOURS: i64 = 8;

// Only the columns actually read by the scan paths are copied into memory; caller
// identity is re-queried on demand by `job_caller`. The DB schema columns stay as-is.

#[derive(Debug, Clone)]
pub(crate) struct JobRow {
    pub id: String,
    pub kind: String,
    pub workspace_name: String,
    pub retry_count: i64,
    pub ticket_id: Option<String>,
    /// The caller agent's stable session pin for sync analyze/implement jobs; NULL for
    /// async/boot-resumed jobs.
    pub caller_agent_id: Option<String>,
    /// Explicit dispatch mode (see [`JobMode`]) — the `caller_agent_id` discriminator.
    mode: JobMode,
    /// The absolute folder a delegated round runs in; `None` = the caller's workspace.
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
        // Defensive fallback: the catalog entry that backfilled `jobs.mode` is retired, so
        // nothing repairs a NULL or bogus value any more — treat it as async rather than
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

// Each durable INSERT form has two producers — one lock-acquiring (conn), one
// lock-held (TxGuard). The SQL const + params builder is the single source of truth
// for the column set; call sites keep their own executor (never delegate a TxGuard
// call to conn.execute — the guard already holds the mutex).

/// pending_jobs envelope INSERT — producers: `message_router::persist_pending`
/// and `complete_job_with_envelope`.
pub(crate) const PENDING_JOB_INSERT_SQL: &str = "INSERT INTO pending_jobs \
     (id, target_agent_id, envelope, created_at) \
     VALUES (?1, ?2, ?3, ?4)";

/// Build the pending_jobs INSERT params from a durable envelope — the target agent id
/// is derived from the envelope (caller identity lives in the serialized envelope).
pub(crate) fn pending_job_params(id: &str, envelope: &AgentJob, now: &str) -> Result<[Value; 4]> {
    let envelope_json = serde_json::to_string(envelope)?;
    Ok([
        id.into(),
        envelope_target(envelope).into(),
        envelope_json.into(),
        now.into(),
    ])
}

/// agents roster INSERT — producers: `spawn_job`'s roster loop, the pipeline's
/// `insert_round_slots`, `open_handover_record`, `attach_conversation` and `upsert_job_agent`.
/// Status is the literal 'launched'.
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

/// Optional kind-specific child row inserted in the SAME tx as the job row
/// (jobs + agents + child are one atomic unit — a crash between a jobs+agents commit
/// and a later child insert would otherwise leave a resumable job without its child).
/// The shared helper takes the child explicitly (never branching on `kind`).
#[derive(Debug, Clone)]
pub(crate) enum SpawnChild {
    /// Pure kind marker — no child row (resume reads the jobs row alone).
    Analyze,
    /// Pure kind marker — no child row. The single worker's roster row holds the
    /// dispatched task; the job row holds the caller identity for resume delivery.
    Implement,
    /// research_jobs (id, state).
    Research,
    /// A research-run cleanup Sanitation agent. No child row: the jobs row's `task` holds
    /// the cleanup prompt, and the row id == the run id (folder name) so the row holds the
    /// run folder until the cleanup completes (folder first, row last).
    ResearchCleanup,
    /// A periodic temp cleaner Sanitation agent (see `crate::temp`). No child row: the jobs row's
    /// `task` holds the prompt. Leftover rows are terminalized at boot (never resumed) — the
    /// cleaner's ephemeral workspace is never a managed workspace.
    TempCleanup,
    /// A per-phase ticket job: `jobs.kind` = `phase.as_ref()` (the single state-machine
    /// authority), `jobs.ticket_id` = the ticket. One short-lived job per phase; roster
    /// grows during the phase, and the job is deleted when the phase completes or resets.
    Phase {
        phase: TicketPhase,
        ticket_id: String,
    },
}

impl SpawnChild {
    /// The `jobs.kind` value — one kind per child row (Analyze is the pure-marker
    /// exception). Deriving kind here closes the drift window of a mismatched pair.
    #[must_use]
    pub fn kind_str(&self) -> &'static str {
        match self {
            Self::Analyze => "analyze",
            Self::Implement => IMPLEMENT_KIND,
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

/// Spawn a job with its pre-generated agent roster in ONE transaction. MUST commit
/// before the agent's first session write. The kind is derived from the child via
/// [`SpawnChild::kind_str`]; a row for a cancelled analyze round is never committed.
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
    spawn_job_with_chain(
        conn,
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
        None,
        None,
    )
    .await
}

/// [`spawn_job`] with a chain step's chain — or the schema its answer must come back as — on the
/// row, so a restart resumes the same chain, or knows what the answer was asked to look like and
/// runs the transformation again (fail-open: a replay can still forward the answer unchanged).
#[expect(clippy::too_many_arguments)]
pub(crate) async fn spawn_job_with_chain(
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
    chain: Option<&crate::launch::Chain>,
    extract: Option<&crate::extract::Extract>,
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
        chain,
        extract,
    )
    .await?;
    // Manual-cancel gate INSIDE the tx (after the INSERT, before the commit), scoped to the
    // child a cancel action can name: the tx holds the single-writer lock, so a dispatch
    // whose row write races the cancel either observes the fired signal here — the row is
    // never committed — or commits first and the cancel sweep's DELETE serializes behind
    // this tx. A cancellation and the analyze row can therefore never both stand. Every
    // other child keeps its dispatch unbounded by cancel signals — a research run's id, for
    // one, is reused by the cleanup row its cancelled-run handoff still has to write.
    if matches!(child, SpawnChild::Analyze) && crate::stop::cancelled(id) {
        // Rolled back HERE rather than left to the guard's drop: a cancellation that abandons
        // its own write leaves no deferred-cleanup WARN behind and no dangling transaction.
        tracing::info!(job = %id, "Job dispatch rolled back — round cancelled");
        tx.rollback().await?;
        return Ok(());
    }
    tx.commit().await?;
    Ok(())
}

/// The jobs-row INSERT + agent-roster loop + child-row match, run inside a live
/// transaction (`tx`). Extracted from [`spawn_job`] so callers that already hold a
/// handoff transaction reuse the exact row model. Does NOT commit.
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
    chain: Option<&crate::launch::Chain>,
    extract: Option<&crate::extract::Extract>,
) -> Result<()> {
    let kind = child.kind_str();
    let ticket_id = child_ticket_id(child);
    // The mode is derived from the caller pin (sync ⇔ pin present). A sync dispatch
    // without a pin resolves to mode=async, preserving the replaced NULL-sentinel behavior.
    let mode = if caller_agent_id.is_some() {
        JobMode::Sync
    } else {
        JobMode::Async
    };
    // JSON for the custom-role launch the round was dispatched by (`NULL` for built-in
    // callers) — read back by `job_caller` on boot resume.
    let custom_launch_json = custom_launch
        .map(serde_json::to_string)
        .transpose()
        .context("serialize jobs.custom_launch")?;
    // The chain this round is a step of, read back on resume so the answer is handed on instead
    // of delivered to the caller.
    let chain_json = chain
        .map(serde_json::to_string)
        .transpose()
        .context("serialize jobs.chain")?;
    // The schema this round's answer must come back as, read back on resume so the round still
    // knows what was asked of it and runs the transformation again — fail-open, so a replay can
    // still forward the answer unchanged.
    let extract_json = extract
        .map(serde_json::to_string)
        .transpose()
        .context("serialize jobs.extract")?;
    let now = db::now();
    tx.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, user_name, channel, role, \
         ticket_id, retry_count, created_at, updated_at, caller_agent_id, mode, exec_dir, \
         custom_launch, chain, extract) \
         VALUES (?1, ?2, 'launched', ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, ?9, ?10, ?11, ?12, ?13, ?14, \
         ?15)",
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
            chain_json,
            extract_json,
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

pub(crate) struct NewAgent {
    pub agent_id: String,
    pub kind: AgentKind,
    pub idx: Option<i64>,
    pub task: String,
}

/// The `jobs.kind` of a command-line handover's record row: the one turn that answers it.
pub(crate) const TURN_KIND: &str = "turn";

/// The `jobs.kind` of a chain's own record row.
pub(crate) const CHAIN_KIND: &str = "chain";

/// The `jobs.kind` of a durable worker round's row: the work row a boot resume re-drives, and — once
/// settled — the record the id `send` printed answers.
pub(crate) const IMPLEMENT_KIND: &str = "implement";

/// The kinds whose row is the durable record of one handover handed to an agent, as the SQL list
/// every record read and write carries. Every other kind shares the `jobs` table as scheduling state
/// only, is converted to its envelope or deleted when it ends, and has no record.
const RECORD_KINDS_SQL: &str = "'turn', 'chain', 'implement'";

/// The record kinds that are receipts only: no work hangs off their row, so the interrupted-record
/// sweep and the purge own them outright. An [`IMPLEMENT_KIND`] row is work while it is `launched`.
const RECEIPT_KINDS_SQL: &str = "'turn', 'chain'";

/// The `WHERE` fragment that keeps only work still able to move: a settled record is a record, not
/// work, and every work row is deleted or converted when it ends.
const ACTIVE_STATUSES_SQL: &str = "status NOT IN ('done', 'done_no_answer', 'stopped', 'failed')";

/// Open the record row a command-line handover's id answers: the `jobs` row itself, kind `turn`,
/// plus the one roster row naming the conversation that will hold its answer. Written after the
/// handover is durable and before its work leaves, so a completion that arrives first cannot be
/// missed, and a re-issued handover finds the row it already has.
///
/// The message's text is deliberately absent: the envelope and the conversation already hold it, and
/// a record of what happened has no business keeping a third copy.
pub(crate) async fn open_handover_record(
    id: &str,
    job: &AgentJob,
    target_agent_id: &str,
) -> Result<()> {
    let conn = &crate::session::store().conn;
    let now = db::now();
    // The schema the handover asked its answer to match rides on the row, so the answer's shape is
    // derived from what was actually asked for.
    let extract_json = job
        .answer_extract()
        .map(serde_json::to_string)
        .transpose()
        .context("serialize the handover's jobs.extract")?;
    let tx = conn.begin_tx().await?;
    tx.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, user_name, channel, role, \
         retry_count, extract, created_at, updated_at) \
         VALUES (?1, ?2, 'launched', ?3, ?4, ?5, ?6, ?7, 0, ?8, ?9, ?9) \
         ON CONFLICT (id) DO NOTHING",
        params![
            id,
            TURN_KIND,
            "",
            job.workspace_name.as_str(),
            job.user_name.as_str(),
            job.channel.as_str(),
            job.role.as_str(),
            extract_json,
            now,
        ],
    )
    .await
    .with_context(|| format!("failed to open the handover record {id}"))?;
    tx.execute(
        &format!("{AGENT_INSERT_SQL} ON CONFLICT DO NOTHING"),
        agent_params(id, target_agent_id, AgentKind::Turn, None, ""),
    )
    .await
    .with_context(|| format!("failed to record the handover's conversation for {id}"))?;
    tx.commit().await?;
    Ok(())
}

/// Open a chain's own record row: kind `chain`, keyed by the chain's id (its content), with the
/// chain's length on it. A re-issued identical chain is a new run of the same chain, so it re-arms
/// the row it already has — `launched` again, with no reason and a fresh answer window. Only a chain
/// re-arms its creation time: every other handover has a fresh id and a fresh row.
pub(crate) async fn open_chain_record(chain: &crate::launch::Chain) -> Result<()> {
    let chain_steps = i64::try_from(chain.total()).ok();
    let now = db::now();
    let tx = crate::session::store().conn.begin_tx().await?;
    tx.execute(
        "INSERT INTO jobs (id, kind, status, task, workspace_name, user_name, channel, role, \
         retry_count, chain_steps, created_at, updated_at) \
         VALUES (?1, ?2, 'launched', ?3, ?4, ?5, ?6, ?7, 0, ?8, ?9, ?9) \
         ON CONFLICT (id) DO NOTHING",
        params![
            chain.id.as_str(),
            CHAIN_KIND,
            chain.task.as_str(),
            chain.workspace.as_str(),
            chain.user.as_str(),
            chain.channel.as_str(),
            Role::Assistant.as_str(),
            chain_steps,
            now.clone(),
        ],
    )
    .await
    .with_context(|| format!("failed to open the chain record {}", chain.id))?;
    // The re-arm, in the same tx: the row may be the previous run's, settled with a reason and an
    // older answer window.
    tx.execute(
        "UPDATE jobs SET status = ?2, reason = NULL, chain_steps = ?3, created_at = ?4, \
         updated_at = ?4 WHERE id = ?1",
        params![
            chain.id.as_str(),
            Outcome::Running.as_str(),
            chain_steps,
            now
        ],
    )
    .await
    .with_context(|| format!("failed to re-arm the chain record {}", chain.id))?;
    tx.commit().await?;
    Ok(())
}

/// Name the conversation that holds a record row's answer: the chain's final step, whose assistant
/// message IS the chain's answer, and the schema that step asked its answer to match. One roster row
/// per conversation, so a replayed finish is a no-op.
pub(crate) async fn attach_conversation(
    job_id: &str,
    agent_id: &str,
    extract: Option<&crate::extract::Extract>,
) -> Result<()> {
    let conn = &crate::session::store().conn;
    let tx = conn.begin_tx().await?;
    tx.execute(
        &format!("{AGENT_INSERT_SQL} ON CONFLICT DO NOTHING"),
        agent_params(job_id, agent_id, AgentKind::Turn, None, ""),
    )
    .await
    .with_context(|| format!("failed to record the answer's conversation for {job_id}"))?;
    if let Some(extract) = extract {
        let extract_json =
            serde_json::to_string(extract).context("serialize the chain's jobs.extract")?;
        tx.execute(
            "UPDATE jobs SET extract = ?2 WHERE id = ?1",
            params![job_id, extract_json],
        )
        .await
        .with_context(|| format!("failed to record the answer's schema for {job_id}"))?;
    }
    tx.commit().await?;
    Ok(())
}

/// How long a settled record row is kept after it settles — or after it was opened, when it never
/// settled.
pub const RECORD_RETENTION_HOURS: i64 = 24;

/// What a record left `launched` by a stopped process is told: the handover never got its answer.
const STOPPED_MID_FLIGHT: &str = "the product stopped before this handover's answer was delivered";

/// How many expired records one delete removes: the core store is one shared writer, so the sweep
/// bounds itself rather than holding the connection through a long pass.
const RECORD_PURGE_BATCH: i64 = 500;

/// The pause between two purge batches, so a backlog cannot monopolise the connection.
const RECORD_PURGE_PAUSE: std::time::Duration = std::time::Duration::from_millis(50);

/// How long one sweep may keep deleting before it yields to the next one, checked between batches.
/// A backlog too large for one sweep is drained over several of them.
const RECORD_PURGE_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

/// A command-line handover's acknowledgement: its own words, plus the id it printed — the shape an
/// analysis dispatch already uses, so a script has one thing to hold and one thing to ask about. An
/// agent's own `send` and the one-way `guest` address open no record, so they print no id.
#[must_use]
pub(crate) fn acknowledged(answer: String, id: Option<&str>) -> String {
    match id {
        Some(id) => format!("{answer} (job {id})"),
        None => answer,
    }
}

/// Settle a `launched` record: what became of the handover, and the reason when there is one. Only a
/// `launched` row moves, so a later turn under the same id cannot overwrite a report already made.
/// `selector` names the row — its id, or the conversation whose roster names it. Neither the answer
/// nor its shape is stored: the answer is the conversation's last assistant message already.
async fn settle_where(selector: &str, key: &str, state: Outcome, reason: Option<&str>) {
    let result = crate::session::store()
        .conn
        .execute(
            &format!(
                "UPDATE jobs SET status = ?2, reason = ?3, updated_at = ?4 \
                 WHERE {selector} AND status = 'launched' AND kind IN ({RECORD_KINDS_SQL})"
            ),
            params![key, state.as_str(), reason, db::now()],
        )
        .await;
    if let Err(e) = result {
        warn!(key, error = %e, "Failed to settle a handover's record");
    }
}

/// Settle the record row a handover's id names.
pub(crate) async fn settle_record(id: &str, state: Outcome, reason: Option<&str>) {
    settle_where("id = ?1", id, state, reason).await;
}

/// Settle the record whose conversation is `agent_id` — keyed by the conversation instead of the id.
/// A recovered session knows only its own agent id, so this is how an answer that arrives through a
/// recovery settles the handover it answers. Only a recovery retry turn calls this: every ordinary
/// turn knows the ids of the records it answers ([`turn_record_ids`](crate::agent::message_router)),
/// and the conversation-keyed write scans `agents` with no index to help it. Only a record whose
/// envelope is already gone can be settled this way: that is the handover whose message landed and
/// whose answer is still coming. Every other turn shares its conversation with the handovers queued
/// behind it, and settling by the conversation would end theirs before their message was answered.
pub(crate) async fn settle_conversation(agent_id: &str, state: Outcome, reason: Option<&str>) {
    settle_where(
        "id IN (SELECT job_id FROM agents WHERE agent_id = ?1) \
         AND id NOT IN (SELECT id FROM pending_jobs)",
        agent_id,
        state,
        reason,
    )
    .await;
}

/// A handover the router refused to run has no answer coming, so its record must not read as
/// launched for a day. `None` — a job that was never a command-line handover — records nothing.
pub(crate) async fn settle_dropped(pending_job_id: Option<&str>, reason: &str) {
    if let Some(id) = pending_job_id {
        settle_record(id, Outcome::Failed, Some(reason)).await;
    }
}

/// The last assistant message of the conversations this record dispatched, created after the record
/// was opened: the answer, as the conversation holds it. `None` when no conversation has answered
/// yet, or when the answer is blank — a finished turn with nothing to report.
async fn record_answer(id: &str, created_at: &str) -> Result<Option<String>> {
    let answer = crate::session::store()
        .conn
        .query_optional(
            "SELECT content FROM sessions WHERE role = 'assistant' AND created_at > ?2 AND agent_id \
             IN (SELECT agent_id FROM agents WHERE job_id = ?1) ORDER BY id DESC LIMIT 1",
            params![id, created_at],
            |row| row.get::<String>(0),
        )
        .await
        .context("read a record's answer from its conversation")?;
    Ok(answer.filter(|answer| !answer.trim().is_empty()))
}

/// The one existence probe: does the row answering `question` still exist? `question` is what the
/// row would prove, phrased as a question ("does a handover's work still stand"), and it is also
/// what the one warning a failed read produces names. `None` is that failure: a read that failed
/// proves nothing, so the caller decides which side to take.
async fn row_answers(
    question: &str,
    sql: &str,
    params: impl IntoParams + Send + 'static,
) -> Option<bool> {
    match crate::session::store()
        .conn
        .query_optional(sql, params, |_| Ok::<(), anyhow::Error>(()))
        .await
    {
        Ok(found) => Some(found.is_some()),
        Err(e) => {
            warn!(error = %e, question, "Could not read a record's row while settling it");
            None
        }
    }
}

/// The question [`row_answers`] is asked about a queued envelope, and the statement that answers it.
const QUEUED_ENVELOPE_QUESTION: &str = "does a queued envelope still stand";
const QUEUED_ENVELOPE_SQL: &str = "SELECT 1 FROM pending_jobs WHERE id = ?1 LIMIT 1";

/// Settle every record a previous process left `launched`, so nothing reads as running forever: done
/// when the conversation holds the answer, failed when the handover never got one, and left alone
/// while its work still stands — an envelope still queued, or a message that landed unanswered and
/// so is dead-session recovery's to answer. Runs BEFORE the boot replay, so a drained handover keeps
/// its record and is completed by the replay. A chain's record is left alone while any step still
/// has its record, because the step that is still running is what the replay resumes.
pub async fn settle_interrupted_records() {
    // Only the receipt kinds: an `implement` row is work, and the boot scan that runs after this
    // sweep is what resumes it. Settling it here would fail a round the replay is about to run.
    let rows = match crate::session::store()
        .conn
        .query(
            &format!(
                "SELECT id, chain_steps, created_at FROM jobs WHERE status = 'launched' AND \
                 kind IN ({RECEIPT_KINDS_SQL})"
            ),
            params![],
        )
        .await
    {
        Ok(rows) => rows,
        Err(e) => {
            warn!(error = %e, "Failed to read interrupted command-line send records");
            return;
        }
    };
    for row in rows {
        let Ok(id) = row.get::<String>(0) else {
            continue;
        };
        let chain_steps = row
            .get::<Option<i64>>(1)
            .ok()
            .flatten()
            .and_then(|steps| usize::try_from(steps).ok());
        let created_at = row.get::<String>(2).unwrap_or_default();
        match record_answer(&id, &created_at).await {
            Ok(Some(_)) => {
                settle_record(&id, Outcome::Done, None).await;
                continue;
            }
            Ok(None) => {}
            // A read that failed proves nothing, so the record is left as it is.
            Err(_) => continue,
        }
        // A read failure proves nothing, so the record is left `launched`.
        let in_flight = match chain_steps {
            Some(steps) => chain_has_records(&id, 1..=steps).await != Some(false),
            // A message that landed unanswered is what dead-session recovery answers, and that
            // answer settles this record; otherwise the handover's own envelope is what the replay
            // runs. Failing a record either of them will answer would be the record contradicting
            // the conversation.
            None => {
                match row_answers(
                    "does a handover's conversation still stand",
                    "SELECT 1 FROM sessions WHERE role = 'user' AND created_at > ?2 AND agent_id \
                     IN (SELECT agent_id FROM agents WHERE job_id = ?1) LIMIT 1",
                    params![id.as_str(), created_at.as_str()],
                )
                .await
                {
                    // The message never landed: only its queued envelope can still answer it, and
                    // a failed read proves nothing — both leave the record alone.
                    Some(false) => row_answers(
                        QUEUED_ENVELOPE_QUESTION,
                        QUEUED_ENVELOPE_SQL,
                        params![id.as_str()],
                    )
                    .await
                    .unwrap_or(true),
                    Some(true) | None => true,
                }
            }
        };
        if !in_flight {
            settle_record(&id, Outcome::Failed, Some(STOPPED_MID_FLIGHT)).await;
        }
    }
}

/// Delete the record rows past their retention: settled before `cutoff`, or a receipt opened before
/// it and never settled — `updated_at` moves on both, so one bound covers each. A launched
/// [`IMPLEMENT_KIND`] row is work the unpause or the boot resume still re-drives, so it stays.
/// `cutoff` is an RFC 3339 timestamp, the shape `db::now()` writes.
pub async fn purge_expired_records(cutoff: String) -> Result<u64> {
    let conn = &crate::session::store().conn;
    let deadline = std::time::Instant::now() + RECORD_PURGE_BUDGET;
    let mut total = 0u64;
    loop {
        // Turso 0.8.3 cannot parse `DELETE … LIMIT`, so the bound is an id subquery. The `agents`
        // rows go with the job's own (the schema's cascade).
        let deleted = conn
            .execute(
                &format!(
                    "DELETE FROM jobs WHERE id IN (SELECT id FROM jobs WHERE \
                     (kind IN ({RECEIPT_KINDS_SQL}) OR status <> 'launched') AND \
                     kind IN ({RECORD_KINDS_SQL}) AND updated_at < ?1 LIMIT ?2)"
                ),
                params![cutoff.as_str(), RECORD_PURGE_BATCH],
            )
            .await
            .context("delete expired command-line send records")?;
        total += deleted;
        if deleted == 0 || std::time::Instant::now() >= deadline {
            break;
        }
        // A drain or a stop ends the sweep between batches rather than through the task's
        // cancellation: the next one starts where this left off.
        if !crate::shutdown::sleep_or_shutdown_or_drain(RECORD_PURGE_PAUSE).await {
            break;
        }
    }
    Ok(total)
}

/// Checkpoint a job row: bump retry_count and touch `updated_at` (a recency marker —
/// the phase-only purge keys off it).
async fn checkpoint_job(conn: &Connection, id: &str, retry_count: i64) -> Result<()> {
    // Boot resume re-arms the row unconditionally — status is reset to Launched
    // regardless of its prior value.
    let status = Outcome::Running;
    let now = db::now();
    conn.execute(
        "UPDATE jobs SET status = ?1, retry_count = ?2, updated_at = ?3 WHERE id = ?4",
        params![status.as_str(), retry_count, now, id],
    )
    .await
    .with_context(|| format!("failed to checkpoint job {id}"))?;
    Ok(())
}

/// Write one agent's terminal outcome (agents.outcome per-agent at round collection —
/// no read-modify-write race).
pub(crate) async fn write_agent_outcome(
    conn: &Connection,
    job_id: &str,
    agent_id: &str,
    status: Outcome,
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

/// Complete a job whose result is a durable envelope: ONE tx — INSERT pending_jobs (id = source
/// jobs.id), settle a record kind with the round's own verdict, and DELETE every other kind's row.
/// Exactly-once persistence boundary.
///
/// On INSERT failure the tx rolls back, so the job row SURVIVES and the caller routes the
/// envelope best-effort. The surviving row is resumed at the next boot and delivers a SECOND
/// envelope — an accepted duplicate-delivery window on the DB-error path.
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
    // The pending row is the durable record for an envelope kind, so the job row goes with it. A
    // record kind stays as the record of the handover: settled with the round's own verdict from
    // its roster row, so the id it printed and the roster never disagree — and removed later by
    // [`purge_expired_records`]. The answer is never copied onto it: the conversation holds it, and
    // a record of what happened is not a second copy. Only a round that failed or was stopped keeps
    // its reason, which is what a record of a handover that produced no answer is for. A roster row
    // still `launched` carries no verdict yet (a round that panicked before it could checkpoint
    // one), so the fallback stands for it and the reason stays NULL.
    tx.execute(
        &format!(
            "UPDATE jobs SET status = COALESCE((SELECT status FROM agents WHERE job_id = ?1 \
             AND status <> 'launched' LIMIT 1), ?2), reason = (SELECT outcome FROM agents WHERE \
             job_id = ?1 AND status IN ('failed', 'stopped') LIMIT 1), updated_at = ?3 \
             WHERE id = ?1 AND kind IN ({RECORD_KINDS_SQL})"
        ),
        params![job_id, Outcome::Done.as_str(), now],
    )
    .await
    .with_context(|| format!("failed to settle job {job_id}"))?;
    tx.execute(
        &format!("DELETE FROM jobs WHERE id = ?1 AND kind NOT IN ({RECORD_KINDS_SQL})"),
        params![job_id],
    )
    .await
    .with_context(|| format!("failed to terminalize job {job_id}"))?;
    // Manual-cancel gate INSIDE the tx (after the INSERT, before the commit): the tx
    // holds the single-writer lock, so a cancel that fired after the caller's
    // pre-completion check is still observed here and rolls the tx back — the pending row
    // is never committed and the job row survives for the cancel sweep, whose DELETE then
    // serializes behind this tx. Job kinds that never register a cancel signal (a worker
    // round) keep their unbounded completion.
    if crate::stop::cancelled(job_id) {
        tracing::info!(job = %job_id, "Job completion rolled back — run manually cancelled");
        tx.rollback().await?;
        return Ok(());
    }
    tx.commit().await?;
    Ok(())
}

/// COMPLETE a durable analyze/research job in one tx: INSERT the pending_jobs envelope (id = job
/// id) and DELETE the jobs row (see [`complete_job_with_envelope`]). Returns the envelope with
/// `pending_job_id` set when the row persisted, cleared on INSERT failure so the caller routes
/// best-effort. A cancelled round's in-tx gate rolls the write back and still returns Ok; both
/// call sites check their cancel flag before routing.
///
/// `custom` is the calling custom role's launch, written into the envelope so
/// [`envelope_target`] resolves the custom branch.
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
        pause_notice_id: None,
        originating_workspace: None,
        custom_role: custom,
        reply_to_custom: None,
        urgent: false,
        extract: None,
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

/// Terminalize a job: DELETE the row entirely — CASCADE removes the agent roster and child rows; a
/// no-op when the row is already absent. Safe for any jobs.kind. Ticket-phase call sites use it both
/// on success (the phase body transitions the ticket and deletes its own short-lived job; the puller
/// re-creates a fresh attempt on the next tick) and on hard-failure cleanup. Envelope kinds
/// (analyze/research) are terminal only when the row is gone — the pending row IS the durable record.
pub(crate) async fn terminalize_job(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute("DELETE FROM jobs WHERE id = ?1", params![job_id])
        .await
        .context("terminalize job")?;
    Ok(())
}

/// The envelope queued for `id`, when one waits — no feature module hand-writes this
/// table's shape. An unparsable envelope answers `None`: it addresses nothing.
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

/// The one delete of a durable record, by exact id, in the caller's transaction: its queued
/// envelope first (the boot replay would otherwise deliver it), then its `jobs` row (a surviving
/// row would be resumed, and its deletion cascades the roster). Idempotent; a failure rolls the
/// caller's transaction back.
pub(crate) async fn delete_records_tx(tx: &crate::db::TxGuard<'_>, ids: &[Value]) -> Result<()> {
    let placeholders = db::sql_in_placeholders(ids.len());
    for table in ["pending_jobs", "jobs"] {
        tx.execute(
            &format!("DELETE FROM {table} WHERE id IN ({placeholders})"),
            ids.to_vec(),
        )
        .await
        .with_context(|| format!("delete the durable record from {table}"))?;
    }
    Ok(())
}

/// [`delete_records_tx`] in a transaction of its own.
pub(crate) async fn delete_records(conn: &Connection, ids: &[Value]) -> Result<()> {
    let tx = conn.begin_tx().await?;
    delete_records_tx(&tx, ids).await?;
    tx.commit().await?;
    Ok(())
}

/// Remove a gone round's own durable state: its queued result and its `jobs` row.
pub(crate) async fn delete_round_rows(conn: &Connection, job_id: &str) -> Result<()> {
    delete_records(conn, &[Value::from(job_id)]).await
}

// Sync caller-owned jobs are never purged by time — only the caller's /clear or the
// deletion of the owning workspace abandons them. This is that path.

/// User-initiated /clear or session-row delete abandons the session's unfinished sync
/// jobs before the session is deleted — no silent orphaning. CASCADE removes the roster
/// rows; sync analyze/implement jobs carry no run folders or pending envelopes.
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

/// Atomic cancel-handoff row transition for a research run being cancelled: DELETE the research
/// job row (and its pending envelope) and INSERT the `research_cleanup` row reusing id == run_id —
/// in ONE tx, so the folder can never be orphaned and boot resume picks cleanup up. The reverse
/// order is impossible (`jobs.id` is a PK).
pub(crate) async fn transition_research_to_cleanup(
    conn: &Connection,
    run_id: &str,
    cleanup_task: &str,
    workspace_name: &str,
) -> Result<()> {
    let tx = conn.begin_tx().await?;
    delete_records_tx(&tx, &[Value::from(run_id)]).await?;
    insert_job_tx(
        &tx,
        run_id,
        cleanup_task,
        workspace_name,
        "",
        "",
        Role::Sanitation,
        &[NewAgent {
            agent_id: crate::address::AgentAddress::cleanup(run_id).id(),
            kind: AgentKind::Sanitation,
            idx: None,
            task: cleanup_task.to_string(),
        }],
        &SpawnChild::ResearchCleanup,
        None,
        None,
        None,
        None,
        None,
    )
    .await?;
    tx.commit().await?;
    Ok(())
}

/// Cancel the workspace's research/research_cleanup runs before its workspace-row deletion: fires
/// each run's cancel signal, cancels its agents, then performs the cancelled-run cleanup handoff
/// (a live orchestrator does its own). Returns the cancelled-run count; the remaining job rows are
/// deleted by the CALLER's workspace-row transaction.
///
/// Known race: the handoff runs asynchronously for a live orchestrator, but the cleanup agent then
/// uses the orchestrator's in-memory [`Workspace`] handle, so the folder is still released.
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

// A sync analyze/implement job is owned by the caller session that launched it
// (`jobs.caller_agent_id` = the caller's stable session pin). The JOBS ROW is
// the source of truth for ownership. The caller's universal resume-completion
// step locates every launched caller-owned job by this pin and slot-resumes it.

/// Launched sync analyze/implement jobs owned by `caller_agent_id` (the caller's stable session
/// pin), newest first, awaiting the caller's resume-completion step. There is no freshness gate: a
/// job is dropped only by explicit abandon. The caller scopes its BINDING to its own dangling frame
/// ([`JobRow::created_at`]), so a pool entry that frame did not dispatch is abandoned.
pub(crate) async fn find_owned_launched_jobs(
    conn: &Connection,
    caller_agent_id: &str,
) -> Result<Vec<JobRow>> {
    let rows = conn
        .query(
            &format!(
                "SELECT {JOB_COLUMNS} FROM jobs j \
                 WHERE j.caller_agent_id = ?1 AND j.kind IN ('analyze','{IMPLEMENT_KIND}') \
                   AND j.status = 'launched' AND j.mode = 'sync' \
                 ORDER BY j.created_at DESC"
            ),
            params![caller_agent_id],
        )
        .await
        .context("find owned launched jobs")?;
    rows.iter().map(job_row_from).collect()
}

// A ticket phase job is a short-lived `jobs` row whose `kind` equals the ticket's
// current phase and whose `ticket_id` links it to the ticket. `tickets.phase` is the
// sole durable running truth; the job is created by the single puller when a ticket in
// a working phase has no job, and deleted when the phase completes or is reset.

/// A phase job id (kind = phase, ticket_id set).
#[derive(Debug, Clone)]
pub(crate) struct TicketJobRow {
    pub id: String,
}

fn ticket_job_row_from(row: &Row) -> anyhow::Result<TicketJobRow> {
    Ok(TicketJobRow { id: row.get(0)? })
}

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

/// Terminalize every launched phase job for a ticket (idempotent) so no launched row
/// lingers until the stale purge.
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

/// Update the phase job's stored `task`; phase dispatch re-derives its prompt from
/// live state, so the stored value is not read for re-dispatch.
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

/// Record the profile a verification round started on, so a resumed round finishes on the
/// settings it began with instead of the ones in force when it happens to resume.
pub(crate) async fn set_round_profile(job_id: &str, profile: &str) -> Result<()> {
    let conn = &crate::session::store().conn;
    conn.execute(
        "UPDATE jobs SET round_profile = ?2, updated_at = ?3 WHERE id = ?1",
        params![job_id, profile, db::now()],
    )
    .await
    .context("record the round's profile")?;
    Ok(())
}

/// The profile a verification round started on. `None` for a job that recorded none — one created
/// before the column existed, or a stage that takes no profile — so the caller reads the live one.
pub(crate) async fn round_profile(job_id: &str) -> Result<Option<String>> {
    let conn = &crate::session::store().conn;
    let stored = conn
        .query_optional(
            "SELECT round_profile FROM jobs WHERE id = ?1",
            params![job_id],
            |row| row.get::<Option<String>>(0),
        )
        .await
        .context("read the round's profile")?;
    Ok(stored.flatten())
}

/// Upsert a job-bound roster row for a currently-running stage agent — the
/// active-agent marker that replaces `tickets.assigned_to`. The roster is keyed on
/// (job_id, agent_id) so the NULL-seat engineer anchor remains a distinct row.
///
/// The update half is the re-registration of an existing `(job_id, agent_id)`: the
/// agent id is derived from the ticket and role, so the puller re-driving a job it left
/// in place registers the same pair again and updates the row. The insert half is the one roster
/// INSERT ([`AGENT_INSERT_SQL`]), so a row that does not exist yet is created `launched` — the
/// status this call carries applies to the update half.
pub(crate) async fn upsert_job_agent(
    conn: &Connection,
    job_id: &str,
    agent_id: &str,
    kind: AgentKind,
    status: Outcome,
) -> Result<()> {
    let insert = format!("{AGENT_INSERT_SQL} ON CONFLICT(job_id, agent_id) DO NOTHING");
    conn.upsert_row(
        "UPDATE agents SET status = ?1 WHERE job_id = ?2 AND agent_id = ?3",
        || params![status.as_str(), job_id, agent_id],
        &insert,
        agent_params(job_id, agent_id, kind, None, ""),
    )
    .await
    .with_context(|| format!("failed to upsert job agent {agent_id} for job {job_id}"))?;
    Ok(())
}

/// List the currently-running agent ids for a ticket (`status='launched'` roster rows
/// across its phase jobs) — drives comment routing to running agents.
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

/// True when a job has a currently-running agent — the re-dispatch guard.
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

// An interrupted parallel round is resumable from its roster: not-Done slots are
// re-run with their stored per-slot `task`, while the phase body reconstructs the Done
// half from each slot's stored `outcome`.

/// Mark a phase job's stale `launched` roster rows as `failed` so the re-dispatch
/// guard sees no running agents while the rows — and their stored per-slot tasks —
/// survive for a slot-resume. Unlike [`clear_launched_agents_for_job`] this preserves
/// the not-Done slots. Used on pause-freeze and boot recovery.
pub(crate) async fn interrupt_phase_job_roster(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute(
        "UPDATE agents SET status = 'failed' WHERE job_id = ?1 AND status = 'launched'",
        params![job_id],
    )
    .await
    .context("mark interrupted phase roster rows failed")?;
    Ok(())
}

/// Re-arm a resumed round's not-Done roster slots as `launched` so the running-signal
/// blocks a concurrent re-drive and the Running Agents view lists them. Also touches the
/// job's `updated_at` — phase jobs are purge-eligible on stale updated_at. The stored
/// `idx`/`task` are preserved.
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

/// Clear a job's stale 'launched' roster rows after a stage-completion handoff; the job
/// row survives. For an interrupted PARALLEL round use [`interrupt_phase_job_roster`]
/// instead so the not-Done slots' stored tasks survive for a slot-resume.
pub(crate) async fn clear_launched_agents_for_job(conn: &Connection, job_id: &str) -> Result<()> {
    conn.execute(
        "DELETE FROM agents WHERE job_id = ?1 AND status = 'launched'",
        params![job_id],
    )
    .await
    .context("clear launched agents for job")?;
    Ok(())
}

/// Delete the jobs row (CASCADE destroys research_jobs child rows) inside an existing
/// transaction. The agents roster is cascade-deleted with it.
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
    crate::address::AgentAddress::routed(user_name, role, workspace_name).id()
}

/// Resolve the consumer-loop agent ID for an envelope (Manager role → the workspace Manager
/// session; pinned Assistant in a personal workspace → that user's assistant session).
/// [`crate::address::AgentAddress::routed`] already branches on the manager role, so this is the
/// single canonical form.
///
/// A custom-role launch's address is carried in the envelope's `custom_role` and returned
/// directly; boot replay uses the STORED `target_agent_id`, so both agree.
pub(crate) fn envelope_target(job: &AgentJob) -> String {
    target_agent_id(
        job.custom_role.as_ref(),
        &job.user_name,
        job.role.as_str(),
        &job.workspace_name,
    )
}

/// Caller identity + task of a job, read from the `jobs` row ALONE — a job whose child
/// row is missing (a crash between the spawn tx and a child insert) still resumes and
/// delivers to the original caller instead of being stranded.
pub(crate) struct JobCaller {
    pub task: String,
    pub role: String,
    pub user_name: String,
    pub channel: String,
    /// The custom-role launch the job was dispatched by, `None` for every built-in caller
    /// and for a NULL/unreadable value — a resume addresses the custom branch instead of a
    /// fabricated built-in address. Boxed to keep the resume-outcome enums small.
    pub custom_launch: Option<Box<crate::Launch>>,
    /// The chain this round is a step of, `None` for an ordinary round — a resume hands the
    /// answer on instead of delivering it to the caller. An unreadable value is its own state,
    /// never "no chain": such a round is abandoned rather than delivered as an ordinary one.
    pub chain: crate::launch::Stored,
    /// The schema this round's answer must come back as, when its caller asked for one — a resume
    /// still knows what was asked of it and runs the transformation again, which is fail-open: a
    /// replay can still forward the answer unchanged. An unreadable value is logged and treated as
    /// absent: the answer is still delivered, whole.
    pub extract: Option<crate::extract::Extract>,
}

/// Discriminated resume preamble: a sync caller never mistakes a gone job for a
/// drain-cut, or a transient DB read error for an abandon.
pub(crate) enum ResumePreamble {
    /// Caller identity + parsed role — the resume may proceed.
    Proceed(JobCaller, Role),
    /// Drain/shutdown active — quiet abort, the job row is untouched.
    DrainCut,
    /// The jobs row is absent — only via an explicit abandon (session /clear or workspace
    /// deletion) or a cancel whose own row deletion raced this resume; no terminalization
    /// happens. The old terminalize-on-missing compensation existed only for purges.
    Gone,
    /// The job row could not be READ (transient DB error) — NOT an abandon.
    /// The row is untouched; the error is carried for infra-failure reporting.
    Unreadable(anyhow::Error),
}

/// Terminal outcome of a resumed sync (analyze/implement) round.
pub(crate) enum SyncResumeOutcome {
    /// The round reached a terminal result (Ok text or Err) — the caller owns delivery and
    /// terminalization.
    Terminal(Role, JobCaller, anyhow::Result<String>),
    /// Drain-cut mid-resume — outcomes checkpointed, the job stays launched.
    DrainCut,
    /// The job row was gone between the ownership lookup and the resume — an explicit
    /// abandon, or a cancel that removed the row while this resume ran.
    Gone,
}

/// Parse a `jobs.custom_launch` value — the ONE policy for every reader: NULL and a
/// corrupted value both mean "no launch", the corruption warned and left to the
/// reader's refusal of that shape. A resume passes it on and is refused downstream in
/// the consumer loop's custom-role funnel; an ownership check derives the fabricated
/// `…_{role}` address, which no live custom/worker session carries (`Launch::agent_id`
/// is always a `role_…`/`rolerun_…`/`worker_…` address), so it refuses the row too.
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

/// The steps among `steps` that still have a live record — a worker round's `jobs` row, a queued
/// launch envelope, or the conversation a role or worker-branch step runs in (it has no round of
/// its own) — and whether the conversations could be read too. `None` when the rows themselves
/// could not be read.
async fn live_chain_steps(
    chain_id: &str,
    steps: impl IntoIterator<Item = usize>,
) -> Option<(BTreeSet<usize>, bool)> {
    let steps: BTreeSet<usize> = steps.into_iter().collect();
    if steps.is_empty() {
        return Some((steps, true));
    }
    let ids: Vec<Value> = steps
        .iter()
        .map(|step| Value::from(crate::launch::record_id(chain_id, *step)))
        .collect();
    let placeholders = db::sql_in_placeholders(ids.len());
    // The same ids bind twice: a worker round's row and a queued envelope are both a step's record.
    let mut bound = ids.clone();
    bound.extend(ids);
    let rows = crate::session::store()
        .conn
        .query(
            &format!(
                "SELECT id FROM jobs WHERE id IN ({placeholders}) \
                 UNION SELECT id FROM pending_jobs WHERE id IN ({placeholders})"
            ),
            bound,
        )
        .await;
    let mut live = BTreeSet::new();
    match rows {
        Ok(rows) => {
            for row in rows {
                let step = row
                    .get::<String>(0)
                    .ok()
                    .and_then(|id| crate::launch::record_parts(&id).map(|(_, step)| step));
                if let Some(step) = step
                    && steps.contains(&step)
                {
                    live.insert(step);
                }
            }
        }
        Err(e) => {
            tracing::warn!(chain = %chain_id, error = %e, "Failed to read a chain's records");
            return None;
        }
    }
    match chain_sessions(chain_id).await {
        Ok(sessions) => {
            live.extend(
                sessions
                    .into_iter()
                    .map(|(_, step)| step)
                    .filter(|step| steps.contains(step)),
            );
            Some((live, true))
        }
        Err(e) => {
            tracing::warn!(chain = %chain_id, error = %e, "Failed to read a chain's conversations");
            Some((live, false))
        }
    }
}

/// Whether any of `steps` still has its durable record. `Some(false)` only when a complete read
/// proved them gone, `None` when it could not say. The step ids are deterministic, so one call is
/// both the hand-off's idempotency check (the next step alone) and a chain's liveness check (every
/// step).
pub(crate) async fn chain_has_records(
    chain_id: &str,
    steps: impl IntoIterator<Item = usize>,
) -> Option<bool> {
    let (live, conversations_read) = live_chain_steps(chain_id, steps).await?;
    if !live.is_empty() {
        // A durable row is proof on its own: a conversation read that failed cannot take it back,
        // and the hand-off's check must never re-dispatch over a step that exists.
        return Some(true);
    }
    conversations_read.then_some(false)
}

/// The conversations a chain's steps are running in, with each one's step number.
pub(crate) async fn chain_sessions(chain_id: &str) -> Result<Vec<(String, usize)>> {
    let conn = &crate::session::store().conn;
    let rows = conn
        .query(
            "SELECT agent_id, chain FROM session_metadata WHERE chain IS NOT NULL",
            db::params![],
        )
        .await
        .context("read the conversations of a chain")?;
    Ok(rows
        .into_iter()
        .filter_map(|row| {
            let agent_id = row.get::<String>(0).ok()?;
            let chain = crate::launch::Stored::parse(row.get::<Option<String>>(1).ok().flatten());
            match chain {
                crate::launch::Stored::Step(chain) if chain.id == chain_id => {
                    Some((agent_id, chain.step))
                }
                _ => None,
            }
        })
        .collect())
}

/// Remove one step's durable record from whichever table holds it.
pub(crate) async fn delete_chain_step(id: &str) -> Result<()> {
    delete_records(&crate::session::store().conn, &[Value::from(id)]).await
}

/// Remove every durable record of one chain, so no step replays. The ids are exact — one per
/// step — never a pattern, so another chain's step in the same conversation is untouched.
pub(crate) async fn delete_chain_rows(chain_id: &str) -> Result<()> {
    let ids: Vec<Value> = (1..=crate::launch::MAX_STEPS)
        .map(|step| Value::from(crate::launch::record_id(chain_id, step)))
        .collect();
    delete_records(&crate::session::store().conn, &ids).await
}

/// Drop the chain pointer from the conversation of one step: the step is over, so a later turn
/// there is an ordinary turn again. The next step's conversation is untouched.
pub(crate) async fn clear_chain_step(chain_id: &str, step: usize) -> Result<()> {
    for (agent_id, _) in chain_sessions(chain_id)
        .await?
        .into_iter()
        .filter(|(_, at)| *at == step)
    {
        crate::session::store()
            .set_chain(&agent_id, None)
            .await
            .context("clear a chain step's conversation mark")?;
    }
    Ok(())
}

/// The boot replay's repair of the one window between a step's message landing and its
/// conversation recording the chain: the dedup path is the last write before the dead-session
/// poller takes over, and the pointer is what tells it the step is in flight. The step's group is
/// re-asserted here too, so a resumed chain is visible and stoppable before any dispatch runs.
async fn repair_chain_step(row: &PendingJobRow, job: &AgentJob) {
    let Some(chain) = job
        .custom_role
        .as_ref()
        .and_then(|launch| launch.chain.as_deref())
    else {
        return;
    };
    if let Err(e) = record_chain_step(&row.target_agent_id, chain).await {
        warn!(
            pending_job = %row.id,
            error = %e,
            "Failed to record a chain step's conversation mark",
        );
    }
    crate::tools::chain::keep_visible(chain);
}

/// Record on a conversation that it is a step of `chain`. The boot replay repairs the one window
/// between a step's message landing and its conversation recording the chain, so a restart still
/// knows the step is in flight instead of resuming it as an ordinary delegation.
pub(crate) async fn record_chain_step(agent_id: &str, chain: &crate::launch::Chain) -> Result<()> {
    crate::session::store()
        .set_chain(agent_id, Some(chain))
        .await
        .context("record a chain step's conversation mark")
}

/// Retire every conversation of a chain: its dangling turn must never resume as an ordinary
/// delegation, and nothing may hand its answer on. Called when the chain ends — failed or stopped.
pub(crate) async fn retire_chain_sessions(chain_id: &str) -> Result<()> {
    for (agent_id, _) in chain_sessions(chain_id).await? {
        retire_chain_session(&agent_id).await?;
    }
    Ok(())
}

/// Retire ONE conversation from its chain step: the deliberate-stop mark keeps automatic recovery
/// away from it, and the pointer goes so a later launch there is an ordinary branch again.
pub(crate) async fn retire_chain_session(agent_id: &str) -> Result<()> {
    crate::session::store()
        .set_launch_stopped(agent_id, true)
        .await?;
    crate::session::store()
        .set_chain(agent_id, None)
        .await
        .context("clear a retired chain step's conversation mark")
}

/// Load the task + caller identity of a job from the `jobs` row alone.
async fn job_caller(conn: &Connection, job_id: &str) -> Result<Option<JobCaller>> {
    conn.query_optional(
        "SELECT task, role, user_name, channel, custom_launch, chain, extract FROM jobs \
         WHERE id = ?1",
        params![job_id],
        |r| {
            Ok::<_, anyhow::Error>(JobCaller {
                task: r.get::<String>(0)?,
                role: r.get::<String>(1)?,
                user_name: r.get::<String>(2)?,
                channel: r.get::<String>(3)?,
                custom_launch: parse_custom_launch(r.get::<Option<String>>(4)?).map(Box::new),
                chain: crate::launch::Stored::parse(r.get::<Option<String>>(5)?),
                extract: crate::extract::Extract::from_stored(r.get::<Option<String>>(6)?),
            })
        },
    )
    .await
    .context("load job caller")
}

/// A live analyze round of the caller's: the cancel action needs the row's ownership
/// AND the execution directory a cancelled round's synthetic search engine is named
/// after ([`crate::tools::delegation::release_placed_engine`]) — both from one read,
/// because neither may be looked up after the rows are gone.
pub(crate) struct CallerAnalyzeRound {
    /// The round's `place`, when it was dispatched with one.
    pub(crate) exec_dir: Option<String>,
}

/// The analyze round named by `job_id` when it is `caller`'s to stop: its `jobs` row
/// must be an analyze round whose delivery address ([`target_agent_id`]) is the caller's
/// session — the ONE derivation, so a cancel can never drift from where the result would
/// be delivered. The `kind = 'analyze'` filter is load-bearing: it refuses the `deep`
/// depth's `research` job and every worker round, however the id was named. An absent row
/// answers `None`; a row whose `custom_launch` cannot be read is read as launch-less.
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

/// Discriminating core of [`resume_job_preamble`]: same order and log texts, but the
/// caller gets an explicit [`ResumePreamble`] so it can tell a gone job from a drain-cut.
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
            // A row that carries a launch resumes as its custom branch: the persisted launch is the
            // address authority, never the tool's own construction role.
            caller_role = caller_role.for_launch(caller.custom_launch.as_deref());
            ResumePreamble::Proceed(caller, caller_role)
        }
        // The row is gone: an explicit abandon, or a cancel, raced this resume.
        Ok(None) => {
            tracing::info!(job = %job_id, "{missing_site}: job row gone (abandoned or cancelled) — skipping resume");
            ResumePreamble::Gone
        }
        // A transient DB read error is NOT an abandon — warn and surface it as an infra failure
        // (fail-safe: no deletion, the row is untouched either way).
        Err(e) => {
            tracing::warn!(job = %job_id, error = %e, "{missing_site}: failed to read job row — skipping resume");
            ResumePreamble::Unreadable(e)
        }
    }
}

/// Read a job's boot-resume retry count (0 = never boot-resumed) — the real resume
/// signal behind the research report's telemetry gate.
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

/// `Outcome::Done` as this module's SQL spells it — `as_str` is a `const fn`, so the terminal
/// status has one spelling here and cannot drift from the enum.
const DONE_STATUS: &str = Outcome::Done.as_str();

/// Load every job that can still move — the boot scan's and the unpause re-drive's one set. A settled
/// record is excluded: nothing will move it again, so it is neither resumed nor reported unknown.
async fn list_active_jobs(conn: &Connection) -> Result<Vec<JobRow>> {
    let rows = conn
        .query(
            &format!(
                "SELECT {JOB_COLUMNS} FROM jobs WHERE {ACTIVE_STATUSES_SQL} ORDER BY created_at"
            ),
            (),
        )
        .await
        .context("list active jobs")?;
    rows.iter().map(job_row_from).collect()
}

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

/// The roster agent ids of every job that can still move, minus the slots that already finished: the
/// ids a cut round resumes IN PLACE from ([`Outcome::Done`] slots are reconstructed from their stored
/// outcomes). A settled record drops out — the same set [`list_active_jobs`] takes — so a handover's
/// conversation stops protecting its namespaces the moment the handover settles.
pub(crate) async fn resumable_roster_agent_ids(conn: &Connection) -> Result<Vec<String>> {
    let rows = conn
        .query(
            &format!(
                "SELECT DISTINCT a.agent_id FROM agents a \
                 WHERE a.job_id IN (SELECT id FROM jobs WHERE {ACTIVE_STATUSES_SQL}) \
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

/// Waits for the drain flag, then polls the agent registry AND the non-agent call registry. Clean
/// exit: no agent registered AND no orchestrator-only LLM call in flight (the research orchestrator
/// holds a whole-run guard so the token never fires into an inter-phase window) → fires the global
/// token, which the GUI subscription turns into window exit. Cap expiry (10 min): force-cancel.
///
/// The jobs table is NOT polled: drain-cut rounds leave their jobs status='launched' for boot
/// resume, so a count-based wait would hold the window open for the full cap.
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
        // Drain-cut leftovers are covered: cut rounds leave their agents unregistered AND
        // their jobs status='launched', so the registries drain without them.
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

/// What a handover dropped with its stopped launch is told.
const DROPPED_BY_STOPPED_LAUNCH: &str = "the handover was dropped when its launch stopped";

/// Delete every deferred envelope addressed to `target_agent_id`, returning how many
/// rows were dropped. The deliberate-stop path is the ONE place that drops durable
/// envelopes by address, and it waives the at-least-once guarantee (a stopped launch
/// must not be replayed at boot); every other deletion stays by row id. A dropped
/// envelope is work nobody will answer, so its outcome record is settled once the rows
/// are gone — a chain step's record is the chain's own, so dropping the step ends it.
#[expect(clippy::cast_possible_truncation)] // u64 row count fits usize on all supported targets
pub(crate) async fn delete_pending_jobs_for(target_agent_id: &str) -> Result<usize> {
    let conn = &crate::session::store().conn;
    // One transaction: an envelope persisted into the gap between the read and the delete would
    // vanish without its record being settled.
    let tx = conn.begin_tx().await?;
    let dropped: Vec<(String, String)> = tx
        .query(
            "SELECT id, envelope FROM pending_jobs WHERE target_agent_id = ?1",
            params![target_agent_id],
        )
        .await
        .context("list pending jobs for agent")?
        .iter()
        .filter_map(|row| Some((row.get::<String>(0).ok()?, row.get::<String>(1).ok()?)))
        .collect();
    let deleted = tx
        .execute(
            "DELETE FROM pending_jobs WHERE target_agent_id = ?1",
            params![target_agent_id],
        )
        .await
        .context("delete pending jobs for agent")?;
    tx.commit().await?;
    // After the delete: a crash in between leaves no row, and the boot sweep fails the record.
    for (id, envelope) in &dropped {
        // A chain step's record is the chain's own, so dropping the step ends the chain.
        let chain = serde_json::from_str::<AgentJob>(envelope)
            .ok()
            .and_then(|job| job.chain().cloned());
        match chain {
            Some(chain) => crate::tools::chain::fail(&chain, DROPPED_BY_STOPPED_LAUNCH).await,
            None => crate::jobs::settle_dropped(Some(id), DROPPED_BY_STOPPED_LAUNCH).await,
        }
    }
    info!(
        target_agent_id,
        count = deleted,
        "dropped deferred envelopes for a stopped launch"
    );
    Ok(deleted as usize)
}

/// Outcome of a boot recovery scan: jobs selected for resume.
///
/// Ticket phase jobs are NOT resumed at boot — the single puller re-drives a ticket in
/// a working phase (its row is left in place with stale roster rows cleared). Only the
/// non-ticket envelope kinds carry their own resume semantics.
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
/// recognizing both the legacy prefix format and the current suffix format. Returns the
/// input unchanged when neither is present. The timestamp body is validated against the
/// [`crate::session::render_timestamp`] shape so content that merely mimics the wrapper
/// is never mis-stripped. Residual: a suffix message whose content begins with a
/// legacy-shaped block is byte-identical to the legacy format and strips from the front
/// — unreachable in practice, failing toward duplicate delivery (never loss).
fn strip_timestamp_wrapper(content: &str) -> &str {
    // Suffix format — only for non-legacy-prefixed messages: a legacy message whose content
    // ends with a timestamp-shaped block must not be stripped from the tail.
    if !content.starts_with("<timestamp>")
        && let Some(body) = content.strip_suffix("</timestamp>")
        && let Some(start) = body.rfind("\n\n<timestamp>")
        && is_timestamp_body(&body[start + "\n\n<timestamp>".len()..])
    {
        return &body[..start];
    }
    // Legacy prefix format — requires the exact `\n\n` separator so a suffix-format message
    // whose content merely STARTS with a timestamp block is never mis-stripped.
    if let Some(body) = content.strip_prefix("<timestamp>")
        && let Some(end) = body.find("</timestamp>")
        && is_timestamp_body(&body[..end])
        && let Some(after) = body[end..].strip_prefix("</timestamp>\n\n")
    {
        return after;
    }
    content
}

/// Whether `s` matches the `%Y-%m-%d %H:%M:%S (%Z)` shape of
/// [`crate::session::render_timestamp`]. The tz must be non-empty — an empty one would
/// fail validation and boot-replay dedup would degrade to duplicate delivery.
fn is_timestamp_body(s: &str) -> bool {
    let Some((datetime, tz)) = s.strip_suffix(')').and_then(|rest| rest.rsplit_once(" (")) else {
        return false;
    };
    !tz.is_empty() && chrono::NaiveDateTime::parse_from_str(datetime, "%Y-%m-%d %H:%M:%S").is_ok()
}

/// Dedup check: was the pending envelope already appended to the target session? Every
/// user message persists with a `<timestamp>…</timestamp>` block, so exact equality
/// would always be false — a SUFFIX match after stripping the wrapper is required. Plus
/// a created_at tiebreaker: skip only if pending.created_at <= the last user message's
/// created_at (closes the silent at-most-once hole for identical consecutive messages).
/// Both are compared via chrono parsing rather than lexical strings.
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

/// Deliver the envelopes one workspace kept across a freeze: the turns a pause cut short. Called on
/// resume, so the owner's kept message is answered without restarting the product.
pub(crate) async fn replay_pending_for_workspace(name: &str) -> Result<usize> {
    replay_pending_jobs(&crate::session::store().conn, Some(name)).await
}

/// (0) Replay outstanding pending_jobs — the boot reclaim path for durable envelopes
/// (at-least-once delivery). Runs before anything that could purge.
/// Dedup semantics: the suffix + created_at tiebreaker closes the
/// listener-interleaving race — the scan and channel listeners may run
/// concurrently, the consumer serializes per agent.
///
/// A paused workspace's envelopes are kept, never delivered: the pause is what stopped them, and
/// resuming the workspace is what answers them ([`replay_pending_for_workspace`]). That is what
/// keeps a restart from re-routing a frozen workspace's kept messages and telling the owner a
/// second time that they are kept.
///
/// `only_workspace` narrows the replay to one workspace's envelopes — the resume path. It differs
/// in two ways that only matter there: a message the freeze cut mid-turn was already appended to
/// its conversation, so it is delivered again rather than deduped away (a repeated answer is
/// accepted, a lost message is not — the consumer survived the pause, so the dead-session poller
/// will never look at it), and an envelope whose workspace cannot be read is left alone rather than
/// settled, so resuming one workspace never touches another's work.
async fn replay_pending_jobs(conn: &Connection, only_workspace: Option<&str>) -> Result<usize> {
    let rows = list_pending_jobs(conn).await?;
    let mut replayed = 0usize;
    for row in &rows {
        let Ok(mut job) = serde_json::from_str::<AgentJob>(&row.envelope) else {
            // A scoped replay cannot tell whose row this is, and settling it would reach into
            // another workspace's work: the boot replay settles it.
            if only_workspace.is_some() {
                continue;
            }
            warn!(pending_job = %row.id, "Pending job envelope unreadable — skipping");
            // The row is never replayed, so nothing will answer it: a record under its id must
            // not read as running.
            crate::jobs::settle_dropped(
                Some(&row.id),
                "the handover's stored envelope could not be read",
            )
            .await;
            continue;
        };
        job.pending_job_id = Some(row.id.clone());
        if only_workspace.is_some_and(|workspace| job.workspace_name != workspace) {
            continue;
        }
        // Kept unless the workspace is proven running: a frozen one waits for its resume, and a
        // pause that cannot be read is no reason to deliver blind.
        match crate::workspace::paused_now(&job.workspace_name).await {
            Ok(false) => {}
            Ok(true) => {
                debug!(
                    pending_job = %row.id,
                    workspace = %job.workspace_name,
                    "Pending job kept — the workspace is frozen",
                );
                continue;
            }
            Err(e) => {
                debug!(
                    pending_job = %row.id,
                    workspace = %job.workspace_name,
                    error = %e,
                    "Pending job kept — the workspace's pause could not be read",
                );
                continue;
            }
        }
        if pending_already_appended(conn, row).await {
            if only_workspace.is_some() {
                // A freeze cut this turn mid-flight: the message landed, no answer came, and the
                // consumer survived the pause. Deliver it again — the owner asked for the answer,
                // and a second one is better than none.
                crate::agent::message_router::route(&row.target_agent_id, job);
                replayed += 1;
                continue;
            }
            // The message already reached the session — dedup prevents a
            // duplicate append. The row is reclaimed here (delivery complete;
            // response recovery is the dead-session poller's domain).
            repair_chain_step(row, &job).await;
            debug!(pending_job = %row.id, "Pending job already appended — skipping");
            if let Err(e) = delete_pending_job(conn, &row.id).await {
                warn!(pending_job = %row.id, error = %e, "Failed to delete deduped pending job");
            }
            // The message landed, so the handover's answer is still coming: dead-session recovery
            // re-runs the conversation, and the turn that answers it settles the record by the
            // conversation it names — so nothing has to settle it here.
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
            crate::jobs::settle_dropped(
                Some(&row.id),
                "the handover names no user it could run as",
            )
            .await;
            continue;
        };
        job.workspace_name = workspace_name;
        // Crash-window cleanup dispatch: a genuinely-undelivered research-completion envelope
        // whose run has NO `research_cleanup` row means the daemon died between
        // complete_durable_job and dispatch_research_cleanup. Create the durable row NOW (the
        // dedup marker + folder-hold); the `research_cleanup` boot-scan arm is the SOLE
        // dispatcher (spawning here would double-run the same `cleanup_{run_id}` id).
        // RUN-FOLDER EXISTENCE is the crash-window classifier: a folder still present means the
        // cleanup never ran; a folder gone means it COMPLETED in a previous lifetime while the
        // envelope stayed undelivered — no re-dispatch, hence no duplicate LLM round.
        if job.kind == MessageKind::ResearchResult
            && !crate::research_cleanup::research_cleanup_row_exists(conn, &row.id)
                .await
                .unwrap_or(true)
        {
            crate::research_cleanup::dispatch_cleanup_for_pending_envelope(&row.id, &job).await;
        }
        // The last look this path can take: a discarded round's queued row is its only trace
        // (its invocation has long exited), and a removal committing between the read above and
        // the route below is the one window this replay cannot close. A failed read proves nothing,
        // so the safe side is to deliver.
        if row_answers(
            QUEUED_ENVELOPE_QUESTION,
            QUEUED_ENVELOPE_SQL,
            params![row.id.as_str()],
        )
        .await
            == Some(false)
        {
            debug!(
                pending_job = %row.id,
                "Pending job discarded by a cancel — not replayed",
            );
            continue;
        }
        // The dedup above is the authoritative "in session" check: a row whose content is NOT in
        // the session was never appended — route the FULL content. Replaying empty here would
        // silently lose the envelope forever (at-most-once violation).
        crate::agent::message_router::route(&row.target_agent_id, job);
        replayed += 1;
    }
    Ok(replayed)
}

/// Stale-job purge.
///
/// Cutoff = 8h. Deletes only stale TICKET-PHASE jobs whose updated_at predates the cutoff AND
/// whose roster agents' sessions are all stale too (live sessions referenced by unfinished jobs
/// are NEVER purged — the agents table IS the marker). The puller re-creates a purged job on the
/// next tick; paused workspaces' phase jobs are purge-immune. Unfinished non-phase jobs are
/// never time-deleted.
pub async fn purge_stale_jobs(cutoff: &str) -> Result<u64> {
    let conn = &crate::session::store().conn;

    // A paused (frozen) phase job is purge-immune: it holds the workspace's running slot.
    // The kind filter mirrors `is_ticket_phase_kind`.
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
    // Paused workspace names: a frozen phase job must survive the stale purge. Best-effort
    // — a `workspaces` read failure falls back to purging.
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

/// Sync jobs (mode='sync') are settled by the owner's live resume-completion step
/// ([`find_owned_launched_jobs`]) — boot resume must never race it. Research jobs always
/// spawn with mode='async', so only the analyze half can fire.
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

/// Binding 3: a resume-eligible job whose workspace can no longer be resolved is left in place,
/// not terminalized — the boot loop skips it without bumping retry_count or touching the roster.
///
/// Not reported as a failure: the row is parked, not lost, and a merely unmounted workspace resumes
/// on a later boot (a custom role's launch reaches the consumer loop instead).
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

/// What the boot scan decided about one active job row.
enum ResumeDecision {
    /// The row is a resume target, checkpointed as it was classified.
    Resume(ResumableJob),
    /// The row is not this pass's to resume, and why has been reported or is known: work a pause
    /// deferred, a sync job its caller owns, a workspace that cannot be resolved, or a pause that
    /// cannot be read — the row is left alone and the next boot or resume asks again.
    Skip,
    /// A kind this build has no resume path for — reported as unknown, because it is.
    Unknown,
}

/// Whether one active job row is a resume target, checkpointed as it is classified — the one
/// classification the boot scan and a workspace's unpause share, so resuming after a freeze and
/// resuming after a restart can never disagree about what is resumable. Each skip is reported as
/// what it is, so a job the scan cannot act on is never mistaken for an unknown kind.
async fn resumable_of(conn: &Connection, job: &JobRow) -> ResumeDecision {
    // The freeze reaches the boot resume: a paused workspace's work waits for the unpause, which
    // re-drives exactly it ([`resumable_for_workspace`]) instead of writing the frozen tree now.
    match crate::workspace::paused_now(&job.workspace_name).await {
        Ok(false) => {}
        Ok(true) => {
            debug!(
                job = %job.id,
                workspace = %job.workspace_name,
                "workspace paused — its work waits for the unpause",
            );
            return ResumeDecision::Skip;
        }
        // A pause that cannot be read is not one: the row is left alone, and the next boot or
        // resume asks again.
        Err(_) => return ResumeDecision::Skip,
    }
    match job.kind.as_str() {
        // Resume at the roster/state level (dispatch re-enters the orchestrator with the stored
        // task). Always bump retry_count.
        "research" | "analyze" => {
            if skip_sync_job(job) || workspace_unresolvable(job).await {
                return ResumeDecision::Skip;
            }
            let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
            ResumeDecision::Resume(if job.kind == "research" {
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
            })
        }
        // A single-worker implement round: resume it like any other durable envelope kind. Its row is
        // both the work row this resume re-drives and, once settled, the record the id `send` printed
        // answers — a `launched` one is work, so the receipt sweep leaves it alone.
        IMPLEMENT_KIND => {
            if skip_sync_job(job) || workspace_unresolvable(job).await {
                return ResumeDecision::Skip;
            }
            let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
            ResumeDecision::Resume(ResumableJob::Implement {
                job_id: job.id.clone(),
                workspace_name: job.workspace_name.clone(),
                exec_dir: job.exec_dir.clone(),
            })
        }
        // A research-run cleanup Sanitation agent: the folder stays held until the resumed tail
        // releases it; a row removed by the dump-guarded cancel sweep deliberately leaves a
        // dump-holding folder.
        "research_cleanup" => {
            if workspace_unresolvable(job).await {
                return ResumeDecision::Skip;
            }
            let _ = checkpoint_job(conn, &job.id, job.retry_count + 1).await;
            ResumeDecision::Resume(ResumableJob::ResearchCleanup {
                job_id: job.id.clone(),
                workspace_name: job.workspace_name.clone(),
            })
        }
        // Nothing else has a resume path: every other kind is terminalized, deleted or settled when
        // it ends (record kinds are handled by the caller before they reach here).
        _ => ResumeDecision::Unknown,
    }
}

/// The resume targets of one workspace — what a freeze stopped mid-round, so resuming the
/// workspace re-drives exactly the work it stopped instead of waiting for the next restart. A run
/// that registers itself as live (an analysis or a research round) is not one of them: a pause
/// never stops a reading round, so resuming during one would otherwise start a second copy under
/// the same job id and share its single run registration.
pub(crate) async fn resumable_for_workspace(name: &str) -> Result<Vec<ResumableJob>> {
    let conn = &crate::session::store().conn;
    let mut resumable = Vec::new();
    for job in list_active_jobs(conn)
        .await?
        .iter()
        .filter(|job| job.workspace_name == name)
    {
        if crate::stop::is_registered(&job.id) {
            debug!(job = %job.id, "run is live — not re-driven by the resume");
            continue;
        }
        if let ResumeDecision::Resume(stage) = resumable_of(conn, job).await {
            resumable.push(stage);
        }
    }
    Ok(resumable)
}

/// Boot recovery scan: first statement of run_management. Order: (0) replay pending_jobs; (1) one
/// scan over the active jobs — ticket phase jobs only get their stale launched roster cleared (the
/// puller re-drives them), the research/analyze/implement/research_cleanup kinds resume, and
/// temp_cleanup rows are terminalized. An unresolvable workspace skips in place (binding 3).
///
/// Every resumed job gets updated_at = now (the ONLY protection for the pre-first-commit window)
/// and retry_count + 1.
pub(crate) async fn recover_from_restart() -> Result<Vec<ResumableJob>> {
    let start = std::time::Instant::now();
    let conn = &crate::session::store().conn;

    // (0) Deliverable replay — before anything that could purge.
    let replayed = replay_pending_jobs(conn, None).await?;

    // (1) Scan over the active jobs. Ticket phase jobs are NOT kind-discriminated resume
    // targets — the single puller re-drives them; we only clear their stale launched roster
    // rows so the puller re-dispatches. The remaining kinds resume or terminalize in line.
    let jobs = list_active_jobs(conn).await?;
    let mut resumable: Vec<ResumableJob> = Vec::new();
    let mut resumed_other = 0usize;
    for job in &jobs {
        if is_ticket_phase_kind(&job.kind) {
            // A ticket phase job interrupted by a crash: 'launched' roster rows from the previous
            // process are marked 'failed' (NOT deleted) so the puller re-dispatches a slot-resume
            // that reuses the stored per-slot tasks and already-Done outcomes.
            if let Err(e) = interrupt_phase_job_roster(conn, &job.id).await {
                warn!(
                    job = %job.id,
                    ticket = %job.ticket_id.as_deref().unwrap_or("?"),
                    error = %e,
                    "Failed to mark stale launched agents on ticket phase boot scan",
                );
            }
        } else if job.kind == "temp_cleanup" {
            // A periodic temp cleaner interrupted by a crash: the cleanup is best-effort and the
            // next due pass re-runs it, so a leftover row is terminalized (never resumed). The
            // cleaner's synthetic workspace has no `workspaces` row, so resuming would hit
            // run_management's unresolvable-workspace path.
            info!(
                job = %job.id,
                "Temp cleaner row left over from a previous lifetime — terminalizing (no resume)",
            );
            let _ = terminalize_job(conn, &job.id).await;
        } else if matches!(job.kind.as_str(), TURN_KIND | CHAIN_KIND) {
            // A record is not work: still `launched` means the handover's answer is still coming,
            // which [`settle_interrupted_records`] left for its conversation to settle. Resuming it
            // would re-dispatch an answer; reporting it unknown would be permanent noise.
        } else {
            match resumable_of(conn, job).await {
                ResumeDecision::Resume(stage) => {
                    resumable.push(stage);
                    resumed_other += 1;
                }
                ResumeDecision::Skip => {}
                ResumeDecision::Unknown => {
                    warn!(job = %job.id, kind = %job.kind, "Unknown job kind — skipping");
                }
            }
        }
    }

    let elapsed = start.elapsed();
    info!(
        duration_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
        resumed_other,
        replayed_pending = replayed,
        "Boot recovery scan complete",
    );

    // Boot sweep for session pins (S5 terminal deletion): pins for tickets already in a
    // terminal phase are removed so the TTL guard stops protecting their sessions (the
    // 5-min archive loop re-runs this — boot just catches up after downtime).
    let _ = purge_terminal_session_pins().await;

    Ok(resumable)
}

/// Deterministic session-pin agent ID: the per-ticket accumulated session
/// `ticket_{ticket_id}_{role}`, permanently job_id = NULL.
#[must_use]
pub(crate) fn session_pin_id(ticket_id: &str, role: Role) -> String {
    crate::address::AgentAddress::ticket(ticket_id.to_owned(), role.as_str().to_owned()).id()
}

/// Upsert the session pin for `role` (engineer/sanitation). NEVER sets job_id (that removes the
/// row from the partial-index scope → later NULL insert no longer conflicts → duplicate anchor).
/// The DDL and this write use the IDENTICAL `job_id IS NULL` form, so it can never touch the
/// job-bound roster row with the same agent id.
///
/// The update half is the normal path: the claim-time write creates the pin and every later
/// `run_stage_agent` re-upserts it, each bounce/reset round included.
pub(crate) async fn upsert_session_pin(
    conn: &Connection,
    ticket_id: &str,
    task: &str,
    status: Outcome,
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

/// Terminal deletion of session-pins (S5 — permanently-NULL seats): pins protect the
/// accumulated engineer/sanitation session across bounce rounds and re-attempts. Once
/// the ticket reaches a terminal phase (no future bounces/resets), the pin is
/// removed so the TTL guard stops protecting the session. Centralized here (5-min
/// archive loop + boot sweep). Idempotent: parse ticket_id from the pin agent_id,
/// phase-check on the board side, delete.
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
    // Guarded access — the purge may run before BOARD is initialized (the archive loop
    // checks later in the same tick; boot sweep runs after init).
    let board_ready = crate::pipeline::board::BOARD.get().is_some();
    for row in &rows {
        let Ok(agent_id) = row.get::<String>(0) else {
            continue;
        };
        let Ok(kind) = row.get::<String>(1) else {
            continue;
        };
        // The pin's id names the ticket it protects: the address module's split (a ticket id may
        // carry `_`, only the trailing role is stripped), with that role checked against `kind`.
        let Some(crate::address::AgentAddress::Ticket {
            ticket: ticket_id,
            role,
        }) = crate::address::AgentAddress::parse(&agent_id)
        else {
            continue;
        };
        if role != kind || ticket_id.is_empty() {
            continue;
        }
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

    /// A verification round's profile snapshot round-trips on the job row, and a job that recorded
    /// none reads as none — the fallback a job created before the column existed relies on.
    #[tokio::test]
    async fn a_round_profile_is_recorded_and_read_back() {
        crate::util::test::init_management_test_stores().await;
        let job_id = "round_profile_job";
        spawn_job(
            &crate::session::store().conn,
            job_id,
            "verify it",
            "round_profile_ws",
            crate::users::ADMIN_USER_NAME,
            "gui",
            crate::Role::Reviewer,
            &[],
            &SpawnChild::Analyze,
            None,
            None,
            None,
        )
        .await
        .expect("seed the job row");

        assert_eq!(
            round_profile(job_id)
                .await
                .expect("read the round's profile"),
            None,
            "a job that recorded no profile reads as none",
        );
        set_round_profile(job_id, r#"{"pass_threshold":6}"#)
            .await
            .expect("record the round's profile");
        assert_eq!(
            round_profile(job_id)
                .await
                .expect("read the round's profile")
                .as_deref(),
            Some(r#"{"pass_threshold":6}"#),
        );
    }

    /// The boot replay repairs the one window between a step's message landing and its
    /// conversation recording the chain, so a restart still knows the step is in flight instead of
    /// resuming it as an ordinary delegation whose answer would reach the owner.
    #[serial_test::serial(recover_from_restart)]
    #[tokio::test]
    async fn the_boot_replay_records_a_chain_step_on_its_conversation() {
        crate::util::test::init_management_test_stores().await;
        let agent_id = crate::address::AgentAddress::worker_branch("replayed".to_owned()).id();
        let chain = crate::launch::Chain {
            id: "replayed_step".to_string(),
            task: "draft it".to_string(),
            user: crate::users::ADMIN_USER_NAME.to_string(),
            channel: "gui".to_string(),
            workspace: "replayed_ws".to_string(),
            inherited: None,
            current: crate::launch::Step {
                addressee: crate::tools::send::Addressee::WorkerBranch,
                place: None,
                branch: Some("replayed".to_string()),
                name: None,
                extract: None,
            },
            step: 1,
            next: Vec::new(),
        };
        let job = AgentJob {
            content: "draft it".to_string(),
            workspace_name: "replayed_ws".to_string(),
            user_name: chain.user.clone(),
            channel: chain.channel.clone(),
            kind: MessageKind::UserMessage,
            role: crate::Role::Worker,
            reply_target: None,
            pending_job_id: None,
            pause_notice_id: None,
            originating_workspace: None,
            custom_role: Some(crate::Launch {
                role: crate::Role::Worker,
                name: String::new(),
                identifier: Some("replayed".to_string()),
                prompt: String::new(),
                tools: Vec::new(),
                agent_id: agent_id.clone(),
                reply_to: None,
                chain: Some(Box::new(chain.clone())),
                extract: None,
            }),
            reply_to_custom: None,
            urgent: false,
            extract: None,
        };
        let row = PendingJobRow {
            id: crate::launch::record_id(&chain.id, 1),
            target_agent_id: agent_id.clone(),
            envelope: serde_json::to_string(&job).expect("encode the envelope"),
            created_at: db::now(),
        };
        // The crash window: the step's message landed, its conversation never recorded the chain.
        crate::util::test::seed_session_row(
            &crate::session::store().conn,
            &agent_id,
            "user",
            "draft it",
        )
        .await;

        repair_chain_step(&row, &job).await;
        repair_chain_step(&row, &job).await;

        assert_eq!(
            chain_sessions(&chain.id)
                .await
                .expect("read the chain's conversations"),
            vec![(agent_id.clone(), 1)],
            "the replay recorded the chain on the step's conversation, once"
        );
        // And the chain's Running Agents group is back before anything is dispatched again, so a
        // chain resumed by the boot replay is visible and stoppable straight away.
        assert!(
            crate::agent::registry::NON_AGENT_CALLS
                .list()
                .iter()
                .any(|call| call.parent_key
                    == Some(crate::agent::registry::ParentKey::Chain(chain.id.clone()))),
            "the resumed chain is one group in Running Agents"
        );
        crate::jobs::retire_chain_session(&agent_id)
            .await
            .expect("retire the test's conversation");
        crate::tools::chain::stop(&chain.id)
            .await
            .expect("drop the test's chain group");
    }

    /// A poisoned envelope the boot replay refuses is work that will never run: its record is
    /// failed, rather than left reading as launched until it expires.
    #[serial_test::serial(recover_from_restart, handover_records)]
    #[tokio::test]
    async fn the_boot_replay_fails_a_refused_handovers_record() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        // A pinned role with an empty user is the one refusal this path drops the row for.
        let job = AgentJob {
            content: "do it".to_string(),
            workspace_name: "unpinned".to_string(),
            user_name: String::new(),
            channel: "cli".to_string(),
            kind: MessageKind::UserMessage,
            role: crate::Role::Assistant,
            reply_target: None,
            pending_job_id: None,
            pause_notice_id: None,
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
            urgent: false,
            extract: None,
        };
        let id = "refused_handover";
        let created_at = db::now();
        conn.execute(
            PENDING_JOB_INSERT_SQL,
            pending_job_params(id, &job, &created_at).expect("encode the envelope"),
        )
        .await
        .expect("seed the envelope");
        open_handover_record(id, &job, &envelope_target(&job))
            .await
            .expect("the handover's record row");

        replay_pending_jobs(conn, None)
            .await
            .expect("the replay runs");

        let (status, reason) = crate::util::test::record_row(id)
            .await
            .expect("the handover opened one");
        assert_eq!(status, Outcome::Failed.as_str());
        assert_eq!(
            reason.as_deref(),
            Some("the handover names no user it could run as"),
        );
    }

    /// The crash window between the message being appended and the answer leaving: the replay
    /// reclaims the row without re-running it, and dead-session recovery answers the message. The
    /// record is not failed — failing it would contradict the answer that arrives — and the boot
    /// sweep settles it the moment the conversation holds that answer.
    #[serial_test::serial(recover_from_restart, handover_records)]
    #[tokio::test]
    async fn the_boot_replay_leaves_a_record_whose_message_already_landed_running() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let job = AgentJob {
            content: "do it".to_string(),
            workspace_name: "appended_ws".to_string(),
            user_name: crate::users::ADMIN_USER_NAME.to_string(),
            channel: "cli".to_string(),
            kind: MessageKind::UserMessage,
            role: crate::Role::Assistant,
            reply_target: None,
            pending_job_id: None,
            pause_notice_id: None,
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
            urgent: false,
            extract: None,
        };
        let agent_id = envelope_target(&job);
        let id = "appended_handover";
        let created_at = db::now();
        conn.execute(
            PENDING_JOB_INSERT_SQL,
            pending_job_params(id, &job, &created_at).expect("encode the envelope"),
        )
        .await
        .expect("seed the envelope");
        open_handover_record(id, &job, &agent_id)
            .await
            .expect("the handover's record row");
        // The crash window: the message landed (a later timestamp, so the dedup sees it), the
        // answer never left.
        crate::util::test::seed_session_row(conn, &agent_id, "user", "do it").await;

        replay_pending_jobs(conn, None)
            .await
            .expect("the replay runs");

        let (status, _) = crate::util::test::record_row(id)
            .await
            .expect("the handover opened one");
        assert_eq!(
            status,
            Outcome::Running.as_str(),
            "the message landed, so the answer is still coming: the record must not read as a \
             failure the conversation will contradict",
        );
        assert!(
            !row_answers(QUEUED_ENVELOPE_QUESTION, QUEUED_ENVELOPE_SQL, params![id])
                .await
                .unwrap_or(true),
            "the deduped row is reclaimed"
        );
        // The answer arrives afterwards, and the boot sweep settles the record from it — the one
        // record, never left launched over an answer that is already there.
        crate::util::test::seed_session_row(conn, &agent_id, "assistant", "done").await;
        settle_interrupted_records().await;
        let (status, _) = crate::util::test::record_row(id)
            .await
            .expect("the handover's record stands");
        assert_eq!(status, Outcome::Done.as_str());
    }

    /// A stopped launch's waiting envelopes are dropped without an answer, so every handover among
    /// them is failed — the custom role deleted while its envelope waited is the same discard.
    #[serial_test::serial(handover_records)]
    #[tokio::test]
    async fn a_stopped_launch_fails_the_handovers_it_drops() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let target = "dropped_handover_agent";
        let job = AgentJob {
            content: "wait for me".to_string(),
            workspace_name: "dropped_ws".to_string(),
            user_name: crate::users::ADMIN_USER_NAME.to_string(),
            channel: "cli".to_string(),
            kind: MessageKind::UserMessage,
            role: crate::Role::Worker,
            reply_target: None,
            pending_job_id: None,
            pause_notice_id: None,
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
            urgent: false,
            extract: None,
        };
        let id = "dropped_handover";
        let envelope = serde_json::to_string(&job).expect("encode the envelope");
        conn.execute(
            PENDING_JOB_INSERT_SQL,
            params![id, target, envelope, db::now()],
        )
        .await
        .expect("seed the envelope");
        open_handover_record(id, &job, target)
            .await
            .expect("the handover's record row");

        assert_eq!(
            delete_pending_jobs_for(target)
                .await
                .expect("drop the envelopes"),
            1
        );

        let (status, reason) = crate::util::test::record_row(id)
            .await
            .expect("the handover opened one");
        assert_eq!(status, Outcome::Failed.as_str());
        assert!(
            reason
                .as_deref()
                .is_some_and(|reason| reason.contains("dropped when its launch stopped")),
            "the record names the discard: {reason:?}"
        );
        assert!(
            !row_answers(QUEUED_ENVELOPE_QUESTION, QUEUED_ENVELOPE_SQL, params![id])
                .await
                .unwrap_or(true),
            "the row is gone"
        );
    }

    /// An envelope the replay cannot read is never re-run, so a record under its id is failed
    /// rather than left launched.
    #[serial_test::serial(recover_from_restart, handover_records)]
    #[tokio::test]
    async fn the_boot_replay_fails_a_record_whose_envelope_is_unreadable() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let id = "unreadable_handover";
        conn.execute(
            PENDING_JOB_INSERT_SQL,
            params![id, "unreadable_agent", "{ not an envelope", db::now()],
        )
        .await
        .expect("seed the envelope");
        crate::util::test::JobRowBuilder::new(conn, id, TURN_KIND, "assistant", "unreadable_ws")
            .timestamps(db::now())
            .insert()
            .await
            .expect("the handover's record row");

        replay_pending_jobs(conn, None)
            .await
            .expect("the replay runs");

        let (status, reason) = crate::util::test::record_row(id)
            .await
            .expect("the handover opened one");
        assert_eq!(status, Outcome::Failed.as_str());
        assert!(
            reason
                .as_deref()
                .is_some_and(|reason| reason.contains("could not be read")),
            "the record names the unreadable envelope: {reason:?}"
        );
        assert!(
            row_answers(QUEUED_ENVELOPE_QUESTION, QUEUED_ENVELOPE_SQL, params![id])
                .await
                .unwrap_or(false),
            "the unreadable row itself is left in place"
        );
    }

    /// A paused workspace's envelopes are kept — no second delivery, and so no second notice
    /// telling the owner a message is kept — and a resume delivers exactly its own workspace's:
    /// the workspace-scoped path touches nothing else.
    #[serial_test::serial(recover_from_restart, handover_records)]
    #[tokio::test]
    async fn a_resume_delivers_only_its_own_workspaces_kept_messages() {
        crate::util::test::init_management_test_stores().await;
        let _ = crate::agent::message_router::init_global();
        let conn = &crate::session::store().conn;
        let mut names = Vec::new();
        let mut inboxes = Vec::new();
        for name in ["resume_kept_ws", "resume_other_ws"] {
            crate::util::test::create_test_workspace(&format!("/tmp/{name}"), name).await;
            crate::workspace::store()
                .set_paused(name, true)
                .await
                .expect("freeze the workspace");
            let job = handover_job(name);
            // A local inbox keeps the delivery from spawning a real consumer.
            inboxes.push(crate::agent::message_router::register_agent(
                &envelope_target(&job),
            ));
            let id = format!("{name}_kept_handover");
            conn.execute(
                PENDING_JOB_INSERT_SQL,
                pending_job_params(&id, &job, &db::now()).expect("serialize the envelope"),
            )
            .await
            .expect("seed the envelope");
            names.push(name);
        }

        // A frozen workspace's message is kept, not delivered.
        assert_eq!(
            replay_pending_for_workspace(names[0])
                .await
                .expect("the scoped replay"),
            0,
            "a frozen workspace's envelope waits for its resume",
        );

        // Resuming one workspace delivers only its own messages.
        conn.execute("UPDATE workspaces SET paused = 0", ())
            .await
            .expect("resume both workspaces");
        assert_eq!(
            replay_pending_for_workspace(names[0])
                .await
                .expect("the scoped replay"),
            1,
            "the resumed workspace's message is delivered",
        );
        assert!(
            row_answers(
                QUEUED_ENVELOPE_QUESTION,
                QUEUED_ENVELOPE_SQL,
                params![format!("{}_kept_handover", names[1])],
            )
            .await
            .unwrap_or(false),
            "the other workspace's message is untouched",
        );
        assert_eq!(
            replay_pending_for_workspace(names[1])
                .await
                .expect("the scoped replay"),
            1,
            "and its own resume delivers it",
        );
        drop(inboxes);
    }

    /// The message a freeze cut mid-turn is answered on resume: the turn's own envelope was already
    /// appended to the conversation, so the scoped replay delivers it again instead of deduping it
    /// away — a repeated answer is accepted, a lost message is not.
    #[serial_test::serial(recover_from_restart, handover_records)]
    #[tokio::test]
    async fn a_resume_answers_a_message_the_freeze_cut_mid_turn() {
        crate::util::test::init_management_test_stores().await;
        let _ = crate::agent::message_router::init_global();
        let conn = &crate::session::store().conn;
        let name = "resume_cut_mid_turn_ws";
        crate::util::test::create_test_workspace(&format!("/tmp/{name}"), name).await;
        let job = handover_job(name);
        let target = envelope_target(&job);
        let mut inbox = crate::agent::message_router::register_agent(&target);
        let id = "cut_mid_turn_handover";
        conn.execute(
            PENDING_JOB_INSERT_SQL,
            pending_job_params(id, &job, &db::now()).expect("serialize the envelope"),
        )
        .await
        .expect("seed the envelope");
        // The message landed before the freeze cut the turn, and no answer came with it.
        crate::util::test::seed_session_row(conn, &target, "user", &job.content).await;

        assert_eq!(
            replay_pending_for_workspace(name)
                .await
                .expect("the scoped replay"),
            1,
            "the resume delivers the kept message instead of deduping it away",
        );
        assert!(
            inbox.try_recv().is_ok(),
            "and the conversation is handed it again",
        );
        assert!(
            row_answers(QUEUED_ENVELOPE_QUESTION, QUEUED_ENVELOPE_SQL, params![id])
                .await
                .unwrap_or(false),
            "the envelope stands until its answer is delivered",
        );
    }

    /// The round re-drive a resume performs is scoped to one workspace, so resuming one never
    /// picks up another's work — and it re-drives only what a freeze actually stopped: a round
    /// whose run is still live (a pause never stops a reading pass) is left to finish.
    #[serial_test::serial(recover_from_restart)]
    #[tokio::test]
    async fn a_resume_re_drives_only_its_own_workspaces_stopped_rounds() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        for workspace in ["resume_scoped_ws", "resume_foreign_ws"] {
            crate::util::test::create_test_workspace(&format!("/tmp/{workspace}"), workspace).await;
        }
        for (id, workspace) in [
            ("resume_scoped_analyze", "resume_scoped_ws"),
            ("resume_foreign_analyze", "resume_foreign_ws"),
            ("resume_live_analyze", "resume_scoped_ws"),
        ] {
            crate::util::test::JobRowBuilder::new(conn, id, "analyze", "analyst", workspace)
                .timestamps(db::now())
                .insert()
                .await
                .expect("the round's row");
        }
        // The live round's registration is the guard: it drops, and unregisters, at scope end.
        let _live = crate::stop::register("resume_live_analyze", None);

        let resumable = resumable_for_workspace("resume_scoped_ws")
            .await
            .expect("the scoped re-drive");
        assert!(
            resumable
                .iter()
                .any(|job| matches!(job, ResumableJob::Analyze { job_id, .. } if job_id == "resume_scoped_analyze")),
            "the resumed workspace's own stopped round is re-driven",
        );
        assert!(
            !resumable
                .iter()
                .any(|job| matches!(job, ResumableJob::Analyze { job_id, .. } if job_id == "resume_live_analyze")),
            "a round that is still running is not started twice",
        );
        assert!(
            !resumable
                .iter()
                .any(|job| matches!(job, ResumableJob::Analyze { job_id, .. } if job_id == "resume_foreign_analyze")),
            "another workspace's round is left alone",
        );
    }

    /// A dispatch whose id is already cancelled commits no row: the cancel gate inside
    /// `spawn_job`'s transaction refuses it, roster included.
    #[serial_test::serial(recover_from_restart)]
    #[tokio::test]
    async fn spawn_job_refuses_a_cancelled_rounds_row() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let job_id = "spawn_gate_cancelled_round";
        let _cancel_guard = crate::stop::register(job_id, None);
        crate::stop::fire(job_id);

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
        let _cleanup_guard = crate::stop::register(cleanup_id, None);
        crate::stop::fire(cleanup_id);
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

    /// A round's schema survives its own row: the job a restart resumes still knows its answer must
    /// come back as JSON matching it, and a round with no schema reads back as one that asked for
    /// nothing — the answer is delivered whole.
    #[serial_test::serial(recover_from_restart)]
    #[tokio::test]
    async fn a_rounds_schema_survives_its_row() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        let extract = crate::extract::Extract::from_text(
            r#"{"type":"object","properties":{"summary":{"type":"string"}}}"#,
        )
        .expect("the schema is accepted");
        let job_id = "extract_row_round";
        spawn_job_with_chain(
            conn,
            job_id,
            "summarize it",
            "extract_row_ws",
            crate::users::ADMIN_USER_NAME,
            "gui",
            crate::Role::Worker,
            &[],
            &SpawnChild::Implement,
            None,
            None,
            None,
            None,
            Some(&extract),
        )
        .await
        .expect("the round's row is written");

        let (caller, role) = resume_job_preamble(conn, job_id, "test resume", "test resume")
            .await
            .expect("the round resumes from its row");
        assert_eq!(role, crate::Role::Worker);
        assert_eq!(
            caller.extract.as_ref(),
            Some(&extract),
            "the row carries the schema its answer must come back as"
        );
        delete_round_rows(conn, job_id)
            .await
            .expect("clean up this test's row");
    }

    /// Insert a minimal ticket row so a [`SpawnChild::Phase`] job's `ticket_id` FK resolves.
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

    /// (f) Boot recovery resumes only async analyze/implement/research jobs. Sync
    /// caller-owned jobs are skipped — their caller session re-drives them — so they stay
    /// `launched` with `retry_count` untouched; the async job is check-pointed.
    ///
    /// Serial with the other `recover_from_restart` test: a concurrent boot scan rescans the
    /// shared `jobs` table and check-points this test's async analyze a second time.
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
    /// unresolvable is SKIPPED in place, not resumed: it stays `launched` with `retry_count`
    /// 0 and is absent from the resumable list.
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

    /// The /clear abandon path deletes ONLY the cleared session's sync jobs — async and
    /// ticket-phase jobs for the same workspace survive, and the roster cascades away.
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

    /// Workspace deletion abandons every job for the workspace uniformly: sync, phase,
    /// research and research_cleanup rows all go, run folders are released via the run
    /// cancel, and the rosters cascade away.
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

        // Full integration: the WorkspaceStore built over the shared test conn runs the
        // research-run sweep fail-closed, then deletes the remaining job rows atomically with the
        // workspace row.
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

    /// A record row settled with each ending reads that ending, and a failure carries its reason.
    /// An empty answer is `done_no_answer` — a normal ending, not a failure.
    #[tokio::test]
    #[serial_test::serial(handover_records)]
    async fn a_settled_record_reads_its_outcome_and_reason() {
        crate::util::test::init_management_test_stores().await;
        for (id, state, reason, expected) in [
            ("settled_done", Outcome::Done, None, "done"),
            ("settled_empty", Outcome::Empty, None, "done_no_answer"),
            (
                "settled_stopped",
                Outcome::Stopped,
                Some("the workspace was paused"),
                "stopped",
            ),
            (
                "settled_failed",
                Outcome::Failed,
                Some("the folder is gone"),
                "failed",
            ),
        ] {
            crate::util::test::JobRowBuilder::new(
                &crate::session::store().conn,
                id,
                TURN_KIND,
                "assistant",
                "record_ws",
            )
            .timestamps(db::now())
            .insert()
            .await
            .expect("the record row");

            settle_record(id, state, reason).await;

            let (status, stored) = crate::util::test::record_row(id)
                .await
                .expect("the record row stands");
            assert_eq!(status, expected, "{id}");
            assert_eq!(stored.as_deref(), reason, "{id}");
        }
    }

    /// Settlement only ever moves a `launched` record of a record kind: a report already made cannot
    /// be overwritten by a later ending, and another kind of work sharing the `jobs` table is never
    /// written here.
    #[tokio::test]
    #[serial_test::serial(handover_records)]
    async fn only_a_launched_record_moves() {
        crate::util::test::init_management_test_stores().await;
        crate::util::test::JobRowBuilder::new(
            &crate::session::store().conn,
            "settle_guarded",
            TURN_KIND,
            "assistant",
            "record_ws",
        )
        .timestamps(db::now())
        .insert()
        .await
        .expect("the record row");

        settle_record("settle_guarded", Outcome::Failed, Some("the turn failed")).await;
        settle_record("settle_guarded", Outcome::Done, None).await;
        let (status, reason) = crate::util::test::record_row("settle_guarded")
            .await
            .expect("the record row stands");
        assert_eq!(
            status,
            Outcome::Failed.as_str(),
            "a later ending changes nothing"
        );
        assert_eq!(reason.as_deref(), Some("the turn failed"));

        // Work that is not a handover's record is scheduling state: its row is deleted or converted
        // when it ends, and never written here. A worker round's row IS a record, so it moves.
        crate::util::test::JobRowBuilder::new(
            &crate::session::store().conn,
            "settle_analysis_round",
            "analyze",
            "analyst",
            "record_ws",
        )
        .timestamps(db::now())
        .insert()
        .await
        .expect("the round's own row");
        settle_record("settle_analysis_round", Outcome::Done, None).await;
        let (status, _) = crate::util::test::record_row("settle_analysis_round")
            .await
            .expect("the round's row stands");
        assert_eq!(
            status,
            Outcome::Running.as_str(),
            "an analysis round is not a handover's record"
        );
    }

    /// The SQL fragments spell exactly the kinds and endings the code means: a kind or status added
    /// to one without the other would silently stop matching.
    #[test]
    fn the_record_sql_spells_the_kinds_and_endings() {
        let quoted = |items: &[&str]| {
            items
                .iter()
                .map(|item| format!("'{item}'"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        assert_eq!(
            RECORD_KINDS_SQL,
            quoted(&[TURN_KIND, CHAIN_KIND, IMPLEMENT_KIND])
        );
        assert_eq!(
            RECEIPT_KINDS_SQL,
            quoted(&[TURN_KIND, CHAIN_KIND]),
            "the receipts are the record kinds whose row is not work"
        );
        let settled = [
            Outcome::Done,
            Outcome::Empty,
            Outcome::Stopped,
            Outcome::Failed,
        ]
        .map(Outcome::as_str);
        assert_eq!(
            ACTIVE_STATUSES_SQL,
            format!("status NOT IN ({})", quoted(&settled))
        );
    }

    /// The `turn` envelope a command-line handover carries.
    fn handover_job(workspace: &str) -> crate::agent::message_router::AgentJob {
        crate::agent::message_router::AgentJob {
            content: "draft it".to_string(),
            workspace_name: workspace.to_string(),
            user_name: crate::users::ADMIN_USER_NAME.to_string(),
            channel: "cli".to_string(),
            kind: crate::agent::message_router::MessageKind::UserMessage,
            role: crate::Role::Assistant,
            reply_target: None,
            pending_job_id: None,
            pause_notice_id: None,
            originating_workspace: None,
            custom_role: None,
            reply_to_custom: None,
            urgent: false,
            extract: None,
        }
    }

    /// A record is settled by the conversation it names, so an answer that arrives through dead
    /// session recovery — which knows only its own agent id — ends the handover it answers.
    #[tokio::test]
    #[serial_test::serial(handover_records)]
    async fn a_record_is_settled_by_its_conversation() {
        crate::util::test::init_management_test_stores().await;
        let job = handover_job("record_ws");
        open_handover_record("settle_by_conversation", &job, "recovered_agent")
            .await
            .expect("the handover's record row");

        settle_conversation("some_other_agent", Outcome::Done, None).await;
        let (status, _) = crate::util::test::record_row("settle_by_conversation")
            .await
            .expect("the record row stands");
        assert_eq!(
            status,
            Outcome::Running.as_str(),
            "another conversation settles nothing"
        );

        settle_conversation("recovered_agent", Outcome::Done, None).await;
        let (status, _) = crate::util::test::record_row("settle_by_conversation")
            .await
            .expect("the record row stands");
        assert_eq!(
            status,
            Outcome::Done.as_str(),
            "the conversation that answers the handover settles its record"
        );
    }

    /// A handover queued behind a turn in the same conversation is not the turn's to settle: its
    /// envelope is still there, so its message has not landed and no answer is coming yet. Only the
    /// record whose envelope is already gone — the one dead-session recovery answers — moves.
    #[tokio::test]
    #[serial_test::serial(handover_records)]
    async fn a_queued_handovers_record_is_not_settled_by_its_conversation() {
        crate::util::test::init_management_test_stores().await;
        let job = handover_job("queued_record_ws");
        open_handover_record("settle_queued_handover", &job, "shared_conversation")
            .await
            .expect("the handover's record row");
        crate::agent::message_router::persist_pending(&job, "settle_queued_handover".to_string())
            .await
            .expect("the handover's envelope");

        settle_conversation("shared_conversation", Outcome::Done, None).await;
        let (status, _) = crate::util::test::record_row("settle_queued_handover")
            .await
            .expect("the record row stands");
        assert_eq!(
            status,
            Outcome::Running.as_str(),
            "a handover still queued has not been answered yet"
        );

        // The message landed: the envelope is gone, so the answer now comes through recovery.
        crate::session::store()
            .conn
            .execute(
                "DELETE FROM pending_jobs WHERE id = ?1",
                params!["settle_queued_handover"],
            )
            .await
            .expect("the envelope leaves with the message");
        settle_conversation("shared_conversation", Outcome::Done, None).await;
        let (status, _) = crate::util::test::record_row("settle_queued_handover")
            .await
            .expect("the record row stands");
        assert_eq!(
            status,
            Outcome::Done.as_str(),
            "the record whose envelope is gone is the recovery case"
        );
    }

    /// A worker round's record carries the round's own verdict: a failed round's printed id says
    /// `failed`, exactly as its roster row does, instead of claiming the work was done. A round that
    /// panicked before it could checkpoint a verdict has none, so the record still ends — never left
    /// `launched` for a boot to retry.
    #[tokio::test]
    #[serial_test::serial(handover_records)]
    async fn a_failed_worker_round_settles_its_record_with_its_own_verdict() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        crate::util::test::JobRowBuilder::new(
            conn,
            "failed_round_record",
            IMPLEMENT_KIND,
            "worker",
            "failed_round_ws",
        )
        .timestamps(db::now())
        .insert()
        .await
        .expect("the round's row");
        conn.execute(
            AGENT_INSERT_SQL,
            agent_params(
                "failed_round_record",
                "failed_round_worker",
                AgentKind::Worker,
                None,
                "do it",
            ),
        )
        .await
        .expect("the round's roster row");
        write_agent_outcome(
            conn,
            "failed_round_record",
            "failed_round_worker",
            Outcome::Failed,
            Some("the round went wrong"),
        )
        .await
        .expect("the round's own verdict");

        complete_job_with_envelope(
            conn,
            "failed_round_record",
            &handover_job("failed_round_ws"),
        )
        .await
        .expect("the round completes");

        let (status, reason) = crate::util::test::record_row("failed_round_record")
            .await
            .expect("the record stands");
        assert_eq!(
            status,
            Outcome::Failed.as_str(),
            "the record copies the round's verdict instead of asserting one"
        );
        assert_eq!(
            reason.as_deref(),
            Some("the round went wrong"),
            "a failed round's own text is the record's reason, not an empty one"
        );

        // A round that panicked never checkpointed a verdict: its roster row is still `launched`,
        // and the record must still end rather than wait for a boot to resume a round that is gone.
        crate::util::test::JobRowBuilder::new(
            conn,
            "panicked_round_record",
            IMPLEMENT_KIND,
            "worker",
            "failed_round_ws",
        )
        .timestamps(db::now())
        .insert()
        .await
        .expect("the round's row");
        conn.execute(
            AGENT_INSERT_SQL,
            agent_params(
                "panicked_round_record",
                "panicked_round_worker",
                AgentKind::Worker,
                None,
                "do it",
            ),
        )
        .await
        .expect("the round's roster row");
        complete_job_with_envelope(
            conn,
            "panicked_round_record",
            &handover_job("failed_round_ws"),
        )
        .await
        .expect("the round completes");
        let (status, reason) = crate::util::test::record_row("panicked_round_record")
            .await
            .expect("the record stands");
        assert_eq!(
            status,
            Outcome::Done.as_str(),
            "a round with no verdict settles on the fallback, never on `launched`"
        );
        assert_eq!(
            reason, None,
            "a round with no outcome gives the record no reason"
        );
    }

    /// A settled record never stores the answer: the conversation holds it, and a record of what
    /// happened is not a second copy. A round that failed or was stopped keeps its reason, which is
    /// what a record of a handover that produced no answer is for.
    #[serial_test::serial(handover_records)]
    #[tokio::test]
    async fn a_settled_record_keeps_an_answer_out_of_its_row_and_a_reason_in_it() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        for (id, ending, outcome) in [
            ("answered_round_record", Outcome::Done, "the whole answer"),
            (
                "stopped_round_record",
                Outcome::Stopped,
                crate::agent::PAUSED_REASON,
            ),
            (
                "empty_round_record",
                Outcome::Empty,
                "nothing to report, still not a reason",
            ),
        ] {
            crate::util::test::JobRowBuilder::new(conn, id, IMPLEMENT_KIND, "worker", "settled_ws")
                .timestamps(db::now())
                .insert()
                .await
                .expect("the round's row");
            conn.execute(
                AGENT_INSERT_SQL,
                agent_params(
                    id,
                    &format!("{id}_worker"),
                    AgentKind::Worker,
                    None,
                    "do it",
                ),
            )
            .await
            .expect("the round's roster row");
            write_agent_outcome(conn, id, &format!("{id}_worker"), ending, Some(outcome))
                .await
                .expect("the round's own verdict");
            complete_job_with_envelope(conn, id, &handover_job("settled_ws"))
                .await
                .expect("the round completes");

            let (status, reason) = crate::util::test::record_row(id)
                .await
                .expect("the record stands");
            assert_eq!(status, ending.as_str(), "the record copies the verdict");
            let expected = matches!(ending, Outcome::Failed | Outcome::Stopped).then_some(outcome);
            assert_eq!(
                reason.as_deref(),
                expected,
                "only a failed or stopped round's record keeps a reason",
            );
        }
    }

    /// A re-issued identical chain is a new run of the same chain: opening its record re-arms the
    /// row it already has — launched again, with no reason and a fresh answer window.
    #[tokio::test]
    #[serial_test::serial(handover_records)]
    async fn a_reissued_chain_re_arms_its_record() {
        crate::util::test::init_management_test_stores().await;
        let chain = crate::launch::Chain {
            id: "re_armed_chain".to_string(),
            task: "draft it".to_string(),
            user: crate::users::ADMIN_USER_NAME.to_string(),
            channel: "cli".to_string(),
            workspace: "re_armed_ws".to_string(),
            inherited: None,
            current: crate::launch::Step::worker(),
            step: 1,
            next: Vec::new(),
        };
        open_chain_record(&chain).await.expect("the chain's row");
        settle_record(&chain.id, Outcome::Failed, Some("the step failed")).await;
        let opened = "2020-01-01T00:00:00+00:00";
        crate::session::store()
            .conn
            .execute(
                "UPDATE jobs SET created_at = ?1 WHERE id = ?2",
                params![opened, chain.id.as_str()],
            )
            .await
            .expect("backdate the first run");

        open_chain_record(&chain).await.expect("the re-issued row");

        let (status, reason) = crate::util::test::record_row(&chain.id)
            .await
            .expect("the chain's record stands");
        assert_eq!(status, Outcome::Running.as_str());
        assert_eq!(reason, None, "the previous run's reason is gone");
        let created = crate::session::store()
            .conn
            .query_optional(
                "SELECT created_at FROM jobs WHERE id = ?1",
                params![chain.id.as_str()],
                |row| row.get::<String>(0),
            )
            .await
            .expect("read the row")
            .expect("the row stands");
        assert_ne!(created, opened, "the answer window starts again");
    }

    /// The boot sweep settles what a stopped process left: a record whose work is gone is failed
    /// with the sentence a script reads, one whose envelope still stands is left for the replay,
    /// one whose conversation answered is done, one whose message landed unanswered is left for
    /// dead-session recovery, and a chain with a step still standing is left alone.
    #[tokio::test]
    #[serial_test::serial(handover_records)]
    async fn the_boot_sweep_settles_what_a_stopped_process_left() {
        crate::util::test::init_management_test_stores().await;
        let conn = &crate::session::store().conn;
        for id in [
            "sweep_gone",
            "sweep_live",
            "sweep_answered",
            "sweep_unanswered",
            "sweep_chain",
            "sweep_chain_gone",
        ] {
            crate::util::test::JobRowBuilder::new(conn, id, TURN_KIND, "assistant", "sweep_ws")
                .timestamps(db::now())
                .insert()
                .await
                .expect("the record row");
        }
        // The drained handover's own work still waits for the replay.
        conn.execute(
            "INSERT INTO pending_jobs (id, target_agent_id, envelope, created_at) \
             VALUES ('sweep_live', 'sweep_agent', '{}', ?1)",
            params![db::now()],
        )
        .await
        .expect("the queued envelope");
        // The answer the third record's conversation holds.
        attach_conversation("sweep_answered", "sweep_answered_agent", None)
            .await
            .expect("the conversation");
        crate::util::test::seed_session_row(conn, "sweep_answered_agent", "assistant", "done")
            .await;
        // The fourth record's message landed and nothing answered it: dead-session recovery will,
        // and that answer settles the record — so the sweep must not call it a failure first.
        attach_conversation("sweep_unanswered", "sweep_unanswered_agent", None)
            .await
            .expect("the conversation");
        crate::util::test::seed_session_row(conn, "sweep_unanswered_agent", "user", "draft it")
            .await;
        // A chain's row is the chain's, and its first step still stands; another chain's step is
        // gone, so nothing will ever resume it.
        for id in ["sweep_chain", "sweep_chain_gone"] {
            conn.execute(
                "UPDATE jobs SET kind = ?1, chain_steps = 1 WHERE id = ?2",
                params![CHAIN_KIND, id],
            )
            .await
            .expect("the chain's row");
        }
        crate::util::test::JobRowBuilder::new(
            conn,
            crate::launch::record_id("sweep_chain", 1),
            "worker",
            "worker",
            "sweep_ws",
        )
        .timestamps(db::now())
        .insert()
        .await
        .expect("the chain's first step");

        settle_interrupted_records().await;

        let (gone, reason) = crate::util::test::record_row("sweep_gone")
            .await
            .expect("the record row stands");
        assert_eq!(gone, Outcome::Failed.as_str());
        assert_eq!(reason.as_deref(), Some(STOPPED_MID_FLIGHT));
        let (chain_gone, _) = crate::util::test::record_row("sweep_chain_gone")
            .await
            .expect("the chain's record row stands");
        assert_eq!(chain_gone, Outcome::Failed.as_str());
        for (id, expected) in [
            ("sweep_live", Outcome::Running),
            ("sweep_answered", Outcome::Done),
            ("sweep_unanswered", Outcome::Running),
            ("sweep_chain", Outcome::Running),
        ] {
            let (status, _) = crate::util::test::record_row(id)
                .await
                .expect("the record row stands");
            assert_eq!(status, expected.as_str(), "{id}");
        }
    }

    /// The purge removes a settled record past the cutoff and keeps a running one, so finished
    /// handovers do not accumulate.
    #[tokio::test]
    #[serial_test::serial(handover_records)]
    async fn the_purge_removes_only_records_past_their_retention() {
        crate::util::test::init_management_test_stores().await;
        for id in ["purge_done", "purge_running", "purge_never_settled"] {
            crate::util::test::JobRowBuilder::new(
                &crate::session::store().conn,
                id,
                TURN_KIND,
                "assistant",
                "purge_ws",
            )
            .timestamps(db::now())
            .insert()
            .await
            .expect("the record row");
        }
        settle_record("purge_done", Outcome::Done, None).await;
        // Backdate only the settled record: the running one was created now, so a cutoff between
        // the two removes the settled one and keeps the running one.
        crate::session::store()
            .conn
            .execute(
                "UPDATE jobs SET updated_at = '2020-01-01T00:00:00+00:00' \
                 WHERE id IN ('purge_done', 'purge_never_settled')",
                params![],
            )
            .await
            .expect("backdate the expired records");

        assert_eq!(
            purge_expired_records("2021-01-01T00:00:00+00:00".to_string())
                .await
                .expect("purge once"),
            2,
            "a settled record and one that never settled are both past their retention"
        );
        assert!(
            crate::util::test::record_row("purge_done").await.is_none(),
            "the settled record is gone"
        );
        assert!(
            crate::util::test::record_row("purge_never_settled")
                .await
                .is_none(),
            "a record that never settled is not kept forever"
        );
        assert!(
            crate::util::test::record_row("purge_running")
                .await
                .is_some(),
            "a running record inside its retention is kept"
        );
    }

    /// An acknowledgement names the id a script holds, and leaves the words alone without one.
    #[test]
    fn acknowledged_names_the_id_it_printed() {
        assert_eq!(
            acknowledged("Message handed over".to_string(), Some("job_1")),
            "Message handed over (job job_1)"
        );
        assert_eq!(
            acknowledged("Message handed over".to_string(), None),
            "Message handed over",
            "a handover that opened no record prints no id"
        );
    }
}
