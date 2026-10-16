//! The single, strictly-linear, append-only schema catalog for all Turso
//! stores (core.db and logs.db).
//!
//! # Core principle
//!
//! Every schema creation, change, and removal in the codebase is an entry in
//! [`MIGRATIONS`]. There is **no other place** that defines or alters a table
//! schema. `schema_migrations` records which ids have run per database; an
//! entry is applied exactly once because the only control flow is the id-based
//! applied check — there are **no runner guards** (no environment checks, no
//! fresh-vs-upgrade branching). Idempotent DDL (`IF NOT EXISTS`) lives in the
//! SQL text, and per-body idempotency logic (e.g. the [`run_add_chat_history_reply_columns`]
//! existence probe, needed because Turso has no `ADD COLUMN IF NOT EXISTS`) is
//! part of a migration body, never a conditional in the runner.
//!
//! # Current-shape baseline
//!
//! The catalog no longer walks a 0.4.2→0.5.x delta chain. The old chain (ids
//! `1`–`23`) and the one-time consolidation import are **retired**: every
//! existing install is already fully current (either the 0.5.0 shape — no
//! `chat_history` reply columns, one delta behind — the current-main shape, or
//! a fresh file), and the id-only applied check makes their recorded ids
//! `1`–`23` harmless no-ops. The baseline (`24`–`26`) creates the exact
//! current shape on fresh installs and is a strict no-op on existing ones,
//! except entries `25`/`27`–`34`/`38`/`42`, which genuinely upfill the retired
//! delta `23`'s `chat_history` reply columns, the `workspaces.maintainer_recommendations`
//! and `jobs.caller_agent_id` / `session_metadata.created_at` columns, the
//! per-user `image_gen_model` / `video_model` columns, the backfilled
//! `jobs.mode` discriminator, the `session_metadata.sleep_ended` marker, the
//! `alarms.command` column, the `chat_history.broadcast_id` column, the
//! `tickets.last_transition_actor` / `ticket_chronicle.actor` columns, the
//! nullable `users.granted_tools` column on one-delta-behind / current
//! databases, and the `alarms.trigger` column that replaces the retired shell
//! command. The tail's data migrations are
//! `36`, which detaches guests from shared workspaces by clearing
//! `users.selected_workspace`, `37`, which rewrites the legacy `DeepSeek`
//! provider-routing display name to the canonical `deepseek` slug, and `41`,
//! which deletes every command-armed `alarms` row. The tail's
//! drops are `39`/`40`, which remove `users.permissions` and
//! `users.selected_role` — account kind is the admin's name and no account
//! stores an agent role — and `43`, which retires the `alarms.command` column.
//! The tail's stage-merge entries are `44`, which rewrites the retired
//! `in_review`/`in_qa` stages to the merged `verification` stage (deleting the
//! retired job rows, since a rename would collide on the per-ticket unique
//! index, and the retired→retired chronicle hops, which never happened), and
//! `45`, which invalidates the stored per-workspace lifecycle descriptions
//! that still name a retired stage. The tail's index rebuild is `46`, which
//! drops and recreates the ticket-title FTS index in the engine's 0.8.0
//! on-disk layout (see [`REBUILD_TICKET_TITLE_FTS_INDEX`]), and the tail's
//! newest upfill is `47`, which adds the `jobs.exec_dir` column. The tail's
//! newest entries are `48`, which merges the retired `in_diagnostics` stage
//! into `verification`, `49`, which drops the stored workspace contexts
//! that claim the project commands already ran before the round, and `50`,
//! which rewrites the retired image-generation default to its successor in the
//! picker list and in every account's pick. The newest entries are the
//! custom-role tail `51`–`53` (`custom_roles` plus the custom-role columns on
//! `session_metadata` and `alarms`), `54`, which adds `jobs.custom_launch`
//! so a boot resume re-addresses a durable round to its custom branch, `55`,
//! which adds `session_metadata.personal_changes`, and `56`, which adds
//! `session_metadata.reply_to` so a coder branch's report survives a restart,
//! and `57`, which adds `session_metadata.context_block_len` so a launch
//! session's cached leading context block is measured from what was written
//! rather than re-derived from live workspace state. `58` adds the
//! `tool_calls.source` column — the marker that tells a call a command-line
//! operation performed from an agent's own. `59` adds the `telegram_intake`
//! table — the Telegram intake's durable record of the incoming updates it has
//! taken up — and `60` upfills its `claimed_by` column on the installs whose
//! table was created by the earlier revision of `59`. `61` adds the
//! `telegram_bursts` table — the durable record of every released burst. `62`
//! moves the personal-workspace change baseline off the `session_metadata` row
//! into its own `session_personal_changes` table, carrying every recorded
//! baseline across, and `63` drops the retired column. `64` drops the logs
//! store's observability indexes that no product query's plan reaches (see
//! [`DROP_UNUSED_LOGS_INDEXES`]), and `65` does the same for the core store
//! (see [`DROP_UNUSED_CORE_INDEXES`]).
//!
//! Future schema changes resume the chain at id `66` with monotonically
//! increasing, unique integer ids, never reused across any store for the
//! lifetime of the catalog.
//!
//! # Failure semantics
//!
//! A catalog/logic error is a hard boot failure — it is never a repair finding
//! and never a reason to replace the store.

use anyhow::Context;
use futures_util::future::BoxFuture;
use std::collections::HashSet;

use crate::db::{Connection, params};

/// Which physical database a migration targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetDb {
    Core,
    Logs,
}

/// A Rust-function migration body. The function is called without a
/// surrounding transaction (it owns its own transactional/FK semantics).
type RustFn = for<'a> fn(&'a Connection) -> BoxFuture<'a, anyhow::Result<()>>;

/// The body of a migration: ready SQL text, or a Rust function.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MigrationBody {
    Sql(&'static str),
    Rust(RustFn),
}

/// One ordered schema migration: an id, a target database, and a body.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Migration {
    pub(crate) id: &'static str,
    pub(crate) target: TargetDb,
    pub(crate) body: MigrationBody,
}

// ── Current-shape baseline DDL ─────────────────────────────────────────
//
// The baseline encodes the exact current shape of the consolidated domain
// database (core.db) and the logs database (logs.db) directly, without a
// 0.4.2 delta chain. Every statement is `IF NOT EXISTS`, so on an already
// current database the whole batch is a no-op. On a fresh install it
// reproduces the shape that the retired chain (ids `1`–`23`) used to assemble
// incrementally via ALTERs.

/// The full current-shape core schema: every domain table in its final shape
/// (the 0.4.2 baseline minus the retired `assigned_to` / `pipeline_reservation`
/// / `paused_frozen` / `user_roles` / `config_role` / `ticket_jobs` /
/// `ticket_stage_jobs` artifacts, plus the `jobs.ticket_id` /
/// `jobs.caller_agent_id`, `session_metadata.created_at`, `chat_history`
/// `timestamp`/`reply_*`/`broadcast_id`, and `ticket_chronicle`/`alarms` additions) and every
/// core index a product query's plan reaches — including the tickets FTS index
/// and the `idx_jobs_phase_ticket` unique index. The unreached ones a later
/// delta drops are never created here (see [`DROP_UNUSED_CORE_INDEXES`]).
///
/// `jobs` is created WITH `caller_agent_id`, `mode`, `exec_dir` and
/// `custom_launch` as its trailing columns, matching the shape the retired
/// `ALTER TABLE ...` plus deltas `28`/`30`/`47`/`54` produce, so
/// `idx_jobs_phase_ticket` (which references `jobs.ticket_id`) is safe to
/// create in the same batch. Every statement is `IF NOT EXISTS`.
const BASELINE_CORE_SCHEMA: &str = "\
-- ── Board (tickets, comments, counters) ────────────────────────────────
CREATE TABLE IF NOT EXISTS tickets (
    id               TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    description      TEXT NOT NULL,
    phase            TEXT NOT NULL DEFAULT 'backlog',
    workspace_name   TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    prerequisites    TEXT NOT NULL DEFAULT '[]',
    supersedes       TEXT,
    superseded_by    TEXT,
    commit_hash      TEXT,
    lines_added      INTEGER,
    lines_removed    INTEGER,
    reporter         TEXT NOT NULL DEFAULT '',
    is_archived      INTEGER NOT NULL DEFAULT 0,
    embedding        BLOB,
    priority         INTEGER NOT NULL DEFAULT 1,
    reviewed_head    TEXT,
    reviewed_tree    TEXT,
    done_at          TEXT,
    bounce_count     INTEGER NOT NULL DEFAULT 0,
    last_transition_actor TEXT
);
CREATE TABLE IF NOT EXISTS ticket_comments (
    id          TEXT PRIMARY KEY,
    ticket_id   TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id)
);
CREATE TABLE IF NOT EXISTS ticket_counters (
    workspace_name TEXT PRIMARY KEY,
    next_id        INTEGER NOT NULL DEFAULT 1
);
-- ── Sessions / jobs / agents ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_metadata (
    agent_id      TEXT PRIMARY KEY,
    last_activity TEXT NOT NULL,
    channel       TEXT,
    user_name     TEXT,
    workspace_name TEXT,
    role          TEXT,
    active_models TEXT,
    token_length  INTEGER,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT,
    sleep_ended   TEXT,
    custom_role   TEXT,
    custom_identifier TEXT,
    launch_stopped TEXT,
    reply_to       TEXT,
    context_block_len INTEGER
);
CREATE TABLE IF NOT EXISTS session_personal_changes (
    agent_id TEXT PRIMARY KEY REFERENCES session_metadata(agent_id) ON DELETE CASCADE,
    baseline TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'launched',
    task           TEXT NOT NULL DEFAULT '',
    workspace_name TEXT NOT NULL,
    user_name      TEXT NOT NULL DEFAULT '',
    channel        TEXT NOT NULL DEFAULT '',
    role           TEXT NOT NULL,
    retry_count    INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    ticket_id      TEXT REFERENCES tickets(id),
    caller_agent_id TEXT,
    mode           TEXT,
    exec_dir       TEXT,
    custom_launch  TEXT
);
CREATE TABLE IF NOT EXISTS agents (
    job_id     TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    agent_id   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    idx        INTEGER,
    status     TEXT NOT NULL DEFAULT 'launched',
    outcome    TEXT,
    task       TEXT NOT NULL,
    PRIMARY KEY (job_id, agent_id)
);
CREATE TABLE IF NOT EXISTS pending_jobs (
    id              TEXT PRIMARY KEY,
    target_agent_id TEXT NOT NULL,
    envelope        TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_jobs (
    id    TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    state TEXT NOT NULL
);
-- ── Workspaces / editor ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS workspaces (
    name       TEXT PRIMARY KEY,
    path       TEXT NOT NULL UNIQUE,
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    maintenance INTEGER NOT NULL DEFAULT 0,
    paused      INTEGER NOT NULL DEFAULT 1,
    maintainer_debounce_mins INTEGER NOT NULL DEFAULT 5,
    maintainer_last_run_at TEXT,
    maintainer_recommendations TEXT,
    diagnostics TEXT,
    diagnostics_generation INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    last_analyzed_commit TEXT,
    discovery_generation INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS workspace_contexts (
    workspace_name TEXT NOT NULL REFERENCES workspaces(name) ON DELETE CASCADE,
    role           TEXT,
    content        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    UNIQUE(workspace_name, role)
);
CREATE TABLE IF NOT EXISTS editor_tabs (
    workspace_name TEXT NOT NULL REFERENCES workspaces(name) ON DELETE CASCADE,
    file_path      TEXT NOT NULL,
    tab_order      INTEGER NOT NULL DEFAULT 0,
    is_active      INTEGER NOT NULL DEFAULT 0,
    is_dirty       INTEGER NOT NULL DEFAULT 0,
    dirty_content  TEXT,
    PRIMARY KEY (workspace_name, file_path)
);
-- ── Users / channels ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    name                TEXT PRIMARY KEY,
    selected_workspace  TEXT,
    image_gen_model     TEXT,
    video_model         TEXT,
    granted_tools       TEXT
);
CREATE TABLE IF NOT EXISTS user_channels (
    user_name   TEXT NOT NULL REFERENCES users(name),
    channel     TEXT NOT NULL,
    identifier  TEXT NOT NULL,
    reply_target TEXT,
    UNIQUE(channel, identifier)
);
-- ── Config ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS config_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS config_model_routing (
    model              TEXT PRIMARY KEY,
    provider_order     TEXT,
    allow_fallbacks    INTEGER
);
-- ── Chat history ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id    TEXT NOT NULL UNIQUE,
    user_name     TEXT NOT NULL,
    direction     TEXT NOT NULL,
    content       TEXT NOT NULL,
    agent_role    TEXT,
    broadcast_id  TEXT,
    workspace     TEXT NOT NULL,
    timestamp     TEXT,
    reply_author  TEXT,
    reply_snippet TEXT
);
-- ── Telegram intake ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS telegram_intake (
    account_id TEXT NOT NULL,
    update_id  INTEGER NOT NULL,
    state      TEXT NOT NULL,
    payload    TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    claimed_by TEXT,
    PRIMARY KEY (account_id, update_id)
);
CREATE TABLE IF NOT EXISTS telegram_bursts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT NOT NULL,
    user_name    TEXT NOT NULL,
    reply_target TEXT NOT NULL,
    parts        INTEGER NOT NULL,
    released_by  TEXT NOT NULL,
    released_at  TEXT NOT NULL,
    part_times   TEXT NOT NULL
);
-- ── Ticket chronicle / alarms ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ticket_chronicle (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id      TEXT NOT NULL,
    workspace_name TEXT NOT NULL,
    source_phase   TEXT NOT NULL,
    target_phase   TEXT NOT NULL,
    at             TEXT NOT NULL,
    actor          TEXT
);
CREATE TABLE IF NOT EXISTS alarms (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    user_name        TEXT NOT NULL,
    kind             TEXT NOT NULL,
    text             TEXT NOT NULL,
    fire_at          TEXT,
    interval_seconds INTEGER,
    next_fire_at     TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'active',
    created_at       TEXT NOT NULL,
    trigger          TEXT,
    role             TEXT,
    identifier       TEXT,
    workspace        TEXT
);
CREATE TABLE IF NOT EXISTS custom_roles (
    name       TEXT PRIMARY KEY,
    prompt     TEXT NOT NULL,
    tools      TEXT NOT NULL,
    behavior   TEXT NOT NULL,
    created_at TEXT NOT NULL
);
-- ── Indexes ────────────────────────────────────────────────────────────
-- Only the indexes a product query's plan reaches, plus the uniqueness ones.
-- The unreached rest are dropped by [`DROP_UNUSED_CORE_INDEXES`], which
-- carries the audit.
CREATE INDEX IF NOT EXISTS idx_ticket_comments_ticket_id ON ticket_comments(ticket_id);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id, id);
CREATE INDEX IF NOT EXISTS idx_jobs_kind_status ON jobs(kind, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agents_anchor ON agents(agent_id) WHERE job_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_pending_jobs_agent_created ON pending_jobs(target_agent_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS workspace_contexts_null_role ON workspace_contexts(workspace_name) WHERE role IS NULL;
CREATE INDEX IF NOT EXISTS idx_chat_history_user_ws_id ON chat_history(user_name, workspace, id);
CREATE INDEX IF NOT EXISTS idx_tickets_title_fts ON tickets USING fts (title) WITH (tokenizer = 'ngram');
CREATE INDEX IF NOT EXISTS idx_tickets_board_active ON tickets (is_archived, priority ASC, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_phase_ticket ON jobs(kind, ticket_id) WHERE ticket_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ticket_chronicle_ws_id ON ticket_chronicle(workspace_name, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ticket_chronicle_dedup ON ticket_chronicle(ticket_id, workspace_name, source_phase, target_phase, at);
CREATE INDEX IF NOT EXISTS idx_alarms_due ON alarms(status, next_fire_at);
CREATE INDEX IF NOT EXISTS idx_tickets_workspace_phase ON tickets (workspace_name, phase, is_archived, priority ASC, created_at DESC);";

/// The full current-shape logs schema: `logs`, `tool_calls`, `llm_requests`,
/// `llm_failures` (the observability tables) plus the `grep_telemetry` table
/// and all of their indexes. Every statement is `IF NOT EXISTS`, so on an
/// already-current database the whole batch is a no-op.
const BASELINE_LOGS_SCHEMA: &str = "\
-- ── Logs / tool calls / LLM requests ───────────────────────────────────
CREATE TABLE IF NOT EXISTS logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    level       TEXT NOT NULL,
    target      TEXT NOT NULL,
    message     TEXT NOT NULL,
    fields      TEXT NOT NULL DEFAULT '{}',
    agent_id    TEXT NOT NULL DEFAULT '',
    agent_role  TEXT NOT NULL DEFAULT '',
    workspace   TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id       TEXT NOT NULL,
    role           TEXT NOT NULL,
    source         TEXT NOT NULL DEFAULT 'agent',
    tool_name      TEXT NOT NULL,
    arguments      TEXT NOT NULL DEFAULT '{}',
    duration_ms    INTEGER NOT NULL DEFAULT 0,
    success        INTEGER NOT NULL DEFAULT 1,
    error_message  TEXT,
    workspace      TEXT NOT NULL DEFAULT '',
    recorded_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_requests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at         TEXT NOT NULL,
    purpose             TEXT NOT NULL,
    agent_id            TEXT NOT NULL DEFAULT '',
    role                TEXT NOT NULL DEFAULT '',
    workspace           TEXT NOT NULL DEFAULT '',
    ticket_id           TEXT,
    model               TEXT NOT NULL,
    routing             TEXT NOT NULL DEFAULT '',
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cached_input_tokens INTEGER,
    cache_miss_tokens   INTEGER,
    duration_ms         INTEGER NOT NULL,
    retry_attempts      INTEGER NOT NULL,
    finish_reason       TEXT,
    failure_class       TEXT,
    success             INTEGER NOT NULL DEFAULT 1,
    cost                REAL,
    cost_details        TEXT,
    upstream_provider   TEXT,
    system_fingerprint  TEXT
);
-- ── Grep telemetry ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS grep_telemetry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at   TEXT NOT NULL,
    command       TEXT NOT NULL,
    served        INTEGER NOT NULL DEFAULT 0,
    reason        TEXT NOT NULL DEFAULT '',
    recursive     INTEGER NOT NULL DEFAULT 0,
    piped         INTEGER NOT NULL DEFAULT 0,
    operand_count INTEGER NOT NULL DEFAULT 0,
    flags         TEXT NOT NULL DEFAULT '',
    mode          TEXT NOT NULL DEFAULT '',
    workspace     TEXT NOT NULL DEFAULT '',
    grep_count    INTEGER NOT NULL DEFAULT 0,
    served_count  INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER,
    exit_code     INTEGER
);
-- ── Indexes ────────────────────────────────────────────────────────────
-- Only the indexes a product query's plan reaches. The write-only rest are
-- dropped by [`DROP_UNUSED_LOGS_INDEXES`], which carries the audit.
CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
CREATE INDEX IF NOT EXISTS idx_tool_calls_recorded_at ON tool_calls(recorded_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_error_message ON tool_calls(error_message);
CREATE INDEX IF NOT EXISTS idx_llm_requests_recorded_at ON llm_requests(recorded_at);
CREATE INDEX IF NOT EXISTS idx_grep_telemetry_recorded_at ON grep_telemetry(recorded_at);
-- ── Per-attempt LLM failure trail ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS llm_failures (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id        TEXT NOT NULL,
    recorded_at         TEXT NOT NULL,
    purpose             TEXT NOT NULL,
    agent_id            TEXT NOT NULL DEFAULT '',
    role                TEXT NOT NULL DEFAULT '',
    workspace           TEXT NOT NULL DEFAULT '',
    ticket_id           TEXT,
    model               TEXT NOT NULL,
    routing             TEXT NOT NULL DEFAULT '',
    attempt             INTEGER NOT NULL,
    attempt_duration_ms INTEGER,
    failure_class       TEXT NOT NULL,
    finish_reason       TEXT,
    retry_after_ms      INTEGER,
    error_chain         TEXT NOT NULL DEFAULT ''
);";

/// Delta `35` (existing installs): the per-attempt `llm_failures` trail for
/// the logs store. Fresh installs create the table in the baseline (`26`),
/// making this a no-op there; every statement is `IF NOT EXISTS`.
///
/// The table carries no index — no production statement reads it (see
/// [`DROP_UNUSED_LOGS_INDEXES`]).
const ADD_LLM_FAILURES: &str = "CREATE TABLE IF NOT EXISTS llm_failures (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id        TEXT NOT NULL,
    recorded_at         TEXT NOT NULL,
    purpose             TEXT NOT NULL,
    agent_id            TEXT NOT NULL DEFAULT '',
    role                TEXT NOT NULL DEFAULT '',
    workspace           TEXT NOT NULL DEFAULT '',
    ticket_id           TEXT,
    model               TEXT NOT NULL,
    routing             TEXT NOT NULL DEFAULT '',
    attempt             INTEGER NOT NULL,
    attempt_duration_ms INTEGER,
    failure_class       TEXT NOT NULL,
    finish_reason       TEXT,
    retry_after_ms      INTEGER,
    error_chain         TEXT NOT NULL DEFAULT ''
);";

/// Catalog `64`: drop the logs store's indexes that no product query's plan
/// reaches.
///
/// # The audit
///
/// Every production statement against these tables was planned through the
/// debug surface's `EXPLAIN QUERY PLAN`. The kept indexes are the ones a
/// predicate or an ordering was planned onto:
///
/// - `logs` — the Issues tab's count and list (`level IN (…)`) and the INFO
///   retention (`level = 'INFO' AND timestamp < ?`) all plan on
///   `idx_logs_level`. The tab's `target`/`since` filters are never set, and
///   `agent_id`/`agent_role`/`workspace` appear in no predicate; `timestamp`
///   loses to `idx_logs_level` for the retention range.
/// - `tool_calls` — the Tool Failures count plans on
///   `idx_tool_calls_error_message` (covering) and its list on
///   `idx_tool_calls_recorded_at` (the `ORDER BY recorded_at DESC`); the two
///   retention rules search exactly those (`error_message = ?`, then
///   `recorded_at < ?`). `agent_id`/`role`/`tool_name`/`workspace` appear in no
///   predicate or ordering.
/// - `llm_requests` — the retention delete's `recorded_at < ?` plans on
///   `idx_llm_requests_recorded_at`. `agent_id`/`model`/`purpose` are written
///   and never read.
/// - `llm_failures` — no production statement reads this table at all; only
///   `record_llm_failure_row` writes it, and retention deliberately keeps it.
/// - `grep_telemetry` — the retention delete's `recorded_at < ?` plans on
///   `idx_grep_telemetry_recorded_at`. `served`/`reason`/`command` are written
///   and never read.
///
/// A filterless `COUNT(*)` (the All Logs total, and `mahbot debug`'s schema
/// dump) plans onto whichever index it happens to pick as a covering scan — it
/// has no predicate and is served by any index or by the table itself, so that
/// pick is not a dependency and its total is unchanged once the index is gone.
///
/// # Effect
///
/// Only indexes are dropped — no table, column or row changes, and the freed
/// pages go to the free pool rather than shrinking the file. Every drop is
/// `IF EXISTS`, so the entry is a no-op on a store whose baseline (`26`) and
/// delta (`35`) already carry only the kept set; a fresh install never builds
/// an index this entry would drop.
const DROP_UNUSED_LOGS_INDEXES: &str = "\
DROP INDEX IF EXISTS idx_logs_timestamp;
DROP INDEX IF EXISTS idx_logs_target;
DROP INDEX IF EXISTS idx_logs_agent_role;
DROP INDEX IF EXISTS idx_logs_agent_id;
DROP INDEX IF EXISTS idx_logs_workspace;
DROP INDEX IF EXISTS idx_tool_calls_agent_id;
DROP INDEX IF EXISTS idx_tool_calls_role;
DROP INDEX IF EXISTS idx_tool_calls_tool_name;
DROP INDEX IF EXISTS idx_tool_calls_workspace;
DROP INDEX IF EXISTS idx_llm_requests_agent_id;
DROP INDEX IF EXISTS idx_llm_requests_model;
DROP INDEX IF EXISTS idx_llm_requests_purpose;
DROP INDEX IF EXISTS idx_llm_failures_recorded_at;
DROP INDEX IF EXISTS idx_llm_failures_operation_id;
DROP INDEX IF EXISTS idx_llm_failures_failure_class;
DROP INDEX IF EXISTS idx_grep_telemetry_served;
DROP INDEX IF EXISTS idx_grep_telemetry_reason;
DROP INDEX IF EXISTS idx_grep_telemetry_command;";

/// Catalog `65`: drop the core store's indexes that no product query's plan
/// reaches.
///
/// # The audit
///
/// Every production statement against the core store's tables — every page,
/// tool, background and periodic pass, retention pass and startup pass — was
/// planned through the debug surface's `EXPLAIN QUERY PLAN`, a mutating
/// statement through an equivalent read-only form. The store carries no
/// `ANALYZE` statistics, so these are the shipped engine's stats-free
/// heuristics and do not depend on row counts. The kept indexes are the ones a
/// predicate or an ordering was planned onto:
///
/// - `ticket_comments` — `get_comments`' `WHERE ticket_id = ?` plans on
///   `idx_ticket_comments_ticket_id`.
/// - `sessions` — every read is scoped by `agent_id`, and the paginated ones
///   also seek `id` within it: `idx_sessions_agent_id`.
/// - `jobs` — the research-run lookup, the phase-job lookup, the caller's
///   launched-job read and the stale purge all plan on
///   `idx_jobs_kind_status (kind=?)`; `idx_jobs_phase_ticket` is unique.
/// - `agents` — the session-pin upsert and its delete (`WHERE agent_id = ? AND
///   job_id IS NULL`) plan on the unique `idx_agents_anchor`.
/// - `pending_jobs` — the per-target delete plans on
///   `idx_pending_jobs_agent_created (target_agent_id=?)`.
/// - `chat_history` — the paged conversation load plans on the composite
///   `idx_chat_history_user_ws_id`.
/// - `tickets` — the board listings plan on `idx_tickets_board_active`, which
///   also supplies their `priority ASC, created_at DESC` ordering, and the
///   workspace-scoped claim candidates plan on `idx_tickets_workspace_phase`;
///   the title search goes through the FTS index `idx_tickets_title_fts`.
/// - `ticket_chronicle` — the trim, the cursor seed and the drain plan on
///   `idx_ticket_chronicle_ws_id`; `idx_ticket_chronicle_dedup` is unique.
/// - `alarms` — the due sweep and the per-target listings plan on
///   `idx_alarms_due`.
/// - `workspace_contexts` — the `role IS NULL` read plans on
///   `workspace_contexts_null_role`, which is unique.
/// - The remaining core tables (`users`, `workspaces`, `session_metadata`,
///   `session_personal_changes`, `config_kv`, `config_model_routing`,
///   `editor_tabs`, `user_channels`, `research_jobs`, `ticket_counters`,
///   `custom_roles`, `telegram_intake`, `telegram_bursts`) carry no product
///   index beyond their primary key's own.
///
/// The removed set, with the statements that name their columns but are served
/// by another path:
///
/// - `idx_jobs_caller_agent` — `abandon_session_jobs`' `DELETE … WHERE mode =
///   'sync' AND caller_agent_id = ?1` scans `jobs`, and `find_owned_launched_jobs`'
///   read of the caller's launched sync jobs searches
///   `idx_jobs_kind_status (kind=?)`. A bare `WHERE caller_agent_id = ?` scans
///   too: the planner does not derive the partial index's
///   `caller_agent_id IS NOT NULL` from an equality, so no statement reaches
///   this index.
/// - `idx_jobs_updated_at` — the only production statement constraining the
///   column is `purge_stale_jobs`, and its `updated_at < ?1 AND kind IN (…)`
///   plan searches `idx_jobs_kind_status (kind=?)`. `updated_at` is otherwise
///   only ever a SET target, and nothing orders by it.
/// - `idx_chat_history_user` — a strict prefix of the kept composite
///   `idx_chat_history_user_ws_id`; no production statement filters
///   `user_name` alone.
/// - `idx_chat_history_workspace` — no production statement filters
///   `chat_history` by `workspace` alone; the conversation-load
///   `user_name = ?1 AND workspace = ?2` predicate is served by the composite.
///
/// An index that exists to enforce a guarantee is kept whatever the plans say:
/// `idx_jobs_phase_ticket` and `idx_ticket_chronicle_dedup` are enforced on
/// write and named by no plan. The engine's own indexes — the constraint
/// autoindexes, the FTS backing objects and the change-capture bookkeeping —
/// are not the product's to drop. Every statement that names a removed index's
/// columns has its chosen path in another index or in the table itself, so no
/// screen's rows, order or totals change.
///
/// # Effect
///
/// Only indexes are dropped — no table, column or row changes, and the freed
/// pages go to the free pool rather than shrinking the file. Every drop is
/// `IF EXISTS`, so the entry is a no-op on a store that already carries only
/// the kept set; a fresh install never builds an index this entry would drop,
/// because the baseline (`24`) no longer creates the three it owned and delta
/// `28` no longer creates the fourth.
const DROP_UNUSED_CORE_INDEXES: &str = "\
DROP INDEX IF EXISTS idx_jobs_caller_agent;
DROP INDEX IF EXISTS idx_jobs_updated_at;
DROP INDEX IF EXISTS idx_chat_history_user;
DROP INDEX IF EXISTS idx_chat_history_workspace;";

/// One-shot data migration (`37`).
///
/// Earlier versions wrote the DeepSeek provider routing value as OpenRouter's
/// *display name* (`DeepSeek`), which is not the provider slug OpenRouter
/// resolves, so those pins silently did not apply (and, with
/// `allow_fallbacks: false`, failed the request). Only that exact value is
/// rewritten; every other stored order survives verbatim.
const REWRITE_LEGACY_DEEPSEEK_ROUTING_SLUG: &str = "UPDATE config_model_routing SET provider_order = 'deepseek' WHERE provider_order = 'DeepSeek';";

/// Catalog `44`: rewrite the retired `in_review`/`in_qa` stages to the merged
/// `verification` stage — a data migration; no schema change.
///
/// `tickets.phase`, `ticket_chronicle.source_phase` and
/// `ticket_chronicle.target_phase` are untyped `TEXT` with no CHECK constraint,
/// so only the stored strings move — a ticket's stage state is entirely its
/// phase string, so nothing else about it has to change. Ticket comments are NOT
/// touched: the retired stage labels a past round's comment carries are
/// history, they are displayed as written, and no read path parses them.
///
/// `last_transition_actor` is deliberately left alone, unlike every service-side
/// phase write (see `board::PIPELINE_ACTOR`'s invariant): that column records
/// the actor of a transition, and this rewrite is not one — the ticket's last
/// real transition still is. The invariant's misattribution hazard cannot arise
/// here either: the catalog runs before CDC capture is enabled on the connection
/// (see `db::open_consolidated_store`), so this UPDATE is never captured and
/// materializes no chronicle hop to attribute.
///
/// The retired `jobs` rows are DELETED, **never renamed**. `jobs.kind` for a
/// ticket phase job equals the ticket's phase, and the partial unique index
/// `idx_jobs_phase_ticket` on `jobs(kind, ticket_id)` WHERE
/// `ticket_id IS NOT NULL` admits one row per (kind, ticket_id) pair. A single
/// ticket can hold both an `in_review` and an `in_qa` row — the window between
/// a stage's transition commit and its job deletion, and stale rows are
/// purge-immune while their workspace is paused — so renaming both to
/// `verification` would collide and abort this entry's transaction, and a
/// failed SQL migration is a HARD BOOT REFUSAL, which this change must never
/// cause.
///
/// Deleting the retired rows also discards their roster (`agents.job_id`
/// references `jobs(id)` `ON DELETE CASCADE`), which is exactly what the merge
/// requires: a round interrupted by the upgrade is re-driven as a whole — the
/// puller creates a fresh `verification` job for a ticket now in the
/// verification phase — so partial work from the old two-stage arrangement is
/// never mistaken for a completed check.
///
/// The chronicle's retired→retired hops (`in_review → in_qa` and the like) are
/// DELETED *before* the rewrite, not rewritten into `verification →
/// verification` self-transitions: the two ends are one stage now, so the hop
/// between them never happened. The remaining rewrites cannot collide on the
/// unique `idx_ticket_chronicle_dedup` (`ticket_id, workspace_name,
/// source_phase, target_phase, at`): `at` is the transition's own `updated_at`
/// (microsecond resolution), so no two hops of one ticket share it — and without
/// that, one ticket's `in_review → X` and `in_qa → X` rows would be exactly the
/// pair sharing every other key column, which the rewrite merges. Idempotent:
/// re-running matches no rows.
///
/// One-way, like every phase rename: a binary that predates this entry parses
/// phase strings strictly and cannot read `verification`, so a manual downgrade
/// (not a supported path) leaves tickets in this phase unreadable — the same
/// hazard the `ready_for_development` → `queued` rename carried.
const REWRITE_RETIRED_STAGES_TO_VERIFICATION: &str = "\
DELETE FROM jobs WHERE kind IN ('in_review', 'in_qa');\
DELETE FROM ticket_chronicle WHERE source_phase IN ('in_review', 'in_qa') \
  AND target_phase IN ('in_review', 'in_qa');\
UPDATE tickets SET phase = 'verification' WHERE phase IN ('in_review', 'in_qa');\
UPDATE ticket_chronicle SET source_phase = 'verification' WHERE source_phase IN ('in_review', 'in_qa');\
UPDATE ticket_chronicle SET target_phase = 'verification' WHERE target_phase IN ('in_review', 'in_qa');";

/// Catalog `46`: rebuild the ticket-title FTS index in the engine's 0.8.0
/// on-disk layout. A no-op on stores that never had it (a fresh install's
/// baseline already created it) and a one-shot rebuild on every store that
/// did.
///
/// Turso 0.8.0's FTS rewrite added two document-identity fast fields to every
/// segment without bumping the storage format number, so an index written by
/// an earlier engine is not refused at open — it fails the moment a segment
/// carrying it is read, and only some operations touch that path (search can
/// break while the rest of the store works). `CREATE INDEX IF NOT EXISTS`
/// alone could not fix it: it returns before touching an index that already
/// exists, so the rebuild has to DROP first. The DDL is repeated from
/// [`BASELINE_CORE_SCHEMA`] by design — the catalog is the only place that
/// defines schema, and a `CREATE` that diverged from the baseline would leave
/// the store's `sqlite_master` text changed. Running inside the entry's
/// transaction makes the drop and the recreate atomic: a failed rebuild rolls
/// the drop back and the boot is refused with the index still intact.
const REBUILD_TICKET_TITLE_FTS_INDEX: &str = "\
DROP INDEX IF EXISTS idx_tickets_title_fts; \
CREATE INDEX IF NOT EXISTS idx_tickets_title_fts ON tickets USING fts (title) WITH (tokenizer = 'ngram');";

/// Catalog `48`: rewrite the retired `in_diagnostics` stage to the merged
/// `verification` stage — a data migration; no schema change.
///
/// `tickets.phase`, `ticket_chronicle.source_phase` and
/// `ticket_chronicle.target_phase` are untyped `TEXT` with no CHECK constraint,
/// so only the stored strings move — a ticket's stage state is entirely its
/// phase string, so nothing else about it has to change. Ticket comments are NOT
/// touched: the retired stage label a past round's comment carries is history,
/// it is displayed as written, and no read path parses it.
///
/// The retired `jobs` rows are DELETED, **never renamed**. `jobs.kind` for a
/// ticket phase job equals the ticket's phase, and the partial unique index
/// `idx_jobs_phase_ticket` on `jobs(kind, ticket_id)` WHERE
/// `ticket_id IS NOT NULL` admits one row per (kind, ticket_id) pair: renaming
/// an `in_diagnostics` row to `verification` would collide with a row already
/// carrying the merged stage (the window between a stage's transition commit and
/// its job deletion, and stale rows are purge-immune while their workspace is
/// paused) and abort this entry's transaction, and a failed SQL migration is a
/// HARD BOOT REFUSAL, which this change must never cause.
///
/// Deleting the retired rows also discards their roster (`agents.job_id`
/// references `jobs(id)` `ON DELETE CASCADE`), which is exactly what the merge
/// requires: the ticket moves to the `verification` phase and the puller creates
/// a fresh `verification` job for it, so partial work from the retired phase is
/// never mistaken for a completed check.
///
/// The `in_diagnostics → verification` hop is DELETED *before* the rewrite, not
/// rewritten into a `verification → verification` self-transition: the two ends
/// are one stage now, so the hop between them never happened. The remaining
/// rewrites cannot collide on the unique `idx_ticket_chronicle_dedup`
/// (`ticket_id, workspace_name, source_phase, target_phase, at`): `at` is the
/// transition's own microsecond timestamp, so no two hops of one ticket share
/// it.
///
/// `last_transition_actor` is deliberately left alone, unlike every service-side
/// phase write (see `board::PIPELINE_ACTOR`'s invariant): that column records
/// the actor of a transition, and this rewrite is not one — the ticket's last
/// real transition still is. The invariant's misattribution hazard cannot arise
/// here either: the catalog runs before CDC capture is enabled on the connection
/// (see `db::open_consolidated_store`), so these UPDATEs are never captured and
/// materialize no chronicle hop to attribute.
///
/// Idempotent: re-running matches no rows.
const RETIRE_IN_DIAGNOSTICS_PHASE: &str = "\
DELETE FROM jobs WHERE kind = 'in_diagnostics';\
DELETE FROM ticket_chronicle WHERE source_phase = 'in_diagnostics' \
  AND target_phase = 'verification';\
UPDATE tickets SET phase = 'verification' WHERE phase = 'in_diagnostics';\
UPDATE ticket_chronicle SET source_phase = 'verification' WHERE source_phase = 'in_diagnostics';\
UPDATE ticket_chronicle SET target_phase = 'verification' WHERE target_phase = 'in_diagnostics';";

/// Delta `51` (existing installs): the `custom_roles` table for
/// administrator-defined agent roles. The baseline (`24`) already declares it,
/// so on a fresh install this is `IF NOT EXISTS` no-op; on an existing install
/// it creates the table.
const CUSTOM_ROLES_TABLE: &str = "CREATE TABLE IF NOT EXISTS custom_roles (
    name       TEXT PRIMARY KEY,
    prompt     TEXT NOT NULL,
    tools      TEXT NOT NULL,
    behavior   TEXT NOT NULL,
    created_at TEXT NOT NULL
);";

/// Delta `59` (existing installs): the `telegram_intake` table in its current
/// shape, `claimed_by` included. The baseline (`24`) already declares it, so on a
/// fresh install this is an `IF NOT EXISTS` no-op; on an existing install it
/// creates the table. An install that applied this entry in its earlier revision,
/// before the column was added, is upfilled by delta `60` — an applied id is
/// never re-run.
const TELEGRAM_INTAKE_TABLE: &str = "CREATE TABLE IF NOT EXISTS telegram_intake (
    account_id TEXT NOT NULL,
    update_id  INTEGER NOT NULL,
    state      TEXT NOT NULL,
    payload    TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    claimed_by TEXT,
    PRIMARY KEY (account_id, update_id)
);";

/// Delta `61` (existing installs): the `telegram_bursts` table — the durable
/// record of every released burst, so a complaint about a split burst can be
/// read against the timing that released it. The baseline (`24`) already
/// declares it, so on a fresh install this is an `IF NOT EXISTS` no-op; on an
/// existing install it creates the table. An applied id is never re-run.
const TELEGRAM_BURSTS_TABLE: &str = "CREATE TABLE IF NOT EXISTS telegram_bursts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id   TEXT NOT NULL,
    user_name    TEXT NOT NULL,
    reply_target TEXT NOT NULL,
    parts        INTEGER NOT NULL,
    released_by  TEXT NOT NULL,
    released_at  TEXT NOT NULL,
    part_times   TEXT NOT NULL
);";

/// Delta `62` (existing installs): the personal-workspace change baseline moves
/// out of the `session_metadata` row into a table of its own, and every recorded
/// baseline is carried across. The engine rewrites a row whole on any update, so
/// a baseline held on the session's own row made every small bookkeeping update
/// (activity time, message count, token length) rewrite megabytes and capture a
/// before/after image of it. The baseline is now stored where it changes — once
/// per turn — and read back by joining the session's own row. The baseline
/// (`24`) already declares the new table, so on a fresh install this is an
/// `IF NOT EXISTS` no-op whose copy matches no row.
const PERSONAL_CHANGES_TO_OWN_TABLE: &str = "\
CREATE TABLE IF NOT EXISTS session_personal_changes (
    agent_id TEXT PRIMARY KEY REFERENCES session_metadata(agent_id) ON DELETE CASCADE,
    baseline TEXT NOT NULL
);
INSERT INTO session_personal_changes (agent_id, baseline)
    SELECT agent_id, personal_changes FROM session_metadata WHERE personal_changes IS NOT NULL;";

/// The complete, strictly-linear catalog. **Order is application order.**
///
/// Entries `24`–`26` are the consolidated current-shape baseline; entries
/// `27`–`31` are the upfill deltas that follow it:
/// - `24` builds the full core schema; on existing installs every statement is
///   `IF NOT EXISTS`, so it is a strict no-op.
/// - `25` doubles as the baseline for fresh installs and the genuine upfill of
///   the retired delta `23`'s `chat_history` reply columns — the one migration
///   a 0.5.0 (one-delta-behind) database needs; it is a guarded, idempotent
///   no-op on current-main and fresh installs.
/// - `26` builds the logs baseline; a strict no-op on existing logs stores.
/// - `27` adds `workspaces.maintainer_recommendations` via a guarded Rust body —
///   a real upfill on existing installs, a no-op on fresh ones.
/// - `28` adds `jobs.caller_agent_id` / `session_metadata.created_at` via a
///   guarded Rust body.
/// - `29` adds the per-user `image_gen_model` / `video_model` columns.
/// - `30` adds `jobs.mode` and backfills it from the `caller_agent_id`
///   NULL-sentinel (sync = has caller pin, async = NULL).
/// - `31` adds the `session_metadata.sleep_ended` marker via a guarded Rust
///   body — a real upfill on existing installs, a no-op on fresh ones.
/// - `32` adds the nullable `alarms.command` column — retired storage: the
///   baseline no longer declares it and entry `43` drops it again.
/// - `33` adds the nullable `chat_history.broadcast_id` column.
/// - `34` adds the `tickets.last_transition_actor` and
///   `ticket_chronicle.actor` columns for phase-transition actor attribution.
/// - `35` adds the per-attempt `llm_failures` table to the logs store
///   (baseline `26` already carries it for fresh installs).
/// - `36` is the admin-only workspace-membership data migration.
/// - `37` rewrites the legacy `DeepSeek` provider-routing display name to the
///   canonical `deepseek` slug (a data migration; a no-op on new installs).
/// - `38` adds the nullable `users.granted_tools` column.
/// - `39`/`40` drop the `users.permissions` and `users.selected_role` columns —
///   account kind is the admin's name, and no account stores an agent role.
///   Two entries, never one, so each drop keeps its own body (see
///   [`drop_column_if_missing`]).
/// - `41` deletes every command-armed `alarms` row (a data migration: the
///   command-armed alarm feature is gone, so those rows are removed before any
///   of them can fire again).
/// - `42` adds the nullable `alarms.trigger` column, the alarm's tool-call
///   trigger that replaced the retired shell command (a no-op on fresh
///   installs, whose baseline already declares it).
/// - `43` drops the retired `alarms.command` column.
/// - `44` rewrites the retired `in_review`/`in_qa` stages to the merged
///   `verification` stage: the retired job rows are deleted (a rename would
///   collide on `idx_jobs_phase_ticket`), the retired→retired chronicle hops
///   are deleted, and the remaining ticket/chronicle phase strings are
///   rewritten (see [`REWRITE_RETIRED_STAGES_TO_VERIFICATION`]).
/// - `45` invalidates the stored per-workspace lifecycle descriptions in
///   `workspace_contexts` that still name a retired stage, so no agent is
///   handed the retired two-stage arrangement.
/// - `46` rebuilds the ticket-title FTS index in the engine's 0.8.0 on-disk
///   layout (see [`REBUILD_TICKET_TITLE_FTS_INDEX`]).
/// - `47` adds the nullable `jobs.exec_dir` column, the absolute directory a
///   targeted delegation round runs in (a no-op on fresh installs, whose
///   baseline already declares it).
/// - `48` rewrites the retired `in_diagnostics` stage to the merged
///   `verification` stage: the retired job rows are deleted (a rename could
///   collide on `idx_jobs_phase_ticket`), the `in_diagnostics → verification`
///   chronicle hop is deleted, and the remaining ticket/chronicle phase strings
///   are rewritten (see [`RETIRE_IN_DIAGNOSTICS_PHASE`]).
/// - `49` invalidates the stored per-workspace contexts in `workspace_contexts`
///   that still claim the workspace's project commands already ran before the
///   round, so no agent is handed the retired arrangement.
/// - `50` rewrites the retired image-generation default to its successor in the
///   stored `image_gen_models` picker list and in every account's
///   `users.image_gen_model` pick, matched on the exact (trimmed) id, so a
///   near-miss id and the owner's other picks survive.
/// - `51` adds the `custom_roles` table (fresh installs get it from entry
///   `24`'s `CREATE TABLE`, so this is a no-op there).
/// - `52` adds the `session_metadata.custom_role` /
///   `session_metadata.custom_identifier` / `session_metadata.launch_stopped`
///   columns (a no-op on fresh installs, whose baseline already declares them).
/// - `53` adds the `alarms.role` / `alarms.identifier` / `alarms.workspace`
///   columns that address an alarm to a custom role (a no-op on fresh installs,
///   whose baseline already declares them).
/// - `54` adds the nullable `jobs.custom_launch` column — the JSON
///   [`crate::Launch`] a durable round was dispatched by, so a boot
///   resume re-addresses the round to the caller's custom branch (a no-op on
///   fresh installs, whose baseline already declares it).
/// - `55` adds the nullable `session_metadata.personal_changes` column — the
///   personal-workspace change baseline: retired storage, whose home is the
///   `session_personal_changes` table entry `62` moves it to, so the baseline no
///   longer declares it and entry `63` drops it again.
/// - `56` adds the nullable `session_metadata.reply_to` column — the JSON
///   conversation that asked for a branch's launch (the asking custom role's
///   branch), so a coder branch's report survives a restart and returns there
///   (a no-op on fresh installs, whose baseline already declares it).
/// - `57` adds the nullable `session_metadata.context_block_len` column — the
///   length of a launch session's cached leading context block, recorded when
///   the block is written so a later folder change rebuilds it from what was
///   written rather than from a re-render of live workspace state (a no-op on
///   fresh installs, whose baseline already declares it).
/// - `58` adds the `tool_calls.source` column — which surface performed a
///   recorded call (`agent` or `cli`), so a change a command-line operation
///   made is distinguishable from one the assistant made (a no-op on fresh
///   installs, whose baseline already declares it).
/// - `59` adds the `telegram_intake` table — the Telegram intake's durable
///   record of the incoming updates it has taken up (fresh installs get it from
///   entry `24`'s `CREATE TABLE`, so this is a no-op there).
/// - `60` adds the `telegram_intake.claimed_by` column — the run that holds an event
///   — to a table created without it (entry `59` carries why), a no-op wherever the
///   table already has it.
/// - `61` adds the `telegram_bursts` table — the durable record of every released
///   burst, so a complaint about a split burst can be read against the timing that
///   released it (fresh installs get it from entry `24`'s `CREATE TABLE`, so this is
///   a no-op there).
/// - `62` moves the personal-workspace change baseline off the `session_metadata`
///   row into its own `session_personal_changes` table (see
///   [`PERSONAL_CHANGES_TO_OWN_TABLE`] for why), carrying every recorded
///   baseline across (fresh installs get the table from entry `24`'s
///   `CREATE TABLE`, so the copy matches no row there).
/// - `63` drops the retired `session_metadata.personal_changes` column, a no-op
///   on fresh installs, whose baseline never declares it.
/// - `64` drops the logs store's observability indexes that no product query's
///   plan reaches (see [`DROP_UNUSED_LOGS_INDEXES`]).
/// - `65` drops the core store's indexes that no product query's plan reaches
///   (see [`DROP_UNUSED_CORE_INDEXES`]).
pub(crate) const MIGRATIONS: &[Migration] = &[
    Migration {
        id: "24",
        target: TargetDb::Core,
        body: MigrationBody::Sql(BASELINE_CORE_SCHEMA),
    },
    Migration {
        id: "25",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_chat_history_reply_columns),
    },
    Migration {
        id: "26",
        target: TargetDb::Logs,
        body: MigrationBody::Sql(BASELINE_LOGS_SCHEMA),
    },
    Migration {
        id: "27",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_workspaces_maintainer_recommendations),
    },
    Migration {
        id: "28",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_jobs_caller_agent_and_session_created_at),
    },
    Migration {
        id: "29",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_users_image_and_video_models),
    },
    Migration {
        id: "30",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_jobs_mode),
    },
    Migration {
        id: "31",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_session_metadata_sleep_ended),
    },
    Migration {
        id: "32",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_alarms_command),
    },
    Migration {
        id: "33",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_chat_history_broadcast_id),
    },
    Migration {
        id: "34",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_ticket_transition_actor),
    },
    Migration {
        id: "35",
        target: TargetDb::Logs,
        body: MigrationBody::Sql(ADD_LLM_FAILURES),
    },
    Migration {
        id: "36",
        target: TargetDb::Core,
        body: MigrationBody::Rust(detach_guest_workspaces),
    },
    Migration {
        id: "37",
        target: TargetDb::Core,
        body: MigrationBody::Sql(REWRITE_LEGACY_DEEPSEEK_ROUTING_SLUG),
    },
    Migration {
        id: "38",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_users_granted_tools),
    },
    Migration {
        id: "39",
        target: TargetDb::Core,
        body: MigrationBody::Rust(drop_users_permissions),
    },
    Migration {
        id: "40",
        target: TargetDb::Core,
        body: MigrationBody::Rust(drop_users_selected_role),
    },
    Migration {
        id: "41",
        target: TargetDb::Core,
        body: MigrationBody::Rust(remove_command_alarms),
    },
    Migration {
        id: "42",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_alarms_trigger),
    },
    Migration {
        id: "43",
        target: TargetDb::Core,
        body: MigrationBody::Rust(drop_alarms_command),
    },
    Migration {
        id: "44",
        target: TargetDb::Core,
        body: MigrationBody::Sql(REWRITE_RETIRED_STAGES_TO_VERIFICATION),
    },
    Migration {
        id: "45",
        target: TargetDb::Core,
        body: MigrationBody::Rust(drop_retired_lifecycle_descriptions),
    },
    Migration {
        id: "46",
        target: TargetDb::Core,
        body: MigrationBody::Sql(REBUILD_TICKET_TITLE_FTS_INDEX),
    },
    Migration {
        id: "47",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_jobs_exec_dir),
    },
    Migration {
        id: "48",
        target: TargetDb::Core,
        body: MigrationBody::Sql(RETIRE_IN_DIAGNOSTICS_PHASE),
    },
    Migration {
        id: "49",
        target: TargetDb::Core,
        body: MigrationBody::Rust(drop_retired_commands_claim_descriptions),
    },
    Migration {
        id: "50",
        target: TargetDb::Core,
        body: MigrationBody::Rust(retire_old_image_gen_model),
    },
    Migration {
        id: "51",
        target: TargetDb::Core,
        body: MigrationBody::Sql(CUSTOM_ROLES_TABLE),
    },
    Migration {
        id: "52",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_session_metadata_custom_role),
    },
    Migration {
        id: "53",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_alarms_custom_target),
    },
    Migration {
        id: "54",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_jobs_custom_launch),
    },
    Migration {
        id: "55",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_session_metadata_personal_changes),
    },
    Migration {
        id: "56",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_session_metadata_reply_to),
    },
    Migration {
        id: "57",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_session_metadata_context_block_len),
    },
    Migration {
        id: "58",
        target: TargetDb::Logs,
        body: MigrationBody::Rust(add_tool_calls_source),
    },
    Migration {
        id: "59",
        target: TargetDb::Core,
        body: MigrationBody::Sql(TELEGRAM_INTAKE_TABLE),
    },
    Migration {
        id: "60",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_telegram_intake_claimed_by),
    },
    Migration {
        id: "61",
        target: TargetDb::Core,
        body: MigrationBody::Sql(TELEGRAM_BURSTS_TABLE),
    },
    Migration {
        id: "62",
        target: TargetDb::Core,
        body: MigrationBody::Sql(PERSONAL_CHANGES_TO_OWN_TABLE),
    },
    Migration {
        id: "63",
        target: TargetDb::Core,
        body: MigrationBody::Rust(drop_personal_changes_column),
    },
    Migration {
        id: "64",
        target: TargetDb::Logs,
        body: MigrationBody::Sql(DROP_UNUSED_LOGS_INDEXES),
    },
    Migration {
        id: "65",
        target: TargetDb::Core,
        body: MigrationBody::Sql(DROP_UNUSED_CORE_INDEXES),
    },
];

/// Apply the migration catalog to `conn` for one physical database.
///
/// Iterates [`MIGRATIONS`] in order, skipping entries that target a different
/// database or whose id is already recorded in `schema_migrations`. Each
/// SQL entry runs inside its own transaction (schema change + tracking row
/// commit atomically); each Rust entry runs without a surrounding transaction.
/// A failure propagates — it is a hard boot failure, never a reason to replace
/// the store.
pub(crate) async fn run_migrations(conn: &Connection, db: TargetDb) -> anyhow::Result<()> {
    run_catalog(conn, db, MIGRATIONS).await
}

/// The ledger table every product install carries: the catalog records applied
/// ids in it, and the boot usability gate ([`crate::db::has_product_schema`])
/// uses it as the marker of "built by this product".
pub(crate) const MIGRATIONS_TABLE: &str = "schema_migrations";

/// The migration loop, parameterized by a catalog so tests can replay the
/// retired `1`–`23` chain (see the test fixtures). Creates [`MIGRATIONS_TABLE`],
/// reads the applied ids, and runs each entry once, in catalog order, targeting
/// only `db`.
async fn run_catalog(conn: &Connection, db: TargetDb, catalog: &[Migration]) -> anyhow::Result<()> {
    conn.execute(
        &format!(
            "CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (\
             id         TEXT PRIMARY KEY,\
             applied_at TEXT NOT NULL\
         )"
        ),
        (),
    )
    .await
    .context("Failed to create schema_migrations tracking table")?;

    let applied: HashSet<String> = conn
        .query(&format!("SELECT id FROM {MIGRATIONS_TABLE}"), ())
        .await
        .context("Failed to read applied migrations")?
        .into_iter()
        .filter_map(|row| row.get::<String>(0).ok())
        .collect();

    for migration in catalog {
        if migration.target != db {
            continue;
        }
        if applied.contains(migration.id) {
            continue;
        }
        match migration.body {
            MigrationBody::Sql(sql) => {
                let tx = conn.begin_tx().await.with_context(|| {
                    format!("Migration '{}': failed to begin transaction", migration.id)
                })?;
                tx.execute_batch(sql)
                    .await
                    .with_context(|| format!("Migration '{}' failed", migration.id))?;
                tx.execute(
                    &format!("INSERT INTO {MIGRATIONS_TABLE} (id, applied_at) VALUES (?1, ?2)"),
                    params![migration.id, crate::db::now()],
                )
                .await
                .with_context(|| {
                    format!("Migration '{}': failed to record as applied", migration.id)
                })?;
                tx.commit()
                    .await
                    .with_context(|| format!("Migration '{}': failed to commit", migration.id))?;
            }
            MigrationBody::Rust(run) => {
                // Unlike an SQL entry, a Rust body and its tracking row are not
                // recorded atomically (the body runs without a transaction). This
                // is safe because every Rust body probes (or is written to be
                // indifferent to) the state it applies, so it is idempotent and
                // re-runnable: a crash between body success and id recording
                // re-runs the body without corruption on the next boot.
                run(conn)
                    .await
                    .with_context(|| format!("Migration '{}' failed", migration.id))?;
                conn.execute(
                    &format!("INSERT INTO {MIGRATIONS_TABLE} (id, applied_at) VALUES (?1, ?2)"),
                    params![migration.id, crate::db::now()],
                )
                .await
                .with_context(|| {
                    format!("Migration '{}': failed to record as applied", migration.id)
                })?;
            }
        }
    }
    Ok(())
}

// ── Rust-function migrations (called without a surrounding transaction) ──

fn add_chat_history_reply_columns(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_chat_history_reply_columns(conn))
}

/// Upfill the retired delta `23` (`chat_history.reply_author` /
/// `reply_snippet`) for databases that were fully current before the catalog
/// consolidation: a 0.5.0 database has neither column, a current-main database
/// already has both, and a fresh install got them from entry `24`'s
/// `CREATE TABLE`. Guarded by [`add_column_if_missing`], so this body is
/// idempotent (non-transactional like every Rust body, re-runnable until
/// recorded).
async fn run_add_chat_history_reply_columns(conn: &Connection) -> anyhow::Result<()> {
    for column in ["reply_author", "reply_snippet"] {
        add_column_if_missing(conn, "chat_history", column, "TEXT").await?;
    }
    Ok(())
}

fn add_chat_history_broadcast_id(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_chat_history_broadcast_id(conn))
}

/// Upfill the `chat_history.broadcast_id` column for databases created before
/// delta `33`: an id tagging the delivery dispatch that wrote an agent row.
/// Fresh installs get the column from entry `24`'s `CREATE TABLE`, so the probe
/// makes this a no-op there. Guarded by [`add_column_if_missing`], so this body
/// is idempotent (non-transactional like every Rust body, re-runnable until
/// recorded).
async fn run_add_chat_history_broadcast_id(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "chat_history", "broadcast_id", "TEXT").await
}

fn add_ticket_transition_actor(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_ticket_transition_actor(conn))
}

/// Upfill the phase-transition actor columns for databases created before
/// delta `34`: `tickets.last_transition_actor` (the acting identity written by
/// every phase-changing UPDATE) and `ticket_chronicle.actor` (the same value
/// copied by the CDC materializer). Fresh installs get both columns from entry
/// `24`'s `CREATE TABLE`, so the probes make this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional
/// like every Rust body, re-runnable until recorded). No backfill — legacy
/// rows fall back to `"system"` at render time.
async fn run_add_ticket_transition_actor(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "tickets", "last_transition_actor", "TEXT").await?;
    add_column_if_missing(conn, "ticket_chronicle", "actor", "TEXT").await
}

fn add_workspaces_maintainer_recommendations(
    conn: &Connection,
) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_workspaces_maintainer_recommendations(conn))
}

/// Upfill `workspaces.maintainer_recommendations` for databases created before
/// delta `27` (0.5.x live installs). Fresh installs get the column from entry
/// `24`'s `CREATE TABLE`, so the probe makes this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional
/// like every Rust body, re-runnable until recorded). Holds a replace-only
/// JSON blob of maintainer recommendations:
/// {"recommendations": [...], "generated_at": RFC3339}.
async fn run_add_workspaces_maintainer_recommendations(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "workspaces", "maintainer_recommendations", "TEXT").await
}

/// Return whether `column` exists in `table` (via `PRAGMA table_info`).
///
/// This is the idempotency probe used by the guarded Rust migration bodies:
/// Turso has no `ADD COLUMN IF NOT EXISTS` (or `DROP COLUMN IF EXISTS`), so a
/// body probes before altering rather than relying on an `IF NOT EXISTS`.
async fn column_exists(conn: &Connection, table: &str, column: &str) -> anyhow::Result<bool> {
    let found = conn
        .query(&format!("PRAGMA table_info({table})"), ())
        .await
        .with_context(|| format!("Failed to probe {table}.{column}"))?
        .into_iter()
        .any(|row| row.get::<String>(1).ok().as_deref() == Some(column));
    Ok(found)
}

/// Idempotently add a column of SQL type `ty` to `table` (Turso has no
/// `ADD COLUMN IF NOT EXISTS`): probe with [`column_exists`], then
/// `ALTER TABLE`. Used by the guarded migration bodies above and below.
async fn add_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
    ty: &str,
) -> anyhow::Result<()> {
    if !column_exists(conn, table, column).await? {
        conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {column} {ty}"), ())
            .await
            .with_context(|| format!("Failed to add {table}.{column}"))?;
    }
    Ok(())
}

/// Drop `table.column` when it still exists. Turso has no
/// `DROP COLUMN IF [NOT] EXISTS`, and a baseline `CREATE TABLE` already omits
/// a dropped column on fresh installs, so the probe is what makes a drop body
/// idempotent (non-transactional like every Rust body, re-runnable until
/// recorded).
///
/// The table is rewritten by the drop, so a drop body holds exactly one
/// statement. Dropping is one-way: an older binary whose queries still name the
/// dropped column finds no such column, which the catalog treats as a hard
/// failure — a column drop has no downgrade path.
async fn drop_column_if_missing(
    conn: &Connection,
    table: &str,
    column: &str,
) -> anyhow::Result<()> {
    if column_exists(conn, table, column).await? {
        conn.execute(&format!("ALTER TABLE {table} DROP COLUMN {column}"), ())
            .await
            .with_context(|| format!("Failed to drop {table}.{column}"))?;
    }
    Ok(())
}

fn add_jobs_caller_agent_and_session_created_at(
    conn: &Connection,
) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_jobs_caller_agent_and_session_created_at(conn))
}

/// Upfill the sync-resume primitives for databases created before delta `28`:
/// `jobs.caller_agent_id` (the caller agent's stable session pin for sync
/// analyze/implement jobs; NULL for async/boot-resumed jobs) and
/// `session_metadata.created_at` (the session's creation timestamp). Fresh
/// installs get both columns from entry `24`'s `CREATE TABLE`, so the probes
/// make this a no-op there. Idempotent, non-transactional like every Rust body
/// (re-runnable until recorded).
async fn run_add_jobs_caller_agent_and_session_created_at(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "jobs", "caller_agent_id", "TEXT").await?;
    add_column_if_missing(conn, "session_metadata", "created_at", "TEXT").await?;
    Ok(())
}

fn add_users_image_and_video_models(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_users_image_and_video_models(conn))
}

/// Upfill the per-user image/video model columns for databases created before
/// delta `29`. Fresh installs get both columns from entry `24`'s `CREATE
/// TABLE`, so the probe makes this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional
/// like every Rust body, re-runnable until recorded).
async fn run_add_users_image_and_video_models(conn: &Connection) -> anyhow::Result<()> {
    for column in ["image_gen_model", "video_model"] {
        add_column_if_missing(conn, "users", column, "TEXT").await?;
    }
    Ok(())
}

fn add_jobs_mode(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_jobs_mode(conn))
}

/// Upfill the explicit `jobs.mode` sync/async discriminator for databases
/// created before delta `30`. Existing rows are backfilled from the
/// `caller_agent_id` NULL-sentinel (sync = has caller pin, async = NULL).
/// Fresh installs get the column from entry `24`'s `CREATE TABLE`, so the
/// probe makes this a no-op there. Guarded by [`add_column_if_missing`], so
/// this body is idempotent (non-transactional like every Rust body,
/// re-runnable until recorded).
async fn run_add_jobs_mode(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "jobs", "mode", "TEXT").await?;
    // Backfill idempotently: rows created before the column existed map from the
    // caller_agent_id NULL-sentinel (sync = has caller pin, async = NULL).
    conn.execute(
        "UPDATE jobs \
         SET mode = CASE WHEN caller_agent_id IS NOT NULL THEN 'sync' ELSE 'async' END \
         WHERE mode IS NULL",
        (),
    )
    .await
    .with_context(|| "Failed to backfill jobs.mode")?;
    Ok(())
}

fn add_jobs_exec_dir(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_jobs_exec_dir(conn))
}

/// Upfill the delegation `jobs.exec_dir` column for databases created
/// before delta `47`: the absolute directory a targeted delegation round
/// runs in (NULL = the caller's own workspace, `jobs.workspace_name`).
/// Fresh installs get the column from entry `24`'s `CREATE TABLE`, so the
/// probe makes this a no-op there. Guarded by [`add_column_if_missing`], so
/// this body is idempotent (non-transactional like every Rust body,
/// re-runnable until recorded).
async fn run_add_jobs_exec_dir(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "jobs", "exec_dir", "TEXT").await
}

fn add_session_metadata_custom_role(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_session_metadata_custom_role(conn))
}

/// Upfill the per-session `custom_role` (the custom-role record this session
/// belongs to), `custom_identifier` (the named branch it continues, when it was
/// named) and `launch_stopped` (its launch was stopped before it produced
/// anything) columns for databases created before delta `52`. Fresh installs
/// get the columns from entry `24`'s `CREATE TABLE`, so the probe makes this a
/// no-op there. Guarded by [`add_column_if_missing`], so this body is
/// idempotent (non-transactional like every Rust body, re-runnable until
/// recorded).
async fn run_add_session_metadata_custom_role(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "session_metadata", "custom_role", "TEXT").await?;
    add_column_if_missing(conn, "session_metadata", "custom_identifier", "TEXT").await?;
    add_column_if_missing(conn, "session_metadata", "launch_stopped", "TEXT").await
}

fn add_alarms_custom_target(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_alarms_custom_target(conn))
}

/// Upfill the nullable `alarms.role` / `alarms.identifier` / `alarms.workspace`
/// columns for databases created before delta `53` — the custom role, its
/// branch identifier and the workspace an alarm is addressed to. Fresh installs
/// get the columns from entry `24`'s `CREATE TABLE`, so the probe makes this a
/// no-op there. Guarded by [`add_column_if_missing`], so this body is
/// idempotent (non-transactional like every Rust body, re-runnable until
/// recorded).
async fn run_add_alarms_custom_target(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "alarms", "role", "TEXT").await?;
    add_column_if_missing(conn, "alarms", "identifier", "TEXT").await?;
    add_column_if_missing(conn, "alarms", "workspace", "TEXT").await
}

fn add_jobs_custom_launch(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_jobs_custom_launch(conn))
}

/// Upfill the nullable `jobs.custom_launch` column for databases created
/// before delta `54` — the JSON [`crate::Launch`] a durable round was
/// dispatched by (`NULL` for every built-in caller). Fresh installs get the
/// column from entry `24`'s `CREATE TABLE`, so the probe makes this a no-op
/// there. Guarded by [`add_column_if_missing`], so this body is idempotent
/// (non-transactional like every Rust body, re-runnable until recorded).
async fn run_add_jobs_custom_launch(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "jobs", "custom_launch", "TEXT").await
}

fn add_session_metadata_personal_changes(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_session_metadata_personal_changes(conn))
}

/// Upfill the nullable `session_metadata.personal_changes` column for databases
/// created before delta `55` — the per-user personal-workspace change baseline
/// (JSON: relative path → size/mtime/content-digest) the Assistant reads at the
/// head of a turn and refreshes after it. The baseline no longer declares the
/// column — its home is now the `session_personal_changes` table — so on a fresh
/// install this adds `personal_changes`, delta `62` carries it across and delta
/// `63` drops it again. Guarded by [`add_column_if_missing`], so this body is
/// idempotent (non-transactional like every Rust body, re-runnable until
/// recorded).
async fn run_add_session_metadata_personal_changes(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "session_metadata", "personal_changes", "TEXT").await
}

fn add_session_metadata_reply_to(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_session_metadata_reply_to(conn))
}

/// Upfill the nullable `session_metadata.reply_to` column for databases created
/// before delta `56` — the JSON conversation that asked for a branch's launch
/// (the asking custom role's branch and the folder its turn ran in), so a coder
/// branch's report survives a restart and returns there. Fresh installs get the
/// column from entry `24`'s `CREATE TABLE`, so the probe makes this a no-op
/// there. Guarded by [`add_column_if_missing`], so this body is idempotent
/// (non-transactional like every Rust body, re-runnable until recorded).
async fn run_add_session_metadata_reply_to(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "session_metadata", "reply_to", "TEXT").await
}

fn add_session_metadata_context_block_len(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_session_metadata_context_block_len(conn))
}

/// Upfill the nullable `session_metadata.context_block_len` column for databases
/// created before delta `57` — the length of a launch session's cached leading
/// context block, recorded when the block is written so a later folder change
/// rebuilds it from what was written rather than from a re-render of live
/// workspace state. Fresh installs get the column from entry `24`'s
/// `CREATE TABLE`, so the probe makes this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional like
/// every Rust body, re-runnable until recorded).
async fn run_add_session_metadata_context_block_len(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "session_metadata", "context_block_len", "INTEGER").await
}

fn add_tool_calls_source(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_tool_calls_source(conn))
}

/// Upfill the `tool_calls.source` column for databases created before delta
/// `58` — which surface performed the recorded call (`agent` or `cli`), so a
/// change a command-line operation made is distinguishable from one the
/// assistant made. Fresh installs get the column (with its `agent` default)
/// from entry `26`'s `CREATE TABLE`, so the probe makes this a no-op there.
/// Guarded by [`add_column_if_missing`], so this body is idempotent
/// (non-transactional like every Rust body, re-runnable until recorded).
async fn run_add_tool_calls_source(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(
        conn,
        "tool_calls",
        "source",
        "TEXT NOT NULL DEFAULT 'agent'",
    )
    .await
}

fn add_telegram_intake_claimed_by(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_telegram_intake_claimed_by(conn))
}

/// Guarantee the `telegram_intake.claimed_by` column — the run that holds an event
/// — on a database whose table was created without it (see entry `59`). A table
/// that already carries the column (every fresh install, from the baseline's
/// `CREATE TABLE`) is left alone. Guarded by [`add_column_if_missing`], so this
/// body is idempotent (non-transactional like every Rust body, re-runnable until
/// recorded).
async fn run_add_telegram_intake_claimed_by(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "telegram_intake", "claimed_by", "TEXT").await
}

fn add_session_metadata_sleep_ended(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_session_metadata_sleep_ended(conn))
}

/// Upfill the per-session `sleep_ended` marker column for databases created
/// before delta `31`. Fresh installs get the column from entry `24`'s
/// `CREATE TABLE`, so the probe makes this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional
/// like every Rust body, re-runnable until recorded).
async fn run_add_session_metadata_sleep_ended(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "session_metadata", "sleep_ended", "TEXT").await
}

fn add_alarms_command(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_alarms_command(conn))
}

/// Upfill the nullable `alarms.command` column for databases created before
/// delta `32`. The baseline no longer declares the column — the command-armed
/// alarm feature is retired and the baseline carries its replacement `trigger`
/// — so on a fresh install this adds `command` and delta `43` drops it again.
/// Guarded by [`add_column_if_missing`], so this body is idempotent
/// (non-transactional like every Rust body, re-runnable until recorded).
async fn run_add_alarms_command(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "alarms", "command", "TEXT").await
}

fn detach_guest_workspaces(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_detach_guest_workspaces(conn))
}

/// Data migration: workspace membership becomes admin-only.
///
/// Detaches every guest from shared workspaces by clearing
/// `users.selected_workspace` (NULL = the account's personal workspace).
///
/// The admin predicate is exactly the account-name test — the same single
/// predicate the runtime guards use, so migration and runtime can never
/// disagree. A body naming the legacy account-kind token would not parse on a
/// fresh install: entry `24`'s baseline never creates it, and entries `39`/`40`
/// drop it on every existing install.
/// Accounts, chat history, personal workspaces and alarms are preserved —
/// detach, not deletion. Idempotent: re-running matches no rows
/// (non-transactional like every Rust body, re-runnable until recorded).
async fn run_detach_guest_workspaces(conn: &Connection) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE users SET selected_workspace = NULL \
         WHERE selected_workspace IS NOT NULL AND name <> ?1",
        params![crate::users::ADMIN_USER_NAME],
    )
    .await
    .with_context(|| "Failed to detach guests from shared workspaces")?;
    Ok(())
}

fn add_users_granted_tools(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_users_granted_tools(conn))
}

/// Upfill the nullable `users.granted_tools` column for databases created
/// before delta `38`. Fresh installs get the column from entry `24`'s
/// `CREATE TABLE`, so the probe makes this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional
/// like every Rust body, re-runnable until recorded). Holds the JSON array of
/// custom tool names granted to the user; NULL/empty means no grants.
async fn run_add_users_granted_tools(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "users", "granted_tools", "TEXT").await
}

fn drop_users_permissions(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_drop_users_permissions(conn))
}

/// Drop `users.permissions` for databases created before delta `39`.
///
/// The column held the account-kind token (`'full'` = admin, nothing = guest),
/// duplicating what the account's name already says; the admin's name is the
/// only account-kind marker now. An account that held `'full'` under a name
/// other than the admin's becomes a guest and keeps its stored workspace
/// selection, which the runtime clamp resolves to its personal workspace — the
/// clamp is the enforcement point, this body does not rewrite the selection.
async fn run_drop_users_permissions(conn: &Connection) -> anyhow::Result<()> {
    drop_column_if_missing(conn, "users", "permissions").await
}

fn drop_users_selected_role(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_drop_users_selected_role(conn))
}

/// Drop `users.selected_role` for databases created before delta `40`.
///
/// The column stored a per-account agent-role selection that always resolved
/// to the single Assistant and therefore decided nothing.
async fn run_drop_users_selected_role(conn: &Connection) -> anyhow::Result<()> {
    drop_column_if_missing(conn, "users", "selected_role").await
}

fn remove_command_alarms(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_remove_command_alarms(conn))
}

/// Data migration: delete every command-armed alarm row — active or long
/// settled — before any of them can fire again. The command-armed alarm
/// feature is gone, so nothing is converted and no notice is left on those
/// rows; entry `43` then retires the column itself.
///
/// Guarded by [`column_exists`]: every normal path reaches this entry after
/// `32` added the column, so the probe passes, but a hard-coded
/// `DELETE … WHERE command IS NOT NULL` would be a hard boot failure on a store
/// whose `alarms` table lacks it — and catalog errors are never absorbed.
async fn run_remove_command_alarms(conn: &Connection) -> anyhow::Result<()> {
    if !column_exists(conn, "alarms", "command").await? {
        return Ok(());
    }
    conn.execute("DELETE FROM alarms WHERE command IS NOT NULL", ())
        .await
        .with_context(|| "Failed to delete command-armed alarms")?;
    Ok(())
}

fn add_alarms_trigger(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_alarms_trigger(conn))
}

/// Upfill the nullable `alarms.trigger` column for databases created before
/// delta `42`. Fresh installs get the column from entry `24`'s `CREATE TABLE`,
/// so the probe makes this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional
/// like every Rust body, re-runnable until recorded). Holds the alarm's
/// tool-call trigger: `{"tool": ..., "args": {...}}`.
async fn run_add_alarms_trigger(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "alarms", "trigger", "TEXT").await
}

fn drop_alarms_command(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_drop_alarms_command(conn))
}

/// Drop `alarms.command` for databases created before delta `43`.
///
/// The column armed an alarm with a shell command, a feature that no longer
/// exists; entry `41` has already deleted every row that carried one, so the
/// drop retires storage only. Nothing else references the column: it is not
/// indexed and not part of a PK/UNIQUE/CHECK/FK constraint.
async fn run_drop_alarms_command(conn: &Connection) -> anyhow::Result<()> {
    drop_column_if_missing(conn, "alarms", "command").await
}

fn drop_personal_changes_column(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_drop_personal_changes_column(conn))
}

/// Drop `session_metadata.personal_changes` once delta `62` has carried every
/// value into `session_personal_changes`. The baseline no longer declares the
/// column — its home is that table — so on a fresh install there is nothing to
/// drop; delta `55` re-adds the column there and this body removes it again (see
/// [`run_add_session_metadata_personal_changes`]). Guarded by
/// [`drop_column_if_missing`], so this body is idempotent (non-transactional
/// like every Rust body, re-runnable until recorded).
async fn run_drop_personal_changes_column(conn: &Connection) -> anyhow::Result<()> {
    drop_column_if_missing(conn, "session_metadata", "personal_changes").await
}

/// Case-folded markers of a stored description of the retired two-stage
/// review-then-QA arrangement, matched as substrings of the content after
/// [`describes_retired_lifecycle`]'s normalization.
///
/// They cover the spellings the retired phases were actually written in
/// (`in_review`, `in review`, `InReview`), the lifecycle-list spellings that
/// drop the `in` (`→ Review`, `Review →`, `Review/QA`, `review and QA`), and the
/// retired stages named as the two ends of one pipeline step.
const RETIRED_LIFECYCLE_MARKERS: &[&str] = &[
    "in_review",
    "in review",
    "inreview",
    "in_qa",
    "in qa",
    "inqa",
    "→ review",
    "review →",
    "→ qa",
    "qa →",
    "review/qa",
    "qa/review",
];

/// Whether a stored workspace context describes the retired arrangement.
///
/// Deliberately greedy pipeline vocabulary, not a parser: the content is
/// case-folded, its stage-list separators (` and `, `, `, ` then `) are
/// normalized to `/`, and the `reviewer`/`reviewers` role names are folded out —
/// so `Review/QA`, `review and QA` and `review then QA` all hit, while a role
/// list like `analyst/coder/qa/reviewer/sanitation` does not (the reviewer role
/// survives the merge; only the retired *stage* names are probed, and the fold's
/// placeholder carries no stage word).
fn describes_retired_lifecycle(content: &str) -> bool {
    let mut folded = content.to_ascii_lowercase();
    for separator in [" and ", ", ", " then "] {
        folded = folded.replace(separator, "/");
    }
    folded = folded
        .replace("reviewers", "role_name")
        .replace("reviewer", "role_name");
    RETIRED_LIFECYCLE_MARKERS
        .iter()
        .any(|marker| folded.contains(marker))
}

fn drop_retired_lifecycle_descriptions(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_drop_retired_lifecycle_descriptions(conn))
}

/// Data migration (`45`): no stored workspace context may hand an agent the
/// retired two-stage arrangement.
///
/// `workspace_contexts` holds discovery-generated prose per
/// (`workspace_name`, `role`), and a row that names the retired stages
/// describes the pipeline's lifecycle — including the module split and the
/// review-then-QA order that the merge removed. No partial rewrite can make such
/// a description true, so the row is DELETED: the value is a derived cache that
/// the workspace's next re-analysis (the manual Re-analyze, or the nightly
/// new-commit pass) regenerates from the current source, a missing context
/// degrades to the file-derived fallback rather than to an error, and a stored
/// description of the retired arrangement is worse than a missing one.
///
/// The probe behind it is a heuristic in both directions: prose that describes
/// the retired order without naming a retired stage or joining the two as one
/// lifecycle step (e.g. "the code is reviewed, then tested") survives and waits
/// for the next re-analysis, while a context that uses `review` as its own
/// subject (e.g. "review → merge") is dropped unnecessarily. Idempotent and
/// non-transactional like every Rust body: a second run matches nothing.
async fn run_drop_retired_lifecycle_descriptions(conn: &Connection) -> anyhow::Result<()> {
    let rows = conn
        .query("SELECT rowid, content FROM workspace_contexts", ())
        .await
        .context("Failed to read stored workspace contexts")?;
    for row in rows {
        let content = row
            .get::<String>(1)
            .context("Failed to read workspace_contexts.content")?;
        if !describes_retired_lifecycle(&content) {
            continue;
        }
        let rowid = row
            .get::<i64>(0)
            .context("Failed to read workspace_contexts.rowid")?;
        conn.execute(
            "DELETE FROM workspace_contexts WHERE rowid = ?1",
            params![rowid],
        )
        .await
        .context("Failed to drop a stale stored workspace context")?;
    }
    Ok(())
}

/// Markers of the retired claim that the workspace's project commands already
/// ran before the round, matched as substrings of the case-folded content.
const RETIRED_COMMANDS_CLAIM_MARKERS: &[&str] = &[
    "already ran",
    "already been run",
    "already verified",
    "already been verified",
    "ran before",
    "run before this round",
    "diagnostics phase",
    "diagnostics stage",
    "no need to repeat",
];

/// Whether a stored workspace context claims the project commands ran before
/// the round.
///
/// Deliberately phrase-level, never a bare `ran`: the content is case-folded
/// and the markers cover the spellings the retired arrangement was actually
/// written in, so prose that merely mentions a command running in this round
/// survives.
fn claims_commands_ran_before_the_round(content: &str) -> bool {
    let folded = content.to_ascii_lowercase();
    RETIRED_COMMANDS_CLAIM_MARKERS
        .iter()
        .any(|marker| folded.contains(marker))
}

fn drop_retired_commands_claim_descriptions(
    conn: &Connection,
) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_drop_retired_commands_claim_descriptions(conn))
}

/// Data migration (`49`): no stored workspace context may hand an agent the
/// retired claim that the workspace's project commands already ran before the
/// round.
///
/// `workspace_contexts` holds discovery-generated prose per
/// (`workspace_name`, `role`). Since the merge the project commands run INSIDE
/// the verification round, in parallel with the participants, and their result
/// is the same round comment — so a row that claims they already ran (and tells
/// the agent not to repeat them) describes the retired arrangement and
/// contradicts the round's own brief. No partial rewrite can make such a claim
/// true, so the row is DELETED: the value is a derived cache that the
/// workspace's next re-analysis (the manual Re-analyze, or the nightly
/// new-commit pass) regenerates from the current source, and a missing context
/// degrades to the file-derived fallback rather than to an error.
///
/// The probe behind it is a heuristic in both directions: prose that describes
/// the same thing without naming a marker phrase survives and waits for the next
/// re-analysis, and the markers are deliberately phrase-level (never a bare
/// `ran`), so a context that merely mentions a command running in this round is
/// kept. Idempotent and non-transactional like every Rust body: a second run
/// matches nothing.
async fn run_drop_retired_commands_claim_descriptions(conn: &Connection) -> anyhow::Result<()> {
    let rows = conn
        .query("SELECT rowid, content FROM workspace_contexts", ())
        .await
        .context("Failed to read stored workspace contexts")?;
    for row in rows {
        let content = row
            .get::<String>(1)
            .context("Failed to read workspace_contexts.content")?;
        if !claims_commands_ran_before_the_round(&content) {
            continue;
        }
        let rowid = row
            .get::<i64>(0)
            .context("Failed to read workspace_contexts.rowid")?;
        conn.execute(
            "DELETE FROM workspace_contexts WHERE rowid = ?1",
            params![rowid],
        )
        .await
        .context("Failed to drop a stale stored workspace context")?;
    }
    Ok(())
}

/// Catalog `50`: the image-generation default that the curated set retired.
const RETIRED_IMAGE_GEN_MODEL: &str = "google/gemini-3.1-flash-image";
/// Catalog `50`: the model the retired one is replaced by.
const SUCCESSOR_IMAGE_GEN_MODEL: &str = "google/gemini-nano-banana-2.1";
/// Catalog `50`: the `config_kv` key holding the newline-separated picker list.
const IMAGE_GEN_MODELS_KV_KEY: &str = "image_gen_models";

fn retire_old_image_gen_model(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_retire_old_image_gen_model(conn))
}

/// Data migration (`50`): the retired image-generation default is replaced by
/// its successor on both user-visible surfaces — the `image_gen_models`
/// `config_kv` picker list and every account's `users.image_gen_model` pick.
///
/// The picker list is owner-edited free text, so only the exact id is replaced:
/// repetition and order are the owner's, a duplicate the replacement produces is
/// left alone, and a near-miss id (e.g. `…-preview`) survives. Matching is on
/// the trimmed line, the same normalisation the read path applies, so a
/// whitespace-padded pick is still recognised and becomes the bare successor;
/// every other line is kept byte-identical. The list row is only written when it
/// actually changed, so a second run is a strict no-op on it.
///
/// Runs once, ever: the guarantee comes from the `schema_migrations` ledger —
/// a later deliberate re-add of the retired id by the owner is kept, which an
/// every-boot rewrite could not honour.
async fn run_retire_old_image_gen_model(conn: &Connection) -> anyhow::Result<()> {
    let mut list_rewritten = false;

    let rows = conn
        .query(
            "SELECT value FROM config_kv WHERE key = ?1",
            params![IMAGE_GEN_MODELS_KV_KEY],
        )
        .await
        .context("Failed to read the stored image-generation picker list")?;
    if let Some(row) = rows.into_iter().next() {
        let value = row
            .get::<String>(0)
            .context("Failed to read config_kv.value")?;
        let rewritten = value
            .split('\n')
            .map(|line| {
                if line.trim() == RETIRED_IMAGE_GEN_MODEL {
                    SUCCESSOR_IMAGE_GEN_MODEL
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        if rewritten != value {
            conn.execute(
                "UPDATE config_kv SET value = ?1 WHERE key = ?2",
                params![rewritten, IMAGE_GEN_MODELS_KV_KEY],
            )
            .await
            .context("Failed to rewrite the stored image-generation picker list")?;
            list_rewritten = true;
        }
    }

    let mut rewritten_accounts = 0usize;
    let accounts = conn
        .query(
            "SELECT name, image_gen_model FROM users WHERE image_gen_model IS NOT NULL",
            (),
        )
        .await
        .context("Failed to read the stored per-account image-generation picks")?;
    for row in accounts {
        let name = row.get::<String>(0).context("Failed to read users.name")?;
        let pick = row
            .get::<String>(1)
            .context("Failed to read users.image_gen_model")?;
        if pick.trim() != RETIRED_IMAGE_GEN_MODEL {
            continue;
        }
        conn.execute(
            "UPDATE users SET image_gen_model = ?1 WHERE name = ?2",
            params![SUCCESSOR_IMAGE_GEN_MODEL, name],
        )
        .await
        .context("Failed to rewrite a stored per-account image-generation pick")?;
        rewritten_accounts += 1;
    }

    if list_rewritten || rewritten_accounts > 0 {
        tracing::info!(
            list_rewritten,
            accounts = rewritten_accounts,
            "retired image-generation model rewritten to its successor",
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Connection;

    // ── Shared schema-inspection helpers ──────────────────────────────

    async fn column_names(conn: &Connection, table: &str) -> Vec<String> {
        conn.query(&format!("PRAGMA table_info({table})"), ())
            .await
            .expect("read table_info")
            .into_iter()
            .map(|row| row.get::<String>(1).expect("column name"))
            .collect()
    }

    async fn applied_ids(conn: &Connection) -> Vec<String> {
        conn.query("SELECT id FROM schema_migrations", ())
            .await
            .expect("read schema_migrations")
            .into_iter()
            .map(|row| row.get::<String>(0).expect("id"))
            .collect()
    }

    /// Strip SQL `--` line comments (respecting single-quoted string literals)
    /// and remove all whitespace — a cosmetic normalization used by the
    /// structural DDL comparisons, so semantically identical DDL text compares
    /// equal without being brittle to `--` inside a string literal.
    fn normalize_ddl(sql: &str) -> String {
        let mut out = String::new();
        let mut in_string = false;
        let mut chars = sql.chars().peekable();
        while let Some(c) = chars.next() {
            if in_string {
                if c == '\'' {
                    in_string = false;
                }
                out.push(c);
            } else if c == '\'' {
                in_string = true;
                out.push(c);
            } else if c == '-' && chars.peek() == Some(&'-') {
                // Consume the rest of the line comment.
                for c2 in chars.by_ref() {
                    if c2 == '\n' {
                        break;
                    }
                }
            } else if !c.is_whitespace() {
                out.push(c);
            }
        }
        out
    }

    /// Explicitly-created indexes: name → normalized SQL (comments/whitespace
    /// stripped). Excludes constraint autoindexes. This captures the index's
    /// target table/columns, uniqueness, and any partial `WHERE` predicate.
    async fn index_defs(conn: &Connection) -> std::collections::BTreeMap<String, String> {
        conn.query(
            "SELECT name, sql FROM sqlite_master \
             WHERE type = 'index' AND name NOT LIKE 'sqlite_autoindex_%' \
             AND name NOT LIKE '__turso_internal_%' AND name NOT LIKE 'turso_cdc%'",
            (),
        )
        .await
        .expect("read index definitions")
        .into_iter()
        .map(|row| {
            let name = row.get::<String>(0).expect("index name");
            let sql = row
                .get::<Option<String>>(1)
                .expect("index sql")
                .unwrap_or_default();
            (name, normalize_ddl(&sql))
        })
        .collect()
    }

    /// Per-table full DDL: name → normalized `CREATE TABLE` SQL (comments and
    /// whitespace stripped, `--` inside string literals preserved). This
    /// captures the complete table shape — columns, ordering, default/not-null,
    /// PRIMARY KEY, inline UNIQUE, CHECK, and FK constraints.
    async fn table_defs(conn: &Connection) -> std::collections::BTreeMap<String, String> {
        conn.query(
            "SELECT name, sql FROM sqlite_master \
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%' \
             AND name NOT LIKE '__turso_internal_%' AND name NOT LIKE 'turso_cdc%'",
            (),
        )
        .await
        .expect("read table definitions")
        .into_iter()
        .map(|row| {
            let name = row.get::<String>(0).expect("table name");
            let sql = row
                .get::<Option<String>>(1)
                .expect("table sql")
                .unwrap_or_default();
            (name, normalize_ddl(&sql))
        })
        .collect()
    }

    /// User table names (excludes `sqlite_*` internal tables and Turso runtime
    /// artifacts such as CDC/internal tables).
    async fn table_names(conn: &Connection) -> Vec<String> {
        conn.query(
            "SELECT name FROM sqlite_master \
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%' \
             AND name NOT LIKE '__turso_internal_%' AND name NOT LIKE 'turso_cdc%'",
            (),
        )
        .await
        .expect("read tables")
        .into_iter()
        .map(|row| row.get::<String>(0).expect("table name"))
        .collect()
    }

    // ── Expected-shape fixtures (hand-maintained, independent of the baseline)
    //
    // These are deliberately typed out literally, NOT derived from the baseline
    // DDL constants above, so a wrong baseline cannot validate itself.

    /// Hand-maintained expected current core shape: table → its full column
    /// set. Column sets are compared order-insensitively because ALTER-produced
    /// shapes (the retired chain) must compare equal to CREATE-produced ones
    /// (the baseline).
    const EXPECTED_CORE_TABLE_COLUMNS: &[(&str, &[&str])] = &[
        (
            "agents",
            &[
                "job_id", "agent_id", "kind", "idx", "status", "outcome", "task",
            ],
        ),
        (
            "alarms",
            &[
                "id",
                "session_id",
                "user_name",
                "kind",
                "text",
                "fire_at",
                "interval_seconds",
                "next_fire_at",
                "status",
                "created_at",
                "trigger",
                "role",
                "identifier",
                "workspace",
            ],
        ),
        (
            "chat_history",
            &[
                "id",
                "message_id",
                "user_name",
                "direction",
                "content",
                "agent_role",
                "broadcast_id",
                "workspace",
                "timestamp",
                "reply_author",
                "reply_snippet",
            ],
        ),
        ("config_kv", &["key", "value"]),
        (
            "config_model_routing",
            &["model", "provider_order", "allow_fallbacks"],
        ),
        (
            "custom_roles",
            &["name", "prompt", "tools", "behavior", "created_at"],
        ),
        (
            "editor_tabs",
            &[
                "workspace_name",
                "file_path",
                "tab_order",
                "is_active",
                "is_dirty",
                "dirty_content",
            ],
        ),
        (
            "jobs",
            &[
                "id",
                "kind",
                "status",
                "task",
                "workspace_name",
                "user_name",
                "channel",
                "role",
                "retry_count",
                "created_at",
                "updated_at",
                "ticket_id",
                "caller_agent_id",
                "mode",
                "exec_dir",
                "custom_launch",
            ],
        ),
        (
            "pending_jobs",
            &["id", "target_agent_id", "envelope", "created_at"],
        ),
        ("research_jobs", &["id", "state"]),
        ("schema_migrations", &["id", "applied_at"]),
        (
            "session_metadata",
            &[
                "agent_id",
                "last_activity",
                "channel",
                "user_name",
                "workspace_name",
                "role",
                "active_models",
                "token_length",
                "message_count",
                "created_at",
                "sleep_ended",
                "custom_role",
                "custom_identifier",
                "launch_stopped",
                "reply_to",
                "context_block_len",
            ],
        ),
        ("session_personal_changes", &["agent_id", "baseline"]),
        (
            "sessions",
            &["id", "agent_id", "role", "content", "created_at"],
        ),
        (
            "telegram_bursts",
            &[
                "id",
                "account_id",
                "user_name",
                "reply_target",
                "parts",
                "released_by",
                "released_at",
                "part_times",
            ],
        ),
        (
            "telegram_intake",
            &[
                "account_id",
                "update_id",
                "state",
                "payload",
                "created_at",
                "decided_at",
                "claimed_by",
            ],
        ),
        (
            "ticket_chronicle",
            &[
                "id",
                "ticket_id",
                "workspace_name",
                "source_phase",
                "target_phase",
                "at",
                "actor",
            ],
        ),
        (
            "ticket_comments",
            &["id", "ticket_id", "role", "content", "created_at"],
        ),
        ("ticket_counters", &["workspace_name", "next_id"]),
        (
            "tickets",
            &[
                "id",
                "title",
                "description",
                "phase",
                "workspace_name",
                "created_at",
                "updated_at",
                "prerequisites",
                "supersedes",
                "superseded_by",
                "commit_hash",
                "lines_added",
                "lines_removed",
                "reporter",
                "is_archived",
                "embedding",
                "priority",
                "reviewed_head",
                "reviewed_tree",
                "done_at",
                "bounce_count",
                "last_transition_actor",
            ],
        ),
        (
            "user_channels",
            &["user_name", "channel", "identifier", "reply_target"],
        ),
        (
            "users",
            &[
                "name",
                "selected_workspace",
                "image_gen_model",
                "video_model",
                "granted_tools",
            ],
        ),
        (
            "workspace_contexts",
            &["workspace_name", "role", "content", "created_at"],
        ),
        (
            "workspaces",
            &[
                "name",
                "path",
                "status",
                "created_at",
                "updated_at",
                "maintenance",
                "paused",
                "maintainer_debounce_mins",
                "maintainer_last_run_at",
                "maintainer_recommendations",
                "diagnostics",
                "diagnostics_generation",
                "notes",
                "last_analyzed_commit",
                "discovery_generation",
            ],
        ),
    ];

    /// The indexes a fresh core store carries: only the ones a product query's
    /// plan reaches, plus the uniqueness ones (see
    /// [`DROP_UNUSED_CORE_INDEXES`]).
    const EXPECTED_CORE_INDEXES: &[&str] = &[
        "idx_ticket_comments_ticket_id",
        "idx_sessions_agent_id",
        "idx_jobs_kind_status",
        "idx_agents_anchor",
        "idx_pending_jobs_agent_created",
        "workspace_contexts_null_role",
        "idx_chat_history_user_ws_id",
        "idx_tickets_title_fts",
        "idx_tickets_board_active",
        "idx_jobs_phase_ticket",
        "idx_ticket_chronicle_ws_id",
        "idx_ticket_chronicle_dedup",
        "idx_alarms_due",
        "idx_tickets_workspace_phase",
    ];

    /// The indexes delta [`DROP_UNUSED_CORE_INDEXES`] drops — the audit's
    /// outcome. With [`EXPECTED_CORE_INDEXES`] they are the pre-delta-`65` set a
    /// store upgrading across this release carries.
    const REMOVED_CORE_INDEXES: &[&str] = &[
        "idx_jobs_caller_agent",
        "idx_jobs_updated_at",
        "idx_chat_history_user",
        "idx_chat_history_workspace",
    ];

    const EXPECTED_LOGS_TABLE_COLUMNS: &[(&str, &[&str])] = &[
        (
            "grep_telemetry",
            &[
                "id",
                "recorded_at",
                "command",
                "served",
                "reason",
                "recursive",
                "piped",
                "operand_count",
                "flags",
                "mode",
                "workspace",
                "grep_count",
                "served_count",
                "skipped_count",
                "duration_ms",
                "exit_code",
            ],
        ),
        (
            "llm_failures",
            &[
                "id",
                "operation_id",
                "recorded_at",
                "purpose",
                "agent_id",
                "role",
                "workspace",
                "ticket_id",
                "model",
                "routing",
                "attempt",
                "attempt_duration_ms",
                "failure_class",
                "finish_reason",
                "retry_after_ms",
                "error_chain",
            ],
        ),
        (
            "llm_requests",
            &[
                "id",
                "recorded_at",
                "purpose",
                "agent_id",
                "role",
                "workspace",
                "ticket_id",
                "model",
                "routing",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "cache_miss_tokens",
                "duration_ms",
                "retry_attempts",
                "finish_reason",
                "failure_class",
                "success",
                "cost",
                "cost_details",
                "upstream_provider",
                "system_fingerprint",
            ],
        ),
        (
            "logs",
            &[
                "id",
                "timestamp",
                "level",
                "target",
                "message",
                "fields",
                "agent_id",
                "agent_role",
                "workspace",
            ],
        ),
        ("schema_migrations", &["id", "applied_at"]),
        (
            "tool_calls",
            &[
                "id",
                "agent_id",
                "role",
                "source",
                "tool_name",
                "arguments",
                "duration_ms",
                "success",
                "error_message",
                "workspace",
                "recorded_at",
            ],
        ),
    ];

    /// The indexes a fresh logs store carries: only the ones a product query's
    /// plan reaches (see [`DROP_UNUSED_LOGS_INDEXES`]).
    const EXPECTED_LOGS_INDEXES: &[&str] = &[
        "idx_logs_level",
        "idx_tool_calls_recorded_at",
        "idx_tool_calls_error_message",
        "idx_llm_requests_recorded_at",
        "idx_grep_telemetry_recorded_at",
    ];

    /// All expected core table names.
    fn expected_core_table_names() -> Vec<&'static str> {
        EXPECTED_CORE_TABLE_COLUMNS
            .iter()
            .map(|(n, _)| *n)
            .collect()
    }

    /// Expected core table names, excluding `schema_migrations` (whose row
    /// count legitimately grows when a new catalog entry is applied).
    fn expected_core_domain_tables() -> Vec<&'static str> {
        expected_core_table_names()
            .into_iter()
            .filter(|n| *n != "schema_migrations")
            .collect()
    }

    /// Tables a reopen introduces which the pre-reopen fixture cannot carry
    /// (delta `51`'s `custom_roles`, delta `59`'s `telegram_intake`, delta
    /// `61`'s `telegram_bursts` and delta `62`'s `session_personal_changes`).
    /// They have no "before" row count to compare, so
    /// the reopen snapshot skips them; their arrival is asserted by the schema
    /// comparison instead.
    const REOPEN_NEW_CORE_TABLES: &[&str] = &[
        "custom_roles",
        "telegram_intake",
        "telegram_bursts",
        "session_personal_changes",
    ];

    /// Domain tables whose row counts a reopen snapshot compares: every
    /// expected domain table except the ones the reopen introduces.
    fn counted_core_domain_tables() -> Vec<&'static str> {
        expected_core_domain_tables()
            .into_iter()
            .filter(|n| !REOPEN_NEW_CORE_TABLES.contains(n))
            .collect()
    }

    fn expected_logs_domain_tables() -> Vec<&'static str> {
        EXPECTED_LOGS_TABLE_COLUMNS
            .iter()
            .map(|(n, _)| *n)
            .filter(|n| *n != "schema_migrations")
            .collect()
    }

    // ── Schema assertion helper ────────────────────────────────────────

    /// Assert that `conn`'s schema matches `expected_tables` (table name-set
    /// equality + per-table column-name SET equality, order-insensitive) and
    /// `expected_indexes` (explicit index name-set equality, see
    /// [`assert_index_set`]). Errors list the missing/extra items with
    /// `context` for a helpful message.
    async fn assert_schema_matches(
        conn: &Connection,
        expected_tables: &[(&str, &[&str])],
        expected_indexes: &[&str],
        context: &str,
    ) {
        let actual_names: Vec<String> = table_names(conn).await;
        let actual_set: std::collections::HashSet<&str> =
            actual_names.iter().map(String::as_str).collect();
        let expected_set: std::collections::HashSet<&str> =
            expected_tables.iter().map(|(n, _)| *n).collect();
        let missing: Vec<&str> = expected_set.difference(&actual_set).copied().collect();
        let extra: Vec<&str> = actual_set.difference(&expected_set).copied().collect();
        assert!(missing.is_empty(), "{context}: missing tables {missing:?}");
        assert!(extra.is_empty(), "{context}: unexpected tables {extra:?}");

        for (table, expected_cols) in expected_tables {
            let actual = column_names(conn, table).await;
            let actual_set: std::collections::HashSet<&str> =
                actual.iter().map(String::as_str).collect();
            let expected_set: std::collections::HashSet<&str> =
                expected_cols.iter().copied().collect();
            let miss: Vec<&str> = expected_set.difference(&actual_set).copied().collect();
            let ex: Vec<&str> = actual_set.difference(&expected_set).copied().collect();
            assert!(
                miss.is_empty(),
                "{context}: table '{table}' missing columns {miss:?}"
            );
            assert!(
                ex.is_empty(),
                "{context}: table '{table}' has extra columns {ex:?}"
            );
        }

        assert_index_set(conn, expected_indexes, context).await;
    }

    /// Assert that `conn`'s explicit index name set is exactly `expected`.
    /// Errors list the missing/extra names with `context`.
    async fn assert_index_set(conn: &Connection, expected: &[&str], context: &str) {
        let names: Vec<String> = index_defs(conn).await.into_keys().collect();
        let actual: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();
        let expected: std::collections::HashSet<&str> = expected.iter().copied().collect();
        let missing: Vec<&str> = expected.difference(&actual).copied().collect();
        let extra: Vec<&str> = actual.difference(&expected).copied().collect();
        assert!(missing.is_empty(), "{context}: missing indexes {missing:?}");
        assert!(extra.is_empty(), "{context}: unexpected indexes {extra:?}");
    }

    /// Table → ordered column names, for before/after snapshots.
    async fn column_sets(
        conn: &Connection,
        tables: &[&str],
    ) -> std::collections::BTreeMap<String, Vec<String>> {
        let mut m = std::collections::BTreeMap::new();
        for t in tables {
            m.insert(t.to_string(), column_names(conn, t).await);
        }
        m
    }

    /// Table → row count, for before/after data snapshots.
    async fn table_row_counts(
        conn: &Connection,
        tables: &[&str],
    ) -> std::collections::BTreeMap<String, i64> {
        let mut m = std::collections::BTreeMap::new();
        for t in tables {
            let count: i64 = conn
                .query(&format!("SELECT COUNT(*) FROM {t}"), ())
                .await
                .expect("count rows")[0]
                .get::<i64>(0)
                .expect("count");
            m.insert(t.to_string(), count);
        }
        m
    }

    // ── Old-catalog fixture (simulated pre-consolidation databases) ─────
    //
    // The production catalog was rewritten to the current-shape baseline
    // (ids `24`–`26`), so the retired `1`–`23` chain can no longer be observed
    // by running `run_migrations`. These fixtures replay that chain (verbatim
    // old DDL + old Rust bodies) so the baseline can be proven to be a strict
    // no-op against a database built by the REAL pre-consolidation DDL text,
    // not a hand-shaped imitation. They cover the three supported states:
    // a 0.5.0 DB (one delta behind), a current-main DB (all columns), and a
    // fresh install.

    const OLD_BASELINE_BOARD_TABLES: &str = "\
CREATE TABLE IF NOT EXISTS tickets (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    phase          TEXT NOT NULL DEFAULT 'backlog',
    assigned_to     TEXT,
    workspace_name  TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    prerequisites   TEXT NOT NULL DEFAULT '[]',
    supersedes      TEXT,
    superseded_by   TEXT,
    commit_hash     TEXT,
    lines_added     INTEGER,
    lines_removed   INTEGER,
    reporter        TEXT NOT NULL DEFAULT '',
    is_archived     INTEGER NOT NULL DEFAULT 0,
    embedding       BLOB,
    pipeline_reservation INTEGER NOT NULL DEFAULT 0,
    priority        INTEGER NOT NULL DEFAULT 1,
    reviewed_head   TEXT,
    reviewed_tree   TEXT,
    done_at         TEXT,
    bounce_count    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ticket_comments (
    id          TEXT PRIMARY KEY,
    ticket_id   TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id)
);
CREATE TABLE IF NOT EXISTS ticket_counters (
    workspace_name TEXT PRIMARY KEY,
    next_id        INTEGER NOT NULL DEFAULT 1
);";

    const OLD_BASELINE_SESSION_TABLES: &str = "\
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_metadata (
    agent_id      TEXT PRIMARY KEY,
    last_activity TEXT NOT NULL,
    channel       TEXT,
    user_name     TEXT,
    workspace_name TEXT,
    role          TEXT,
    active_models TEXT,
    token_length  INTEGER,
    message_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'launched',
    task           TEXT NOT NULL DEFAULT '',
    workspace_name TEXT NOT NULL,
    user_name      TEXT NOT NULL DEFAULT '',
    channel        TEXT NOT NULL DEFAULT '',
    role           TEXT NOT NULL,
    retry_count    INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    job_id     TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    agent_id   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    idx        INTEGER,
    status     TEXT NOT NULL DEFAULT 'launched',
    outcome    TEXT,
    task       TEXT NOT NULL,
    PRIMARY KEY (job_id, agent_id)
);
CREATE TABLE IF NOT EXISTS pending_jobs (
    id              TEXT PRIMARY KEY,
    target_agent_id TEXT NOT NULL,
    envelope        TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ticket_stage_jobs (
    id          TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    ticket_id   TEXT NOT NULL,
    stage       TEXT NOT NULL,
    phase       TEXT NOT NULL,
    round       INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS research_jobs (
    id    TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    state TEXT NOT NULL
);";

    const OLD_BASELINE_WORKSPACE_TABLES: &str = "\
CREATE TABLE IF NOT EXISTS workspaces (
    name       TEXT PRIMARY KEY,
    path       TEXT NOT NULL UNIQUE,
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    maintenance INTEGER NOT NULL DEFAULT 0,
    paused      INTEGER NOT NULL DEFAULT 1,
    maintainer_debounce_mins INTEGER NOT NULL DEFAULT 5,
    maintainer_last_run_at TEXT,
    diagnostics TEXT,
    diagnostics_generation INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    last_analyzed_commit TEXT,
    discovery_generation INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS workspace_contexts (
    workspace_name TEXT NOT NULL REFERENCES workspaces(name) ON DELETE CASCADE,
    role           TEXT,
    content        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    UNIQUE(workspace_name, role)
);
CREATE TABLE IF NOT EXISTS editor_tabs (
    workspace_name TEXT NOT NULL REFERENCES workspaces(name) ON DELETE CASCADE,
    file_path      TEXT NOT NULL,
    tab_order      INTEGER NOT NULL DEFAULT 0,
    is_active      INTEGER NOT NULL DEFAULT 0,
    is_dirty       INTEGER NOT NULL DEFAULT 0,
    dirty_content  TEXT,
    PRIMARY KEY (workspace_name, file_path)
);";

    const OLD_BASELINE_USERS_TABLES: &str = "\
CREATE TABLE IF NOT EXISTS users (
    name                TEXT PRIMARY KEY,
    permissions         TEXT,
    selected_workspace  TEXT,
    selected_role       TEXT
);
CREATE TABLE IF NOT EXISTS user_channels (
    user_name   TEXT NOT NULL REFERENCES users(name),
    channel     TEXT NOT NULL,
    identifier  TEXT NOT NULL,
    reply_target TEXT,
    UNIQUE(channel, identifier)
);
CREATE TABLE IF NOT EXISTS user_roles (
    user_name   TEXT NOT NULL REFERENCES users(name),
    role        TEXT NOT NULL,
    PRIMARY KEY (user_name, role)
);";

    const OLD_BASELINE_CONFIG_TABLES: &str = "\
CREATE TABLE IF NOT EXISTS config_kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS config_role (
    role             TEXT PRIMARY KEY,
    model            TEXT,
    reasoning_effort TEXT
);
CREATE TABLE IF NOT EXISTS config_model_routing (
    model              TEXT PRIMARY KEY,
    provider_order     TEXT,
    allow_fallbacks    INTEGER
);";

    const OLD_BASELINE_CHAT_HISTORY_TABLES: &str = "\
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    user_name TEXT NOT NULL,
    direction TEXT NOT NULL,
    content TEXT NOT NULL,
    agent_role TEXT,
    workspace TEXT NOT NULL
);";

    const OLD_BASELINE_CORE_INDEXES: &str = "\
CREATE INDEX IF NOT EXISTS idx_ticket_comments_ticket_id ON ticket_comments(ticket_id);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id, id);
CREATE INDEX IF NOT EXISTS idx_jobs_kind_status ON jobs(kind, status);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agents_anchor ON agents(agent_id) WHERE job_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_pending_jobs_agent_created ON pending_jobs(target_agent_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS workspace_contexts_null_role ON workspace_contexts(workspace_name) WHERE role IS NULL;
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_name);
CREATE INDEX IF NOT EXISTS idx_chat_history_workspace ON chat_history(workspace);
CREATE INDEX IF NOT EXISTS idx_chat_history_user_ws_id ON chat_history(user_name, workspace, id);";

    const OLD_BASELINE_LOGS_TABLES: &str = "\
CREATE TABLE IF NOT EXISTS logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    level       TEXT NOT NULL,
    target      TEXT NOT NULL,
    message     TEXT NOT NULL,
    fields      TEXT NOT NULL DEFAULT '{}',
    agent_id    TEXT NOT NULL DEFAULT '',
    agent_role  TEXT NOT NULL DEFAULT '',
    workspace   TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id       TEXT NOT NULL,
    role           TEXT NOT NULL,
    tool_name      TEXT NOT NULL,
    arguments      TEXT NOT NULL DEFAULT '{}',
    duration_ms    INTEGER NOT NULL DEFAULT 0,
    success        INTEGER NOT NULL DEFAULT 1,
    error_message  TEXT,
    workspace      TEXT NOT NULL DEFAULT '',
    recorded_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_requests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at         TEXT NOT NULL,
    purpose             TEXT NOT NULL,
    agent_id            TEXT NOT NULL DEFAULT '',
    role                TEXT NOT NULL DEFAULT '',
    workspace           TEXT NOT NULL DEFAULT '',
    ticket_id           TEXT,
    model               TEXT NOT NULL,
    routing             TEXT NOT NULL DEFAULT '',
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cached_input_tokens INTEGER,
    cache_miss_tokens   INTEGER,
    duration_ms         INTEGER NOT NULL,
    retry_attempts      INTEGER NOT NULL,
    finish_reason       TEXT,
    failure_class       TEXT,
    success             INTEGER NOT NULL DEFAULT 1,
    cost                REAL,
    cost_details        TEXT,
    upstream_provider   TEXT,
    system_fingerprint  TEXT
);";

    const OLD_BASELINE_LOGS_INDEXES: &str = "\
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
CREATE INDEX IF NOT EXISTS idx_logs_target ON logs(target);
CREATE INDEX IF NOT EXISTS idx_logs_agent_role ON logs(agent_role);
CREATE INDEX IF NOT EXISTS idx_logs_agent_id ON logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_logs_workspace ON logs(workspace);
CREATE INDEX IF NOT EXISTS idx_tool_calls_agent_id ON tool_calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_role ON tool_calls(role);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_recorded_at ON tool_calls(recorded_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_workspace ON tool_calls(workspace);
CREATE INDEX IF NOT EXISTS idx_tool_calls_error_message ON tool_calls(error_message);
CREATE INDEX IF NOT EXISTS idx_llm_requests_recorded_at ON llm_requests(recorded_at);
CREATE INDEX IF NOT EXISTS idx_llm_requests_agent_id ON llm_requests(agent_id);
CREATE INDEX IF NOT EXISTS idx_llm_requests_model ON llm_requests(model);
CREATE INDEX IF NOT EXISTS idx_llm_requests_purpose ON llm_requests(purpose);";

    const OLD_DELTA_DROP_CONFIG_ROLE: &str = "DROP TABLE IF EXISTS config_role;";

    const OLD_DELTA_FTS_INDEX: &str = "CREATE INDEX IF NOT EXISTS idx_tickets_title_fts ON tickets \
USING fts (title) WITH (tokenizer = 'ngram');";

    const OLD_DELTA_BOARD_ACTIVE_INDEX: &str = "CREATE INDEX IF NOT EXISTS idx_tickets_board_active ON tickets \
(is_archived, priority ASC, created_at DESC);";

    const OLD_DELTA_GREP_TELEMETRY: &str = "\
CREATE TABLE IF NOT EXISTS grep_telemetry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at   TEXT NOT NULL,
    command       TEXT NOT NULL,
    served        INTEGER NOT NULL DEFAULT 0,
    reason        TEXT NOT NULL DEFAULT '',
    recursive     INTEGER NOT NULL DEFAULT 0,
    piped         INTEGER NOT NULL DEFAULT 0,
    operand_count INTEGER NOT NULL DEFAULT 0,
    flags         TEXT NOT NULL DEFAULT '',
    mode          TEXT NOT NULL DEFAULT '',
    workspace     TEXT NOT NULL DEFAULT '',
    grep_count    INTEGER NOT NULL DEFAULT 0,
    served_count  INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER,
    exit_code     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_grep_telemetry_recorded_at ON grep_telemetry(recorded_at);
CREATE INDEX IF NOT EXISTS idx_grep_telemetry_served ON grep_telemetry(served);
CREATE INDEX IF NOT EXISTS idx_grep_telemetry_reason ON grep_telemetry(reason);
CREATE INDEX IF NOT EXISTS idx_grep_telemetry_command ON grep_telemetry(command);";

    const OLD_DELTA_ALARMS: &str = "\
CREATE TABLE IF NOT EXISTS alarms (
    id               TEXT PRIMARY KEY,
    session_id       TEXT NOT NULL,
    user_name        TEXT NOT NULL,
    kind             TEXT NOT NULL,
    text             TEXT NOT NULL,
    fire_at          TEXT,
    interval_seconds INTEGER,
    next_fire_at     TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'active',
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alarms_due ON alarms(status, next_fire_at);";

    const OLD_DELTA_CHAT_HISTORY_TIMESTAMP: &str =
        "ALTER TABLE chat_history ADD COLUMN timestamp TEXT;";

    const OLD_DELTA_RENAME_READY_FOR_DEVELOPMENT_TO_QUEUED: &str = "\
UPDATE tickets SET phase = 'queued' WHERE phase = 'ready_for_development';\
UPDATE ticket_chronicle SET source_phase = 'queued' WHERE source_phase = 'ready_for_development';\
UPDATE ticket_chronicle SET target_phase = 'queued' WHERE target_phase = 'ready_for_development';\
UPDATE jobs SET kind = 'queued' WHERE kind = 'ready_for_development';";

    const OLD_DELTA_TICKETS_WORKSPACE_PHASE_INDEX: &str = "CREATE INDEX IF NOT EXISTS idx_tickets_workspace_phase \
ON tickets (workspace_name, phase, is_archived, priority ASC, created_at DESC);";

    // The one-time consolidation import is retired and not preserved in tests:
    // its schema effects (dropping user_roles/config_role/ticket_jobs/
    // ticket_stage_jobs) are fully duplicated by the later chain entries
    // (19, 10, consolidate_005, 15), and its data effects are simulated by
    // direct seeding. The no-op yields the identical final shape.
    fn noop_import(_conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn cleanup_legacy_ticket_jobs(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
        Box::pin(run_import_cleanup(conn))
    }

    /// The post-import cleanup: discard pre-rework ticket job rows (FK ON, so
    /// the cascade fires) and report — never fail on — any orphan rows the
    /// bulk load may have left behind.
    async fn run_import_cleanup(conn: &Connection) -> anyhow::Result<()> {
        conn.execute(
            "DELETE FROM jobs WHERE kind IN \
                 ('ticket_stage', 'ticket_analysis', 'ticket_implementation')",
            (),
        )
        .await
        .context("Failed to discard legacy ticket jobs after consolidation import")?;

        let violations = conn
            .query("PRAGMA foreign_key_check", ())
            .await
            .context("Failed to run PRAGMA foreign_key_check after consolidation import")?;
        if !violations.is_empty() {
            tracing::warn!(
                count = violations.len(),
                "consolidation import found orphan rows violating the new FKs; \
                 preserving them as soft references (future writes still enforce the FK)",
            );
        }
        Ok(())
    }

    fn drop_jobs_paused_frozen(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
        Box::pin(run_drop_jobs_paused_frozen(conn))
    }

    /// Idempotently drop `jobs.paused_frozen` — the finalized add/drop
    /// round-trip cleanup of the retired chain. Turso has no
    /// `DROP COLUMN IF EXISTS`, so the existence probe is [`column_exists`].
    async fn run_drop_jobs_paused_frozen(conn: &Connection) -> anyhow::Result<()> {
        if column_exists(conn, "jobs", "paused_frozen").await? {
            conn.execute("ALTER TABLE jobs DROP COLUMN paused_frozen", ())
                .await
                .context("Failed to drop jobs.paused_frozen")?;
        }
        Ok(())
    }

    fn drop_user_roles_and_seed_onboarding(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
        Box::pin(run_drop_user_roles_and_seed_onboarding(conn))
    }

    /// Idempotently drop `user_roles` and seed the onboarding state for
    /// existing (non-fresh) installs. Fresh-vs-existing discriminator: the
    /// `users` table is EMPTY on a fresh install (the admin is seeded by
    /// `ensure_admin_user` as a POST-OPEN hook, AFTER migrations) and
    /// NON-EMPTY on an existing install.
    async fn run_drop_user_roles_and_seed_onboarding(conn: &Connection) -> anyhow::Result<()> {
        let user_count: i64 = conn
            .query("SELECT COUNT(*) FROM users", ())
            .await
            .context("Failed to probe users count")?
            .into_iter()
            .next()
            .and_then(|row| row.get::<i64>(0).ok())
            .unwrap_or(0);

        if user_count > 0 {
            conn.execute(
                "INSERT OR REPLACE INTO config_kv (key, value) VALUES (?1, ?2)",
                params![
                    crate::config::CONFIG_KEY_ONBOARDING_STATE,
                    crate::config::OnboardingState::Finished.as_str(),
                ],
            )
            .await
            .context("Failed to set onboarding_state=finished")?;

            let rows = conn
                .query("SELECT name, permissions, selected_role FROM users", ())
                .await?;
            for row in rows {
                let name: String = row.get(0)?;
                let permissions: Option<String> = row.get(1)?;
                let selected_role: Option<String> = row.get(2)?;
                let sel = selected_role.unwrap_or_default();
                let needs_default = sel.is_empty() || {
                    // 'artist' stays accepted here: this migration body already
                    // ran on older stores, and re-running it on a mid-chain
                    // upgrade must not clobber legacy 'artist' rows. New writes
                    // can never produce 'artist'.
                    let in_pool = if permissions.as_deref() == Some("full") {
                        matches!(sel.as_str(), "support" | "assistant" | "manager" | "artist")
                    } else {
                        matches!(sel.as_str(), "assistant" | "artist")
                    };
                    !in_pool
                };
                if needs_default {
                    conn.execute(
                        "UPDATE users SET selected_role = 'assistant' WHERE name = ?1",
                        params![name],
                    )
                    .await
                    .context("Failed to normalize selected_role")?;
                }
            }
        }

        conn.execute("DROP TABLE IF EXISTS user_roles", ())
            .await
            .context("Failed to drop user_roles")?;
        Ok(())
    }

    fn rewrite_analysis_verdicts(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
        Box::pin(run_rewrite_analysis_verdicts(conn))
    }

    /// One-time data migration: rewrite legacy score-based analysis verdict
    /// rows to the score-less graded shape. Idempotent and re-runnable until
    /// applied.
    async fn run_rewrite_analysis_verdicts(conn: &Connection) -> anyhow::Result<()> {
        let rows = conn
            .query(
                "SELECT job_id, agent_id, outcome FROM agents \
                 WHERE kind = 'analyst' AND outcome LIKE '{\"verdict\":%'",
                (),
            )
            .await
            .context("Failed to read analysis verdict rows for migration")?;
        for row in rows {
            let job_id: String = row.get(0)?;
            let agent_id: String = row.get(1)?;
            let outcome: String = row.get(2)?;
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&outcome) else {
                continue;
            };
            let Some(verdict) = value.get("verdict") else {
                continue;
            };
            let Some(score) = verdict.get("score").and_then(serde_json::Value::as_u64) else {
                continue;
            };
            let Some(issues) = verdict.get("issues").and_then(serde_json::Value::as_array) else {
                continue;
            };
            let grade = if score < 7 { "blocker" } else { "minor" };
            let graded: Vec<serde_json::Value> = issues
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|text| serde_json::json!({ "text": text, "grade": grade }))
                .collect();
            let rewritten = serde_json::json!({ "verdict": { "issues": graded } }).to_string();
            conn.execute(
                "UPDATE agents SET outcome = ?1 WHERE job_id = ?2 AND agent_id = ?3",
                params![rewritten, job_id, agent_id],
            )
            .await
            .context("Failed to rewrite analysis verdict row")?;
        }
        Ok(())
    }

    /// The ENTIRE retired `1`–`23` catalog in original order, except entry
    /// `consolidate_001_import_domain_stores` (see [`noop_import`]). The
    /// relative order is preserved exactly — the add-before-drop round-trips
    /// (`007`/`consolidate_006`, the `ticket_jobs` rename/drop) depend on it.
    const OLD_CATALOG: &[Migration] = &[
        Migration {
            id: "1",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_BASELINE_BOARD_TABLES),
        },
        Migration {
            id: "2",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_BASELINE_SESSION_TABLES),
        },
        Migration {
            id: "3",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_BASELINE_WORKSPACE_TABLES),
        },
        Migration {
            id: "4",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_BASELINE_USERS_TABLES),
        },
        Migration {
            id: "5",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_BASELINE_CONFIG_TABLES),
        },
        Migration {
            id: "6",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_BASELINE_CHAT_HISTORY_TABLES),
        },
        Migration {
            id: "7",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_BASELINE_CORE_INDEXES),
        },
        Migration {
            id: "001_drop_ticket_pipeline_reservation",
            target: TargetDb::Core,
            body: MigrationBody::Sql("ALTER TABLE tickets DROP COLUMN pipeline_reservation;"),
        },
        Migration {
            id: "002_drop_ticket_assigned_to",
            target: TargetDb::Core,
            body: MigrationBody::Sql("ALTER TABLE tickets DROP COLUMN assigned_to;"),
        },
        Migration {
            id: "003_drop_ticket_stage_jobs_round",
            target: TargetDb::Core,
            body: MigrationBody::Sql("ALTER TABLE ticket_stage_jobs DROP COLUMN round;"),
        },
        Migration {
            id: "005_drop_ticket_stage_jobs_phase",
            target: TargetDb::Core,
            body: MigrationBody::Sql("ALTER TABLE ticket_stage_jobs DROP COLUMN phase;"),
        },
        Migration {
            id: "006_rename_ticket_stage_jobs_to_ticket_jobs",
            target: TargetDb::Core,
            body: MigrationBody::Sql(
                "DROP TABLE IF EXISTS ticket_jobs; ALTER TABLE ticket_stage_jobs RENAME TO ticket_jobs;",
            ),
        },
        Migration {
            id: "consolidate_002_drop_ticket_jobs_stage",
            target: TargetDb::Core,
            body: MigrationBody::Sql("ALTER TABLE ticket_jobs DROP COLUMN stage;"),
        },
        Migration {
            id: "007_jobs_paused_frozen",
            target: TargetDb::Core,
            body: MigrationBody::Sql("ALTER TABLE jobs ADD COLUMN paused_frozen INTEGER;"),
        },
        Migration {
            id: "consolidate_003_jobs_ticket_id",
            target: TargetDb::Core,
            body: MigrationBody::Sql(
                "ALTER TABLE jobs ADD COLUMN ticket_id TEXT REFERENCES tickets(id);",
            ),
        },
        Migration {
            id: "consolidate_005_drop_ticket_jobs",
            target: TargetDb::Core,
            body: MigrationBody::Sql("DROP TABLE IF EXISTS ticket_jobs;"),
        },
        Migration {
            id: "consolidate_006_drop_jobs_paused_frozen",
            target: TargetDb::Core,
            body: MigrationBody::Sql("ALTER TABLE jobs DROP COLUMN paused_frozen;"),
        },
        Migration {
            id: "consolidate_007_jobs_phase_ticket_index",
            target: TargetDb::Core,
            body: MigrationBody::Sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_phase_ticket ON jobs(kind, ticket_id) WHERE ticket_id IS NOT NULL;",
            ),
        },
        Migration {
            id: "consolidate_008_create_ticket_chronicle",
            target: TargetDb::Core,
            body: MigrationBody::Sql(
                "CREATE TABLE IF NOT EXISTS ticket_chronicle (\
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,\
                    ticket_id      TEXT NOT NULL,\
                    workspace_name TEXT NOT NULL,\
                    source_phase   TEXT NOT NULL,\
                    target_phase   TEXT NOT NULL,\
                    at             TEXT NOT NULL\
                 );\
                 CREATE INDEX IF NOT EXISTS idx_ticket_chronicle_ws_id \
                    ON ticket_chronicle(workspace_name, id);\
                 CREATE UNIQUE INDEX IF NOT EXISTS idx_ticket_chronicle_dedup \
                    ON ticket_chronicle(ticket_id, workspace_name, source_phase, target_phase, at);",
            ),
        },
        Migration {
            id: "10",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_DELTA_DROP_CONFIG_ROLE),
        },
        Migration {
            id: "11",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_DELTA_FTS_INDEX),
        },
        Migration {
            id: "12",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_DELTA_BOARD_ACTIVE_INDEX),
        },
        Migration {
            id: "consolidate_001_import_domain_stores",
            target: TargetDb::Core,
            body: MigrationBody::Rust(noop_import),
        },
        Migration {
            id: "003_reset_nonterminal_tickets",
            target: TargetDb::Core,
            body: MigrationBody::Sql(
                "UPDATE tickets SET phase = 'backlog' WHERE phase NOT IN ('done','cancelled','failed') AND is_archived = 0;",
            ),
        },
        Migration {
            id: "13",
            target: TargetDb::Core,
            body: MigrationBody::Rust(cleanup_legacy_ticket_jobs),
        },
        Migration {
            id: "15",
            target: TargetDb::Core,
            body: MigrationBody::Sql("DROP TABLE IF EXISTS ticket_stage_jobs;"),
        },
        Migration {
            id: "16",
            target: TargetDb::Core,
            body: MigrationBody::Rust(drop_jobs_paused_frozen),
        },
        Migration {
            id: "8",
            target: TargetDb::Logs,
            body: MigrationBody::Sql(OLD_BASELINE_LOGS_TABLES),
        },
        Migration {
            id: "9",
            target: TargetDb::Logs,
            body: MigrationBody::Sql(OLD_BASELINE_LOGS_INDEXES),
        },
        Migration {
            id: "14",
            target: TargetDb::Logs,
            body: MigrationBody::Sql(OLD_DELTA_GREP_TELEMETRY),
        },
        Migration {
            id: "17",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_DELTA_ALARMS),
        },
        Migration {
            id: "18",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_DELTA_CHAT_HISTORY_TIMESTAMP),
        },
        Migration {
            id: "19",
            target: TargetDb::Core,
            body: MigrationBody::Rust(drop_user_roles_and_seed_onboarding),
        },
        Migration {
            id: "20",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_DELTA_RENAME_READY_FOR_DEVELOPMENT_TO_QUEUED),
        },
        Migration {
            id: "21",
            target: TargetDb::Core,
            body: MigrationBody::Rust(rewrite_analysis_verdicts),
        },
        Migration {
            id: "22",
            target: TargetDb::Core,
            body: MigrationBody::Sql(OLD_DELTA_TICKETS_WORKSPACE_PHASE_INDEX),
        },
        Migration {
            id: "23",
            target: TargetDb::Core,
            body: MigrationBody::Sql(
                "ALTER TABLE chat_history ADD COLUMN reply_author TEXT; ALTER TABLE chat_history ADD COLUMN reply_snippet TEXT;",
            ),
        },
    ];

    /// [`OLD_CATALOG`] minus entry `"23"` — the chain state of a 0.5.0 release
    /// database (one delta behind: no `chat_history` reply columns). `Migration`
    /// is `Copy`, so this is a cheap filter.
    fn old_catalog_without_reply_delta() -> Vec<Migration> {
        OLD_CATALOG
            .iter()
            .filter(|m| m.id != "23")
            .copied()
            .collect()
    }

    // ── Tests ──────────────────────────────────────────────────────────

    /// A fresh install runs the baseline (`24`) plus the `25`/`27`–`34`/`38`/`42`
    /// upfills and the `36`–`57`/`59`–`63`/`65` tail, and converges to the exact current
    /// core shape: the table set (which also proves the required absences of
    /// `user_roles` / `config_role` / `ticket_jobs` / `ticket_stage_jobs`) and
    /// the per-table column sets (which prove the absences of `assigned_to` /
    /// `pipeline_reservation` / `paused_frozen` and of the account-kind
    /// `users.permissions` / `users.selected_role` / `alarms.command`).
    #[tokio::test]
    async fn fresh_install_converges_to_expected_shape() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .expect("fresh consolidated store");

        assert_schema_matches(
            &conn,
            EXPECTED_CORE_TABLE_COLUMNS,
            EXPECTED_CORE_INDEXES,
            "fresh core",
        )
        .await;

        let mut applied = applied_ids(&conn).await;
        applied.sort();
        assert_eq!(
            applied,
            [
                "24", "25", "27", "28", "29", "30", "31", "32", "33", "34", "36", "37", "38", "39",
                "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53",
                "54", "55", "56", "57", "59", "60", "61", "62", "63", "65"
            ]
            .map(String::from),
            "fresh core applies the 24–34 baseline + the 36–57/59–63/65 tail exactly"
        );
    }

    /// Delta `65` must have nothing to remove on a fresh install: the only
    /// creators of the four indexes it drops were the old baseline and delta
    /// `28`'s body, and this replays the catalog up to the drop to show that
    /// neither creates them any more. A `CREATE` left behind in either would
    /// make a fresh store build an index only for `65` to drop it again — the
    /// only place that regression is observable, since a dropped index leaves no
    /// trace in the store. Only the absence is asserted: what else a pre-`65`
    /// store carries is [`fresh_install_converges_to_expected_shape`]'s to say,
    /// and holding this one to the current shape would fail it on the next index
    /// delta instead of on a regression.
    #[tokio::test]
    async fn a_fresh_core_store_before_delta_65_creates_none_of_the_dropped_indexes() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        let entry = MIGRATIONS
            .iter()
            .position(|m| m.id == "65")
            .expect("entry 65 in the catalog");

        run_catalog(&conn, TargetDb::Core, &MIGRATIONS[..entry])
            .await
            .expect("catalog through delta 64");

        let indexes = index_defs(&conn).await;
        // The prefix really built the store — a `position` that missed would
        // leave the check below vacuously true.
        assert!(
            indexes.contains_key("idx_tickets_board_active"),
            "the catalog through delta 64 must have built the core store"
        );
        for idx in REMOVED_CORE_INDEXES {
            assert!(
                !indexes.contains_key(*idx),
                "the core catalog before delta 65 must not create {idx}"
            );
        }
    }

    /// The install that ran delta `28` before its index creation was retired
    /// carries `idx_jobs_caller_agent`, and the reopen must take it away — the
    /// other three indexes `65` drops are the old baseline's, which the fleet
    /// pin reopens with.
    #[tokio::test]
    async fn delta_65_drops_the_index_the_retired_delta_28_created() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .expect("fresh consolidated store");
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_caller_agent ON jobs(caller_agent_id) \
             WHERE caller_agent_id IS NOT NULL",
            (),
        )
        .await
        .expect("the delta-28 index as it was created before its retirement");
        conn.execute("DELETE FROM schema_migrations WHERE id = '65'", ())
            .await
            .expect("forget delta 65");

        run_migrations(&conn, TargetDb::Core).await.expect("reopen");

        assert_index_set(
            &conn,
            EXPECTED_CORE_INDEXES,
            "core reopen with the retired delta-28 index",
        )
        .await;
    }

    /// A database that applied the `telegram_intake` table in the shape delta `59` first
    /// landed it — without `claimed_by` — is upgraded by entry `60`. An applied id is
    /// never re-run, so the column cannot be folded into `59`: `60` is what carries it to
    /// an install that ran the earlier revision, which is why the entry exists.
    #[tokio::test]
    async fn a_telegram_intake_table_without_claimed_by_is_upgraded() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .expect("fresh consolidated store");
        // The state such an install is in: the table of the earlier `59`, with `60` —
        // added to the catalog later — not yet applied.
        conn.execute("DROP TABLE telegram_intake", ())
            .await
            .expect("drop the table");
        conn.execute(
            "CREATE TABLE telegram_intake (\
                 account_id TEXT NOT NULL,\
                 update_id  INTEGER NOT NULL,\
                 state      TEXT NOT NULL,\
                 payload    TEXT,\
                 created_at TEXT NOT NULL,\
                 decided_at TEXT,\
                 PRIMARY KEY (account_id, update_id)\
             )",
            (),
        )
        .await
        .expect("the table as delta 59 first created it");
        conn.execute("DELETE FROM schema_migrations WHERE id = '60'", ())
            .await
            .expect("forget delta 60");
        assert!(
            !column_names(&conn, "telegram_intake")
                .await
                .contains(&"claimed_by".to_string()),
            "the fixture must lack the column the entry adds"
        );

        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("reopen the catalog");

        assert!(
            column_names(&conn, "telegram_intake")
                .await
                .contains(&"claimed_by".to_string()),
            "delta 60 must add claimed_by to a table delta 59 created without it"
        );
    }

    /// Delta `62` must copy each recorded `session_metadata.personal_changes`
    /// value into `session_personal_changes` byte-for-byte before delta `63`
    /// drops the source column: the baseline is the Assistant's change record,
    /// so a value that failed to land would silently reset every session's
    /// change detection. A row whose baseline is NULL gets no child row. The
    /// copy runs once — a second reopen must not disturb the table.
    ///
    /// One seeded baseline is megabytes of JSON, the size the split exists for:
    /// a value of that size is an overflow chain of its own, so the copy and the
    /// drop rewrite it out of the session's row rather than editing it in place.
    #[tokio::test]
    async fn a_recorded_baseline_moves_to_its_own_table() {
        use std::fmt::Write as _;

        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        let entry = MIGRATIONS
            .iter()
            .position(|m| m.id == "62")
            .expect("entry 62 in the catalog");
        run_catalog(&conn, TargetDb::Core, &MIGRATIONS[..entry])
            .await
            .expect("catalog up to delta 62");

        let baseline = r#"{"a.txt":{"size":7,"mtime_ns":11,"digest":13}}"#;
        // A real baseline is megabytes of JSON: a value that size fills an
        // overflow chain the copy and the drop must carry intact.
        let mut large = String::new();
        while large.len() < 6 * 1024 * 1024 {
            let at = large.len();
            let _ = write!(
                large,
                r#""dir{at}/file.txt":{{"size":7,"mtime_ns":11,"digest":13}},"#
            );
        }
        let now = crate::db::now();
        for (agent_id, value) in [
            ("pc_with_baseline", Some(baseline)),
            ("pc_large", Some(large.as_str())),
            ("pc_without", None),
        ] {
            conn.execute(
                "INSERT INTO session_metadata (agent_id, last_activity, personal_changes) \
                 VALUES (?1, ?2, ?3)",
                params![agent_id, now.as_str(), value],
            )
            .await
            .unwrap();
        }

        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("finish the catalog");

        let moved = baseline_rows(&conn).await;
        assert_eq!(
            moved.get("pc_with_baseline").map(String::as_str),
            Some(baseline),
            "delta 62 must carry the baseline across byte-for-byte"
        );
        assert_eq!(
            moved.get("pc_large").map(String::as_str),
            Some(large.as_str()),
            "a multi-megabyte baseline must survive the copy and the drop intact"
        );
        assert!(
            !moved.contains_key("pc_without"),
            "a NULL baseline must not get a child row"
        );
        assert!(
            !column_exists(&conn, "session_metadata", "personal_changes")
                .await
                .unwrap(),
            "delta 63 must drop the retired column"
        );

        let before = baseline_rows(&conn).await;
        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("reopen the catalog");
        assert_eq!(
            baseline_rows(&conn).await,
            before,
            "a recorded baseline must not be rewritten by a later reopen"
        );
    }

    async fn baseline_rows(conn: &Connection) -> std::collections::BTreeMap<String, String> {
        conn.query(
            "SELECT agent_id, baseline FROM session_personal_changes",
            (),
        )
        .await
        .expect("read the baseline table")
        .into_iter()
        .map(|row| {
            (
                row.get::<String>(0).expect("agent_id"),
                row.get::<String>(1).expect("baseline"),
            )
        })
        .collect()
    }

    /// A fresh logs store runs the baseline (`26`) plus the delta `35` and
    /// delta `58` upfills and the delta `64` index drops, and converges to the
    /// exact current logs shape (logs/tool_calls/llm_requests/llm_failures +
    /// grep_telemetry, carrying only the indexes a product query's plan
    /// reaches).
    #[tokio::test]
    async fn fresh_logs_install_converges_to_expected_shape() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::LOG_DB_NAME),
            "",
        )
        .await
        .expect("open logs");
        run_migrations(&conn, TargetDb::Logs)
            .await
            .expect("run logs catalog");

        assert_schema_matches(
            &conn,
            EXPECTED_LOGS_TABLE_COLUMNS,
            EXPECTED_LOGS_INDEXES,
            "fresh logs",
        )
        .await;

        let mut applied = applied_ids(&conn).await;
        applied.sort();
        assert_eq!(
            applied,
            vec![
                "26".to_string(),
                "35".to_string(),
                "58".to_string(),
                "64".to_string()
            ],
            "fresh logs applies baseline 26 + delta 35 + delta 58 + delta 64 exactly"
        );
    }

    /// The upgrade a live install takes: a current-shape store that still
    /// carries the full pre-`64` index set drops exactly the unused ones on
    /// reopen and ends with the fresh store's index set — the fresh/existing
    /// convergence `64` guarantees.
    #[tokio::test]
    async fn a_pre_delta_64_store_converges_to_the_kept_index_set_on_reopen() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::LOG_DB_NAME),
            "",
        )
        .await
        .expect("open logs");
        run_migrations(&conn, TargetDb::Logs)
            .await
            .expect("catalog");
        // Rebuild the pre-`64` shape: every index the old baseline and the grep
        // telemetry upfill created, plus the three the old `llm_failures` DDL
        // carried, with the drop no longer recorded.
        conn.execute_batch(OLD_BASELINE_LOGS_INDEXES)
            .await
            .expect("old baseline index DDL");
        conn.execute_batch(OLD_DELTA_GREP_TELEMETRY)
            .await
            .expect("old grep telemetry DDL");
        for ddl in [
            "CREATE INDEX IF NOT EXISTS idx_llm_failures_recorded_at ON llm_failures(recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_llm_failures_operation_id ON llm_failures(operation_id)",
            "CREATE INDEX IF NOT EXISTS idx_llm_failures_failure_class ON llm_failures(failure_class)",
        ] {
            conn.execute(ddl, ())
                .await
                .expect("old llm_failures index DDL");
        }
        conn.execute("DELETE FROM schema_migrations WHERE id = '64'", ())
            .await
            .expect("forget delta 64");
        let before = index_defs(&conn).await;
        for idx in [
            "idx_logs_agent_id",
            "idx_tool_calls_tool_name",
            "idx_llm_requests_purpose",
            "idx_llm_failures_operation_id",
            "idx_grep_telemetry_reason",
        ] {
            assert!(
                before.contains_key(idx),
                "the fixture must carry the index delta 64 removes, missing {idx}"
            );
        }

        run_migrations(&conn, TargetDb::Logs).await.expect("reopen");

        assert_schema_matches(
            &conn,
            EXPECTED_LOGS_TABLE_COLUMNS,
            EXPECTED_LOGS_INDEXES,
            "pre-delta-64 reopen",
        )
        .await;
    }

    /// The common upgrade path: a database that already applied the logs
    /// baseline (`26`) — no `llm_failures` — gains the table through delta
    /// `35` on reopen (the `tool_calls.source` column through delta `58`, the
    /// unused-index drops through delta `64`), converging to the current shape
    /// with no index on the write-only `llm_failures` table.
    #[tokio::test]
    async fn baseline_logs_db_gains_llm_failures_via_delta_35() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::LOG_DB_NAME),
            "",
        )
        .await
        .expect("open logs");
        // Apply only through the baseline: the catalog prefix 24–26 (the
        // core-target entries are skipped, so exactly entry `26` lands). The
        // CURRENT baseline DDL already carries `llm_failures` — drop it to
        // reproduce the previous release's baseline-26 shape.
        run_catalog(&conn, TargetDb::Logs, &MIGRATIONS[..3])
            .await
            .expect("baseline only");
        conn.execute("DROP TABLE llm_failures", ()).await.unwrap();
        assert!(!table_defs(&conn).await.contains_key("llm_failures"));

        run_migrations(&conn, TargetDb::Logs)
            .await
            .expect("delta 35");

        let mut applied = applied_ids(&conn).await;
        applied.sort();
        assert_eq!(
            applied,
            vec![
                "26".to_string(),
                "35".to_string(),
                "58".to_string(),
                "64".to_string()
            ],
            "reopen must record exactly 26 + 35 + 58 + 64"
        );
        // The reopen converges to the current logs shape, which proves both the
        // delta-35 `llm_failures` upfill and delta 64's exact index set (none
        // of them on `llm_failures`).
        assert_schema_matches(
            &conn,
            EXPECTED_LOGS_TABLE_COLUMNS,
            EXPECTED_LOGS_INDEXES,
            "reopened baseline logs db",
        )
        .await;
    }

    /// Seed representative domain rows into a fully current-shape core
    /// database, so the no-op reopen has data to protect.
    async fn seed_current_core_rows(conn: &Connection) {
        let now = crate::db::now();
        conn.execute(
            "INSERT INTO tickets (id, title, description, phase, workspace_name, created_at, updated_at) \
             VALUES ('T1', 't', 'd', 'backlog', 'ws', ?1, ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO ticket_comments (id, ticket_id, role, content, created_at) \
             VALUES ('C1', 'T1', 'manager', 'ship it', ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO jobs (id, kind, role, workspace_name, task, user_name, channel, retry_count, \
             status, created_at, updated_at, ticket_id) \
             VALUES ('J1', 'analysis', 'analyst', 'ws', '', 'bob', '', 0, 'launched', ?1, ?1, 'T1')",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO agents (job_id, agent_id, kind, idx, status, outcome, task) \
             VALUES ('J1', 'A1', 'analyst', 0, 'done', NULL, 'task')",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO users (name, selected_workspace) VALUES ('bob', 'ws')",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO chat_history (message_id, user_name, direction, content, agent_role, workspace) \
             VALUES ('m1', 'bob', 'in', 'hello', NULL, 'ws')",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO ticket_chronicle (ticket_id, workspace_name, source_phase, target_phase, at) \
             VALUES ('T1', 'ws', 'backlog', 'queued', ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO alarms (id, session_id, user_name, kind, text, next_fire_at, created_at) \
             VALUES ('a1', 's1', 'bob', 'reminder', 'ping', ?1, ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute("INSERT INTO config_kv (key, value) VALUES ('k', 'v')", ())
            .await
            .unwrap();
    }

    /// A snapshot of everything the reopen must leave untouched.
    struct CoreSnapshot {
        tables: std::collections::BTreeMap<String, String>,
        indexes: std::collections::BTreeMap<String, String>,
        cols: std::collections::BTreeMap<String, Vec<String>>,
        counts: std::collections::BTreeMap<String, i64>,
        chat: Vec<String>,
        tickets: Vec<(String, String)>,
    }

    async fn snapshot_core_state(conn: &Connection) -> CoreSnapshot {
        CoreSnapshot {
            tables: table_defs(conn).await,
            indexes: index_defs(conn).await,
            cols: column_sets(conn, &expected_core_table_names()).await,
            counts: table_row_counts(conn, &counted_core_domain_tables()).await,
            chat: conn
                .query("SELECT content FROM chat_history", ())
                .await
                .unwrap()
                .into_iter()
                .map(|r| r.get::<String>(0).unwrap())
                .collect(),
            tickets: conn
                .query("SELECT title, phase FROM tickets", ())
                .await
                .unwrap()
                .into_iter()
                .map(|r| (r.get::<String>(0).unwrap(), r.get::<String>(1).unwrap()))
                .collect(),
        }
    }

    /// Filter a `name → value` map down to the entries whose key is not in
    /// `exclude` — used to compare a catalog snapshot "except" certain tables
    /// whose DDL/columns legitimately change on reopen (delta `27`).
    fn without<V: Clone>(
        map: &std::collections::BTreeMap<String, V>,
        exclude: &[&str],
    ) -> std::collections::BTreeMap<String, V> {
        map.iter()
            .filter(|(name, _)| !exclude.contains(&name.as_str()))
            .map(|(name, v)| (name.clone(), v.clone()))
            .collect()
    }

    /// Assert that reopening the core catalog leaves the database unchanged,
    /// except for `except_tables`, whose table DDL and column set may differ on
    /// reopen (the delta-27 `workspaces` column upfill, the delta-28
    /// `jobs`/`session_metadata` column upfills, the delta-47 `jobs.exec_dir`
    /// column, the delta-54 `jobs.custom_launch` column, the delta-55
    /// `session_metadata.personal_changes` column moved into delta-62's
    /// `session_personal_changes` table by the delta-63 drop, the delta-56
    /// `session_metadata.reply_to` column, the delta-57
    /// `session_metadata.context_block_len` column, and the
    /// delta-51/52/53 tail — `custom_roles` plus the custom-role columns on
    /// `session_metadata` and
    /// `alarms`). Row counts, `chat_history` content and `tickets` content are
    /// asserted to be strictly unchanged. The index set must lose exactly
    /// [`REMOVED_CORE_INDEXES`] and keep every other index byte-identical; that
    /// the result is [`EXPECTED_CORE_INDEXES`] is asserted separately by the
    /// caller — reopen must converge to it, not merely keep what the pre-reopen
    /// shape carried (delta `65`).
    async fn assert_core_catalog_unchanged(
        conn: &Connection,
        before: &CoreSnapshot,
        except_tables: &[&str],
    ) {
        assert_eq!(
            without(&table_defs(conn).await, except_tables),
            without(&before.tables, except_tables),
            "core table DDL (except {except_tables:?}) must be unchanged on reopen"
        );
        assert_eq!(
            without(&before.indexes, REMOVED_CORE_INDEXES),
            index_defs(conn).await,
            "core index set must lose exactly the unreached indexes on reopen"
        );
        assert_eq!(
            without(
                &column_sets(conn, &expected_core_table_names()).await,
                except_tables
            ),
            without(&before.cols, except_tables),
            "core column sets (except {except_tables:?}) must be unchanged on reopen"
        );
        assert_eq!(
            table_row_counts(conn, &counted_core_domain_tables()).await,
            before.counts,
            "core row counts must be unchanged on reopen"
        );
        let after_chat: Vec<String> = conn
            .query("SELECT content FROM chat_history", ())
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.get::<String>(0).unwrap())
            .collect();
        assert_eq!(
            after_chat, before.chat,
            "chat_history content must be unchanged"
        );
        let after_tickets: Vec<(String, String)> = conn
            .query("SELECT title, phase FROM tickets", ())
            .await
            .unwrap()
            .into_iter()
            .map(|r| (r.get::<String>(0).unwrap(), r.get::<String>(1).unwrap()))
            .collect();
        assert_eq!(
            after_tickets, before.tickets,
            "tickets content must be unchanged"
        );
    }

    /// The core fleet-wide boot-safety pin: a database shaped by the REAL
    /// retired `1`–`23` chain (logged ids 1–23 recorded) must reopen through
    /// the new baseline (`24`/`25`/`27`–`34`/`36`–`57`/`59`–`63`/`65`) as a
    /// STRICT no-op except the delta-27 `workspaces.maintainer_recommendations`
    /// column upfill, the delta-28 `jobs.caller_agent_id` /
    /// `session_metadata.created_at` column upfills, the delta-29 per-user model
    /// columns, the
    /// delta-30 `jobs.mode` column upfill + backfill, the delta-31
    /// `sleep_ended` column upfill, the delta-32 `alarms.command` column
    /// upfill, the delta-33 `chat_history.broadcast_id` column upfill, the
    /// delta-34 `tickets.last_transition_actor` / `ticket_chronicle.actor`
    /// column upfill, the delta-38 `users.granted_tools` column upfill, the
    /// delta-39/40 `users` column drops, and the delta-41/42/43 `alarms`
    /// command retirement (delete the command-armed rows, add `trigger`, drop
    /// `command`) —
    /// plus the delta-36 data rewrite, which detaches the seeded guest from
    /// `users.selected_workspace` (row counts unchanged; the snapshot compares
    /// only counts, chat content and tickets), and the delta-65 index drops,
    /// which converge the index set onto the fresh store's. All asserted
    /// explicitly. The
    /// entry-46 FTS rebuild is invisible to the snapshot by construction (the
    /// index keeps its name and DDL text and the internal backing objects are
    /// excluded), so this also proves Turso honors `IF NOT EXISTS` on the FTS
    /// index when the baseline re-runs it and that dropping and recreating the
    /// index leaves the rest of the store alone.
    ///
    /// The stage-merge and phase-retirement tail (`44`/`45`/`48`/`49`), the
    /// image-model retirement (`50`) and the custom-role / custom-launch /
    /// personal-changes / context-block-length columns (`51`–`57`, with deltas
    /// `62`/`63` moving the personal-changes baseline into a table of its own)
    /// are a no-op here too: the seeded rows
    /// name none of the retired stages and claim no earlier command run, so no job
    /// is deleted, no phase is rewritten and the empty `workspace_contexts` stays
    /// empty, and the seeded `config_kv` row is an unrelated key while the
    /// seeded account's pick is NULL.
    #[tokio::test]
    #[expect(clippy::too_many_lines)]
    async fn old_catalog_current_db_reopens_as_noop() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        run_catalog(&conn, TargetDb::Core, OLD_CATALOG)
            .await
            .expect("old catalog");
        seed_current_core_rows(&conn).await;

        let before_ids = applied_ids(&conn).await;
        let before = snapshot_core_state(&conn).await;

        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("new catalog");

        let mut expected_ids = before_ids.clone();
        for id in [
            "24", "25", "27", "28", "29", "30", "31", "32", "33", "34", "36", "37", "38", "39",
            "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53",
            "54", "55", "56", "57", "59", "60", "61", "62", "63", "65",
        ] {
            expected_ids.push(id.to_string());
        }
        expected_ids.sort();
        let mut after_ids = applied_ids(&conn).await;
        after_ids.sort();
        assert_eq!(
            after_ids, expected_ids,
            "reopen must record exactly old ids ∪ 24/25/27..34/36..57/59–63/65"
        );

        // Everything else is a strict no-op; only workspaces (delta 27),
        // jobs/session_metadata (delta 28), users (deltas 29/38/39/40),
        // jobs.mode (delta 30), session_metadata.sleep_ended (delta 31),
        // alarms.command (delta 32, retired by deltas 41/42/43),
        // chat_history.broadcast_id (delta 33),
        // tickets/ticket_chronicle (delta 34), jobs.exec_dir (delta 47),
        // jobs.custom_launch (delta 54), session_metadata.personal_changes
        // (delta 55, moved to `session_personal_changes` by deltas 62/63),
        // session_metadata.reply_to (delta 56),
        // session_metadata.context_block_len (delta 57),
        // telegram_intake (delta 59) and the custom-role tail (delta 51's
        // `custom_roles` table plus the session_metadata/alarms columns of
        // deltas 52/53) change in shape, and delta `65` drops the unreached
        // indexes. Delta `36` rewrites `users.selected_workspace` data
        // (detaching the seeded guest), which leaves row counts unchanged — the
        // snapshot compares only counts, chat content and tickets, so the no-op
        // assertions still hold. The seeded alarm row carries no command, so the
        // delta-41 delete matches no rows and the count is unchanged there too.
        assert_core_catalog_unchanged(
            &conn,
            &before,
            &[
                "workspaces",
                "jobs",
                "session_metadata",
                "users",
                "alarms",
                "chat_history",
                "tickets",
                "ticket_chronicle",
                // Introduced by delta 51 — absent from the pre-reopen fixture.
                "custom_roles",
                // Introduced by delta 59 — absent from the pre-reopen fixture.
                "telegram_intake",
                // Introduced by delta 61 — absent from the pre-reopen fixture.
                "telegram_bursts",
                // Introduced by delta 62 — absent from the pre-reopen fixture.
                "session_personal_changes",
            ],
        )
        .await;

        // Delta `65` must drop the unreached core indexes: the old shape carries
        // the old baseline's three of them, and the reopen converges the index
        // set onto the fresh store's — the fresh/existing equality the entry
        // guarantees. (`idx_jobs_caller_agent`, the fourth, is delta `28`'s and
        // is covered by `delta_65_drops_the_index_the_retired_delta_28_created`.)
        for idx in [
            "idx_jobs_updated_at",
            "idx_chat_history_user",
            "idx_chat_history_workspace",
        ] {
            assert!(
                before.indexes.contains_key(idx),
                "the old shape must carry the index delta 65 removes, missing {idx}"
            );
        }
        assert_index_set(&conn, EXPECTED_CORE_INDEXES, "core reopen").await;
        let after_ws_cols = column_sets(&conn, &["workspaces"]).await;
        let mut expected_ws_cols = before.cols["workspaces"].clone();
        expected_ws_cols.push("maintainer_recommendations".to_string());
        assert_eq!(
            after_ws_cols["workspaces"], expected_ws_cols,
            "reopen must append exactly maintainer_recommendations to workspaces columns"
        );
        let after_jobs_cols = column_sets(&conn, &["jobs"]).await;
        let mut expected_jobs_cols = before.cols["jobs"].clone();
        expected_jobs_cols.push("caller_agent_id".to_string());
        expected_jobs_cols.push("mode".to_string());
        expected_jobs_cols.push("exec_dir".to_string());
        expected_jobs_cols.push("custom_launch".to_string());
        assert_eq!(
            after_jobs_cols["jobs"], expected_jobs_cols,
            "reopen must append exactly caller_agent_id/mode/exec_dir/custom_launch to jobs columns"
        );
        let after_sm_cols = column_sets(&conn, &["session_metadata"]).await;
        let mut expected_sm_cols = before.cols["session_metadata"].clone();
        expected_sm_cols.push("created_at".to_string());
        expected_sm_cols.push("sleep_ended".to_string());
        expected_sm_cols.push("custom_role".to_string());
        expected_sm_cols.push("custom_identifier".to_string());
        expected_sm_cols.push("launch_stopped".to_string());
        expected_sm_cols.push("reply_to".to_string());
        expected_sm_cols.push("context_block_len".to_string());
        assert_eq!(
            after_sm_cols["session_metadata"], expected_sm_cols,
            "reopen must append exactly created_at/sleep_ended/custom_role/custom_identifier/\
             launch_stopped/reply_to/context_block_len to session_metadata columns"
        );
        let after_users_cols = column_sets(&conn, &["users"]).await;
        let mut expected_users_cols = before.cols["users"].clone();
        expected_users_cols.retain(|c| c != "permissions" && c != "selected_role");
        expected_users_cols.push("image_gen_model".to_string());
        expected_users_cols.push("video_model".to_string());
        expected_users_cols.push("granted_tools".to_string());
        assert_eq!(
            after_users_cols["users"], expected_users_cols,
            "reopen must append exactly image_gen_model/video_model/granted_tools \
             and drop permissions/selected_role from users columns"
        );
        let after_alarms_cols = column_sets(&conn, &["alarms"]).await;
        let mut expected_alarms_cols = before.cols["alarms"].clone();
        expected_alarms_cols.push("trigger".to_string());
        expected_alarms_cols.push("role".to_string());
        expected_alarms_cols.push("identifier".to_string());
        expected_alarms_cols.push("workspace".to_string());
        assert_eq!(
            after_alarms_cols["alarms"], expected_alarms_cols,
            "reopen must trade command (delta 32) for trigger (deltas 42/43) and add the \
             custom-role target columns (delta 53) on alarms columns"
        );
        let after_chat_cols = column_sets(&conn, &["chat_history"]).await;
        let mut expected_chat_cols = before.cols["chat_history"].clone();
        expected_chat_cols.push("broadcast_id".to_string());
        assert_eq!(
            after_chat_cols["chat_history"], expected_chat_cols,
            "reopen must append exactly broadcast_id to chat_history columns"
        );
        let after_tickets_cols = column_sets(&conn, &["tickets"]).await;
        let mut expected_tickets_cols = before.cols["tickets"].clone();
        expected_tickets_cols.push("last_transition_actor".to_string());
        assert_eq!(
            after_tickets_cols["tickets"], expected_tickets_cols,
            "reopen must append exactly last_transition_actor to tickets columns"
        );
        let after_chronicle_cols = column_sets(&conn, &["ticket_chronicle"]).await;
        let mut expected_chronicle_cols = before.cols["ticket_chronicle"].clone();
        expected_chronicle_cols.push("actor".to_string());
        assert_eq!(
            after_chronicle_cols["ticket_chronicle"], expected_chronicle_cols,
            "reopen must append exactly actor to ticket_chronicle columns"
        );
    }

    /// The logs fleet-wide boot-safety pin: an old-catalog logs store (ids
    /// `8`/`9`/`14`) reopens through the logs baseline (`26`), the delta `35`
    /// upfill, the delta `58` column upfill and the delta `64` index drops —
    /// the table DDL is unchanged except the new `llm_failures` table and the
    /// appended `tool_calls.source` column, and the index set converges to the
    /// kept indexes (every write-only one the old shape carried is gone).
    #[tokio::test]
    async fn old_catalog_logs_db_reopens_as_noop() {
        const NEW_TABLE: &str = "llm_failures";
        // `llm_failures` does not exist in the old-catalog shape, and delta `58`
        // appends a column to `tool_calls` — exclude both from the
        // unchanged-reopen comparison; their arrival on reopen is asserted
        // below. Row counts are compared over the tables that exist on both
        // sides.
        let counts_tables: Vec<&str> = expected_logs_domain_tables()
            .into_iter()
            .filter(|n| *n != NEW_TABLE)
            .collect();
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::LOG_DB_NAME),
            "",
        )
        .await
        .expect("open logs");
        run_catalog(&conn, TargetDb::Logs, OLD_CATALOG)
            .await
            .expect("old catalog");
        conn.execute(
            "INSERT INTO grep_telemetry (recorded_at, command) VALUES (?1, 'grep foo')",
            params![crate::db::now()],
        )
        .await
        .unwrap();

        let before_ids = applied_ids(&conn).await;
        let before_tables = table_defs(&conn).await;
        let before_indexes = index_defs(&conn).await;
        let before_tool_calls_cols = column_names(&conn, "tool_calls").await;
        let before_counts = table_row_counts(&conn, &counts_tables).await;

        run_migrations(&conn, TargetDb::Logs)
            .await
            .expect("new catalog");

        let mut expected_ids = before_ids.clone();
        expected_ids.push("26".to_string());
        expected_ids.push("35".to_string());
        expected_ids.push("58".to_string());
        expected_ids.push("64".to_string());
        expected_ids.sort();
        let mut after_ids = applied_ids(&conn).await;
        after_ids.sort();
        assert_eq!(
            after_ids, expected_ids,
            "logs reopen must record exactly old ids ∪ 26 ∪ 35 ∪ 58 ∪ 64"
        );

        assert_eq!(
            without(&table_defs(&conn).await, &[NEW_TABLE, "tool_calls"]),
            without(&before_tables, &[NEW_TABLE, "tool_calls"]),
            "logs table DDL (except {NEW_TABLE} and the delta-58 tool_calls column) must be \
             unchanged on reopen"
        );
        assert_eq!(
            table_row_counts(&conn, &counts_tables).await,
            before_counts,
            "logs row counts must be unchanged on reopen"
        );

        // Delta `58` must land the call-source discriminator on existing
        // installs — appended by the `ALTER` (a fresh install's baseline
        // declares it after `role` instead).
        let mut expected_cols = before_tool_calls_cols;
        expected_cols.push("source".to_string());
        assert_eq!(
            column_names(&conn, "tool_calls").await,
            expected_cols,
            "delta 58 must append source to tool_calls"
        );

        // Delta `35` must land the per-attempt failure trail on existing
        // installs — the whole point of the two-sided migration.
        let after_tables = table_defs(&conn).await;
        assert!(
            after_tables.contains_key(NEW_TABLE),
            "delta 35 must create {NEW_TABLE} on existing installs"
        );

        // Delta `64` must drop the unread observability indexes: the old shape
        // carries `idx_logs_agent_id`, and the reopen converges the index set
        // onto the fresh store's.
        assert!(
            before_indexes.contains_key("idx_logs_agent_id"),
            "the old shape must carry an index delta 64 removes"
        );
        assert_index_set(&conn, EXPECTED_LOGS_INDEXES, "logs reopen").await;
    }

    /// The 0.5.0 (one-delta-behind) upgrade: a database built by the retired
    /// chain WITHOUT entry `23` has no `chat_history` reply columns; the
    /// baseline entry `25` upfills exactly those two columns, entry `27` adds
    /// `workspaces.maintainer_recommendations`, entry `28` adds
    /// `jobs.caller_agent_id` / `session_metadata.created_at`, entry `29` adds
    /// the per-user `image_gen_model` / `video_model` columns, entry `30` adds
    /// `jobs.mode`, entry `31` adds `session_metadata.sleep_ended`, entry `32`
    /// adds `alarms.command`, and entry `33` adds `chat_history.broadcast_id`,
    /// and entry `34` adds
    /// `tickets.last_transition_actor` / `ticket_chronicle.actor`. The tail is
    /// entries `36`–`57`/`59`–`63`/`65`: the data migrations (detaching guests from
    /// shared workspaces, rewriting the legacy `DeepSeek` routing slug,
    /// deleting the command-armed alarm rows, rewriting the retired
    /// `in_review`/`in_qa` stages to `verification`, invalidating the
    /// stored workspace lifecycle descriptions that still name them, and
    /// replacing the retired image-generation model in the picker list and in
    /// the account picks), the `users.granted_tools`
    /// column, the drops of the account-kind `users.permissions` /
    /// `users.selected_role` columns, the `alarms.trigger` upfill plus the
    /// `alarms.command` drop that retire the command-armed alarm feature, the
    /// custom-role tail (`custom_roles` plus the `session_metadata` and
    /// `alarms` columns), and the `jobs.exec_dir` / `jobs.custom_launch` /
    /// `session_metadata.reply_to` /
    /// `session_metadata.context_block_len` columns, and the `telegram_intake`
    /// table of delta `59` (its `claimed_by` column guaranteed by `60`), and the
    /// `telegram_bursts` table of delta `61`, and deltas `62`/`63`, which move
    /// the personal-changes baseline into `session_personal_changes`, and `65`,
    /// which drops the indexes no production query's plan reaches.
    #[expect(clippy::too_many_lines)] // large table-driven migration fixture
    #[tokio::test]
    async fn one_delta_behind_db_upgrades_reply_columns() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        run_catalog(&conn, TargetDb::Core, &old_catalog_without_reply_delta())
            .await
            .expect("old catalog minus 23");

        let now = crate::db::now();
        conn.execute(
            "INSERT INTO chat_history (message_id, user_name, direction, content, agent_role, workspace, timestamp) \
             VALUES ('m1', 'alice', 'in', 'hello', NULL, 'ws', ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO chat_history (message_id, user_name, direction, content, agent_role, workspace, timestamp) \
             VALUES ('m2', 'assistant', 'out', 'hi alice', 'assistant', 'ws', ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();

        let before_chat_cols = column_names(&conn, "chat_history").await;
        assert!(
            !before_chat_cols.contains(&"reply_author".to_string()),
            "0.5.0 DB must not have reply_author"
        );
        assert!(
            !before_chat_cols.contains(&"reply_snippet".to_string()),
            "0.5.0 DB must not have reply_snippet"
        );
        let untouched_tables: Vec<&str> = expected_core_table_names()
            .into_iter()
            .filter(|n| {
                *n != "chat_history"
                    && *n != "workspaces"
                    && *n != "jobs"
                    && *n != "session_metadata"
                    && *n != "users"
                    && *n != "alarms"
                    && *n != "tickets"
                    && *n != "ticket_chronicle"
                    && *n != "schema_migrations"
                    // Introduced by delta `51` — absent before the upgrade.
                    && *n != "custom_roles"
                    // Introduced by delta `59` — absent from the pre-reopen fixture.
                    && *n != "telegram_intake"
                    // Introduced by delta `61` — absent from the pre-reopen fixture.
                    && *n != "telegram_bursts"
                    // Introduced by delta `62` — absent from the pre-reopen fixture.
                    && *n != "session_personal_changes"
            })
            .collect();
        let before_other = column_sets(&conn, &untouched_tables).await;
        let mut before_ids = applied_ids(&conn).await;
        before_ids.sort();

        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("new catalog");

        let after_chat_cols = column_names(&conn, "chat_history").await;
        assert!(
            after_chat_cols.contains(&"reply_author".to_string()),
            "reply_author must be added"
        );
        assert!(
            after_chat_cols.contains(&"reply_snippet".to_string()),
            "reply_snippet must be added"
        );
        let after_ws_cols = column_names(&conn, "workspaces").await;
        assert!(
            after_ws_cols.contains(&"maintainer_recommendations".to_string()),
            "maintainer_recommendations must be added to workspaces"
        );

        let reply_nulls: i64 = conn
            .query(
                "SELECT COUNT(*) FROM chat_history \
                 WHERE reply_author IS NULL AND reply_snippet IS NULL",
                (),
            )
            .await
            .unwrap()[0]
            .get::<i64>(0)
            .unwrap();
        assert_eq!(reply_nulls, 2, "both rows must keep NULL reply columns");

        let contents: Vec<String> = conn
            .query("SELECT content FROM chat_history ORDER BY id", ())
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.get::<String>(0).unwrap())
            .collect();
        assert_eq!(contents, vec!["hello".to_string(), "hi alice".to_string()]);

        let mut expected_ids = before_ids.clone();
        for id in [
            "24", "25", "27", "28", "29", "30", "31", "32", "33", "34", "36", "37", "38", "39",
            "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53",
            "54", "55", "56", "57", "59", "60", "61", "62", "63", "65",
        ] {
            expected_ids.push(id.to_string());
        }
        expected_ids.sort();
        let mut after_ids = applied_ids(&conn).await;
        after_ids.sort();
        assert_eq!(
            after_ids, expected_ids,
            "upgrade must record exactly old ids ∪ 24/25/27..34/36..57/59–63/65"
        );

        let after_users_cols = column_names(&conn, "users").await;
        assert!(
            after_users_cols.contains(&"image_gen_model".to_string()),
            "image_gen_model must be added to users"
        );
        assert!(
            after_users_cols.contains(&"video_model".to_string()),
            "video_model must be added to users"
        );
        assert!(
            after_users_cols.contains(&"granted_tools".to_string()),
            "granted_tools must be added to users"
        );
        assert!(
            !after_users_cols.contains(&"permissions".to_string())
                && !after_users_cols.contains(&"selected_role".to_string()),
            "the account-kind columns must be dropped from users"
        );
        let after_alarms_cols = column_names(&conn, "alarms").await;
        assert!(
            after_alarms_cols.contains(&"trigger".to_string())
                && !after_alarms_cols.contains(&"command".to_string()),
            "the retired command column must be replaced by trigger on alarms"
        );
        let after_chat_cols = column_names(&conn, "chat_history").await;
        assert!(
            after_chat_cols.contains(&"broadcast_id".to_string()),
            "broadcast_id must be added to chat_history"
        );
        let after_tickets_cols = column_names(&conn, "tickets").await;
        assert!(
            after_tickets_cols.contains(&"last_transition_actor".to_string()),
            "last_transition_actor must be added to tickets"
        );
        let after_chronicle_cols = column_names(&conn, "ticket_chronicle").await;
        assert!(
            after_chronicle_cols.contains(&"actor".to_string()),
            "actor must be added to ticket_chronicle"
        );

        assert_eq!(
            column_sets(&conn, &untouched_tables).await,
            before_other,
            "only chat_history (delta 25/33), workspaces (delta 27), \
             jobs/session_metadata (delta 28), users (deltas 29/38), jobs.mode \
             (delta 30), session_metadata.sleep_ended (delta 31), \
             alarms.command → trigger (deltas 32/41/42/43) and \
             tickets/ticket_chronicle (delta 34) \
             columns may change on the 0.5.0 upgrade (plus the custom-role \
             columns of deltas 52/53)"
        );
    }

    /// Read every `users` row as `name → selected_workspace` for the
    /// delta-36 behavioral pin.
    async fn users_selected_workspaces(
        conn: &Connection,
    ) -> std::collections::BTreeMap<String, Option<String>> {
        conn.query("SELECT name, selected_workspace FROM users", ())
            .await
            .expect("read users")
            .into_iter()
            .map(|row| {
                (
                    row.get::<String>(0).expect("user name"),
                    row.get::<Option<String>>(1).expect("selected_workspace"),
                )
            })
            .collect()
    }

    /// Model → provider order for every routing row, ordered by model.
    async fn model_routing_orders(conn: &Connection) -> Vec<(String, String)> {
        conn.query(
            "SELECT model, provider_order FROM config_model_routing ORDER BY model",
            (),
        )
        .await
        .expect("read config_model_routing")
        .into_iter()
        .map(|row| {
            (
                row.get::<String>(0).expect("model"),
                row.get::<String>(1).expect("provider_order"),
            )
        })
        .collect()
    }

    /// Behavioral pin for delta `36`: workspace membership is admin-only, so the
    /// migration detaches every guest from shared workspaces (clearing
    /// `selected_workspace` → NULL) while leaving the admin attached.
    /// Idempotent: re-running matches no rows and changes nothing.
    #[tokio::test]
    async fn detach_guest_workspaces_clears_only_guests() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        // Catalog front entries 24 (baseline core schema — creates `users`) and
        // 25 (chat_history reply columns) suffice to bring up the table.
        run_catalog(&conn, TargetDb::Core, &MIGRATIONS[..2])
            .await
            .expect("baseline catalog");

        // The admin with a shared workspace, a guest sharing it, and a guest
        // already on their personal workspace (NULL).
        conn.execute(
            "INSERT INTO users (name, selected_workspace) VALUES ('admin', 'shared_ws')",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO users (name, selected_workspace) VALUES ('guest_shared', 'shared_ws')",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO users (name, selected_workspace) VALUES ('guest_null', NULL)",
            (),
        )
        .await
        .unwrap();

        run_detach_guest_workspaces(&conn).await.unwrap();

        let after = users_selected_workspaces(&conn).await;
        assert_eq!(
            after["admin"],
            Some("shared_ws".to_string()),
            "the admin keeps their shared workspace"
        );
        assert_eq!(
            after["guest_shared"], None,
            "guests are detached from shared workspaces"
        );
        assert_eq!(after["guest_null"], None, "already-personal stays NULL");

        // Idempotence: re-running matches no rows and changes nothing.
        run_detach_guest_workspaces(&conn).await.unwrap();
        assert_eq!(users_selected_workspaces(&conn).await, after);
    }

    /// Behavioral pin for catalog entries `39`/`40`: dropping
    /// `users.permissions` and `users.selected_role` rewrites the populated
    /// table, so every other account column must come through intact, a second
    /// execution must be a no-op (the bodies probe for their column).
    #[tokio::test]
    async fn drop_account_kind_columns_preserves_row_data() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        conn.execute(
            "CREATE TABLE users (\
             name TEXT PRIMARY KEY, permissions TEXT, selected_workspace TEXT, \
             selected_role TEXT, image_gen_model TEXT, video_model TEXT, granted_tools TEXT\
             )",
            (),
        )
        .await
        .expect("create pre-39 users table");
        conn.execute(
            "INSERT INTO users \
             (name, permissions, selected_workspace, selected_role, image_gen_model, \
              video_model, granted_tools) \
             VALUES ('admin', 'full', 'proj', 'assistant', 'img-model', 'vid-model', '[\"t\"]')",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO users (name, permissions, selected_workspace, selected_role) \
             VALUES ('guest', NULL, NULL, 'engineer')",
            (),
        )
        .await
        .unwrap();

        run_drop_users_permissions(&conn).await.unwrap();
        run_drop_users_selected_role(&conn).await.unwrap();

        let expected_columns: Vec<String> = [
            "name",
            "selected_workspace",
            "image_gen_model",
            "video_model",
            "granted_tools",
        ]
        .iter()
        .map(|c| (*c).to_string())
        .collect();
        assert_eq!(column_names(&conn, "users").await, expected_columns);
        let rows = conn
            .query(
                "SELECT name, selected_workspace, image_gen_model, video_model, granted_tools \
                 FROM users ORDER BY name",
                (),
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get::<String>(0).unwrap(), "admin");
        assert_eq!(
            rows[0].get::<Option<String>>(1).unwrap(),
            Some("proj".to_string()),
            "the admin's selected workspace must survive the table rewrite"
        );
        assert_eq!(
            rows[0].get::<Option<String>>(2).unwrap(),
            Some("img-model".to_string())
        );
        assert_eq!(
            rows[0].get::<Option<String>>(3).unwrap(),
            Some("vid-model".to_string())
        );
        assert_eq!(
            rows[0].get::<Option<String>>(4).unwrap(),
            Some("[\"t\"]".to_string())
        );
        assert_eq!(rows[1].get::<String>(0).unwrap(), "guest");
        assert_eq!(rows[1].get::<Option<String>>(1).unwrap(), None);
        assert_eq!(rows[1].get::<Option<String>>(4).unwrap(), None);

        // Idempotence: both bodies probe for their column.
        run_drop_users_permissions(&conn).await.unwrap();
        run_drop_users_selected_role(&conn).await.unwrap();
        assert_eq!(column_names(&conn, "users").await, expected_columns);
    }

    /// Behavioral pin for the alarm-tail deltas: entry `41` deletes every row
    /// that carried a shell command — active and long settled alike — while
    /// command-free rows survive, and entry `43` drops the column, leaving the
    /// new end state. Delta `42` is pinned by the one-delta-behind reopen test
    /// instead: the baseline seeded here already declares `trigger`.
    #[tokio::test]
    async fn command_armed_alarms_are_deleted_and_the_command_column_is_dropped() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        // Seed into the catalog front through delta `32`: by then the table has
        // the retired `alarms.command` (added by `32`, dropped only by `43`) and
        // the tail that follows re-runs entry `41`'s delete over what is seeded.
        // The baseline already declares `trigger`, so the seeded table carries
        // both columns — the delete only ever looks at `command`.
        let through_command = MIGRATIONS
            .iter()
            .position(|m| m.id == "32")
            .expect("entry 32 in the catalog");
        run_catalog(&conn, TargetDb::Core, &MIGRATIONS[..=through_command])
            .await
            .expect("catalog through delta 32");
        let now = crate::db::now();
        for (id, status, command) in [
            ("armed_active", "active", Some("rm -rf /tmp/x")),
            ("armed_settled", "fired", Some("say done")),
            ("reminder", "active", None),
        ] {
            conn.execute(
                "INSERT INTO alarms \
                 (id, session_id, user_name, kind, text, next_fire_at, status, created_at, command) \
                 VALUES (?1, 's1', 'bob', 'reminder', 'ping', ?2, ?3, ?2, ?4)",
                params![id, now.clone(), status, command],
            )
            .await
            .unwrap();
        }

        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("new catalog");

        let remaining: Vec<String> = conn
            .query("SELECT id FROM alarms ORDER BY id", ())
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.get::<String>(0).unwrap())
            .collect();
        assert_eq!(
            remaining,
            vec!["reminder".to_string()],
            "delta 41 must delete every command-armed alarm row and keep the rest"
        );
        // The end state carries `trigger` and no `command` (delta `43`). The
        // `trigger` half is the baseline's own declaration, so it is the reopen
        // test — not this fixture — that would catch a missing delta `42`.
        let cols = column_names(&conn, "alarms").await;
        assert!(!cols.contains(&"command".to_string()), "got: {cols:?}");
    }

    /// Behavioral pin for catalog entry `37`: the legacy DeepSeek routing value
    /// written by earlier versions (OpenRouter's display name, not the slug it
    /// resolves) is rewritten, while every other stored provider order — an
    /// already-canonical one and a genuine user choice — survives verbatim.
    /// Idempotent: a second execution matches no rows.
    #[tokio::test]
    async fn rewrite_legacy_deepseek_routing_slug_rewrites_only_the_display_name() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        // Catalog front entry 24 (baseline core schema) creates
        // `config_model_routing`.
        run_catalog(&conn, TargetDb::Core, &MIGRATIONS[..1])
            .await
            .expect("baseline catalog");
        for (model, order) in [
            ("deepseek/deepseek-v4-flash", "DeepSeek"),
            ("deepseek/deepseek-v4.1-flash", "deepseek"),
            ("qwen/qwen3.8-flash", "GMICloud"),
        ] {
            conn.execute(
                "INSERT INTO config_model_routing (model, provider_order) VALUES (?1, ?2)",
                params![model, order],
            )
            .await
            .unwrap();
        }

        let entry_37 = MIGRATIONS
            .iter()
            .find(|m| m.id == "37")
            .expect("entry 37 in the catalog");
        run_catalog(&conn, TargetDb::Core, std::slice::from_ref(entry_37))
            .await
            .expect("apply entry 37");
        let rewritten = model_routing_orders(&conn).await;
        assert_eq!(
            rewritten,
            vec![
                (
                    "deepseek/deepseek-v4-flash".to_string(),
                    "deepseek".to_string()
                ),
                (
                    "deepseek/deepseek-v4.1-flash".to_string(),
                    "deepseek".to_string()
                ),
                ("qwen/qwen3.8-flash".to_string(), "GMICloud".to_string()),
            ],
            "only the legacy display-name value may be rewritten"
        );

        // The ledger records `37` after the first application, so re-run the SQL
        // body directly to pin that it matches no rows a second time.
        conn.execute(REWRITE_LEGACY_DEEPSEEK_ROUTING_SLUG, ())
            .await
            .expect("re-run entry 37 body");
        assert_eq!(model_routing_orders(&conn).await, rewritten);
    }

    /// The stage-merge description probe catches the spellings the retired
    /// phases were written in (underscored, spaced, fused, any casing) and the
    /// lifecycle-list spellings that drop the `in`, in the separator styles the
    /// stored prose uses (` and `, `, `, ` then `), and leaves a description
    /// that names no retired stage alone.
    #[test]
    fn retired_lifecycle_description_probe_handles_the_stored_shapes() {
        for retired in [
            "a fixed lifecycle — `Backlog → Analysis → InReview → InQa → InSanitation → Done`",
            "Each phase (`in_development`, `in_diagnostics`, `in_review`, `in_qa`) owns a job",
            "the lifecycle is in development → in diagnostics → in review → in QA → in sanitation",
            "(Analysis → Planning → Queued → Development → Diagnostics → Review → QA → Sanitation)",
            "the dev/review/QA pipeline",
            "the board keeps IN_REVIEW and IN_QA columns",
            "comments come from the analysis, engineer, diagnostics, review and QA",
            "two separate checks: review, qa",
            "review then QA",
            // Greedy by design: `review` given its own arrow is invalidated even
            // when the prose is not about this pipeline — the accepted cost.
            "the docs are updated on each review → merge cycle",
        ] {
            assert!(
                describes_retired_lifecycle(retired),
                "must be invalidated: {retired:?}"
            );
        }
        for kept in [
            "the code reviewers check the change and one functional tester runs the product",
            "every phase owns a short-lived `jobs` row",
            "reviewing the diff is what the reviewer does",
            "the reviewer and the analyst read the same tree",
            "the model slots split inspector roles (analyst/coder/qa/reviewer/sanitation) from the rest",
            "`review.rs` and `qa.rs` were merged",
        ] {
            assert!(
                !describes_retired_lifecycle(kept),
                "must survive verbatim: {kept:?}"
            );
        }
    }

    /// The retired-commands-claim probe invalidates the stored shapes that tell
    /// an agent the project commands already ran before the round, and leaves
    /// verbatim both the same-round arrangement and prose that merely mentions a
    /// command running in this round.
    #[test]
    fn retired_commands_claim_probe_handles_the_stored_shapes() {
        for retired in [
            "Diagnostics (format, lint, type-check, build, unit test) already ran and are reported in the ticket's diagnostics comment; do not repeat them",
            "this is exactly what the Diagnostics phase ran before this round",
            "format, lint and build have already been verified",
            "the diagnostics stage passes before this one",
        ] {
            assert!(
                claims_commands_ran_before_the_round(retired),
                "must be invalidated: {retired:?}"
            );
        }
        for kept in [
            "the project commands run in the same round as the check",
            "run `cargo test` if you need the unit suite",
            "the local service is already running on port 3000",
            "the reviewer and one functional tester run together",
        ] {
            assert!(
                !claims_commands_ran_before_the_round(kept),
                "must survive verbatim: {kept:?}"
            );
        }
    }

    /// Role (empty for the NULL role) → content for every stored workspace
    /// context.
    async fn workspace_context_rows(
        conn: &Connection,
    ) -> std::collections::BTreeMap<String, String> {
        conn.query("SELECT role, content FROM workspace_contexts", ())
            .await
            .expect("read workspace_contexts")
            .into_iter()
            .map(|row| {
                (
                    row.get::<Option<String>>(0)
                        .expect("role")
                        .unwrap_or_default(),
                    row.get::<String>(1).expect("content"),
                )
            })
            .collect()
    }

    /// Behavioral pin for the stage-merge tail: entry `44` rewrites the retired
    /// `in_review`/`in_qa` stages to `verification` on the ticket and on both
    /// chronicle sides, DELETES the retired job rows — a ticket can hold
    /// both an `in_review` and an `in_qa` row, so a rename would collide on the
    /// unique `jobs(kind, ticket_id)` index and abort the migration — taking
    /// their cascaded `agents` roster with them, and DELETES the retired→retired
    /// chronicle hop, which never happened. Entry `45` deletes every stored
    /// workspace lifecycle description that names a retired stage and leaves a
    /// description naming none verbatim. Idempotent: re-running both bodies
    /// matches no rows.
    #[tokio::test]
    #[expect(clippy::too_many_lines)] // large table-driven migration fixture
    async fn retired_stages_are_merged_into_verification() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        // Catalog front entry 24 (baseline core schema) creates workspaces,
        // tickets, jobs, agents, ticket_chronicle and workspace_contexts.
        run_catalog(&conn, TargetDb::Core, &MIGRATIONS[..1])
            .await
            .expect("baseline catalog");
        let now = crate::db::now();
        conn.execute(
            "INSERT INTO workspaces (name, path, created_at, updated_at) VALUES ('ws', '/ws', ?1, ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        for (id, phase) in [("T1", "in_review"), ("T2", "in_qa"), ("T3", "backlog")] {
            conn.execute(
                "INSERT INTO tickets \
                 (id, title, description, phase, workspace_name, created_at, updated_at) \
                 VALUES (?1, ?1, '', ?2, 'ws', ?3, ?3)",
                params![id, phase, now.clone()],
            )
            .await
            .unwrap();
        }
        // T1 holds BOTH retired job rows — the collision a rename would hit.
        for (job, kind, ticket) in [
            ("J1", "in_review", Some("T1")),
            ("J2", "in_qa", Some("T1")),
            ("J3", "in_qa", Some("T2")),
            ("J4", "analysis", Some("T3")),
        ] {
            conn.execute(
                "INSERT INTO jobs \
                 (id, kind, role, workspace_name, task, user_name, channel, retry_count, status, \
                  created_at, updated_at, ticket_id) \
                 VALUES (?1, ?2, 'qa', 'ws', '', 'bob', '', 0, 'launched', ?3, ?3, ?4)",
                params![job, kind, now.clone(), ticket],
            )
            .await
            .unwrap();
        }
        conn.execute(
            "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
             VALUES ('J1', 'A1', 'verifier', 0, 'done', 'task')",
            (),
        )
        .await
        .unwrap();
        for (ticket, source, target) in [
            ("T1", "in_diagnostics", "in_review"),
            ("T2", "in_qa", "in_sanitation"),
            ("T3", "backlog", "queued"),
            // The retired→retired hop the merge turned into one stage: it must
            // be dropped rather than rewritten into a `verification →
            // verification` self-transition.
            ("T1", "in_review", "in_qa"),
        ] {
            conn.execute(
                "INSERT INTO ticket_chronicle \
                 (ticket_id, workspace_name, source_phase, target_phase, at) VALUES (?1, 'ws', ?2, ?3, ?4)",
                params![ticket, source, target, now.clone()],
            )
            .await
            .unwrap();
        }
        let arrow_lifecycle =
            "lifecycle — `Backlog → Analysis → InReview → InQa → InSanitation → Done`";
        let phase_list =
            "Each phase (`in_development`, `in_diagnostics`, `in_review`, `in_qa`) owns a job";
        let spaced_lifecycle =
            "the pipeline runs development → diagnostics → review → QA → sanitation";
        for (role, content) in [
            (None, arrow_lifecycle),
            (Some("engineer"), phase_list),
            (Some("qa"), "the board keeps IN_REVIEW and IN_QA columns"),
            (Some("manager"), spaced_lifecycle),
            (Some("coder"), "no retired stage is named here"),
        ] {
            conn.execute(
                "INSERT INTO workspace_contexts (workspace_name, role, content, created_at) \
                 VALUES ('ws', ?1, ?2, ?3)",
                params![role, content, now.clone()],
            )
            .await
            .unwrap();
        }

        let merge_entries: Vec<Migration> = MIGRATIONS
            .iter()
            .filter(|m| m.id == "44" || m.id == "45")
            .copied()
            .collect();
        run_catalog(&conn, TargetDb::Core, &merge_entries)
            .await
            .expect("stage-merge tail");

        let phases: Vec<(String, String)> = conn
            .query("SELECT id, phase FROM tickets ORDER BY id", ())
            .await
            .unwrap()
            .into_iter()
            .map(|row| (row.get::<String>(0).unwrap(), row.get::<String>(1).unwrap()))
            .collect();
        assert_eq!(
            phases,
            vec![
                ("T1".to_string(), "verification".to_string()),
                ("T2".to_string(), "verification".to_string()),
                ("T3".to_string(), "backlog".to_string()),
            ],
            "the retired phases become verification; every other phase survives"
        );

        let remaining_jobs: Vec<String> = conn
            .query("SELECT id FROM jobs ORDER BY id", ())
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.get::<String>(0).unwrap())
            .collect();
        assert_eq!(
            remaining_jobs,
            vec!["J4".to_string()],
            "the retired job rows must be deleted, never renamed"
        );
        let orphan_rosters: i64 = conn.query("SELECT COUNT(*) FROM agents", ()).await.unwrap()[0]
            .get::<i64>(0)
            .unwrap();
        assert_eq!(
            orphan_rosters, 0,
            "the deleted jobs must cascade their roster away"
        );

        let chronicle: Vec<(String, String, String)> = conn
            .query(
                "SELECT ticket_id, source_phase, target_phase FROM ticket_chronicle ORDER BY ticket_id",
                (),
            )
            .await
            .unwrap()
            .into_iter()
            .map(|row| {
                (
                    row.get::<String>(0).unwrap(),
                    row.get::<String>(1).unwrap(),
                    row.get::<String>(2).unwrap(),
                )
            })
            .collect();
        assert_eq!(
            chronicle,
            vec![
                (
                    "T1".to_string(),
                    "in_diagnostics".to_string(),
                    "verification".to_string()
                ),
                (
                    "T2".to_string(),
                    "verification".to_string(),
                    "in_sanitation".to_string()
                ),
                (
                    "T3".to_string(),
                    "backlog".to_string(),
                    "queued".to_string()
                ),
            ],
            "both chronicle sides must be rewritten, the retired→retired hop \
             dropped, and unretired edges kept"
        );

        let contexts = workspace_context_rows(&conn).await;
        assert!(
            !contexts.contains_key(""),
            "the arrow lifecycle list names a retired stage — the row must go, not be patched"
        );
        assert!(
            !contexts.contains_key("engineer"),
            "the code-span slug list names a retired stage"
        );
        assert!(
            !contexts.contains_key("qa"),
            "an upper-cased retired stage name still instructs the old arrangement"
        );
        assert!(
            !contexts.contains_key("manager"),
            "the spaced lifecycle prose names a retired stage"
        );
        assert_eq!(
            contexts.get("coder"),
            Some(&"no retired stage is named here".to_string()),
            "a description naming no retired stage survives verbatim"
        );

        // The ledger records `44`/`45` after the first application, so re-run
        // both bodies directly to pin that they match no rows a second time.
        conn.execute_batch(REWRITE_RETIRED_STAGES_TO_VERIFICATION)
            .await
            .expect("re-run entry 44 body");
        run_drop_retired_lifecycle_descriptions(&conn)
            .await
            .expect("re-run entry 45 body");
        assert_eq!(workspace_context_rows(&conn).await, contexts);
    }

    /// Behavioral pin for the phase-retirement tail: entry `48` rewrites the
    /// retired `in_diagnostics` stage to `verification` on the ticket and on both
    /// chronicle sides, DELETES the retired job row — renaming it could collide
    /// on the unique `jobs(kind, ticket_id)` index and abort the migration —
    /// taking its cascaded `agents` roster with it, and DELETES the
    /// `in_diagnostics → verification` hop, which the merge turned into one
    /// stage. Entry `49` deletes every stored workspace context that claims the
    /// project commands already ran before the round and leaves a same-round
    /// description verbatim. Idempotent: re-running both bodies matches no rows.
    #[tokio::test]
    #[expect(clippy::too_many_lines)] // large table-driven migration fixture
    async fn retired_diagnostics_phase_is_merged_into_verification() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        // Catalog front entry 24 (baseline core schema) creates workspaces,
        // tickets, jobs, agents, ticket_chronicle and workspace_contexts.
        run_catalog(&conn, TargetDb::Core, &MIGRATIONS[..1])
            .await
            .expect("baseline catalog");
        let now = crate::db::now();
        conn.execute(
            "INSERT INTO workspaces (name, path, created_at, updated_at) VALUES ('ws', '/ws', ?1, ?1)",
            params![now.clone()],
        )
        .await
        .unwrap();
        for (id, phase) in [
            ("T1", "in_diagnostics"),
            ("T2", "verification"),
            ("T3", "backlog"),
        ] {
            conn.execute(
                "INSERT INTO tickets \
                 (id, title, description, phase, workspace_name, created_at, updated_at) \
                 VALUES (?1, ?1, '', ?2, 'ws', ?3, ?3)",
                params![id, phase, now.clone()],
            )
            .await
            .unwrap();
        }
        // J1's ticket moves to `verification`; a rename of J1 would collide with
        // the already-`verification` J2, which is why the retired row is deleted.
        for (job, kind, ticket) in [
            ("J1", "in_diagnostics", Some("T1")),
            ("J2", "verification", Some("T2")),
            ("J3", "analysis", Some("T3")),
        ] {
            conn.execute(
                "INSERT INTO jobs \
                 (id, kind, role, workspace_name, task, user_name, channel, retry_count, status, \
                  created_at, updated_at, ticket_id) \
                 VALUES (?1, ?2, 'verifier', 'ws', '', 'bob', '', 0, 'launched', ?3, ?3, ?4)",
                params![job, kind, now.clone(), ticket],
            )
            .await
            .unwrap();
        }
        conn.execute(
            "INSERT INTO agents (job_id, agent_id, kind, idx, status, task) \
             VALUES ('J1', 'A1', 'verifier', 0, 'done', 'task')",
            (),
        )
        .await
        .unwrap();
        for (ticket, source, target) in [
            ("T1", "in_development", "in_diagnostics"),
            ("T2", "backlog", "queued"),
            ("T3", "in_diagnostics", "in_development"),
            // The merge's own hop: `in_diagnostics` and `verification` are one
            // stage now, so it must be dropped rather than rewritten into a
            // `verification → verification` self-transition.
            ("T1", "in_diagnostics", "verification"),
        ] {
            conn.execute(
                "INSERT INTO ticket_chronicle \
                 (ticket_id, workspace_name, source_phase, target_phase, at) VALUES (?1, 'ws', ?2, ?3, ?4)",
                params![ticket, source, target, now.clone()],
            )
            .await
            .unwrap();
        }
        let already_ran = "Diagnostics (format, lint, type-check, build, unit test) already ran \
                           and are reported in the ticket's diagnostics comment; do not repeat them";
        let diagnostics_phase = "this is exactly what the Diagnostics phase ran before this round";
        let same_round = "the project commands run in the same round as the check";
        let running_service = "the local service is already running on port 3000";
        for (role, content) in [
            ("engineer", already_ran),
            ("qa", diagnostics_phase),
            ("coder", same_round),
            ("manager", running_service),
        ] {
            conn.execute(
                "INSERT INTO workspace_contexts (workspace_name, role, content, created_at) \
                 VALUES ('ws', ?1, ?2, ?3)",
                params![role, content, now.clone()],
            )
            .await
            .unwrap();
        }

        let merge_entries: Vec<Migration> = MIGRATIONS
            .iter()
            .filter(|m| m.id == "48" || m.id == "49")
            .copied()
            .collect();
        run_catalog(&conn, TargetDb::Core, &merge_entries)
            .await
            .expect("phase-retirement tail");

        let phases: Vec<(String, String)> = conn
            .query("SELECT id, phase FROM tickets ORDER BY id", ())
            .await
            .unwrap()
            .into_iter()
            .map(|row| (row.get::<String>(0).unwrap(), row.get::<String>(1).unwrap()))
            .collect();
        assert_eq!(
            phases,
            vec![
                ("T1".to_string(), "verification".to_string()),
                ("T2".to_string(), "verification".to_string()),
                ("T3".to_string(), "backlog".to_string()),
            ],
            "the retired phase becomes verification; every other phase survives"
        );

        let remaining_jobs: Vec<String> = conn
            .query("SELECT id FROM jobs ORDER BY id", ())
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.get::<String>(0).unwrap())
            .collect();
        assert_eq!(
            remaining_jobs,
            vec!["J2".to_string(), "J3".to_string()],
            "the retired job row must be deleted, never renamed"
        );
        let orphan_rosters: i64 = conn.query("SELECT COUNT(*) FROM agents", ()).await.unwrap()[0]
            .get::<i64>(0)
            .unwrap();
        assert_eq!(
            orphan_rosters, 0,
            "the deleted job must cascade its roster away"
        );

        let chronicle: Vec<(String, String, String)> = conn
            .query(
                "SELECT ticket_id, source_phase, target_phase FROM ticket_chronicle ORDER BY ticket_id",
                (),
            )
            .await
            .unwrap()
            .into_iter()
            .map(|row| {
                (
                    row.get::<String>(0).unwrap(),
                    row.get::<String>(1).unwrap(),
                    row.get::<String>(2).unwrap(),
                )
            })
            .collect();
        assert_eq!(
            chronicle,
            vec![
                (
                    "T1".to_string(),
                    "in_development".to_string(),
                    "verification".to_string()
                ),
                (
                    "T2".to_string(),
                    "backlog".to_string(),
                    "queued".to_string()
                ),
                (
                    "T3".to_string(),
                    "verification".to_string(),
                    "in_development".to_string()
                ),
            ],
            "both chronicle sides must be rewritten, the merge hop dropped, and \
             unretired edges kept"
        );

        let contexts = workspace_context_rows(&conn).await;
        assert!(
            !contexts.contains_key("engineer"),
            "a claim that the commands already ran describes the retired arrangement"
        );
        assert!(
            !contexts.contains_key("qa"),
            "naming the diagnostics phase as a prior stage describes the retired arrangement"
        );
        assert_eq!(
            contexts.get("coder"),
            Some(&same_round.to_string()),
            "a same-round description must survive verbatim"
        );
        assert_eq!(
            contexts.get("manager"),
            Some(&running_service.to_string()),
            "an unrelated already-running service must survive verbatim"
        );

        // The ledger records `48`/`49` after the first application, so re-run
        // both bodies directly to pin that they match no rows a second time.
        conn.execute_batch(RETIRE_IN_DIAGNOSTICS_PHASE)
            .await
            .expect("re-run entry 48 body");
        run_drop_retired_commands_claim_descriptions(&conn)
            .await
            .expect("re-run entry 49 body");
        let phases_after: Vec<(String, String)> = conn
            .query("SELECT id, phase FROM tickets ORDER BY id", ())
            .await
            .unwrap()
            .into_iter()
            .map(|row| (row.get::<String>(0).unwrap(), row.get::<String>(1).unwrap()))
            .collect();
        assert_eq!(phases_after, phases, "a re-run must not move another phase");
        let jobs_after: Vec<String> = conn
            .query("SELECT id FROM jobs ORDER BY id", ())
            .await
            .unwrap()
            .into_iter()
            .map(|row| row.get::<String>(0).unwrap())
            .collect();
        assert_eq!(
            jobs_after, remaining_jobs,
            "a re-run must not delete another job"
        );
        let chronicle_after: Vec<(String, String, String)> = conn
            .query(
                "SELECT ticket_id, source_phase, target_phase FROM ticket_chronicle ORDER BY ticket_id",
                (),
            )
            .await
            .unwrap()
            .into_iter()
            .map(|row| {
                (
                    row.get::<String>(0).unwrap(),
                    row.get::<String>(1).unwrap(),
                    row.get::<String>(2).unwrap(),
                )
            })
            .collect();
        assert_eq!(
            chronicle_after, chronicle,
            "a re-run must not rewrite another hop"
        );
        assert_eq!(workspace_context_rows(&conn).await, contexts);
    }

    /// The stored `config_kv.value` for the `image_gen_models` owner-editable
    /// picker list, for the delta-50 behavioral pin.
    async fn image_gen_models_list(conn: &Connection) -> String {
        conn.query(
            "SELECT value FROM config_kv WHERE key = ?1",
            params![IMAGE_GEN_MODELS_KV_KEY],
        )
        .await
        .expect("read config_kv")
        .into_iter()
        .next()
        .expect("image_gen_models row")
        .get::<String>(0)
        .expect("value")
    }

    /// `name → users.image_gen_model` (including NULL) for the delta-50
    /// behavioral pin.
    async fn users_image_gen_models(
        conn: &Connection,
    ) -> std::collections::BTreeMap<String, Option<String>> {
        conn.query("SELECT name, image_gen_model FROM users", ())
            .await
            .expect("read users")
            .into_iter()
            .map(|row| {
                (
                    row.get::<String>(0).expect("user name"),
                    row.get::<Option<String>>(1).expect("image_gen_model"),
                )
            })
            .collect()
    }

    /// Behavioral pin for catalog entry `50`: the retired image-generation
    /// default is replaced by its successor on both user-visible surfaces — the
    /// owner-edited `image_gen_models` picker list and every account's
    /// `users.image_gen_model` pick — while a near-miss id, the owner's other
    /// picks and a NULL pick survive verbatim, and blank lines/repeats/order are
    /// preserved. Matching is on the trimmed line, so the whitespace-padded pick
    /// becomes the bare successor. A successor that was already in the list
    /// stays beside the replacement — duplicates are the owner's business. The
    /// `schema_migrations` ledger makes it one-shot: a later deliberate re-add
    /// of the retired id is kept.
    #[tokio::test]
    async fn retire_old_image_gen_model_rewrites_both_picker_surfaces_once() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        // Apply the catalog through entry `49` so only `50` is left to run.
        let entry_50 = MIGRATIONS
            .iter()
            .position(|m| m.id == "50")
            .expect("entry 50 in the catalog");
        run_catalog(&conn, TargetDb::Core, &MIGRATIONS[..entry_50])
            .await
            .expect("catalog through 49");

        // A realistic-messy owner list: blank lines, a repeat, a near-miss id, a
        // whitespace-padded pick and a trailing newline.
        conn.execute(
            "INSERT INTO config_kv (key, value) VALUES (?1, ?2)",
            params![
                IMAGE_GEN_MODELS_KV_KEY,
                "bogus/keep-me\n\ngoogle/gemini-3.1-flash-image\n\
                 google/gemini-3.1-flash-image-preview\n  google/gemini-3.1-flash-image  \n\
                 microsoft/mai-image-2.6\ngoogle/gemini-nano-banana-2.1\ngoogle/gemini-3.1-flash-image\n"
            ],
        )
        .await
        .unwrap();
        for (name, pick) in [
            ("alice", Some(RETIRED_IMAGE_GEN_MODEL)),
            ("bob", Some("microsoft/mai-image-2.6")),
            ("carol", None),
        ] {
            conn.execute(
                "INSERT INTO users (name, image_gen_model) VALUES (?1, ?2)",
                params![name, pick],
            )
            .await
            .unwrap();
        }

        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("full catalog");

        assert_eq!(
            image_gen_models_list(&conn).await,
            "bogus/keep-me\n\ngoogle/gemini-nano-banana-2.1\n\
             google/gemini-3.1-flash-image-preview\ngoogle/gemini-nano-banana-2.1\n\
             microsoft/mai-image-2.6\ngoogle/gemini-nano-banana-2.1\ngoogle/gemini-nano-banana-2.1\n",
            "only the exact retired id may be replaced, on the trimmed line; the \
             near-miss, the order, the blanks and the repeats must survive — and \
             an already-present successor may end up duplicated (no dedupe)"
        );
        let picks = users_image_gen_models(&conn).await;
        assert_eq!(
            picks.get("alice"),
            Some(&Some(SUCCESSOR_IMAGE_GEN_MODEL.to_string())),
            "the account holding the retired id must be rewritten"
        );
        assert_eq!(
            picks.get("bob"),
            Some(&Some("microsoft/mai-image-2.6".to_string())),
            "another model pick must be untouched"
        );
        assert_eq!(
            picks.get("carol"),
            Some(&None),
            "a NULL pick must be untouched"
        );

        // The ledger recorded `50`, so a deliberate re-add by the owner must
        // survive the next boot's catalog run.
        conn.execute(
            "UPDATE config_kv SET value = ?1 WHERE key = ?2",
            params![
                format!("{RETIRED_IMAGE_GEN_MODEL}\n{SUCCESSOR_IMAGE_GEN_MODEL}"),
                IMAGE_GEN_MODELS_KV_KEY
            ],
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE users SET image_gen_model = ?1 WHERE name = 'alice'",
            params![RETIRED_IMAGE_GEN_MODEL],
        )
        .await
        .unwrap();
        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("second catalog run");
        assert!(
            image_gen_models_list(&conn)
                .await
                .starts_with(RETIRED_IMAGE_GEN_MODEL),
            "a later deliberate re-add of the retired id must be kept: one-shot"
        );
        assert_eq!(
            users_image_gen_models(&conn).await.get("alice"),
            Some(&Some(RETIRED_IMAGE_GEN_MODEL.to_string())),
            "a later deliberate re-add of the retired id must be kept: one-shot"
        );
    }
}
