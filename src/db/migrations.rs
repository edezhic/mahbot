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
//! SQL text, and per-body idempotency logic (e.g. the [`add_column_if_missing`]
//! existence probe, needed because Turso has no `ADD COLUMN IF NOT EXISTS`) is
//! part of a migration body, never a conditional in the runner.
//!
//! # Current-shape baseline
//!
//! Entry `24` builds the whole core schema and entry `26` the whole logs
//! schema — every domain table, column and index in its current shape, every
//! statement `IF NOT EXISTS`. A fresh install converges in one pass, and on an
//! existing store the two baselines are strict no-ops: a baseline cannot upfill
//! a file that already exists.
//!
//! # What stays in the catalog
//!
//! An entry stays only if it changes the end state of a brand-new database, or
//! if it can still be pending on a database that has run a 0.10.x build. The
//! two halves select different entries, and confusing them is how a column,
//! table or default disappears from an installation that has not yet reached
//! the newest release. The catalog holds nine entries:
//!
//! - `24` / `26` — the core and logs baselines. They always stay: they are what
//!   a fresh install is built from, and the upgrade half alone would retire
//!   them.
//! - `69` — writes `audio_transcription_use_local = "false"` wherever the row is
//!   absent, which on a supported store is only a brand-new database, since
//!   every 0.10.x store recorded the entry (see
//!   [`seed_audio_transcription_disabled`]). It is the only automatic writer of
//!   that row, and absence of the row means "enabled", so retiring it would turn
//!   on a multi-gigabyte recogniser download for a fresh install.
//! - `79` — the `guest_reports` table, which arrived in 0.10.4 and is therefore
//!   still pending on a store created by 0.10.0–0.10.3 (see
//!   [`GUEST_REPORTS_TABLE`]).
//! - `80` — the `session_metadata.provider` / `provider_model` columns, which
//!   arrived in 0.10.5 (see [`add_session_metadata_provider`]).
//! - `81` — deletes a stored provider priority list (see
//!   [`DROP_DEEPSEEK_V41_FLASH_1017_ROUTING`]). It only touches data, but it is
//!   still pending on a store created before 0.10.5, so it stays: "only touches
//!   data" retires an entry the upgrade half has already retired, it is never a
//!   reason by itself.
//! - `82` — overwrites the image/video model lists with the shipped ones (see
//!   [`run_set_media_model_lists`]). It deliberately discards a list the owner may
//!   have edited, because otherwise the new defaults never reach such an install.
//!   Its body reads the same constants as the fresh-install seed, so the two
//!   cannot drift.
//! - `83` — re-runs the same body as `82` because `82` never shipped in a release,
//!   so a store that already recorded it would otherwise keep the old list.
//! - `84` — the `jobs.chain` / `session_metadata.chain` columns a `send` chain rides
//!   on (see [`run_add_chain_columns`]). They arrive with this release, so every
//!   existing store is still pending on them.
//! - `85` — the `jobs.extract` / `session_metadata.extract` columns a `send` schema
//!   rides on (see [`run_add_extract_columns`]), arriving with the same release.
//!
//! # The id → release record, and the retired ids
//!
//! Nothing in the source records which release added which id, so it is written
//! here — the next prune depends on it:
//!
//! - `24`–`78` — recorded by 0.10.0 (0.10.1–0.10.3 shipped the same catalog).
//! - `79` — added by 0.10.4.
//! - `80`, `81` — added by 0.10.5.
//! - `82`, `83` — added after the 0.10.6 tag, so no shipped build records either.
//! - `84` — the `send` chain's columns; no shipped build records it.
//! - `85` — the `send` schema's columns; no shipped build records it.
//! - Next id: `86`.
//!
//! Retired, and never to be reused:
//!
//! - before the collapse to the current-shape baseline: the 0.4.2→0.5.x chain
//!   (`1`–`23`) and its named entries — `001_drop_ticket_pipeline_reservation`,
//!   `001_session_token_length`, `002_drop_ticket_assigned_to`,
//!   `002_session_message_count`, `003_drop_ticket_stage_jobs_round`,
//!   `003_reset_nonterminal_tickets`, `004_drop_tickets_review_base_count`,
//!   `004_reset_implementation_jobs`, `005_drop_ticket_stage_jobs_phase`,
//!   `006_rename_ticket_stage_jobs_to_ticket_jobs`, `007_jobs_paused_frozen`,
//!   `consolidate_002_drop_ticket_jobs_stage`, `consolidate_003_jobs_ticket_id`,
//!   `consolidate_004_discard_old_ticket_jobs`,
//!   `consolidate_005_drop_ticket_jobs`,
//!   `consolidate_006_drop_jobs_paused_frozen`,
//!   `consolidate_007_jobs_phase_ticket_index`,
//!   `consolidate_008_create_ticket_chronicle` — plus the one-time consolidation
//!   import (`consolidate_001_import_domain_stores`);
//! - by that collapse: `25`, `27`, `28`, `29`, `30`, `31`, `32`, `33`, `34`, `35`,
//!   `36`, `37`, `38`, `39`, `40`, `41`, `42`, `43`, `44`, `45`, `46`, `47`, `48`,
//!   `49`, `50`, `51`, `52`, `53`, `54`, `55`, `56`, `57`, `58`, `59`, `60`, `61`,
//!   `62`, `63`, `64`, `65`, `66`, `67`, `68`, `70`, `71`, `72`, `73`, `74`, `75`,
//!   `76`, `77`, `78`.
//!
//! None of them is still needed on a supported store: a store a 0.10.0 release
//! created is built by the baselines, and a store that ran the chain recorded
//! whichever parts it ran — the parts it skipped were never needed, since it
//! reached the current shape without them.
//!
//! Retired ids are retired, not recycled: the runner skips a recorded id, so a
//! reused id would mean different things on different installations — skipped
//! where its body already ran, executed where it did not, and always executed on
//! a brand-new database, whose ledger is empty. A new id is no better: it would be
//! pending on every existing database, costing each install a full integrity scan
//! of its stores on the next boot.
//!
//! # Support floor
//!
//! A store is supported when it has run a 0.10.x build at least once — not when
//! it was created at 0.10 or later, because an older install updates straight
//! forward and skips the window. The consequences are accepted, stated rather
//! than implied:
//!
//! - A store created before 0.10.0 is not supported. It still opens — its ledger
//!   is non-empty, so the boot gate accepts it — and then fails at the first
//!   query needing a column it never received. It is worse than one column: such
//!   a store also lacks the logs store's `ticket` column and keeps the `alarms`
//!   table, because the rename never runs, so logging and the whole automations
//!   surface break. No detection gate is wanted: the boot gate deliberately asks
//!   only whether the ledger is non-empty.
//! - An entry whose body ran but whose ledger row was lost to a crash is never
//!   re-run, so such a store keeps whatever partial state it had.
//! - A downgrade from this catalog back to a 0.10.x binary is not supported: the
//!   retired entries would run again, including the destructive data rewrites.
//!
//! # Coverage, and the frozen fixture
//!
//! The v0.10.0 fixture in the tests builds a database from that release's own
//! baseline text with its ids recorded, and proves the survivors take it to a
//! fresh install's schema and catalog-written rows. It replaces the shape
//! coverage the per-entry tests gave. The retired entries that rewrote or moved
//! stored data are gone with no test standing in for them, which is accepted:
//! they can never run again on a supported store, and the fixture comparison
//! pins the end state a store must reach.
//!
//! The fixture is a copy of the two baseline constants at tag `v0.10.0`. It must
//! be regenerated whenever the support floor moves; that is the standing cost of
//! this shape, and the id → release record above is what makes it possible.
//!
//! # Failure semantics
//!
//! A catalog/logic error is a hard boot failure — it is never a repair finding
//! and never a reason to replace the store.

use anyhow::Context;
use futures_util::future::BoxFuture;
use std::collections::HashSet;

use crate::db::{Connection, params};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetDb {
    Core,
    Logs,
}

/// Runs without a surrounding transaction: the body owns its transactional/FK semantics.
type RustFn = for<'a> fn(&'a Connection) -> BoxFuture<'a, anyhow::Result<()>>;

#[derive(Debug, Clone, Copy)]
pub(crate) enum MigrationBody {
    Sql(&'static str),
    Rust(RustFn),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Migration {
    pub(crate) id: &'static str,
    pub(crate) target: TargetDb,
    pub(crate) body: MigrationBody,
}

/// Entry `24`: the full current-shape core schema — every domain table in its
/// final shape and every core index a product query's plan reaches. Every
/// statement is `IF NOT EXISTS`, so on an existing store the batch is a no-op.
///
/// `jobs` is created WITH `caller_agent_id`, `mode`, `exec_dir` and
/// `custom_launch` as its trailing columns, and `session_metadata` WITH
/// `provider` / `provider_model`, so `idx_jobs_phase_ticket` (which references
/// `jobs.ticket_id`) is safe to create in the same batch.
///
/// # The kept-index audit
///
/// Every production statement against these tables was planned through the debug
/// surface's `EXPLAIN QUERY PLAN`. The declared indexes are the ones a predicate
/// or an ordering was planned onto:
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
/// - `chat_history` — the conversation read and the removal that holds its cap
///   both plan on the composite `idx_chat_history_user_ws_id`.
/// - `tickets` — the board listings plan on `idx_tickets_board_active`, which
///   also supplies their `priority ASC, created_at DESC` ordering, and the
///   workspace-scoped claim candidates plan on `idx_tickets_workspace_phase`;
///   the title search goes through the FTS index `idx_tickets_title_fts`.
/// - `ticket_chronicle` — the trim, the cursor seed and the drain plan on
///   `idx_ticket_chronicle_ws_id`; `idx_ticket_chronicle_dedup` is unique.
/// - `automations` — the due sweep and the per-target listings plan on
///   `idx_automations_due`.
/// - `workspace_contexts` — the `role IS NULL` read plans on
///   `workspace_contexts_null_role`, which is unique.
/// - The remaining core tables (`users`, `workspaces`, `session_metadata`,
///   `session_personal_changes`, `config_kv`, `config_model_routing`,
///   `editor_tabs`, `user_channels`, `research_jobs`, `ticket_counters`,
///   `custom_roles`, `telegram_intake`, `telegram_bursts`) carry no product
///   index beyond their primary key's own.
///
/// Four indexes are deliberately not declared, because no statement reaches
/// them: `idx_jobs_caller_agent` (the planner does not derive the partial
/// index's `caller_agent_id IS NOT NULL` from an equality), `idx_jobs_updated_at`
/// (`purge_stale_jobs` plans on `idx_jobs_kind_status (kind=?)`, and `updated_at`
/// is otherwise only a SET target), `idx_chat_history_user` (a strict prefix of
/// the composite above) and `idx_chat_history_workspace` (no statement filters
/// `workspace` alone).
///
/// An index that exists to enforce a guarantee is kept whatever the plans say:
/// `idx_jobs_phase_ticket` and `idx_ticket_chronicle_dedup` are enforced on write
/// and named by no plan. The engine's own indexes — the constraint autoindexes,
/// the FTS backing objects and the change-capture bookkeeping — are not the
/// product's to drop.
///
/// The ticket-title FTS index is declared in the engine's current on-disk
/// layout. No supported store carries the older segment layout: Turso 0.8.0's
/// FTS rewrite added two document-identity fast fields to every segment without
/// bumping the storage format number, so an index written by an earlier engine
/// fails the moment a segment carrying it is read.
///
/// The personal-workspace change baseline has a table of its own
/// (`session_personal_changes`, a child of the session's `session_metadata`
/// row): the engine rewrites a row whole on any update, so a baseline held on
/// the session's own row made every small bookkeeping update rewrite megabytes.
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
    priority         INTEGER NOT NULL DEFAULT 1,
    label            TEXT NOT NULL DEFAULT 'feat',
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
    context_block_len INTEGER,
    user_activity  TEXT,
    image_count    INTEGER,
    provider       TEXT,
    provider_model TEXT,
    chain          TEXT,
    extract        TEXT
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
    custom_launch  TEXT,
    chain          TEXT,
    extract        TEXT
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
    granted_tools       TEXT,
    memory_inventory_day TEXT
);
CREATE TABLE IF NOT EXISTS user_channels (
    user_name   TEXT NOT NULL REFERENCES users(name),
    channel     TEXT NOT NULL,
    identifier  TEXT NOT NULL,
    reply_target TEXT,
    UNIQUE(channel, identifier)
);
CREATE TABLE IF NOT EXISTS guest_reports (
    user_name TEXT NOT NULL,
    day       TEXT NOT NULL,
    digest    TEXT NOT NULL,
    PRIMARY KEY (user_name, day, digest)
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
-- ── Ticket chronicle / automations ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS ticket_chronicle (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id      TEXT NOT NULL,
    workspace_name TEXT NOT NULL,
    source_phase   TEXT NOT NULL,
    target_phase   TEXT NOT NULL,
    at             TEXT NOT NULL,
    actor          TEXT
);
CREATE TABLE IF NOT EXISTS automations (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    user_name           TEXT NOT NULL,
    kind                TEXT NOT NULL,
    text                TEXT NOT NULL,
    fire_at             TEXT,
    interval_seconds    INTEGER,
    next_fire_at        TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
    created_at          TEXT NOT NULL,
    trigger             TEXT,
    role                TEXT,
    identifier          TEXT,
    workspace           TEXT,
    retry_at            TEXT,
    unfinished_attempts INTEGER NOT NULL DEFAULT 0
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
-- The audit of what is kept and what is deliberately absent is this
-- constant's doc comment.
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
CREATE INDEX IF NOT EXISTS idx_automations_due ON automations(status, next_fire_at);
CREATE INDEX IF NOT EXISTS idx_tickets_workspace_phase ON tickets (workspace_name, phase, is_archived, priority ASC, created_at DESC);";

/// Entry `26`: the full current-shape logs schema — `logs`, `tool_calls`,
/// `llm_requests`, `llm_failures` and `grep_telemetry` with only the indexes a
/// product query's plan reaches. Every statement is `IF NOT EXISTS`, so on an
/// existing logs store the batch is a no-op.
///
/// # The kept-index audit
///
/// The kept indexes are the ones a predicate or an ordering was planned onto:
///
/// - `logs` — the Issues view's count and list (`level IN (…)`) and the INFO
///   retention (`level = 'INFO' AND timestamp < ?`) plan on `idx_logs_level`;
///   `timestamp` loses to it for the retention range. The listing's search
///   predicate is `LIKE '%…%'`, and a leading wildcard is unplannable, so the
///   scope columns gain none; `target`/`since` are never set and
///   `agent_id`/`agent_role` appear in no predicate.
/// - `tool_calls` — the two retention rules cut on `recorded_at < ?`
///   (`idx_tool_calls_recorded_at`); `idx_tool_calls_error_message` serves the
///   ad-hoc debug surface. `agent_id`/`role`/`tool_name`/`workspace` appear in
///   no predicate or ordering.
/// - `llm_requests` — the retention delete plans on
///   `idx_llm_requests_recorded_at`; `agent_id`/`model`/`purpose` are written
///   and never read.
/// - `llm_failures` — no production statement reads this table; only the
///   recorder writes it and retention keeps it, so it carries no index.
/// - `grep_telemetry` — the retention delete plans on
///   `idx_grep_telemetry_recorded_at`; `served`/`reason`/`command` are written
///   and never read.
///
/// A filterless `COUNT(*)` (the All Logs total, `mahbot debug`'s schema dump)
/// plans onto whichever index it happens to pick as a covering scan — it is
/// served by any index or by the table itself, so that pick is not a dependency.
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
    workspace   TEXT NOT NULL DEFAULT '',
    ticket      TEXT NOT NULL DEFAULT ''
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
-- Only the indexes a product query's plan reaches; the audit of what is
-- kept and what is deliberately absent is this constant's doc comment.
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

/// Entry `81`: drop the stored provider priority list of
/// `deepseek/deepseek-v4.1-flash-1017`, which now has a built-in list of its own
/// ([`crate::config::default_model_routing`]) — the owner's earlier value is
/// discarded by design, and the field is his to set again. Matched on the exact
/// model id, so the other DeepSeek identifiers keep their rows.
const DROP_DEEPSEEK_V41_FLASH_1017_ROUTING: &str =
    "DELETE FROM config_model_routing WHERE model = 'deepseek/deepseek-v4.1-flash-1017';";

/// Entry `79` (existing installs): the `guest_reports` table — the one-day ledger behind a
/// guest assistant's `report_gap` tool's daily cap and duplicate check. It holds no report
/// text, only the account, the local day and a digest of what was filed, and every write
/// prunes the days before. The baseline (`24`) already declares it, so on a fresh install
/// this is an `IF NOT EXISTS` no-op.
const GUEST_REPORTS_TABLE: &str = "CREATE TABLE IF NOT EXISTS guest_reports (
    user_name TEXT NOT NULL,
    day       TEXT NOT NULL,
    digest    TEXT NOT NULL,
    PRIMARY KEY (user_name, day, digest)
);";

/// The complete, strictly-linear catalog. **Order is application order.**
///
/// Nine entries, split by the module header's rule: the fresh-install half is
/// `24`, `26` and `69`; the upgrade half is `79`, `80`, `81`, `82`, `83` and
/// `84` — the entries that can still be pending on a store that has run a
/// 0.10.x build. The header also carries the id → release record and the
/// retired ids.
///
/// - `24` builds the full core schema; on an existing store every statement is
///   `IF NOT EXISTS`, so it is a strict no-op.
/// - `26` builds the full logs schema, likewise a strict no-op on an existing
///   logs store.
/// - `69` seeds the transcription-off flag a brand-new core store needs (see
///   [`seed_audio_transcription_disabled`]).
/// - `79` creates `guest_reports` on an existing core store (see
///   [`GUEST_REPORTS_TABLE`]).
/// - `80` adds the two conversation-pin columns (see
///   [`add_session_metadata_provider`]).
/// - `81` deletes one stored routing row (see
///   [`DROP_DEEPSEEK_V41_FLASH_1017_ROUTING`]).
/// - `82` overwrites the image/video model lists (see
///   [`run_set_media_model_lists`]).
/// - `83` re-runs the same body, because `82` never shipped in a release (see
///   [`run_set_media_model_lists`]).
/// - `84` adds the nullable `jobs.chain` and `session_metadata.chain` columns a `send` chain
///   rides on (see [`run_add_chain_columns`]).
pub(crate) const MIGRATIONS: &[Migration] = &[
    Migration {
        id: "24",
        target: TargetDb::Core,
        body: MigrationBody::Sql(BASELINE_CORE_SCHEMA),
    },
    Migration {
        id: "26",
        target: TargetDb::Logs,
        body: MigrationBody::Sql(BASELINE_LOGS_SCHEMA),
    },
    Migration {
        id: "69",
        target: TargetDb::Core,
        body: MigrationBody::Rust(seed_audio_transcription_disabled),
    },
    Migration {
        id: "79",
        target: TargetDb::Core,
        body: MigrationBody::Sql(GUEST_REPORTS_TABLE),
    },
    Migration {
        id: "80",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_session_metadata_provider),
    },
    Migration {
        id: "81",
        target: TargetDb::Core,
        body: MigrationBody::Sql(DROP_DEEPSEEK_V41_FLASH_1017_ROUTING),
    },
    Migration {
        id: "82",
        target: TargetDb::Core,
        body: MigrationBody::Rust(set_media_model_lists),
    },
    Migration {
        id: "83",
        target: TargetDb::Core,
        body: MigrationBody::Rust(set_media_model_lists),
    },
    Migration {
        id: "84",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_chain_columns),
    },
    Migration {
        id: "85",
        target: TargetDb::Core,
        body: MigrationBody::Rust(add_extract_columns),
    },
];

/// Apply the catalog to `conn` for one physical database. Each SQL entry's
/// schema change and tracking row commit in one transaction; a Rust entry runs
/// outside one.
pub(crate) async fn run_migrations(conn: &Connection, db: TargetDb) -> anyhow::Result<()> {
    run_catalog(conn, db, MIGRATIONS).await
}

/// The applied-id ledger, and the boot usability gate's "built by this product" marker.
pub(crate) const MIGRATIONS_TABLE: &str = "schema_migrations";

/// True when entries remain for `db`. Boot asks before writing schema DDL, so
/// an unreadable store is refused and repaired first (see `db::open_store`).
pub(crate) async fn has_pending(conn: &Connection, db: TargetDb) -> anyhow::Result<bool> {
    let applied = applied_ids(conn).await?;
    Ok(MIGRATIONS
        .iter()
        .any(|migration| migration.target == db && !applied.contains(migration.id)))
}

/// The ids already recorded in [`MIGRATIONS_TABLE`].
async fn applied_ids(conn: &Connection) -> anyhow::Result<HashSet<String>> {
    Ok(conn
        .query(&format!("SELECT id FROM {MIGRATIONS_TABLE}"), ())
        .await
        .context("Failed to read applied migrations")?
        .into_iter()
        .filter_map(|row| row.get::<String>(0).ok())
        .collect())
}

/// The migration loop, catalog-parameterized so a test can apply the baseline
/// prefix or one entry in isolation.
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

    let applied = applied_ids(conn).await?;

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
                // Not atomic with its tracking row: every Rust body is
                // idempotent, so a crash between the two re-runs it safely.
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

/// Return whether `column` exists in `table` (via `PRAGMA table_info`).
///
/// This is the idempotency probe used by the guarded Rust migration bodies:
/// Turso has no `ADD COLUMN IF NOT EXISTS`, so a body probes before altering
/// rather than relying on an `IF NOT EXISTS`.
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
/// `ALTER TABLE`. The guarded bodies call it so a re-run is a no-op.
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

fn add_chain_columns(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_chain_columns(conn))
}

/// Entry `84`: add the nullable `jobs.chain` and `session_metadata.chain` columns — the remaining
/// steps of a `send` chain, its final recipient and where it is, carried by the record a hop
/// already has (a worker round's row, a branch session) rather than a store of its own. A
/// conversation or job with no chain (every one predating the columns) behaves exactly as before.
/// Guarded by [`add_column_if_missing`], so this body is idempotent.
async fn run_add_chain_columns(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "jobs", "chain", "TEXT").await?;
    add_column_if_missing(conn, "session_metadata", "chain", "TEXT").await
}

fn add_extract_columns(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_extract_columns(conn))
}

/// Entry `85`: add the nullable `jobs.extract` and `session_metadata.extract` columns — the JSON
/// schema a `send` asked its answer to match, carried by the record the delegation already has (a
/// worker round's row, a branch session) rather than a store of its own. A job or conversation with
/// no schema (every one predating the columns) behaves exactly as before, and a build that predates
/// the columns never selects them: a schema-carrying row resumed there delivers prose. Guarded by
/// [`add_column_if_missing`], so this body is idempotent.
async fn run_add_extract_columns(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "jobs", "extract", "TEXT").await?;
    add_column_if_missing(conn, "session_metadata", "extract", "TEXT").await
}

fn add_session_metadata_provider(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_add_session_metadata_provider(conn))
}

/// Entry `80`: add the nullable `session_metadata.provider` and
/// `session_metadata.provider_model` columns — the provider slug this conversation is pinned to,
/// and the chat model it was chosen for. A model change re-selects, because the previous provider
/// may not serve the new model, so the model the pin was made for has to be stored beside it.
/// Both are nullable with no backfill: a conversation with no pin (every conversation predating
/// the columns) selects on its next run and records its pick then. Fresh installs get the columns
/// from entry `24`'s `CREATE TABLE`, so the probes make this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional like every Rust body,
/// re-runnable until recorded).
async fn run_add_session_metadata_provider(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "session_metadata", "provider", "TEXT").await?;
    add_column_if_missing(conn, "session_metadata", "provider_model", "TEXT").await
}

fn seed_audio_transcription_disabled(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_seed_audio_transcription_disabled(conn))
}

/// Entry `69`: persist the transcription-off flag wherever the row is absent, so
/// an owner who never chose a value is not treated as having local speech
/// enabled — the ~2 GB Qwen3-ASR recogniser is not loaded, and the voice stack
/// (wake word, microphone recording, transcription) stays off until the owner
/// turns it on. The row keeps its meaning afterwards: absence still means
/// enabled, which is what enabling the setting writes.
///
/// `DO NOTHING` makes this an insert-if-absent: an existing row (the user's
/// `"false"` or anything else they wrote) is never overwritten, and a row the
/// user deletes by enabling is not re-written later because the catalog's id
/// check makes the body run once.
async fn run_seed_audio_transcription_disabled(conn: &Connection) -> anyhow::Result<()> {
    conn.execute(
        "INSERT INTO config_kv (key, value) VALUES (?1, 'false') ON CONFLICT (key) DO NOTHING",
        params![crate::config::CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL],
    )
    .await?;
    Ok(())
}

/// Entries `82` and `83`: bring every installation to the shipped image/video model lists,
/// discarding lists edited by hand — that is intended, because otherwise the new defaults never
/// reach an existing install. `DO UPDATE` reaches installs whose `config_kv` has no row for either
/// key at all. The fresh-install seed writes the same two constants, so both paths end on one
/// list. Entry `83` exists because `82` never shipped in a release: a store built from the source
/// tree may already have recorded `82`, and it must still arrive at the shipped list.
async fn run_set_media_model_lists(conn: &Connection) -> anyhow::Result<()> {
    for (key, list) in [
        (
            crate::config::CONFIG_KEY_IMAGE_GEN_MODELS,
            crate::config::SHIPPED_IMAGE_GEN_MODELS,
        ),
        (
            crate::config::CONFIG_KEY_VIDEO_MODELS,
            crate::config::SHIPPED_VIDEO_MODELS,
        ),
    ] {
        conn.execute(
            "INSERT INTO config_kv (key, value) VALUES (?1, ?2) \
             ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            params![key, list],
        )
        .await?;
    }
    Ok(())
}

fn set_media_model_lists(conn: &Connection) -> BoxFuture<'_, anyhow::Result<()>> {
    Box::pin(run_set_media_model_lists(conn))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Connection;

    async fn recorded_ids(conn: &Connection) -> Vec<String> {
        conn.query("SELECT id FROM schema_migrations", ())
            .await
            .expect("read schema_migrations")
            .into_iter()
            .map(|row| row.get::<String>(0).expect("id"))
            .collect()
    }

    /// Every `config_kv` row as (key, value), sorted so comparisons are stable.
    async fn config_kv_rows(conn: &Connection) -> Vec<(String, String)> {
        let mut rows: Vec<(String, String)> = conn
            .query("SELECT key, value FROM config_kv", ())
            .await
            .expect("read config_kv")
            .into_iter()
            .map(|row| {
                (
                    row.get::<String>(0).expect("key"),
                    row.get::<String>(1).expect("value"),
                )
            })
            .collect();
        rows.sort();
        rows
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

    // Expected-shape fixtures (hand-maintained, independent of the baseline)
    //
    // These are deliberately typed out literally, NOT derived from the baseline
    // DDL constants above, so a wrong baseline cannot validate itself.

    /// Hand-maintained expected current core shape: table → its full column
    /// set. Column sets are compared order-insensitively because an
    /// ALTER-produced shape must compare equal to the baseline's CREATE-produced
    /// one.
    const EXPECTED_CORE_TABLE_COLUMNS: &[(&str, &[&str])] = &[
        (
            "agents",
            &[
                "job_id", "agent_id", "kind", "idx", "status", "outcome", "task",
            ],
        ),
        (
            "automations",
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
                "retry_at",
                "unfinished_attempts",
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
        ("guest_reports", &["user_name", "day", "digest"]),
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
                "chain",
                "extract",
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
                "chain",
                "context_block_len",
                "user_activity",
                "image_count",
                "provider",
                "provider_model",
                "extract",
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
                "priority",
                "label",
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
                "memory_inventory_day",
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

    /// Hand-maintained expected current core index set. It is the independent
    /// authority for today's fresh shape — never derive it from the baseline.
    /// `BASELINE_CORE_SCHEMA`'s doc carries the audit of what is kept and what
    /// is deliberately absent.
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
        "idx_automations_due",
        "idx_tickets_workspace_phase",
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
                "ticket",
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
    /// plan reaches (`BASELINE_LOGS_SCHEMA`'s doc carries the audit).
    const EXPECTED_LOGS_INDEXES: &[&str] = &[
        "idx_logs_level",
        "idx_tool_calls_recorded_at",
        "idx_tool_calls_error_message",
        "idx_llm_requests_recorded_at",
        "idx_grep_telemetry_recorded_at",
    ];

    /// Assert `conn`'s table and per-table column name sets, plus its explicit
    /// index name set, against the expected ones; errors list the difference.
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
            let actual = column_shapes(conn, table).await;
            let actual_set: std::collections::HashSet<&str> =
                actual.keys().map(String::as_str).collect();
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

    /// A fresh install converges to the exact current core shape: the table set
    /// (which proves the retired tables absent) and the per-table column sets
    /// (which prove the retired columns absent).
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

        let mut applied = recorded_ids(&conn).await;
        applied.sort();
        assert_eq!(
            applied,
            ["24", "69", "79", "80", "81", "82", "83", "84", "85"].map(String::from),
            "fresh core applies the baseline, the 69 seed and the seven upgrade entries exactly"
        );
    }

    /// A fresh logs store converges to the exact current shape: the five tables
    /// and only the indexes a product query's plan reaches.
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

        let mut applied = recorded_ids(&conn).await;
        applied.sort();
        assert_eq!(
            applied,
            vec!["26".to_string()],
            "fresh logs applies the baseline exactly"
        );
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

    /// Behavioral pin for catalog entry `81`: the stored priority list of the model that now has a
    /// built-in list of its own is deleted, while every other routing row — the DeepSeek
    /// identifiers whose ids share its prefix included — survives verbatim. Idempotent: a second
    /// execution matches no rows.
    #[tokio::test]
    async fn drop_deepseek_v41_flash_1017_routing_deletes_only_that_model() {
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
            ("deepseek/deepseek-v4.1-flash-1017", "deepinfra/fp8/flex"),
            ("deepseek/deepseek-v4.1-flash", "deepseek"),
            ("deepseek/deepseek-v4.1-flash-1017-preview", "deepseek"),
            ("qwen/qwen3.8-flash", "GMICloud"),
        ] {
            conn.execute(
                "INSERT INTO config_model_routing (model, provider_order) VALUES (?1, ?2)",
                params![model, order],
            )
            .await
            .unwrap();
        }

        let entry = MIGRATIONS
            .iter()
            .find(|m| m.id == "81")
            .expect("entry 81 in the catalog");
        run_catalog(&conn, TargetDb::Core, std::slice::from_ref(entry))
            .await
            .expect("apply entry 81");
        let remaining = model_routing_orders(&conn).await;
        assert_eq!(
            remaining,
            vec![
                (
                    "deepseek/deepseek-v4.1-flash".to_string(),
                    "deepseek".to_string()
                ),
                (
                    "deepseek/deepseek-v4.1-flash-1017-preview".to_string(),
                    "deepseek".to_string()
                ),
                ("qwen/qwen3.8-flash".to_string(), "GMICloud".to_string()),
            ],
            "only the exact model id may be dropped — prefix and family neighbours keep their row"
        );

        // The ledger records `81` after the first application, so re-run the SQL
        // body directly to pin that it matches no rows a second time.
        conn.execute(DROP_DEEPSEEK_V41_FLASH_1017_ROUTING, ())
            .await
            .expect("re-run entry 81 body");
        assert_eq!(model_routing_orders(&conn).await, remaining);
    }

    /// The boot open's schema reason to run its integrity scan: a catalog entry
    /// the store does not carry. A store at the catalog's head has none.
    #[tokio::test]
    async fn has_pending_reads_the_catalog_against_the_ledger() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(tmp.path(), crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        run_migrations(&conn, TargetDb::Core).await.unwrap();
        assert!(
            !has_pending(&conn, TargetDb::Core).await.unwrap(),
            "a store at the catalog's head has nothing pending"
        );
        let newest = MIGRATIONS
            .iter()
            .rev()
            .find(|migration| migration.target == TargetDb::Core)
            .expect("the core catalog has entries")
            .id;
        conn.execute(
            &format!("DELETE FROM schema_migrations WHERE id = '{newest}'"),
            (),
        )
        .await
        .unwrap();
        assert!(
            has_pending(&conn, TargetDb::Core).await.unwrap(),
            "an entry the ledger no longer records is pending again"
        );
    }

    // ── Frozen v0.10.0 fixture ─────────────────────────────────────────────
    //
    // `V0_10_0_CORE_SCHEMA` and `V0_10_0_LOGS_SCHEMA` are tag `v0.10.0`'s
    // `BASELINE_CORE_SCHEMA` / `BASELINE_LOGS_SCHEMA`, copied verbatim, and the
    // two id lists are what that release recorded per store. This is the oldest
    // store the reduced catalog still supports.
    //
    // It coheres as an upgrade fixture only because 0.10.0's baseline already
    // declared every column the retired upfills add: a baseline cannot upfill an
    // existing file. The one survivor that adds columns (`80`) adds the provider
    // pin, which the starting text does not have, and it runs on the fixture.
    // Never derive it from today's BASELINE_* constants — today's baseline
    // already carries the later additions and would agree with itself.
    // Regenerate it from the new floor's tag when the support floor moves.
    const V0_10_0_CORE_SCHEMA: &str = "\
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
    priority         INTEGER NOT NULL DEFAULT 1,
    label            TEXT NOT NULL DEFAULT 'feat',
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
    context_block_len INTEGER,
    user_activity  TEXT,
    image_count    INTEGER
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
    granted_tools       TEXT,
    memory_inventory_day TEXT
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
-- ── Ticket chronicle / automations ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS ticket_chronicle (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id      TEXT NOT NULL,
    workspace_name TEXT NOT NULL,
    source_phase   TEXT NOT NULL,
    target_phase   TEXT NOT NULL,
    at             TEXT NOT NULL,
    actor          TEXT
);
CREATE TABLE IF NOT EXISTS automations (
    id                  TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    user_name           TEXT NOT NULL,
    kind                TEXT NOT NULL,
    text                TEXT NOT NULL,
    fire_at             TEXT,
    interval_seconds    INTEGER,
    next_fire_at        TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',
    created_at          TEXT NOT NULL,
    trigger             TEXT,
    role                TEXT,
    identifier          TEXT,
    workspace           TEXT,
    retry_at            TEXT,
    unfinished_attempts INTEGER NOT NULL DEFAULT 0
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
CREATE INDEX IF NOT EXISTS idx_automations_due ON automations(status, next_fire_at);
CREATE INDEX IF NOT EXISTS idx_tickets_workspace_phase ON tickets (workspace_name, phase, is_archived, priority ASC, created_at DESC);";
    const V0_10_0_LOGS_SCHEMA: &str = "\
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
    workspace   TEXT NOT NULL DEFAULT '',
    ticket      TEXT NOT NULL DEFAULT ''
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
    /// The core ids 0.10.0 recorded: 24–78 minus the logs-target entries.
    const V0_10_0_CORE_IDS: &[&str] = &[
        "24", "25", "27", "28", "29", "30", "31", "32", "33", "34", "36", "37", "38", "39", "40",
        "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55",
        "56", "57", "59", "60", "61", "62", "63", "65", "66", "67", "68", "69", "70", "72", "73",
        "74", "75", "76", "77", "78",
    ];
    /// The logs ids 0.10.0 recorded.
    const V0_10_0_LOGS_IDS: &[&str] = &["26", "35", "58", "64", "71"];

    /// One column's declared shape, read from `PRAGMA table_info`: what the DDL
    /// says, not where in the row it sits.
    #[derive(Debug, PartialEq, Eq)]
    struct ColumnShape {
        declared: String,
        not_null: i64,
        default: Option<String>,
        pk: i64,
    }

    /// One foreign key, read from `PRAGMA foreign_key_list`.
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct ForeignKey {
        referenced_table: String,
        from: String,
        to: Option<String>,
        on_update: String,
        on_delete: String,
    }

    /// Table → column name → that column's declared shape.
    async fn column_shapes(
        conn: &Connection,
        table: &str,
    ) -> std::collections::BTreeMap<String, ColumnShape> {
        conn.query(&format!("PRAGMA table_info({table})"), ())
            .await
            .expect("read table_info")
            .into_iter()
            .map(|row| {
                (
                    row.get::<String>(1).expect("column name"),
                    ColumnShape {
                        declared: row.get::<String>(2).expect("declared type"),
                        not_null: row.get::<i64>(3).expect("notnull"),
                        default: row.get::<Option<String>>(4).expect("default"),
                        pk: row.get::<i64>(5).expect("pk"),
                    },
                )
            })
            .collect()
    }

    /// A table's foreign keys, sorted so the comparison ignores row order.
    async fn foreign_keys(conn: &Connection, table: &str) -> Vec<ForeignKey> {
        let mut keys: Vec<ForeignKey> = conn
            .query(&format!("PRAGMA foreign_key_list({table})"), ())
            .await
            .expect("read foreign_key_list")
            .into_iter()
            .map(|row| ForeignKey {
                referenced_table: row.get::<String>(2).expect("referenced table"),
                from: row.get::<String>(3).expect("from"),
                to: row.get::<Option<String>>(4).expect("to"),
                on_update: row.get::<String>(5).expect("on update"),
                on_delete: row.get::<String>(6).expect("on delete"),
            })
            .collect();
        keys.sort();
        keys
    }

    /// Assert two stores carry the same schema: the same tables, the same
    /// per-table columns with their declared type, nullability, default and
    /// primary-key position, the same foreign keys, and the same index
    /// definitions. Column order is deliberately not compared — an
    /// ALTER-appended column sits in a different position than a
    /// CREATE-declared one, while every fact that decides behaviour is.
    async fn assert_same_schema(upgraded: &Connection, fresh: &Connection, context: &str) {
        let mut upgraded_tables = table_names(upgraded).await;
        let mut fresh_tables = table_names(fresh).await;
        upgraded_tables.sort();
        fresh_tables.sort();
        assert_eq!(
            upgraded_tables, fresh_tables,
            "{context}: table sets differ"
        );
        for table in &fresh_tables {
            assert_eq!(
                column_shapes(upgraded, table).await,
                column_shapes(fresh, table).await,
                "{context}: table '{table}' column shapes differ"
            );
            assert_eq!(
                foreign_keys(upgraded, table).await,
                foreign_keys(fresh, table).await,
                "{context}: table '{table}' foreign keys differ"
            );
        }
        assert_eq!(
            index_defs(upgraded).await,
            index_defs(fresh).await,
            "{context}: index definitions differ"
        );
    }

    /// A store in the v0.10.0 shape: that release's baseline text, its recorded
    /// ids, and nothing else — the caller adds the rows the entries left behind.
    async fn v0_10_0_store(db_path: &std::path::Path, schema: &str, ids: &[&str]) -> Connection {
        let conn = crate::db::open_with_schema(db_path, "")
            .await
            .expect("open the v0.10.0 fixture store");
        conn.execute_batch(schema).await.expect("v0.10.0 baseline");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (\
             id TEXT PRIMARY KEY, applied_at TEXT NOT NULL)",
            (),
        )
        .await
        .expect("create the fixture ledger");
        for id in ids {
            conn.execute(
                "INSERT INTO schema_migrations (id, applied_at) VALUES (?1, ?2)",
                params![*id, crate::db::now()],
            )
            .await
            .expect("record a v0.10.0 id");
        }
        conn
    }

    /// The upgrade half of the reduced catalog: a database at the 0.10.0 shape
    /// reaches a fresh install's schema and the same catalog-written rows once
    /// the survivors run. The two rows are seeded because a v0.10.0 store
    /// already carries them — the transcription default entry `69` wrote, and
    /// the routing row entry `81` deletes; without them that half of the
    /// comparison would be vacuous for exactly the entries it protects.
    #[tokio::test]
    async fn v0_10_0_store_upgrades_to_the_fresh_shape() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = v0_10_0_store(
            &crate::db::store_db_path(&tmp.path().join("upgrade"), crate::db::CONSOLIDATED_DB_NAME),
            V0_10_0_CORE_SCHEMA,
            V0_10_0_CORE_IDS,
        )
        .await;
        conn.execute(
            "INSERT INTO config_kv (key, value) VALUES (?1, 'false')",
            params![crate::config::CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL],
        )
        .await
        .expect("seed entry 69's row as 0.10.0 left it");
        conn.execute(
            "INSERT INTO config_model_routing (model, provider_order) \
             VALUES ('deepseek/deepseek-v4.1-flash-1017', 'deepinfra/fp8/flex')",
            (),
        )
        .await
        .expect("seed the routing row entry 81 deletes");

        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("upgrade the v0.10.0 store");

        let fresh = crate::db::open_consolidated_store(&tmp.path().join("fresh"))
            .await
            .expect("fresh store");
        assert_same_schema(&conn, &fresh, "v0.10.0 upgrade vs fresh core").await;
        assert_eq!(
            config_kv_rows(&conn).await,
            config_kv_rows(&fresh).await,
            "the catalog-written config rows must match a fresh install"
        );
        assert_eq!(
            model_routing_orders(&conn).await,
            model_routing_orders(&fresh).await,
            "entry 81 must delete the routing row a v0.10.0 store carries"
        );
    }

    /// The logs half: a v0.10.0 logs store already carries the whole logs
    /// catalog, so the reduced catalog leaves it at the fresh shape. This is the
    /// guard that no logs entry a 0.10.x store still needs was retired.
    #[tokio::test]
    async fn v0_10_0_logs_store_matches_the_fresh_shape() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = v0_10_0_store(
            &crate::db::store_db_path(&tmp.path().join("upgrade"), crate::db::LOG_DB_NAME),
            V0_10_0_LOGS_SCHEMA,
            V0_10_0_LOGS_IDS,
        )
        .await;
        run_migrations(&conn, TargetDb::Logs)
            .await
            .expect("upgrade the v0.10.0 logs store");

        let fresh = crate::db::open_with_schema(
            &crate::db::store_db_path(&tmp.path().join("fresh"), crate::db::LOG_DB_NAME),
            "",
        )
        .await
        .expect("fresh logs store");
        run_migrations(&fresh, TargetDb::Logs)
            .await
            .expect("fresh logs catalog");

        assert_same_schema(&conn, &fresh, "v0.10.0 logs upgrade vs fresh logs").await;
    }

    /// Entry `69` is the only automatic writer of the transcription-off row: a
    /// fresh install ships disabled, and a value the owner wrote is never
    /// overwritten, which is what keeps the row's absence meaning "enabled".
    #[tokio::test]
    async fn seed_audio_transcription_disabled_writes_the_off_default_once() {
        let key = crate::config::CONFIG_KEY_AUDIO_TRANSCRIPTION_USE_LOCAL;
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .expect("fresh store");
        // Rows other catalog entries write are not this entry's business.
        let transcription = |rows: Vec<(String, String)>| {
            rows.into_iter()
                .filter(|(stored, _)| stored == key)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            transcription(config_kv_rows(&conn).await),
            vec![(key.to_string(), "false".to_string())],
            "a fresh install ships transcription off"
        );

        conn.execute(
            "UPDATE config_kv SET value = 'true' WHERE key = ?1",
            params![key],
        )
        .await
        .unwrap();
        run_seed_audio_transcription_disabled(&conn)
            .await
            .expect("re-run entry 69");
        assert_eq!(
            transcription(config_kv_rows(&conn).await),
            vec![(key.to_string(), "true".to_string())],
            "an owner's value is never overwritten"
        );
    }

    /// Entry `82` puts every install on the shipped image/video model lists: a hand-edited list is
    /// overwritten by design, and an install with no row at all gets the same lists.
    #[tokio::test]
    async fn set_media_model_lists_overwrites_hand_edited_lists() {
        let image_key = crate::config::CONFIG_KEY_IMAGE_GEN_MODELS;
        let video_key = crate::config::CONFIG_KEY_VIDEO_MODELS;
        let shipped = vec![
            (
                image_key.to_string(),
                crate::config::SHIPPED_IMAGE_GEN_MODELS.to_string(),
            ),
            (
                video_key.to_string(),
                crate::config::SHIPPED_VIDEO_MODELS.to_string(),
            ),
        ];
        let lists = |rows: Vec<(String, String)>| {
            rows.into_iter()
                .filter(|(key, _)| key == image_key || key == video_key)
                .collect::<Vec<_>>()
        };

        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .expect("fresh store");
        assert_eq!(
            lists(config_kv_rows(&conn).await),
            shipped,
            "a fresh store must run the catalog onto the shipped lists"
        );

        conn.execute(
            "UPDATE config_kv SET value = 'hand/pick' WHERE key IN (?1, ?2)",
            params![image_key, video_key],
        )
        .await
        .unwrap();
        run_set_media_model_lists(&conn)
            .await
            .expect("re-run entry 82");
        assert_eq!(
            lists(config_kv_rows(&conn).await),
            shipped,
            "a hand-edited list is replaced, otherwise the new defaults never reach the install"
        );

        conn.execute(
            "DELETE FROM config_kv WHERE key IN (?1, ?2)",
            params![image_key, video_key],
        )
        .await
        .unwrap();
        run_set_media_model_lists(&conn)
            .await
            .expect("re-run entry 82 with no rows");
        assert_eq!(
            lists(config_kv_rows(&conn).await),
            shipped,
            "an install predating the rows must receive them too"
        );
    }

    /// A store built from the source tree can already have recorded `82` with the old list. Entry
    /// `83` re-runs the same body, so such a store still arrives at the shipped list.
    #[tokio::test]
    async fn a_store_that_recorded_the_media_list_step_still_arrives_at_the_new_list() {
        let video_key = crate::config::CONFIG_KEY_VIDEO_MODELS;

        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .expect("fresh store");
        conn.execute(
            "UPDATE config_kv SET value = 'minimax/hailuo-3' WHERE key = ?1",
            params![video_key],
        )
        .await
        .unwrap();
        conn.execute("DELETE FROM schema_migrations WHERE id = '83'", ())
            .await
            .unwrap();

        run_migrations(&conn, TargetDb::Core)
            .await
            .expect("re-run the core catalog");

        let video = config_kv_rows(&conn)
            .await
            .into_iter()
            .find(|(key, _)| key == video_key)
            .map(|(_, value)| value);
        assert_eq!(
            video.as_deref(),
            Some(crate::config::SHIPPED_VIDEO_MODELS),
            "a store that already recorded 82 must still reach the shipped video list"
        );
        assert!(
            recorded_ids(&conn).await.iter().any(|id| id == "83"),
            "entry 83 must be recorded again"
        );
    }
}
