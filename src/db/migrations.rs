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
//! except entries `25`/`27`–`34`, which genuinely upfill the retired delta
//! `23`'s `chat_history` reply columns, the `workspaces.maintainer_recommendations`
//! and `jobs.caller_agent_id` / `session_metadata.created_at` columns, the
//! per-user `image_gen_model` / `video_model` columns, the backfilled
//! `jobs.mode` discriminator, the `session_metadata.sleep_ended` marker, the
//! `alarms.command` column, the `chat_history.broadcast_id` column, and the
//! `tickets.last_transition_actor` / `ticket_chronicle.actor` columns on
//! one-delta-behind / current databases.
//!
//! Future schema changes resume the chain at id `35` with monotonically
//! increasing, unique integer ids, never reused across any store for the
//! lifetime of the catalog.
//!
//! # Failure semantics
//!
//! A catalog/logic error is a hard boot failure — it is **never** absorbed by
//! the heal/quarantine/`catch_unwind` wrappers and never triggers a recreate.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Context;
use futures_util::future::BoxFuture;

use crate::db::{Connection, params};

/// Which physical database a migration targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetDb {
    Core,
    Logs,
}

/// A Rust-function migration body. The function is called without a
/// surrounding transaction (it owns its own transactional/FK semantics).
type RustFn = for<'a> fn(&'a Connection, &'a Path) -> BoxFuture<'a, anyhow::Result<()>>;

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
/// core index — including the tickets FTS index and the `idx_jobs_phase_ticket`
/// unique index. The `idx_jobs_caller_agent` index is created by delta `28`
/// (which also upfills `jobs.caller_agent_id` / `session_metadata.created_at`
/// on upgraded databases), not in this batch — the batch cannot reference a
/// column that a later Rust delta adds.
///
/// `jobs` is created WITH `caller_agent_id` and `mode` as its trailing
/// columns, matching the shape the retired `ALTER TABLE ...` plus deltas
/// `28`/`30` produce, so `idx_jobs_phase_ticket` (which references
/// `jobs.ticket_id`) is safe to create in the same batch. Every statement
/// is `IF NOT EXISTS`.
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
    sleep_ended   TEXT
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
    mode           TEXT
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
    permissions         TEXT,
    selected_workspace  TEXT,
    selected_role       TEXT,
    image_gen_model     TEXT,
    video_model         TEXT
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
    command          TEXT
);
-- ── Indexes ────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ticket_comments_ticket_id ON ticket_comments(ticket_id);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id, id);
CREATE INDEX IF NOT EXISTS idx_jobs_kind_status ON jobs(kind, status);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agents_anchor ON agents(agent_id) WHERE job_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_pending_jobs_agent_created ON pending_jobs(target_agent_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS workspace_contexts_null_role ON workspace_contexts(workspace_name) WHERE role IS NULL;
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_name);
CREATE INDEX IF NOT EXISTS idx_chat_history_workspace ON chat_history(workspace);
CREATE INDEX IF NOT EXISTS idx_chat_history_user_ws_id ON chat_history(user_name, workspace, id);
CREATE INDEX IF NOT EXISTS idx_tickets_title_fts ON tickets USING fts (title) WITH (tokenizer = 'ngram');
CREATE INDEX IF NOT EXISTS idx_tickets_board_active ON tickets (is_archived, priority ASC, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_phase_ticket ON jobs(kind, ticket_id) WHERE ticket_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ticket_chronicle_ws_id ON ticket_chronicle(workspace_name, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ticket_chronicle_dedup ON ticket_chronicle(ticket_id, workspace_name, source_phase, target_phase, at);
CREATE INDEX IF NOT EXISTS idx_alarms_due ON alarms(status, next_fire_at);
CREATE INDEX IF NOT EXISTS idx_tickets_workspace_phase ON tickets (workspace_name, phase, is_archived, priority ASC, created_at DESC);";

/// The full current-shape logs schema: `logs`, `tool_calls`, `llm_requests`
/// (the observability tables) plus the `grep_telemetry` table and all of their
/// indexes. Every statement is `IF NOT EXISTS`, so on an already-current
/// database the whole batch is a no-op.
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
CREATE INDEX IF NOT EXISTS idx_llm_requests_purpose ON llm_requests(purpose);
CREATE INDEX IF NOT EXISTS idx_grep_telemetry_recorded_at ON grep_telemetry(recorded_at);
CREATE INDEX IF NOT EXISTS idx_grep_telemetry_served ON grep_telemetry(served);
CREATE INDEX IF NOT EXISTS idx_grep_telemetry_reason ON grep_telemetry(reason);
CREATE INDEX IF NOT EXISTS idx_grep_telemetry_command ON grep_telemetry(command);";

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
///   guarded Rust body plus the `idx_jobs_caller_agent` index.
/// - `29` adds the per-user `image_gen_model` / `video_model` columns.
/// - `30` adds `jobs.mode` and backfills it from the `caller_agent_id`
///   NULL-sentinel (sync = has caller pin, async = NULL).
/// - `31` adds the `session_metadata.sleep_ended` marker via a guarded Rust
///   body — a real upfill on existing installs, a no-op on fresh ones.
/// - `32` adds the nullable `alarms.command` column.
/// - `33` adds the nullable `chat_history.broadcast_id` column.
/// - `34` adds the `tickets.last_transition_actor` and
///   `ticket_chronicle.actor` columns for phase-transition actor attribution.
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
];

/// Apply the migration catalog to `conn` for one physical database.
///
/// Iterates [`MIGRATIONS`] in order, skipping entries that target a different
/// database or whose id is already recorded in `schema_migrations`. Each
/// SQL entry runs inside its own transaction (schema change + tracking row
/// commit atomically); each Rust entry runs without a surrounding transaction.
/// A failure propagates — it is a hard boot failure, never healed/quarantined.
pub(crate) async fn run_migrations(
    conn: &Connection,
    db: TargetDb,
    root: &Path,
) -> anyhow::Result<()> {
    run_catalog(conn, db, root, MIGRATIONS).await
}

/// The migration loop, parameterized by a catalog so tests can replay the
/// retired `1`–`23` chain (see the test fixtures). Creates `schema_migrations`,
/// reads the applied ids, and runs each entry once, in catalog order, targeting
/// only `db`.
async fn run_catalog(
    conn: &Connection,
    db: TargetDb,
    root: &Path,
    catalog: &[Migration],
) -> anyhow::Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (\
             id         TEXT PRIMARY KEY,\
             applied_at TEXT NOT NULL\
         )",
        (),
    )
    .await
    .context("Failed to create schema_migrations tracking table")?;

    let applied: HashSet<String> = conn
        .query("SELECT id FROM schema_migrations", ())
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
                    "INSERT INTO schema_migrations (id, applied_at) VALUES (?1, ?2)",
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
                // is safe because every Rust body — the `chat_history` reply-column
                // upfill, which probes before altering — is idempotent and
                // re-runnable, so a crash between body success and id recording
                // re-runs the body without corruption on the next boot.
                run(conn, root)
                    .await
                    .with_context(|| format!("Migration '{}' failed", migration.id))?;
                conn.execute(
                    "INSERT INTO schema_migrations (id, applied_at) VALUES (?1, ?2)",
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

fn add_chat_history_reply_columns<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
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
        add_column_if_missing(conn, "chat_history", column).await?;
    }
    Ok(())
}

fn add_chat_history_broadcast_id<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
    Box::pin(run_add_chat_history_broadcast_id(conn))
}

/// Upfill the `chat_history.broadcast_id` column for databases created before
/// delta `33`: a shared id tagging all per-user copies of one broadcast
/// dispatch so the workspace stream can dedup agent rows exactly. Fresh
/// installs get the column from entry `24`'s `CREATE TABLE`, so the probe makes
/// this a no-op there. Guarded by [`add_column_if_missing`], so this body is
/// idempotent (non-transactional like every Rust body, re-runnable until
/// recorded).
async fn run_add_chat_history_broadcast_id(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "chat_history", "broadcast_id").await
}

fn add_ticket_transition_actor<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
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
    add_column_if_missing(conn, "tickets", "last_transition_actor").await?;
    add_column_if_missing(conn, "ticket_chronicle", "actor").await
}

fn add_workspaces_maintainer_recommendations<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
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
    add_column_if_missing(conn, "workspaces", "maintainer_recommendations").await
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

/// Idempotently add a plain `TEXT` column to `table` (Turso has no
/// `ADD COLUMN IF NOT EXISTS`): probe with [`column_exists`], then
/// `ALTER TABLE`. Used by the guarded migration bodies above and below.
async fn add_column_if_missing(conn: &Connection, table: &str, column: &str) -> anyhow::Result<()> {
    if !column_exists(conn, table, column).await? {
        conn.execute(&format!("ALTER TABLE {table} ADD COLUMN {column} TEXT"), ())
            .await
            .with_context(|| format!("Failed to add {table}.{column}"))?;
    }
    Ok(())
}

fn add_jobs_caller_agent_and_session_created_at<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
    Box::pin(run_add_jobs_caller_agent_and_session_created_at(conn))
}

/// Upfill the sync-resume primitives for databases created before delta `28`:
/// `jobs.caller_agent_id` (the caller agent's stable session pin for sync
/// analyze/implement jobs; NULL for async/boot-resumed jobs) and
/// `session_metadata.created_at` (the session's creation timestamp). Fresh
/// installs get both columns from entry `24`'s
/// `CREATE TABLE`, so the probes make this a no-op there — and the partial
/// `idx_jobs_caller_agent` index is likewise a no-op on fresh installs.
/// Idempotent, non-transactional like every Rust body (re-runnable until
/// recorded).
async fn run_add_jobs_caller_agent_and_session_created_at(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "jobs", "caller_agent_id").await?;
    add_column_if_missing(conn, "session_metadata", "created_at").await?;
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_jobs_caller_agent \
         ON jobs(caller_agent_id) WHERE caller_agent_id IS NOT NULL;",
    )
    .await
    .with_context(|| "Failed to create idx_jobs_caller_agent")?;
    Ok(())
}

fn add_users_image_and_video_models<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
    Box::pin(run_add_users_image_and_video_models(conn))
}

/// Upfill the per-user image/video model columns for databases created before
/// delta `29`. Fresh installs get both columns from entry `24`'s `CREATE
/// TABLE`, so the probe makes this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional
/// like every Rust body, re-runnable until recorded).
async fn run_add_users_image_and_video_models(conn: &Connection) -> anyhow::Result<()> {
    for column in ["image_gen_model", "video_model"] {
        add_column_if_missing(conn, "users", column).await?;
    }
    Ok(())
}

fn add_jobs_mode<'a>(conn: &'a Connection, _root: &'a Path) -> BoxFuture<'a, anyhow::Result<()>> {
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
    add_column_if_missing(conn, "jobs", "mode").await?;
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

fn add_session_metadata_sleep_ended<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
    Box::pin(run_add_session_metadata_sleep_ended(conn))
}

/// Upfill the per-session `sleep_ended` marker column for databases created
/// before delta `31`. Fresh installs get the column from entry `24`'s
/// `CREATE TABLE`, so the probe makes this a no-op there. Guarded by
/// [`add_column_if_missing`], so this body is idempotent (non-transactional
/// like every Rust body, re-runnable until recorded).
async fn run_add_session_metadata_sleep_ended(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "session_metadata", "sleep_ended").await
}

fn add_alarms_command<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
    Box::pin(run_add_alarms_command(conn))
}

/// Upfill the nullable `alarms.command` column for databases created before
/// delta `32`. Fresh installs get the column from entry `24`'s `CREATE TABLE`,
/// so the probe makes this a no-op there. Guarded by [`add_column_if_missing`],
/// so this body is idempotent (non-transactional like every Rust body,
/// re-runnable until recorded).
async fn run_add_alarms_command(conn: &Connection) -> anyhow::Result<()> {
    add_column_if_missing(conn, "alarms", "command").await
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
                "command",
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
            ],
        ),
        (
            "sessions",
            &["id", "agent_id", "role", "content", "created_at"],
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
                "permissions",
                "selected_workspace",
                "selected_role",
                "image_gen_model",
                "video_model",
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

    const EXPECTED_CORE_INDEXES: &[&str] = &[
        "idx_ticket_comments_ticket_id",
        "idx_sessions_agent_id",
        "idx_jobs_kind_status",
        "idx_jobs_updated_at",
        "idx_agents_anchor",
        "idx_pending_jobs_agent_created",
        "workspace_contexts_null_role",
        "idx_chat_history_user",
        "idx_chat_history_workspace",
        "idx_chat_history_user_ws_id",
        "idx_tickets_title_fts",
        "idx_tickets_board_active",
        "idx_jobs_phase_ticket",
        "idx_jobs_caller_agent",
        "idx_ticket_chronicle_ws_id",
        "idx_ticket_chronicle_dedup",
        "idx_alarms_due",
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

    const EXPECTED_LOGS_INDEXES: &[&str] = &[
        "idx_logs_timestamp",
        "idx_logs_level",
        "idx_logs_target",
        "idx_logs_agent_role",
        "idx_logs_agent_id",
        "idx_logs_workspace",
        "idx_tool_calls_agent_id",
        "idx_tool_calls_role",
        "idx_tool_calls_tool_name",
        "idx_tool_calls_recorded_at",
        "idx_tool_calls_workspace",
        "idx_tool_calls_error_message",
        "idx_llm_requests_recorded_at",
        "idx_llm_requests_agent_id",
        "idx_llm_requests_model",
        "idx_llm_requests_purpose",
        "idx_grep_telemetry_recorded_at",
        "idx_grep_telemetry_served",
        "idx_grep_telemetry_reason",
        "idx_grep_telemetry_command",
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
    /// `expected_indexes` (explicit index name-set equality). Errors list the
    /// missing/extra items with `context` for a helpful message.
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

        let idx_names: Vec<String> = index_defs(conn).await.into_keys().collect();
        let actual_idx: std::collections::HashSet<&str> =
            idx_names.iter().map(String::as_str).collect();
        let expected_idx: std::collections::HashSet<&str> =
            expected_indexes.iter().copied().collect();
        let missing_idx: Vec<&str> = expected_idx.difference(&actual_idx).copied().collect();
        let extra_idx: Vec<&str> = actual_idx.difference(&expected_idx).copied().collect();
        assert!(
            missing_idx.is_empty(),
            "{context}: missing indexes {missing_idx:?}"
        );
        assert!(
            extra_idx.is_empty(),
            "{context}: unexpected indexes {extra_idx:?}"
        );
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
    fn noop_import<'a>(
        _conn: &'a Connection,
        _root: &'a Path,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        Box::pin(async { Ok(()) })
    }

    fn cleanup_legacy_ticket_jobs<'a>(
        conn: &'a Connection,
        _root: &'a Path,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
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

    fn drop_jobs_paused_frozen<'a>(
        conn: &'a Connection,
        _root: &'a Path,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
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

    fn drop_user_roles_and_seed_onboarding<'a>(
        conn: &'a Connection,
        _root: &'a Path,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
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

    fn rewrite_analysis_verdicts<'a>(
        conn: &'a Connection,
        _root: &'a Path,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
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

    /// A fresh install runs the baseline (`24`–`31`) and converges to the
    /// exact current core shape: the table set (which also proves the required
    /// absences of `user_roles` / `config_role` / `ticket_jobs` /
    /// `ticket_stage_jobs`) and the per-table column sets (which prove the
    /// absences of `assigned_to` / `pipeline_reservation` / `paused_frozen`).
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
            vec![
                "24".to_string(),
                "25".to_string(),
                "27".to_string(),
                "28".to_string(),
                "29".to_string(),
                "30".to_string(),
                "31".to_string(),
                "32".to_string(),
                "33".to_string(),
                "34".to_string()
            ],
            "fresh core applies baseline 24/25/27/28/29/30/31/32/33/34 exactly"
        );
    }

    /// A fresh logs store runs the baseline (`26`) and converges to the exact
    /// current logs shape (logs/tool_calls/llm_requests + grep_telemetry).
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
        run_migrations(&conn, TargetDb::Logs, root)
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
            vec!["26".to_string()],
            "fresh logs applies baseline 26 exactly"
        );
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
            "INSERT INTO users (name, permissions, selected_workspace, selected_role) \
             VALUES ('bob', NULL, 'ws', 'assistant')",
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
            counts: table_row_counts(conn, &expected_core_domain_tables()).await,
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
    /// `jobs`/`session_metadata` column upfills), and `except_indexes`, whose
    /// definitions may be added on reopen. Row counts, `chat_history` content
    /// and `tickets` content are asserted to be strictly unchanged.
    async fn assert_core_catalog_unchanged(
        conn: &Connection,
        before: &CoreSnapshot,
        except_tables: &[&str],
        except_indexes: &[&str],
    ) {
        assert_eq!(
            without(&table_defs(conn).await, except_tables),
            without(&before.tables, except_tables),
            "core table DDL (except {except_tables:?}) must be unchanged on reopen"
        );
        assert_eq!(
            without(&index_defs(conn).await, except_indexes),
            without(&before.indexes, except_indexes),
            "core index definitions (except {except_indexes:?}) must be unchanged on reopen"
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
            table_row_counts(conn, &expected_core_domain_tables()).await,
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
    /// the new baseline (`24`/`25`/`27`/`28`/`29`/`30`/`31`/`32`/`33`) as a
    /// STRICT no-op except the delta-27 `workspaces.maintainer_recommendations`
    /// column upfill, the delta-28 `jobs.caller_agent_id` /
    /// `session_metadata.created_at` column upfills (plus the delta-28
    /// `idx_jobs_caller_agent` index), the delta-29 per-user model columns, and
    /// the delta-30 `jobs.mode` column upfill + backfill, the delta-31
    /// `sleep_ended` column upfill, the delta-32 `alarms.command` column
    /// upfill, and the delta-33 `chat_history.broadcast_id` column upfill, and
    /// the delta-34 `tickets.last_transition_actor` /
    /// `ticket_chronicle.actor` column upfill, all
    /// asserted explicitly. This also proves
    /// Turso honors `IF NOT EXISTS` on the FTS index when the baseline re-runs it.
    #[tokio::test]
    async fn old_catalog_current_db_reopens_as_noop() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        run_catalog(&conn, TargetDb::Core, root, OLD_CATALOG)
            .await
            .expect("old catalog");
        seed_current_core_rows(&conn).await;

        let before_ids = applied_ids(&conn).await;
        let before = snapshot_core_state(&conn).await;

        run_migrations(&conn, TargetDb::Core, root)
            .await
            .expect("new catalog");

        let mut expected_ids = before_ids.clone();
        expected_ids.push("24".to_string());
        expected_ids.push("25".to_string());
        expected_ids.push("27".to_string());
        expected_ids.push("28".to_string());
        expected_ids.push("29".to_string());
        expected_ids.push("30".to_string());
        expected_ids.push("31".to_string());
        expected_ids.push("32".to_string());
        expected_ids.push("33".to_string());
        expected_ids.push("34".to_string());
        expected_ids.sort();
        let mut after_ids = applied_ids(&conn).await;
        after_ids.sort();
        assert_eq!(
            after_ids, expected_ids,
            "reopen must record exactly old ids ∪ 24/25/27/28/29/30/31/32/33/34"
        );

        // Everything else is a strict no-op; only workspaces (delta 27),
        // jobs/session_metadata (delta 28), users (delta 29), jobs.mode
        // (delta 30), session_metadata.sleep_ended (delta 31),
        // alarms.command (delta 32) and chat_history.broadcast_id (delta 33)
        // gain their columns, appended at the end by the ALTER, and the
        // delta-28 `idx_jobs_caller_agent` index is added.
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
            ],
            &["idx_jobs_caller_agent"],
        )
        .await;
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
        assert_eq!(
            after_jobs_cols["jobs"], expected_jobs_cols,
            "reopen must append exactly caller_agent_id/mode to jobs columns"
        );
        let after_sm_cols = column_sets(&conn, &["session_metadata"]).await;
        let mut expected_sm_cols = before.cols["session_metadata"].clone();
        expected_sm_cols.push("created_at".to_string());
        expected_sm_cols.push("sleep_ended".to_string());
        assert_eq!(
            after_sm_cols["session_metadata"], expected_sm_cols,
            "reopen must append exactly created_at/sleep_ended to session_metadata columns"
        );
        let after_users_cols = column_sets(&conn, &["users"]).await;
        let mut expected_users_cols = before.cols["users"].clone();
        expected_users_cols.push("image_gen_model".to_string());
        expected_users_cols.push("video_model".to_string());
        assert_eq!(
            after_users_cols["users"], expected_users_cols,
            "reopen must append exactly image_gen_model/video_model to users columns"
        );
        let after_alarms_cols = column_sets(&conn, &["alarms"]).await;
        let mut expected_alarms_cols = before.cols["alarms"].clone();
        expected_alarms_cols.push("command".to_string());
        assert_eq!(
            after_alarms_cols["alarms"], expected_alarms_cols,
            "reopen must append exactly command to alarms columns"
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
    /// `8`/`9`/`14`) reopens through the logs baseline (`26`) as a strict
    /// no-op — schema and data unchanged.
    #[tokio::test]
    async fn old_catalog_logs_db_reopens_as_noop() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::LOG_DB_NAME),
            "",
        )
        .await
        .expect("open logs");
        run_catalog(&conn, TargetDb::Logs, root, OLD_CATALOG)
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
        let before_counts = table_row_counts(&conn, &expected_logs_domain_tables()).await;

        run_migrations(&conn, TargetDb::Logs, root)
            .await
            .expect("new catalog");

        let mut expected_ids = before_ids.clone();
        expected_ids.push("26".to_string());
        expected_ids.sort();
        let mut after_ids = applied_ids(&conn).await;
        after_ids.sort();
        assert_eq!(
            after_ids, expected_ids,
            "logs reopen must record exactly old ids ∪ 26"
        );

        assert_eq!(
            table_defs(&conn).await,
            before_tables,
            "logs table DDL must be unchanged on reopen"
        );
        assert_eq!(
            index_defs(&conn).await,
            before_indexes,
            "logs index definitions must be unchanged on reopen"
        );
        assert_eq!(
            table_row_counts(&conn, &expected_logs_domain_tables()).await,
            before_counts,
            "logs row counts must be unchanged on reopen"
        );
    }

    /// The 0.5.0 (one-delta-behind) upgrade: a database built by the retired
    /// chain WITHOUT entry `23` has no `chat_history` reply columns; the
    /// baseline entry `25` upfills exactly those two columns, entry `27` adds
    /// `workspaces.maintainer_recommendations`, entry `28` adds
    /// `jobs.caller_agent_id` / `session_metadata.created_at` (plus its
    /// partial index), entry `30` adds `jobs.mode`, entry `31` adds
    /// `session_metadata.sleep_ended`, entry `32` adds `alarms.command`, and
    /// entry `33` adds `chat_history.broadcast_id`, and entry `34` adds
    /// `tickets.last_transition_actor` / `ticket_chronicle.actor`,
    /// leaving every row and every other table's schema untouched.
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
        run_catalog(
            &conn,
            TargetDb::Core,
            root,
            &old_catalog_without_reply_delta(),
        )
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
            })
            .collect();
        let before_other = column_sets(&conn, &untouched_tables).await;
        let mut before_ids = applied_ids(&conn).await;
        before_ids.sort();

        run_migrations(&conn, TargetDb::Core, root)
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
        expected_ids.push("24".to_string());
        expected_ids.push("25".to_string());
        expected_ids.push("27".to_string());
        expected_ids.push("28".to_string());
        expected_ids.push("29".to_string());
        expected_ids.push("30".to_string());
        expected_ids.push("31".to_string());
        expected_ids.push("32".to_string());
        expected_ids.push("33".to_string());
        expected_ids.push("34".to_string());
        expected_ids.sort();
        let mut after_ids = applied_ids(&conn).await;
        after_ids.sort();
        assert_eq!(
            after_ids, expected_ids,
            "upgrade must record exactly old ids ∪ 24/25/27/28/29/30/31/32/33/34"
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
        let after_alarms_cols = column_names(&conn, "alarms").await;
        assert!(
            after_alarms_cols.contains(&"command".to_string()),
            "command must be added to alarms"
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
             jobs/session_metadata (delta 28), users (delta 29), jobs.mode \
             (delta 30), session_metadata.sleep_ended (delta 31), \
             alarms.command (delta 32) and tickets/ticket_chronicle (delta 34) \
             columns may change on the 0.5.0 upgrade"
        );
    }
}
