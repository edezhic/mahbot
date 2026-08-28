//! The single, strictly-linear, append-only schema catalog for all Turso
//! stores (core.db and logs.db).
//!
//! # Core principle
//!
//! Every schema creation, change, and removal in the codebase is an entry in
//! [`MIGRATIONS`]. There is **no other place** that defines or alters a table
//! schema. The catalog is a **strictly linear history** anchored at the **0.4.2
//! baseline** (commit `4a68782`): the starter create-table entries encode the
//! 0.4.2 table shapes, and every subsequent entry brings a database from 0.4.2
//! to the current shape. Application order is **catalog position**, end to end.
//!
//! # Id-only applied check
//!
//! An entry is identified solely by its [`Migration::id`]. The runner's
//! mechanism is exactly:
//!
//! - if the id is already recorded in that database's `schema_migrations` →
//!   skip (already applied);
//! - otherwise → run the entry's SQL text (or Rust function) and record its id.
//!
//! There are **no runner guards** (no environment checks, no fresh-vs-upgrade
//! branching): the only control flow is the id-based applied check. Idempotent
//! DDL (`IF NOT EXISTS`) simply lives in the SQL text, and per-body idempotency
//! logic (e.g. the [`run_drop_jobs_paused_frozen`] existence probe, needed
//! because Turso has no `DROP COLUMN IF EXISTS`) is part of a migration body,
//! never a conditional in the runner.
//!
//! # Ids
//!
//! Existing already-deployed ids (e.g. `consolidate_003_jobs_ticket_id`,
//! `001_drop_ticket_pipeline_reservation`) are preserved verbatim so they act
//! as no-ops on databases that already recorded them. New migrations use plain
//! monotonically increasing integer ids, unique and never reused across all
//! physical Turso stores for the lifetime of the catalog. `schema_migrations.id`
//! is `TEXT PRIMARY KEY`, so mixed string/integer ids coexist. Integer ids are
//! assigned in the order entries were added, but the array groups entries by
//! store/target for readability — numeric id order need not match array
//! position (application order is always catalog position).
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
/// surrounding transaction (it owns its own transactional/FK semantics, e.g.
/// the non-transactional consolidation import).
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

// ── 0.4.2 baseline DDL ───────────────────────────────────────────────────
//
// The shape at commit `4a68782` (the 0.4.2 version bump). Fresh databases
// create these tables, then the delta entries below evolve them to current.

/// Board baseline (0.4.2): `tickets` carries `assigned_to` and
/// `pipeline_reservation`; `review_base_count` was added and removed *before*
/// 0.4.2 and is deliberately absent (pre-baseline drift, out of scope).
const BASELINE_BOARD_TABLES: &str = "\
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

/// Session/jobs baseline (0.4.2): `jobs` has no `ticket_id` and no
/// `paused_frozen`; `ticket_stage_jobs` still carries `round`/`phase`/`stage`.
const BASELINE_SESSION_TABLES: &str = "\
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

/// Workspace baseline (0.4.2).
const BASELINE_WORKSPACE_TABLES: &str = "\
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

/// Users baseline (0.4.2).
const BASELINE_USERS_TABLES: &str = "\
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

/// Config baseline (0.4.2): `config_role` still exists (dropped by a delta).
const BASELINE_CONFIG_TABLES: &str = "\
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

/// Chat-history baseline (0.4.2).
const BASELINE_CHAT_HISTORY_TABLES: &str = "\
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    user_name TEXT NOT NULL,
    direction TEXT NOT NULL,
    content TEXT NOT NULL,
    agent_role TEXT,
    workspace TEXT NOT NULL
);";

/// 0.4.2 indexes / unique constraints that reference only 0.4.2 columns.
/// `idx_jobs_phase_ticket` (references `jobs.ticket_id`), the FTS index, and
/// `idx_tickets_board_active` are all **post-0.4.2** and live in the delta
/// chain below.
const BASELINE_CORE_INDEXES: &str = "\
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

/// Logs baseline (0.4.2): `logs`, `tool_calls`, and `llm_requests` (which
/// already carried the cost/upstream_provider/system_fingerprint observability
/// columns). `grep_telemetry` did not exist at 0.4.2 and is a delta.
const BASELINE_LOGS_TABLES: &str = "\
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

const BASELINE_LOGS_INDEXES: &str = "\
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

// ── Post-0.4.2 delta DDL ─────────────────────────────────────────────────

/// `config_role` is dropped after the consolidation (it was mapped to `None`).
const DELTA_DROP_CONFIG_ROLE: &str = "DROP TABLE IF EXISTS config_role;";

/// `ticket_stage_jobs` is dropped by the delta chain (renamed to `ticket_jobs`,
/// then dropped). On an already-current consolidated DB the baseline re-creates it
/// (the drop/rename deltas are already-recorded and skipped), so this idempotent
/// entry removes the leaked table. It is a safe no-op on fresh/0.4.2 installs.
const DELTA_DROP_TICKET_STAGE_JOBS: &str = "DROP TABLE IF EXISTS ticket_stage_jobs;";

/// The tickets FTS index must exist BEFORE the consolidation import so the
/// bulk-inserted tickets maintain it (turso's `CREATE INDEX ... USING fts`
/// does not backfill pre-existing rows). The runtime tokenizer-repair hook
/// (`ensure_fts_index`) stays a repair, not a schema path.
const DELTA_FTS_INDEX: &str = "CREATE INDEX IF NOT EXISTS idx_tickets_title_fts ON tickets \
USING fts (title) WITH (tokenizer = 'ngram');";

/// `idx_tickets_board_active` is post-0.4.2; it references only baseline
/// `tickets` columns, so it can run at any point after the baseline.
const DELTA_BOARD_ACTIVE_INDEX: &str = "CREATE INDEX IF NOT EXISTS idx_tickets_board_active ON tickets \
(is_archived, priority ASC, created_at DESC);";

/// `grep_telemetry` is the sole 0.4.2→current logs delta.
const DELTA_GREP_TELEMETRY: &str = "\
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

/// Alarm/reminder table backing the `add_alarm`/`list_alarms`/`remove_alarm`
/// tools and the periodic sweep that routes due reminders back into the
/// Assistant's own session.
const DELTA_ALARMS: &str = "\
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

/// The id of the one-time consolidation import (a Rust-function migration).
pub(crate) const CONSOLIDATION_IMPORT_ID: &str = "consolidate_001_import_domain_stores";

/// The complete, strictly-linear catalog. **Order is application order.**
pub(crate) const MIGRATIONS: &[Migration] = &[
    // ── 0.4.2 baseline (head) ─────────────────────────────────────────
    Migration {
        id: "1",
        target: TargetDb::Core,
        body: MigrationBody::Sql(BASELINE_BOARD_TABLES),
    },
    Migration {
        id: "2",
        target: TargetDb::Core,
        body: MigrationBody::Sql(BASELINE_SESSION_TABLES),
    },
    Migration {
        id: "3",
        target: TargetDb::Core,
        body: MigrationBody::Sql(BASELINE_WORKSPACE_TABLES),
    },
    Migration {
        id: "4",
        target: TargetDb::Core,
        body: MigrationBody::Sql(BASELINE_USERS_TABLES),
    },
    Migration {
        id: "5",
        target: TargetDb::Core,
        body: MigrationBody::Sql(BASELINE_CONFIG_TABLES),
    },
    Migration {
        id: "6",
        target: TargetDb::Core,
        body: MigrationBody::Sql(BASELINE_CHAT_HISTORY_TABLES),
    },
    Migration {
        id: "7",
        target: TargetDb::Core,
        body: MigrationBody::Sql(BASELINE_CORE_INDEXES),
    },
    // ── 0.4.2→current deltas (existing deployed ids preserved) ─────────
    // Board deltas: drop the pre-job-centric reservation/assignment columns.
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
    // The non-terminal-ticket reset (`003_reset_nonterminal_tickets`) is moved
    // BELOW the consolidation import: it is a DATA reset, and only the imported
    // legacy tickets (copied into the consolidated DB by consolidate_001) need
    // it. On a current DB it is already-recorded and skipped, so live in-progress
    // tickets are never touched.
    // Session deltas: ticket_stage_jobs rework, then the job-per-phase rework.
    Migration {
        id: "003_drop_ticket_stage_jobs_round",
        target: TargetDb::Core,
        body: MigrationBody::Sql("ALTER TABLE ticket_stage_jobs DROP COLUMN round;"),
    },
    // Note: the historical `004_reset_implementation_jobs` /
    // `consolidate_004_discard_old_ticket_jobs` deletes are deliberately NOT
    // catalog entries. They ran on the (empty) pre-import `jobs` table, and the
    // post-import `13` cleanup deletes exactly the union of the kinds they
    // removed. Keeping them would be dead code — a recorded id replays
    // automatically via the id-only applied check, so their absence changes no
    // behavior and does not create a divergent schema path (the append-only
    // guarantee covers schema evolution, which this data-only reset does not).
    Migration {
        id: "005_drop_ticket_stage_jobs_phase",
        target: TargetDb::Core,
        body: MigrationBody::Sql("ALTER TABLE ticket_stage_jobs DROP COLUMN phase;"),
    },
    Migration {
        id: "006_rename_ticket_stage_jobs_to_ticket_jobs",
        target: TargetDb::Core,
        body: MigrationBody::Sql(
            "DROP TABLE IF EXISTS ticket_jobs; \
             ALTER TABLE ticket_stage_jobs RENAME TO ticket_jobs;",
        ),
    },
    Migration {
        id: "consolidate_002_drop_ticket_jobs_stage",
        target: TargetDb::Core,
        body: MigrationBody::Sql("ALTER TABLE ticket_jobs DROP COLUMN stage;"),
    },
    // `paused_frozen` is ADDed then DROPped (historical add/drop round-trip).
    // The add must precede the drop so a fresh 0.4.2 install never hits a
    // `DROP COLUMN` on an absent column.
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
    // `jobs.ticket_id` must exist before `idx_jobs_phase_ticket` runs — the
    // ALTER (consolidate_003) precedes this unique index in the catalog.
    Migration {
        id: "consolidate_006_drop_jobs_paused_frozen",
        target: TargetDb::Core,
        body: MigrationBody::Sql("ALTER TABLE jobs DROP COLUMN paused_frozen;"),
    },
    Migration {
        id: "consolidate_007_jobs_phase_ticket_index",
        target: TargetDb::Core,
        body: MigrationBody::Sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_phase_ticket \
             ON jobs(kind, ticket_id) WHERE ticket_id IS NOT NULL;",
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
    // ── Post-0.4.2 deltas that are new catalog entries (integer ids) ────
    Migration {
        id: "10",
        target: TargetDb::Core,
        body: MigrationBody::Sql(DELTA_DROP_CONFIG_ROLE),
    },
    Migration {
        id: "11",
        target: TargetDb::Core,
        body: MigrationBody::Sql(DELTA_FTS_INDEX),
    },
    Migration {
        id: "12",
        target: TargetDb::Core,
        body: MigrationBody::Sql(DELTA_BOARD_ACTIVE_INDEX),
    },
    // The one-time consolidation import: non-transactional (FK off/on bulk
    // load from the 6 legacy per-store files), re-runnable until applied, and
    // recorded only after it succeeds. Positioned AFTER the FTS/index entries
    // so imported tickets are already FTS-indexed, and BEFORE the post-import
    // reset + cleanup below (which act on the imported rows).
    Migration {
        id: CONSOLIDATION_IMPORT_ID,
        target: TargetDb::Core,
        body: MigrationBody::Rust(import_domain_stores),
    },
    // Reset non-terminal legacy tickets to backlog. MUST run AFTER the import —
    // the consolidated tickets table is empty until consolidate_001 fills it, so
    // this reset only has an effect on the imported rows. On a current DB it is
    // already-recorded and skipped.
    Migration {
        id: "003_reset_nonterminal_tickets",
        target: TargetDb::Core,
        body: MigrationBody::Sql(
            "UPDATE tickets SET phase = 'backlog' \
             WHERE phase NOT IN ('done','cancelled','failed') AND is_archived = 0;",
        ),
    },
    // Post-import cleanup: discard pre-rework ticket job rows the import just
    // rolled in. A DISTINCT id (not the earlier reset id), run with FK ON.
    // This id is a new integer id never recorded by the old runner, so it also
    // fires on a re-open of an already-current consolidated DB — a safe no-op
    // there, because the job-per-phase model never produces these legacy kinds
    // and the FK check is warn-only.
    Migration {
        id: "13",
        target: TargetDb::Core,
        body: MigrationBody::Rust(cleanup_legacy_ticket_jobs),
    },
    // Idempotent final-shape cleanups. The baseline re-creates 0.4.2 artifacts
    // (these are NEW integer ids, never recorded by the old runner); on an
    // already-current consolidated DB the drops/renames that removed them are
    // already-recorded and skipped, so these entries remove the leaked table /
    // column. They are safe no-ops on fresh and 0.4.2 installs.
    Migration {
        id: "15",
        target: TargetDb::Core,
        body: MigrationBody::Sql(DELTA_DROP_TICKET_STAGE_JOBS),
    },
    Migration {
        id: "16",
        target: TargetDb::Core,
        body: MigrationBody::Rust(drop_jobs_paused_frozen),
    },
    // ── 0.4.2 logs baseline + delta ────────────────────────────────────
    Migration {
        id: "8",
        target: TargetDb::Logs,
        body: MigrationBody::Sql(BASELINE_LOGS_TABLES),
    },
    Migration {
        id: "9",
        target: TargetDb::Logs,
        body: MigrationBody::Sql(BASELINE_LOGS_INDEXES),
    },
    Migration {
        id: "14",
        target: TargetDb::Logs,
        body: MigrationBody::Sql(DELTA_GREP_TELEMETRY),
    },
    Migration {
        id: "17",
        target: TargetDb::Core,
        body: MigrationBody::Sql(DELTA_ALARMS),
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

    for migration in MIGRATIONS {
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
                // is safe because every Rust body — the consolidation import,
                // the post-import cleanup, and the `jobs.paused_frozen` drop — is
                // idempotent and re-runnable, so a crash between body success and
                // id recording re-runs the body without corruption on the next boot.
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

fn import_domain_stores<'a>(
    conn: &'a Connection,
    root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
    Box::pin(run_consolidation_import(conn, root))
}

fn cleanup_legacy_ticket_jobs<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
    Box::pin(run_import_cleanup(conn))
}

fn drop_jobs_paused_frozen<'a>(
    conn: &'a Connection,
    _root: &'a Path,
) -> BoxFuture<'a, anyhow::Result<()>> {
    Box::pin(run_drop_jobs_paused_frozen(conn))
}

/// The one-time consolidation import: copy user tables from the six legacy
/// per-store files into the consolidated database. Non-transactional (FK
/// enforcement is OFF during the bulk load and ON afterwards), re-runnable
/// until applied.
async fn run_consolidation_import(conn: &Connection, root: &Path) -> anyhow::Result<()> {
    let any_legacy = crate::db::DOMAIN_STORE_NAMES
        .iter()
        .any(|name| crate::db::legacy_store_db_path(root, name).exists());
    if !any_legacy {
        return Ok(());
    }
    crate::db::import_legacy_stores(conn, root).await
}

/// The post-import cleanup: discard pre-rework ticket job rows (FK ON, so the
/// cascade fires) and report — never fail on — any orphan rows the bulk load
/// may have left behind.
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

/// Idempotently drop `jobs.paused_frozen`.
///
/// On a fresh/0.4.2 install the historical add/drop round-trip (`007_jobs_paused_frozen`
/// → `consolidate_006`) already removes the column, so this is a no-op. On an
/// already-current consolidated DB the catalog's baseline re-creates `jobs` without
/// the column, delta `007` re-adds it, but `consolidate_006` is already recorded
/// and skipped — leaving a dangling column. Turso has no `DROP COLUMN IF EXISTS`,
/// so the existence probe lives inside this single self-contained migration body
/// (idempotent cleanup, not a runner guard).
async fn run_drop_jobs_paused_frozen(conn: &Connection) -> anyhow::Result<()> {
    let has = conn
        .query("PRAGMA table_info(jobs)", ())
        .await
        .context("Failed to probe jobs.paused_frozen")?
        .into_iter()
        .any(|row| row.get::<String>(1).ok().as_deref() == Some("paused_frozen"));
    if has {
        conn.execute("ALTER TABLE jobs DROP COLUMN paused_frozen", ())
            .await
            .context("Failed to drop jobs.paused_frozen")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Connection;

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
             WHERE type = 'index' AND name NOT LIKE 'sqlite_autoindex_%'",
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
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
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

    /// User table names (excludes `sqlite_*` internal tables).
    async fn table_names(conn: &Connection) -> Vec<String> {
        conn.query(
            "SELECT name FROM sqlite_master \
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
            (),
        )
        .await
        .expect("read tables")
        .into_iter()
        .map(|row| row.get::<String>(0).expect("table name"))
        .collect()
    }

    // ── 0.4.2 backward-compat baseline (verbatim 0.4.2 SCHEMA DDL) ───────────
    //
    // The golden upgrade test seeds the legacy per-store files from these
    // constants, which are the EXACT `SCHEMA` DDL each store shipped at the
    // 0.4.2 baseline (the commit `4a68782` bump; the DDL is unchanged from its
    // parent, which is where these were transcribed from). They are
    // deliberately independent of the catalog's BASELINE_* constants so the
    // test catches a mistaken "reverse-engineered" baseline instead of
    // validating it against itself.
    const LEGACY_0_4_2_BOARD_SCHEMA: &str = r#"
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
CREATE INDEX IF NOT EXISTS idx_ticket_comments_ticket_id ON ticket_comments(ticket_id);
CREATE TABLE IF NOT EXISTS ticket_counters (
    workspace_name TEXT PRIMARY KEY,
    next_id        INTEGER NOT NULL DEFAULT 1
);
"#;

    const LEGACY_0_4_2_SESSIONS_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON sessions(agent_id, id);

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
CREATE INDEX IF NOT EXISTS idx_jobs_kind_status ON jobs(kind, status);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);

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
CREATE UNIQUE INDEX IF NOT EXISTS idx_agents_anchor ON agents(agent_id) WHERE job_id IS NULL;

CREATE TABLE IF NOT EXISTS pending_jobs (
    id              TEXT PRIMARY KEY,
    target_agent_id TEXT NOT NULL,
    envelope        TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_jobs_agent_created ON pending_jobs(target_agent_id, created_at);

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
);
"#;

    const LEGACY_0_4_2_WORKSPACES_SCHEMA: &str = r#"
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
CREATE UNIQUE INDEX IF NOT EXISTS workspace_contexts_null_role ON workspace_contexts(workspace_name) WHERE role IS NULL;
CREATE TABLE IF NOT EXISTS editor_tabs (
    workspace_name TEXT NOT NULL REFERENCES workspaces(name) ON DELETE CASCADE,
    file_path      TEXT NOT NULL,
    tab_order      INTEGER NOT NULL DEFAULT 0,
    is_active      INTEGER NOT NULL DEFAULT 0,
    is_dirty       INTEGER NOT NULL DEFAULT 0,
    dirty_content  TEXT,
    PRIMARY KEY (workspace_name, file_path)
);
"#;

    const LEGACY_0_4_2_USERS_SCHEMA: &str = r#"
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
);
"#;

    const LEGACY_0_4_2_CONFIG_SCHEMA: &str = r#"
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
);
"#;

    const LEGACY_0_4_2_CHAT_HISTORY_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    user_name TEXT NOT NULL,
    direction TEXT NOT NULL,
    content TEXT NOT NULL,
    agent_role TEXT,
    workspace TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_name);
CREATE INDEX IF NOT EXISTS idx_chat_history_workspace ON chat_history(workspace);
CREATE INDEX IF NOT EXISTS idx_chat_history_user_ws_id ON chat_history(user_name, workspace, id);
"#;

    const LEGACY_0_4_2_LOGS_SCHEMA: &str = r#"
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
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
CREATE INDEX IF NOT EXISTS idx_logs_target ON logs(target);
CREATE INDEX IF NOT EXISTS idx_logs_agent_role ON logs(agent_role);
CREATE INDEX IF NOT EXISTS idx_logs_agent_id ON logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_logs_workspace ON logs(workspace);

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
CREATE INDEX IF NOT EXISTS idx_tool_calls_agent_id ON tool_calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_role ON tool_calls(role);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_recorded_at ON tool_calls(recorded_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_workspace ON tool_calls(workspace);
CREATE INDEX IF NOT EXISTS idx_tool_calls_error_message ON tool_calls(error_message);

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
CREATE INDEX IF NOT EXISTS idx_llm_requests_recorded_at ON llm_requests(recorded_at);
CREATE INDEX IF NOT EXISTS idx_llm_requests_agent_id ON llm_requests(agent_id);
CREATE INDEX IF NOT EXISTS idx_llm_requests_model ON llm_requests(model);
CREATE INDEX IF NOT EXISTS idx_llm_requests_purpose ON llm_requests(purpose);
"#;

    /// A fresh 0.4.2-anchored install must converge exactly to the current
    /// shape: 0.4.2 tables are created, then the full delta chain is applied.
    #[tokio::test]
    async fn fresh_install_converges_to_current_shape() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_consolidated_store(tmp.path())
            .await
            .expect("fresh consolidated store");

        for table in [
            "tickets",
            "ticket_comments",
            "ticket_counters",
            "sessions",
            "session_metadata",
            "jobs",
            "agents",
            "pending_jobs",
            "research_jobs",
            "workspaces",
            "workspace_contexts",
            "editor_tabs",
            "users",
            "user_channels",
            "user_roles",
            "config_kv",
            "config_model_routing",
            "chat_history",
            "ticket_chronicle",
        ] {
            assert!(
                crate::db::table_exists(&conn, table).await.unwrap(),
                "missing {table}"
            );
        }
        for table in ["config_role", "ticket_jobs", "ticket_stage_jobs"] {
            assert!(
                !crate::db::table_exists(&conn, table).await.unwrap(),
                "{table} must be gone"
            );
        }

        let jobs_cols = column_names(&conn, "jobs").await;
        assert!(jobs_cols.contains(&"ticket_id".to_string()));
        assert!(!jobs_cols.contains(&"paused_frozen".to_string()));
        let tickets_cols = column_names(&conn, "tickets").await;
        assert!(!tickets_cols.contains(&"pipeline_reservation".to_string()));
        assert!(!tickets_cols.contains(&"assigned_to".to_string()));

        for id in [
            "001_drop_ticket_pipeline_reservation",
            "002_drop_ticket_assigned_to",
            "003_reset_nonterminal_tickets",
            "003_drop_ticket_stage_jobs_round",
            "005_drop_ticket_stage_jobs_phase",
            "006_rename_ticket_stage_jobs_to_ticket_jobs",
            "consolidate_002_drop_ticket_jobs_stage",
            "007_jobs_paused_frozen",
            "consolidate_003_jobs_ticket_id",
            "consolidate_005_drop_ticket_jobs",
            "consolidate_006_drop_jobs_paused_frozen",
            "consolidate_007_jobs_phase_ticket_index",
            "consolidate_008_create_ticket_chronicle",
            CONSOLIDATION_IMPORT_ID,
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "10",
            "11",
            "12",
            "13",
            "15",
            "16",
        ] {
            assert!(
                applied_ids(&conn).await.contains(&id.to_string()),
                "missing applied id {id}"
            );
        }
    }

    /// A fresh logs store runs the catalog and gets the current logs shape
    /// (logs/tool_calls/llm_requests + the `grep_telemetry` delta).
    #[tokio::test]
    async fn fresh_logs_install_runs_catalog() {
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

        for table in [
            "logs",
            "tool_calls",
            "llm_requests",
            "grep_telemetry",
            "schema_migrations",
        ] {
            assert!(
                crate::db::table_exists(&conn, table).await.unwrap(),
                "missing {table}"
            );
        }
        let applied = applied_ids(&conn).await;
        for id in ["8", "9", "14"] {
            assert!(applied.contains(&id.to_string()), "missing applied id {id}");
        }
    }

    /// The golden upgrade: bring a 0.4.2 per-store install (multi-file, per-store
    /// `SESSION_MIGRATIONS`) to the current consolidated shape without losing
    /// data. This is the one-time validation of the 0.4.2→current delta chain.
    ///
    /// The legacy stores are seeded from the [`LEGACY_0_4_2_*`] constants, copied
    /// verbatim from the `SCHEMA` DDL each store shipped at commit `4a68782`.
    /// These fixtures are independent of the catalog's [`BASELINE_*`] constants,
    /// so the test catches a mistaken "reverse-engineered" baseline instead of
    /// validating it against itself.
    #[tokio::test]
    async fn golden_upgrade_from_0_4_2_per_store() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let now = crate::db::now();

        let board = crate::db::open_with_schema(
            &crate::db::legacy_store_db_path(root, "board"),
            LEGACY_0_4_2_BOARD_SCHEMA,
        )
        .await
        .expect("legacy board");
        board
            .execute(
                "INSERT INTO tickets (id, title, description, phase, assigned_to, workspace_name, \
                 created_at, updated_at, pipeline_reservation, priority) \
                 VALUES ('T1', 'Done ticket', 'd', 'done', 'eng', 'ws', ?1, ?1, 0, 1)",
                params![now.clone()],
            )
            .await
            .unwrap();
        board
            .execute(
                "INSERT INTO tickets (id, title, description, phase, assigned_to, workspace_name, \
                 created_at, updated_at, pipeline_reservation, priority) \
                 VALUES ('T2', 'Wip ticket', 'd', 'in_development', 'eng', 'ws', ?1, ?1, 0, 1)",
                params![now.clone()],
            )
            .await
            .unwrap();
        board
            .execute(
                "INSERT INTO ticket_comments (id, ticket_id, role, content, created_at) \
                 VALUES ('C1', 'T1', 'manager', 'ship it', ?1)",
                params![now.clone()],
            )
            .await
            .unwrap();

        let sessions = crate::db::open_with_schema(
            &crate::db::legacy_store_db_path(root, "sessions"),
            LEGACY_0_4_2_SESSIONS_SCHEMA,
        )
        .await
        .expect("legacy sessions");
        sessions
            .execute(
                "INSERT INTO sessions (agent_id, role, content, created_at) \
                 VALUES ('sess1', 'assistant', 'hello', ?1)",
                params![now.clone()],
            )
            .await
            .unwrap();
        sessions
            .execute(
                "INSERT INTO session_metadata (agent_id, last_activity, role, message_count) \
                 VALUES ('sess1', ?1, 'assistant', 1)",
                params![now.clone()],
            )
            .await
            .unwrap();
        sessions
            .execute(
                "INSERT INTO jobs (id, kind, role, workspace_name, task, user_name, channel, \
                 retry_count, status, created_at, updated_at) \
                 VALUES ('job_research', 'research', '', 'ws', '', '', '', 0, 'launched', ?1, ?1)",
                params![now.clone()],
            )
            .await
            .unwrap();
        sessions
            .execute(
                "INSERT INTO jobs (id, kind, role, workspace_name, task, user_name, channel, \
                 retry_count, status, created_at, updated_at) \
                 VALUES ('job_stage', 'ticket_stage', '', 'ws', '', '', '', 0, 'launched', ?1, ?1)",
                params![now.clone()],
            )
            .await
            .unwrap();
        sessions
            .execute(
                "INSERT INTO ticket_stage_jobs (id, ticket_id, stage, phase, round) \
                 VALUES ('job_stage', 'T1', 'implementation', 'in_development', 1)",
                (),
            )
            .await
            .unwrap();

        let workspaces = crate::db::open_with_schema(
            &crate::db::legacy_store_db_path(root, "workspaces"),
            LEGACY_0_4_2_WORKSPACES_SCHEMA,
        )
        .await
        .expect("legacy workspaces");
        workspaces
            .execute(
                "INSERT INTO workspaces (name, path, created_at, updated_at) \
                 VALUES ('ws', '/ws', ?1, ?1)",
                params![now.clone()],
            )
            .await
            .unwrap();

        let users = crate::db::open_with_schema(
            &crate::db::legacy_store_db_path(root, "users"),
            LEGACY_0_4_2_USERS_SCHEMA,
        )
        .await
        .expect("legacy users");
        users
            .execute(
                "INSERT INTO users (name, selected_workspace) VALUES ('alice', 'ws')",
                (),
            )
            .await
            .unwrap();

        let config = crate::db::open_with_schema(
            &crate::db::legacy_store_db_path(root, "config"),
            LEGACY_0_4_2_CONFIG_SCHEMA,
        )
        .await
        .expect("legacy config");
        config
            .execute("INSERT INTO config_kv (key, value) VALUES ('k', 'v')", ())
            .await
            .unwrap();
        config
            .execute(
                "INSERT INTO config_role (role, model) VALUES ('manager', 'm')",
                (),
            )
            .await
            .unwrap();

        let chat_history = crate::db::open_with_schema(
            &crate::db::legacy_store_db_path(root, "chat_history"),
            LEGACY_0_4_2_CHAT_HISTORY_SCHEMA,
        )
        .await
        .expect("legacy chat_history");
        chat_history
            .execute(
                "INSERT INTO chat_history (message_id, user_name, direction, content, workspace) \
                 VALUES ('m1', 'alice', 'in', 'hi', 'ws')",
                (),
            )
            .await
            .unwrap();

        drop(board);
        drop(sessions);
        drop(workspaces);
        drop(users);
        drop(config);
        drop(chat_history);

        let conn = crate::db::open_consolidated_store(root)
            .await
            .expect("consolidate 0.4.2 install");

        assert_eq!(
            conn.query("SELECT count(*) FROM tickets WHERE id='T1'", ())
                .await
                .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "ticket survives"
        );
        assert_eq!(
            conn.query("SELECT count(*) FROM ticket_comments WHERE id='C1'", ())
                .await
                .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "ticket comment survives"
        );
        assert_eq!(
            conn.query("SELECT count(*) FROM sessions WHERE agent_id='sess1'", ())
                .await
                .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "session survives"
        );
        assert_eq!(
            conn.query(
                "SELECT count(*) FROM session_metadata WHERE agent_id='sess1'",
                ()
            )
            .await
            .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "session_metadata survives"
        );
        assert_eq!(
            conn.query("SELECT count(*) FROM workspaces WHERE name='ws'", ())
                .await
                .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "workspace survives"
        );
        assert_eq!(
            conn.query("SELECT count(*) FROM users WHERE name='alice'", ())
                .await
                .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "user survives"
        );
        assert_eq!(
            conn.query("SELECT count(*) FROM config_kv WHERE key='k'", ())
                .await
                .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "config_kv survives"
        );
        assert_eq!(
            conn.query(
                "SELECT count(*) FROM chat_history WHERE message_id='m1'",
                ()
            )
            .await
            .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "chat_history survives"
        );
        // The AUTOINCREMENT watermark is carried across the consolidation so
        // future `chat_history` inserts do not collide with the moved rows.
        assert_eq!(
            conn.query(
                "SELECT seq FROM sqlite_sequence WHERE name='chat_history'",
                ()
            )
            .await
            .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "AUTOINCREMENT watermark preserved after consolidation"
        );
        assert_eq!(
            conn.query("SELECT count(*) FROM jobs WHERE kind='research'", ())
                .await
                .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            1,
            "research job survives"
        );

        // The deliberate reset/drop contract holds on the imported rows: the
        // non-terminal legacy ticket is reset to backlog, and the pre-rework
        // `ticket_stage` job the import rolled in is dropped.
        assert_eq!(
            conn.query("SELECT phase FROM tickets WHERE id='T2'", ())
                .await
                .unwrap()[0]
                .get::<String>(0)
                .unwrap(),
            "backlog",
            "non-terminal legacy ticket reset to backlog"
        );
        assert_eq!(
            conn.query("SELECT count(*) FROM jobs WHERE kind='ticket_stage'", ())
                .await
                .unwrap()[0]
                .get::<i64>(0)
                .unwrap(),
            0,
            "pre-rework ticket_stage jobs dropped after import"
        );

        assert!(!crate::db::table_exists(&conn, "config_role").await.unwrap());
        assert!(!crate::db::table_exists(&conn, "ticket_jobs").await.unwrap());
        assert!(
            !crate::db::table_exists(&conn, "ticket_stage_jobs")
                .await
                .unwrap()
        );

        assert_eq!(
            conn.query("PRAGMA foreign_key_check", ())
                .await
                .unwrap()
                .len(),
            0,
            "no orphan rows after the import"
        );
    }

    /// A current-code consolidated DB — shaped by the pre-catalog SCHEMA +
    /// migration runner, holding ONLY the historical ids in `schema_migrations`
    /// — must CONVERGE when the catalog re-runs. The new integer baseline ids
    /// re-create 0.4.2 artifacts (`ticket_stage_jobs`, `jobs.paused_frozen`)
    /// that the already-recorded drops/renames then skip; the idempotent
    /// cleanup entries (`15`, `16`) remove them so the reopen leaves no stray
    /// table or column.
    #[tokio::test]
    async fn current_consolidated_db_reopen_converges() {
        // The ids the OLD per-store + consolidated runner recorded on a
        // current-main DB (from the pre-change src/db/migrations.rs). Note
        // `007_jobs_paused_frozen` is ABSENT — the old runner dropped the ADD
        // from its list at the job-per-phase refactor — while
        // `consolidate_006_drop_jobs_paused_frozen` is present.
        const OLD_DEPLOYED_IDS: &[&str] = &[
            "001_session_token_length",
            "002_session_message_count",
            "003_drop_ticket_stage_jobs_round",
            "004_reset_implementation_jobs",
            "005_drop_ticket_stage_jobs_phase",
            "006_rename_ticket_stage_jobs_to_ticket_jobs",
            "consolidate_002_drop_ticket_jobs_stage",
            "consolidate_003_jobs_ticket_id",
            "consolidate_004_discard_old_ticket_jobs",
            "consolidate_005_drop_ticket_jobs",
            "consolidate_006_drop_jobs_paused_frozen",
            "consolidate_007_jobs_phase_ticket_index",
            "001_drop_ticket_pipeline_reservation",
            "002_drop_ticket_assigned_to",
            "003_reset_nonterminal_tickets",
            "004_drop_tickets_review_base_count",
            "consolidate_008_create_ticket_chronicle",
            CONSOLIDATION_IMPORT_ID,
        ];

        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::CONSOLIDATED_DB_NAME),
            "",
        )
        .await
        .expect("open core");
        run_migrations(&conn, TargetDb::Core, root)
            .await
            .expect("fresh catalog");

        // Simulate a pre-change "current" DB: the same current shape, but
        // `schema_migrations` holds ONLY the historical ids.
        conn.execute("DELETE FROM schema_migrations", ())
            .await
            .unwrap();
        for id in OLD_DEPLOYED_IDS {
            conn.execute(
                "INSERT INTO schema_migrations (id, applied_at) VALUES (?1, ?2)",
                params![id, crate::db::now()],
            )
            .await
            .unwrap();
        }

        // Reopen: the catalog re-runs on the existing current-shape DB.
        run_migrations(&conn, TargetDb::Core, root)
            .await
            .expect("reopen catalog");

        assert!(
            !crate::db::table_exists(&conn, "ticket_stage_jobs")
                .await
                .unwrap()
        );
        assert!(!crate::db::table_exists(&conn, "config_role").await.unwrap());
        let jobs_cols = column_names(&conn, "jobs").await;
        assert!(jobs_cols.contains(&"ticket_id".to_string()));
        assert!(!jobs_cols.contains(&"paused_frozen".to_string()));
        let tickets_cols = column_names(&conn, "tickets").await;
        assert!(!tickets_cols.contains(&"pipeline_reservation".to_string()));
        assert!(!tickets_cols.contains(&"assigned_to".to_string()));

        let applied = applied_ids(&conn).await;
        for id in ["15", "16"] {
            assert!(applied.contains(&id.to_string()), "missing applied id {id}");
        }
    }

    /// The catalog's 0.4.2 baseline must be structurally identical to the
    /// verbatim per-store `SCHEMA` fixtures (including the `SESSION_MIGRATIONS`
    /// 001/002 effects, already folded into `session_metadata`). This is the
    /// "double-check all schemas" verification requested on the ticket: each
    /// legacy store's full per-table DDL (which pins columns, ordering,
    /// defaults, PK, inline UNIQUE, CHECK, and FK clauses) and explicit index
    /// definitions (SQL/uniqueness) are compared against the consolidated
    /// baseline, in BOTH directions (so a baseline-extra table/index is caught).
    /// The fixtures are independently byte-verified against commit `24c92d2`, so
    /// this test validates the baseline structure and the delta chain, not the
    /// fixtures themselves.
    #[tokio::test]
    async fn catalog_baseline_matches_verbatim_0_4_2_schema() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();

        // Build the consolidated 0.4.2 anchor from ONLY the BASELINE_* DDL
        // (not the delta chain), so it represents the baseline shape.
        let core = crate::db::open_with_schema(&root.join("baseline-core.db"), "")
            .await
            .expect("open baseline core");
        for sql in [
            BASELINE_BOARD_TABLES,
            BASELINE_SESSION_TABLES,
            BASELINE_WORKSPACE_TABLES,
            BASELINE_USERS_TABLES,
            BASELINE_CONFIG_TABLES,
            BASELINE_CHAT_HISTORY_TABLES,
            BASELINE_CORE_INDEXES,
        ] {
            core.execute_batch(sql).await.expect("apply baseline core");
        }
        let logs = crate::db::open_with_schema(&root.join("baseline-logs.db"), "")
            .await
            .expect("open baseline logs");
        logs.execute_batch(BASELINE_LOGS_TABLES)
            .await
            .expect("apply baseline logs tables");
        logs.execute_batch(BASELINE_LOGS_INDEXES)
            .await
            .expect("apply baseline logs indexes");

        // Core: every legacy 0.4.2 per-store table/index must appear in the
        // consolidated baseline with the exact definition — and the baseline
        // must not carry a table/index that 0.4.2 did not have. The full-DDL
        // `table_defs` comparison pins columns, ordering, defaults, PK, inline
        // UNIQUE, CHECK, and FK clauses in one shot; the index comparison pins
        // index SQL/uniqueness.
        let mut legacy_core_tables: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut legacy_core_indexes: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
        let mut legacy_core_table_defs: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::new();
        for (store, schema) in [
            ("board", LEGACY_0_4_2_BOARD_SCHEMA),
            ("sessions", LEGACY_0_4_2_SESSIONS_SCHEMA),
            ("workspaces", LEGACY_0_4_2_WORKSPACES_SCHEMA),
            ("users", LEGACY_0_4_2_USERS_SCHEMA),
            ("config", LEGACY_0_4_2_CONFIG_SCHEMA),
            ("chat_history", LEGACY_0_4_2_CHAT_HISTORY_SCHEMA),
        ] {
            let legacy =
                crate::db::open_with_schema(&crate::db::legacy_store_db_path(root, store), schema)
                    .await
                    .expect("open legacy store");
            for table in table_names(&legacy).await {
                assert!(
                    legacy_core_tables.insert(table.clone()),
                    "duplicate table {table} across legacy 0.4.2 stores"
                );
            }
            for (name, sql) in index_defs(&legacy).await {
                assert!(
                    legacy_core_indexes.insert(name.clone(), sql).is_none(),
                    "duplicate index {name} across legacy 0.4.2 stores"
                );
            }
            for t in table_defs(&legacy).await {
                assert!(
                    legacy_core_table_defs.insert(t.0, t.1).is_none(),
                    "duplicate table DDL across legacy 0.4.2 stores"
                );
            }
        }
        let core_tables: std::collections::HashSet<String> =
            table_names(&core).await.into_iter().collect();
        assert_eq!(
            core_tables, legacy_core_tables,
            "baseline core tables diverge from the 0.4.2 per-store schemas",
        );
        assert_eq!(
            index_defs(&core).await,
            legacy_core_indexes,
            "baseline core indexes diverge from the 0.4.2 per-store schemas",
        );
        assert_eq!(
            table_defs(&core).await,
            legacy_core_table_defs,
            "baseline core table DDL diverges from the 0.4.2 per-store schemas",
        );

        // Logs: the 0.4.2 logs store must be captured by the logs baseline,
        // bidirectionally.
        let legacy_logs =
            crate::db::open_with_schema(&root.join("legacy-logs.db"), LEGACY_0_4_2_LOGS_SCHEMA)
                .await
                .expect("open legacy logs");
        let legacy_logs_tables: std::collections::HashSet<String> =
            table_names(&legacy_logs).await.into_iter().collect();
        let logs_tables: std::collections::HashSet<String> =
            table_names(&logs).await.into_iter().collect();
        assert_eq!(
            logs_tables, legacy_logs_tables,
            "baseline logs tables diverge from the 0.4.2 logs schema",
        );
        assert_eq!(
            index_defs(&logs).await,
            index_defs(&legacy_logs).await,
            "baseline logs indexes diverge from the 0.4.2 logs schema",
        );
        assert_eq!(
            table_defs(&logs).await,
            table_defs(&legacy_logs).await,
            "baseline logs table DDL diverges from the 0.4.2 logs schema",
        );
    }

    /// A 0.4.2-era logs store (no `schema_migrations`, no `grep_telemetry`)
    /// upgraded by the catalog: the baseline entries no-op on the already-created
    /// tables and the sole logs delta (`grep_telemetry`) is added, with existing
    /// log rows preserved. This covers the 0.4.2→current logs upgrade path.
    #[tokio::test]
    async fn logs_upgrade_from_0_4_2_adds_grep_telemetry() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let conn = crate::db::open_with_schema(
            &crate::db::store_db_path(root, crate::db::LOG_DB_NAME),
            LEGACY_0_4_2_LOGS_SCHEMA,
        )
        .await
        .expect("open legacy logs");
        conn.execute(
            "INSERT INTO logs (timestamp, level, target, message) \
             VALUES (?1, 'INFO', 'test', 'hello')",
            params![crate::db::now()],
        )
        .await
        .unwrap();

        run_migrations(&conn, TargetDb::Logs, root)
            .await
            .expect("upgrade logs catalog");

        assert!(
            crate::db::table_exists(&conn, "grep_telemetry")
                .await
                .unwrap()
        );
        assert!(
            crate::db::table_exists(&conn, "llm_requests")
                .await
                .unwrap()
        );
        let count = conn
            .query("SELECT COUNT(*) FROM logs WHERE message = 'hello'", ())
            .await
            .expect("count logs")
            .into_iter()
            .next()
            .map(|row| row.get::<i64>(0).expect("count"))
            .unwrap_or(0);
        assert_eq!(count, 1, "existing logs rows must survive the logs upgrade");
        let applied = applied_ids(&conn).await;
        for id in ["8", "9", "14"] {
            assert!(applied.contains(&id.to_string()), "missing applied id {id}");
        }
    }
}
