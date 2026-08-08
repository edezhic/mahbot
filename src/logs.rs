//! Turso-backed log storage.
//!
//! Each log entry is inserted asynchronously via a background channel task.
//! A broadcast channel feeds live log entries to the Iced native GUI dashboard.

use crate::turso;
use crate::util::UnwrapPoison;
use crate::util::json;
use anyhow::Context;
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use std::io;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::OnceCell;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::warn;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
use turso::{Row, Value, params};

/// Schema for a single log entry stored in Turso.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub target: String,
    pub message: String,
    #[serde(default)]
    pub fields: serde_json::Value,
    #[serde(default)]
    pub agent_id: String,
    #[serde(default)]
    pub agent_role: String,
    #[serde(default)]
    pub workspace: String,
}

// Column definitions for `logs` SELECT queries.
crate::columns! {
    LOGS_COLUMNS [LOGS] {
        TIMESTAMP   => "timestamp",
        LEVEL       => "level",
        TARGET      => "target",
        MESSAGE     => "message",
        FIELDS      => "fields",
        AGENT_ID    => "agent_id",
        AGENT_ROLE  => "agent_role",
        WORKSPACE   => "workspace",
    }
}

/// Turso-backed log store.
///
/// NOTE: This store does NOT use `define_store!` or `global_store!`. The store
/// is opened manually inside [`init_tracing()`] because bootstrapping order
/// requires logs to be available before other stores are initialized. See
/// [`LOG_STORE`] for details.
#[derive(Clone, Debug)]
pub struct LogStore {
    pub(crate) conn: crate::turso::Connection,
}

/// Global log store, set during [`init_tracing()`].
///
/// # Access model
///
/// This store is initialized inside [`init_tracing()`] — it does NOT have an
/// `init_global()` like other stores. Do NOT add one. Calling `init_tracing()`
/// already opens `logs.db`. A second open via `init_global()` would create a
/// second connection to the same database, causing `.tshm` coordination
/// conflicts between the two connections.
///
/// In addition to this global, [`crate::gui::BOOT_LOG_STORE`] holds another clone of
/// the same `LogStore`, and [`init_tracing()`] returns a third `Arc<LogStore>`
/// to its caller. All three point to the same underlying connection (which
/// is cheaply cloneable since `Connection` wraps an `Arc` internally).
pub static LOG_STORE: OnceCell<LogStore> = OnceCell::const_new();

const LOGS_SCHEMA: &str = "\
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
-- Consolidated tool-call / retry-failure stats (formerly stats.db). Both the
-- normal open path and the quarantine-recreate branch execute this schema, so
-- a quarantine silently recreates the stats tables too.
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
CREATE TABLE IF NOT EXISTS retry_failures (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt           INTEGER NOT NULL,
    failure_class     TEXT NOT NULL,
    error_chain       TEXT NOT NULL,
    http_version      TEXT,
    content_length    INTEGER,
    actual_body_len   INTEGER,
    content_encoding  TEXT,
    transfer_encoding TEXT,
    elapsed_ms        INTEGER NOT NULL,
    body_head         TEXT NOT NULL DEFAULT '',
    body_tail         TEXT NOT NULL DEFAULT '',
    finish_reason     TEXT,
    completion_tokens INTEGER,
    retry_after_ms    INTEGER,
    recorded_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retry_failures_recorded_at ON retry_failures(recorded_at);
CREATE INDEX IF NOT EXISTS idx_retry_failures_class ON retry_failures(failure_class);
-- Per-operation LLM request stats (all purposes: agent runs, verdict
-- extraction, summarization, consolidation). Metadata only — no request
-- inputs/outputs are stored. Auto-created on existing databases at next
-- store open (CREATE TABLE IF NOT EXISTS), including quarantine recreation.
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
    success             INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_llm_requests_recorded_at ON llm_requests(recorded_at);
CREATE INDEX IF NOT EXISTS idx_llm_requests_agent_id ON llm_requests(agent_id);
CREATE INDEX IF NOT EXISTS idx_llm_requests_model ON llm_requests(model);
CREATE INDEX IF NOT EXISTS idx_llm_requests_purpose ON llm_requests(purpose);
-- Per-call image-generation stats, recorded at call time by the image_gen
-- tool so records survive agent cancel/crash (the session-finalize flush
-- drops stats when an agent is dropped without finalization).
CREATE TABLE IF NOT EXISTS image_gen_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at   TEXT NOT NULL,
    model         TEXT NOT NULL,
    workspace     TEXT NOT NULL DEFAULT '',
    duration_ms   INTEGER NOT NULL,
    success       INTEGER NOT NULL DEFAULT 1,
    attempts      INTEGER NOT NULL DEFAULT 1,
    failure_class TEXT,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_image_gen_calls_recorded_at ON image_gen_calls(recorded_at);
CREATE INDEX IF NOT EXISTS idx_image_gen_calls_model ON image_gen_calls(model);
-- Shadow-instrumentation rows for the joint-verdict pipeline: per-agent
-- verdict scores per stage round, so inter-agent agreement can be
-- re-measured and the dynamic-count formula validated after launch.
CREATE TABLE IF NOT EXISTS verdict_scores (
    id          TEXT PRIMARY KEY,
    ticket_id   TEXT NOT NULL,
    stage       TEXT NOT NULL,
    agent_index INTEGER NOT NULL,
    score       INTEGER NOT NULL,
    issues      TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verdict_scores_ticket_id ON verdict_scores(ticket_id);
CREATE INDEX IF NOT EXISTS idx_verdict_scores_created_at ON verdict_scores(created_at);";

impl LogStore {
    /// Open (or create) the log database at `root/db/logs.db`.
    ///
    /// `pub(crate)` (matching every other store's generated `open`) so tests in
    /// other modules can create a real log store via [`crate::open_test_store!`].
    ///
    /// Boot-time quarantine: if the existing store fails integrity verification
    /// (corruption-class `quick_check` output, or the open/verify path
    /// panicking — opening a corrupt store under multiprocess_wal can panic),
    /// the whole artifact family (database plus `-wal`/`-shm`/`-tshm`
    /// sidecars) is moved aside to a timestamped quarantine name and a fresh
    /// store is created. Logs-only by construction: this lives in the logs
    /// open path and is never reachable from the shared store helpers. Boot
    /// never fails because of the quarantine mechanism — rename failures are
    /// logged and the store is recreated (or, in the extreme case, the
    /// pre-quarantine error is surfaced).
    pub(crate) async fn open(root: &Path) -> anyhow::Result<Self> {
        match open_verified_logs_store(root).await {
            Ok(conn) => Ok(Self { conn }),
            Err(OpenFailure::Corrupt(reason)) => {
                warn!(
                    error = %reason,
                    "logs store failed integrity verification — quarantining artifact family \
                     and recreating a fresh store",
                );
                quarantine_logs_artifacts(root);
                let conn = crate::turso::open_store(root, "logs", LOGS_SCHEMA)
                    .await
                    .context("Failed to recreate logs store after quarantine")?;
                Ok(Self { conn })
            }
            Err(OpenFailure::Other(e)) => Err(e),
        }
    }

    /// Insert a batch of log entries in a single transaction.
    ///
    /// Each entry would otherwise commit (and fsync the WAL) individually —
    /// in WAL mode the per-commit fsync dominates the insert cost. Batching
    /// the diagnostics logs into one transaction reduces N commits to one.
    /// On failure the whole batch is dropped (the caller's [`spawn_log_writer`]
    /// clears it regardless) — matching the existing drop-on-failure semantics;
    /// log entries are diagnostics, not durable state.
    async fn insert_batch(&self, entries: &[LogEntry]) -> anyhow::Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .begin_tx()
            .await
            .context("Failed to begin log insert transaction")?;
        for entry in entries {
            tx.execute(
                "INSERT INTO logs (timestamp, level, target, message, fields, agent_id, agent_role, workspace) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    entry.timestamp.clone(),
                    entry.level.clone(),
                    entry.target.clone(),
                    entry.message.clone(),
                    serde_json::to_string(&entry.fields)
                        .expect("log entry fields serialization failed; this should not happen"),
                    entry.agent_id.clone(),
                    entry.agent_role.clone(),
                    entry.workspace.clone(),
                ],
            )
            .await
            .context("Failed to insert log entry in batch")?;
        }
        tx.commit()
            .await
            .context("Failed to commit log insert transaction")?;
        Ok(())
    }

    /// Build the data query with LIKE-based search using bind params.
    fn build_query(filters: &LogQuery) -> (String, Vec<Value>) {
        let (where_sql, mut values) = build_where_clause(filters);

        let limit: i64 = i64::try_from(filters.limit.unwrap_or(100).min(1000))
            .expect("log query limit overflowed i64; limit must be <= i64::MAX");
        let offset: i64 = i64::try_from(filters.offset.unwrap_or(0))
            .expect("log query offset overflowed i64; offset must be <= i64::MAX");
        values.push(Value::Integer(limit));
        values.push(Value::Integer(offset));

        let sql = format!(
            "SELECT {LOGS_COLUMNS} FROM logs {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
        );

        (sql, values)
    }

    /// Count matching rows.
    async fn count_matching(&self, filters: &LogQuery) -> Result<usize, ::turso::Error> {
        let (where_sql, values) = build_where_clause(filters);
        let sql = format!("SELECT COUNT(*) FROM logs {where_sql}");
        self.conn
            .query_row(&sql, values, |row| row.get::<i64>(0))
            .await
            .map(|n| usize::try_from(n).unwrap_or(0))
    }

    /// Delete log entries matching a given `level` whose `timestamp` is older than the given
    /// RFC 3339 `cutoff`. Returns the number of deleted rows.
    pub async fn delete_older_than(&self, level: &str, cutoff: &str) -> anyhow::Result<u64> {
        let n = self
            .conn
            .execute(
                "DELETE FROM logs WHERE level = ?1 AND timestamp < ?2",
                params![level, cutoff],
            )
            .await
            .context("Failed to delete old log entries")?;
        Ok(n)
    }

    /// Persist a single per-agent verdict score (shadow instrumentation for
    /// the joint-verdict pipeline). Best-effort — callers log and swallow
    /// failures; the verdict pipeline must never block on instrumentation.
    /// Written outside the comment+transition transaction so the store write
    /// lock is never held for it.
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) async fn record_verdict_score(
        &self,
        ticket_id: &str,
        stage: &str,
        agent_index: usize,
        score: u8,
        issues: &[String],
    ) -> anyhow::Result<()> {
        let id = format!(
            "{ticket_id}_{stage}_{}_{agent_index}",
            crate::generate_suffix()
        );
        let now = crate::turso::now();
        let issues_json = serde_json::to_string(issues)?;
        self.conn
            .execute(
                "INSERT INTO verdict_scores (id, ticket_id, stage, agent_index, score, issues, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    id,
                    ticket_id,
                    stage,
                    agent_index as i64,
                    i64::from(score),
                    issues_json,
                    now,
                ],
            )
            .await
            .context("Failed to record verdict score")?;
        Ok(())
    }

    /// Query log entries with optional filters.
    ///
    /// Uses LIKE-based search on target and message columns.
    ///
    /// Returns `(entries, total_count)` where `entries` respects pagination
    /// and `total_count` is the total number of entries matching the same filters.
    pub async fn query(&self, filters: &LogQuery) -> anyhow::Result<(Vec<LogEntry>, usize)> {
        let total = self.count_matching(filters).await?;
        if total == 0 {
            return Ok((vec![], 0));
        }

        let (data_sql, data_values) = Self::build_query(filters);
        let rows = self
            .conn
            .query(&data_sql, data_values)
            .await
            .context("Data query failed")?;

        let mut entries = Vec::new();
        for row in rows {
            entries.push(log_entry_from_row(&row)?);
        }

        Ok((entries, total))
    }
}

/// Outcome of opening + verifying the logs store at boot.
enum OpenFailure {
    /// Quarantine-worthy: integrity verification failed, the open failed with
    /// a corruption-class error, or the open/verify path panicked (opening a
    /// corrupt store under multiprocess_wal can panic).
    Corrupt(String),
    /// Non-quarantine failures (busy/locked/IO) — propagate unchanged.
    Other(anyhow::Error),
}

/// Open the logs store and verify its integrity.
///
/// On a corruption-class failure the connection is dropped before returning so
/// the caller can rename the artifact family (POSIX open-file rename
/// semantics) and recreate the store — the recreated store must be the one
/// registered in `LOG_STORE`/`iter_checkpoint_stores`.
///
/// The open itself is panic-absorbed: opening a corrupt store under
/// multiprocess_wal can panic (e.g. the shared-WAL frame-index invariant), and
/// a boot-time panic here must quarantine rather than crash startup.
async fn open_verified_logs_store(root: &Path) -> Result<crate::turso::Connection, OpenFailure> {
    let open = AssertUnwindSafe(crate::turso::open_store(root, "logs", LOGS_SCHEMA))
        .catch_unwind()
        .await;
    let conn = match open {
        Ok(Ok(conn)) => conn,
        Ok(Err(e)) => {
            let db_path = turso::store_db_path(root, "logs");
            // A store that exists but cannot be opened at all is corrupt —
            // quarantine so boot can proceed with a fresh store. A missing
            // file (first boot) opens fine, so this path implies an existing
            // file that is unreadable. Busy/locked/IO-class open errors
            // (disk full, transient permission) never quarantine — route
            // through the same classifier as the quick_check path.
            if db_path.exists() && is_corruption_class(&e) {
                return Err(OpenFailure::Corrupt(format!("open failed: {e:#}")));
            }
            return Err(OpenFailure::Other(e));
        }
        Err(payload) => {
            return Err(OpenFailure::Corrupt(format!(
                "open panicked: {}",
                crate::util::panic_message(&*payload)
            )));
        }
    };
    let verify = AssertUnwindSafe(conn.quick_check()).catch_unwind().await;
    match verify {
        Ok(Ok(())) => Ok(conn),
        Ok(Err(e)) if is_corruption_class(&e) => Err(OpenFailure::Corrupt(format!("{e:#}"))),
        Ok(Err(e)) => Err(OpenFailure::Other(e)),
        Err(payload) => Err(OpenFailure::Corrupt(format!(
            "integrity check panicked: {}",
            crate::util::panic_message(&*payload)
        ))),
    }
}

/// True when a `quick_check` failure is corruption-class rather than a
/// busy/locked/IO failure of the PRAGMA itself.
///
/// Two forms qualify: the PRAGMA returned a non-`ok` row (our
/// `Database integrity check failed` bail), or the PRAGMA failed to execute
/// with a message that is not a lock/busy or I/O condition (e.g. a page-level
/// error reading a zeroed page — `Invalid page type`). Only corruption-class
/// failures quarantine — a busy or locked store must never trigger the rename
/// path.
///
/// Note: this is a negation-based substring heuristic. A corruption message
/// that happens to contain `busy`/`locked` would be misclassified as
/// non-corruption (boot fails rather than quarantines), and an I/O error
/// lacking the expected tokens could false-quarantine. The open-failure path
/// has a sharper inversion: a corrupt store that fails to OPEN with an
/// I/O-flavored message (e.g. a zeroed page read surfacing as `i/o error`)
/// is classified non-corruption and boot fails without quarantine — the
/// opposite of the intended resilience. Conversely an error lacking the
/// tokens (e.g. `No space left on device`, `too many open files`) false-
/// quarantines a healthy store. Bounded to the logs-only boot path (the
/// primary target — `Invalid page type` from zeroed pages — is caught);
/// accepted as pragmatic.
fn is_corruption_class(e: &anyhow::Error) -> bool {
    let msg = format!("{e:#}");
    if msg.contains("Database integrity check failed") {
        return true;
    }
    let lower = msg.to_lowercase();
    !(lower.contains("busy")
        || lower.contains("locked")
        || lower.contains("i/o error")
        || lower.contains("no such file")
        || lower.contains("permission denied"))
}

/// Move the logs store's whole artifact family aside to a timestamped
/// quarantine name. Best-effort: a rename failure is logged, never fatal.
///
/// The timestamp is second-granularity and `rename` would silently overwrite
/// an existing destination, so a monotonically increasing sequence suffix
/// guarantees uniqueness across multiple quarantines within the same second.
fn quarantine_logs_artifacts(root: &Path) {
    static QUARANTINE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let db_path = turso::store_db_path(root, "logs");
    let sidecars = turso::store_sidecars(&db_path);
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let seq = QUARANTINE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let base = if seq == 0 {
        format!("logs.db.quarantine-{stamp}")
    } else {
        format!("logs.db.quarantine-{stamp}-{seq}")
    };
    for (src, suffix) in [
        (&db_path, ""),
        (&sidecars.wal, "-wal"),
        (&sidecars.shm, "-shm"),
        (&sidecars.tshm, "-tshm"),
    ] {
        if !src.exists() {
            continue;
        }
        let dst = db_path.with_file_name(format!("{base}{suffix}"));
        if let Err(e) = std::fs::rename(src, &dst) {
            warn!(
                error = %e,
                from = %src.display(),
                to = %dst.display(),
                "Failed to quarantine corrupt logs store artifact",
            );
        }
    }
}

/// Parameters for filtering log queries.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LogQuery {
    pub level: Option<String>,
    pub target: Option<String>,
    pub search: Option<String>,
    pub since: Option<String>,
    pub until: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub agent_id: Option<String>,
    pub agent_role: Option<String>,
    pub workspace: Option<String>,
}

// ── Shared helpers ───────────────────────────────────────────────────────────

/// Build WHERE clause and bind values from `LogQuery` filters.
/// Returns `(WHERE ...`, `[values]`) — an empty string when no filters are set.
fn build_where_clause(filters: &LogQuery) -> (String, Vec<Value>) {
    let mut conditions: Vec<String> = Vec::new();
    let mut values: Vec<Value> = Vec::new();

    if let Some(ref levels_str) = filters.level
        && !levels_str.is_empty()
    {
        let levels: Vec<Value> = levels_str
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| Value::Text(s.to_string()))
            .collect();
        if !levels.is_empty() {
            conditions.push(format!(
                "level IN ({})",
                turso::sql_in_placeholders(levels.len()),
            ));
            values.extend(levels);
        }
    }

    if let Some(ref target) = filters.target {
        conditions.push("target LIKE ?".into());
        values.push(Value::Text(format!("{target}%")));
    }

    if let Some(ref search) = filters.search
        && !search.is_empty()
    {
        let val = Value::Text(format!("%{search}%"));
        conditions.push("(target LIKE ? OR message LIKE ?)".into());
        values.push(val.clone());
        values.push(val);
    }

    if let Some(ref since) = filters.since {
        conditions.push("timestamp >= ?".into());
        values.push(Value::Text(since.clone()));
    }

    if let Some(ref until) = filters.until {
        conditions.push("timestamp <= ?".into());
        values.push(Value::Text(until.clone()));
    }

    if let Some(ref agent_id) = filters.agent_id {
        conditions.push("agent_id LIKE ?".into());
        values.push(Value::Text(format!("%{agent_id}%")));
    }

    if let Some(ref agent_role) = filters.agent_role {
        conditions.push("agent_role = ?".into());
        values.push(Value::Text(agent_role.clone()));
    }

    if let Some(ref workspace) = filters.workspace
        && !workspace.is_empty()
    {
        conditions.push("workspace = ?".into());
        values.push(Value::Text(workspace.clone()));
    }

    if conditions.is_empty() {
        (String::new(), values)
    } else {
        (format!("WHERE {}", conditions.join(" AND ")), values)
    }
}

fn log_entry_from_row(row: &Row) -> anyhow::Result<LogEntry> {
    let timestamp = row.get::<String>(COL_LOGS_TIMESTAMP)?;
    let level = row.get::<String>(COL_LOGS_LEVEL)?;
    let target = row.get::<String>(COL_LOGS_TARGET)?;
    let message = row.get::<String>(COL_LOGS_MESSAGE)?;
    let fields_str = row.get::<String>(COL_LOGS_FIELDS)?;
    let fields: serde_json::Value =
        serde_json::from_str(&fields_str).unwrap_or(serde_json::Value::Null);

    let agent_id = row.get::<String>(COL_LOGS_AGENT_ID)?;
    let agent_role = row.get::<String>(COL_LOGS_AGENT_ROLE)?;
    let workspace = row.get::<String>(COL_LOGS_WORKSPACE)?;

    Ok(LogEntry {
        timestamp,
        level,
        target,
        message,
        fields,
        agent_id,
        agent_role,
        workspace,
    })
}

// ── Tracing initialization ──────────────────────────────────────────

/// Initialize tracing: JSON to Turso store only (no terminal output).
/// Returns the [`LogStore`] for querying and a broadcast sender
/// for live log streaming to the Iced native GUI dashboard.
pub async fn init_tracing(
    storage_root: &Path,
) -> anyhow::Result<(Arc<LogStore>, tokio::sync::broadcast::Sender<String>)> {
    let store = LogStore::open(storage_root).await?;
    LOG_STORE
        .set(store.clone())
        .map_err(|_| anyhow::anyhow!("LOG_STORE already initialized"))?;
    let log_store = Arc::new(store);
    let (log_tx, log_rx) = tokio::sync::mpsc::unbounded_channel();
    let (broadcast_tx, _) = tokio::sync::broadcast::channel(256);

    spawn_log_writer(Arc::clone(&log_store), log_rx, broadcast_tx.clone());

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "info,turso_core=warn,tantivy=warn,ort=warn,fff_search=warn,fff_search::grep=error",
        )
    });

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            fmt::Layer::new()
                .json()
                .with_writer(make_log_writer(log_tx))
                .with_ansi(false),
        )
        .init();

    Ok((log_store, broadcast_tx))
}

// ── Tracing integration ──────────────────────────────────────────

/// A [`MakeWriter`] that sends JSON log lines over an unbounded channel.
const fn make_log_writer(tx: UnboundedSender<String>) -> LogWriter {
    LogWriter { tx }
}

#[derive(Clone)]
struct LogWriter {
    tx: UnboundedSender<String>,
}

impl io::Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let line = String::from_utf8_lossy(buf).to_string();
        let _ = self.tx.send(line);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl MakeWriter<'_> for LogWriter {
    type Writer = Self;

    fn make_writer(&self) -> Self::Writer {
        self.clone()
    }
}

/// Maximum number of log entries accumulated before a forced DB flush.
///
/// Bounds the write-lock hold: one flush inserts at most this many rows in a
/// single transaction.
const LOG_BATCH_MAX: usize = 50;

/// Maximum age of an accumulated batch before a timer flush.
///
/// Keeps the DB insert path fresh under low log volume while the GUI live
/// broadcast (which stays per-message, unbuffered) is unaffected.
const LOG_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

/// Spawn a background task that receives JSON log lines and writes them to Turso
/// and broadcasts them over the channel to the Iced GUI dashboard.
///
/// The GUI live broadcast stays per-message (no GUI lag). Only the DB-insert
/// path batches: entries accumulate in [`LOG_BATCH_MAX`]-sized batches flushed
/// by [`LOG_FLUSH_INTERVAL`], on batch-cap, or on channel close (flush-before-
/// shutdown). A crash mid-batch loses at most [`LOG_BATCH_MAX`] entries —
/// acceptable, since DB log inserts are diagnostics, not durable state.
///
/// A storage-layer panic (e.g. the shared-WAL frame-index invariant violation)
/// must not silently freeze persistence: the flush is panic-absorbed with a
/// bounded number of restarts and backoff, recorded on the visible failure
/// surface, and past the bound the writer stops flushing with a terminal banner
/// instead of spinning on a broken connection. During backoff sleeps the writer
/// is not polling the channel, so lines accumulate in the unbounded channel —
/// bounded by the backoff schedule, drained once flushing resumes. In the
/// terminal stopped state the writer keeps draining the channel (broadcast +
/// drop), so the channel does not grow indefinitely.
fn spawn_log_writer(
    store: Arc<LogStore>,
    rx: UnboundedReceiver<String>,
    broadcast: tokio::sync::broadcast::Sender<String>,
) {
    spawn_log_writer_with_interval(store, rx, broadcast, LOG_FLUSH_INTERVAL);
}

/// [`spawn_log_writer`] with an explicit flush interval — tests inject a
/// non-production interval (very long, or very short) to exercise the
/// batch-cap and timer flush paths without racing the production timer.
fn spawn_log_writer_with_interval(
    store: Arc<LogStore>,
    mut rx: UnboundedReceiver<String>,
    broadcast: tokio::sync::broadcast::Sender<String>,
    flush_interval: std::time::Duration,
) {
    tokio::spawn(async move {
        let mut batch: Vec<LogEntry> = Vec::new();
        let mut flush_timer = tokio::time::interval(flush_interval);
        flush_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first interval tick fires immediately — consume it so the timer
        // only fires after the first full interval.
        flush_timer.tick().await;

        loop {
            tokio::select! {
                maybe_line = rx.recv() => {
                    let Some(line) = maybe_line else {
                        // Channel closed (all senders dropped, e.g. tracing
                        // teardown on shutdown) — flush remaining and exit.
                        if !log_writer_stopped() {
                            absorb_flush(&store, &mut batch).await;
                        }
                        break;
                    };

                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let Some(entry) = parse_tracing_json(trimmed) else {
                        continue;
                    };

                    // Broadcast to dashboard subscribers before inserting (fast path)
                    let _ = broadcast.send(serde_json::to_string(&entry).expect(
                        "log entry broadcast serialization failed; this should not happen",
                    ));

                    if log_writer_stopped() {
                        // Terminal state: keep draining + broadcasting, but drop
                        // entries instead of pushing them onto a batch that will
                        // never be flushed (prevents unbounded memory growth).
                        continue;
                    }

                    batch.push(entry);
                    if batch.len() >= LOG_BATCH_MAX {
                        absorb_flush(&store, &mut batch).await;
                    }
                }
                _ = flush_timer.tick() => {
                    if !batch.is_empty() && !log_writer_stopped() {
                        absorb_flush(&store, &mut batch).await;
                    }
                }
            }
        }
    });
}

/// Flush the accumulated batch, absorbing storage-layer panics.
///
/// A panic here indicates a broken connection (e.g. the frame-index invariant
/// violation); it is recorded on the failure surface, the batch is dropped, and
/// the writer backs off before the next attempt. After
/// [`LOG_WRITER_MAX_CONSECUTIVE_PANICS`] consecutive panics the writer enters
/// the terminal stopped state (visible banner) rather than spinning forever.
/// A successful flush resets the consecutive-panic counter.
async fn absorb_flush(store: &LogStore, batch: &mut Vec<LogEntry>) {
    let result = AssertUnwindSafe(flush_log_batch(store, batch))
        .catch_unwind()
        .await;
    match result {
        Ok(()) => reset_log_writer_panic_state(),
        Err(payload) => {
            batch.clear();
            let message = format!(
                "log writer storage panic: {}",
                crate::util::panic_message(&*payload)
            );
            let consecutive = record_log_writer_panic(&message);
            if log_writer_stopped() {
                eprintln!(
                    "[mahbot] log store writer stopped after {consecutive} consecutive storage \
                     panics: {message}"
                );
            } else {
                tokio::time::sleep(log_writer_panic_backoff(consecutive)).await;
            }
        }
    }
}

/// Insert all accumulated entries in one transaction and clear the batch.
///
/// On persistent failure the batch is still cleared (entries are dropped) —
/// log entries are diagnostics, not durable state. Failures are **not**
/// swallowed: they are recorded on the [`log_write_error_info`] surface
/// (rendered on the GUI Logs page) and reported to stderr at a bounded rate.
/// No `tracing!` call is made from here — the writer task consumes the tracing
/// channel, so tracing from inside it would recurse into itself.
async fn flush_log_batch(store: &LogStore, batch: &mut Vec<LogEntry>) {
    if batch.is_empty() {
        return;
    }

    let mut last_error: Option<anyhow::Error> = None;
    for attempt in 0..LOG_INSERT_MAX_ATTEMPTS {
        match store.insert_batch(batch).await {
            Ok(()) => {
                batch.clear();
                return;
            }
            Err(e) => {
                last_error = Some(e);
                if attempt + 1 < LOG_INSERT_MAX_ATTEMPTS {
                    tokio::time::sleep(LOG_INSERT_RETRY_BACKOFF).await;
                }
            }
        }
    }

    record_log_write_failure(last_error);
    batch.clear();
}

// ── Log-writer error observability ──────────────────────────────────

/// Maximum number of insert attempts (including the first) for one log batch.
///
/// Retrying inside the writer is safe: the batch is only dropped after all
/// attempts fail. The total added latency is bounded by
/// `(LOG_INSERT_MAX_ATTEMPTS - 1) × LOG_INSERT_RETRY_BACKOFF`.
const LOG_INSERT_MAX_ATTEMPTS: usize = 3;

/// Backoff between log-batch insert retry attempts.
const LOG_INSERT_RETRY_BACKOFF: std::time::Duration = std::time::Duration::from_millis(250);

/// Minimum interval between stderr warnings about log-write failures.
const LOG_WRITE_STDERR_WARN_INTERVAL_MS: u64 = 60_000;

/// Consecutive storage-panic restarts before the writer stops flushing
/// permanently. A storage-layer panic indicates a broken connection (e.g. the
/// shared-WAL frame-index invariant violation); retrying past this bound would
/// spin forever on a connection that keeps panicking.
const LOG_WRITER_MAX_CONSECUTIVE_PANICS: u32 = 5;

/// Base backoff after a storage panic (doubles per consecutive panic, capped).
const LOG_WRITER_PANIC_BACKOFF_MS: u64 = 500;

/// Consecutive-panic restart state machine for the log writer.
///
/// Pure and unit-testable; the global writer state ([`LOG_WRITE_LAST_ERROR`])
/// mirrors this struct.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LogWriterPanicState {
    /// Consecutive storage-panic restarts in progress (0 when the last flush
    /// succeeded).
    pub consecutive_panics: u32,
    /// True once the writer stopped flushing permanently after exceeding
    /// [`LOG_WRITER_MAX_CONSECUTIVE_PANICS`] — the terminal banner state.
    pub writer_stopped: bool,
}

impl LogWriterPanicState {
    /// Record a storage panic; returns the updated consecutive-panic count.
    #[must_use]
    pub fn record_panic(&mut self) -> u32 {
        self.consecutive_panics += 1;
        if self.consecutive_panics >= LOG_WRITER_MAX_CONSECUTIVE_PANICS {
            self.writer_stopped = true;
        }
        self.consecutive_panics
    }

    /// Reset the consecutive-panic counter after a successful flush. The
    /// terminal stopped state is sticky — a stopped writer never flushes again
    /// (the broken connection cannot be healed without reopening the store).
    pub fn reset(&mut self) {
        self.consecutive_panics = 0;
    }
}

/// Snapshot of the log-writer failure surface, for display and tests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LogWriteErrorInfo {
    /// Total number of batch insert failures recorded since startup.
    pub count: u64,
    /// RFC 3339 timestamp of the most recent failure.
    pub last_timestamp: Option<String>,
    /// Message of the most recent failure.
    pub last_message: Option<String>,
    /// Writer panic-restart state.
    pub panic_state: LogWriterPanicState,
}

/// Most recent log-batch insert failure observed by the writer task.
///
/// Count, timestamp, and message live behind a single mutex so readers always
/// observe a consistent triple — a torn pair (timestamp from one failure with
/// the message from the next) would be misleading on the observability
/// surface.
static LOG_WRITE_LAST_ERROR: std::sync::Mutex<Option<LogWriteErrorInfo>> =
    std::sync::Mutex::new(None);

/// Unix millis of the last stderr warning (rate limiter).
static LOG_WRITE_LAST_STDERR_WARN_MS: AtomicU64 = AtomicU64::new(0);

/// Read the log-writer failure surface.
///
/// This is the sanctioned surface for log-persistence outages: the GUI Logs
/// page renders a warning banner from it. It is safe to call from anywhere
/// (no tracing involved), including from inside the writer task itself.
#[must_use]
pub fn log_write_error_info() -> LogWriteErrorInfo {
    LOG_WRITE_LAST_ERROR
        .lock()
        .unwrap_poison()
        .clone()
        .unwrap_or_default()
}

/// Record a failed log-batch insert on the observability surface.
fn record_log_write_failure(error: Option<anyhow::Error>) {
    let message = error.map_or_else(
        || "unknown log insert failure".to_string(),
        |e| format!("{e:#}"),
    );
    record_log_write_failure_impl(&message, LogFailureKind::Insert);
}

/// Record a storage-layer panic absorbed by the writer. Returns the updated
/// consecutive-panic count. The terminal stopped state additionally gets an
/// unconditional banner from the caller.
fn record_log_writer_panic(message: &str) -> u32 {
    record_log_write_failure_impl(message, LogFailureKind::WriterPanic)
}

/// Which failure kind is being recorded — drives the stderr label and whether
/// the storage-panic restart counter is bumped.
#[derive(Clone, Copy)]
enum LogFailureKind {
    Insert,
    WriterPanic,
}

impl LogFailureKind {
    fn label(self) -> &'static str {
        match self {
            Self::Insert => "insert failure",
            Self::WriterPanic => "writer panic",
        }
    }

    fn records_panic(self) -> bool {
        matches!(self, Self::WriterPanic)
    }
}

/// Shared failure-recording core: bumps the counter, stamps timestamp/message,
/// optionally records a storage-panic restart, and emits a rate-limited stderr
/// warning (`kind` labels the failure on the stderr line). stderr is not
/// routed through tracing, so this cannot recurse into the writer task.
fn record_log_write_failure_impl(message: &str, kind: LogFailureKind) -> u32 {
    let (count, consecutive) = {
        let mut guard = LOG_WRITE_LAST_ERROR.lock().unwrap_poison();
        let entry = guard.get_or_insert(LogWriteErrorInfo::default());
        entry.count += 1;
        entry.last_timestamp = Some(turso::now());
        entry.last_message = Some(message.to_string());
        let consecutive = if kind.records_panic() {
            entry.panic_state.record_panic()
        } else {
            entry.panic_state.consecutive_panics
        };
        (entry.count, consecutive)
    };
    emit_stderr_warning(count, message, kind.label());
    consecutive
}

/// Reset the consecutive-panic counter (and the terminal stopped flag) after a
/// successful flush.
fn reset_log_writer_panic_state() {
    let mut guard = LOG_WRITE_LAST_ERROR.lock().unwrap_poison();
    if let Some(entry) = guard.as_mut() {
        entry.panic_state.reset();
    }
}

/// True once the writer has permanently stopped flushing (terminal banner).
fn log_writer_stopped() -> bool {
    LOG_WRITE_LAST_ERROR
        .lock()
        .unwrap_poison()
        .as_ref()
        .is_some_and(|e| e.panic_state.writer_stopped)
}

/// Backoff after the `n`-th consecutive storage panic: 500ms, 1s, 2s, … capped
/// at 30s. The terminal bound ([`LOG_WRITER_MAX_CONSECUTIVE_PANICS`]) ends the
/// sequence before the cap engages today; the cap guards a future bound
/// increase.
fn log_writer_panic_backoff(consecutive: u32) -> std::time::Duration {
    let shift = consecutive.saturating_sub(1).min(6);
    let ms = LOG_WRITER_PANIC_BACKOFF_MS.saturating_mul(1 << shift);
    std::time::Duration::from_millis(ms.min(30_000))
}

/// Rate-limited stderr warning (stderr bypasses tracing, so this cannot
/// recurse into the writer task).
fn emit_stderr_warning(count: u64, message: &str, kind: &str) {
    let now_ms = crate::util::unix_millis();
    let last_warn_ms = LOG_WRITE_LAST_STDERR_WARN_MS.load(Ordering::SeqCst);
    if now_ms.saturating_sub(last_warn_ms) >= LOG_WRITE_STDERR_WARN_INTERVAL_MS {
        LOG_WRITE_LAST_STDERR_WARN_MS.store(now_ms, Ordering::SeqCst);
        eprintln!("[mahbot] log store {kind} #{count}: {message}");
    }
}

/// Extract a string field from a JSON value, defaulting to `""`.
fn get_str_or_empty(val: &serde_json::Value, key: &str) -> String {
    json::get_opt_str(val, key).unwrap_or("").to_string()
}

/// Parse a tracing-subscriber JSON line into a `LogEntry`.
fn parse_tracing_json(line: &str) -> Option<LogEntry> {
    let val: serde_json::Value = serde_json::from_str(line).ok()?;

    let timestamp = get_str_or_empty(&val, "timestamp");
    let level = get_str_or_empty(&val, "level");
    let target = get_str_or_empty(&val, "target");

    let mut fields = val
        .get("fields")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let message = get_str_or_empty(&fields, "message");

    if let Some(obj) = fields.as_object_mut() {
        obj.remove("message");
    }
    let fields = if fields.as_object().is_some_and(serde_json::Map::is_empty) {
        serde_json::Value::Null
    } else {
        fields
    };

    // Extract agent_id, agent_role and workspace from the innermost span
    let (agent_id, agent_role, workspace) = extract_agent_from_span(&val);

    Some(LogEntry {
        timestamp,
        level,
        target,
        message,
        fields,
        agent_id,
        agent_role,
        workspace,
    })
}

/// Extract the three agent-related fields from a span JSON object.
fn extract_agent_fields(span: &serde_json::Value) -> (String, String, String) {
    (
        get_str_or_empty(span, "agent_id"),
        get_str_or_empty(span, "role"),
        get_str_or_empty(span, "workspace"),
    )
}

/// Extract `agent_id`, `role`, and `workspace` from the current span data
/// in tracing JSON.
///
/// `tracing-subscriber` JSON format puts the current span's fields under
/// `span.agent_id`, `span.role`, and `span.workspace` (or `spans[last].*`).
/// The three sources are merged per-component — the event's own fields win
/// where present, the span fills the gaps — so every corner attributes
/// correctly:
///
/// - full event fields (e.g. `run_agent`'s "Agent failed" entry, ask-tool
///   sub-agent failures) name the failing agent directly, beating the
///   inherited caller span;
/// - workspace-only events (e.g. "Search index capacity exhausted") keep the
///   span's agent attribution;
/// - agent_id-only events (e.g. session-persistence warnings) keep the span's
///   role/workspace attribution.
fn extract_agent_from_span(val: &serde_json::Value) -> (String, String, String) {
    let mut agent_id = String::new();
    let mut role = String::new();
    let mut workspace = String::new();
    for candidate in std::iter::once(val.get("fields"))
        .chain(std::iter::once(val.get("span")))
        .chain(std::iter::once(
            val.get("spans")
                .and_then(|v| v.as_array())
                .and_then(|a| a.last()),
        ))
        .flatten()
    {
        let (id, r, ws) = extract_agent_fields(candidate);
        if agent_id.is_empty() {
            agent_id = id;
        }
        if role.is_empty() {
            role = r;
        }
        if workspace.is_empty() {
            workspace = ws;
        }
        if !agent_id.is_empty() && !role.is_empty() && !workspace.is_empty() {
            break;
        }
    }
    (agent_id, role, workspace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tracing_json_full() {
        let line = r#"{"timestamp":"2025-05-06T12:34:56.000000Z","level":"INFO","target":"mahbot::orchestrator","span":{"name":"agent","agent_id":"00000000-0000-0000-0000-000000000000","role":"lead","workspace":"/some/workspace"},"fields":{"message":"Hello world","key":"value"}}"#;
        let entry = parse_tracing_json(line).unwrap();
        assert_eq!(entry.timestamp, "2025-05-06T12:34:56.000000Z");
        assert_eq!(entry.level, "INFO");
        assert_eq!(entry.target, "mahbot::orchestrator");
        assert_eq!(entry.message, "Hello world");
        assert_eq!(entry.fields, serde_json::json!({"key": "value"}));
        assert_eq!(entry.agent_id, "00000000-0000-0000-0000-000000000000");
        assert_eq!(entry.agent_role, "lead");
        assert_eq!(entry.workspace, "/some/workspace");
    }

    #[test]
    fn test_parse_tracing_json_no_fields() {
        let line = r#"{"timestamp":"2025-05-06T12:34:56.000000Z","level":"WARN","target":"test","fields":{"message":"warning"}}"#;
        let entry = parse_tracing_json(line).unwrap();
        assert_eq!(entry.message, "warning");
        assert_eq!(entry.fields, serde_json::Value::Null);
        assert_eq!(entry.agent_id, "");
        assert_eq!(entry.agent_role, "");
        assert_eq!(entry.workspace, "");
    }

    #[test]
    fn test_parse_tracing_json_lenient() {
        let entry = parse_tracing_json(r#"{"incomplete": true}"#).unwrap();
        assert_eq!(entry.timestamp, "");
        assert_eq!(entry.level, "");
        assert_eq!(entry.target, "");
        assert_eq!(entry.message, "");
        assert_eq!(entry.fields, serde_json::Value::Null);
        assert_eq!(entry.agent_id, "");
        assert_eq!(entry.agent_role, "");
        assert_eq!(entry.workspace, "");
    }

    /// Agent-attribution corners for `parse_tracing_json`: the event's own
    /// fields win where present, the span fills the gaps, and the `spans`
    /// array is the last resort. Each case: (name, line, agent_id, role,
    /// workspace).
    #[test]
    fn test_parse_tracing_json_agent_attribution() {
        let cases = [
            (
                "span only",
                r#"{"timestamp":"...","level":"INFO","target":"test","span":{"name":"agent","agent_id":"abc-123","role":"analyst"},"fields":{"message":"researching"}}"#,
                "abc-123",
                "analyst",
                "",
            ),
            (
                "spans array",
                r#"{"timestamp":"...","level":"INFO","target":"test","spans":[{"name":"parent"},{"name":"agent","agent_id":"xyz-456","role":"coder","workspace":"/ws"}],"fields":{"message":"writing code"}}"#,
                "xyz-456",
                "coder",
                "/ws",
            ),
            (
                "event fields without span",
                r#"{"timestamp":"...","level":"ERROR","target":"mahbot::agent","fields":{"message":"Agent failed","agent_id":"ticket_123_engineer","role":"engineer","workspace":"my-ws","classification":"transport"}}"#,
                "ticket_123_engineer",
                "engineer",
                "my-ws",
            ),
            (
                "event beats inherited span",
                r#"{"timestamp":"...","level":"ERROR","target":"mahbot::agent","span":{"name":"agent","agent_id":"caller_42","role":"engineer","workspace":"parent-ws"},"fields":{"message":"Agent failed","agent_id":"ask_ws_1_2_analyst","role":"analyst","workspace":"my-ws","classification":"runtime"}}"#,
                "ask_ws_1_2_analyst",
                "analyst",
                "my-ws",
            ),
            (
                "workspace-only event keeps span agent",
                r#"{"timestamp":"...","level":"WARN","target":"mahbot::tools::edit","span":{"name":"agent","agent_id":"ticket_7_engineer","role":"engineer","workspace":"my-ws"},"fields":{"message":"Search index capacity exhausted","workspace":"my-ws","path":"src/a.rs"}}"#,
                "ticket_7_engineer",
                "engineer",
                "my-ws",
            ),
            (
                "agent_id-only event merges span role/workspace",
                r#"{"timestamp":"...","level":"WARN","target":"mahbot::agent","span":{"name":"agent","agent_id":"ticket_7_engineer","role":"engineer","workspace":"my-ws"},"fields":{"message":"Failed to persist incoming messages to session DB","agent_id":"ticket_7_engineer","error":"io"}}"#,
                "ticket_7_engineer",
                "engineer",
                "my-ws",
            ),
        ];
        for (name, line, id, role, ws) in cases {
            let entry = parse_tracing_json(line).unwrap();
            assert_eq!(entry.agent_id, id, "{name}: agent_id");
            assert_eq!(entry.agent_role, role, "{name}: agent_role");
            assert_eq!(entry.workspace, ws, "{name}: workspace");
        }
    }

    /// Create a temporary LogStore for tests.
    /// Returns the store and a TempDir that must be held to prevent premature cleanup.
    async fn test_store() -> (Arc<LogStore>, tempfile::TempDir) {
        let (store, dir) = crate::open_test_store!(LogStore, "log");
        (Arc::new(store), dir)
    }

    // Helper to seed log entries in tests
    async fn seed_entries(store: &LogStore, entries: &[LogEntry]) {
        store.insert_batch(entries).await.unwrap();
    }

    #[tokio::test]
    async fn test_spawn_log_writer_writes_to_store() {
        let (store, _dir) = test_store().await;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let (broadcast_tx, _) = tokio::sync::broadcast::channel(256);

        spawn_log_writer(store.clone(), rx, broadcast_tx);

        tx.send(
            r#"{"timestamp":"2025-01-01T00:00:00Z","level":"INFO","target":"test","fields":{"message":"hi"}}"#
                .to_string(),
        )
        .unwrap();
        tx.send(
            r#"{"timestamp":"2025-01-01T00:00:01Z","level":"ERROR","target":"test","fields":{"message":"oh no","err":"boom"}}"#
                .to_string(),
        )
        .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        drop(tx);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let (entries, total) = store.query(&LogQuery::default()).await.unwrap();
        assert_eq!(total, 2);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].message, "oh no");
        assert_eq!(entries[1].message, "hi");
    }

    /// Poll `store.query` until the total entry count reaches `expected` or the
    /// timeout elapses. Avoids the fixed-sleep races of the earlier version
    /// (the writer and the test share a runtime, so wall-clock sleeps can be
    /// skewed by writer-side DB work on loaded machines).
    async fn wait_for_total(store: &LogStore, expected: usize, timeout: std::time::Duration) {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let (_, total) = store.query(&LogQuery::default()).await.unwrap();
            if total == expected {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for total == {expected}, got {total}"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn test_flush_log_batch_records_failure_on_surface() {
        let (store, _dir) = test_store().await;
        let baseline = log_write_error_info().count;

        let entry = LogEntry {
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            level: "INFO".to_string(),
            target: "test".to_string(),
            message: "should not persist".to_string(),
            fields: serde_json::Value::Null,
            agent_id: String::new(),
            agent_role: String::new(),
            workspace: String::new(),
        };

        // Deterministic insert failure: drop the logs table through the store's
        // own connection, so the batch INSERT fails at prepare time.
        store
            .conn
            .execute("DROP TABLE logs", ())
            .await
            .expect("drop logs table for failure test");

        let mut batch = vec![entry];
        flush_log_batch(&store, &mut batch).await;

        // The batch must be dropped (entries are diagnostics) and the failure
        // must be visible on the sanctioned surface.
        assert!(batch.is_empty(), "failed batch must still be cleared");
        let info = log_write_error_info();
        assert!(
            info.count > baseline,
            "failure count must advance: baseline {baseline}, now {}",
            info.count
        );
        assert!(
            info.last_message.is_some(),
            "last-error message must be recorded"
        );
    }

    #[tokio::test]
    async fn test_flush_log_batch_retries_then_records() {
        // After restoring write access, a retried flush must succeed without
        // recording a failure — the bounded retry absorbs transient errors.
        let (store, _dir) = test_store().await;
        let baseline = log_write_error_info().count;

        let mut batch = vec![LogEntry {
            timestamp: "2025-01-01T00:00:01Z".to_string(),
            level: "INFO".to_string(),
            target: "test".to_string(),
            message: "persisted".to_string(),
            fields: serde_json::Value::Null,
            agent_id: String::new(),
            agent_role: String::new(),
            workspace: String::new(),
        }];
        flush_log_batch(&store, &mut batch).await;

        assert!(batch.is_empty());
        assert_eq!(
            log_write_error_info().count,
            baseline,
            "no failure recorded"
        );
        let (entries, total) = store.query(&LogQuery::default()).await.unwrap();
        assert_eq!(total, 1);
        assert_eq!(entries[0].message, "persisted");
    }

    #[tokio::test]
    async fn test_log_writer_batches_and_timer_flushes() {
        // Writer 1: a flush interval long enough that the timer can never fire
        // during the test — only the batch-cap path can insert rows.
        let (store, _dir) = test_store().await;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let (broadcast_tx, _) = tokio::sync::broadcast::channel(256);
        spawn_log_writer_with_interval(
            store.clone(),
            rx,
            broadcast_tx,
            std::time::Duration::from_secs(60),
        );

        // One entry first — below the batch cap, so it stays buffered.
        tx.send(
            r#"{"timestamp":"2025-01-01T00:00:02Z","level":"INFO","target":"test","fields":{"message":"timer flush"}}"#
                .to_string(),
        )
        .unwrap();

        // Then LOG_BATCH_MAX more entries — the batch-cap flush fires as soon
        // as the writer accumulates a full batch (the timer cannot fire here).
        for i in 0..LOG_BATCH_MAX {
            tx.send(
                format!(
                    r#"{{"timestamp":"2025-01-01T00:00:03Z","level":"INFO","target":"test","fields":{{"message":"batch {i}"}}}}"#
                ),
            )
            .unwrap();
        }

        // The cap flush must insert exactly LOG_BATCH_MAX entries; the lone
        // earlier entry is still buffered (60s timer never fired).
        wait_for_total(&store, LOG_BATCH_MAX, std::time::Duration::from_secs(10)).await;

        // Writer 2: a very short flush interval — the timer path must flush a
        // lone buffered entry without needing the cap or channel close.
        let (store2, _dir2) = test_store().await;
        let (tx2, rx2) = tokio::sync::mpsc::unbounded_channel();
        let (broadcast_tx2, _) = tokio::sync::broadcast::channel(256);
        spawn_log_writer_with_interval(
            store2.clone(),
            rx2,
            broadcast_tx2,
            std::time::Duration::from_millis(50),
        );
        tx2.send(
            r#"{"timestamp":"2025-01-01T00:00:04Z","level":"INFO","target":"test","fields":{"message":"timer fired"}}"#
                .to_string(),
        )
        .unwrap();
        wait_for_total(&store2, 1, std::time::Duration::from_secs(10)).await;

        drop(tx);
        drop(tx2);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_like_search_substring() {
        let (store, _dir) = test_store().await;

        let entries = vec![
            LogEntry {
                timestamp: "2025-01-01T00:00:00Z".into(),
                level: "INFO".into(),
                target: "module_a".into(),
                message: "processing request".into(),
                fields: serde_json::Value::Null,
                agent_id: String::new(),
                agent_role: String::new(),
                workspace: String::new(),
            },
            LogEntry {
                timestamp: "2025-01-01T00:00:01Z".into(),
                level: "ERROR".into(),
                target: "module_b".into(),
                message: "failed to process".into(),
                fields: serde_json::Value::Null,
                agent_id: String::new(),
                agent_role: String::new(),
                workspace: String::new(),
            },
            LogEntry {
                timestamp: "2025-01-01T00:00:02Z".into(),
                level: "INFO".into(),
                target: "module_c".into(),
                message: "started".into(),
                fields: serde_json::Value::Null,
                agent_id: String::new(),
                agent_role: String::new(),
                workspace: String::new(),
            },
        ];

        seed_entries(&store, &entries).await;

        // LIKE %...% matches substrings: "proc" matches "processing" and "process"
        let (results, total) = store
            .query(&LogQuery {
                search: Some("proc".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 2, "substring 'proc' should match both entries");
        assert_eq!(results.len(), 2);
        let (results, total) = store
            .query(&LogQuery {
                search: Some("request".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(results[0].message, "processing request");

        // LIKE matches the target column too
        let (_results, total) = store
            .query(&LogQuery {
                search: Some("module".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 3, "all targets contain 'module'");
    }

    #[tokio::test]
    async fn test_like_search_combined_filters() {
        let (store, _dir) = test_store().await;

        let entries = vec![
            LogEntry {
                timestamp: "2025-01-01T00:00:00Z".into(),
                level: "INFO".into(),
                target: "mahbot::orchestrator".into(),
                message: "processing request".into(),
                fields: serde_json::Value::Null,
                agent_id: String::new(),
                agent_role: String::new(),
                workspace: String::new(),
            },
            LogEntry {
                timestamp: "2025-01-01T00:00:01Z".into(),
                level: "ERROR".into(),
                target: "mahbot::tools".into(),
                message: "failed to process".into(),
                fields: serde_json::json!({"code": 1}),
                agent_id: String::new(),
                agent_role: String::new(),
                workspace: String::new(),
            },
            LogEntry {
                timestamp: "2025-01-01T00:00:02Z".into(),
                level: "INFO".into(),
                target: "mahbot::api".into(),
                message: "started".into(),
                fields: serde_json::Value::Null,
                agent_id: String::new(),
                agent_role: String::new(),
                workspace: String::new(),
            },
        ];

        seed_entries(&store, &entries).await;

        // LIKE + level filter
        let (results, total) = store
            .query(&LogQuery {
                level: Some("ERROR".into()),
                search: Some("process".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 1, "only ERROR log matching 'process'");
        assert_eq!(results[0].message, "failed to process");
        let (_results, total) = store
            .query(&LogQuery {
                target: Some("mahbot::tools".into()),
                search: Some("process".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 1, "only tools target entry matching 'process'");

        // LIKE + since
        let (_results, total) = store
            .query(&LogQuery {
                since: Some("2025-01-01T00:00:01Z".into()),
                search: Some("process".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 1, "only entry after timestamp matching 'process'");
    }

    #[tokio::test]
    async fn test_like_search_with_special_chars() {
        let (store, _dir) = test_store().await;

        let entries = vec![
            LogEntry {
                timestamp: "2025-01-01T00:00:00Z".into(),
                level: "INFO".into(),
                target: "module_a".into(),
                message: "processing `Hello ${name}` template".into(),
                fields: serde_json::Value::Null,
                agent_id: String::new(),
                agent_role: String::new(),
                workspace: String::new(),
            },
            LogEntry {
                timestamp: "2025-01-01T00:00:01Z".into(),
                level: "ERROR".into(),
                target: "module_b".into(),
                message: "normal log entry".into(),
                fields: serde_json::Value::Null,
                agent_id: String::new(),
                agent_role: String::new(),
                workspace: String::new(),
            },
        ];

        seed_entries(&store, &entries).await;

        // LIKE is literal substring — backtick and ${} match as-is
        let (results, total) = store
            .query(&LogQuery {
                search: Some("template".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 1, "LIKE should match partial word in message");
        assert!(
            results[0].message.contains("template"),
            "should match the correct entry"
        );

        // Empty search returns all entries
        let (_results, total) = store
            .query(&LogQuery {
                search: None,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 2, "no search filter should return all entries");
    }

    /// The writer's panic-restart bound: after enough consecutive panics the
    /// writer enters the terminal stopped state (banner), and a successful
    /// flush resets the counter (the stopped state is sticky).
    #[test]
    fn test_log_writer_panic_state_machine() {
        let mut state = LogWriterPanicState::default();
        assert!(!state.writer_stopped);
        for i in 1..=LOG_WRITER_MAX_CONSECUTIVE_PANICS {
            let _ = state.record_panic();
            assert_eq!(state.consecutive_panics, i);
        }
        assert!(
            state.writer_stopped,
            "writer must stop after the consecutive-panic bound"
        );
        state.reset();
        assert_eq!(state.consecutive_panics, 0);
        assert!(
            state.writer_stopped,
            "terminal stopped state is sticky across reset"
        );
    }

    /// Boot-time quarantine: a logs store whose main DB file is corrupt is
    /// moved aside to a timestamped quarantine name and a fresh store is
    /// created in its place, without failing the open.
    #[tokio::test]
    async fn test_log_store_open_quarantines_corrupt_store() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let root = tmp.path();

        // Build a store with real data, then checkpoint so all pages live in
        // the main DB file (not the WAL), then close it.
        {
            let store = LogStore::open(root).await.expect("open healthy store");
            store
                .insert_batch(&[LogEntry {
                    timestamp: "2025-01-01T00:00:00Z".into(),
                    level: "INFO".into(),
                    target: "test".into(),
                    message: "pre-corruption".into(),
                    fields: serde_json::Value::Null,
                    agent_id: String::new(),
                    agent_role: String::new(),
                    workspace: String::new(),
                }])
                .await
                .expect("seed entry");
            store
                .conn
                .checkpoint()
                .await
                .expect("checkpoint so pages land in the main DB file");
        }

        // Corrupt a b-tree page in the main DB file (zero page 2; the header
        // page 1 stays intact so the file still opens).
        let db_path = turso::store_db_path(root, "logs");
        let bytes = std::fs::read(&db_path).expect("read db file");
        assert!(bytes.len() > 8192, "test needs a multi-page db file");
        let mut corrupted = bytes.clone();
        corrupted[4096..8192].fill(0);
        std::fs::write(&db_path, corrupted).expect("corrupt db file");

        // Open must succeed with a fresh store, leaving the corrupt artifact
        // family quarantined.
        let store = LogStore::open(root).await.expect("open must not fail");
        let (_, total) = store
            .query(&LogQuery::default())
            .await
            .expect("fresh store query");
        assert_eq!(total, 0, "fresh store must be empty");
        // The consolidated stats tables are part of the logs schema — a
        // quarantine recreate must recreate them too (not silently discard).
        let stats_tables: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type='table' AND name IN ('tool_calls','retry_failures')",
                params![],
                |row| row.get::<i64>(0),
            )
            .await
            .expect("count consolidated stats tables");
        assert_eq!(
            stats_tables, 2,
            "consolidated stats tables must exist after quarantine recreate"
        );
        let quarantined: Vec<_> = std::fs::read_dir(root.join("db"))
            .expect("read db dir")
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("quarantine-"))
            .collect();
        assert!(
            !quarantined.is_empty(),
            "corrupt artifact family must be quarantined, found: {quarantined:?}"
        );
    }

    /// Boot-time quarantine via the open-failure path: a store whose header is
    /// so corrupt it cannot even be opened (invalid page-size field) is still
    /// quarantined and recreated, rather than failing boot.
    #[tokio::test]
    async fn test_log_store_open_quarantines_unopenable_store() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let root = tmp.path();

        {
            let store = LogStore::open(root).await.expect("open healthy store");
            store
                .insert_batch(&[LogEntry {
                    timestamp: "2025-01-01T00:00:00Z".into(),
                    level: "INFO".into(),
                    target: "test".into(),
                    message: "pre-corruption".into(),
                    fields: serde_json::Value::Null,
                    agent_id: String::new(),
                    agent_role: String::new(),
                    workspace: String::new(),
                }])
                .await
                .expect("seed entry");
            store
                .conn
                .checkpoint()
                .await
                .expect("checkpoint so pages land in the main DB file");
        }

        // Zero the header's page-size field (big-endian u16 at byte offset 16)
        // so the file cannot be opened at all — Limbo bails with a corruption
        // error ("invalid page size in database header") rather than an IO error.
        let db_path = turso::store_db_path(root, "logs");
        let mut bytes = std::fs::read(&db_path).expect("read db file");
        bytes[16] = 0;
        bytes[17] = 0;
        std::fs::write(&db_path, bytes).expect("corrupt db file");

        let store = LogStore::open(root).await.expect("open must not fail");
        let (_, total) = store
            .query(&LogQuery::default())
            .await
            .expect("fresh store query");
        assert_eq!(total, 0, "fresh store must be empty");
        let quarantined: Vec<_> = std::fs::read_dir(root.join("db"))
            .expect("read db dir")
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("quarantine-"))
            .collect();
        assert!(
            !quarantined.is_empty(),
            "unopenable store must be quarantined, found: {quarantined:?}"
        );
    }
}
