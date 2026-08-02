//! Turso-backed log storage.
//!
//! Each log entry is inserted asynchronously via a background channel task.
//! A broadcast channel feeds live log entries to the Iced native GUI dashboard.

use crate::turso;
use crate::util::UnwrapPoison;
use crate::util::json;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::OnceCell;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
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
CREATE INDEX IF NOT EXISTS idx_logs_workspace ON logs(workspace);";

impl LogStore {
    /// Open (or create) the log database at `root/db/logs.db` and run schema migrations.
    ///
    /// `pub(crate)` (matching every other store's generated `open`) so tests in
    /// other modules can create a real log store via [`crate::open_test_store!`].
    pub(crate) async fn open(root: &Path) -> anyhow::Result<Self> {
        let conn = crate::turso::open_store(root, "logs", LOGS_SCHEMA).await?;
        Ok(Self { conn })
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
            entries.push(row_to_entry(&row)?);
        }

        Ok((entries, total))
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

fn row_to_entry(row: &Row) -> anyhow::Result<LogEntry> {
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
                        flush_log_batch(&store, &mut batch).await;
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

                    batch.push(entry);
                    if batch.len() >= LOG_BATCH_MAX {
                        flush_log_batch(&store, &mut batch).await;
                    }
                }
                _ = flush_timer.tick() => {
                    if !batch.is_empty() {
                        flush_log_batch(&store, &mut batch).await;
                    }
                }
            }
        }
    });
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

/// Snapshot of the log-writer failure surface, for display and tests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LogWriteErrorInfo {
    /// Total number of batch insert failures recorded since startup.
    pub count: u64,
    /// RFC 3339 timestamp of the most recent failure.
    pub last_timestamp: Option<String>,
    /// Message of the most recent failure.
    pub last_message: Option<String>,
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
///
/// Updates the counter and last-error fields and emits a rate-limited stderr
/// warning. stderr is not routed through tracing, so this cannot recurse into
/// the writer task.
fn record_log_write_failure(error: Option<anyhow::Error>) {
    let message = error.map_or_else(
        || "unknown log insert failure".to_string(),
        |e| format!("{e:#}"),
    );
    let count = {
        let mut guard = LOG_WRITE_LAST_ERROR.lock().unwrap_poison();
        let entry = guard.get_or_insert(LogWriteErrorInfo::default());
        entry.count += 1;
        entry.last_timestamp = Some(turso::now());
        entry.last_message = Some(message.clone());
        entry.count
    };

    let now_ms = unix_millis();
    let last_warn_ms = LOG_WRITE_LAST_STDERR_WARN_MS.load(Ordering::SeqCst);
    if now_ms.saturating_sub(last_warn_ms) >= LOG_WRITE_STDERR_WARN_INTERVAL_MS {
        LOG_WRITE_LAST_STDERR_WARN_MS.store(now_ms, Ordering::SeqCst);
        eprintln!("[mahbot] log store insert failure #{count}: {message}");
    }
}

/// Current Unix time in milliseconds.
fn unix_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        d.as_secs().saturating_mul(1_000) + u64::from(d.subsec_millis())
    })
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
fn extract_agent_from_span(val: &serde_json::Value) -> (String, String, String) {
    // Prefer the innermost span (direct `span` key)
    if let Some(span) = val.get("span") {
        let (id, role, ws) = extract_agent_fields(span);
        if !id.is_empty() || !role.is_empty() || !ws.is_empty() {
            return (id, role, ws);
        }
    }

    // Fall back to the last entry in the `spans` array
    if let Some(spans) = val.get("spans").and_then(|v| v.as_array())
        && let Some(last_span) = spans.last()
    {
        let (id, role, ws) = extract_agent_fields(last_span);
        return (id, role, ws);
    }

    (String::new(), String::new(), String::new())
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

    #[test]
    fn test_parse_tracing_json_with_span_agent() {
        let line = r#"{"timestamp":"...","level":"INFO","target":"test","span":{"name":"agent","agent_id":"abc-123","role":"analyst"},"fields":{"message":"researching"}}"#;
        let entry = parse_tracing_json(line).unwrap();
        assert_eq!(entry.agent_id, "abc-123");
        assert_eq!(entry.agent_role, "analyst");
        assert_eq!(entry.workspace, "");
    }

    #[test]
    fn test_parse_tracing_json_with_spans_array() {
        let line = r#"{"timestamp":"...","level":"INFO","target":"test","spans":[{"name":"parent"},{"name":"agent","agent_id":"xyz-456","role":"coder","workspace":"/ws"}],"fields":{"message":"writing code"}}"#;
        let entry = parse_tracing_json(line).unwrap();
        assert_eq!(entry.agent_id, "xyz-456");
        assert_eq!(entry.agent_role, "coder");
        assert_eq!(entry.workspace, "/ws");
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
}
