//! Turso-backed log storage.
//!
//! Each log entry is inserted asynchronously via a background channel task.
//! A broadcast channel feeds live log entries to the Iced native GUI dashboard.

use crate::turso::{self, row_text};
use crate::util::json;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;
use std::sync::Arc;
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

/// Column list for log SELECT queries.
///
/// The column order here must match the positional indices defined in
/// [`COL_LOGS_TIMESTAMP`] through [`COL_LOGS_WORKSPACE`], which are used
/// in [`row_to_entry`].
const LOGS_COLUMNS: &str =
    "timestamp, level, target, message, fields, agent_id, agent_role, workspace";

/// Column-index constants for [`LOGS_COLUMNS`].
///
/// These replace hardcoded positional indices in [`row_to_entry`].
/// With named constants, the compiler catches references to undefined
/// column constants — for instance, removing a constant but forgetting to
/// update a `row_text()` call produces a compile error rather than a silent
/// field mapping bug.
const COL_LOGS_TIMESTAMP: usize = 0;
const COL_LOGS_LEVEL: usize = 1;
const COL_LOGS_TARGET: usize = 2;
const COL_LOGS_MESSAGE: usize = 3;
const COL_LOGS_FIELDS: usize = 4;
const COL_LOGS_AGENT_ID: usize = 5;
const COL_LOGS_AGENT_ROLE: usize = 6;
const COL_LOGS_WORKSPACE: usize = 7;

/// Turso-backed log store.
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

/// Returns a reference to the global log store.
///
/// # Panics
///
/// Panics if the log store has not been initialized. [`init_tracing()`] is
/// called during bootstrap — before any code accesses this store — so this
/// panic only occurs due to a programming error.
#[must_use]
pub fn store() -> &'static LogStore {
    LOG_STORE
        .get()
        .expect("LOG_STORE not initialized — call init_tracing() first")
}

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
    /// Open (or create) the log database at `db_path` and run schema migrations.
    async fn open(db_path: &Path) -> anyhow::Result<Self> {
        let conn = crate::turso::open_with_schema(db_path, LOGS_SCHEMA).await?;
        Ok(Self { conn })
    }

    /// Insert a single log entry into the database.
    async fn insert(&self, entry: &LogEntry) -> anyhow::Result<()> {
        self.conn
            .execute(
                "INSERT INTO logs (timestamp, level, target, message, fields, agent_id, agent_role, workspace) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    entry.timestamp.clone(),
                    entry.level.clone(),
                    entry.target.clone(),
                    entry.message.clone(),
                    serde_json::to_string(&entry.fields).unwrap_or_default(),
                    entry.agent_id.clone(),
                    entry.agent_role.clone(),
                    entry.workspace.clone(),
                ],
            )
            .await
            .context("Failed to insert log entry")?;
        Ok(())
    }

    /// Build the data query with LIKE-based search using bind params.
    fn build_query(filters: &LogQuery) -> (String, Vec<Value>) {
        let (where_sql, mut values) = build_where_clause(filters);

        let limit: i64 = i64::try_from(filters.limit.unwrap_or(100).min(1000)).unwrap_or(100);
        let offset: i64 = i64::try_from(filters.offset.unwrap_or(0)).unwrap_or(0);
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
        let rows = self.conn.query(&sql, values).await?;
        let count = match rows.into_iter().next() {
            Some(row) => match row.get_value(0)? {
                Value::Integer(n) => usize::try_from(n).unwrap_or(0),
                _ => 0,
            },
            None => 0,
        };
        Ok(count)
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
    let timestamp = row_text(row, COL_LOGS_TIMESTAMP)?;
    let level = row_text(row, COL_LOGS_LEVEL)?;
    let target = row_text(row, COL_LOGS_TARGET)?;
    let message = row_text(row, COL_LOGS_MESSAGE)?;
    let fields_str = match row.get_value(COL_LOGS_FIELDS)? {
        Value::Text(s) => s,
        Value::Null => "{}".to_string(),
        _ => anyhow::bail!("expected text for fields"),
    };
    let fields: serde_json::Value =
        serde_json::from_str(&fields_str).unwrap_or(serde_json::Value::Null);

    let agent_id = row_text(row, COL_LOGS_AGENT_ID)?;
    let agent_role = row_text(row, COL_LOGS_AGENT_ROLE)?;
    let workspace = row_text(row, COL_LOGS_WORKSPACE)?;

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
    let store = LogStore::open(&storage_root.join("db/logs.db")).await?;
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

/// Spawn a background task that receives JSON log lines and writes them to Turso
/// and broadcasts them over the channel to the Iced GUI dashboard.
///
/// Each log entry is inserted and broadcast immediately — no batching.
fn spawn_log_writer(
    store: Arc<LogStore>,
    mut rx: UnboundedReceiver<String>,
    broadcast: tokio::sync::broadcast::Sender<String>,
) {
    tokio::spawn(async move {
        while let Some(line) = rx.recv().await {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let Some(entry) = parse_tracing_json(trimmed) else {
                continue;
            };

            // Broadcast to dashboard subscribers before inserting (fast path)
            let _ = broadcast.send(serde_json::to_string(&entry).unwrap_or_default());

            if let Err(e) = store.insert(&entry).await {
                // Log insertion failed — nowhere useful to report since
                // terminal output is disabled and we can't use tracing here
                // (that would recurse).
                let _ = e;
            }
        }
    });
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

    /// Verify that the number of columns in [`LOGS_COLUMNS`] matches the highest
    /// column-index constant + 1. If this test fails, a column was added or removed
    /// from the string list without updating the corresponding `COL_LOGS_*` constants,
    /// or vice versa — a silent data corruption hazard.
    #[test]
    fn logs_columns_count_matches_column_constants() {
        crate::assert_column_count!(LOGS_COLUMNS, COL_LOGS_WORKSPACE);
    }

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

    // Helper to seed log entries in tests
    async fn seed_entries(store: &LogStore, entries: &[LogEntry]) {
        for e in entries {
            store.insert(e).await.unwrap();
        }
    }

    #[test]
    fn test_spawn_log_writer_writes_to_store() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::TempDir::new().unwrap();
            let db_path = dir.path().join("db/logs.db");
            let store = Arc::new(LogStore::open(&db_path).await.unwrap());
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
        });
    }

    #[test]
    fn test_like_search_substring() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::TempDir::new().unwrap();
            let db_path = dir.path().join("db/logs.db");
            let store = Arc::new(LogStore::open(&db_path).await.unwrap());

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
        });
    }

    #[test]
    fn test_like_search_combined_filters() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::TempDir::new().unwrap();
            let db_path = dir.path().join("db/logs.db");
            let store = Arc::new(LogStore::open(&db_path).await.unwrap());

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
        });
    }

    #[test]
    fn test_like_search_with_special_chars() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let dir = tempfile::TempDir::new().unwrap();
            let db_path = dir.path().join("db/logs.db");
            let store = Arc::new(LogStore::open(&db_path).await.unwrap());

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
        });
    }
}
