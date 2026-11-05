//! Turso-backed log storage.
//!
//! Each log entry is inserted asynchronously via a background channel task; a
//! broadcast channel feeds live entries to the Iced dashboard.
//!
//! # Row scope
//!
//! A row's *scope* — the agent, that agent's role, the workspace and the ticket — is a
//! column of the row, never a field of the record: [`parse_tracing_json`] promotes every
//! spelling of those values out of the record and its span chain and drops the record's
//! own copy, so a value lives in the scope or in `fields`, never both. A record with no
//! such context (a process-level line, a failure record written without a workspace on
//! purpose) carries none, and nothing is invented for it.
//!
//! `workspace` is the workspace's own name — `personal:<user>` for an account's folder —
//! never an absolute path. Write-time only: rows written before the rule keep what they
//! hold, and age out with retention except the warnings, which are never pruned.

use crate::db;
use crate::util::UnwrapPoison;
use crate::util::json;
use anyhow::Context;
use db::{Row, Value, params};
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use std::io;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::OnceCell;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::{EnvFilter, Layer, fmt, layer::SubscriberExt};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    #[serde(default)]
    pub ticket: String,
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
        TICKET      => "ticket",
    }
}

impl LogEntry {
    /// Whether this row matches the Logs page's search text, over the same parts as
    /// [`SEARCH_PARTS`]. An empty query matches every row.
    #[must_use]
    pub fn matches_search(&self, query: &str) -> bool {
        let query = query.to_lowercase();
        query.is_empty()
            || SEARCH_PARTS
                .iter()
                .any(|(_, value)| value(self).to_lowercase().contains(&query))
    }
}

/// Turso-backed log store.
///
/// NOTE: opened manually inside [`init_tracing()`] rather than through `define_store!`, because
/// bootstrapping order needs logs available before the other stores. See [`LOG_STORE`].
#[derive(Clone, Debug)]
pub struct LogStore {
    pub(crate) conn: crate::db::Connection,
}

/// Global log store, set during [`init_tracing()`].
///
/// It does NOT have an `init_global()` like other stores, and must not get one: a store file is
/// opened once per process, because closing ANY descriptor for it drops every record lock the
/// process holds (see [`crate::db::wal_guard`]). One `Database` may serve many connections —
/// which is how the core store's readers work — but a second *open* of the file must never
/// happen. This store keeps no reader connections of its own and serves its writes on its own
/// writing connection: the tracing writer's batches, the telemetry drain's records
/// ([`crate::telemetry`]), the retention pass, the views the Logs page runs, and the periodic
/// round's maintenance one ([`crate::db::Maintenance`]). [`init_tracing()`] also returns the
/// store to its caller, and it travels on to the dashboard, so both share one connection.
pub static LOG_STORE: OnceCell<LogStore> = OnceCell::const_new();

/// One greppable shell call's grep-engine decision, persisted to the dedicated `grep_telemetry`
/// table (self-contained — no `tool_calls.id` correlation). The caller builds it from the serve
/// outcomes and the runtime facts; every platform's grep engine reports.
#[derive(Debug)]
pub(crate) struct GrepTelemetryRow {
    pub command: String,
    pub served: bool,
    pub reason: String,
    pub recursive: bool,
    pub piped: bool,
    pub operand_count: usize,
    pub flags: String,
    pub mode: String,
    pub workspace: String,
    pub grep_count: usize,
    pub served_count: usize,
    pub skipped_count: usize,
    pub duration_ms: Option<i64>,
    pub exit_code: Option<i32>,
    /// The instant the search ran — stamped where it ran, not where the row is written.
    pub recorded_at: String,
}

/// The facts of one event that outlived the call that produced it — the process a command or a
/// chrome-use call left running — recorded by [`record_stray_event`] as ONE row of its own. A
/// chrome-use call that started itself needs none: it fails, and its failure text is its record.
pub(crate) struct StrayEvent<'a> {
    /// The surface that reported it ([`STRAY_SOURCE_SHELL`], [`STRAY_SOURCE_CHROME_TOOL`]).
    pub source: &'a str,
    /// What outlived the call, stated plainly — the row's `message`. Metadata only: no raw
    /// command lines, no page content, no credentials.
    pub message: &'a str,
    /// The stop recipe for what was left running, carried verbatim into `fields` as `detail` —
    /// even where the call's own note offers none, because the mode's guard keeps it out there.
    pub recipe: &'a str,
    /// What must be addressed to stop it (a leftover's process group root, or the chrome-use
    /// session whose helper was left behind); empty when unknown.
    pub scope: &'a str,
    /// The physical chrome-use session driven; empty for the shell.
    pub session: &'a str,
    /// The workspace the call ran in — empty only where the surface has none. A path is
    /// accepted: [`record_stray_event`] normalizes it to the workspace's name.
    pub workspace: &'a str,
    /// How long the recorded call took, when the surface measured it.
    pub duration_ms: Option<i64>,
}

/// The durable record's `target` — the surface that reported the event.
pub(crate) const STRAY_SOURCE_SHELL: &str = "shell";
pub(crate) const STRAY_SOURCE_CHROME_TOOL: &str = "chrome-tool";

impl LogStore {
    /// Open (or create) the log database at `root/db/logs.db`. `pub(crate)` so tests elsewhere
    /// can create a real store ([`crate::open_test_store!`]). An existing store must be usable
    /// ([`crate::db::open_store`] refuses otherwise); a missing file is a first launch.
    pub(crate) async fn open(root: &Path) -> anyhow::Result<Self> {
        let conn = crate::db::open_store(root, "logs", "").await?;
        // The catalog owns the logs schema; a catalog failure is a hard boot failure.
        crate::db::migrations::run_migrations(&conn, crate::db::migrations::TargetDb::Logs).await?;
        Ok(Self { conn })
    }

    /// Insert a batch of log entries in a single transaction: a per-entry commit would fsync the
    /// WAL once each, which dominates the insert cost. On failure the whole batch is dropped —
    /// log entries are diagnostics, not durable state. A caller that cannot lose a row inserts
    /// here directly instead.
    pub(crate) async fn insert_batch(&self, entries: &[LogEntry]) -> anyhow::Result<()> {
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
                "INSERT INTO logs (timestamp, level, target, message, fields, agent_id, agent_role, workspace, ticket) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
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
                    entry.ticket.clone(),
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

    /// Delete stored log records in one Logs view's scope, returning the number of rows deleted.
    /// `level_filter` uses [`LogQuery::level`] syntax; `None` means every level (All Logs).
    ///
    /// The predicate comes from the same [`build_where_clause`] as the listing query and holds
    /// the level filter alone: the search text is deliberately never part of it, because the
    /// button empties the whole view rather than the searched subset. A queued warning row at a
    /// level this scope covers goes with it ([`crate::telemetry::wipe`]). The tracing writer's
    /// batched lines and [`write_warn_row`] are not queued, so a line produced around the wipe
    /// can still land after it.
    pub(crate) async fn clear_logs(&self, level_filter: Option<&str>) -> anyhow::Result<u64> {
        let filters = LogQuery {
            level: level_filter.map(str::to_owned),
            ..LogQuery::default()
        };
        let (where_sql, values) = build_where_clause(&filters);
        let levels = level_filter.map_or_else(Vec::new, level_names);
        crate::telemetry::wipe(
            |record| match record {
                crate::telemetry::TelemetryRecord::WarningRow(entry) => {
                    // Keep only a row this scope leaves in the store.
                    !level_in_scope(&levels, &entry.level)
                }
                _ => true,
            },
            async {
                self.conn
                    .execute(&format!("DELETE FROM logs {where_sql}"), values)
                    .await
                    .context("Failed to clear log entries")
            },
        )
        .await
    }

    /// Whether any row carrying `message` holds `reason` in its `reason` field. The rows
    /// carrying one message are one per distinct reason, so they are read and parsed here
    /// rather than matched in SQL against the JSON they store.
    pub(crate) async fn has_reason(&self, message: &str, reason: &str) -> anyhow::Result<bool> {
        /// The `reason` field of a row's stored `fields`, or `None` when it holds none.
        fn reason_of(fields: &str) -> Option<String> {
            serde_json::from_str::<serde_json::Value>(fields)
                .ok()?
                .get("reason")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        }
        let rows = self
            .conn
            .query(
                "SELECT fields FROM logs WHERE message = ?1",
                params![message],
            )
            .await
            .context("Failed to read the log rows carrying a message")?;
        for row in &rows {
            // The one selected column, so its position is its own.
            if reason_of(&row.get::<String>(0)?).as_deref() == Some(reason) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Query log entries with optional filters. Search is a plain substring match over the
    /// row's readable parts — module, message, workspace, ticket ([`build_where_clause`]).
    /// Returns `(entries, total_count)`, the count matching the same filters.
    pub async fn query(&self, filters: &LogQuery) -> anyhow::Result<(Vec<LogEntry>, usize)> {
        let (where_sql, values) = build_where_clause(filters);

        let count_sql = format!("SELECT COUNT(*) FROM logs {where_sql}");
        let total = self
            .conn
            .query_row(&count_sql, values.clone(), |row| row.get::<i64>(0))
            .await
            .map(|n| usize::try_from(n).unwrap_or(0))?;
        if total == 0 {
            return Ok((vec![], 0));
        }

        let limit: i64 = i64::try_from(filters.limit.unwrap_or(100).min(1000))
            .expect("log query limit overflowed i64; limit must be <= i64::MAX");
        let offset: i64 = i64::try_from(filters.offset.unwrap_or(0))
            .expect("log query offset overflowed i64; offset must be <= i64::MAX");
        let mut data_values = values;
        data_values.push(Value::Integer(limit));
        data_values.push(Value::Integer(offset));

        let data_sql = format!(
            "SELECT {LOGS_COLUMNS} FROM logs {where_sql} ORDER BY id DESC LIMIT ? OFFSET ?",
        );
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

    /// Persist one greppable shell call's grep-engine decision. Metadata only; the telemetry
    /// drain task is the caller and drops the row on a failure.
    pub(crate) async fn record_grep_telemetry(&self, row: &GrepTelemetryRow) -> anyhow::Result<()> {
        self.conn
            .execute(
                "INSERT INTO grep_telemetry \
                 (recorded_at, command, served, reason, recursive, piped, operand_count, flags, \
                  mode, workspace, grep_count, served_count, skipped_count, duration_ms, exit_code) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
                params![
                    row.recorded_at.clone(),
                    row.command.clone(),
                    i64::from(row.served),
                    row.reason.clone(),
                    i64::from(row.recursive),
                    i64::from(row.piped),
                    i64::try_from(row.operand_count)?,
                    row.flags.clone(),
                    row.mode.clone(),
                    row.workspace.clone(),
                    i64::try_from(row.grep_count)?,
                    i64::try_from(row.served_count)?,
                    i64::try_from(row.skipped_count)?,
                    row.duration_ms,
                    row.exit_code.map(i64::from),
                ],
            )
            .await?;
        Ok(())
    }
}

/// Record one event that outlived its call as ONE row of its own: a `WARN` row, which the
/// Issues view shows and retention keeps, and which the tracing writer never carries because
/// that writer batches lines. The row says what happened and carries the recipe that stops it.
/// Filed through [`record_warn_row`], so a process with no logs store (the `mahbot chrome` CLI
/// dispatches before [`init_tracing`] runs) records nothing.
pub(crate) fn record_stray_event(event: &StrayEvent<'_>) {
    record_warn_row(
        event.source,
        event.message,
        serde_json::json!({
            "detail": event.recipe,
            "scope": event.scope,
            "session": event.session,
            "duration_ms": event.duration_ms,
        }),
        event.workspace,
    );
}

/// File ONE durable `WARN` row of its own in the Issues view, straight through the telemetry
/// queue — never the batching tracing writer — for a fact that must outlive the call that
/// produced it. `fields` is the row's own object and `workspace` a path or a name, normalised
/// here. The row is attributed to the calling agent here
/// ([`crate::agent::tool_record_attribution`]), never by the caller. Never deduped, unlike
/// [`record_issue_once`]: each call is one occurrence, and counting them is what the row is for.
/// Fail-open: a process with no logs store files nothing.
pub(crate) fn record_warn_row(
    target: &str,
    message: &str,
    fields: serde_json::Value,
    workspace: &str,
) {
    if !crate::telemetry::recording() {
        return;
    }
    let attribution = crate::agent::tool_record_attribution();
    crate::telemetry::enqueue(crate::telemetry::TelemetryRecord::WarningRow(LogEntry {
        timestamp: crate::db::now(),
        level: "WARN".to_string(),
        target: target.to_string(),
        message: message.to_string(),
        fields,
        workspace: workspace_name(workspace),
        agent_id: attribution.agent_id,
        agent_role: attribution.agent_role,
        ticket: attribution.ticket,
    }));
}

/// What became of one fact the Issues view is owed ([`record_issue_once`]).
pub(crate) enum IssueWrite {
    /// This call is what put the row in the store.
    Written,
    /// The store already held the fact: nothing was written now.
    AlreadyRecorded,
    /// No row carries the fact: this process has no logs store ([`LOG_STORE`] unset), or the
    /// write was attempted and failed.
    NotRecorded,
}

/// Write one durable `WARN` row straight to the logs store — never through the tracing writer,
/// which batches lines and may drop them. `WARN` is what the Issues view shows and retention
/// keeps, so this is where a fact the product must not lose outlives the process;
/// [`record_issue_once`] is its one caller. Returns `Ok(())` when this call put the row in the
/// store, `Err` with the store's own error text when the write was attempted and failed.
async fn write_warn_row(store: &LogStore, entry: LogEntry) -> Result<(), String> {
    let entry = LogEntry {
        timestamp: crate::db::now(),
        level: "WARN".to_string(),
        ..entry
    };
    store
        .insert_batch(&[entry])
        .await
        .map_err(|e| e.to_string())
}

/// Record one durable row for the product's own Issues view, at most once for a
/// `(message, reason)` pair: written straight to the store at `WARN`, which the Issues view
/// shows and retention never reaps, so a repeat is a duplicate rather than news. A store read
/// that fails records anyway and says so at `WARN`. Fail-open: a process with no store gets
/// [`IssueWrite::NotRecorded`].
///
/// `fields` is the row's `fields` object; `reason` is written into it here, so the dedupe key
/// and the stored row can never disagree. The row has no tool attribution (empty agent and
/// workspace): it comes from a background task, not a tool call.
///
/// Reports what became of the fact ([`IssueWrite`]). On a failed write this warns with the
/// store's own error text — a warning about the store, not about what was lost — and a caller
/// whose process exits immediately after adds its own line naming the fact.
pub(crate) async fn record_issue_once(
    message: &str,
    reason: &str,
    target: &str,
    fields: serde_json::Value,
) -> IssueWrite {
    let Some(store) = LOG_STORE.get() else {
        return IssueWrite::NotRecorded;
    };
    match store.has_reason(message, reason).await {
        // This reason is already in the Issues view: one fact, told once.
        Ok(true) => return IssueWrite::AlreadyRecorded,
        Ok(false) => {}
        Err(e) => tracing::warn!(
            error = %e,
            "could not read whether this issue was already recorded — recording it again"
        ),
    }
    // `reason` goes into the row's own `fields` here, so the dedupe key ([`LogStore::has_reason`])
    // and the stored row can never disagree.
    let mut fields = fields;
    if let Some(object) = fields.as_object_mut() {
        object.insert(
            "reason".to_string(),
            serde_json::Value::String(reason.to_string()),
        );
    }
    match write_warn_row(
        store,
        LogEntry {
            target: target.to_string(),
            message: message.to_string(),
            fields,
            ..LogEntry::default()
        },
    )
    .await
    {
        Ok(()) => IssueWrite::Written,
        Err(error) => {
            tracing::warn!(%error, "could not record an issue for the Issues view");
            IssueWrite::NotRecorded
        }
    }
}

/// Parameters for filtering log queries.
#[derive(Debug, Clone, Default)]
pub struct LogQuery {
    pub level: Option<String>,
    pub target: Option<String>,
    pub search: Option<String>,
    pub since: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// The level names a [`LogQuery::level`] filter holds, parsed as the listing query parses them.
fn level_names(level_filter: &str) -> Vec<String> {
    level_filter
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Whether a row at `level` is inside a clear scope — the Rust side of the `level IN (…)`
/// [`build_where_clause`] writes. An empty scope is every level (All Logs).
fn level_in_scope(levels: &[String], level: &str) -> bool {
    levels.is_empty() || levels.iter().any(|covered| covered == level)
}

/// One part of the Logs page's search: the column it is stored in, and where the entry holds it.
type SearchPart = (&'static str, fn(&LogEntry) -> &str);

/// The row's parts the search covers: module and message, plus the two scope values —
/// workspace and ticket. Both halves of the search read this list.
const SEARCH_PARTS: [SearchPart; 4] = [
    ("target", |entry| &entry.target),
    ("message", |entry| &entry.message),
    ("workspace", |entry| &entry.workspace),
    ("ticket", |entry| &entry.ticket),
];

/// A `LIKE` pattern matching `text` literally anywhere inside a column: `%`, `_` and the escape
/// character itself are escaped, so a wildcard typed into the box is that character.
fn like_contains(text: &str) -> String {
    let mut pattern = String::with_capacity(text.len() + 2);
    pattern.push('%');
    for ch in text.chars() {
        if matches!(ch, '%' | '_' | '\\') {
            pattern.push('\\');
        }
        pattern.push(ch);
    }
    pattern.push('%');
    pattern
}

/// Build WHERE clause and bind values from `LogQuery` filters.
/// Returns `(WHERE ...`, `[values]`) — an empty string when no filters are set.
fn build_where_clause(filters: &LogQuery) -> (String, Vec<Value>) {
    let mut conditions: Vec<String> = Vec::new();
    let mut values: Vec<Value> = Vec::new();

    if let Some(ref levels_str) = filters.level
        && !levels_str.is_empty()
    {
        let levels: Vec<Value> = level_names(levels_str)
            .into_iter()
            .map(Value::Text)
            .collect();
        if !levels.is_empty() {
            conditions.push(format!(
                "level IN ({})",
                db::sql_in_placeholders(levels.len()),
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
        let predicate = SEARCH_PARTS
            .iter()
            .map(|(column, _)| format!("{column} LIKE ? ESCAPE '\\'"))
            .collect::<Vec<_>>()
            .join(" OR ");
        conditions.push(format!("({predicate})"));
        values.extend(std::iter::repeat_n(
            Value::Text(like_contains(search)),
            SEARCH_PARTS.len(),
        ));
    }

    if let Some(ref since) = filters.since {
        conditions.push("timestamp >= ?".into());
        values.push(Value::Text(since.clone()));
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
    let ticket = row.get::<String>(COL_LOGS_TICKET)?;

    Ok(LogEntry {
        timestamp,
        level,
        target,
        message,
        fields,
        agent_id,
        agent_role,
        workspace,
        ticket,
    })
}

/// The default filter, applied as a LAYER filter on the JSON log layer — see [`log_layers`] for
/// why it must not be a global subscriber filter. `tantivy` reports through the standard `log`
/// interface, so its directive only matches records that arrive over the bridge
/// [`install_log_bridge`] installs.
///
/// `fff` mutes the file-search library the tools and the editor search with — every module of
/// the family, since EnvFilter matches a target as a raw prefix. It must stay `off` with no
/// narrower directive: the most specific directive wins, so `fff_search::grep=error` would
/// re-open that module. A level cut is not enough — the pre-existing `fff_search=error` let the
/// library's in-memory file table, once full after a burst of new files, write 24 473 identical
/// ERROR rows in under two seconds, which buried every real problem in the owner's view. The
/// product's own record for that condition is the edit tool's notice that a written file could
/// not be indexed (target `mahbot::tools::edit`), and it is unaffected.
///
/// `pdf_extract` warns once per page on a font table whose glyphs and Unicode disagree, so a
/// single document produces hundreds of near-identical records: it is held to ERROR.
///
/// `usvg`/`resvg` complain per input about one caller's file rather than anything the product
/// did; their quieter levels are off entirely and a real refusal reaches the caller as that
/// tool's own answer.
///
/// `iced`'s warning channel reports recoverable progress (a published message, an invalidated
/// widget) that the product's own handlers never produce: every layout invalidation sits on an
/// input event and there is no repaint loop. It is held to ERROR too, and one directive covers
/// every `iced_*` crate.
pub(crate) const DEFAULT_LOG_FILTER: &str =
    "info,turso_core=warn,tantivy=warn,fff=off,pdf_extract=error,usvg=off,resvg=off,iced=error";

/// The production layer stack: the JSON log layer, with the log filter applied PER LAYER, plus
/// the engine-cause capture layer ([`crate::db::checkpoint_cause`]). The filter must be a layer
/// filter and never the subscriber's: installed globally it would drop the engine's DEBUG
/// checkpoint-failure event. Extracted from [`init_tracing`] so a test can install the real stack.
pub(crate) fn log_layers<W>(
    writer: W,
    env_filter: EnvFilter,
) -> impl tracing::Subscriber + Send + Sync + 'static
where
    W: for<'a> fmt::MakeWriter<'a> + Send + Sync + 'static,
{
    tracing_subscriber::registry()
        .with(
            fmt::Layer::new()
                .json()
                .with_writer(writer)
                .with_ansi(false)
                .with_filter(env_filter),
        )
        .with(
            crate::db::checkpoint_cause::CauseCaptureLayer
                .with_filter(crate::db::checkpoint_cause::filter()),
        )
}

/// Install the `log` → `tracing` bridge: the libraries this product depends on report through
/// the standard `log` interface, and without this bridge their records never reach the store. A
/// failed install is a boot failure, because silently losing every dependency's records is the
/// regression the bridge exists to prevent.
///
/// `level` must be the log layer's own filter level ([`log_bridge_level`]) — never
/// [`LevelFilter::current`], which the engine-cause capture layer raises to DEBUG process-wide.
fn install_log_bridge(level: LevelFilter) -> anyhow::Result<()> {
    use tracing_log::AsLog;

    tracing_log::LogTracer::builder()
        .with_max_level(level.as_log())
        .init()
        .map_err(|e| anyhow::anyhow!("failed to install the log→tracing bridge: {e}"))
}

/// The level the `log` bridge gates on: the log layer's own filter level, read before the filter
/// moves into [`log_layers`]. A filter with no level hint leaves the bridge at `TRACE`: the
/// layers still decide what is written.
fn log_bridge_level(env_filter: &EnvFilter) -> LevelFilter {
    env_filter.max_level_hint().unwrap_or(LevelFilter::TRACE)
}

/// Initialize tracing: JSON to the Turso store only (no terminal output). Returns the
/// [`LogStore`] for querying and a broadcast sender for live streaming to the dashboard.
pub async fn init_tracing(
    storage_root: &Path,
) -> anyhow::Result<(Arc<LogStore>, tokio::sync::broadcast::Sender<String>)> {
    let store = match LogStore::open(storage_root).await {
        Ok(store) => store,
        Err(e) => {
            // The pre-tracing diagnostics were already handed to stderr; the logs-store replay
            // can never run, so the buffer is dropped.
            crate::boot::clear_boot_diagnostics();
            return Err(e);
        }
    };
    LOG_STORE
        .set(store.clone())
        .map_err(|_| anyhow::anyhow!("LOG_STORE already initialized"))?;
    // The telemetry queue and its drain task start with the store they write to: a record
    // produced before this is dropped without being counted ([`crate::telemetry`]).
    crate::telemetry::start();
    let log_store = Arc::new(store);
    let (log_tx, log_rx) = tokio::sync::mpsc::unbounded_channel();
    let (broadcast_tx, _) = tokio::sync::broadcast::channel(256);

    spawn_log_writer(Arc::clone(&log_store), log_rx, broadcast_tx.clone());

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER));
    let bridge_level = log_bridge_level(&env_filter);
    tracing::subscriber::set_global_default(log_layers(make_log_writer(log_tx), env_filter))
        .expect("Unable to install the global tracing subscriber");
    install_log_bridge(bridge_level)?;
    crate::boot::mark_tracing_initialized();

    // Surface the pre-tracing boot diagnostics in the logs store for the GUI boot log.
    crate::boot::replay_boot_diagnostics();

    Ok((log_store, broadcast_tx))
}

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

/// Maximum number of log entries accumulated before a forced DB flush: one flush inserts at
/// most this many rows in a single transaction.
const LOG_BATCH_MAX: usize = 50;

/// Maximum age of an accumulated batch before a timer flush, so the insert path stays fresh
/// under low log volume while the live broadcast stays per-message.
const LOG_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);

/// Spawn a background task that receives JSON log lines and writes them to Turso and broadcasts
/// them to the dashboard. Only the DB-insert path batches ([`LOG_BATCH_MAX`]-sized, or on the
/// timer); the GUI broadcast stays per-message.
///
/// A storage-layer panic (e.g. the shared-WAL frame-index invariant violation) must not silently
/// freeze persistence: the flush is panic-absorbed with a bounded number of restarts and
/// backoff, recorded on the visible failure surface, and past the bound the writer stops
/// flushing with a terminal banner instead of spinning on a broken connection. During backoff
/// the writer is not polling, so lines accumulate in the unbounded channel — bounded by the
/// backoff schedule. In the stopped state it keeps draining (broadcast + drop).
fn spawn_log_writer(
    store: Arc<LogStore>,
    rx: UnboundedReceiver<String>,
    broadcast: tokio::sync::broadcast::Sender<String>,
) {
    spawn_log_writer_with_interval(store, rx, broadcast, LOG_FLUSH_INTERVAL);
}

/// [`spawn_log_writer`] with an explicit flush interval — tests inject a very long or very
/// short one to exercise the batch-cap and timer paths without racing the production timer.
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
        // The first interval tick fires immediately — consume it so the timer only fires after
        // the first full interval.
        flush_timer.tick().await;

        loop {
            tokio::select! {
                maybe_line = rx.recv() => {
                    let Some(line) = maybe_line else {
                        // Channel closed (tracing teardown on shutdown) — flush and exit.
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
                        // Terminal state: keep draining and broadcasting, but drop entries
                        // instead of pushing them onto a batch that will never be flushed.
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

/// Flush the accumulated batch, absorbing storage-layer panics: a panic here means a broken
/// connection, so it is recorded on the failure surface, the batch is dropped and the writer
/// backs off. After [`LOG_WRITER_MAX_CONSECUTIVE_PANICS`] it enters the terminal stopped state;
/// a successful flush resets the counter.
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
                crate::boot::timestamped_stderr(&format!(
                    "log store writer stopped after {consecutive} consecutive storage \
                     panics: {message}"
                ));
            } else {
                tokio::time::sleep(log_writer_panic_backoff(consecutive)).await;
            }
        }
    }
}

/// Insert all accumulated entries in one transaction and clear the batch. On persistent failure
/// the batch is still cleared — log entries are diagnostics. Failures are **not** swallowed:
/// they are recorded on the [`log_write_error_info`] surface (the GUI Logs page banner) and
/// reported to stderr at a bounded rate. No `tracing!` call is made from here — the writer task
/// consumes the tracing channel, so tracing from inside it would recurse.
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

/// Maximum number of insert attempts (including the first) for one log batch. Retrying is safe:
/// the batch is only dropped after all attempts fail.
const LOG_INSERT_MAX_ATTEMPTS: usize = 3;

/// Backoff between log-batch insert retry attempts.
const LOG_INSERT_RETRY_BACKOFF: std::time::Duration = std::time::Duration::from_millis(250);

/// Minimum interval between stderr warnings about log-write failures.
const LOG_WRITE_STDERR_WARN_INTERVAL_MS: u64 = 60_000;

/// Consecutive storage-panic restarts before the writer stops flushing permanently: retrying
/// past this bound would spin forever on a connection that keeps panicking.
const LOG_WRITER_MAX_CONSECUTIVE_PANICS: u32 = 5;

/// Base backoff after a storage panic (doubles per consecutive panic, capped).
const LOG_WRITER_PANIC_BACKOFF_MS: u64 = 500;

/// Consecutive-panic restart state machine for the log writer. Pure and unit-testable; the
/// global writer state ([`LOG_WRITE_LAST_ERROR`]) mirrors it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LogWriterPanicState {
    /// Consecutive storage-panic restarts in progress (0 when the last flush succeeded).
    consecutive_panics: u32,
    /// True once the writer stopped flushing permanently — the terminal banner state.
    pub writer_stopped: bool,
}

impl LogWriterPanicState {
    /// Record a storage panic; returns the updated consecutive-panic count.
    #[must_use]
    fn record_panic(&mut self) -> u32 {
        self.consecutive_panics += 1;
        if self.consecutive_panics >= LOG_WRITER_MAX_CONSECUTIVE_PANICS {
            self.writer_stopped = true;
        }
        self.consecutive_panics
    }

    /// Reset the consecutive-panic counter after a successful flush. The terminal stopped state
    /// is sticky: a stopped writer never flushes again.
    fn reset(&mut self) {
        self.consecutive_panics = 0;
    }
}

/// Snapshot of the log-writer failure surface, for display and tests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct LogWriteErrorInfo {
    /// Total failures recorded since startup — batch insert failures, writer panics, and panics
    /// out of one telemetry record's own write.
    pub count: u64,
    /// Rows the telemetry queue dropped because it was full ([`crate::telemetry`]), counted
    /// apart from write failures: a healthy store can also drop for want of room.
    pub dropped: u64,
    /// RFC 3339 timestamp of the most recent failure.
    pub last_timestamp: Option<String>,
    /// Message of the most recent failure.
    pub last_message: Option<String>,
    /// Writer panic-restart state.
    pub panic_state: LogWriterPanicState,
}

/// Most recent log-batch insert failure observed by the writer task. Count, timestamp and
/// message live behind one mutex so readers never observe a torn pair.
static LOG_WRITE_LAST_ERROR: std::sync::Mutex<LogWriteErrorInfo> =
    std::sync::Mutex::new(LogWriteErrorInfo {
        count: 0,
        dropped: 0,
        last_timestamp: None,
        last_message: None,
        panic_state: LogWriterPanicState {
            consecutive_panics: 0,
            writer_stopped: false,
        },
    });

/// Rate-limited stderr reporting for this surface. stderr is not routed through tracing, so
/// this cannot recurse into the writer task, and it is best-effort delivery — which is why
/// [`log_write_error_info`] is the surface a user can always read. One reporter per kind of
/// line: a burst of one kind must not hide the other.
struct StderrReporter {
    last_ms: AtomicU64,
}

impl StderrReporter {
    const fn new() -> Self {
        Self {
            last_ms: AtomicU64::new(0),
        }
    }

    /// Print the line `line` builds unless this reporter's last one went out within
    /// [`LOG_WRITE_STDERR_WARN_INTERVAL_MS`]. Built only when it will be printed.
    fn emit(&self, line: impl FnOnce() -> String) {
        let now_ms = crate::util::unix_millis();
        if now_ms.saturating_sub(self.last_ms.load(Ordering::SeqCst))
            >= LOG_WRITE_STDERR_WARN_INTERVAL_MS
        {
            self.last_ms.store(now_ms, Ordering::SeqCst);
            crate::boot::timestamped_stderr(&line());
        }
    }
}

/// The line about a log-store write failure.
static LOG_WRITE_STDERR: StderrReporter = StderrReporter::new();

/// The line about telemetry records the queue dropped.
static TELEMETRY_DROP_STDERR: StderrReporter = StderrReporter::new();

/// Rows the telemetry queue dropped that no line has reported yet. The producing path only
/// adds here; the drain task prints ([`report_telemetry_drops`]).
static TELEMETRY_DROPS_UNREPORTED: AtomicU64 = AtomicU64::new(0);

/// Read the log-writer failure surface — the sanctioned one, which the GUI Logs page renders a
/// banner from. Safe to call from anywhere, including inside the writer task.
#[must_use]
pub(crate) fn log_write_error_info() -> LogWriteErrorInfo {
    LOG_WRITE_LAST_ERROR.lock().unwrap_poison().clone()
}

/// Record a failed log-batch insert on the observability surface.
fn record_log_write_failure(error: Option<anyhow::Error>) {
    let message = error.map_or_else(
        || "unknown log insert failure".to_string(),
        |e| format!("{e:#}"),
    );
    record_log_write_failure_impl(&message, LogFailureKind::Insert);
}

/// Record a storage-layer panic absorbed by the writer. Returns the updated consecutive-panic
/// count. The terminal stopped state additionally gets an unconditional banner from the caller.
fn record_log_writer_panic(message: &str) -> u32 {
    record_log_write_failure_impl(message, LogFailureKind::WriterPanic)
}

/// Record a storage-layer panic absorbed by the telemetry drain task's own write of one record
/// ([`crate::telemetry`]). A broken connection must not take the drain task with it: the record
/// is dropped like any failed write and the next one is still attempted.
pub(crate) fn record_telemetry_write_panic(message: &str) {
    record_log_write_failure_impl(message, LogFailureKind::RecordPanic);
}

/// Which failure kind is being recorded — drives the stderr label and whether the
/// storage-panic restart counter is bumped.
#[derive(Clone, Copy)]
enum LogFailureKind {
    Insert,
    WriterPanic,
    /// A panic out of one telemetry record's own write — the drain task's, so it leaves the
    /// restart counter alone.
    RecordPanic,
}

impl LogFailureKind {
    fn label(self) -> &'static str {
        match self {
            Self::Insert => "insert failure",
            Self::WriterPanic => "writer panic",
            Self::RecordPanic => "record write panic",
        }
    }

    fn records_panic(self) -> bool {
        matches!(self, Self::WriterPanic)
    }
}

/// Record rows the telemetry queue dropped because it was full ([`crate::telemetry`]).
///
/// Not a write failure and deliberately not counted as one: the store may be perfectly healthy
/// while a burst has nowhere to go, so the two numbers are kept apart and reported through the
/// same surface — the GUI Logs page banner and the line [`report_telemetry_drops`] prints.
pub(crate) fn record_telemetry_drop(rows: u64) {
    LOG_WRITE_LAST_ERROR.lock().unwrap_poison().dropped += rows;
    TELEMETRY_DROPS_UNREPORTED.fetch_add(rows, Ordering::Relaxed);
}

/// Print the line about the drops the producing path only counted — the drain task's job, so no
/// thread doing the work writes to stderr about its own bookkeeping. A report the limiter
/// suppresses is still in the counter.
pub(crate) fn report_telemetry_drops() {
    let rows = TELEMETRY_DROPS_UNREPORTED.swap(0, Ordering::Relaxed);
    if rows == 0 {
        return;
    }
    let total = LOG_WRITE_LAST_ERROR.lock().unwrap_poison().dropped;
    TELEMETRY_DROP_STDERR
        .emit(|| format!("telemetry queue full: {rows} row(s) dropped, {total} in total"));
}

/// Shared failure-recording core: bumps the counter, stamps timestamp and message, optionally
/// records a storage-panic restart, and emits a rate-limited stderr warning.
fn record_log_write_failure_impl(message: &str, kind: LogFailureKind) -> u32 {
    let (count, consecutive) = {
        let mut guard = LOG_WRITE_LAST_ERROR.lock().unwrap_poison();
        guard.count += 1;
        guard.last_timestamp = Some(db::now());
        guard.last_message = Some(message.to_string());
        let consecutive = if kind.records_panic() {
            guard.panic_state.record_panic()
        } else {
            guard.panic_state.consecutive_panics
        };
        (guard.count, consecutive)
    };
    LOG_WRITE_STDERR.emit(|| format!("log store {} #{count}: {message}", kind.label()));
    consecutive
}

/// Reset the consecutive-panic counter after a successful flush. A stopped writer never flushes
/// again, so only the counter is cleared.
fn reset_log_writer_panic_state() {
    let mut guard = LOG_WRITE_LAST_ERROR.lock().unwrap_poison();
    guard.panic_state.reset();
}

/// True once the writer has permanently stopped flushing (terminal banner).
fn log_writer_stopped() -> bool {
    LOG_WRITE_LAST_ERROR
        .lock()
        .unwrap_poison()
        .panic_state
        .writer_stopped
}

/// Backoff after the `n`-th consecutive storage panic: 500ms, 1s, 2s, … capped at 30s. The terminal
/// bound ends the sequence before the cap engages today; the cap guards a future bound increase.
fn log_writer_panic_backoff(consecutive: u32) -> std::time::Duration {
    let shift = consecutive.saturating_sub(1).min(6);
    let ms = LOG_WRITER_PANIC_BACKOFF_MS.saturating_mul(1 << shift);
    std::time::Duration::from_millis(ms.min(30_000))
}

/// Extract a string field from a JSON value, defaulting to `""`.
fn get_str_or_empty(val: &serde_json::Value, key: &str) -> String {
    json::get_opt_str(val, key).unwrap_or("").to_string()
}

/// Parse a tracing-subscriber JSON line into a `LogEntry`, promoting the row's scope out of the
/// record and its span chain.
fn parse_tracing_json(line: &str) -> Option<LogEntry> {
    let val: serde_json::Value = serde_json::from_str(line).ok()?;

    let timestamp = get_str_or_empty(&val, "timestamp");
    let level = get_str_or_empty(&val, "level");
    let target = get_str_or_empty(&val, "target");

    let scope = extract_scope(&val);

    let mut fields = val
        .get("fields")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let message = get_str_or_empty(&fields, "message");

    if let Some(obj) = fields.as_object_mut() {
        obj.remove("message");
        strip_scope_fields(obj);
    }
    let fields = if fields.as_object().is_some_and(serde_json::Map::is_empty) {
        serde_json::Value::Null
    } else {
        fields
    };

    Some(LogEntry {
        timestamp,
        level,
        target,
        message,
        fields,
        agent_id: scope.agent_id,
        agent_role: scope.agent_role,
        workspace: scope.workspace,
        ticket: scope.ticket,
    })
}

/// The field names each scope value may be carried under, most specific first: a record's own
/// copy of any of them is the value the row's column carries.
const AGENT_ID_KEYS: &[&str] = &["agent_id"];
const AGENT_ROLE_KEYS: &[&str] = &["role", "agent_role"];
const WORKSPACE_KEYS: &[&str] = &["workspace", "workspace_name", "workspace_path"];
const TICKET_KEYS: &[&str] = &["ticket", "ticket_id"];

/// A row's scope: the agent, its role, the workspace and the ticket — the values the row
/// carries itself and the Logs page shows as its context.
#[derive(Default)]
struct LogScope {
    agent_id: String,
    agent_role: String,
    workspace: String,
    ticket: String,
}

impl LogScope {
    /// Fill whatever this scope does not hold yet from one source, leaving the rest alone.
    fn promote(&mut self, source: &serde_json::Value) {
        fill(&mut self.agent_id, source, AGENT_ID_KEYS);
        fill(&mut self.agent_role, source, AGENT_ROLE_KEYS);
        fill(&mut self.ticket, source, TICKET_KEYS);
        if self.workspace.is_empty()
            && let Some(value) = first_value(source, WORKSPACE_KEYS)
        {
            self.workspace = workspace_name(value);
        }
    }
}

/// Fill `slot` from the first non-empty value `source` holds under `keys`.
fn fill(slot: &mut String, source: &serde_json::Value, keys: &[&str]) {
    if slot.is_empty() {
        *slot = first_value(source, keys).unwrap_or_default().to_string();
    }
}

/// The first non-empty string `source` holds under `keys`.
fn first_value<'a>(source: &'a serde_json::Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| json::get_opt_str(source, key).filter(|value| !value.is_empty()))
}

/// Promote the row's scope out of the record's own fields and its span chain: the record's own
/// fields win where present, then the spans innermost outwards, so the agent that is running
/// wins over the context it runs in and each value is taken from the first source that carries
/// it. `tracing-subscriber`'s JSON formatter writes both `span` and `spans` (root to leaf).
fn extract_scope(val: &serde_json::Value) -> LogScope {
    let spans = val
        .get("spans")
        .and_then(serde_json::Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let mut scope = LogScope::default();
    for source in std::iter::once(val.get("fields"))
        .chain(spans.iter().rev().map(Some))
        .chain(std::iter::once(val.get("span")))
        .flatten()
    {
        scope.promote(source);
    }
    scope
}

/// Whether `key` is one of the names a scope value is carried under — such a key is never a
/// field of the record.
fn is_scope_key(key: &str) -> bool {
    [AGENT_ID_KEYS, AGENT_ROLE_KEYS, WORKSPACE_KEYS, TICKET_KEYS]
        .iter()
        .any(|keys| keys.contains(&key))
}

/// Drop the record's own copy of every scope value: the row carries it now, and a scope value
/// is never also a field of the same record.
fn strip_scope_fields(fields: &mut serde_json::Map<String, serde_json::Value>) {
    fields.retain(|key, _| !is_scope_key(key));
}

/// The row's `workspace` value: the workspace's name, never a path. A folder in the userspaces
/// directory ([`crate::users::userspaces_root`]) is an account's personal workspace; any other
/// path's last component is the folder it points at.
fn workspace_name(value: &str) -> String {
    let mut parts = value.rsplit(['/', '\\']).filter(|part| !part.is_empty());
    let Some(last) = parts.next() else {
        return String::new();
    };
    let userspaces = crate::users::userspaces_root();
    if parts.next() == userspaces.file_name().and_then(std::ffi::OsStr::to_str) {
        return crate::users::personal_workspace_name(last);
    }
    last.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The row a written line becomes. A record's scope is the row's, not the record's: every
    /// spelling is taken from the record's own fields where present, then from its span chain, and
    /// never repeated as a field. A workspace recorded as a path becomes the name it stands for.
    /// Each case: name, line, the four scope columns, and the fields the row must be left holding
    /// (`None` where the case makes no claim).
    #[expect(clippy::too_many_lines)] // one table row per record shape
    #[test]
    fn parse_tracing_json_shapes_each_record() {
        struct Case {
            name: &'static str,
            line: &'static str,
            agent_id: &'static str,
            role: &'static str,
            workspace: &'static str,
            ticket: &'static str,
            fields: Option<serde_json::Value>,
        }
        let cases = [
            Case {
                name: "promoted scope out of the record",
                line: r#"{"timestamp":"...","level":"INFO","target":"test",
                    "spans":[
                        {"name":"ticket","ticket":"mahbot-42","workspace":"mahbot"},
                        {"name":"agent","agent_id":"ticket_mahbot-42_engineer","role":"engineer"}
                    ],
                    "fields":{"message":"writing code","agent_id":"ticket_mahbot-42_engineer",
                        "role":"engineer","workspace_name":"/Users/x/Desktop/mahbot/",
                        "ticket_id":"mahbot-42","path":"src/a.rs","attempt":2}}"#,
                agent_id: "ticket_mahbot-42_engineer",
                role: "engineer",
                workspace: "mahbot",
                ticket: "mahbot-42",
                fields: Some(serde_json::json!({"path": "src/a.rs", "attempt": 2})),
            },
            Case {
                name: "ticket from the span and a personal workspace",
                line: r#"{"timestamp":"...","level":"INFO","target":"mahbot::pipeline::verification",
                    "spans":[{"name":"ticket","ticket":"mahbot-7","workspace":"personal:lyuba"},
                        {"name":"agent","agent_id":"cli","role":"cli"}],
                    "fields":{"message":"project commands finished","passed":3}}"#,
                agent_id: "cli",
                role: "cli",
                workspace: "personal:lyuba",
                ticket: "mahbot-7",
                fields: Some(serde_json::json!({"passed": 3})),
            },
            Case {
                name: "span only",
                line: r#"{"timestamp":"...","level":"INFO","target":"test","span":{"name":"agent","agent_id":"abc-123","role":"analyst"},"fields":{"message":"researching"}}"#,
                agent_id: "abc-123",
                role: "analyst",
                workspace: "",
                ticket: "",
                fields: None,
            },
            Case {
                name: "spans array",
                line: r#"{"timestamp":"...","level":"INFO","target":"test","spans":[{"name":"parent"},{"name":"agent","agent_id":"xyz-456","role":"worker","workspace":"my-ws"}],"fields":{"message":"writing code"}}"#,
                agent_id: "xyz-456",
                role: "worker",
                workspace: "my-ws",
                ticket: "",
                fields: None,
            },
            Case {
                name: "event fields without span",
                line: r#"{"timestamp":"...","level":"ERROR","target":"mahbot::agent","fields":{"message":"Agent failed","agent_id":"ticket_123_engineer","role":"engineer","workspace":"my-ws","classification":"transport"}}"#,
                agent_id: "ticket_123_engineer",
                role: "engineer",
                workspace: "my-ws",
                ticket: "",
                fields: None,
            },
            Case {
                name: "event beats inherited span",
                line: r#"{"timestamp":"...","level":"ERROR","target":"mahbot::agent","span":{"name":"agent","agent_id":"caller_42","role":"engineer","workspace":"parent-ws"},"fields":{"message":"Agent failed","agent_id":"analyze_ws_1_2_analyst","role":"analyst","workspace":"my-ws","classification":"runtime"}}"#,
                agent_id: "analyze_ws_1_2_analyst",
                role: "analyst",
                workspace: "my-ws",
                ticket: "",
                fields: None,
            },
            Case {
                name: "workspace-only event keeps span agent",
                line: r#"{"timestamp":"...","level":"WARN","target":"mahbot::tools::edit","span":{"name":"agent","agent_id":"ticket_7_engineer","role":"engineer","workspace":"my-ws"},"fields":{"message":"Search index capacity exhausted","workspace":"my-ws","path":"src/a.rs"}}"#,
                agent_id: "ticket_7_engineer",
                role: "engineer",
                workspace: "my-ws",
                ticket: "",
                fields: None,
            },
            Case {
                name: "agent_id-only event merges span role/workspace",
                line: r#"{"timestamp":"...","level":"WARN","target":"mahbot::agent","span":{"name":"agent","agent_id":"ticket_7_engineer","role":"engineer","workspace":"my-ws"},"fields":{"message":"Failed to persist incoming messages to session DB","agent_id":"ticket_7_engineer","error":"io"}}"#,
                agent_id: "ticket_7_engineer",
                role: "engineer",
                workspace: "my-ws",
                ticket: "",
                fields: None,
            },
        ];
        for case in cases {
            let entry = parse_tracing_json(case.line).unwrap();
            assert_eq!(entry.agent_id, case.agent_id, "{}: agent_id", case.name);
            assert_eq!(entry.agent_role, case.role, "{}: agent_role", case.name);
            assert_eq!(entry.workspace, case.workspace, "{}: workspace", case.name);
            assert_eq!(entry.ticket, case.ticket, "{}: ticket", case.name);
            if let Some(fields) = case.fields {
                assert_eq!(
                    entry.fields, fields,
                    "{}: the promoted values are not fields of the same record",
                    case.name
                );
            }
        }
    }

    /// Create a temporary LogStore for tests. Returns the store and a TempDir that must be held.
    async fn test_store() -> (Arc<LogStore>, tempfile::TempDir) {
        let (store, dir) = crate::open_test_store!(LogStore, "log");
        (Arc::new(store), dir)
    }

    // Helper to seed log entries in tests
    async fn seed_entries(store: &LogStore, entries: &[LogEntry]) {
        store.insert_batch(entries).await.unwrap();
    }

    /// The per-view clear scopes: the Issues scope deletes warnings and errors only, the All
    /// Logs scope deletes every stored record.
    #[tokio::test]
    // the clear drives a wipe, which purges the process-global telemetry queue
    #[serial_test::serial(provider)]
    async fn clear_logs_scopes_match_the_views() {
        let (store, _dir) = test_store().await;
        seed_entries(
            &store,
            &[
                LogEntry {
                    level: "INFO".into(),
                    ..Default::default()
                },
                LogEntry {
                    level: "WARN".into(),
                    ..Default::default()
                },
                LogEntry {
                    level: "ERROR".into(),
                    ..Default::default()
                },
            ],
        )
        .await;

        // Issues scope: warnings and errors only — the informational row survives.
        assert_eq!(store.clear_logs(Some("ERROR,WARN")).await.unwrap(), 2);
        let (entries, total) = store.query(&LogQuery::default()).await.unwrap();
        assert_eq!(total, 1);
        assert_eq!(entries[0].level, "INFO");

        // All Logs scope: every stored record.
        assert_eq!(store.clear_logs(None).await.unwrap(), 1);
        assert_eq!(store.query(&LogQuery::default()).await.unwrap().1, 0);
    }

    /// [`LogStore::has_reason`] finds a reason any row carrying a message holds, so a reason
    /// already recorded is never recorded again.
    #[tokio::test]
    async fn a_reason_already_recorded_is_found_and_a_new_one_is_not() {
        let (store, _dir) = test_store().await;

        store
            .insert_batch(&[
                LogEntry {
                    message: "the update stopped".into(),
                    fields: serde_json::json!({ "reason": "first" }),
                    ..Default::default()
                },
                LogEntry {
                    message: "the update stopped".into(),
                    fields: serde_json::json!({ "reason": "second" }),
                    ..Default::default()
                },
                // A row with no reason at all is not evidence for one.
                LogEntry {
                    message: "the update stopped".into(),
                    ..Default::default()
                },
                // A different message is its own fact, whatever its reason.
                LogEntry {
                    message: "another fact".into(),
                    fields: serde_json::json!({ "reason": "first" }),
                    ..Default::default()
                },
            ])
            .await
            .unwrap();

        for (message, reason, found) in [
            ("the update stopped", "first", true),
            ("the update stopped", "second", true),
            ("the update stopped", "third", false),
            ("another fact", "first", true),
            ("another fact", "second", false),
            ("never recorded", "first", false),
        ] {
            assert_eq!(
                store.has_reason(message, reason).await.unwrap(),
                found,
                "{message:?} + {reason:?}"
            );
        }
    }

    /// Open a healthy store in a fresh temp dir, seed one entry, and checkpoint so all pages
    /// land in the main DB file, then close it. Returns the dir, which must be held.
    async fn seeded_closed_store() -> tempfile::TempDir {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        {
            let store = LogStore::open(tmp.path())
                .await
                .expect("open healthy store");
            seed_entries(
                &store,
                &[LogEntry {
                    timestamp: "2025-01-01T00:00:00Z".into(),
                    level: "INFO".into(),
                    target: "test".into(),
                    message: "pre-corruption".into(),
                    ..Default::default()
                }],
            )
            .await;
            store
                .conn
                .checkpoint_ungated()
                .await
                .expect("checkpoint so pages land in the main DB file");
        }
        tmp
    }

    /// Poll `store.query` until the total entry count reaches `expected` or the timeout elapses,
    /// rather than sleeping: the writer and the test share a runtime.
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

    /// [`flush_log_batch`] on both outcomes: a healthy flush persists the rows and records no
    /// failure; a batch the store refuses is dropped all the same and the refusal is visible on
    /// the sanctioned surface. The retry-then-succeed branch is deliberately untested.
    #[tokio::test]
    #[serial_test::serial(log_surface)] // serializes the process-global write-failure counter
    async fn flush_log_batch_persists_and_records_only_a_refused_batch() {
        let baseline = log_write_error_info().count;

        let (store, _dir) = test_store().await;
        let mut batch = vec![LogEntry {
            timestamp: "2025-01-01T00:00:01Z".to_string(),
            level: "INFO".to_string(),
            target: "test".to_string(),
            message: "persisted".to_string(),
            ..Default::default()
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

        // Deterministic insert failure: drop the logs table through the store's own connection.
        let (store, _dir) = test_store().await;
        store
            .conn
            .execute("DROP TABLE logs", ())
            .await
            .expect("drop logs table for failure test");

        let mut batch = vec![LogEntry {
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            level: "INFO".to_string(),
            target: "test".to_string(),
            message: "should not persist".to_string(),
            ..Default::default()
        }];
        flush_log_batch(&store, &mut batch).await;

        // The batch must be dropped and the failure visible on the sanctioned surface.
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

    /// The writer's three ways into the store: closing the channel flushes what is buffered, a
    /// full batch flushes at the cap while the timer holds the odd earlier entry, and the timer
    /// flushes a lone buffered entry.
    #[tokio::test]
    async fn the_log_writer_flushes_on_close_the_batch_cap_and_the_timer() {
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

        drop(tx);
        wait_for_total(&store, 2, std::time::Duration::from_secs(2)).await;

        let (entries, total) = store.query(&LogQuery::default()).await.unwrap();
        assert_eq!(total, 2);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].message, "oh no");
        assert_eq!(entries[1].message, "hi");

        // Writer 1: a flush interval long enough that the timer can never fire — only the
        // batch-cap path can insert rows.
        let (store, _dir) = test_store().await;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let (broadcast_tx, _) = tokio::sync::broadcast::channel(256);
        spawn_log_writer_with_interval(
            store.clone(),
            rx,
            broadcast_tx,
            std::time::Duration::from_mins(1),
        );

        tx.send(
            r#"{"timestamp":"2025-01-01T00:00:02Z","level":"INFO","target":"test","fields":{"message":"timer flush"}}"#
                .to_string(),
        )
        .unwrap();

        // Then LOG_BATCH_MAX more entries — the batch-cap flush fires as soon as a full batch
        // accumulates (the timer cannot fire here).
        for i in 0..LOG_BATCH_MAX {
            tx.send(
                format!(
                    r#"{{"timestamp":"2025-01-01T00:00:03Z","level":"INFO","target":"test","fields":{{"message":"batch {i}"}}}}"#
                ),
            )
            .unwrap();
        }

        wait_for_total(&store, LOG_BATCH_MAX, std::time::Duration::from_secs(10)).await;

        // Writer 2: a very short flush interval — the timer path must flush a lone entry.
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

    /// What the search box reads over stored rows: a plain substring over the message and the
    /// target, a wildcard character that is just that character (an empty search returning every
    /// row), the row's own scope, and the level, target and since filters it combines with.
    #[expect(clippy::too_many_lines)] // four fixtures, one claim each
    #[tokio::test]
    async fn stored_search_matches_substrings_targets_scope_and_filters() {
        let (store, _dir) = test_store().await;
        seed_entries(
            &store,
            &[
                LogEntry {
                    timestamp: "2025-01-01T00:00:00Z".into(),
                    level: "INFO".into(),
                    target: "module_a".into(),
                    message: "processing request".into(),
                    ..Default::default()
                },
                LogEntry {
                    timestamp: "2025-01-01T00:00:01Z".into(),
                    level: "ERROR".into(),
                    target: "module_b".into(),
                    message: "failed to process".into(),
                    ..Default::default()
                },
                LogEntry {
                    timestamp: "2025-01-01T00:00:02Z".into(),
                    level: "INFO".into(),
                    target: "module_c".into(),
                    message: "started".into(),
                    ..Default::default()
                },
            ],
        )
        .await;
        for (search, total) in [("proc", 2), ("request", 1), ("module", 3)] {
            let (results, found) = store
                .query(&LogQuery {
                    search: Some(search.into()),
                    ..Default::default()
                })
                .await
                .unwrap();
            assert_eq!(found, total, "search {search:?}");
            assert_eq!(results.len(), total, "search {search:?}");
            if search == "request" {
                assert_eq!(results[0].message, "processing request");
            }
        }

        let (store, _dir) = test_store().await;
        seed_entries(
            &store,
            &[
                LogEntry {
                    timestamp: "2025-01-01T00:00:00Z".into(),
                    level: "INFO".into(),
                    target: "mahbot::orchestrator".into(),
                    message: "processing request".into(),
                    ..Default::default()
                },
                LogEntry {
                    timestamp: "2025-01-01T00:00:01Z".into(),
                    level: "ERROR".into(),
                    target: "mahbot::tools".into(),
                    message: "failed to process".into(),
                    fields: serde_json::json!({"code": 1}),
                    ..Default::default()
                },
                LogEntry {
                    timestamp: "2025-01-01T00:00:02Z".into(),
                    level: "INFO".into(),
                    target: "mahbot::api".into(),
                    message: "started".into(),
                    ..Default::default()
                },
            ],
        )
        .await;
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
        let (_results, total) = store
            .query(&LogQuery {
                since: Some("2025-01-01T00:00:01Z".into()),
                search: Some("process".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 1, "only entry after timestamp matching 'process'");

        let (store, _dir) = test_store().await;
        seed_entries(
            &store,
            &[
                LogEntry {
                    target: "mahbot::pipeline".into(),
                    message: "claimed".into(),
                    workspace: "mahbot".into(),
                    ticket: "mahbot-42".into(),
                    ..Default::default()
                },
                LogEntry {
                    target: "mahbot::pipeline".into(),
                    message: "claimed".into(),
                    workspace: "mahbot_rlx".into(),
                    ticket: "mahbot-43".into(),
                    ..Default::default()
                },
            ],
        )
        .await;
        for (search, total) in [
            ("mahbot-42", 1),  // the ticket
            ("mahbot_rlx", 1), // the workspace
            ("mahbot-4", 2),   // a fragment of both tickets
            ("mahbot_4", 0),   // `_` is a character, not a wildcard
            ("%", 0),          // and so is `%`
        ] {
            let (_results, found) = store
                .query(&LogQuery {
                    search: Some(search.into()),
                    ..Default::default()
                })
                .await
                .unwrap();
            assert_eq!(found, total, "search {search:?}");
        }

        let (store, _dir) = test_store().await;
        seed_entries(
            &store,
            &[
                LogEntry {
                    timestamp: "2025-01-01T00:00:00Z".into(),
                    level: "INFO".into(),
                    target: "module_a".into(),
                    message: "processing `Hello ${name}` template".into(),
                    ..Default::default()
                },
                LogEntry {
                    timestamp: "2025-01-01T00:00:01Z".into(),
                    level: "ERROR".into(),
                    target: "module_b".into(),
                    message: "normal log entry".into(),
                    ..Default::default()
                },
            ],
        )
        .await;
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
        let (_results, total) = store
            .query(&LogQuery {
                search: None,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(total, 2, "no search filter should return all entries");
    }

    /// The live filter matches the same four parts the stored-row query binds, so a ticket or a
    /// workspace typed into the box finds a live row exactly as it finds a stored one.
    #[test]
    fn matches_search_covers_the_row_scope() {
        let entry = LogEntry {
            target: "mahbot::pipeline".into(),
            message: "Claimed ticket".into(),
            workspace: "mahbot".into(),
            ticket: "mahbot-42".into(),
            ..Default::default()
        };
        for query in ["pipeline", "Claimed", "MAHBOT", "mahbot-4", "MAHBOT-42", ""] {
            assert!(entry.matches_search(query), "{query}");
        }
        assert!(!entry.matches_search("mahbot_rlx"));
    }

    /// The writer's panic-restart bound: past it the writer enters the terminal stopped state
    /// (banner), and a successful flush resets the counter.
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

    /// Boot-time refusal: a logs store whose main DB file is corrupt is never quarantined or
    /// recreated — `LogStore::open` errors with a `StoreRefusal` naming its integrity check and
    /// leaves the file byte for byte as it was. The open only scans when the start has a reason
    /// for it (see `db::open_present_store`); here it is one catalog entry still to apply, the
    /// deterministic stand-in for a start that is not an ordinary clean one.
    #[tokio::test]
    async fn test_log_store_open_refuses_corrupt_store() {
        let tmp = seeded_closed_store().await;
        let root = tmp.path();

        let pending = crate::db::migrations::MIGRATIONS
            .iter()
            .rev()
            .find(|migration| migration.target == crate::db::migrations::TargetDb::Logs)
            .expect("the logs catalog has entries")
            .id;
        // Put the store one entry behind, then hand it off clean again (the checkpoint keeps the
        // journal empty, so the pending catalog entry is the one reason left).
        {
            let store = LogStore::open(root).await.expect("reopen healthy store");
            store
                .conn
                .execute(
                    &format!("DELETE FROM schema_migrations WHERE id = '{pending}'"),
                    (),
                )
                .await
                .expect("record the newest logs migration as unapplied");
            store
                .conn
                .checkpoint_ungated()
                .await
                .expect("checkpoint the fixture store");
        }

        let db_path = db::store_db_path(root, "logs");
        let bytes = std::fs::read(&db_path).expect("read db file");
        assert!(bytes.len() > 8192, "test needs a multi-page db file");
        let mut corrupted = bytes.clone();
        corrupted[4096..8192].fill(0);
        std::fs::write(&db_path, &corrupted).expect("corrupt db file");

        let err = LogStore::open(root)
            .await
            .expect_err("a corrupt store must be refused, not recreated");
        let refusal = err
            .downcast_ref::<crate::db::StoreRefusal>()
            .unwrap_or_else(|| panic!("expected a StoreRefusal, got: {err:#}"));
        assert!(
            refusal
                .reason
                .contains("integrity check cannot read a table"),
            "the warranted scan, not an earlier gate, must be what refuses: {}",
            refusal.reason
        );
        assert_eq!(
            std::fs::read(&db_path).expect("read db file"),
            corrupted,
            "the refused store's main file must be unchanged"
        );
        crate::db::test_support::assert_not_quarantined(&root.join("db"), "a refused logs store");
    }

    /// A record written through the standard third-party `log` interface reaches the log layer
    /// under its real target, through the production stack and bridge, and the line the layer
    /// wrote is a row the store's own parser accepts.
    ///
    /// The bridge installs a process-global `log` logger (which is why the production
    /// [`install_log_bridge`] treats an existing one as a failure), while the stack goes on this
    /// thread's dispatcher. The real `EnvFilter` is in the stack, so the assertion on the target
    /// also pins the `tracing-log` feature: without it the layer stores the literal target `log`.
    #[test]
    fn dependency_log_records_reach_the_log_store() {
        const MESSAGE: &str = "a dependency record through the standard logging interface";

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let env_filter = EnvFilter::new(DEFAULT_LOG_FILTER);
        let bridge_level = log_bridge_level(&env_filter);
        let _guard = tracing::subscriber::set_default(log_layers(make_log_writer(tx), env_filter));
        install_log_bridge(bridge_level).expect("install the log→tracing bridge");

        tracing_log::log::warn!("{MESSAGE}");

        let mut written = String::new();
        while let Ok(line) = rx.try_recv() {
            written.push_str(&line);
        }
        let record = written
            .lines()
            .find(|line| line.contains(MESSAGE))
            .unwrap_or_else(|| {
                panic!(
                    "a record written through the standard `log` interface must reach the log \
                     layer: {written}"
                )
            });
        let entry = parse_tracing_json(record).expect("the log store must parse the written line");
        assert_eq!(
            entry.message, MESSAGE,
            "the store row must carry the record's message",
        );
        assert_eq!(
            entry.target,
            module_path!(),
            "the record's real target must survive the bridge, not become the literal `log`",
        );
    }

    /// The line the layer really writes carries the row's scope: a record inside the agent's span
    /// chain parses with the agent, role, workspace and ticket, and the record's own copy of
    /// those values is not repeated as a field. A record with no agent invents none of them, and
    /// a workspace stated as a path becomes the name it stands for.
    #[test]
    fn a_record_inside_the_agent_span_carries_the_rows_scope() {
        const WRITING: &str = "writing code";
        const FINISHED: &str = "project commands finished";

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _guard = tracing::subscriber::set_default(log_layers(
            make_log_writer(tx),
            EnvFilter::new(DEFAULT_LOG_FILTER),
        ));
        let phase = tracing::info_span!("ticket", ticket = %"mahbot-42", workspace = %"mahbot");
        let agent = tracing::info_span!(
            parent: &phase,
            "agent",
            agent_id = %"ticket_mahbot-42_engineer",
            role = %"engineer",
            workspace = %"mahbot",
            ticket = Some("mahbot-42")
        );
        agent.in_scope(|| {
            tracing::warn!(ticket = %"mahbot-42", path = "src/a.rs", "{}", WRITING);
        });
        phase.in_scope(|| {
            tracing::info!(
                workspace = %"/Users/x/.mahbot/userspaces/lyuba",
                passed = 3,
                "{}",
                FINISHED
            );
        });

        let mut written = String::new();
        while let Ok(line) = rx.try_recv() {
            written.push_str(&line);
        }
        let row = |message: &str| {
            let line = written
                .lines()
                .find(|line| line.contains(message))
                .unwrap_or_else(|| panic!("no written line carries {message:?}: {written}"));
            parse_tracing_json(line).expect("the log store must parse the written line")
        };

        let scoped = row(WRITING);
        assert_eq!(scoped.agent_id, "ticket_mahbot-42_engineer");
        assert_eq!(scoped.agent_role, "engineer");
        assert_eq!(scoped.workspace, "mahbot");
        assert_eq!(scoped.ticket, "mahbot-42");
        assert_eq!(
            scoped.fields,
            serde_json::json!({ "path": "src/a.rs" }),
            "a promoted value is not repeated as a field of the same record"
        );

        let stage = row(FINISHED);
        assert_eq!(stage.agent_id, "", "a stage line has no agent of its own");
        assert_eq!(stage.agent_role, "");
        assert_eq!(
            stage.workspace, "personal:lyuba",
            "an account's folder keeps its own spelling, never the path"
        );
        assert_eq!(stage.ticket, "mahbot-42");
        assert_eq!(stage.fields, serde_json::json!({ "passed": 3 }));
    }

    /// The level cut [`DEFAULT_LOG_FILTER`] applies to the PDF text layer: that dependency's
    /// warnings are dropped before the store, its errors still reach it. A document conversion
    /// cannot be driven hermetically, so the layer's view of the dependency is what is asserted.
    #[test]
    fn the_default_filter_holds_the_pdf_dependency_to_error() {
        const TARGET: &str = "pdf_extract";
        const DROPPED: &str = "a warning the dependency emits once per page";
        const KEPT: &str = "the dependency's own error";
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _guard = tracing::subscriber::set_default(log_layers(
            make_log_writer(tx),
            EnvFilter::new(DEFAULT_LOG_FILTER),
        ));
        tracing::warn!(target: TARGET, "{DROPPED}");
        tracing::error!(target: TARGET, "{KEPT}");
        let mut written = String::new();
        while let Ok(line) = rx.try_recv() {
            written.push_str(&line);
        }
        assert!(
            !written.contains(DROPPED),
            "the dependency's warnings must not reach the log layer: {written}"
        );
        assert!(
            written.contains(KEPT),
            "the dependency's errors must still reach the log layer: {written}"
        );
    }

    /// The whole `fff_*` family is muted outright: not one of its targets reaches the log layer at
    /// any level, while records from everywhere else do, including the product's own notice that a
    /// written file could not be indexed.
    ///
    /// Written with explicit targets on the tracing API rather than through `log`, because the
    /// bridge is a process-global logger that only one test can install
    /// ([`dependency_log_records_reach_the_log_store`] pins it). What is asserted here is the
    /// layer's own verdict, per target and per level.
    #[test]
    fn the_default_filter_silences_the_search_library() {
        const BURST_TARGET: &str = "fff_search::stable_vec";
        const BURST: &str = "StableVec: capacity exhausted — dropping item to prevent reallocation";
        const GREP_TARGET: &str = "fff_search::grep";
        const GREP: &str = "a grep diagnostic from the file-search library";
        const SHARED_TARGET: &str = "fff_search::shared";
        const SHARED: &str = "a watcher diagnostic from the file-search library";
        const FFF_GREP_TARGET: &str = "fff_grep::matcher";
        const FFF_GREP: &str = "a record from the family's grep crate";
        const PARSER_TARGET: &str = "fff_query_parser::parse";
        const PARSER: &str = "a record from the family's query parser";
        const DEBOUNCER_TARGET: &str = "fff_notify_debouncer_full";
        const DEBOUNCER: &str = "a record the family logs over the `log` interface";
        const NOTICE_TARGET: &str = "mahbot::tools::edit";
        const NOTICE: &str =
            "Search index capacity exhausted after file write — background rescan needed";
        const ENGINE_TARGET: &str = "mahbot::workspace_index";
        const ENGINE: &str = "Search engine created — background scan started";

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _guard = tracing::subscriber::set_default(log_layers(
            make_log_writer(tx),
            EnvFilter::new(DEFAULT_LOG_FILTER),
        ));

        tracing::error!(target: BURST_TARGET, len = 1025, capacity = 1025, "{BURST}");
        tracing::error!(target: GREP_TARGET, "{GREP}");
        tracing::warn!(target: SHARED_TARGET, "{SHARED}");
        tracing::info!(target: FFF_GREP_TARGET, "{FFF_GREP}");
        tracing::error!(target: PARSER_TARGET, "{PARSER}");
        tracing::info!(target: DEBOUNCER_TARGET, "{DEBOUNCER}");
        tracing::warn!(target: NOTICE_TARGET, "{NOTICE}");
        tracing::info!(target: ENGINE_TARGET, "{ENGINE}");

        let mut written = String::new();
        while let Ok(line) = rx.try_recv() {
            written.push_str(&line);
        }
        for (target, message) in [
            (BURST_TARGET, BURST),
            (GREP_TARGET, GREP),
            (SHARED_TARGET, SHARED),
            (FFF_GREP_TARGET, FFF_GREP),
            (PARSER_TARGET, PARSER),
            (DEBOUNCER_TARGET, DEBOUNCER),
        ] {
            assert!(
                !written.contains(message),
                "a record from {target} reached the log layer: {written}"
            );
        }
        assert!(
            written.contains(NOTICE),
            "the product's own notice about the search index must still reach the log layer: \
             {written}"
        );
        assert!(
            written.contains(ENGINE),
            "the product's own search-engine record must still reach the log layer: {written}"
        );
    }
}
