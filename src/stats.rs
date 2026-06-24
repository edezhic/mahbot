//! Per-agent tool usage statistics stored in `stats.db`.
//!
//! Stats accumulate in-memory in each [`crate::Agent`] via a
//! `std::sync::Mutex<HashMap<String, ToolUsage>>` and are flushed to
//! the database on session finalization via [`StatsStore::flush_batch`].

use crate::global_store;
use crate::turso::{self, Connection};
use anyhow::{Context, Result};
use std::path::Path;

global_store! {
    /// Global stats store.
    pub static STATS_STORE: StatsStore,
    constructor = StatsStore::open,
}

/// Turso-backed tool usage stats storage.
#[derive(Clone, Debug)]
pub struct StatsStore {
    pub(crate) conn: Connection,
}

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS tool_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT NOT NULL,
    role        TEXT NOT NULL,
    tool_name   TEXT NOT NULL,
    call_count  INTEGER NOT NULL,
    errors      TEXT NOT NULL DEFAULT '[]',
    workspace   TEXT NOT NULL DEFAULT '',
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_usage_agent_id ON tool_usage(agent_id);
CREATE INDEX IF NOT EXISTS idx_tool_usage_role ON tool_usage(role);
CREATE INDEX IF NOT EXISTS idx_tool_usage_recorded_at ON tool_usage(recorded_at);
CREATE INDEX IF NOT EXISTS idx_tool_usage_workspace ON tool_usage(workspace);";

/// Column list for tool error SELECT queries in [`StatsStore::query_tool_errors`].
///
/// The column order here must match the positional indices defined in
/// [`COL_TE_TOOL_NAME`] through [`COL_TE_RECORDED_AT`].
const TOOL_ERROR_COLUMNS: &str =
    "tool_name, role, json_each.value AS error, workspace, recorded_at";

/// Column-index constants for [`TOOL_ERROR_COLUMNS`].
///
/// These replace hardcoded positional indices in [`StatsStore::query_tool_errors`].
/// With named constants, the compiler catches references to undefined column
/// constants — for instance, removing a constant but forgetting to update a
/// `row_text()` call produces a compile error rather than a silent field
/// mapping bug.
const COL_TE_TOOL_NAME: usize = 0;
const COL_TE_ROLE: usize = 1;
const COL_TE_ERROR: usize = 2;
const COL_TE_WORKSPACE: usize = 3;
const COL_TE_RECORDED_AT: usize = 4;

/// Column-index constant for the single-column SELECT in
/// [`StatsStore::query_tool_usage`] (`call_count`).
const COL_TU_CALL_COUNT: usize = 0;

/// Column-index constant for the single-column COUNT(*) SELECT in
/// [`StatsStore::count_tool_errors`].
const COL_TE_COUNT: usize = 0;

/// A single tool error flattened from the `errors` JSON array.
#[derive(Debug, Clone)]
pub struct ToolErrorEntry {
    pub tool_name: String,
    pub role: String,
    pub error: String,
    pub workspace: String,
    pub recorded_at: String,
}

/// Query filters for [`StatsStore::query_tool_errors`] / [`StatsStore::count_tool_errors`].
///
/// All fields are optional — `None` means no filter is applied.
#[derive(Debug, Clone, Default)]
pub struct ToolErrorQuery {
    /// Optional role name filter (exact match via `WHERE role = ?`).
    pub role_filter: Option<String>,
    /// Optional workspace name filter (exact match via `WHERE workspace = ?`).
    pub workspace_filter: Option<String>,
    /// Optional search text filter (substring match via `LIKE` on error text).
    pub search: Option<String>,
}

impl StatsStore {
    /// Open (or create) the stats database at `root/db/stats.db`.
    ///
    /// ## Migration v1
    ///
    /// Adds the `workspace` column to `tool_usage` for existing databases
    /// created before the column was added to the schema constant.
    pub async fn open(root: &Path) -> Result<Self> {
        let db_path = root.join("db/stats.db");
        let conn = turso::open_with_schema(&db_path, SCHEMA).await?;

        // ── Schema migration v1 ────────────────────────────────────────
        //
        // PRAGMA user_version returns 0 for unmigrated databases (SQLite
        // default). We use this as a migration stamp to ensure the migration
        // runs exactly once.
        let user_version: i64 = conn
            .query_row("PRAGMA user_version", turso::params![], |row| {
                row.get::<Option<i64>>(0)
            })
            .await
            .context("Failed to read PRAGMA user_version")?
            .unwrap_or(0);

        if user_version < 1 {
            // Use a silent ALTER TABLE ADD COLUMN — if the column already exists
            // (from a concurrent fresh CREATE TABLE IF NOT EXISTS on a new DB),
            // the error is harmless.
            let _ = conn
                .execute(
                    "ALTER TABLE tool_usage ADD COLUMN workspace TEXT NOT NULL DEFAULT ''",
                    turso::params![],
                )
                .await;
            conn.execute("PRAGMA user_version = 1", turso::params![])
                .await
                .context("Failed to set PRAGMA user_version = 1")?;
        }

        Ok(Self { conn })
    }

    /// Query the most recent `call_count` for a given agent and tool.
    ///
    /// Returns `None` if no row exists for the combination.
    /// When multiple rows exist (e.g. multiple flush calls), uses the most
    /// recent row (`ORDER BY id DESC LIMIT 1`).
    pub async fn query_tool_usage(&self, agent_id: &str, tool_name: &str) -> Result<Option<i64>> {
        match self
            .conn
            .query_row(
                "SELECT call_count FROM tool_usage \
                 WHERE agent_id = ?1 AND tool_name = ?2 \
                 ORDER BY id DESC LIMIT 1",
                turso::params![agent_id, tool_name],
                |row| row.get::<i64>(COL_TU_CALL_COUNT),
            )
            .await
        {
            Ok(count) => Ok(Some(count)),
            Err(::turso::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Build a parameterized WHERE clause and params for tool error queries.
    ///
    /// Returns `(where_clause, params)` where `where_clause` does NOT include
    /// the leading `WHERE` keyword — it is a set of `AND`-joined expressions
    /// suitable for embedding directly into SQL.  All placeholders use unnamed
    /// `?` — use [`turso::params_from_iter`] to bind params positionally.
    ///
    /// This is an associated function (no `&self`) so it can be used without
    /// a store instance if needed.
    #[must_use]
    pub fn build_tool_error_filter(query: &ToolErrorQuery) -> (String, Vec<turso::Value>) {
        let mut clauses = vec!["errors != '[]'".to_string()];
        let mut params = Vec::new();

        if let Some(ref role) = query.role_filter {
            params.push(turso::Value::Text(role.clone()));
            clauses.push("role = ?".to_string());
        }

        if let Some(ref workspace) = query.workspace_filter {
            params.push(turso::Value::Text(workspace.clone()));
            clauses.push("workspace = ?".to_string());
        }

        if let Some(ref search) = query.search {
            params.push(turso::Value::Text(format!("%{search}%")));
            clauses.push("json_each.value LIKE ?".to_string());
        }

        (clauses.join(" AND "), params)
    }

    /// Count the total number of individual errors across all tool_usage rows
    /// matching the optional query filters.
    ///
    /// Uses SQLite's `json_each` to flatten the `errors` JSON array so each
    /// array element counts as one row.
    pub async fn count_tool_errors(&self, query: &ToolErrorQuery) -> Result<usize> {
        let (where_clause, params) = Self::build_tool_error_filter(query);
        let sql = format!(
            "SELECT COUNT(*) FROM tool_usage, json_each(tool_usage.errors) WHERE {where_clause}",
        );
        let rows = self
            .conn
            .query(&sql, turso::params_from_iter(params))
            .await?;
        let count = match rows.into_iter().next() {
            Some(row) => match row.get_value(COL_TE_COUNT)? {
                turso::Value::Integer(n) => usize::try_from(n).unwrap_or(0),
                _ => 0,
            },
            None => 0,
        };
        Ok(count)
    }

    /// Query flattened tool errors with optional filters and pagination.
    ///
    /// Each individual error from the `errors` JSON array appears as its own row.
    /// Returns `(entries, total_count)`.
    pub async fn query_tool_errors(
        &self,
        query: &ToolErrorQuery,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<ToolErrorEntry>, usize)> {
        let total = self.count_tool_errors(query).await?;
        if total == 0 {
            return Ok((vec![], 0));
        }

        let (where_clause, filter_params) = Self::build_tool_error_filter(query);
        let limit_val = i64::try_from(limit).unwrap_or(50);
        let offset_val = i64::try_from(offset).unwrap_or(0);

        // Build the SQL with filter params first, then limit/offset.
        // All placeholders use unnamed `?` — params_from_iter binds positionally.
        let sql = format!(
            "SELECT {TOOL_ERROR_COLUMNS} \
             FROM tool_usage, json_each(tool_usage.errors) \
             WHERE {where_clause} \
             ORDER BY recorded_at DESC \
             LIMIT ? OFFSET ?",
        );

        let mut all_params = filter_params;
        all_params.push(turso::Value::Integer(limit_val));
        all_params.push(turso::Value::Integer(offset_val));

        let rows = self
            .conn
            .query(&sql, turso::params_from_iter(all_params))
            .await?;

        let mut entries = Vec::new();
        for row in rows {
            entries.push(ToolErrorEntry {
                tool_name: turso::row_text(&row, COL_TE_TOOL_NAME)?,
                role: turso::row_text(&row, COL_TE_ROLE)?,
                error: turso::row_text(&row, COL_TE_ERROR)?,
                workspace: turso::row_text(&row, COL_TE_WORKSPACE)?,
                recorded_at: turso::row_text(&row, COL_TE_RECORDED_AT)?,
            });
        }

        Ok((entries, total))
    }

    /// Write a batch of tool usage entries for a single agent flush.
    pub async fn flush_batch(
        &self,
        agent_id: &str,
        role: &str,
        workspace: &str,
        stats: &std::collections::HashMap<String, crate::ToolUsage>,
    ) -> Result<()> {
        let recorded_at = turso::now();
        for (tool_name, usage) in stats {
            let errors_json = serde_json::to_string(&usage.errors).unwrap_or_default();
            self.conn
                .execute(
                    "INSERT INTO tool_usage (agent_id, role, tool_name, call_count, errors, workspace, recorded_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    turso::params![
                        agent_id,
                        role,
                        tool_name.clone(),
                        {
                            let raw_count = usage.call_count;
                            i64::try_from(raw_count).unwrap_or_else(|_| {
                                tracing::warn!(
                                    agent_id = %agent_id,
                                    role = %role,
                                    tool_name = %tool_name,
                                    call_count = raw_count,
                                    "Tool call count overflowed i64, clamping to i64::MAX"
                                );
                                i64::MAX
                            })
                        },
                        errors_json,
                        workspace.to_string(),
                        recorded_at.clone(),
                    ],
                )
                .await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the number of columns in [`TOOL_ERROR_COLUMNS`] matches the highest
    /// column-index constant + 1. If this test fails, a column was added or removed
    /// from the string list without updating the corresponding `COL_TE_*` constants,
    /// or vice versa — a silent data corruption hazard.
    ///
    /// Note: [`COL_TU_CALL_COUNT`] and [`COL_TE_COUNT`] are single-column query
    /// constants and are intentionally excluded from this assertion.
    #[test]
    fn tool_error_columns_count_matches_column_constants() {
        let count = TOOL_ERROR_COLUMNS.split(',').count();
        assert_eq!(
            COL_TE_RECORDED_AT + 1,
            count,
            "TOOL_ERROR_COLUMNS has {count} entries but COL_TE_RECORDED_AT ({}) + 1 = {}",
            COL_TE_RECORDED_AT,
            COL_TE_RECORDED_AT + 1,
        );
    }

    #[test]
    fn test_build_tool_error_filter_no_filters() {
        let query = ToolErrorQuery::default();
        let (clause, params) = StatsStore::build_tool_error_filter(&query);
        assert_eq!(clause, "errors != '[]'");
        assert!(params.is_empty());
    }

    #[test]
    fn test_build_tool_error_filter_role_only() {
        let query = ToolErrorQuery {
            role_filter: Some("Engineer".to_string()),
            workspace_filter: None,
            search: None,
        };
        let (clause, params) = StatsStore::build_tool_error_filter(&query);
        assert_eq!(clause, "errors != '[]' AND role = ?");
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], turso::Value::Text("Engineer".to_string()));
    }

    #[test]
    fn test_build_tool_error_filter_search_only() {
        let query = ToolErrorQuery {
            role_filter: None,
            workspace_filter: None,
            search: Some("timeout".to_string()),
        };
        let (clause, params) = StatsStore::build_tool_error_filter(&query);
        assert_eq!(clause, "errors != '[]' AND json_each.value LIKE ?");
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], turso::Value::Text("%timeout%".to_string()));
    }

    #[test]
    fn test_build_tool_error_filter_workspace_only() {
        let query = ToolErrorQuery {
            role_filter: None,
            workspace_filter: Some("my-workspace".to_string()),
            search: None,
        };
        let (clause, params) = StatsStore::build_tool_error_filter(&query);
        assert_eq!(clause, "errors != '[]' AND workspace = ?");
        assert_eq!(params.len(), 1);
        assert_eq!(params[0], turso::Value::Text("my-workspace".to_string()));
    }

    #[test]
    fn test_build_tool_error_filter_both() {
        let query = ToolErrorQuery {
            role_filter: Some("Analyst".to_string()),
            workspace_filter: None,
            search: Some("connection refused".to_string()),
        };
        let (clause, params) = StatsStore::build_tool_error_filter(&query);
        assert_eq!(
            clause,
            "errors != '[]' AND role = ? AND json_each.value LIKE ?"
        );
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], turso::Value::Text("Analyst".to_string()));
        assert_eq!(
            params[1],
            turso::Value::Text("%connection refused%".to_string())
        );
    }

    #[test]
    fn test_build_tool_error_filter_empty_strings() {
        // Empty strings should be treated as valid filters (caller's
        // responsibility to send None rather than empty).
        let query = ToolErrorQuery {
            role_filter: Some(String::new()),
            workspace_filter: None,
            search: Some(String::new()),
        };
        let (clause, params) = StatsStore::build_tool_error_filter(&query);
        assert_eq!(
            clause,
            "errors != '[]' AND role = ? AND json_each.value LIKE ?"
        );
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], turso::Value::Text(String::new()));
        assert_eq!(params[1], turso::Value::Text("%%".to_string()));
    }
}
