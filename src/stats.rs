//! Per-agent tool usage statistics stored in `stats.db`.
//!
//! Stats accumulate in-memory in each [`crate::Agent`] via a
//! `std::sync::Mutex<HashMap<String, ToolUsage>>` and are flushed to
//! the database on session finalization via [`StatsStore::flush_batch`].

use crate::turso::{self};
use anyhow::Result;

crate::define_store! {
    /// Global stats store.
    pub static STATS_STORE: StatsStore,
    db_name = "stats",
    schema = SCHEMA,
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

// Column definitions for tool_error SELECT queries.
crate::columns! {
    TOOL_ERROR_COLUMNS [TE] {
        TOOL_NAME    => "tool_name",
        ROLE         => "role",
        ERROR        => "json_each.value AS error",
        WORKSPACE    => "workspace",
        RECORDED_AT  => "recorded_at",
    }
}

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
    /// Query the most recent `call_count` for a given agent and tool.
    ///
    /// Returns `None` if no row exists for the combination.
    /// When multiple rows exist (e.g. multiple flush calls), uses the most
    /// recent row (`ORDER BY id DESC LIMIT 1`).
    pub async fn query_tool_usage(&self, agent_id: &str, tool_name: &str) -> Result<Option<i64>> {
        self.conn
            .query_optional(
                "SELECT call_count FROM tool_usage \
                 WHERE agent_id = ?1 AND tool_name = ?2 \
                 ORDER BY id DESC LIMIT 1",
                turso::params![agent_id, tool_name],
                |row| row.get::<i64>(COL_TU_CALL_COUNT),
            )
            .await
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
                tool_name: row.get::<String>(COL_TE_TOOL_NAME)?,
                role: row.get::<String>(COL_TE_ROLE)?,
                error: row.get::<String>(COL_TE_ERROR)?,
                workspace: row.get::<String>(COL_TE_WORKSPACE)?,
                recorded_at: row.get::<String>(COL_TE_RECORDED_AT)?,
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

    /// All 8 combinations of optional filters in [`ToolErrorQuery`].
    ///
    /// Each case verifies the exact SQL clause string and param values produced
    /// by [`StatsStore::build_tool_error_filter`].
    #[test]
    fn build_tool_error_filter_all_combinations() {
        struct Case {
            name: &'static str,
            query: ToolErrorQuery,
            expected_clause: &'static str,
            expected_params: Vec<turso::Value>,
        }

        let cases = [
            Case {
                name: "no_filters",
                query: ToolErrorQuery::default(),
                expected_clause: "errors != '[]'",
                expected_params: vec![],
            },
            Case {
                name: "role_only",
                query: ToolErrorQuery {
                    role_filter: Some("Engineer".to_string()),
                    workspace_filter: None,
                    search: None,
                },
                expected_clause: "errors != '[]' AND role = ?",
                expected_params: vec![turso::Value::Text("Engineer".to_string())],
            },
            Case {
                name: "workspace_only",
                query: ToolErrorQuery {
                    role_filter: None,
                    workspace_filter: Some("my-workspace".to_string()),
                    search: None,
                },
                expected_clause: "errors != '[]' AND workspace = ?",
                expected_params: vec![turso::Value::Text("my-workspace".to_string())],
            },
            Case {
                name: "search_only",
                query: ToolErrorQuery {
                    role_filter: None,
                    workspace_filter: None,
                    search: Some("timeout".to_string()),
                },
                expected_clause: "errors != '[]' AND json_each.value LIKE ?",
                expected_params: vec![turso::Value::Text("%timeout%".to_string())],
            },
            Case {
                name: "role_and_workspace",
                query: ToolErrorQuery {
                    role_filter: Some("Analyst".to_string()),
                    workspace_filter: Some("ws1".to_string()),
                    search: None,
                },
                expected_clause: "errors != '[]' AND role = ? AND workspace = ?",
                expected_params: vec![
                    turso::Value::Text("Analyst".to_string()),
                    turso::Value::Text("ws1".to_string()),
                ],
            },
            Case {
                name: "role_and_search",
                query: ToolErrorQuery {
                    role_filter: Some("Analyst".to_string()),
                    workspace_filter: None,
                    search: Some("connection refused".to_string()),
                },
                expected_clause: "errors != '[]' AND role = ? AND json_each.value LIKE ?",
                expected_params: vec![
                    turso::Value::Text("Analyst".to_string()),
                    turso::Value::Text("%connection refused%".to_string()),
                ],
            },
            Case {
                name: "workspace_and_search",
                query: ToolErrorQuery {
                    role_filter: None,
                    workspace_filter: Some("ws2".to_string()),
                    search: Some("error msg".to_string()),
                },
                expected_clause: "errors != '[]' AND workspace = ? AND json_each.value LIKE ?",
                expected_params: vec![
                    turso::Value::Text("ws2".to_string()),
                    turso::Value::Text("%error msg%".to_string()),
                ],
            },
            Case {
                name: "all_three",
                query: ToolErrorQuery {
                    role_filter: Some("Manager".to_string()),
                    workspace_filter: Some("ws3".to_string()),
                    search: Some("fatal".to_string()),
                },
                expected_clause: "errors != '[]' AND role = ? AND workspace = ? AND json_each.value LIKE ?",
                expected_params: vec![
                    turso::Value::Text("Manager".to_string()),
                    turso::Value::Text("ws3".to_string()),
                    turso::Value::Text("%fatal%".to_string()),
                ],
            },
        ];

        for case in &cases {
            let (clause, params) = StatsStore::build_tool_error_filter(&case.query);
            assert_eq!(clause, case.expected_clause, "case: {}", case.name);
            assert_eq!(params, case.expected_params, "case: {}", case.name);
        }
    }
}
