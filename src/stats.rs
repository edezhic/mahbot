//! Per-agent tool call statistics stored in `stats.db`.
//!
//! Each tool invocation is recorded as an individual row with its full
//! serialized arguments, execution duration, and success/failure outcome.
//! Stats accumulate in-memory in each [`crate::Agent`] via a
//! `std::sync::Mutex<Vec<ToolCallRecord>>` and are flushed to
//! the database on session finalization via [`StatsStore::flush_batch`].

use crate::turso::{self};
use anyhow::Result;

crate::define_store! {
    /// Global stats store.
    pub static STATS_STORE: StatsStore,
    db_name = "stats",
    schema = SCHEMA,
    expect = "STATS_STORE not initialized — call init_global() first",
}

const SCHEMA: &str = "\
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
CREATE INDEX IF NOT EXISTS idx_tool_calls_error_message ON tool_calls(error_message);";

// Column definitions for tool_error SELECT queries.
crate::columns! {
    TOOL_ERROR_COLUMNS [TE] {
        TOOL_NAME      => "tool_name",
        ROLE           => "role",
        ERROR_MESSAGE  => "COALESCE(error_message, '') AS error_message",
        ARGUMENTS      => "arguments",
        DURATION_MS    => "duration_ms",
        SUCCESS        => "success",
        WORKSPACE      => "workspace",
        RECORDED_AT    => "recorded_at",
    }
}

/// A single tool error entry queried from the DB.
#[derive(Debug, Clone)]
pub struct ToolErrorEntry {
    pub tool_name: String,
    pub role: String,
    pub error_message: String,
    pub arguments: String,
    pub duration_ms: i64,
    pub success: bool,
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
    /// Query the count of tool calls for a given agent and tool.
    ///
    /// Uses `COUNT(*)` which always returns a row.
    pub async fn query_tool_usage(&self, agent_id: &str, tool_name: &str) -> Result<i64> {
        self.conn
            .query_optional(
                "SELECT COUNT(*) FROM tool_calls \
                 WHERE agent_id = ?1 AND tool_name = ?2",
                turso::params![agent_id, tool_name],
                |row| row.get::<i64>(0),
            )
            .await
            .map(|opt| opt.unwrap_or(0))
    }

    /// Build a parameterized WHERE clause and params for tool error (failure) queries.
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
        let mut clauses = vec!["error_message IS NOT NULL".to_string()];
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
            clauses.push("error_message LIKE ?".to_string());
        }

        (clauses.join(" AND "), params)
    }

    /// Count the total number of tool call error rows matching the optional
    /// query filters.
    pub async fn count_tool_errors(&self, query: &ToolErrorQuery) -> Result<usize> {
        let (where_clause, params) = Self::build_tool_error_filter(query);
        let sql = format!("SELECT COUNT(*) FROM tool_calls WHERE {where_clause}");
        self.conn
            .query_optional(&sql, turso::params_from_iter(params), |row| {
                row.get::<i64>(0)
            })
            .await
            .map(|opt| opt.unwrap_or(0))
            .map(|n: i64| {
                usize::try_from(n)
                    .expect("count_tool_errors returned negative count; DB invariant violated")
            })
    }

    /// Query tool call error entries with optional filters and pagination.
    ///
    /// Returns `(entries, total_count)` where each entry corresponds to a
    /// single failed tool call (error_message IS NOT NULL).
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
        let limit_val = i64::try_from(limit)
            .expect("query_tool_errors limit overflowed i64; limit must be <= i64::MAX");
        let offset_val = i64::try_from(offset)
            .expect("query_tool_errors offset overflowed i64; offset must be <= i64::MAX");

        let sql = format!(
            "SELECT {TOOL_ERROR_COLUMNS} \
             FROM tool_calls \
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
                error_message: row.get::<String>(COL_TE_ERROR_MESSAGE)?,
                arguments: row.get::<String>(COL_TE_ARGUMENTS)?,
                duration_ms: row.get::<i64>(COL_TE_DURATION_MS)?,
                success: row.get::<i64>(COL_TE_SUCCESS)? != 0,
                workspace: row.get::<String>(COL_TE_WORKSPACE)?,
                recorded_at: row.get::<String>(COL_TE_RECORDED_AT)?,
            });
        }

        Ok((entries, total))
    }

    /// Write a batch of per-call tool records for a single agent flush.
    pub async fn flush_batch(
        &self,
        agent_id: &str,
        role: &str,
        workspace: &str,
        stats: &[crate::ToolCallRecord],
    ) -> Result<()> {
        let recorded_at = turso::now();
        for record in stats {
            self.conn
                .execute(
                    "INSERT INTO tool_calls \
                     (agent_id, role, tool_name, arguments, duration_ms, success, error_message, workspace, recorded_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    turso::params![
                        agent_id,
                        role,
                        record.tool_name.clone(),
                        record.arguments.clone(),
                        record.duration_ms,
                        i64::from(record.success),
                        record.error_message.clone(),
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
    #[allow(clippy::too_many_lines)]
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
                expected_clause: "error_message IS NOT NULL",
                expected_params: vec![],
            },
            Case {
                name: "role_only",
                query: ToolErrorQuery {
                    role_filter: Some("Engineer".to_string()),
                    workspace_filter: None,
                    search: None,
                },
                expected_clause: "error_message IS NOT NULL AND role = ?",
                expected_params: vec![turso::Value::Text("Engineer".to_string())],
            },
            Case {
                name: "workspace_only",
                query: ToolErrorQuery {
                    role_filter: None,
                    workspace_filter: Some("my-workspace".to_string()),
                    search: None,
                },
                expected_clause: "error_message IS NOT NULL AND workspace = ?",
                expected_params: vec![turso::Value::Text("my-workspace".to_string())],
            },
            Case {
                name: "search_only",
                query: ToolErrorQuery {
                    role_filter: None,
                    workspace_filter: None,
                    search: Some("timeout".to_string()),
                },
                expected_clause: "error_message IS NOT NULL AND error_message LIKE ?",
                expected_params: vec![turso::Value::Text("%timeout%".to_string())],
            },
            Case {
                name: "role_and_workspace",
                query: ToolErrorQuery {
                    role_filter: Some("Analyst".to_string()),
                    workspace_filter: Some("ws1".to_string()),
                    search: None,
                },
                expected_clause: "error_message IS NOT NULL AND role = ? AND workspace = ?",
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
                expected_clause: "error_message IS NOT NULL AND role = ? AND error_message LIKE ?",
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
                expected_clause: "error_message IS NOT NULL AND workspace = ? AND error_message LIKE ?",
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
                expected_clause: "error_message IS NOT NULL AND role = ? AND workspace = ? AND error_message LIKE ?",
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

    /// Integration test: write per-call records via flush_batch, then verify
    /// they can be read back via query_tool_usage and query_tool_errors.
    #[allow(clippy::too_many_lines)]
    #[tokio::test]
    async fn flush_and_query_round_trip() {
        let (store, _tmp) = crate::open_test_store!(StatsStore, "stats");

        let records = vec![
            crate::ToolCallRecord {
                tool_name: "read".to_string(),
                arguments: r#"{"path":"src/main.rs"}"#.to_string(),
                duration_ms: 42,
                success: true,
                error_message: None,
            },
            crate::ToolCallRecord {
                tool_name: "create_ticket".to_string(),
                arguments: r#"{"title":"Fix bug"}"#.to_string(),
                duration_ms: 150,
                success: true,
                error_message: None,
            },
            crate::ToolCallRecord {
                tool_name: "write".to_string(),
                arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
                duration_ms: 0,
                success: false,
                error_message: Some("Error executing write: permission denied".to_string()),
            },
        ];

        // Flush the batch
        store
            .flush_batch("test-agent", "Engineer", "my-workspace", &records)
            .await
            .expect("flush_batch");

        // Verify query_tool_usage (COUNT)
        let count = store
            .query_tool_usage("test-agent", "create_ticket")
            .await
            .expect("query_tool_usage");
        assert_eq!(count, 1, "should have 1 create_ticket call");

        let count = store
            .query_tool_usage("test-agent", "read")
            .await
            .expect("query_tool_usage");
        assert_eq!(count, 1, "should have 1 read call");

        let count = store
            .query_tool_usage("test-agent", "nonexistent")
            .await
            .expect("query_tool_usage");
        assert_eq!(count, 0, "should have 0 nonexistent calls");

        // Verify query_tool_errors — only the 'write' call failed
        let (errors, total) = store
            .query_tool_errors(&ToolErrorQuery::default(), 100, 0)
            .await
            .expect("query_tool_errors");
        assert_eq!(total, 1, "should have 1 error");
        assert_eq!(errors.len(), 1, "should return 1 entry");
        assert_eq!(errors[0].tool_name, "write");
        assert!(errors[0].error_message.contains("permission denied"));
        assert!(!errors[0].success);
        assert_eq!(errors[0].duration_ms, 0);
        assert_eq!(errors[0].arguments, r#"{"path":"src/lib.rs"}"#);
        assert_eq!(errors[0].role, "Engineer");
        assert_eq!(errors[0].workspace, "my-workspace");

        // Verify error filtering by role
        let query = ToolErrorQuery {
            role_filter: Some("Engineer".to_string()),
            workspace_filter: None,
            search: None,
        };
        let (errors, total) = store
            .query_tool_errors(&query, 100, 0)
            .await
            .expect("query_tool_errors with role filter");
        assert_eq!(total, 1, "Engineer should have 1 error");
        assert_eq!(errors.len(), 1);

        let query = ToolErrorQuery {
            role_filter: Some("Manager".to_string()),
            workspace_filter: None,
            search: None,
        };
        let (_errors, total) = store
            .query_tool_errors(&query, 100, 0)
            .await
            .expect("query_tool_errors with role filter");
        assert_eq!(total, 0, "Manager should have 0 errors");

        // Verify error filtering by search text
        let query = ToolErrorQuery {
            role_filter: None,
            workspace_filter: None,
            search: Some("permission".to_string()),
        };
        let (errors, total) = store
            .query_tool_errors(&query, 100, 0)
            .await
            .expect("query_tool_errors with search");
        assert_eq!(total, 1, "search 'permission' should find 1 error");
        assert_eq!(errors[0].tool_name, "write");

        let query = ToolErrorQuery {
            role_filter: None,
            workspace_filter: None,
            search: Some("timeout".to_string()),
        };
        let (_errors, total) = store
            .query_tool_errors(&query, 100, 0)
            .await
            .expect("query_tool_errors with search");
        assert_eq!(total, 0, "search 'timeout' should find 0 errors");

        // Verify error filtering by workspace
        let query = ToolErrorQuery {
            role_filter: None,
            workspace_filter: Some("my-workspace".to_string()),
            search: None,
        };
        let (_errors, total) = store
            .query_tool_errors(&query, 100, 0)
            .await
            .expect("query_tool_errors with workspace filter");
        assert_eq!(total, 1, "my-workspace should have 1 error");

        let query = ToolErrorQuery {
            role_filter: None,
            workspace_filter: Some("other-ws".to_string()),
            search: None,
        };
        let (_errors, total) = store
            .query_tool_errors(&query, 100, 0)
            .await
            .expect("query_tool_errors with workspace filter");
        assert_eq!(total, 0, "other-ws should have 0 errors");
    }
}
