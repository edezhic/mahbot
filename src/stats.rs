//! Per-agent tool call statistics, consolidated into the logs database.
//!
//! Each tool invocation is recorded as an individual row with its full
//! serialized arguments, execution duration, and success/failure outcome.
//! Stats accumulate in-memory in each [`crate::Agent`] via a
//! `std::sync::Mutex<Vec<ToolCallRecord>>` and are flushed to the logs
//! store on session finalization via [`crate::logs::LogStore::flush_batch`].
//!
//! The `tool_calls` table (and its indexes) is created
//! by the logs store's schema (`LOGS_SCHEMA` in `logs.rs`), so a logs-store
//! quarantine recreate also recreates it. Consumers access the table
//! through [`crate::logs::LOG_STORE`] with fail-open accessors.

use crate::turso::{self};
use anyhow::Result;

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

/// Query filters for [`crate::logs::LogStore::query_tool_errors`] / [`crate::logs::LogStore::count_tool_errors`].
///
/// All fields are optional — `None` means no filter is applied.
#[derive(Debug, Clone, Default)]
pub struct ToolErrorQuery {
    /// Optional search text filter (substring match via `LIKE` on error text).
    pub search: Option<String>,
}

impl crate::logs::LogStore {
    /// Query the count of tool calls for a given agent and tool.
    ///
    /// Uses `COUNT(*)` which always returns a row.
    pub(crate) async fn query_tool_usage(&self, agent_id: &str, tool_name: &str) -> Result<i64> {
        self.conn
            .query_optional(
                "SELECT COUNT(*) FROM tool_calls \
                 WHERE agent_id = ?1 AND tool_name = ?2",
                turso::params![agent_id, tool_name],
                |row| row.get::<i64>(0),
            )
            .await
            .map(|opt| opt.expect("COUNT(*) aggregate always returns a row"))
    }

    /// Count the total number of tool call error rows matching the optional
    /// query filters.
    pub(crate) async fn count_tool_errors(&self, query: &ToolErrorQuery) -> Result<usize> {
        let (where_clause, params) = build_tool_error_filter(query);
        let sql = format!("SELECT COUNT(*) FROM tool_calls WHERE {where_clause}");
        self.conn
            .query_optional(&sql, params, |row| row.get::<i64>(0))
            .await
            .map(|opt| opt.expect("COUNT(*) aggregate always returns a row"))
            .map(|n: i64| {
                usize::try_from(n)
                    .expect("count_tool_errors returned negative count; DB invariant violated")
            })
    }

    /// Query tool call error entries with optional filters and pagination.
    ///
    /// Returns `(entries, total_count)` where each entry corresponds to a
    /// single failed tool call (error_message IS NOT NULL).
    pub(crate) async fn query_tool_errors(
        &self,
        query: &ToolErrorQuery,
        limit: usize,
        offset: usize,
    ) -> Result<(Vec<ToolErrorEntry>, usize)> {
        let total = self.count_tool_errors(query).await?;
        if total == 0 {
            return Ok((vec![], 0));
        }

        let (where_clause, filter_params) = build_tool_error_filter(query);
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

        let rows = self.conn.query(&sql, all_params).await?;

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
    pub(crate) async fn flush_batch(
        &self,
        agent_id: &str,
        role: &str,
        workspace: &str,
        stats: &[crate::ToolCallRecord],
    ) -> Result<()> {
        if stats.is_empty() {
            return Ok(());
        }

        let recorded_at = turso::now();
        let tx = self.conn.begin_tx().await?;
        for record in stats {
            tx.execute(
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
        tx.commit().await?;
        Ok(())
    }

    /// Persist one per-operation LLM request stat to the dedicated
    /// `llm_requests` table. Metadata only — no request inputs/outputs.
    pub(crate) async fn record_llm_request(&self, rec: &LlmRequestRecord) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO llm_requests \
                 (recorded_at, purpose, agent_id, role, workspace, ticket_id, model, routing, \
                  input_tokens, output_tokens, cached_input_tokens, cache_miss_tokens, \
                  cost, cost_details, upstream_provider, system_fingerprint, \
                  duration_ms, retry_attempts, finish_reason, failure_class, success) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, \
                         ?17, ?18, ?19, ?20, ?21)",
                turso::params![
                    rec.recorded_at.clone(),
                    rec.purpose,
                    rec.agent_id.clone(),
                    rec.role.clone(),
                    rec.workspace.clone(),
                    rec.ticket_id.clone(),
                    rec.model.clone(),
                    rec.routing.clone(),
                    rec.input_tokens.map(i64::try_from).transpose()?,
                    rec.output_tokens.map(i64::try_from).transpose()?,
                    rec.cached_input_tokens.map(i64::try_from).transpose()?,
                    rec.cache_miss_tokens.map(i64::try_from).transpose()?,
                    rec.cost,
                    rec.cost_details.clone(),
                    rec.upstream_provider.clone(),
                    rec.system_fingerprint.clone(),
                    i64::try_from(rec.duration_ms)?,
                    i64::from(rec.retry_attempts),
                    rec.finish_reason.clone(),
                    rec.failure_class.clone(),
                    i64::from(rec.success),
                ],
            )
            .await?;
        Ok(())
    }
}

/// One per-operation LLM request stat row (the `llm_requests` table).
///
/// One row per completed LLM operation (agent iteration, extraction,
/// summarization, consolidation) — retry attempts are aggregated into
/// [`LlmRequestRecord::retry_attempts`]. Metadata only: no request inputs or
/// outputs.
#[derive(Debug, Clone)]
pub(crate) struct LlmRequestRecord {
    pub purpose: &'static str,
    pub agent_id: String,
    pub role: String,
    pub workspace: String,
    pub ticket_id: Option<String>,
    pub model: String,
    /// Requested provider routing (provider_order); "default" when unset.
    /// Requested-vs-actual: the serving upstream is [`LlmRequestRecord::upstream_provider`].
    pub routing: String,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub cache_miss_tokens: Option<u64>,
    /// Billed cost from `usage.cost` — the invoice amount; NULL on failures.
    pub cost: Option<f64>,
    /// Raw `usage.cost_details` JSON from the provider (reference only).
    pub cost_details: Option<String>,
    /// Serving upstream — top-level OpenRouter `provider` field (empirical,
    /// undocumented in the API reference); NULL when the provider omits it.
    pub upstream_provider: Option<String>,
    /// Provider `system_fingerprint`; NULL when the provider omits it.
    pub system_fingerprint: Option<String>,
    pub duration_ms: u64,
    /// Total HTTP attempts made by the operation (1 = no retries).
    pub retry_attempts: u32,
    pub finish_reason: Option<String>,
    pub failure_class: Option<String>,
    pub success: bool,
    pub recorded_at: String,
}

/// Build an [`LlmRequestRecord`] from a request + optional response envelope.
/// `None` when the request carries no metadata (context-free calls — test
/// doubles, ad-hoc requests) — such rows are never recorded.
///
/// Failure-path semantics: `upstream_provider` / `system_fingerprint` /
/// `cost` / `cost_details` are only populated from a successful response
/// envelope — failed requests have no response, so those columns stay NULL.
fn llm_request_record(
    request: &crate::ChatRequest,
    duration_ms: u64,
    attempts: u32,
    response: Option<&crate::ChatResponse>,
    finish_reason: Option<&str>,
    failure_class: Option<&'static str>,
) -> Option<LlmRequestRecord> {
    let meta = request.meta.as_ref()?;
    let usage = response.and_then(|r| r.usage.as_ref());
    Some(LlmRequestRecord {
        purpose: meta.purpose,
        agent_id: meta.agent_id.clone(),
        role: meta.role.clone(),
        workspace: meta.workspace.clone(),
        ticket_id: meta.ticket_id.clone(),
        model: request.model.clone(),
        routing: request
            .provider_order
            .as_deref()
            .unwrap_or("default")
            .to_string(),
        input_tokens: usage.and_then(|u| u.input_tokens),
        output_tokens: usage.and_then(|u| u.output_tokens),
        cached_input_tokens: usage.and_then(|u| u.cached_input_tokens),
        cache_miss_tokens: usage.and_then(|u| u.cache_miss_tokens),
        cost: usage.and_then(|u| u.cost),
        cost_details: usage
            .and_then(|u| u.cost_details.as_ref())
            .map(serde_json::Value::to_string),
        upstream_provider: response.and_then(|r| r.upstream_provider.clone()),
        system_fingerprint: response.and_then(|r| r.system_fingerprint.clone()),
        duration_ms,
        retry_attempts: attempts,
        finish_reason: finish_reason.map(str::to_string),
        failure_class: failure_class.map(str::to_string),
        success: failure_class.is_none(),
        recorded_at: crate::turso::now(),
    })
}

#[cfg(test)]
static TEST_LOG_STORE: std::sync::RwLock<Option<crate::logs::LogStore>> =
    std::sync::RwLock::new(None);

/// Test seam: redirect `record_llm_*` writes to a caller-owned store for the
/// duration of a test (mirrors `swap_provider_for_test`). Returns the previous
/// override so a guard can restore it.
#[cfg(test)]
pub(crate) fn swap_test_log_store(
    store: Option<crate::logs::LogStore>,
) -> Option<crate::logs::LogStore> {
    let mut guard = TEST_LOG_STORE.write().expect("test log store poisoned");
    let previous = guard.take();
    *guard = store;
    previous
}

/// Resolve the logs store for stat writes: test override wins, else the global.
/// Used by the llm_requests persistence path so tests observe its writes
/// through the same override.
pub(crate) fn log_store_for_stats() -> Option<crate::logs::LogStore> {
    #[cfg(test)]
    if let Some(store) = TEST_LOG_STORE
        .read()
        .expect("test log store poisoned")
        .as_ref()
    {
        return Some(store.clone());
    }
    crate::logs::LOG_STORE.get().cloned()
}

/// Emit one per-operation LLM request stat row (best-effort, fail-open).
///
/// Skips when the request carries no metadata (context-free calls — test
/// doubles, ad-hoc requests) and when the logs store is not yet open.
pub(crate) async fn record_llm_operation(
    request: &crate::ChatRequest,
    duration_ms: u64,
    attempts: u32,
    response: Option<&crate::ChatResponse>,
    finish_reason: Option<&str>,
    failure_class: Option<&'static str>,
) {
    let Some(rec) = llm_request_record(
        request,
        duration_ms,
        attempts,
        response,
        finish_reason,
        failure_class,
    ) else {
        return;
    };
    let Some(store) = log_store_for_stats() else {
        return;
    };
    if let Err(e) = store.record_llm_request(&rec).await {
        tracing::debug!(error = %e, "Failed to persist LLM request stat");
    }
}

/// Emit a success LLM request stat for a completed operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) async fn record_llm_success(
    request: &crate::ChatRequest,
    started: std::time::Instant,
    attempt: u32,
    response: &crate::ChatResponse,
) {
    record_llm_operation(
        request,
        started.elapsed().as_millis() as u64,
        attempt,
        Some(response),
        response.finish_reason.as_deref(),
        None,
    )
    .await;
}

/// Emit a failure LLM request stat for an exhausted operation.
#[allow(clippy::cast_possible_truncation)]
pub(crate) async fn record_llm_failure(
    request: &crate::ChatRequest,
    started: std::time::Instant,
    exhausted: &crate::retry::RetryExhausted,
) {
    record_llm_operation(
        request,
        started.elapsed().as_millis() as u64,
        exhausted.failures.len() as u32,
        None,
        exhausted
            .failures
            .last()
            .and_then(|r| r.finish_reason.as_deref()),
        Some(exhausted.final_class.label()),
    )
    .await;
}

/// Build a parameterized WHERE clause and params for tool error (failure) queries.
///
/// Returns `(where_clause, params)` where `where_clause` does NOT include
/// the leading `WHERE` keyword — it is a set of `AND`-joined expressions
/// suitable for embedding directly into SQL.  All placeholders use unnamed
/// `?` — pass a `Vec<turso::Value>` to bind params positionally (it
/// implements [`IntoParams`](crate::turso::IntoParams)).
#[must_use]
fn build_tool_error_filter(query: &ToolErrorQuery) -> (String, Vec<turso::Value>) {
    let mut clauses = vec!["error_message IS NOT NULL".to_string()];
    let mut params = Vec::new();

    if let Some(ref search) = query.search {
        params.push(turso::Value::Text(format!("%{search}%")));
        clauses.push("error_message LIKE ?".to_string());
    }

    (clauses.join(" AND "), params)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both states of the sole remaining optional filter (`search`) in
    /// [`ToolErrorQuery`].
    ///
    /// Each case verifies the exact SQL clause string and param values produced
    /// by [`build_tool_error_filter`].
    #[test]
    fn build_tool_error_filter_with_and_without_search() {
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
                name: "search_only",
                query: ToolErrorQuery {
                    search: Some("timeout".to_string()),
                },
                expected_clause: "error_message IS NOT NULL AND error_message LIKE ?",
                expected_params: vec![turso::Value::Text("%timeout%".to_string())],
            },
        ];

        for case in &cases {
            let (clause, params) = build_tool_error_filter(&case.query);
            assert_eq!(clause, case.expected_clause, "case: {}", case.name);
            assert_eq!(params, case.expected_params, "case: {}", case.name);
        }
    }

    /// Integration test: write per-call records via flush_batch, then verify
    /// they can be read back via query_tool_usage and query_tool_errors.
    /// Uses the logs store (the consolidated home of the stats tables).
    #[allow(clippy::too_many_lines)]
    #[tokio::test]
    async fn flush_and_query_round_trip() {
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");

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

        // Verify error filtering by search text
        let query = ToolErrorQuery {
            search: Some("permission".to_string()),
        };
        let (errors, total) = store
            .query_tool_errors(&query, 100, 0)
            .await
            .expect("query_tool_errors with search");
        assert_eq!(total, 1, "search 'permission' should find 1 error");
        assert_eq!(errors[0].tool_name, "write");

        let query = ToolErrorQuery {
            search: Some("timeout".to_string()),
        };
        let (_errors, total) = store
            .query_tool_errors(&query, 100, 0)
            .await
            .expect("query_tool_errors with search");
        assert_eq!(total, 0, "search 'timeout' should find 0 errors");
    }

    /// `llm_requests` insert round-trip — verifies the schema (including
    /// auto-creation on an existing store via LOGS_SCHEMA) and the
    /// parameterized insert are valid against a real store.
    #[tokio::test]
    async fn record_llm_request_round_trip() {
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
        let rec = LlmRequestRecord {
            purpose: "agent",
            agent_id: "gui_alice_ws1_engineer".to_string(),
            role: "engineer".to_string(),
            workspace: "ws1".to_string(),
            ticket_id: Some("42".to_string()),
            model: "deepseek/deepseek-v4-flash-0731".to_string(),
            routing: "deepseek".to_string(),
            input_tokens: Some(1_000),
            output_tokens: Some(200),
            cached_input_tokens: Some(600),
            cache_miss_tokens: Some(400),
            cost: Some(0.0042),
            cost_details: Some(r#"{"input":0.002,"output":0.0022}"#.to_string()),
            upstream_provider: Some("DeepSeek".to_string()),
            system_fingerprint: Some("fp_44709d6fcb".to_string()),
            duration_ms: 1_234,
            retry_attempts: 2,
            finish_reason: Some("stop".to_string()),
            failure_class: None,
            success: true,
            recorded_at: crate::turso::now(),
        };

        store
            .record_llm_request(&rec)
            .await
            .expect("insert llm request");

        let rows = store
            .conn
            .query(
                "SELECT purpose, agent_id, role, workspace, ticket_id, model, routing, \
                        input_tokens, output_tokens, cached_input_tokens, cache_miss_tokens, \
                        cost, cost_details, upstream_provider, system_fingerprint, \
                        duration_ms, retry_attempts, finish_reason, \
                        failure_class, success \
                 FROM llm_requests WHERE agent_id = ?1",
                crate::turso::params!["gui_alice_ws1_engineer"],
            )
            .await
            .expect("query llm request");
        let mut rows = rows.into_iter();
        let row = rows.next().expect("row must exist");
        assert_eq!(row.get::<String>(0).expect("purpose"), "agent");
        assert_eq!(
            row.get::<String>(1).expect("agent_id"),
            "gui_alice_ws1_engineer"
        );
        assert_eq!(row.get::<String>(2).expect("role"), "engineer");
        assert_eq!(row.get::<String>(3).expect("workspace"), "ws1");
        assert_eq!(
            row.get::<Option<String>>(4).expect("ticket_id"),
            Some("42".to_string())
        );
        assert_eq!(
            row.get::<String>(5).expect("model"),
            "deepseek/deepseek-v4-flash-0731"
        );
        assert_eq!(row.get::<String>(6).expect("routing"), "deepseek");
        assert_eq!(row.get::<Option<i64>>(7).expect("input"), Some(1_000));
        assert_eq!(row.get::<Option<i64>>(8).expect("output"), Some(200));
        assert_eq!(row.get::<Option<i64>>(9).expect("cached"), Some(600));
        assert_eq!(row.get::<Option<i64>>(10).expect("miss"), Some(400));
        let cost = row.get::<Option<f64>>(11).expect("cost");
        assert!((cost.expect("cost value") - 0.0042).abs() < 1e-12);
        assert_eq!(
            row.get::<Option<String>>(12).expect("cost_details"),
            Some(r#"{"input":0.002,"output":0.0022}"#.to_string())
        );
        assert_eq!(
            row.get::<Option<String>>(13).expect("upstream_provider"),
            Some("DeepSeek".to_string())
        );
        assert_eq!(
            row.get::<Option<String>>(14).expect("system_fingerprint"),
            Some("fp_44709d6fcb".to_string())
        );
        assert_eq!(row.get::<i64>(15).expect("duration"), 1_234);
        assert_eq!(row.get::<i64>(16).expect("attempts"), 2);
        assert_eq!(
            row.get::<Option<String>>(17).expect("finish"),
            Some("stop".to_string())
        );
        assert_eq!(row.get::<Option<String>>(18).expect("failure_class"), None);
        assert_eq!(row.get::<i64>(19).expect("success"), 1);
        assert!(rows.next().is_none(), "only one row expected");
    }
}
