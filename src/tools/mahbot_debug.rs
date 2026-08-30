//! In-process read-only SQL query tool for the Analyst role — wraps the
//! `mahbot debug` read-only query mechanism against the daemon's own live
//! connections (the consolidated `core.db` via [`crate::db::DOMAIN_CONN`], the
//! `logs.db` via [`crate::logs::LOG_STORE`]) so a trusted Analyst can query
//! them without a subprocess or a second store instance.

use crate::db::Value;
use crate::db::debug;
use crate::db::ipc;
use crate::util::TOOL_OUTPUT_BUDGET_BYTES;
use crate::{Tool, Workspace};
use anyhow::anyhow;

/// Struct implementing the [`Tool`] trait for the Analyst's read-only DB query.
pub(crate) struct MahbotDebugTool;

#[async_trait::async_trait]
impl Tool for MahbotDebugTool {
    fn name(&self) -> &'static str {
        "mahbot_debug"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        super::tool_params_schema(
            &serde_json::json!({
                "query": {
                    "type": "string",
                    "description": "A single read-only SQL statement (with multi-statement input only the first statement runs — the rest is silently ignored). Any mutating statement or non-read-only PRAGMA is rejected before execution — see the tool description for the full blocklist. Introspect table/column names first (sqlite_master / PRAGMA table_info) instead of guessing."
                },
                "db": {
                    "type": "string",
                    "description": format!(
                        "Target database. One of: {}. Defaults to 'core'.",
                        crate::db::debug_db_names().join(", ")
                    )
                }
            }),
            &["query"],
        )
    }

    fn side_effects(&self) -> bool {
        false // read-only — safe to group with other read-only tools
    }

    fn should_scrub_output(&self, _args: &serde_json::Value) -> bool {
        // The tool scrubs the full result (inline preview AND the on-disk spill
        // file) internally via `scrub_credentials` before spilling, so the
        // agent-level pass is disabled to avoid double-scrubbing the preview.
        false
    }

    async fn execute(&self, _ws: &Workspace, args: serde_json::Value) -> anyhow::Result<String> {
        let db = super::get_opt_str(&args, "db").unwrap_or("core");
        let query = super::get_str(&args, "query")?;

        debug::validate_store_name(db, false)?;
        let conn = if db == crate::db::LOG_DB_NAME {
            crate::logs::LOG_STORE.get().map(|s| &s.conn)
        } else {
            crate::db::DOMAIN_CONN.get()
        }
        .ok_or_else(|| anyhow!("the requested store '{db}' is not initialized"))?;

        let output = run_read_only_query(conn, query).await?;
        // The spill helper's contract is that the caller scrubs credentials
        // first; scrub the full result so the spill file (not just the inline
        // preview) is redacted consistently. Large result sets are then spilled
        // to a file so the agent reads the full output on demand instead of
        // cluttering context.
        let scrubbed = crate::util::scrub_credentials(&output);
        Ok(crate::tools::shell::try_spill_to_file(
            scrubbed,
            TOOL_OUTPUT_BUDGET_BYTES,
        ))
    }
}

/// Validate a read-only query and run it against `conn`, returning the result as
/// pipe-delimited text (same format as `mahbot debug`). Rejects any mutating
/// statement before touching the connection.
async fn run_read_only_query(conn: &crate::db::Connection, sql: &str) -> anyhow::Result<String> {
    debug::validate_read_only(sql)?;
    let rows = conn
        .query_readonly(sql, (), ipc::IPC_ROW_LIMIT)
        .await
        .map_err(|e| anyhow!("query failed: {e}"))?;
    Ok(format_readonly_rows(&rows))
}

/// Render [`crate::db::ReadonlyRows`] as pipe-delimited text: a column-header
/// line, one line per row, and a truncation sentinel line when the row limit was
/// hit. Values follow the CLI display rules (NULL→empty, Blob→lowercase hex).
fn format_readonly_rows(rows: &crate::db::ReadonlyRows) -> String {
    if rows.columns.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(&rows.columns.join("|"));
    out.push('\n');
    for row in &rows.rows {
        let parts: Vec<String> = row.iter().map(format_value).collect();
        out.push_str(&parts.join("|"));
        out.push('\n');
    }
    if rows.truncated {
        out.push_str(&debug::format_truncation_row(rows.columns.len()));
        out.push('\n');
    }
    out
}

/// Convert a [`turso::Value`] to its pipe-delimited display representation,
/// matching the `mahbot debug` CLI (NULL→empty, Integer→decimal,
/// Real→default float, Text→verbatim, Blob→lowercase hex).
fn format_value(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => f.to_string(),
        Value::Text(s) => s.clone(),
        Value::Blob(b) => crate::util::hex_string(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ReadonlyRows;

    #[test]
    fn formats_columns_rows_and_truncation_sentinel() {
        let rows = ReadonlyRows {
            columns: vec!["id".into(), "note".into()],
            rows: vec![
                vec![Value::Integer(1), Value::Text("a".into())],
                vec![Value::Null, Value::Blob(vec![0xde, 0xad])],
            ],
            truncated: true,
        };
        let out = format_readonly_rows(&rows);
        assert_eq!(out, "id|note\n1|a\n|dead\ntruncated|truncated\n");
    }

    #[test]
    fn description_loads_without_panicking() {
        // Guard against the mandatory prompt asset being missing: `load_prompt`
        // panics at agent construction, so `description()` must succeed.
        assert!(!MahbotDebugTool.description().is_empty());
    }

    #[test]
    fn schema_db_list_is_accepted_by_the_validator() {
        // Every `db` value the model-facing schema advertises must be accepted
        // by the runtime validator, and the validator must reject a decoy — so
        // the schema can't drift from the fail-closed accepted set.
        let schema = MahbotDebugTool.parameters_schema();
        let db_desc = schema
            .get("properties")
            .and_then(|v| v.get("db"))
            .and_then(|v| v.get("description"))
            .and_then(|v| v.as_str())
            .expect("`db` parameter must have a string description");
        for name in crate::db::debug_db_names() {
            assert!(db_desc.contains(name), "schema must advertise '{name}'");
            assert!(debug::validate_store_name(name, false).is_ok());
        }
        assert!(debug::validate_store_name("not_a_store", false).is_err());
    }

    #[tokio::test]
    async fn runs_a_read_only_query_on_a_real_store() {
        let (store, _dir) = crate::open_test_store!(crate::logs::LogStore, "log");
        let out = run_read_only_query(&store.conn, "SELECT 1 AS n, 'x' AS s")
            .await
            .expect("read-only query must succeed");
        assert_eq!(out, "n|s\n1|x\n");

        let rejected = run_read_only_query(&store.conn, "DROP TABLE logs").await;
        assert!(rejected.is_err(), "mutating query must be rejected");
    }
}
