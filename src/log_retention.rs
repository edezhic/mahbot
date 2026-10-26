//! Retention for the log store: the classes of records that accumulate merely
//! as a by-product of the product running, the window each is kept for, and the
//! paced sweep that removes the expired ones.
//!
//! The sweep never empties a class in one statement. The engine rejects
//! `DELETE … LIMIT`, so each statement bounds itself with a `LIMIT`ed id
//! subquery, and the sweep pauses between statements: the log store serves the
//! log writer, the telemetry drain, the GUI's pages and the debug IPC from ONE
//! connection, and a backlog must not be able to hold that connection through a
//! long pass.
//!
//! Progress is durable by construction: every batch is its own committed
//! statement and expiry is a predicate over the data rather than a cursor, so a
//! shutdown between two batches keeps what was already removed and the next
//! sweep — in this launch or in the next one — simply re-derives what is left.
//!
//! What this bounds is the store's live data, not its file: the engine hands
//! freed pages back to later writes rather than to the OS, and cannot compact a
//! store in place.

use crate::db::params;
use crate::logs::LogStore;
use anyhow::Context;
use chrono::{Duration as ChronoDuration, Utc};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// The window the accumulating telemetry is kept for.
const TELEMETRY_HOURS: i64 = 8;

/// Failed tool calls are the record of why a tool call failed, read back through
/// `mahbot debug` over the `tool_calls` table, so they are kept far longer than
/// the rest of the telemetry.
const FAILED_TOOL_CALL_HOURS: i64 = 24 * 7;

/// Records removed per statement.
const BATCH_ROWS: u64 = 1_000;

/// Pause between batches.
const BATCH_PAUSE: Duration = Duration::from_millis(50);

/// How long one sweep may keep deleting before it yields to the next one. A
/// backlog too large for one sweep is drained over several of them.
const SWEEP_BUDGET: Duration = Duration::from_secs(30);

/// How often the sweep runs.
const SWEEP_INTERVAL: Duration = Duration::from_mins(10);

/// One class of records and the policy for removing it.
struct RetentionRule {
    table: &'static str,
    /// The `WHERE` body selecting this class's expired records, with `?1` the
    /// cutoff — a class carries its own predicate whenever one table holds two
    /// policies.
    expired: &'static str,
    window_hours: i64,
}

/// The whole retention policy of the log store, one entry per accumulating
/// class of records.
///
/// Deliberately absent: the per-attempt trail of failed model calls
/// (`llm_failures`) and the WARN/ERROR rows of `logs`. Both are small, neither
/// grows, and keeping them is what lets a failure recurring over several days be
/// explained — for the latter also the Issues page's one-row-per-reason history.
const RETENTION_RULES: &[RetentionRule] = &[
    // Successful tool calls — a successful call is a fact of a run that is over,
    // kept only while the recent past is still worth inspecting.
    RetentionRule {
        table: "tool_calls",
        expired: "success = 1 AND error_message IS NULL AND recorded_at < ?1",
        window_hours: TELEMETRY_HOURS,
    },
    // Failed tool calls — a call recorded as failed, or one carrying error text
    // of its own. The two rules on this table have to partition it: a call the
    // failure rule takes must not also be a candidate for the 8 h one.
    RetentionRule {
        table: "tool_calls",
        expired: "(success = 0 OR error_message IS NOT NULL) AND recorded_at < ?1",
        window_hours: FAILED_TOOL_CALL_HOURS,
    },
    // Model-request telemetry.
    RetentionRule {
        table: "llm_requests",
        expired: "recorded_at < ?1",
        window_hours: TELEMETRY_HOURS,
    },
    // Grep telemetry — nothing in the product reads it; it is by-product.
    RetentionRule {
        table: "grep_telemetry",
        expired: "recorded_at < ?1",
        window_hours: TELEMETRY_HOURS,
    },
    // INFO log rows — the window this table has always been trimmed to, and the
    // only level it covers: WARN and ERROR are the Issues history, and the debug
    // levels only ever appear under a `RUST_LOG` override.
    RetentionRule {
        table: "logs",
        expired: "level = 'INFO' AND timestamp < ?1",
        window_hours: TELEMETRY_HOURS,
    },
];

impl RetentionRule {
    /// Remove at most [`BATCH_ROWS`] of this rule's expired records, returning
    /// how many were removed.
    async fn delete_batch(&self, store: &LogStore) -> anyhow::Result<u64> {
        let batch_rows = i64::try_from(BATCH_ROWS).expect("batch size fits in i64");
        let cutoff = (Utc::now() - ChronoDuration::hours(self.window_hours)).to_rfc3339();
        let sql = format!(
            "DELETE FROM {table} WHERE id IN (SELECT id FROM {table} WHERE {expired} LIMIT ?2)",
            table = self.table,
            expired = self.expired,
        );
        store
            .conn
            .execute(&sql, params![cutoff, batch_rows])
            .await
            .with_context(|| format!("Failed to delete expired rows from {}", self.table))
    }
}

/// One sweep, deleting in bounded batches until every rule is satisfied,
/// [`SWEEP_BUDGET`] is spent, or the shutdown/graceful drain begins. Returns how
/// many records were removed.
async fn sweep_expired(store: &LogStore) -> u64 {
    let deadline = Instant::now() + SWEEP_BUDGET;
    let mut removed = 0;
    let mut failed = vec![false; RETENTION_RULES.len()];
    loop {
        // A round takes a batch from every rule, so a rule with a large backlog
        // cannot starve the others; a rule returns a short batch once it has
        // nothing expired left, and that is what ends the sweep.
        let mut saturated = false;
        for (index, rule) in RETENTION_RULES.iter().enumerate() {
            if failed[index] {
                continue;
            }
            match rule.delete_batch(store).await {
                Ok(n) => {
                    removed += n;
                    saturated |= n == BATCH_ROWS;
                }
                // Logged and dropped for the rest of this sweep — a rule that
                // cannot delete must neither stop the rules after it nor be
                // retried round after round. The next sweep tries it again.
                Err(e) => {
                    warn!("Log retention: could not remove expired records: {e:#}");
                    failed[index] = true;
                }
            }
        }
        if !saturated || Instant::now() >= deadline {
            return removed;
        }
        if !crate::shutdown::sleep_or_shutdown_or_drain(BATCH_PAUSE).await {
            return removed;
        }
    }
}

/// Background loop: one paced sweep every [`SWEEP_INTERVAL`], the first one a
/// whole interval after start — never at the same instant the stores are coming
/// up.
pub async fn run_retention_loop(store: Arc<LogStore>) {
    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(SWEEP_INTERVAL).await {
            break;
        }
        match sweep_expired(&store).await {
            0 => debug!("Log retention: nothing to remove"),
            removed => debug!(removed, "Log retention: removed expired records"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run one seed statement, with the row's age in hours as `?1`.
    async fn seed(store: &LogStore, sql: &str, age_hours: i64) {
        let at = (Utc::now() - ChronoDuration::hours(age_hours)).to_rfc3339();
        store.conn.execute(sql, params![at]).await.unwrap();
    }

    async fn count(store: &LogStore, table: &str, predicate: &str) -> i64 {
        store
            .conn
            .query_row(
                &format!("SELECT COUNT(*) FROM {table} WHERE {predicate}"),
                params![],
                |row| row.get::<i64>(0),
            )
            .await
            .unwrap()
    }

    /// Every rule cuts at its own window and nothing else: a failed tool call
    /// outlives the 8 h its successful twin is cut at and goes once its 7 days
    /// are up — whether or not it carries error text — while the classes kept on
    /// purpose are untouched at any age.
    #[tokio::test]
    async fn only_records_past_their_window_are_removed() {
        let (store, _dir) = crate::open_test_store!(LogStore, "log");
        for age in [1, 9] {
            seed(
                &store,
                "INSERT INTO tool_calls (agent_id, role, tool_name, recorded_at) \
                 VALUES ('a', 'Worker', 'read', ?1)",
                age,
            )
            .await;
            seed(
                &store,
                "INSERT INTO tool_calls (agent_id, role, tool_name, error_message, recorded_at) \
                 VALUES ('a', 'Worker', 'read', 'boom', ?1)",
                age,
            )
            .await;
            // A call recorded as failed with no error text of its own.
            seed(
                &store,
                "INSERT INTO tool_calls (agent_id, role, tool_name, success, recorded_at) \
                 VALUES ('a', 'Worker', 'read', 0, ?1)",
                age,
            )
            .await;
            seed(
                &store,
                "INSERT INTO llm_requests (recorded_at, purpose, model, duration_ms, retry_attempts) \
                 VALUES (?1, 'p', 'm', 1, 0)",
                age,
            )
            .await;
            seed(
                &store,
                "INSERT INTO grep_telemetry (recorded_at, command) VALUES (?1, 'grep x')",
                age,
            )
            .await;
            for level in ["INFO", "DEBUG"] {
                seed(
                    &store,
                    &format!(
                        "INSERT INTO logs (timestamp, level, target, message) VALUES (?1, '{level}', 't', 'm')"
                    ),
                    age,
                )
                .await;
            }
        }

        // Both shapes of failed call past their 7 days, plus the two classes that
        // are kept on purpose whatever their age.
        seed(
            &store,
            "INSERT INTO tool_calls (agent_id, role, tool_name, error_message, recorded_at) \
             VALUES ('a', 'Worker', 'read', 'boom', ?1)",
            24 * 8,
        )
        .await;
        seed(
            &store,
            "INSERT INTO tool_calls (agent_id, role, tool_name, success, recorded_at) \
             VALUES ('a', 'Worker', 'read', 0, ?1)",
            24 * 8,
        )
        .await;
        seed(
            &store,
            "INSERT INTO logs (timestamp, level, target, message) VALUES (?1, 'WARN', 't', 'm')",
            24 * 30,
        )
        .await;
        seed(
            &store,
            "INSERT INTO llm_failures (operation_id, recorded_at, purpose, model, attempt, failure_class) \
             VALUES ('op', ?1, 'p', 'm', 1, 'c')",
            24 * 30,
        )
        .await;

        sweep_expired(&store).await;

        assert_eq!(
            count(
                &store,
                "tool_calls",
                "success = 1 AND error_message IS NULL"
            )
            .await,
            1
        );
        assert_eq!(
            count(
                &store,
                "tool_calls",
                "success = 0 OR error_message IS NOT NULL"
            )
            .await,
            4
        );
        assert_eq!(count(&store, "llm_requests", "1 = 1").await, 1);
        assert_eq!(count(&store, "grep_telemetry", "1 = 1").await, 1);
        assert_eq!(count(&store, "logs", "level = 'INFO'").await, 1);
        // The levels the sweep does not settle are left alone, whatever their age.
        assert_eq!(count(&store, "logs", "level = 'DEBUG'").await, 2);
        assert_eq!(count(&store, "logs", "level = 'WARN'").await, 1);
        assert_eq!(count(&store, "llm_failures", "1 = 1").await, 1);
    }

    /// A backlog goes in bounded batches: one statement never removes more than
    /// a batch, and the sweep keeps taking batches until the class is clean.
    #[tokio::test]
    #[serial_test::serial(drain)] // serializes the process-global drain flag
    async fn a_backlog_is_removed_in_bounded_batches() {
        let (store, _dir) = crate::open_test_store!(LogStore, "log");
        let at = (Utc::now() - ChronoDuration::hours(9)).to_rfc3339();
        let tx = store.conn.begin_tx().await.unwrap();
        for _ in 0..2_500 {
            tx.execute(
                "INSERT INTO grep_telemetry (recorded_at, command) VALUES (?1, 'grep x')",
                params![at.clone()],
            )
            .await
            .unwrap();
        }
        tx.commit().await.unwrap();

        let rule = RETENTION_RULES
            .iter()
            .find(|rule| rule.table == "grep_telemetry")
            .expect("grep telemetry is a retained class");
        assert_eq!(
            rule.delete_batch(&store).await.unwrap(),
            BATCH_ROWS,
            "one statement removes exactly one batch"
        );
        assert_eq!(
            count(&store, "grep_telemetry", "1 = 1").await,
            1_500,
            "and leaves the rest of the backlog for the next batch"
        );

        sweep_expired(&store).await;
        assert_eq!(count(&store, "grep_telemetry", "1 = 1").await, 0);
    }
}
