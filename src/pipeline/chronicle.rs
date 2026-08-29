//! Persistent per-ticket phase-history timeline (`ticket_chronicle`), the
//! Manager's source of truth for grouped `<ticket-updates>` notifications.
//!
//! The in-memory transition buffer is gone. The timeline is materialized by a
//! CDC-driven subscriber (see [`start_subscriber`]) that observes `tickets`
//! phase-change events and inserts a `ticket_chronicle` row per transition —
//! the ONLY write path (no transition site writes the table). The subscriber is
//! a **synchronous** callback the CDC drainer invokes before broadcasting
//! ([`crate::db::cdc::register_ticket_materializer`]), not a lossy broadcast
//! channel: a transition is materialized by the time a [`drain`]/flush returns,
//! so the timeline cannot silently lose transitions on a lagged receiver.
//!
//! The Manager delivery reads from the table using a single in-memory watermark
//! per workspace: the monotonic AUTOINCREMENT `id` of the last row delivered,
//! initialized to the highest row id present at service start (so updates before
//! a restart are not re-delivered — a trivial, accepted edge case). `at` is the
//! transition's own RFC 3339 `updated_at` (microsecond precision, stable across
//! a restart re-drain), so a re-delivered transition dedupes instead of gaining a
//! fresh timestamp. Pruning is strictly per-workspace: a workspace's acked rows
//! are deleted only when that workspace's cursor has been seeded, and never
//! across workspaces (so one lagging workspace cannot drop another's history).
//!
//! Delivery is at-least-once + idempotent: the cursor is the monotonic
//! AUTOINCREMENT `id` (selection), and the INSERT OR IGNORE dedup index makes a
//! re-delivered transition a no-op. The terminal drain format is preserved:
//! consecutive hops of the same ticket are grouped (run-based) under a single
//! `• {id}:` header, each hop keeping its full RFC 3339 timestamp.

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::{Mutex, OnceLock};

use crate::db::{Connection, params};
use crate::pipeline::board::{TicketPhase, store as board};
use crate::util::UnwrapPoison;

/// Per-workspace manager delivery cursor: the last delivered chronicle row id.
/// `-1` means "not yet seeded from the service-start baseline". This is the
/// single in-memory watermark (the monotonic AUTOINCREMENT cursor) — strictly
/// more precise than a timestamp and restart-safe.
#[derive(Debug, Clone)]
struct Cursor {
    last_id: i64,
}

/// Per-workspace delivery cursors.
static CURSORS: OnceLock<Mutex<HashMap<String, Cursor>>> = OnceLock::new();

/// Serializes [`drain`]'s cursor read + advance. Without it, two concurrent
/// drains for the same workspace could both read the same watermark, select the
/// same rows, and deliver a duplicate `<ticket-updates>` block (the in-memory
/// buffer consumed atomically before this module became table-backed).
static CURSOR_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Service start timestamp (RFC 3339), captured once at boot. Only rows written
/// after this are ever delivered.
static SERVICE_START: OnceLock<String> = OnceLock::new();

/// Initialize the chronicle (cursor registry + service-start baseline).
///
/// Idempotent. Must be called during startup before any [`drain`].
pub fn init_global() {
    let _ = SERVICE_START.get_or_init(crate::db::now);
    CURSORS.get_or_init(|| Mutex::new(HashMap::new()));
}

fn cursor_for(workspace_name: &str) -> Cursor {
    let mut map = CURSORS
        .get()
        .expect("chronicle not initialized — call init_global() first")
        .lock()
        .unwrap_poison();
    let cursor = map
        .entry(workspace_name.to_string())
        .or_insert_with(|| Cursor { last_id: -1 });
    cursor.clone()
}

fn advance_cursor(workspace_name: &str, last_id: i64) {
    let mut map = CURSORS
        .get()
        .expect("chronicle not initialized — call init_global() first")
        .lock()
        .unwrap_poison();
    if let Some(cursor) = map.get_mut(workspace_name) {
        cursor.last_id = last_id;
    }
}

/// Register the CDC-driven chronicle subscriber (once per process). The
/// subscriber is a synchronous callback invoked by the drainer (not a lossy
/// broadcast channel), so the timeline cannot silently lose transitions on a
/// lagged receiver, and a transition is materialized by the time a flush
/// (`db::cdc::drain_once`) returns.
pub fn start_subscriber() {
    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.get().is_some() {
        return;
    }
    let _ = STARTED.set(());
    let conn = board().conn.clone();
    crate::db::cdc::register_ticket_materializer(move |event| {
        let conn = conn.clone();
        Box::pin(async move {
            // Only prune when a transition was actually materialized, so the
            // per-event DELETE below is skipped for tickets inserts/deletes and
            // no-op updates.
            if apply_change(&conn, event).await? {
                prune_acked(&conn).await?;
            }
            Ok(())
        })
    });
}

/// Materialize one ticket change into a `ticket_chronicle` row when the phase
/// changed. This is the only write path to the table. Returns whether a row was
/// actually inserted (`false` for non-tickets events, non-update events, no
/// phase change, or a duplicate that `INSERT OR IGNORE` skipped).
async fn apply_change(
    conn: &Connection,
    event: &crate::db::cdc::ChangeEvent,
) -> anyhow::Result<bool> {
    if event.table != "tickets" || event.change_type != crate::db::cdc::ChangeType::Update {
        return Ok(false);
    }
    let before_phase = event
        .before
        .as_ref()
        .and_then(|r| r.get("phase"))
        .and_then(crate::db::cdc::CdcValue::as_text);
    let after_phase = event
        .after
        .as_ref()
        .and_then(|r| r.get("phase"))
        .and_then(crate::db::cdc::CdcValue::as_text);
    let (Some(before), Some(after)) = (before_phase, after_phase) else {
        return Ok(false);
    };
    if before == after {
        return Ok(false);
    }
    let source: TicketPhase = before.parse()?;
    let target: TicketPhase = after.parse()?;
    let ticket_id = event.ticket_id().unwrap_or_default();
    let workspace = event
        .after
        .as_ref()
        .and_then(|r| r.get("workspace_name"))
        .and_then(crate::db::cdc::CdcValue::as_text)
        .unwrap_or_default();
    // `at` is the transition's own RFC 3339 `updated_at` from the after record
    // (microsecond precision, stable across a re-drain) — so two distinct
    // same-second transitions of one ticket get distinct values and the dedup
    // index cannot collapse them. Falls back to the CDC unix-second commit time.
    let at = event
        .after
        .as_ref()
        .and_then(|r| r.get("updated_at"))
        .and_then(crate::db::cdc::CdcValue::as_text)
        .map_or_else(
            || {
                chrono::DateTime::<chrono::Utc>::from_timestamp(event.change_time, 0)
                    .map_or_else(crate::db::now, |dt| dt.to_rfc3339())
            },
            str::to_string,
        );
    let rows = conn
        .execute(
            "INSERT OR IGNORE INTO ticket_chronicle (ticket_id, workspace_name, source_phase, \
             target_phase, at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![ticket_id, workspace, source.as_ref(), target.as_ref(), at],
        )
        .await?;
    Ok(rows > 0)
}

/// Prune `ticket_chronicle` rows each workspace's manager has acked, strictly
/// per-workspace. A workspace whose cursor is unseeded (`last_id == -1`) keeps
/// its whole table — it has never drained, so deleting its rows would drop
/// timeline entries from the manager's source of truth.
async fn prune_acked(conn: &Connection) -> anyhow::Result<()> {
    let pairs: Vec<(String, i64)> = {
        let map = CURSORS
            .get()
            .expect("chronicle not initialized — call init_global() first")
            .lock()
            .unwrap_poison();
        map.iter()
            .filter(|(_, c)| c.last_id >= 0)
            .map(|(ws, c)| (ws.clone(), c.last_id))
            .collect()
    };
    for (workspace, last_id) in pairs {
        conn.execute(
            "DELETE FROM ticket_chronicle WHERE workspace_name = ?1 AND id <= ?2",
            params![workspace, last_id],
        )
        .await?;
    }
    Ok(())
}

/// Drain the un-delivered `ticket_chronicle` rows for a workspace, advancing the
/// workspace's watermark. Returns the formatted `<ticket-updates>` block, or an
/// empty string when nothing new is pending.
///
/// The first drain for a workspace seeds the cursor from the service-start
/// baseline (max id of rows written before the service started), so only
/// post-start transitions are delivered.
///
/// # Panics
///
/// Panics if [`init_global`] has not been called, or if the board store is not
/// initialized (the table lives on the shared domain connection).
#[must_use]
pub(crate) async fn drain(workspace_name: &str) -> String {
    let conn = &board().conn;
    // Hold the cursor lock across the read + advance so a concurrent drain for
    // the same workspace cannot read the same watermark and deliver a duplicate
    // block (see [`CURSOR_LOCK`]). The formatted block is built after release.
    let cursor_guard = CURSOR_LOCK.lock().await;
    let mut cursor = cursor_for(workspace_name);
    if cursor.last_id < 0 {
        let service_start = SERVICE_START.get().cloned().unwrap_or_else(crate::db::now);
        let seed: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(id), 0) FROM ticket_chronicle \
                 WHERE workspace_name = ?1 AND at <= ?2",
                params![workspace_name, service_start],
                |row| row.get(0),
            )
            .await
            .unwrap_or(0);
        cursor.last_id = seed;
        // Persist the seed even when there are no rows, so the seed query is
        // not re-run (and re-evaluated) on every subsequent drain.
        advance_cursor(workspace_name, seed);
    }
    let rows = match conn
        .query(
            "SELECT id, ticket_id, source_phase, target_phase, at \
             FROM ticket_chronicle \
             WHERE workspace_name = ?1 AND id > ?2 ORDER BY id",
            params![workspace_name, cursor.last_id],
        )
        .await
    {
        Ok(rows) => rows,
        Err(e) => {
            tracing::warn!(workspace = workspace_name, error = %e, "chronicle drain failed");
            return String::new();
        }
    };
    if rows.is_empty() {
        return String::new();
    }
    let hops: Vec<Hop> = rows
        .iter()
        .map(|row| Hop {
            id: row
                .get_value(1)
                .ok()
                .and_then(|v| v.as_text().cloned())
                .unwrap_or_default(),
            source: row
                .get_value(2)
                .ok()
                .and_then(|v| v.as_text().cloned())
                .unwrap_or_default(),
            target: row
                .get_value(3)
                .ok()
                .and_then(|v| v.as_text().cloned())
                .unwrap_or_default(),
            at: row
                .get_value(4)
                .ok()
                .and_then(|v| v.as_text().cloned())
                .unwrap_or_default(),
        })
        .collect();
    let last_id = rows
        .iter()
        .filter_map(|row| row.get_value(0).ok())
        .filter_map(|v| v.as_integer().copied())
        .max()
        .unwrap_or(cursor.last_id);
    advance_cursor(workspace_name, last_id);
    drop(cursor_guard);
    // Prune acked rows now so a workspace that drains and then goes quiet does
    // not keep its delivered timeline rows until the next transition materializes.
    if let Err(e) = prune_acked(conn).await {
        tracing::warn!(error = %e, "chronicle prune after drain failed");
    }
    format_chronicle(&hops)
}

/// One rendered transition hop.
#[derive(Debug, Clone)]
struct Hop {
    id: String,
    source: String,
    target: String,
    at: String,
}

/// Render hops into the `<ticket-updates>` block. Consecutive hops of the same
/// ticket are grouped (run-based) under a single header; new header on any
/// ticket change. Each hop keeps its full RFC 3339 timestamp.
fn format_chronicle(hops: &[Hop]) -> String {
    if hops.is_empty() {
        return String::new();
    }
    let mut out = String::from("<ticket-updates>\n");
    let mut current: Option<&str> = None;
    for hop in hops {
        if current != Some(hop.id.as_str()) {
            let _ = writeln!(out, "• {}:", hop.id);
            current = Some(&hop.id);
        }
        let _ = writeln!(out, "    {} → {} ({})", hop.source, hop.target, hop.at);
    }
    let _ = writeln!(out, "</ticket-updates>");
    out
}

/// Reset all cursors for test isolation.
#[cfg(test)]
pub fn reset() {
    init_global();
    CURSORS.get().unwrap().lock().unwrap_poison().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_groups_consecutive_same_ticket_hops_under_one_header() {
        let hops = vec![
            Hop {
                id: "mahbot-1736".into(),
                source: "in_development".into(),
                target: "in_diagnostics".into(),
                at: "2026-08-17T08:11:34.225709+00:00".into(),
            },
            Hop {
                id: "mahbot-1736".into(),
                source: "in_diagnostics".into(),
                target: "in_review".into(),
                at: "2026-08-17T08:21:19.225709+00:00".into(),
            },
        ];
        let result = format_chronicle(&hops);
        assert!(result.starts_with("<ticket-updates>\n"));
        assert!(result.ends_with("</ticket-updates>\n"));
        assert_eq!(result.matches("• mahbot-1736:").count(), 1);
        assert!(
            result
                .contains("    in_development → in_diagnostics (2026-08-17T08:11:34.225709+00:00)")
        );
        assert!(
            result.contains("    in_diagnostics → in_review (2026-08-17T08:21:19.225709+00:00)")
        );
    }

    #[test]
    fn format_interleaved_tickets_are_run_grouped() {
        let hops = vec![
            Hop {
                id: "mahbot-A".into(),
                source: "backlog".into(),
                target: "analysis".into(),
                at: "2026-08-17T08:00:00+00:00".into(),
            },
            Hop {
                id: "mahbot-B".into(),
                source: "analysis".into(),
                target: "planning".into(),
                at: "2026-08-17T08:01:00+00:00".into(),
            },
            Hop {
                id: "mahbot-A".into(),
                source: "planning".into(),
                target: "queued".into(),
                at: "2026-08-17T08:02:00+00:00".into(),
            },
        ];
        let result = format_chronicle(&hops);
        assert_eq!(result.matches("• mahbot-A:").count(), 2);
        assert_eq!(result.matches("• mahbot-B:").count(), 1);
        let hop_a1 = result.find("    backlog → analysis").unwrap();
        let hop_b = result.find("    analysis → planning").unwrap();
        let hop_a2 = result.find("    planning → queued").unwrap();
        assert!(hop_a1 < hop_b && hop_b < hop_a2);
    }

    #[test]
    fn format_empty_returns_empty() {
        assert!(format_chronicle(&[]).is_empty());
    }

    /// End-to-end composition: a real ticket phase change is captured, the CDC
    /// drainer materializes it into `ticket_chronicle` (via the synchronous
    /// subscriber), and `drain` delivers the grouped `<ticket-updates>` block.
    /// Polls a few times so the test is deterministic regardless of whether the
    /// background drainer or the explicit `drain_once` wins the race.
    #[tokio::test]
    async fn cdc_materializes_chronicle_and_drain_delivers_grouped_block() {
        crate::util::test::init_test_stores().await;
        let tmp = tempfile::tempdir().unwrap();
        let ws =
            crate::util::test::create_test_workspace(tmp.path().to_str().unwrap(), "cdc-ws").await;
        let board = crate::pipeline::board::store();
        let ticket_id =
            crate::util::test::make_ticket(&board, &ws, "CDC ticket", TicketPhase::Backlog).await;
        board
            .transition_to(
                &ticket_id,
                Some(TicketPhase::Backlog),
                TicketPhase::Analysis,
            )
            .await
            .unwrap();

        let mut count = 0;
        for _ in 0..20 {
            crate::db::cdc::drain_once(&board.conn).await.unwrap();
            count = board
                .conn
                .query("SELECT COUNT(*) FROM ticket_chronicle", ())
                .await
                .unwrap()[0]
                .get_value(0)
                .unwrap()
                .as_integer()
                .copied()
                .unwrap_or(0);
            if count > 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert_eq!(
            count, 1,
            "a phase change materializes exactly one chronicle row"
        );

        let block = drain(ws.name.as_str()).await;
        assert!(block.contains("<ticket-updates>"));
        assert!(block.contains(&format!("• {ticket_id}:")));
        assert!(block.contains("backlog → analysis"));
    }
}
