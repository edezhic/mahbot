//! General CDC (change-data-capture) subscription engine over the Turso CDC log.
//!
//! [`drain_once`] drains the engine-managed `turso_cdc` table (created by
//! enabling `PRAGMA capture_data_changes_conn('full,turso_cdc')`), decodes the
//! business key for each changed row, and broadcasts a [`ChangeEvent`] to
//! subscribers of that table. It prunes consumed `turso_cdc` rows bottom-up so
//! the log stays bounded and the WAL `NewRowid` `change_id` stays monotonic for
//! new rows.
//!
//! This module writes no domain/timeline tables. The only write it performs is
//! pruning `turso_cdc` (which the engine self-excludes from capture) — the
//! "no domain writes" contract. Decoding is deliberately a Rust-side positional
//! read over the `before`/`after` full-record blobs (NOT
//! `bin_record_json_object`, which errors on tickets' embedding BLOB), using a
//! cached `PRAGMA table_info` column/PK map.
//!
//! The drainer invokes any registered synchronous ticket materializer
//! ([`register_ticket_materializer`]) before broadcasting to broadcast-channel
//! subscribers, so a non-lossy subscriber (e.g. the chronicle) observes every
//! tickets transition even when a broadcast receiver lags.
//!
//! ## Pinned-engine caveat (turso `=0.7.2`)
//!
//! - CDC is per-connection and mutually exclusive with MVCC (mahbot is WAL, so
//!   it does not apply).
//! - The pragma's `full` capture mode guarantees `before`/`after` are populated
//!   for every `change_type` (`1`=INSERT, `0`=UPDATE, `-1`=DELETE, `2`=COMMIT),
//!   so business keys are recoverable for all three mutations.
//! - In WAL mode `change_id` is a plain `NewRowid` (not a global monotonic
//!   cursor); the drainer therefore reads + prunes strictly in `change_id`
//!   order and re-drains surviving rows on restart (at-least-once + idempotent
//!   consumers).

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use turso::core::types::{ImmutableRecordRef, ValueRef};

use crate::db::{Connection, params};
use crate::util::UnwrapPoison;

/// Capacity of each per-table change-event broadcast channel.
const CHANNEL_CAPACITY: usize = 1024;

/// Maximum number of `turso_cdc` rows processed per drain batch (bounds the
/// per-cycle read for a large transaction; the prune keeps the table small).
const DRAIN_BATCH: i64 = 1000;

/// Fast poll interval while CDC rows are flowing.
const DRAIN_POLL: Duration = Duration::from_millis(100);

/// Idle poll interval cap, reached after successive empty drains. Board-transition
/// latency is covered by the synchronous flushes in `notify_ticket` / the Manager
/// path, so the drainer may back off when idle.
const DRAIN_IDLE_POLL: Duration = Duration::from_secs(1);

/// The engine-managed CDC table name (created when capture is enabled).
pub(crate) const CDC_TABLE: &str = "turso_cdc";

/// Tables that must never be broadcast (self-feed prevention). Writes to these on
/// the CDC-enabled connection would otherwise re-enter the drainer.
const EXCLUDED_TABLES: &[&str] = &[CDC_TABLE, "turso_cdc_version", "ticket_chronicle"];

/// Which mutation produced a CDC change row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    Insert,
    Update,
    Delete,
}

/// The resolved business key of a changed row.
#[derive(Debug, Clone, PartialEq)]
pub enum Pk {
    Text(String),
    Integer(i64),
}

/// A decoded record value.
#[derive(Debug, Clone, PartialEq)]
pub enum CdcValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl CdcValue {
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            _ => None,
        }
    }
}

/// A decoded full row (`before`/`after`), addressed by column name.
#[derive(Debug, Clone)]
pub struct RowRecord {
    columns: Vec<String>,
    values: Vec<Option<CdcValue>>,
}

impl RowRecord {
    fn from_blob(blob: &[u8], info: &TableInfo) -> anyhow::Result<Self> {
        let record = ImmutableRecordRef::from_bin_record(blob);
        let mut values = Vec::with_capacity(info.columns.len());
        let mut iter = record.iter()?;
        for _ in 0..info.columns.len() {
            match iter.next() {
                Some(Ok(v)) => values.push(Some(value_ref_to_cdc(&v))),
                Some(Err(e)) => return Err(e.into()),
                None => values.push(None),
            }
        }
        Ok(Self {
            columns: info.columns.clone(),
            values,
        })
    }

    /// Look up a column by name. Returns `None` when the column is absent or its
    /// value is SQL NULL.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&CdcValue> {
        self.columns
            .iter()
            .position(|c| c == name)
            .and_then(|idx| self.values[idx].as_ref())
    }
}

/// A CDC change event, broadcast to subscribers of [`Self::table`].
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    pub table: String,
    /// The drainer's ordering key (WAL `NewRowid`). **Not** used as a GUI
    /// watermark: it is a plain `NewRowid` that restarts after `turso_cdc` is
    /// fully pruned, so it is only informational today.
    pub change_id: i64,
    /// The original unix-second commit time of the change. Informational today;
    /// the chronicle subscriber prefers the after-record `updated_at` (microsecond
    /// precision) as the transition timestamp.
    pub change_time: i64,
    pub change_type: ChangeType,
    pub pk: Pk,
    /// Full row before the change (populated for UPDATE/DELETE in `full` mode).
    pub before: Option<RowRecord>,
    /// Full row after the change (populated for INSERT/UPDATE in `full` mode).
    pub after: Option<RowRecord>,
}

impl ChangeEvent {
    /// The tickets-table business key as a string id, when applicable.
    #[must_use]
    pub fn ticket_id(&self) -> Option<&str> {
        if self.table == "tickets" {
            match &self.pk {
                Pk::Text(id) => Some(id),
                Pk::Integer(_) => None,
            }
        } else {
            None
        }
    }
}

/// Cached `PRAGMA table_info` shape for a table. The PK columns in mahbot are
/// leading and never dropped, so the map is stable in practice; it is re-queried
/// when the table's column count drifts (a DDL change) rather than trusted
/// blindly.
#[derive(Debug, Clone)]
struct TableInfo {
    columns: Vec<String>,
    /// 0-based ordinals of the PK columns (parallel to `columns`).
    pk_ordinals: Vec<usize>,
}

impl TableInfo {
    async fn query(conn: &Connection, table: &str) -> anyhow::Result<Self> {
        let rows = conn
            .query(&format!("PRAGMA table_info({table})"), ())
            .await?;
        let mut columns = Vec::with_capacity(rows.len());
        let mut pk_ordinals = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let name: String = row.get_value(1)?.as_text().cloned().unwrap_or_default();
            let pk: i64 = row.get_value(5)?.as_integer().copied().unwrap_or(0);
            columns.push(name);
            if pk > 0 {
                pk_ordinals.push(i);
            }
        }
        Ok(Self {
            columns,
            pk_ordinals,
        })
    }
}

/// Cache of per-table [`TableInfo`], keyed by table name.
static TABLE_INFO: OnceLock<Mutex<HashMap<String, TableInfo>>> = OnceLock::new();

/// Registry of generic (non-tickets) per-table broadcast senders.
static TABLES: OnceLock<Mutex<HashMap<String, tokio::sync::broadcast::Sender<ChangeEvent>>>> =
    OnceLock::new();

/// The tickets-table broadcast sender (the only table the GUI + chronicle
/// consume today). Created on first access.
static TICKET_SENDER: OnceLock<tokio::sync::broadcast::Sender<ChangeEvent>> = OnceLock::new();

/// The shared tickets-table broadcast sender.
#[must_use]
pub(crate) fn ticket_sender() -> &'static tokio::sync::broadcast::Sender<ChangeEvent> {
    TICKET_SENDER.get_or_init(|| {
        let (tx, _rx) = tokio::sync::broadcast::channel(CHANNEL_CAPACITY);
        tx
    })
}

/// The tickets-table sender as a `OnceLock`, for `gui::common::broadcast_stream_producer`.
#[must_use]
pub(crate) fn ticket_sender_lock() -> &'static OnceLock<tokio::sync::broadcast::Sender<ChangeEvent>>
{
    &TICKET_SENDER
}

/// A synchronous subscriber invoked by the drainer for each tickets change
/// before broadcasting to the GUI. This is the chronicle's non-lossy materializer
/// (a broadcast channel would drop events on a lagging receiver, losing
/// transitions from the source-of-truth table). db::cdc itself performs no
/// domain/timeline write — the materializer's code owns its own writes.
type TicketMaterializer = Arc<
    dyn for<'a> Fn(&'a ChangeEvent) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send + 'a>>
        + Send
        + Sync,
>;
static TICKET_MATERIALIZER: OnceLock<TicketMaterializer> = OnceLock::new();

/// Register the synchronous tickets materializer (once, idempotent). Must run
/// before the drainer's first drain (typically right after store init).
pub(crate) fn register_ticket_materializer(
    f: impl for<'a> Fn(&'a ChangeEvent) -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send + 'a>>
    + Send
    + Sync
    + 'static,
) {
    let _ = TICKET_MATERIALIZER.set(Arc::new(f));
}

/// Subscribe to change events for `table`. Multiple subscribers on the same table
/// are independent; the sender is created on first subscribe. This is the general
/// multi-table subscription entry point (currently only the tickets stream has
/// production subscribers via the broadcast Sender); generic non-ticket tables
/// will use it once a consumer exists, so it is kept crate-visible.
#[allow(dead_code)]
pub(crate) fn subscribe(table: &str) -> tokio::sync::broadcast::Receiver<ChangeEvent> {
    if table == "tickets" {
        return ticket_sender().subscribe();
    }
    let mut map = TABLES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_poison();
    map.entry(table.to_string())
        .or_insert_with(|| {
            let (tx, _rx) = tokio::sync::broadcast::channel(CHANNEL_CAPACITY);
            tx
        })
        .subscribe()
}

/// Enable CDC on `conn`: capture mode `full` into [`CDC_TABLE`]. Idempotent (the
/// engine uses `CREATE TABLE IF NOT EXISTS` + `INSERT OR IGNORE`). Must run AFTER
/// migrations / consolidation import / boot hooks so those writes are not
/// captured.
pub(crate) async fn enable_capture(conn: &Connection) -> anyhow::Result<()> {
    conn.execute(
        &format!("PRAGMA capture_data_changes_conn('full,{CDC_TABLE}');"),
        (),
    )
    .await?;
    Ok(())
}

/// Spawn the background drain loop (once per process), pumping [`drain_once`]
/// at an adaptive cadence: fast (`DRAIN_POLL`) while rows are flowing, backing
/// off exponentially up to `DRAIN_IDLE_POLL` when idle. Board-transition latency
/// is covered by the synchronous flushes in `notify_ticket` / the Manager path.
///
/// The connection is the post-heal boot connection (the one every domain store
/// shares via the `DOMAIN_CONN` cell); heals only occur during the boot open,
/// before this runs, and the shared connection is process-stable after that —
/// so no runtime re-pointing is needed.
pub(crate) fn spawn_drainer(conn: Connection) {
    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.get().is_some() {
        return;
    }
    let _ = STARTED.set(());
    tokio::spawn(async move {
        let mut interval = DRAIN_POLL;
        loop {
            let saw_rows = match drain_once(&conn).await {
                Ok(seen) => seen,
                Err(e) => {
                    tracing::warn!(error = %e, "ticket CDC drain failed");
                    true
                }
            };
            if saw_rows {
                interval = DRAIN_POLL;
            } else {
                interval = (interval * 2).min(DRAIN_IDLE_POLL);
            }
            tokio::time::sleep(interval).await;
        }
    });
}

/// Drains a batch of `turso_cdc` rows: decode + broadcast subscribed-table
/// changes, then prune the consumed rows bottom-up. Public so tests can drive a
/// single deterministic drain. Rows that fail to decode (e.g. an engine-managed
/// shadow table) are logged and skipped — the batch still prunes, so a bad row
/// never stalls the drainer.
///
/// A row whose tickets materializer errors is left for the next drain (its
/// `change_id` is not pruned), so a transient DB failure during the chronicle
/// write is retried rather than permanently dropping the transition.
///
/// Concurrent invocations are serialized by [`DRAIN_LOCK`]: the drainer is
/// called both from the background loop and from synchronous flushes
/// (`notify_ticket`, the Manager path), and without serialization two drains
/// could interleave their prune with another drain's full-table prune, letting a
/// freshly-captured row whose `change_id` restarted low (after `turso_cdc` was
/// fully pruned) be deleted before it was ever processed.
///
/// Returns `Ok(true)` when the batch saw rows (including rows skipped by decode
/// or the materializer-failure break), `Ok(false)` when it was empty.
static DRAIN_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub(crate) async fn drain_once(conn: &Connection) -> anyhow::Result<bool> {
    let _drain_guard = DRAIN_LOCK.lock().await;
    let rows = conn
        .query_cached(
            // prepare_cached caches by SQL text, so the same formatted string
            // still hits the engine's per-connection statement cache.
            &format!(
                "SELECT change_id, change_time, change_type, \
                 table_name, id, before, after \
                 FROM {CDC_TABLE} ORDER BY change_id LIMIT ?1"
            ),
            params![DRAIN_BATCH],
        )
        .await?;
    if rows.is_empty() {
        return Ok(false);
    }
    // Cache table_info per batch for multi-row transactions.
    let mut infos: HashMap<String, TableInfo> = HashMap::new();
    // Set of non-ticket tables that have at least one broadcast subscriber, so
    // decode can be skipped (and the row pruned) for tables nobody consumes.
    let subscribed_tables: std::collections::HashSet<String> = {
        let mut s = std::collections::HashSet::new();
        if let Some(map) = TABLES.get().and_then(|m| m.lock().ok()) {
            s.extend(map.keys().cloned());
        }
        s
    };
    // Highest change_id whose row was fully delivered (decode + broadcast + a
    // successful materializer) — the prune bound. Rows before this are safe to
    // drop; rows at/after a materializer failure are left for the next drain so
    // a transient DB error is retried rather than permanently dropping the
    // transition (the at-least-once contract).
    let mut prune_up_to: i64 = -1;
    for row in &rows {
        let Some(change_id) = row.get_value(0).ok().and_then(|v| v.as_integer().copied()) else {
            continue;
        };
        let Some(table) = row.get_value(3).ok().and_then(|v| v.as_text().cloned()) else {
            // No table name — cannot determine the row; drop it so a malformed
            // row never stalls the drainer.
            prune_up_to = change_id;
            continue;
        };
        if EXCLUDED_TABLES.contains(&table.as_str()) {
            prune_up_to = change_id;
            continue;
        }
        let change_type = match row.get_value(2).ok().and_then(|v| v.as_integer().copied()) {
            Some(1) => ChangeType::Insert,
            Some(0) => ChangeType::Update,
            Some(-1) => ChangeType::Delete,
            // COMMIT (2) or an unknown type: no business row to broadcast.
            _ => {
                prune_up_to = change_id;
                continue;
            }
        };
        // Unsubscribed non-ticket tables are pruned but not decoded.
        if table != "tickets" && !subscribed_tables.contains(&table) {
            prune_up_to = change_id;
            continue;
        }
        match decode_row(conn, &mut infos, row, &table, change_type).await {
            Ok(event) => {
                // Non-lossy materializer (e.g. the chronicle) runs before the
                // broadcast so a re-delivered transition is materialized even if
                // a lagging receiver drops the broadcast copy.
                if event.table == "tickets"
                    && let Some(m) = TICKET_MATERIALIZER.get().cloned()
                    && let Err(e) = m(&event).await
                {
                    tracing::warn!(table = %event.table, change_id = event.change_id, error = %e, "ticket materializer failed");
                    // Leave this row (and later ones) for the next drain so the
                    // transition is retried, not permanently dropped.
                    break;
                }
                broadcast_event(&event);
                prune_up_to = change_id;
            }
            Err(e) => {
                tracing::warn!(table = %table, change_id, error = %e, "CDC row skipped (decode failed)");
                prune_up_to = change_id;
            }
        }
    }
    if prune_up_to >= 0 {
        conn.execute_cached(
            &format!("DELETE FROM {CDC_TABLE} WHERE change_id <= ?1"),
            params![prune_up_to],
        )
        .await?;
    }
    Ok(true)
}

/// Decode a single CDC row into a [`ChangeEvent`].
async fn decode_row(
    conn: &Connection,
    infos: &mut HashMap<String, TableInfo>,
    row: &turso::Row,
    table: &str,
    change_type: ChangeType,
) -> anyhow::Result<ChangeEvent> {
    if !infos.contains_key(table) {
        let info = cached_table_info(conn, table).await?;
        infos.insert(table.to_string(), info);
    }
    let info = infos.get(table).expect("inserted above");
    let change_id = row.get_value(0)?.as_integer().copied().unwrap_or_default();
    let change_time = row.get_value(1)?.as_integer().copied().unwrap_or_default();
    let rowid = row.get_value(4)?.as_integer().copied().unwrap_or_default();
    let before = row
        .get_value(5)?
        .as_blob()
        .map(|b| RowRecord::from_blob(b, info))
        .transpose()?;
    let after = row
        .get_value(6)?
        .as_blob()
        .map(|b| RowRecord::from_blob(b, info))
        .transpose()?;
    Ok(ChangeEvent {
        table: table.to_string(),
        change_id,
        change_time,
        change_type,
        pk: resolve_pk(info, rowid, before.as_ref(), after.as_ref(), change_type),
        before,
        after,
    })
}

/// Publish an event to its table's subscriber channel(s).
fn broadcast_event(event: &ChangeEvent) {
    if event.table == "tickets" {
        // Best-effort: a lagging receiver gets `Lagged` → the GUI forces a
        // full refresh rather than silently dropping the delta.
        let _ = ticket_sender().send(event.clone());
        return;
    }
    let Some(map) = TABLES.get() else {
        return;
    };
    let Ok(map) = map.lock() else {
        return;
    };
    if let Some(tx) = map.get(&event.table) {
        let _ = tx.send(event.clone());
    }
}

/// Resolve the business key from the record (before for DELETE, after for
/// INSERT/UPDATE), falling back to the rowid when the record is absent. A single
/// INTEGER PRIMARY KEY rowid alias also carries its value in the record, so the
/// record path covers both rowid-alias and TEXT-PK tables. Composite-PK tables
/// (none in mahbot today) fall back to the raw rowid — a known latent gap, since
/// no current subscriber reads one.
fn resolve_pk(
    info: &TableInfo,
    rowid: i64,
    before: Option<&RowRecord>,
    after: Option<&RowRecord>,
    change_type: ChangeType,
) -> Pk {
    let record = match change_type {
        ChangeType::Delete => before,
        _ => after,
    };
    if let Some(rec) = record
        && info.pk_ordinals.len() == 1
    {
        let idx = info.pk_ordinals[0];
        if let Some(v) = rec.values.get(idx).and_then(Option::as_ref) {
            return match v {
                CdcValue::Text(t) => Pk::Text(t.clone()),
                CdcValue::Integer(i) => Pk::Integer(*i),
                _ => Pk::Integer(rowid),
            };
        }
    }
    Pk::Integer(rowid)
}

/// Build (and cache) the column/PK map for `table`, re-querying when the
/// schema's column count drifts.
async fn cached_table_info(conn: &Connection, table: &str) -> anyhow::Result<TableInfo> {
    // Fast path: a cached map whose schema is unchanged. The clone is released
    // from the mutex guard before the await below, so the future stays `Send`.
    let cached = TABLE_INFO
        .get()
        .and_then(|m| m.lock().ok())
        .and_then(|g| g.get(table).cloned());
    if let Some(info) = cached {
        let fresh_count = conn
            .query(&format!("PRAGMA table_info({table})"), ())
            .await?
            .len();
        if fresh_count == info.columns.len() {
            return Ok(info);
        }
    }
    let info = TableInfo::query(conn, table).await?;
    let map = TABLE_INFO.get_or_init(|| Mutex::new(HashMap::new()));
    map.lock()
        .unwrap_poison()
        .insert(table.to_string(), info.clone());
    Ok(info)
}

/// Convert a Turso record value to our owned [`CdcValue`] via the public
/// `ValueRef` accessors (avoiding the private `Numeric` module).
fn value_ref_to_cdc(v: &ValueRef<'_>) -> CdcValue {
    if matches!(v, ValueRef::Null) {
        return CdcValue::Null;
    }
    if let Some(t) = v.to_text() {
        return CdcValue::Text(t.to_string());
    }
    if let Some(b) = v.to_blob() {
        return CdcValue::Blob(b.to_vec());
    }
    if let Some(i) = v.as_int() {
        return CdcValue::Integer(i);
    }
    CdcValue::Real(v.as_float())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn drain_defaults_to_v2_schema_and_decodes_pk() {
        let tmp = tempfile::TempDir::new().unwrap();
        let conn = crate::db::open_with_schema(
            &tmp.path().join("cdc.db"),
            "CREATE TABLE IF NOT EXISTS t (id TEXT PRIMARY KEY, phase TEXT NOT NULL, updated_at TEXT NOT NULL);",
        )
        .await
        .unwrap();
        enable_capture(&conn).await.unwrap();

        let mut rx = subscribe("t");
        conn.execute(
            "INSERT INTO t (id, phase, updated_at) VALUES ('mahbot-t', 'backlog', 'now')",
            (),
        )
        .await
        .unwrap();

        drain_once(&conn).await.unwrap();

        let event = rx.recv().await.expect("change event");
        assert_eq!(event.table, "t");
        assert_eq!(event.change_type, ChangeType::Insert);
        assert_eq!(event.pk, Pk::Text("mahbot-t".into()));
        // The decoded after record exposes the row by column name.
        let after = event.after.as_ref().expect("after record");
        assert_eq!(after.get("phase"), Some(&CdcValue::Text("backlog".into())));

        // Pruning: the consumed row is gone from turso_cdc.
        let remaining = conn
            .query(&format!("SELECT COUNT(*) FROM {CDC_TABLE}"), ())
            .await
            .unwrap();
        assert_eq!(remaining[0].get_value(0).unwrap().as_integer(), Some(&0));
    }
}
