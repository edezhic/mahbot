//! The records the product writes about its own work: produced on the path that
//! earned them, written by a task of their own, never awaited by that path.
//!
//! Every record here is one nobody on the performing path consumes — a model
//! call's row, a failed attempt's row, one intercepted search's usage row, the
//! batch a session's tool calls finalize into, a command-line operation's row,
//! and the warning row an unexpected shell or browser event leaves behind. They
//! all land in the log store, whose ONE connection also serves that store's
//! periodic pass, its log-line writer and its views: awaiting such a write parks
//! the work that merely reported on itself until whatever holds that connection
//! lets go.
//!
//! So producing one is an [`enqueue`] and never a wait. The facts are read where
//! they happen — including the instant the record describes, which is stamped
//! then and never when the record is written — and one drain task writes what the
//! queue holds, in the order it was produced.
//!
//! # The queue's bound
//!
//! The queue holds at most [`MAX_QUEUED_ITEMS`] records and at most
//! [`MAX_QUEUED_BYTES`] bytes of their own text
//! ([`TelemetryRecord::estimated_bytes`]), so it is bounded in memory even when a
//! single record is large. A record that does not fit is DROPPED as a whole, never
//! awaited and never split, and counted in ROWS — one for a record that writes one
//! row, one per call for a tool-call batch lost with it — on the log store's own
//! write-failure surface, as its own number, so neither count hides the other.
//!
//! The condition is the queue being FULL, never the store being unhealthy: a burst
//! drops records while the store is perfectly healthy. A write that fails once the
//! record is dequeued is noted under its class and dropped with no retry, and a
//! storage-layer panic out of it is absorbed the same way, so an unhealthy store
//! cannot turn into an unbounded loop or take the drain task with it.
//!
//! A record produced while no log store exists — during boot, or in a
//! command-line dispatch that never initializes tracing — is not written and not
//! counted, and never takes a place in the queue.
//!
//! # Ending the process
//!
//! Whatever is still queued when the process ends is lost — the drain task is never
//! awaited on the way out, so nothing keeps the product alive for bookkeeping and
//! nothing grows without bound to avoid that loss. What the product writes on its
//! closing path — the Issues view's rows and the partly-finished-update record —
//! goes straight to the store there and never enters this queue.
//!
//! # Wipes
//!
//! A view that empties a class of records must not be followed by that class
//! landing again ([`wipe`]): the delete and the queue's purge both happen with the
//! drain's writes held off ([`WIPE_GATE`]), so a covered record is either written
//! before the wipe and removed by it, or taken out of the queue — trimmed to what
//! the delete left of it, and dropped whole when that is nothing. Neither is counted
//! as a drop, and a delete that failed purges nothing.

use crate::logs::{GrepTelemetryRow, LOG_STORE, LogEntry, LogStore};
use crate::stats::{CallSource, LlmFailureRow, LlmRequestRecord};
use crate::util::UnwrapPoison;
use futures_util::FutureExt;
use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::sync::OnceLock;

/// Most records the queue holds at once.
const MAX_QUEUED_ITEMS: usize = 4_096;

/// Most bytes of the queue's own text it holds at once. A record that does not
/// fit is dropped rather than queued, so one large record cannot grow the queue
/// past this.
const MAX_QUEUED_BYTES: usize = 1 << 20;

/// What one stored row costs beyond the text it carries: the columns' own
/// bookkeeping and the values the insert is built from.
const ROW_OVERHEAD: usize = 64;

/// The rows one `tool_calls` write contributes: an agent session's whole batch, or
/// the one row a command-line operation adds.
pub(crate) struct ToolCallBatch {
    pub source: CallSource,
    /// `"cli"`/`"cli"` for a command-line operation; the session's agent id and
    /// role otherwise.
    pub agent_id: String,
    pub role: String,
    pub workspace: String,
    /// The instant the batch describes, stamped where the session finalized, as
    /// the one timestamp every row of the batch carries (as it always has).
    pub recorded_at: String,
    pub records: Vec<crate::ToolCallRecord>,
}

/// One record waiting for the drain task.
pub(crate) enum TelemetryRecord {
    /// The per-model-call row (`llm_requests`).
    LlmRequest(LlmRequestRecord),
    /// The per-attempt failure row (`llm_failures`).
    LlmFailure(LlmFailureRow),
    /// One intercepted shell search's usage row (`grep_telemetry`).
    GrepTelemetry(GrepTelemetryRow),
    /// A session's tool calls, or the one row a command-line operation
    /// contributes — one table, told apart by [`ToolCallBatch::source`].
    ToolCalls(ToolCallBatch),
    /// The warning row an unexpected shell or browser event leaves (`logs`).
    StrayEvent(LogEntry),
}

impl TelemetryRecord {
    /// The bytes the queue counts this record as: the text its own columns take
    /// plus one row's bookkeeping. An estimate — the point is a bound on the
    /// queue, not the store's own page accounting — but every string a record
    /// carries is counted here, so a field added to one of them must be added to
    /// its arm.
    fn estimated_bytes(&self) -> usize {
        /// The length of a column that may be absent.
        fn optional(text: Option<&String>) -> usize {
            text.map_or(0, String::len)
        }

        /// The text a stored JSON value takes, counted without building it — a
        /// stray event's `fields` hold text of unbounded length, so the bound has
        /// to see it. A lower bound: JSON's own punctuation and escapes are not
        /// counted, and neither is the space a value takes beyond its text.
        fn json_len(value: &serde_json::Value) -> usize {
            match value {
                serde_json::Value::Null => 4,
                serde_json::Value::Bool(_) | serde_json::Value::Number(_) => 8,
                serde_json::Value::String(text) => text.len(),
                serde_json::Value::Array(items) => items.iter().map(json_len).sum(),
                serde_json::Value::Object(fields) => fields
                    .iter()
                    .map(|(key, value)| key.len() + json_len(value))
                    .sum(),
            }
        }
        let text = match self {
            Self::LlmRequest(rec) => {
                rec.agent_id.len()
                    + rec.role.len()
                    + rec.workspace.len()
                    + rec.model.len()
                    + rec.routing.len()
                    + rec.recorded_at.len()
                    + rec.purpose.len()
                    + optional(rec.ticket_id.as_ref())
                    + optional(rec.cost_details.as_ref())
                    + optional(rec.upstream_provider.as_ref())
                    + optional(rec.system_fingerprint.as_ref())
                    + optional(rec.finish_reason.as_ref())
                    + optional(rec.failure_class.as_ref())
            }
            Self::LlmFailure(rec) => {
                rec.operation_id.len()
                    + rec.agent_id.len()
                    + rec.role.len()
                    + rec.workspace.len()
                    + rec.model.len()
                    + rec.routing.len()
                    + rec.recorded_at.len()
                    + rec.purpose.len()
                    + rec.failure_class.len()
                    + rec.error_chain.len()
                    + optional(rec.ticket_id.as_ref())
                    + optional(rec.finish_reason.as_ref())
            }
            Self::GrepTelemetry(row) => {
                row.command.len()
                    + row.reason.len()
                    + row.flags.len()
                    + row.mode.len()
                    + row.workspace.len()
                    + row.recorded_at.len()
            }
            Self::ToolCalls(batch) => {
                batch.agent_id.len()
                    + batch.role.len()
                    + batch.workspace.len()
                    + batch.recorded_at.len()
                    + batch
                        .records
                        .iter()
                        .map(|call| {
                            call.tool_name.len()
                                + call.arguments.len()
                                + optional(call.error_message.as_ref())
                                + ROW_OVERHEAD
                        })
                        .sum::<usize>()
            }
            Self::StrayEvent(entry) => {
                entry.timestamp.len()
                    + entry.level.len()
                    + entry.target.len()
                    + entry.message.len()
                    + entry.agent_id.len()
                    + entry.agent_role.len()
                    + entry.workspace.len()
                    + entry.ticket.len()
                    + json_len(&entry.fields)
            }
        };
        text + ROW_OVERHEAD
    }

    /// How many rows a drop of this record loses: one for a record that writes
    /// one row, one per call for a batch that is lost with it.
    fn lost_rows(&self) -> u64 {
        match self {
            Self::ToolCalls(batch) => batch.records.len() as u64,
            _ => 1,
        }
    }
}

/// A record waiting to be written, with the bytes it is counted as.
struct Queued {
    record: TelemetryRecord,
    bytes: usize,
}

/// What the queue holds: the waiting records, oldest first, and the bytes they
/// take. One lock, because a record's presence and its bytes are one fact.
#[derive(Default)]
struct QueueState {
    items: VecDeque<Queued>,
    bytes: usize,
}

/// The one queue the drain task reads.
struct Queue {
    state: std::sync::Mutex<QueueState>,
    /// Wakes the drain task. A push that happens after the drain found the queue
    /// empty leaves a permit behind, so a pushed record is never missed.
    wake: tokio::sync::Notify,
}

/// The product's telemetry queue, created once by [`start`] — or by a test's own
/// store install, since no test runs the boot that calls it.
///
/// It and a reachable store are what an [`enqueue`] needs: with no queue there is
/// nothing to hand a record to, and with no store there is nowhere for it to go,
/// so in either case the record is recorded nowhere and counted nowhere.
static QUEUE: OnceLock<Queue> = OnceLock::new();

/// Guards [`start`], so the process has exactly one drain task.
static DRAIN_STARTED: OnceLock<()> = OnceLock::new();

/// Held across the drain's take-and-write and across a class wipe's delete and
/// purge, so a wipe can neither miss a record that is mid-write nor be overtaken
/// by one that was taken before it.
static WIPE_GATE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Start the product's one drain task. Called once the log store is open; a
/// record produced before that — or in a process that never opens one — has
/// nowhere to go and is dropped without being counted.
pub(crate) fn start() {
    let queue = QUEUE.get_or_init(Queue::new);
    if DRAIN_STARTED.set(()).is_ok() {
        tokio::spawn(drain(queue));
    }
}

/// Hand one record to the drain task; nothing here waits for the log store.
pub(crate) fn enqueue(record: TelemetryRecord) {
    if let Some(queue) = sink() {
        queue.push(record);
    }
}

/// Whether [`enqueue`] would take a record — what a caller whose record is not
/// free to build asks before building it (the `mahbot chrome` CLI, which runs
/// before any log store exists, is the product's case).
pub(crate) fn recording() -> bool {
    sink().is_some()
}

/// The queue a record can be handed to, or `None` when nothing would take it: no
/// queue ([`start`] never ran — boot, or a command-line dispatch), or no store for
/// it to reach ([`has_store`]).
fn sink() -> Option<&'static Queue> {
    let queue = QUEUE.get()?;
    has_store().then_some(queue)
}

/// Run one class's wipe — `delete` is the class's own statement, built but not
/// yet polled — and take the records it covers out of the queue.
///
/// `strip` is offered each queued record and returns whether it survives the
/// wipe. It may first trim the covered parts out of a record the delete only
/// partly covers: a covered row left in a record that still lands would reappear
/// behind the wipe, while a row the delete leaves is owed to the store. A record
/// whose every row the delete covers is dropped whole.
pub(crate) async fn wipe<S, D>(strip: S, delete: D) -> anyhow::Result<u64>
where
    S: FnMut(&mut TelemetryRecord) -> bool,
    D: std::future::Future<Output = anyhow::Result<u64>>,
{
    let _gate = WIPE_GATE.lock().await;
    let deleted = delete.await?;
    if let Some(queue) = QUEUE.get() {
        queue.purge(strip);
    }
    Ok(deleted)
}

impl Queue {
    fn new() -> Self {
        Self {
            state: std::sync::Mutex::new(QueueState::default()),
            wake: tokio::sync::Notify::new(),
        }
    }

    /// Queue one record, or drop it when the queue is full.
    ///
    /// The whole of the producing path's cost and of the overflow outcome: a
    /// record that does not fit is dropped whole and counted in the rows it loses
    /// ([`crate::logs::record_telemetry_drop`]); the drain task prints the line.
    fn push(&self, record: TelemetryRecord) {
        let bytes = record.estimated_bytes();
        let mut state = self.state.lock().unwrap_poison();
        if state.items.len() >= MAX_QUEUED_ITEMS || state.bytes + bytes > MAX_QUEUED_BYTES {
            drop(state);
            crate::logs::record_telemetry_drop(record.lost_rows());
            // A drop wakes the drain the way a push does, so the line about it is
            // printed: a record too large for a queue that is otherwise empty is
            // dropped with nothing else for the drain to notice it by.
            self.wake.notify_one();
            return;
        }
        state.bytes += bytes;
        state.items.push_back(Queued { record, bytes });
        drop(state);
        self.wake.notify_one();
    }

    /// Take the oldest waiting record, if any.
    fn take(&self) -> Option<Queued> {
        let mut state = self.state.lock().unwrap_poison();
        let queued = state.items.pop_front()?;
        state.bytes -= queued.bytes;
        Some(queued)
    }

    /// Drop everything waiting, uncounted — the test seam's fresh start.
    #[cfg(test)]
    fn clear(&self) {
        *self.state.lock().unwrap_poison() = QueueState::default();
    }

    /// Take the records `strip` does not keep out of the queue, charging the
    /// bytes the survivors gave up.
    fn purge(&self, mut strip: impl FnMut(&mut TelemetryRecord) -> bool) {
        let mut state = self.state.lock().unwrap_poison();
        let mut freed = 0;
        state.items.retain_mut(|queued| {
            if !strip(&mut queued.record) {
                freed += queued.bytes;
                return false;
            }
            // A record the wipe only trimmed holds less than it did; one it left
            // whole keeps the count it was admitted with, even if `strip` gave it
            // more, so the queue's byte total always matches what it holds.
            let bytes = queued.record.estimated_bytes().min(queued.bytes);
            freed += queued.bytes - bytes;
            queued.bytes = bytes;
            true
        });
        state.bytes -= freed;
    }
}

/// The drain task: write what is queued, then sleep until more arrives.
///
/// It lives on the runtime and is never awaited on the way out, so whatever is
/// still queued when the process ends is simply lost.
async fn drain(queue: &'static Queue) {
    loop {
        write_queued(queue).await;
        queue.wake.notified().await;
    }
}

/// Write everything the queue holds, oldest first. Returns whether any record was
/// taken off the queue — a failed write counts, since the record is gone either
/// way. A test with no drain task ([`drain_for_test`]) calls this until the queue
/// is empty.
async fn write_queued(queue: &Queue) -> bool {
    let mut wrote = false;
    loop {
        // The drops a producing path only counted are reported from here, so no
        // thread doing the work ever writes to stderr about its own bookkeeping.
        crate::logs::report_telemetry_drops();
        // Held across the take AND the write: a wipe either runs before this take
        // (and purges what it finds still queued) or after this write (and
        // removes what it wrote) — never in between.
        let _gate = WIPE_GATE.lock().await;
        let Some(queued) = queue.take() else {
            return wrote;
        };
        let Some(store) = store() else {
            // The store went away under us (a test's seam): nothing is written,
            // and nothing is counted — a drop is the queue being full.
            continue;
        };
        // A storage-layer panic (a broken connection is how the engine reports
        // some of them) must not take the drain task with it: the record is
        // dropped like any failed write and the next one is still attempted.
        if let Err(payload) = AssertUnwindSafe(write(&store, queued.record))
            .catch_unwind()
            .await
        {
            crate::logs::record_telemetry_write_panic(&format!(
                "telemetry record write panic: {}",
                crate::util::panic_message(&*payload)
            ));
        }
        wrote = true;
    }
}

/// Write one record, noting a failure the way its class has always noted one: the
/// note leaves from this task, and the record is dropped with no retry.
async fn write(store: &LogStore, record: TelemetryRecord) {
    let outcome = match &record {
        TelemetryRecord::LlmRequest(rec) => store.record_llm_request(rec).await,
        TelemetryRecord::LlmFailure(rec) => store.record_llm_failure_row(rec).await,
        TelemetryRecord::GrepTelemetry(row) => store.record_grep_telemetry(row).await,
        TelemetryRecord::ToolCalls(batch) => {
            store
                .flush_batch(
                    &batch.agent_id,
                    &batch.role,
                    &batch.workspace,
                    batch.source,
                    &batch.recorded_at,
                    &batch.records,
                )
                .await
        }
        TelemetryRecord::StrayEvent(entry) => store.insert_batch(std::slice::from_ref(entry)).await,
    };
    if let Err(error) = outcome {
        note_write_failure(&record, &error);
    }
}

/// Note a failed write under the class that owns the record.
fn note_write_failure(record: &TelemetryRecord, error: &anyhow::Error) {
    match record {
        TelemetryRecord::LlmRequest(_) => {
            tracing::debug!(error = %error, "Failed to persist LLM request stat");
        }
        TelemetryRecord::LlmFailure(_) => {
            tracing::debug!(error = %error, "Failed to persist LLM attempt failure row");
        }
        TelemetryRecord::GrepTelemetry(_) => {
            tracing::debug!(error = %error, "Failed to persist grep telemetry");
        }
        TelemetryRecord::ToolCalls(batch) => match batch.source {
            CallSource::Agent => tracing::warn!(
                agent_id = %batch.agent_id,
                role = %batch.role,
                error = %error,
                "Failed to flush tool usage stats"
            ),
            CallSource::CommandLine => tracing::warn!(
                error = %error,
                tool = %batch.records.first().map_or("", |call| call.tool_name.as_str()),
                "cli: failed to record the command-line operation"
            ),
        },
        TelemetryRecord::StrayEvent(entry) => tracing::debug!(
            error = %error,
            target = %entry.target,
            "could not persist the leftover-process record"
        ),
    }
}

/// Whether there is a store for a produced record to reach.
///
/// In the product the queue and the store appear together ([`start`] runs with the
/// store open), so a record produced with no store finds no queue either
/// ([`enqueue`]) and is recorded and counted nowhere. What is left for this to
/// answer no to is the test seam ([`swap_test_log_store`]), which can take the
/// store away while the queue stays.
fn has_store() -> bool {
    #[cfg(test)]
    if TEST_LOG_STORE.read().unwrap_poison().is_some() {
        return true;
    }
    LOG_STORE.get().is_some()
}

/// The log store telemetry is written to; `None` while none is open.
fn store() -> Option<LogStore> {
    #[cfg(test)]
    if let Some(store) = TEST_LOG_STORE.read().unwrap_poison().as_ref() {
        return Some(store.clone());
    }
    LOG_STORE.get().cloned()
}

#[cfg(test)]
static TEST_LOG_STORE: std::sync::RwLock<Option<LogStore>> = std::sync::RwLock::new(None);

/// Test seam: point telemetry writes at a caller-owned store for the duration of
/// a test (mirrors `swap_provider_for_test`), and install the queue a test never
/// boots. Returns the previous override so a guard can restore it.
///
/// The redirect is PROCESS-GLOBAL while it is set, so a test holding it must hold
/// the serial keys covering every global it touches, and filter its queries by a
/// test-unique identifier. The queue is emptied here: a record another test
/// produced is not this test's to write.
#[cfg(test)]
pub(crate) fn swap_test_log_store(store: Option<LogStore>) -> Option<LogStore> {
    QUEUE.get_or_init(Queue::new).clear();
    let mut guard = TEST_LOG_STORE.write().unwrap_poison();
    let previous = guard.take();
    *guard = store;
    previous
}

/// Drain the queue with no task of its own — what a test calls before reading the
/// rows a record produces, since no test runs the boot that starts the drain.
#[cfg(test)]
pub(crate) async fn drain_for_test() {
    let Some(queue) = QUEUE.get() else {
        return;
    };
    while write_queued(queue).await {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::params;

    /// Marks the rows these tests write: the seam's store is process-global
    /// while it is installed, so a row another test's record put there first
    /// must not be mistaken for one of ours.
    const MARKER: &str = "telemetry-test";

    /// One intercepted shell call's usage row.
    fn telemetry_row(command: &str) -> GrepTelemetryRow {
        GrepTelemetryRow {
            command: command.to_string(),
            served: true,
            reason: String::new(),
            recursive: true,
            piped: false,
            operand_count: 2,
            flags: "-rn".to_string(),
            mode: "Full".to_string(),
            workspace: "/tmp/ws".to_string(),
            grep_count: 1,
            served_count: 1,
            skipped_count: 0,
            duration_ms: Some(3),
            exit_code: Some(0),
            recorded_at: crate::db::now(),
        }
    }

    /// A grep-usage record naming its run by `command`.
    fn grep_row(command: &str) -> TelemetryRecord {
        TelemetryRecord::GrepTelemetry(telemetry_row(command))
    }

    /// A log row the Logs views' clear can cover, at `level`, naming `message`.
    fn log_row(level: &str, message: &str) -> TelemetryRecord {
        TelemetryRecord::StrayEvent(LogEntry {
            timestamp: crate::db::now(),
            level: level.to_string(),
            target: "shell".to_string(),
            message: message.to_string(),
            ..LogEntry::default()
        })
    }

    /// A log row whose message is large enough to matter to the queue's budget.
    fn large_log_row(message_length: usize) -> TelemetryRecord {
        log_row("WARN", &format!("{MARKER}{}", "x".repeat(message_length)))
    }

    /// How many rows of `table` carry the marker in `column`.
    async fn count_marked(store: &LogStore, table: &str, column: &str) -> i64 {
        store
            .conn
            .query_row(
                &format!("SELECT COUNT(*) FROM {table} WHERE {column} LIKE ?1"),
                params![format!("{MARKER}%")],
                |row| row.get::<i64>(0),
            )
            .await
            .expect("count marked rows")
    }

    /// The messages of the marker's log rows, oldest first.
    async fn marked_messages(store: &LogStore) -> Vec<String> {
        store
            .conn
            .query(
                "SELECT message FROM logs WHERE message LIKE ?1 ORDER BY id",
                params![format!("{MARKER}%")],
            )
            .await
            .expect("read the remaining rows")
            .iter()
            .map(|row| row.get::<String>(0).unwrap())
            .collect()
    }

    /// [`start`] is what writes a record with no test drain driving it — the
    /// wiring a boot does once, and the one thing a record's fate depends on.
    #[tokio::test]
    #[serial_test::serial(provider)] // the telemetry store seam is process-global
    async fn the_task_started_for_the_product_writes_what_is_queued() {
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
        let _seam = crate::util::test::install_test_log_store(store.clone());
        start();

        enqueue(grep_row(&format!("{MARKER}-written-by-the-task")));

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while count_marked(&store, "grep_telemetry", "command").await == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "the drain task must write the queued record"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    /// A full queue drops a record instead of waiting for it or for the store,
    /// and the drop is counted on its own number: it is not a write failure.
    #[tokio::test]
    // the seam is process-global, and so are the failure counters this asserts on
    #[serial_test::serial(provider, log_surface)]
    async fn a_full_queue_drops_records_and_counts_them_apart_from_write_failures() {
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
        let _seam = crate::util::test::install_test_log_store(store.clone());
        let queue = Queue::new();
        let before = crate::logs::log_write_error_info();

        // Six records at a sixth of the byte budget each: five fit, and the sixth
        // is dropped rather than queued or awaited.
        for _ in 0..=5 {
            queue.push(large_log_row(MAX_QUEUED_BYTES / 6));
        }

        let after = crate::logs::log_write_error_info();
        assert!(
            after.dropped > before.dropped,
            "the record that did not fit must be counted as a drop"
        );
        assert_eq!(
            after.count, before.count,
            "a drop is the queue being full, not a store write failure"
        );

        write_queued(&queue).await;
        assert_eq!(
            count_marked(&store, "logs", "message").await,
            5,
            "the queue held only what its byte cap allows, and everything it held \
             must still be written"
        );
    }

    /// A store that refuses a write is noted on that record's own class line and
    /// the record dropped: the drain neither retries it nor stops.
    #[tokio::test]
    #[serial_test::serial(provider)] // the telemetry store seam is process-global
    async fn a_failed_write_is_noted_and_the_drain_keeps_going() {
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
        let _seam = crate::util::test::install_test_log_store(store.clone());
        let queue = Queue::new();
        // Deterministic insert failure: the table this record writes to is gone.
        store
            .conn
            .execute("DROP TABLE logs", ())
            .await
            .expect("drop the logs table for the failure test");

        queue.push(large_log_row(8));
        queue.push(grep_row(&format!("{MARKER}-after-the-failure")));
        // A refused write must not become a retry loop: the drain gets through
        // both records and comes back (that a refused write is not a drop is the
        // count assertion above).
        tokio::time::timeout(std::time::Duration::from_secs(30), write_queued(&queue))
            .await
            .expect("a failed write must not hold the drain");

        assert_eq!(
            count_marked(&store, "grep_telemetry", "command").await,
            1,
            "the next record must still be written"
        );
    }

    /// The headline this design exists for: with the log store's one connection
    /// held — the way any long operation on it holds it — the path that produced a
    /// record still finishes, and the record lands once the connection is free.
    #[tokio::test]
    #[serial_test::serial(provider)] // the telemetry store seam is process-global
    async fn producing_a_record_never_waits_for_a_stalled_store() {
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
        let _seam = crate::util::test::install_test_log_store(store.clone());
        // Wedge the store exactly as a stalled write does: its one writing
        // connection is held until this guard drops.
        let held = store
            .conn
            .begin_tx()
            .await
            .expect("hold the store's connection");
        // The wedge is real: a write does wait for that connection (a 200 ms cap
        // is far more than the insert needs).
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(200),
                store.record_grep_telemetry(&telemetry_row("wedged")),
            )
            .await
            .is_err(),
            "a store write must wait for the held connection"
        );

        let message = format!("{MARKER} stalled store");
        let event = crate::logs::StrayEvent {
            source: crate::logs::STRAY_SOURCE_SHELL,
            message: &message,
            recipe: "",
            scope: "",
            session: "",
            workspace: "/tmp/ws",
            duration_ms: Some(1),
        };
        // A write on this path would park this call for as long as the store is
        // held; producing the record must not.
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            crate::logs::record_stray_event(&event);
        })
        .await
        .expect("producing a record must not wait for the store");

        drop(held);
        tokio::time::timeout(std::time::Duration::from_secs(10), drain_for_test())
            .await
            .expect("the drain must write the record once the store is free");
        assert_eq!(
            count_marked(&store, "logs", "message").await,
            1,
            "the record the wedged store could not take must land once it is free"
        );
    }

    /// The Logs views' clear covers the warning rows the same way: what its scope
    /// reaches never lands after it, and what the scope leaves still does — the
    /// queued-record side of the delete's own `level IN (…)`.
    #[tokio::test]
    #[serial_test::serial(provider)] // the telemetry store seam is process-global
    async fn a_wipe_leaves_no_queued_row_of_the_log_scope_behind() {
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
        let _seam = crate::util::test::install_test_log_store(store.clone());

        // One covered row already written — the delete has to find it — and one row
        // the Issues scope never covers.
        enqueue(log_row("ERROR", &format!("{MARKER} written error")));
        enqueue(log_row("INFO", &format!("{MARKER} written info")));
        drain_for_test().await;
        // Three still queued when the wipe runs: two the scope covers, one it leaves.
        enqueue(log_row("ERROR", &format!("{MARKER} queued error")));
        enqueue(log_row("WARN", &format!("{MARKER} queued warning")));
        enqueue(log_row("INFO", &format!("{MARKER} queued info")));

        let deleted = store
            .clear_logs(Some("ERROR,WARN"))
            .await
            .expect("wipe the Issues scope");
        assert!(deleted >= 1, "the written error row is there to delete");

        drain_for_test().await;
        assert_eq!(
            marked_messages(&store).await,
            vec![
                format!("{MARKER} written info"),
                format!("{MARKER} queued info"),
            ],
            "the queued rows the scope covers must not land after the wipe, and the \
             rows it leaves must"
        );
    }

    /// The records land in the store in the order they were produced — the point
    /// of one queue drained by one task.
    #[tokio::test]
    #[serial_test::serial(provider)] // the telemetry store seam is process-global
    async fn records_are_written_in_the_order_they_were_produced() {
        let (store, _tmp) = crate::open_test_store!(crate::logs::LogStore, "log");
        let _seam = crate::util::test::install_test_log_store(store.clone());
        // A queue of this test's own: nothing another test produces can enter it.
        let queue = Queue::new();

        for command in ["first", "second", "third"] {
            queue.push(grep_row(&format!("{MARKER}-{command}")));
        }
        write_queued(&queue).await;

        let rows = store
            .conn
            .query(
                "SELECT command FROM grep_telemetry WHERE command LIKE ?1 ORDER BY id",
                params![format!("{MARKER}%")],
            )
            .await
            .expect("read the recorded runs");
        let commands: Vec<String> = rows
            .iter()
            .map(|row| row.get::<String>(0).unwrap())
            .collect();
        let expected: Vec<String> = ["first", "second", "third"]
            .iter()
            .map(|command| format!("{MARKER}-{command}"))
            .collect();
        assert_eq!(commands, expected);
    }
}
