//! The durable outcome of one command-line `send`: the record its id answers.
//!
//! A command line's handover is answered and gone the moment it returns, so the only thing a
//! script can hold is the id its acknowledgement printed. This module keeps that id's record:
//! opened when the handover is accepted, before its work leaves; completed when the addressed
//! turn's answer is final, before it is wrapped for delivery; read back by `mahbot outcome`.
//!
//! The record's own existence is what tells a command-line handover apart from every other kind
//! of work that shares a job id: an id belonging to an analysis round, a ticket phase or an
//! agent's own `send` has no record and answers `unknown`, with no other marker invented.
//!
//! It is opened once and completed once. Completion only ever moves a record out of `running`, so
//! a later turn under the same id cannot overwrite an answer already reported. A record is kept
//! [`RETENTION_HOURS`] after it finishes — or after it was created, when it never finished.
//!
//! Nothing here writes a record on its own: the handover's own path opens and completes it, a
//! refused one fails it, and the boot sweep settles what a stopped process left behind.

use crate::db::{self, params};
use anyhow::Context;
use std::collections::HashSet;
use tracing::warn;

/// How long a record is kept after it finishes, or after it was created when it never finished.
pub const RETENTION_HOURS: i64 = 24;

/// What a record left `running` by a stopped process is told: the handover never got its answer.
const STOPPED_MID_FLIGHT: &str = "the product stopped before this handover's answer was delivered";

/// How many expired records one delete removes: the core store is one shared writer, so the sweep
/// bounds itself rather than holding the connection through a long pass.
const PURGE_BATCH: i64 = 500;

/// The pause between two purge batches, so a backlog cannot monopolise the connection.
const PURGE_PAUSE: std::time::Duration = std::time::Duration::from_millis(50);

/// How long one sweep may keep deleting before it yields to the next one, checked between
/// batches. A backlog too large for one sweep is drained over several of them.
const PURGE_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

/// The five states a lookup can report — four of them a record's own, the fifth the absence of
/// one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum State {
    /// The handover was accepted and has not finished: not started, or in flight.
    Running,
    /// The addressed turn produced an answer.
    Done,
    /// The addressed turn finished with nothing to report — a normal `manager` ending.
    DoneNoAnswer,
    /// The handover ended without an answer; [`Record::reason`] says why.
    Failed,
}

impl State {
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => "done",
            Self::DoneNoAnswer => "done_no_answer",
            Self::Failed => "failed",
        }
    }

    /// The state a stored string names, or `None` for a value this build does not know.
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "running" => Some(Self::Running),
            "done" => Some(Self::Done),
            "done_no_answer" => Some(Self::DoneNoAnswer),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

/// What happened to the answer's shape: no schema was asked for, the schema's JSON came back, or
/// the whole answer was forwarded unchanged because it could not be turned into that JSON.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Shape {
    None,
    Json,
    Forwarded,
}

impl Shape {
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Json => "json",
            Self::Forwarded => "forwarded",
        }
    }

    /// The shape of a finished answer: `schema` is whether a schema was asked for at all,
    /// `produced` whether the JSON it asked for was made.
    #[must_use]
    pub(crate) const fn of(schema: bool, produced: bool) -> Self {
        match (schema, produced) {
            (false, _) => Self::None,
            (true, true) => Self::Json,
            (true, false) => Self::Forwarded,
        }
    }

    /// The shape a stored string names, or `None` for a value this build does not know.
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "none" => Some(Self::None),
            "json" => Some(Self::Json),
            "forwarded" => Some(Self::Forwarded),
            _ => None,
        }
    }
}

/// A command-line handover's acknowledgement: its own words, plus the id it printed — the shape an
/// analysis dispatch already uses, so a script has one thing to hold and one thing to ask about.
/// An agent's own `send` and the one-way `guest` address open no record, so they print no id.
#[must_use]
pub(crate) fn acknowledged(answer: String, id: Option<&str>) -> String {
    match id {
        Some(id) => format!("{answer} (job {id})"),
        None => answer,
    }
}

/// One record as a lookup reads it.
pub(crate) struct Record {
    pub state: State,
    pub answer: Option<String>,
    pub reason: Option<String>,
    pub shape: Option<Shape>,
    /// The chain's length when the handover started one, `None` for a single turn.
    pub chain_steps: Option<usize>,
}

/// Open the record a command-line handover just accepted, before its work leaves. A record
/// already under this id is REPLACED: a chain's id is its content, so a re-issued identical chain
/// is a new run of the same chain, and the record reports the most recent one.
pub(crate) async fn begin(id: &str, chain_steps: Option<usize>) {
    let chain_steps = chain_steps.and_then(|steps| i64::try_from(steps).ok());
    let now = db::now();
    let result = crate::session::store()
        .conn
        .upsert_row(
            "UPDATE send_outcomes SET state = 'running', answer = NULL, reason = NULL, \
             shape = NULL, finished_at = NULL, chain_steps = ?2, created_at = ?3 WHERE id = ?1",
            || params![id.to_string(), chain_steps, now.clone()],
            "INSERT INTO send_outcomes (id, state, chain_steps, created_at) \
             VALUES (?1, 'running', ?2, ?3) ON CONFLICT (id) DO NOTHING",
            params![id.to_string(), chain_steps, now.clone()],
        )
        .await;
    if let Err(e) = result {
        // The handover must not fail for its record: the acknowledgement still returns the id,
        // and the status reports `unknown`.
        warn!(id, error = %e, "Failed to open a command-line send's outcome record");
    }
}

/// Complete a record with the answer its turn produced — empty for `done_no_answer`. Only a
/// `running` record moves, so a later turn under the same id cannot overwrite a report.
pub(crate) async fn finish(id: &str, answer: &str, shape: Shape) {
    // Nothing came back, so nothing was shaped: a `done_no_answer` record says no more than that,
    // and the shape describes an answer rather than the schema that was asked for.
    let (answer, state, shape) = if answer.trim().is_empty() {
        (None, State::DoneNoAnswer, None)
    } else {
        (Some(answer.to_string()), State::Done, Some(shape.as_str()))
    };
    let result = crate::session::store()
        .conn
        .execute(
            "UPDATE send_outcomes SET state = ?2, answer = ?3, shape = ?4, finished_at = ?5 \
             WHERE id = ?1 AND state = 'running'",
            params![id.to_string(), state.as_str(), answer, shape, db::now()],
        )
        .await;
    if let Err(e) = result {
        warn!(id, error = %e, "Failed to complete a command-line send's outcome record");
    }
}

/// End a record without an answer: the reason is what the caller is told.
pub(crate) async fn fail(id: &str, reason: &str) {
    let result = crate::session::store()
        .conn
        .execute(
            "UPDATE send_outcomes SET state = 'failed', reason = ?2, finished_at = ?3 \
             WHERE id = ?1 AND state = 'running'",
            params![id.to_string(), reason.to_string(), db::now()],
        )
        .await;
    if let Err(e) = result {
        warn!(id, error = %e, "Failed to end a command-line send's outcome record");
    }
}

/// A handover the router refused to run has no answer coming, so its record must not read as
/// running for a day. `None` — a job that was never a command-line handover — records nothing.
pub(crate) async fn fail_dropped(pending_job_id: Option<&str>, reason: &str) {
    if let Some(id) = pending_job_id {
        fail(id, reason).await;
    }
}

/// The ids among `ids` that still have a `running` record — the one read a turn or a durable round
/// makes before it settles, so nothing writes to the shared core writer for an id that has no
/// record. `None` when the read failed: the caller settles every id and each guarded write decides.
pub(crate) async fn running(ids: &[&str]) -> Option<HashSet<String>> {
    if ids.is_empty() {
        return Some(HashSet::new());
    }
    let bound: Vec<db::Value> = ids.iter().map(|id| db::Value::from(*id)).collect();
    let rows = crate::session::store()
        .conn
        .query(
            &format!(
                "SELECT id FROM send_outcomes WHERE state = 'running' AND id IN ({})",
                db::sql_in_placeholders(ids.len())
            ),
            bound,
        )
        .await;
    match rows {
        Ok(rows) => Some(
            rows.into_iter()
                .filter_map(|row| row.get::<String>(0).ok())
                .collect(),
        ),
        Err(e) => {
            warn!(error = %e, "Failed to read the running command-line send outcomes of a turn");
            None
        }
    }
}

/// One record, or `None` when no record stands under that id.
pub(crate) async fn lookup(id: &str) -> anyhow::Result<Option<Record>> {
    let row = crate::session::store()
        .conn
        .query_optional(
            "SELECT state, answer, reason, shape, chain_steps FROM send_outcomes WHERE id = ?1",
            params![id.to_string()],
            |row| {
                Ok::<_, anyhow::Error>((
                    row.get::<String>(0)?,
                    row.get::<Option<String>>(1)?,
                    row.get::<Option<String>>(2)?,
                    row.get::<Option<String>>(3)?,
                    row.get::<Option<i64>>(4)?,
                ))
            },
        )
        .await
        .context("read a command-line send's outcome record")?;
    let Some((state, answer, reason, shape, chain_steps)) = row else {
        return Ok(None);
    };
    let Some(state) = State::parse(&state) else {
        // A record this build cannot read is a record it cannot report: the caller answers
        // `unknown`, which is honest, rather than inventing a state.
        warn!(id, state = %state, "send_outcomes.state is not one this build knows");
        return Ok(None);
    };
    Ok(Some(Record {
        state,
        answer,
        reason,
        shape: shape.as_deref().and_then(Shape::parse),
        chain_steps: chain_steps.and_then(|steps| usize::try_from(steps).ok()),
    }))
}

/// Whether a single-turn handover's own work is still there. `None` when the read failed, which
/// proves nothing and leaves the record alone.
async fn work_stands(id: &str) -> Option<bool> {
    match crate::session::store()
        .conn
        .query_optional(
            "SELECT 1 FROM pending_jobs WHERE id = ?1 \
             UNION ALL SELECT 1 FROM jobs WHERE id = ?1 LIMIT 1",
            params![id.to_string()],
            |_| Ok::<(), anyhow::Error>(()),
        )
        .await
    {
        Ok(found) => Some(found.is_some()),
        Err(e) => {
            warn!(id, error = %e, "Failed to read a handover's work while settling interrupted records");
            None
        }
    }
}

/// Fail every record a previous process left `running` whose work is not in flight: nothing may
/// read as running forever. Runs BEFORE the boot replay, so a drained handover — whose durable
/// rows still stand — keeps its record `running` and is completed by the replay.
pub async fn fail_interrupted() {
    let rows = match crate::session::store()
        .conn
        .query(
            "SELECT id, chain_steps FROM send_outcomes WHERE state = 'running'",
            params![],
        )
        .await
    {
        Ok(rows) => rows,
        Err(e) => {
            warn!(error = %e, "Failed to read interrupted command-line send outcome records");
            return;
        }
    };
    for row in rows {
        let Ok(id) = row.get::<String>(0) else {
            continue;
        };
        let chain_steps = row
            .get::<Option<i64>>(1)
            .ok()
            .flatten()
            .and_then(|steps| usize::try_from(steps).ok());
        let in_flight = match chain_steps {
            // A read failure proves nothing, so the record is left `running`.
            Some(steps) => crate::jobs::chain_has_records(&id, 1..=steps).await != Some(false),
            None => work_stands(&id).await != Some(false),
        };
        if !in_flight {
            fail(&id, STOPPED_MID_FLIGHT).await;
        }
    }
}

/// Delete the records past their retention: finished before `cutoff`, or created before it and
/// never finished. `cutoff` is an RFC 3339 timestamp, the shape `db::now()` writes.
pub async fn purge_expired(cutoff: String) -> anyhow::Result<u64> {
    let conn = &crate::session::store().conn;
    let deadline = std::time::Instant::now() + PURGE_BUDGET;
    let mut total = 0u64;
    loop {
        // Turso 0.8.3 cannot parse `DELETE … LIMIT`, so the bound is an id subquery.
        let deleted = conn
            .execute(
                "DELETE FROM send_outcomes WHERE id IN (SELECT id FROM send_outcomes WHERE \
                 (finished_at IS NOT NULL AND finished_at < ?1) OR \
                 (finished_at IS NULL AND created_at < ?1) LIMIT ?2)",
                params![cutoff.as_str(), PURGE_BATCH],
            )
            .await
            .context("delete expired command-line send outcome records")?;
        total += deleted;
        if deleted == 0 || std::time::Instant::now() >= deadline {
            break;
        }
        // A drain or a stop ends the sweep between batches rather than through the task's
        // cancellation: the next one starts where this left off.
        if !crate::shutdown::sleep_or_shutdown_or_drain(PURGE_PAUSE).await {
            break;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handover the router refuses to run is failed, and a job that was never a command-line
    /// handover records nothing.
    #[tokio::test]
    async fn fail_dropped_settles_only_a_record_that_exists() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("dropped_handover", None).await;
        fail_dropped(Some("dropped_handover"), "the custom role no longer exists").await;
        fail_dropped(None, "the custom role no longer exists").await;

        let record = lookup("dropped_handover")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::Failed);
        assert_eq!(
            record.reason.as_deref(),
            Some("the custom role no longer exists")
        );
    }

    /// An acknowledgement names the id a script holds, and leaves the words alone without one.
    #[test]
    fn acknowledged_names_the_id_it_printed() {
        assert_eq!(
            acknowledged("Message handed over".to_string(), Some("job_1")),
            "Message handed over (job job_1)"
        );
        assert_eq!(
            acknowledged("Message handed over".to_string(), None),
            "Message handed over",
            "a handover that opened no record prints no id"
        );
    }

    /// `begin` opens a `running` record and `lookup` reads it back.
    #[tokio::test]
    async fn begin_opens_a_running_record() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("record_running", Some(3)).await;
        let record = lookup("record_running")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::Running);
        assert_eq!(record.answer, None);
        assert_eq!(record.reason, None);
        assert_eq!(record.shape, None);
        assert_eq!(record.chain_steps, Some(3));
    }

    /// A finished record carries the answer and the shape of the answer.
    #[tokio::test]
    async fn finish_completes_a_record_with_its_answer() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("record_done", None).await;
        finish("record_done", "the answer", Shape::Json).await;
        let record = lookup("record_done")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::Done);
        assert_eq!(record.answer.as_deref(), Some("the answer"));
        assert_eq!(record.shape, Some(Shape::Json));
        assert_eq!(record.chain_steps, None);
        let finished = crate::session::store()
            .conn
            .query_optional(
                "SELECT finished_at FROM send_outcomes WHERE id = ?1",
                params!["record_done".to_string()],
                |row| row.get::<Option<String>>(0),
            )
            .await
            .expect("read finished_at")
            .flatten();
        assert!(finished.is_some(), "a completed record is stamped");
    }

    /// An empty answer is `done_no_answer`, not a failure and not an empty answer.
    #[tokio::test]
    async fn finish_with_an_empty_answer_is_done_no_answer() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("record_empty", None).await;
        finish("record_empty", "  \n", Shape::Json).await;
        let record = lookup("record_empty")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::DoneNoAnswer);
        assert_eq!(record.answer, None);
        assert_eq!(
            record.shape, None,
            "nothing came back, so nothing was shaped — not even the schema that was asked for"
        );
    }

    /// `fail` ends a record without an answer and carries the reason.
    #[tokio::test]
    async fn fail_ends_a_record_with_its_reason() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("record_failed", None).await;
        fail("record_failed", "stopped from the dashboard").await;
        let record = lookup("record_failed")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::Failed);
        assert_eq!(record.reason.as_deref(), Some("stopped from the dashboard"));
        assert_eq!(record.answer, None);
    }

    /// Completion only moves a record out of `running`: a later `finish` after a `fail` changes
    /// nothing, so a report already made cannot be overwritten.
    #[tokio::test]
    async fn a_later_finish_cannot_overwrite_a_finished_record() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("record_guarded", None).await;
        fail("record_guarded", "the addressed turn failed").await;
        finish("record_guarded", "a later answer", Shape::Json).await;
        let record = lookup("record_guarded")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::Failed);
        assert_eq!(record.reason.as_deref(), Some("the addressed turn failed"));
        assert_eq!(record.answer, None, "the failed report is the final one");
    }

    /// `begin` under an id that already finished replaces it: a re-issued identical chain is a
    /// new run, and the record reports the most recent one.
    #[tokio::test]
    async fn begin_replaces_a_finished_record() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("record_replaced", Some(2)).await;
        finish("record_replaced", "the first answer", Shape::Json).await;
        begin("record_replaced", Some(2)).await;
        let record = lookup("record_replaced")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::Running);
        assert_eq!(record.answer, None, "the old answer is gone");
        assert_eq!(record.reason, None);
        assert_eq!(record.shape, None);
        assert_eq!(record.chain_steps, Some(2));
    }

    /// The sweep removes a finished record past the cutoff, and a running record only once its
    /// own creation is past the cutoff too.
    #[tokio::test]
    #[serial_test::serial(send_outcome)]
    async fn purge_expired_removes_only_what_is_past_its_retention() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        // The store is shared across the suite: clear what other tests left so the counts below
        // are this test's own.
        crate::session::store()
            .conn
            .execute("DELETE FROM send_outcomes", params![])
            .await
            .expect("clear the outcome records");

        begin("purge_done", None).await;
        finish("purge_done", "an answer", Shape::Json).await;
        begin("purge_running", None).await;
        // Backdate only the finished record's finish: the running record was created now, so a
        // cutoff between the two removes the finished one and keeps the running one.
        let old = "2020-01-01T00:00:00+00:00";
        crate::session::store()
            .conn
            .execute(
                "UPDATE send_outcomes SET finished_at = ?1 WHERE id = 'purge_done'",
                params![old],
            )
            .await
            .expect("backdate the finished record");

        let between = "2021-01-01T00:00:00+00:00".to_string();
        assert_eq!(
            purge_expired(between).await.expect("purge once"),
            1,
            "only the finished record is past its retention"
        );
        assert!(
            lookup("purge_done").await.expect("read").is_none(),
            "the finished record is gone"
        );
        assert!(
            lookup("purge_running").await.expect("read").is_some(),
            "a running record is kept until its own creation is past the cutoff"
        );

        let future = "2999-01-01T00:00:00+00:00".to_string();
        assert_eq!(
            purge_expired(future).await.expect("purge again"),
            1,
            "the running record is past the cutoff too"
        );
        assert!(
            lookup("purge_running").await.expect("read").is_none(),
            "the running record is gone once its creation is old enough"
        );
    }

    /// A record a previous process left `running` whose work is gone is failed, so nothing reads as
    /// running forever.
    #[tokio::test]
    #[serial_test::serial(send_outcome)]
    async fn fail_interrupted_fails_a_record_whose_work_is_gone() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("interrupted_gone", None).await;
        fail_interrupted().await;
        let record = lookup("interrupted_gone")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::Failed);
        assert_eq!(record.reason.as_deref(), Some(STOPPED_MID_FLIGHT));
        assert_eq!(record.answer, None);
    }

    /// A drained handover's durable rows still stand, so the boot replay completes its record: the
    /// sweep leaves it `running`.
    #[tokio::test]
    #[serial_test::serial(send_outcome)]
    async fn fail_interrupted_leaves_a_record_whose_work_still_stands() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("interrupted_live", None).await;
        crate::util::test::JobRowBuilder::new(
            &crate::session::store().conn,
            "interrupted_live",
            "worker",
            "worker",
            "interrupted_ws",
        )
        .timestamps(db::now())
        .insert()
        .await
        .expect("the handover's own round stands");

        fail_interrupted().await;
        let record = lookup("interrupted_live")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::Running);
    }

    /// A chain's record is the chain's, so it is left alone while any of its steps still has a
    /// record: the step that is still running is what the boot replay resumes.
    #[tokio::test]
    #[serial_test::serial(send_outcome)]
    async fn fail_interrupted_leaves_a_chain_whose_step_still_stands() {
        let _guard = crate::util::test::STORE_LOCK.lock().await;
        crate::util::test::init_test_stores().await;

        begin("interrupted_chain", Some(2)).await;
        crate::util::test::JobRowBuilder::new(
            &crate::session::store().conn,
            crate::launch::record_id("interrupted_chain", 2),
            "worker",
            "worker",
            "interrupted_chain_ws",
        )
        .timestamps(db::now())
        .insert()
        .await
        .expect("the chain's second step stands");

        fail_interrupted().await;
        let record = lookup("interrupted_chain")
            .await
            .expect("read the record")
            .expect("the record stands");
        assert_eq!(record.state, State::Running);
    }
}
