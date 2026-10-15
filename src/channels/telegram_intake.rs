//! The Telegram intake's durable record.
//!
//! The intake's own guarantee is that an event the platform hands it is not lost: an
//! incoming update is written here before the intake acknowledges it or lets it go, and
//! only an event this record holds is handed on. Telegram confirms a prefix of the update
//! stream on each poll, so an event whose record cannot be written stays unconfirmed and is
//! offered again — this record, not a stream position, is what keeps an event from being
//! taken up twice.
//!
//! A row holds the incoming update itself — its identity and content, never an
//! attachment's bytes, which are fetched again from the platform while it still has them —
//! and drops the payload once the event is resolved, keeping only the decision, so an
//! update the platform offers again after a restart is recognised as already dealt with.
//! The record is keyed by the event's own `update_id` and the bot account it belongs to
//! (the account's own identity, so rotating the token for the same account still recovers).
//!
//! What a run has in hand it claims under its own identity: a hold or a take-over writes
//! the claiming run onto the row, and a run only takes up events it is not itself already
//! handling (see [`Intake::take_over`]). Every listener is such a run, so a listener
//! rebuilt inside one process — a token change included — takes over what the listener it
//! replaced left held. What it must not take over is what the replaced one is still handing
//! on, which is why the replaced listener's frame is awaited first (see `stop_listeners`):
//! a frame whose collector has drained is settled or left held, while a process that ended
//! without draining is a crash, the record's own case.
//!
//! An event the intake cannot decide about right now — the read its decision needs failed —
//! is left held exactly like one a crash left behind, payload and all: settling is for a
//! decision that was actually made, so an undecidable event is taken up by a later run
//! rather than refused and dropped for good.

use anyhow::Result;
use serde_json::Value;
use std::sync::Arc;

use crate::db;

crate::define_store! {
    /// Global Telegram intake store.
    pub(crate) static TELEGRAM_INTAKE: TelegramIntakeStore,
    expect = "TELEGRAM_INTAKE not initialized — call init_all_stores() first",
}

/// How long a row is kept, in either state.
///
/// Long enough that an update the platform still holds unconfirmed (roughly a
/// day) is recognised as done when it comes back, far short of the silence after
/// which Telegram rests `update_id` (roughly a week), so a legitimate later
/// event is never taken for a repeat. An unresolved row is kept because its
/// payload is what a later run takes the event up from, but not forever: the
/// record must not grow without bound.
///
/// That bound is the one way an accepted event still goes unhandled: a row left
/// held past it — one no run could decide about, or one held through a wind-down —
/// is swept with the rest, so the event is dropped if no listener is rebuilt inside
/// the window.
pub(crate) const RETENTION_HOURS: i64 = 48;

/// The `telegram_intake.state` value of an event that is held and not settled.
pub(crate) const IN_PROGRESS: &str = "in_progress";

/// The `telegram_intake.state` value of an event that has been settled — handed
/// on, or deliberately not answered.
const DONE: &str = "done";

/// How long a released burst is kept, in days: a complaint about a split burst
/// arrives days after the fact (see [`BurstRecord`]).
const BURST_RETENTION_DAYS: i64 = 30;

/// One released burst as the durable record keeps it: who sent it, where, how
/// many parts it held, which rule released it, and when each part was taken in.
///
/// This is the evidence a complaint about a split burst is read from — the log
/// line a multi-part release writes survives for hours, this for days. A released
/// group is always exactly one request — the parts travel in the one envelope the
/// collector hands over — so the record carries the parts and not a turn count.
pub(crate) struct BurstRecord {
    pub(crate) user_name: String,
    pub(crate) reply_target: String,
    pub(crate) parts: usize,
    /// [`crate::channels::telegram_group`]'s `Release::as_str`: the window that
    /// applied, or the cap/control/stop that cut the group short of one.
    pub(crate) released_by: &'static str,
    pub(crate) released_at: String,
    /// One entry per part, oldest first: the platform's own time for the update
    /// (`sent_at`, null when the update carried none) and the moment the
    /// collector took the part in (`taken_at`), both RFC 3339 — the two facts
    /// that decide whether a burst split, and where.
    pub(crate) part_times: String,
}

impl TelegramIntakeStore {
    /// Drop rows older than the retention bound. Whichever timestamp a row has
    /// decides its age: a resolved row ages from its decision, a row still held
    /// from its arrival. A row no run ever settled is dropped rather than kept
    /// for good, so the record does not grow without bound.
    ///
    /// This is the record's process-wide retention sweep, the one operation not
    /// scoped to a listener's account — which is why it needs none.
    ///
    /// The burst record is swept with the events, each on its own bound: a
    /// released burst is what the events it stood for resolved to, so the two
    /// belong to the same sweep. Returns the total dropped across both.
    pub(crate) async fn age_out(&self) -> Result<u64> {
        let cutoff = (chrono::Utc::now() - chrono::Duration::hours(RETENTION_HOURS)).to_rfc3339();
        let events = self
            .conn
            .execute(
                "DELETE FROM telegram_intake WHERE COALESCE(decided_at, created_at) < ?1",
                db::params![cutoff],
            )
            .await?;
        let bursts = self
            .conn
            .execute(
                "DELETE FROM telegram_bursts WHERE released_at < ?1",
                db::params![
                    (chrono::Utc::now() - chrono::Duration::days(BURST_RETENTION_DAYS))
                        .to_rfc3339()
                ],
            )
            .await?;
        Ok(events + bursts)
    }

    /// Record one released burst of `account_id` (see [`BurstRecord`]).
    pub(crate) async fn record_burst(&self, account_id: &str, record: &BurstRecord) -> Result<()> {
        #[expect(clippy::cast_possible_wrap)] // a released group holds at most `MAX_PARTS` parts
        let parts = record.parts as i64;
        self.conn
            .execute(
                "INSERT INTO telegram_bursts \
                 (account_id, user_name, reply_target, parts, released_by, released_at, part_times) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                db::params![
                    account_id,
                    record.user_name.as_str(),
                    record.reply_target.as_str(),
                    parts,
                    record.released_by,
                    record.released_at.as_str(),
                    record.part_times.as_str()
                ],
            )
            .await?;
        Ok(())
    }
}

/// The intake's durable record as one listener sees it: where the record lives,
/// the run its rows are claimed under, and the account whose events are its own.
#[derive(Clone)]
pub(crate) struct Intake {
    store: TelegramIntakeStore,
    claim: Arc<str>,
    pub(crate) account_id: Arc<str>,
}

impl Intake {
    pub(crate) fn new(store: TelegramIntakeStore, claim: Arc<str>, account_id: &str) -> Self {
        Self {
            store,
            claim,
            account_id: Arc::from(account_id),
        }
    }

    /// Take the durable hold on one unit of a polled batch, claiming it for the run
    /// that has it in hand, and report the events this call recorded.
    ///
    /// A unit is everything that must not be recorded in parts — an album's members
    /// together, every other update on its own — and it is written in ONE statement,
    /// so it is recorded whole or not at all: a member whose write fails leaves its
    /// album unrecorded instead of handing the album's other parts on as unrelated
    /// messages. The payload is what makes an event takeable up again by a later run,
    /// so the write happening at all is the hold.
    ///
    /// That one statement is the only thing that can refuse the unit. An update the
    /// platform never identified reaches the store as a null `update_id`, which the
    /// column's own `NOT NULL` rejects — the whole statement fails, so no part is
    /// ever recorded without its unit.
    ///
    /// An event the record already holds is left out of the answer rather than
    /// reported as an error: the platform re-offers a batch until the position moves
    /// past it, so a collision is the ordinary case.
    pub(crate) async fn hold(&self, updates: &[&Value]) -> Result<Vec<i64>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        // One value group per update: `account_id`, `update_id`, `state`, `payload`,
        // `created_at` and `claimed_by`, with the literal NULL standing for
        // `decided_at` — only a settled event has one.
        let mut values = Vec::with_capacity(updates.len() * 6);
        for update in updates {
            values.push(db::Value::from(self.account_id.as_ref()));
            values.push(db::Value::from(update["update_id"].as_i64()));
            values.push(db::Value::from(IN_PROGRESS));
            values.push(db::Value::from(serde_json::to_string(update)?));
            values.push(db::Value::from(db::now()));
            values.push(db::Value::from(self.claim.as_ref()));
        }
        let rows = self
            .store
            .conn
            .execute_rows(
                &format!(
                    "INSERT INTO telegram_intake \
                     (account_id, update_id, state, payload, created_at, decided_at, claimed_by) \
                     VALUES {} ON CONFLICT(account_id, update_id) DO NOTHING \
                     RETURNING update_id",
                    vec!["(?, ?, ?, ?, ?, NULL, ?)"; updates.len()].join(", ")
                ),
                values,
            )
            .await?;
        rows.into_iter()
            .map(|row| row.get::<i64>(0).map_err(anyhow::Error::from))
            .collect()
    }

    /// Settle these events — handed on, or deliberately not answered — and log a
    /// decision that cannot be written. The hand-over cannot be atomic with this
    /// write, so a failure here is decided the same way a crash in that instant is:
    /// the event stays held and a later run takes it up again, which risks a second
    /// handling but never a lost event — the side to err on.
    pub(crate) async fn settle(&self, ids: &[i64]) {
        if ids.is_empty() {
            return;
        }
        if let Err(err) = self.resolve(ids).await {
            tracing::warn!(?ids, %err, "Telegram intake: could not settle an event in the record");
        }
    }

    /// Write the durable record of a released burst, and log one that cannot be
    /// written. Fail-open: nothing downstream reads it, so a failure must not cost
    /// the message its hand-over.
    pub(crate) async fn record_burst(&self, record: &BurstRecord) {
        if let Err(err) = self.store.record_burst(&self.account_id, record).await {
            tracing::warn!(%err, "Telegram burst: could not write the release record");
        }
    }

    /// Write the decision for these events in one statement, so a unit settles whole
    /// or not at all: half-settled members of an album would read as a duplicate on a
    /// later start. The payload is dropped because only the decision needs keeping.
    async fn resolve(&self, ids: &[i64]) -> Result<()> {
        let mut values = Vec::with_capacity(ids.len() + 3);
        values.push(db::Value::from(DONE));
        values.push(db::Value::from(db::now()));
        values.push(db::Value::from(self.account_id.as_ref()));
        values.extend(ids.iter().copied().map(db::Value::from));
        self.store
            .conn
            .execute(
                &format!(
                    "UPDATE telegram_intake SET state = ?, payload = NULL, decided_at = ? \
                     WHERE account_id = ? AND update_id IN ({})",
                    db::sql_in_placeholders(ids.len())
                ),
                values,
            )
            .await?;
        Ok(())
    }

    /// Take up the events this account holds and this run is not already handling:
    /// everything a previous run left behind, claimed and read in ONE statement, so
    /// two runs can never take up the same event.
    ///
    /// A row whose payload is missing, unparseable, or carries no integer
    /// `update_id` cannot be taken up at all: it is warned about and skipped. Nothing
    /// this intake writes can be in that state — the store refuses an update it
    /// cannot identify and `resolve` drops a payload only together with the decision
    /// — so the guard is for a row another revision wrote, and it is what keeps a
    /// broken payload out of the parser. The retention sweep removes such a row.
    pub(crate) async fn take_over(&self) -> Result<Vec<Value>> {
        let rows = self
            .store
            .conn
            .execute_rows(
                "UPDATE telegram_intake SET claimed_by = ?1 \
                 WHERE account_id = ?2 AND state = ?3 \
                   AND (claimed_by IS NULL OR claimed_by <> ?1) \
                 RETURNING update_id, payload",
                db::params![self.claim.as_ref(), self.account_id.as_ref(), IN_PROGRESS],
            )
            .await?;
        let mut events = Vec::with_capacity(rows.len());
        for row in rows {
            let id = row.get::<i64>(0)?;
            let takeable = row
                .get::<Option<String>>(1)?
                .and_then(|payload| serde_json::from_str::<Value>(&payload).ok())
                .filter(|update| update["update_id"].as_i64().is_some());
            if let Some(update) = takeable {
                events.push(update);
            } else {
                tracing::warn!(
                    account_id = %self.account_id,
                    update_id = id,
                    "telegram_intake: a held event's payload cannot be taken up"
                );
            }
        }
        // The rows come back in no guaranteed order; the platform orders by
        // `update_id`, and an album's members must regroup in the user's send
        // order.
        events.sort_by_key(|update: &Value| update["update_id"].as_i64());
        Ok(events)
    }
}

/// A claim for a listener of its own: what one listener records and takes over is
/// claimed under this, so a listener rebuilt inside the same process is a different
/// run from the one it replaced and can take over what that one left held.
pub(crate) fn listener_claim() -> String {
    format!("run-{}", crate::generate_id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::params;
    use serde_json::json;

    async fn test_store() -> (TelegramIntakeStore, tempfile::TempDir) {
        crate::open_test_store!(TelegramIntakeStore, "telegram_intake")
    }

    /// A run's own handle onto `store`, claiming under `claim` and keyed by `account`.
    fn intake(store: &TelegramIntakeStore, claim: &str, account: &str) -> Intake {
        Intake::new(store.clone(), Arc::from(claim), account)
    }

    /// A held event is visible through `take_over()`; resolving it leaves `take_over()`.
    #[tokio::test]
    async fn hold_then_resolve_moves_event_out_of_take_over() {
        let (store, _tmp) = test_store().await;
        let update = json!({ "update_id": 7, "message": { "text": "hi" } });

        intake(&store, "run-a", "1")
            .hold(&[&update])
            .await
            .expect("hold");
        assert_eq!(
            intake(&store, "run-b", "1")
                .take_over()
                .await
                .expect("take over"),
            vec![update]
        );

        intake(&store, "run-a", "1").settle(&[7]).await;
        assert!(
            intake(&store, "run-c", "1")
                .take_over()
                .await
                .expect("take over")
                .is_empty()
        );
    }

    /// A claim is scoped to the run that took the event up: the run that holds
    /// it does not take it up again, while another run does — in the platform's
    /// own `update_id` order, whatever order the rows come back in.
    #[tokio::test]
    async fn take_over_claims_only_what_this_run_does_not_hold() {
        let (store, _tmp) = test_store().await;
        let run_a = intake(&store, "run-a", "1");
        run_a
            .hold(&[&json!({ "update_id": 2 })])
            .await
            .expect("hold 2");
        run_a
            .hold(&[&json!({ "update_id": 1 })])
            .await
            .expect("hold 1");

        assert!(run_a.take_over().await.expect("own take-over").is_empty());
        let ids: Vec<i64> = intake(&store, "run-b", "1")
            .take_over()
            .await
            .expect("another run")
            .iter()
            .filter_map(|update| update["update_id"].as_i64())
            .collect();
        assert_eq!(ids, vec![1, 2]);
        // The take-over is also where the claim is written: run-b, which just
        // claimed both rows, takes up nothing further, while run-a — whose claim
        // run-b has replaced — finds them takeable again, exactly as it would a
        // dead run's.
        assert!(
            intake(&store, "run-b", "1")
                .take_over()
                .await
                .expect("run-b again")
                .is_empty()
        );
        let reclaimed: Vec<i64> = run_a
            .take_over()
            .await
            .expect("run-a again")
            .iter()
            .filter_map(|update| update["update_id"].as_i64())
            .collect();
        assert_eq!(reclaimed, vec![1, 2]);
    }

    /// The same `update_id` under two accounts are separate events.
    #[tokio::test]
    async fn same_update_id_is_distinct_per_account() {
        let (store, _tmp) = test_store().await;
        intake(&store, "run-a", "1")
            .hold(&[&json!({ "update_id": 3 })])
            .await
            .expect("hold account 1");
        intake(&store, "run-a", "2")
            .hold(&[&json!({ "update_id": 3 })])
            .await
            .expect("hold account 2");

        assert_eq!(
            intake(&store, "run-b", "1")
                .take_over()
                .await
                .expect("take over 1")
                .len(),
            1
        );
        assert_eq!(
            intake(&store, "run-b", "2")
                .take_over()
                .await
                .expect("take over 2")
                .len(),
            1
        );
    }

    /// A second hold for the same event reports that the record already holds it
    /// rather than failing — the platform's re-offers make the collision ordinary —
    /// while two messages alike in everything but the id the platform gave them are
    /// two events: the record's key is that id, not the content.
    #[tokio::test]
    async fn the_record_tells_events_apart_by_the_id_the_platform_gave_them() {
        let (store, _tmp) = test_store().await;
        let first = json!({ "update_id": 9, "message": { "from": { "id": 5 }, "text": "hi" } });
        let second = json!({ "update_id": 10, "message": { "from": { "id": 5 }, "text": "hi" } });
        assert_eq!(
            intake(&store, "run-a", "1")
                .hold(&[&first, &second])
                .await
                .expect("first hold"),
            vec![9, 10]
        );
        assert!(
            intake(&store, "run-b", "1")
                .hold(&[&first])
                .await
                .expect("second hold")
                .is_empty(),
            "a second hold must report that the record already holds the event"
        );
        // Both are events of the record's own, so a later run takes up both.
        let recovered: Vec<i64> = intake(&store, "run-b", "1")
            .take_over()
            .await
            .expect("take over")
            .iter()
            .filter_map(|update| update["update_id"].as_i64())
            .collect();
        assert_eq!(recovered, vec![9, 10]);
    }

    /// One unit is recorded in one statement, so a member that cannot be written
    /// leaves its unit unrecorded rather than recorded in parts.
    #[tokio::test]
    async fn a_unit_is_recorded_whole_or_not_at_all() {
        let (store, _tmp) = test_store().await;
        let run_a = intake(&store, "run-a", "1");
        let first = json!({ "update_id": 4, "message": { "media_group_id": "g" } });
        let second = json!({ "update_id": 5, "message": { "media_group_id": "g" } });
        // An update with no identity of its own reaches the store as a null
        // `update_id`, which the column's own `NOT NULL` refuses inside the unit's
        // one statement — so neither member is recorded and the platform's re-offer
        // can record the album in one piece.
        let unidentified = json!({ "message": { "media_group_id": "g" } });
        run_a
            .hold(&[&first, &unidentified])
            .await
            .expect_err("an update without an id cannot be recorded");
        // A writer that recorded the unit's members one at a time would leave
        // `first` in the record, where another run finds it.
        assert!(
            intake(&store, "run-b", "1")
                .take_over()
                .await
                .expect("take over")
                .is_empty()
        );

        assert_eq!(
            run_a.hold(&[&first, &second]).await.expect("hold both"),
            vec![4, 5]
        );
    }

    /// The bound covers a row in either state: a resolved row ages from its
    /// decision, a held row from its arrival, so a row no run settled cannot
    /// outlive the record.
    #[tokio::test]
    async fn age_out_drops_rows_past_the_retention_bound() {
        let (store, _tmp) = test_store().await;
        let run_a = intake(&store, "run-a", "1");
        for id in [1, 2, 3] {
            run_a
                .hold(&[&json!({ "update_id": id })])
                .await
                .expect("hold");
        }
        run_a.settle(&[1]).await;

        let old = (chrono::Utc::now() - chrono::Duration::hours(RETENTION_HOURS + 1)).to_rfc3339();
        store
            .conn
            .execute(
                "UPDATE telegram_intake SET decided_at = ?1 WHERE update_id = 1",
                params![old.clone()],
            )
            .await
            .expect("age the resolved row");
        store
            .conn
            .execute(
                "UPDATE telegram_intake SET created_at = ?1 WHERE update_id = 3",
                params![old],
            )
            .await
            .expect("age the held row");

        assert_eq!(store.age_out().await.expect("age_out"), 2);
        // Only the freshly held row survives.
        let held: Vec<i64> = intake(&store, "run-b", "1")
            .take_over()
            .await
            .expect("take over")
            .iter()
            .filter_map(|update| update["update_id"].as_i64())
            .collect();
        assert_eq!(held, vec![2]);
    }

    /// The burst record is swept on its own bound, in days rather than hours: a
    /// release stays readable long after the events it stood for are gone, and one
    /// past the bound goes with them.
    #[tokio::test]
    async fn age_out_drops_bursts_past_their_own_bound() {
        let (store, _tmp) = test_store().await;
        let run_a = intake(&store, "run-a", "1");
        let record = |released_at: String| BurstRecord {
            user_name: "alice".to_string(),
            reply_target: "999".to_string(),
            parts: 2,
            released_by: "extended_window",
            released_at,
            part_times: "[]".to_string(),
        };
        let now = chrono::Utc::now();
        let fresh = now.to_rfc3339();
        run_a
            .record_burst(&record(
                (now - chrono::Duration::days(BURST_RETENTION_DAYS + 1)).to_rfc3339(),
            ))
            .await;
        run_a.record_burst(&record(fresh.clone())).await;

        assert_eq!(store.age_out().await.expect("age_out"), 1);
        let rows = store
            .conn
            .query("SELECT released_at FROM telegram_bursts", ())
            .await
            .expect("read the burst record");
        let kept: Vec<String> = rows
            .into_iter()
            .map(|row| row.get::<String>(0).expect("released_at"))
            .collect();
        assert_eq!(kept, vec![fresh]);
    }
}
