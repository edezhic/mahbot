//! Pre-shrink consistency gate for reclaiming (TRUNCATE) WAL checkpoints.
//!
//! The engine derives that checkpoint's new main-file size from the page-1
//! header (`database_size × page_size` — exactly the `PRAGMA page_count` asked
//! for here) and truncates in place with no validation, so a store that answers
//! 0 pages while its main file is non-empty or its WAL holds frames would be
//! truncated to nothing. The engine answers 0 both for a header that declares no
//! pages and for a header read it could not complete, so the record names both
//! readings. Before any reclaiming checkpoint the product therefore compares that
//! answer with the file sizes it already stats; an inconsistent or unusable
//! answer — a panicking probe included — means it is NOT issued and the refusal
//! is recorded loudly.
//!
//! This is a consistency check, NOT store classification: what counts as a
//! damaged store stays defined in [`crate::db::wal_guard`] alone. A refusal is
//! likewise not a failure — it never enters the checkpoint failure/stop
//! machinery, never marks the store unusable, is never counted as a failed
//! attempt, and never triggers recovery.
//!
//! Nothing here opens a store file (wal_guard's lock rule): the answer is the
//! store's own `PRAGMA page_count` plus [`std::fs::metadata`] stats. On a healthy
//! store that is one pragma and two stats and changes nothing. Accepted
//! boundaries: a refused round also leaves the WAL unreclaimed, because the
//! engine couples the main-file shrink to the reclaiming mode (the service keeps
//! serving and the round is reported); a write landing between the answer and the
//! checkpoint, and a declared size that is individually plausible but stale, stay
//! uncovered; and the engine can also truncate in place on an explicit connection
//! close, which the product never issues.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::Path;

use futures_util::future::FutureExt;
use tracing::{error, warn};

use crate::db::failure_record::{self, FailureKind, FailureReport, RoundCounter};
use crate::db::wal_guard::WAL_HEADER_BYTES;
use crate::db::{Connection, Value};

/// One answer the gate asked for: the value, or why it could not be obtained.
/// That text is what the durable record keeps.
type Answer<T> = Result<T, String>;

/// The store's own answer, and the file sizes, that a reclaiming checkpoint
/// would shrink from.
#[derive(Debug)]
struct ShrinkRefusal {
    /// The store's own answer; `None` when it could not be asked.
    page_count: Option<u64>,
    /// Main-file size in bytes; `None` when it could not be obtained.
    main_bytes: Option<u64>,
    /// `-wal` size in bytes; `None` when it could not be obtained.
    wal_bytes: Option<u64>,
    /// Why the answer is not usable.
    cause: ShrinkRefusalCause,
}

#[derive(Debug)]
enum ShrinkRefusalCause {
    /// The store answered zero pages while its main file is non-empty or its WAL
    /// holds frames: the shrink would truncate it to nothing.
    ZeroPagesWithData,
    /// The store's own page count (or a file size it is compared against)
    /// could not be obtained: the answer is not usable, so no shrink may run.
    /// The payload names the failed probe.
    UnusableAnswer(String),
}

impl ShrinkRefusal {
    /// A refusal from a probe that could not be trusted at all (it panicked):
    /// nothing was measured, so the detail is the only thing the record carries.
    fn unusable(detail: String) -> Self {
        Self {
            page_count: None,
            main_bytes: None,
            wal_bytes: None,
            cause: ShrinkRefusalCause::UnusableAnswer(detail),
        }
    }

    /// Whether the refusal is an external condition rather than a finding about
    /// the store's contents: a probe that could not answer because of a
    /// resource/permission/lock condition, not because the store is damaged. An
    /// unusable answer with no such signal stays unmarked — the record must not
    /// rule damage out. A zero answer stays unmarked too: it may be a failed
    /// header read (see the module docs), not evidence that the store is sound.
    fn is_environmental(&self) -> bool {
        match &self.cause {
            ShrinkRefusalCause::ZeroPagesWithData => false,
            ShrinkRefusalCause::UnusableAnswer(detail) => {
                crate::db::text_is_actionable_signal(detail)
            }
        }
    }

    /// One line for the durable record and the log. The measurements live in the
    /// record's own field lines, so this is the decision, not a second copy of
    /// the numbers — and it stands alone in `error.log`, which is why a zero
    /// answer names both of its readings there.
    fn reason(&self) -> String {
        match &self.cause {
            ShrinkRefusalCause::ZeroPagesWithData => "the store answers 0 pages while its files \
                 hold data — a page-1 header declaring no pages, or a header read the engine \
                 could not complete (it answers 0 for both); a reclaiming checkpoint would shrink \
                 the store to nothing, so none was issued"
                .to_string(),
            ShrinkRefusalCause::UnusableAnswer(detail) => format!(
                "the store's own page count or a file size it is compared against could not be \
                 obtained ({detail}) — a reclaiming checkpoint is not issued on an answer that \
                 cannot be trusted"
            ),
        }
    }
}

/// The gate's decision over the facts it collected. `None` = a reclaiming
/// checkpoint may be issued. The only place a [`ShrinkRefusal`] is built from
/// answers.
fn verdict(
    page_count: Answer<u64>,
    main_bytes: Answer<u64>,
    wal_bytes: Answer<u64>,
) -> Option<ShrinkRefusal> {
    // An answer this gate cannot trust is itself a reason not to shrink: the
    // check cannot vouch for the store, so the reclaiming checkpoint is
    // withheld rather than risked. The real probe text is what the record keeps.
    let errors: Vec<&str> = [&page_count, &main_bytes, &wal_bytes]
        .into_iter()
        .filter_map(|answer| answer.as_ref().err().map(String::as_str))
        .collect();
    let cause = if errors.is_empty() {
        let holds_data = matches!(main_bytes, Ok(bytes) if bytes > 0)
            || matches!(wal_bytes, Ok(bytes) if bytes > WAL_HEADER_BYTES);
        if page_count == Ok(0) && holds_data {
            ShrinkRefusalCause::ZeroPagesWithData
        } else {
            return None;
        }
    } else {
        ShrinkRefusalCause::UnusableAnswer(errors.join("; "))
    };
    Some(ShrinkRefusal {
        page_count: page_count.ok(),
        main_bytes: main_bytes.ok(),
        wal_bytes: wal_bytes.ok(),
        cause,
    })
}

/// The size of `path` in bytes: absent is 0 (a store without a WAL has nothing
/// to reclaim), any other stat failure is an unusable answer.
fn stat_bytes(path: &Path, what: &str) -> Answer<u64> {
    crate::db::wal_guard::stat_size(path).map_err(|e| format!("{what} stat failed: {e}"))
}

/// Ask the store for its own page count and compare it with the file sizes the
/// product already stats. `Some(_)` = a reclaiming checkpoint must not be
/// issued. Reads no store bytes (wal_guard's lock rule).
async fn probe(conn: &Connection) -> Option<ShrinkRefusal> {
    let page_count = match conn.query("PRAGMA page_count;", ()).await {
        Ok(rows) => match rows.first().map(|row| row.get_value(0)) {
            Some(value) => page_count_value(value),
            None => Err("the page count query returned no row".to_string()),
        },
        Err(e) => Err(format!("the page count query failed: {e}")),
    };
    let main_bytes = stat_bytes(conn.db_path(), "main file");
    let wal_bytes = stat_bytes(&crate::db::wal_path(conn.db_path()), "wal");
    verdict(page_count, main_bytes, wal_bytes)
}

/// The store's answer as the gate must read it: only a non-negative integer is
/// usable, so a negative or non-integer value and a failed read each come back
/// as the reason the answer cannot be trusted.
fn page_count_value(value: Result<Value, ::turso::Error>) -> Answer<u64> {
    match value {
        Ok(Value::Integer(n)) => {
            u64::try_from(n).map_err(|_| format!("the page count is negative ({n})"))
        }
        Ok(other) => Err(format!(
            "the page count is not a non-negative integer ({other:?})"
        )),
        Err(e) => Err(format!("the page count read failed: {e}")),
    }
}

/// Per-refusing-file count of rounds whose reclaiming checkpoint was refused,
/// for the process's lifetime.
static REFUSAL_ROUNDS: RoundCounter = RoundCounter::new();

/// Record a refused shrink loudly: the first refusal of a store file in this
/// process files a durable block (the record survives where the ordinary log may
/// already be gone), every further round only warns with the running count. Filed
/// once per condition ([`failure_record::record`]).
///
/// `store` names the logical store; `db_path` is the main file actually checked
/// (the caller passes [`Connection::db_path`]) and keys the count, so a probe of
/// another file — the boot rebuild's temp copy — never consumes this one's
/// first-refusal slot. `root` is the storage root the block is filed under
/// (`None` when unresolvable, in which case the block goes to stderr).
fn record_refusal(
    store: &'static str,
    db_path: &Path,
    refusal: &ShrinkRefusal,
    root: Option<&Path>,
) {
    let further_rounds = REFUSAL_ROUNDS.prior_rounds(&db_path.to_string_lossy());
    if further_rounds > 0 {
        warn!(
            store,
            reason = %refusal.reason(),
            further_rounds,
            "Reclaiming checkpoint refused again — already recorded on its first refusal",
        );
        return;
    }
    error!(
        store,
        reason = %refusal.reason(),
        "Reclaiming checkpoint refused — the store's own page count does not match its files; no shrink was issued",
    );
    let report = FailureReport::new(FailureKind::ShrinkRefused)
        .store(store)
        .db_path(db_path.to_path_buf())
        .reason(refusal.reason())
        .environment(refusal.is_environmental())
        .extra(match refusal.page_count {
            Some(count) => format!("page count: {count}"),
            None => "page count: unavailable".to_string(),
        })
        .extra(match refusal.main_bytes {
            Some(bytes) => format!("main file: {bytes} bytes"),
            None => "main file: unavailable".to_string(),
        })
        .extra(failure_record::artifact_state_line(refusal.wal_bytes));
    failure_record::record_and_point(root, "reclaiming checkpoint refused", &report);
}

/// Probe and, on a refusal, record it loudly; `true` = a reclaiming checkpoint
/// may be issued. `root` is the storage root the refusing store's durable block
/// is filed under (the caller passes the root it already resolved).
pub(crate) async fn shrink_allowed(
    conn: &Connection,
    store: &'static str,
    root: Option<&Path>,
) -> bool {
    shrink_allowed_inner(probe(conn), conn, store, root).await
}

/// The gate over a probe's answer: `true` = a reclaiming checkpoint may be
/// issued; a refusal is recorded loudly first. A probe that panics is an answer
/// that cannot be trusted and is refused like any other — the panic never
/// unwinds into the caller or becomes a checkpoint failure.
///
/// `probe` is a parameter, and this module's only seam, because a panicking
/// probe is not otherwise producible: [`shrink_allowed`] is the single caller
/// and always passes the real probe.
async fn shrink_allowed_inner<F>(
    probe: F,
    conn: &Connection,
    store: &'static str,
    root: Option<&Path>,
) -> bool
where
    F: Future<Output = Option<ShrinkRefusal>>,
{
    let refusal = match AssertUnwindSafe(probe).catch_unwind().await {
        Ok(refusal) => refusal,
        Err(payload) => Some(ShrinkRefusal::unusable(format!(
            "the pre-shrink probe panicked: {}",
            crate::util::panic_message(&*payload)
        ))),
    };
    let Some(refusal) = refusal else {
        return true;
    };
    record_refusal(store, conn.db_path(), &refusal, root);
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gate's pure decision over exhaustive (page count, main size, WAL
    /// size) combinations: it refuses exactly the states a reclaiming checkpoint
    /// would destroy, or that it cannot trust.
    #[test]
    #[expect(clippy::too_many_lines)] // one table-driven case per behavior
    fn verdict_refuses_only_states_a_shrink_would_destroy_or_cannot_trust() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Expected {
            Allowed,
            ZeroPagesWithData,
            UnusableAnswer,
        }
        struct Case {
            name: &'static str,
            page_count: Answer<u64>,
            main_bytes: Answer<u64>,
            wal_bytes: Answer<u64>,
            expected: Expected,
            reason_contains: Option<&'static str>,
        }
        let cases = [
            Case {
                name: "healthy store, header-only WAL",
                page_count: Ok(5),
                main_bytes: Ok(20_480),
                wal_bytes: Ok(WAL_HEADER_BYTES),
                expected: Expected::Allowed,
                reason_contains: None,
            },
            Case {
                name: "healthy store, no WAL",
                page_count: Ok(5),
                main_bytes: Ok(20_480),
                wal_bytes: Ok(0),
                expected: Expected::Allowed,
                reason_contains: None,
            },
            Case {
                name: "empty store",
                page_count: Ok(0),
                main_bytes: Ok(0),
                wal_bytes: Ok(0),
                expected: Expected::Allowed,
                reason_contains: None,
            },
            Case {
                name: "empty store, WAL at exactly its header",
                page_count: Ok(0),
                main_bytes: Ok(0),
                wal_bytes: Ok(WAL_HEADER_BYTES),
                expected: Expected::Allowed,
                reason_contains: None,
            },
            Case {
                name: "zero pages, non-empty main file",
                page_count: Ok(0),
                main_bytes: Ok(159_744),
                wal_bytes: Ok(0),
                expected: Expected::ZeroPagesWithData,
                reason_contains: Some("0 pages"),
            },
            Case {
                name: "zero pages, frames only in the WAL",
                page_count: Ok(0),
                main_bytes: Ok(0),
                wal_bytes: Ok(4096),
                expected: Expected::ZeroPagesWithData,
                reason_contains: Some("0 pages"),
            },
            Case {
                name: "zero pages, data in both files",
                page_count: Ok(0),
                main_bytes: Ok(100),
                wal_bytes: Ok(4096),
                expected: Expected::ZeroPagesWithData,
                reason_contains: None,
            },
            Case {
                name: "page count probe failed",
                page_count: Err("the page count query failed: injected".to_string()),
                main_bytes: Ok(1),
                wal_bytes: Ok(0),
                expected: Expected::UnusableAnswer,
                reason_contains: Some("the page count query failed: injected"),
            },
            Case {
                name: "main file stat failed",
                page_count: Ok(3),
                main_bytes: Err("main file stat failed: injected".to_string()),
                wal_bytes: Ok(0),
                expected: Expected::UnusableAnswer,
                reason_contains: Some("main file stat failed: injected"),
            },
            Case {
                name: "WAL stat failed",
                page_count: Ok(3),
                main_bytes: Ok(1),
                wal_bytes: Err("wal stat failed: injected".to_string()),
                expected: Expected::UnusableAnswer,
                reason_contains: Some("wal stat failed: injected"),
            },
            Case {
                name: "every answer unusable",
                page_count: Err("the page count query failed: injected".to_string()),
                main_bytes: Err("main file stat failed: injected".to_string()),
                wal_bytes: Err("wal stat failed: injected".to_string()),
                expected: Expected::UnusableAnswer,
                reason_contains: Some(
                    "the page count query failed: injected; main file stat failed: injected; \
                     wal stat failed: injected",
                ),
            },
        ];
        for case in cases {
            let result = verdict(case.page_count, case.main_bytes, case.wal_bytes);
            let (actual, reason) = match &result {
                None => (Expected::Allowed, String::new()),
                Some(refusal) => (
                    match refusal.cause {
                        ShrinkRefusalCause::ZeroPagesWithData => Expected::ZeroPagesWithData,
                        ShrinkRefusalCause::UnusableAnswer(_) => Expected::UnusableAnswer,
                    },
                    refusal.reason(),
                ),
            };
            assert_eq!(actual, case.expected, "case {}: {result:?}", case.name);
            if let Some(needle) = case.reason_contains {
                assert!(
                    reason.contains(needle),
                    "case {}: the reason {reason:?} must contain {needle:?}",
                    case.name,
                );
            }
        }
    }

    /// The store's own answer is read strictly: only a non-negative integer is
    /// usable, and every other shape — a negative value, a non-integer value, a
    /// failed read — is a reason to withhold the shrink, carrying the text the
    /// durable record keeps.
    #[test]
    fn an_unusable_page_count_answer_is_refused_rather_than_read() {
        assert_eq!(page_count_value(Ok(Value::Integer(7))), Ok(7));
        assert_eq!(page_count_value(Ok(Value::Integer(0))), Ok(0));
        let negative = page_count_value(Ok(Value::Integer(-1))).expect_err("negative is unusable");
        assert!(negative.contains("negative"), "{negative:?}");
        let text = page_count_value(Ok(Value::Text("three".to_string())))
            .expect_err("a non-integer answer is unusable");
        assert!(text.contains("not a non-negative integer"), "{text:?}");
        let read = page_count_value(Err(::turso::Error::Error(
            "injected read failure".to_string(),
        )))
        .expect_err("a failed read is unusable");
        assert!(read.contains("the page count read failed"), "{read:?}");
    }

    /// The first refusal of a store file in the process files one durable block
    /// with every measured field, in the shape the other per-store blocks use;
    /// later refusals only count. The block must not read as a checkpoint failure
    /// — a refusal is not one.
    #[test]
    fn refusal_is_recorded_once_and_is_not_a_checkpoint_failure() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let name = "shrink_gate_refusal_probe";
        let db_path = tmp.path().join("probe.db");
        let refusal = verdict(Ok(0), Ok(159_744), Ok(0)).expect("the case must be refused");
        for _ in 0..3 {
            record_refusal(name, &db_path, &refusal, Some(tmp.path()));
        }
        let body = std::fs::read_to_string(tmp.path().join("error.log")).expect("error.log");
        assert_eq!(
            body.matches("MahBot store shrink refused").count(),
            1,
            "only the first refusal may file a block: {body}"
        );
        for needle in [
            "store: shrink_gate_refusal_probe",
            "db path:",
            "reason:",
            "page count: 0",
            "main file: 159744 bytes",
            "artifact state: wal_size=0",
        ] {
            assert!(
                body.contains(needle),
                "the refusal block must contain {needle:?}: {body}"
            );
        }
        assert!(
            !body.contains(failure_record::ENVIRONMENT_CAUSE),
            "a store finding must not be marked environment-caused: {body}"
        );
        for forbidden in [
            "MahBot checkpoint failure",
            "MahBot exit checkpoint failure",
        ] {
            assert!(
                !body.contains(forbidden),
                "a refusal must not read as {forbidden:?}: {body}"
            );
        }
    }

    /// A probe that could not answer because of an external condition (here:
    /// lock contention) is marked in the record as environment-caused, so an
    /// environmental refusal is not read as evidence about the store.
    #[test]
    fn an_environmental_refusal_is_marked_as_such() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let refusal = verdict(
            Ok(3),
            Ok(20_480),
            Err("wal stat failed: database is locked".to_string()),
        )
        .expect("the case must be refused");
        record_refusal(
            "shrink_gate_environment_probe",
            &tmp.path().join("probe.db"),
            &refusal,
            Some(tmp.path()),
        );
        let body = std::fs::read_to_string(tmp.path().join("error.log")).expect("error.log");
        assert!(
            body.contains(failure_record::ENVIRONMENT_CAUSE),
            "a lock-contention refusal must be marked environment-caused: {body}"
        );
    }

    /// A probe that panics is an answer that cannot be trusted: the gate refuses
    /// the shrink, records the panic as the refusal's reason, and never unwinds
    /// into the caller — a panicking probe is not a checkpoint failure.
    #[tokio::test]
    async fn a_panicking_probe_is_refused_and_recorded() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let db_path = tmp.path().join("probe.db");
        let conn =
            crate::db::open_with_schema(&db_path, "CREATE TABLE probe (id INTEGER PRIMARY KEY);")
                .await
                .expect("create the probe store");

        let allowed = shrink_allowed_inner(
            async { panic!("injected probe panic") },
            &conn,
            "shrink_gate_panic_probe",
            Some(tmp.path()),
        )
        .await;

        assert!(!allowed, "a panicking probe must refuse the shrink");
        let body = std::fs::read_to_string(tmp.path().join("error.log")).expect("error.log");
        assert!(
            body.contains("MahBot store shrink refused"),
            "the refusal must be filed: {body}"
        );
        assert!(
            body.contains("injected probe panic"),
            "the refusal must carry the panic as its reason: {body}"
        );
        assert!(
            !body.contains("MahBot checkpoint failure"),
            "a panicking probe must not read as a checkpoint failure: {body}"
        );
    }
}
