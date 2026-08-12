//! Deterministic reproduction of the turso multiprocess-WAL macOS lock-drop
//! → Exclusive-repair → **stale durable frame index** defect (class B, with
//! class-A write-path panics as the secondary signature). Escalation artifact
//! for the turso maintainers (version pin `=0.7.2`, deliberate; see
//! "Version pin / upstream status").
//!
//! **Experimental-feature disclaimer**: this exercises the experimental
//! `multiprocess_wal` engine feature (opt-in via
//! `experimental_multiprocess_wal(true)`), which the production daemon enables
//! for cross-process WAL. The defect lives in that opt-in path.
//!
//! # Defect mechanism (established by research, turso_core 0.7.2)
//!
//! On macOS the `.tshm` byte locks are process-scoped POSIX `F_SETLK` locks
//! (on Linux they are OFD locks), so closing ANY fd of the `.tshm` — a
//! read-only open+read+drop, e.g. a background wal-guard probe — releases
//! every lock the process holds on that file (see the sibling
//! `lock_drop_repro` bench for the class-C mechanism and the fcntl layout:
//! byte 0 lifetime, byte 1 writer, byte 2 checkpoint, bytes 3+ reader slots).
//! turso never re-verifies its local bookkeeping, so the owner keeps writing
//! while its locks are actually gone.
//!
//! A second turso process (a read-only mirror, a debug CLI, a restarted
//! daemon) that opens the store inside that window probes byte 0, classifies
//! the open **Exclusive**, and runs `repair_or_reseed_authority_from_local_disk_scan`
//! (`storage/wal.rs`): it clears the transient owner state and, depending on
//! the reconciliation verdict, keeps or rebuilds the **durable frame index**
//! that lives in the `.tshm` (append-only `{page_id, frame_id}` entries, 4096
//! per block, published in the shared mmap). When the rebuild's local WAL
//! scan races an active writer (frames still in the page cache, not yet in
//! the `-wal` file), the rebuilt index silently loses entries — and the
//! writer's subsequent commits append to the stale base. The stale index then
//! drives the checkpoint backfill path, which bakes wrong/older index btree
//! pages into the main DB file.
//!
//! Two symptom classes, both detected by this bench:
//!
//! - **A (write-path panic)** — `record_frame` asserts
//!   `shared WAL frame ids must increase monotonically: new_frame_id=…,
//!   previous_frame_id=…, slot=…, shared_max_frame=…` (`storage/
//!   shared_wal_coordination.rs`) or the high-water-mark asserts
//!   (`connection WAL position must not be behind the committed high-water
//!   mark` in `storage/wal.rs`) when the first append collides with a stale
//!   slot. Prod signature: ~1000+ index entries against `max_frame` 10-17.
//! - **B (durable desync)** — the main DB file ends up with a table btree
//!   and a SQL index btree that disagree (`row N missing from index` /
//!   `wrong # of entries in index` from `PRAGMA integrity_check`, or a
//!   verified count mismatch), persisting across reopen and TRUNCATE.
//!
//! The `.tshm` index-length vs actual WAL frame count divergence is the
//! direct WAL-level signature and is captured before any reopen can heal it.
//!
//! # Empirical calibration (research runs on turso 0.7.2 / macOS)
//!
//! | Run | Topology | Result |
//! |-----|----------|--------|
//! | r7_1 | writer (bulk inserts) + read-only mirror, in-process `.tshm` open+close churn (~12432 opens/30s), 57501 rows | B: table 57501 vs index 57222, `row N missing from index` |
//! | r8_1 | identical but no fd churn | clean (67501 \| 67501) — causality proof |
//! | fast-mirror | ~200-430 disk-scan opens/20s | **6/11** reproduced |
//! | slow-mirror | rare reopens | 0/3 |
//! | owner-only | only the owner churns, mirror never reopens | 0/3 |
//! | r3 family | long open transaction (writer spills uncommitted) | byte-identical double A-panics, 72-110s, gap exactly 4096 (block capacity) |
//! | ss2 | durable B | survives reopen and TRUNCATE (baked into the main DB file) |
//!
//! The bench encodes the fast-mirror cadence (writer churn ~400/s, mirror
//! reopen ~15/s, run 30s, ~1900 rows/s) as documented defaults. Reproduction
//! is **frequency-dependent and machine-speed dependent**: hit rates are
//! reported honestly as x/N and the statistical protocol below exists so a
//! single clean run can never be misreported as "fix present".
//!
//! # How to run
//!
//! Build via the non-default feature gate (excluded from default
//! builds/tests/benches):
//!
//! ```text
//! cargo bench --no-default-features --features wal-stale-index-repro \
//!   --bench wal_stale_index_repro -- --run
//! ```
//!
//! Modes:
//!
//! - `--run` — primary batch repro (macOS): writer + read-only mirror,
//!   N iterations (default 11, the calibration baseline), early stop on the
//!   first reproduction. Writer churns its own `.tshm` fds (simulating the
//!   daemon's wal-guard probes) while the mirror reopens the store at the
//!   fast cadence; each iteration ends with a pre-open `.tshm` inspection, a
//!   fresh-reopen integrity/count check, and — on a detected mismatch — a
//!   TRUNCATE-persistence phase: the parent releases the actors and spawns a
//!   truncate child that checkpoints and re-detects, so the checkpoint is not
//!   busy and the ss2 durability proof can actually complete.
//! - `--control` — control group (macOS): identical run with the churn loop
//!   disabled. **Must be clean** — any signal is a control-group violation.
//! - `--observer writing` — mirror also writes between reopens (the
//!   "writing observer" topology that gives the class-A write-path panic a
//!   chance to fire; read-only is the default). The control group's causality
//!   claim holds for the read-only observer only — `--control --observer
//!   writing` is the two-writer topology (the sibling wal_race_repro defect).
//! - `--selfcheck` — verifies the detection oracles (count access paths,
//!   `INDEXED BY` honored, healthy integrity_check) on a disposable DB;
//!   runs on any Unix platform (the lock probes and signal handling are
//!   POSIX).
//!
//! Flags (`--run` / `--control`): `--iterations N`, `--duration S` (secs),
//! `--churn-hz H`, `--mirror-hz H`, `--rate R` (rows/sec), `--seed S`,
//! `--observer read-only|writing`.
//!
//! # Exit contract
//!
//! - `0` — bug reproduced: a class-A write-path panic (stderr signature) or
//!   class-B durable desync — integrity_check problems / verified count
//!   mismatch on the fresh reopen, OR a store too corrupt to read during
//!   detection (read-side pager symptom: a fresh open or an oracle query
//!   fails with `Invalid page type` / `short read` / corrupt reads) — with
//!   honest x/N reporting.
//! - `1` — clean: a full `--run` batch (≥11 trials) with zero mechanism
//!   signals, or a `--control` group that stayed clean (exit-code basis for
//!   "fix present").
//! - `2` — harness error: bad arguments, failed open/insert/probe, oracle
//!   self-check failure, or a **control-group violation** (printed loudly).
//! - `3` — skip: `--run`/`--control` on a non-macOS platform (the F_SETLK
//!   close-drop is macOS-specific; a Linux clean run would prove nothing).
//! - `4` — inconclusive: the batch finished without A/B damage but the
//!   mechanism still fired (stale-index evidence or lock-drop panics), or
//!   fewer than 11 clean trials ran — "fix present" cannot be distinguished
//!   from bad luck (with the 6/11 calibration, P(0/11 | broken) ≈ 0.02%).
//! - `5` / `6` — internal detect/truncate-child codes (evidence-only and
//!   proof-skipped; parent-visible only, never a batch exit).
//!
//! Bare invocation (no arguments) prints this usage and exits `2` — nothing
//! runs silently.
//!
//! # Statistical protocol (probabilistic reproduction)
//!
//! Reproduction is frequency-dependent (6/11 fast-mirror baseline), so a
//! single trial cannot distinguish "fix present" from "bad luck". The bench
//! therefore runs a batch (default 11 trials = the calibration denominator),
//! stops early at the first reproduction, and classifies every trial:
//!
//! - `reproduced-B`: integrity/count mismatch on the fresh reopen, or the
//!   store is corrupt during detection (read-side pager panic / failed
//!   oracle) — the durability phase reports whether it survives TRUNCATE.
//! - `reproduced-A`: monotonicity / high-water-mark panic in any actor's
//!   stderr.
//! - `evidence`: no durable damage, but the pre-open `.tshm` inspection
//!   shows the durable index diverged from the actual WAL — the mechanism
//!   fired without damage (reported loudly, never a silent clean).
//! - `panic-void`: a class-C reader-slot panic aborted the trial — the
//!   lock-drop fired but the damage is not class A/B (reported distinctly;
//!   class C is the sibling `lock_drop_repro` defect). Read-side panics in
//!   the writer/mirror actors are likewise reported distinctly, and the
//!   detection phase still runs on the leftover state.
//! - `clean`: no mechanism signal at all.
//!
//! Each trial uses a fresh disposable temp dir (full cleanup), and a
//! deterministic per-trial jitter seeded by `--seed` decorrelates the actor
//! cadences across trials.
//!
//! # Version pin / upstream status
//!
//! The repo pins `turso = "=0.7.2"` deliberately so the storage layer cannot
//! drift under this artifact (same pin as the wal_race_repro / lock_drop_repro
//! escalations). The class-B stale-index mechanism is unfixed upstream; this
//! artifact is the first confirmed report of the mechanism.
//!
//! # False-clean traps addressed
//!
//! - Auto-checkpoint (>1000 committed frames) can fold the desync into the
//!   main DB file: the high-volume default workload (rate ~1900 rows/s)
//!   keeps auto-checkpoints firing during the run, which is exactly what
//!   produces the durable (reopen-surviving) form; the TRUNCATE phase then
//!   verifies it explicitly.
//! - A fresh reopen can heal a WAL-level desync: the pre-open `.tshm`
//!   inspection captures the index-vs-WAL divergence before any reopen, and
//!   the writer holds its connection (stopped) until the inspection is done,
//!   so no close-time checkpoint can truncate the evidence first.
//! - `nbackfills > 0` forces RebuildFromDisk (healing): reported in the
//!   inspection output; trials that heal produce `evidence` rather than a
//!   silent clean.
//! - `COUNT(*)` silently falling back to a table scan would mask index
//!   desyncs: the bench uses three verified access paths (`SCAN t`,
//!   `SCAN t USING INDEX idx_t_val`, `SEARCH t USING INTEGER PRIMARY KEY`),
//!   confirmed by `--selfcheck` via `EXPLAIN QUERY PLAN`, plus
//!   `PRAGMA integrity_check` — both-oracle detection.
//!
//! # Standalone
//!
//! Self-contained: only `turso` (`=0.7.2`), `tempfile`, `libc` (unix) and
//! `std` are used, so a copy of this file in a fresh project (`harness =
//! false`, deps `turso = "=0.7.2"`, `tempfile`, `libc`) builds and runs
//! unchanged. Each run uses a fresh disposable database under a temp dir;
//! live stores are never touched and concurrent runs cannot interfere.
use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, Write};
use std::os::fd::AsRawFd;
use std::os::unix::process::ExitStatusExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use turso::core::{Database, DatabaseOpts, OpenFlags, PlatformIO};

// ── Symptom signatures (turso_core 0.7.2, verbatim) ───────────────────────

/// Write-path monotonicity assert (`storage/shared_wal_coordination.rs`,
/// `record_frame`): the stale durable index's slot chain disagrees with the
/// next append. Class A.
const A_SIG_MONOTONIC: &str = "shared WAL frame ids must increase monotonically";
/// Write-path WAL-position asserts (`storage/wal.rs`). Class A.
const A_SIG_HIGH_WATER: &str = "must not be behind the committed high-water mark";
/// Reader-slot assert (`unregister_reader`); the sibling class-C defect.
const C_SIG_READER_SLOT: &str = "reader slot released by non-owner";
/// Durable-B integrity_check messages (`translate/integrity_check.rs`).
const B_SIG_WRONG_ENTRIES: &str = "wrong # of entries in index";
const B_SIG_ROW_MISSING: &str = "missing from index";
/// Read-side pager panic heuristic: any panic naming storage internals that
/// is not A or C. Actor read-side panics are reported distinctly (the review
/// protocol: not conflated with class B) and the detection phase decides the
/// trial; a store too corrupt to read during the detect child's own oracles
/// IS class B (the read-side pager symptom).
const READ_SIDE_HINTS: [&str; 9] = [
    "pager",
    "wal frame",
    "btree",
    "corrupt",
    "checksum",
    "page should be",
    "invalid page type",
    "short read",
    "i/o error",
];

// ── Workload shape (mirrors the r7_1 recipe) ──────────────────────────────
// Padded rows force one WAL frame per row (4 KiB page, ~4 KiB payload), so a
// 64-row commit appends 64 frames — the auto-checkpoint threshold (>1000
// committed frames) is crossed within a second and checkpoint churn keeps the
// stale index baking wrong pages into the main DB during the run.
const PAD_BYTES: usize = 4000;
const BATCH_ROWS: usize = 64;
/// Per-reopen inserts of the writing observer (keeps its local WAL scan
/// "matching" the authority so the stale index survives — the class-A path).
const MIRROR_INSERT_ROWS: usize = 8;
const WAL_HEADER_BYTES: u64 = 32;
const WAL_FRAME_BYTES: u64 = 24 + 4096;

// `.tshm` map-header offsets (turso_core 0.7.2 `SharedWalCoordinationMapHeader`,
// `#[repr(C)]`, the on-file mmap layout; `frame_index_len` u32, `snapshot_seq`
// u64, `max_frame` u64, `nbackfills` u64).
const TSHM_FRAME_INDEX_LEN_OFFSET: u64 = 40;
const TSHM_SNAPSHOT_SEQ_OFFSET: u64 = 48;
const TSHM_MAX_FRAME_OFFSET: u64 = 56;
const TSHM_NBACKFILLS_OFFSET: u64 = 64;

// ── Statistical protocol defaults (calibration baseline) ──────────────────
const DEFAULT_ITERATIONS: usize = 11; // 6/11 fast-mirror calibration denominator
const CONTROL_ITERATIONS: usize = 3;
const DEFAULT_DURATION_SECS: u64 = 30; // r7_1 ran 30s
const DEFAULT_CHURN_HZ: u64 = 400; // r7_1 ~12432 opens/30s ≈ 414/s
const DEFAULT_MIRROR_HZ: u64 = 15; // fast-mirror ~200-430 opens/20s
const DEFAULT_ROWS_PER_SEC: u64 = 1900; // r7_1 57501 rows / 30s
const DEFAULT_SEED: u64 = 0x5EED;
/// The mirror stops reopening this long before the writer stops inserting, so
/// the last (possibly stale) rebuild is followed by fresh appends that keep
/// the divergence visible in the `.tshm` — a late correct rebuild must not
/// erase the evidence before the inspection.
const MIRROR_LEAD_SECS: u64 = 1;
/// Index-vs-WAL divergence tolerance. The healthy invariant is
/// `index_len == wal_frames` (every WAL frame is one index entry); the stop
/// race can leave ≤ one 64-frame batch un-indexed.
const EVIDENCE_TOLERANCE_ABS: u64 = 128;
/// Cadence cap: the jitter divisor `period_ns / 5` must stay >= 1, i.e.
/// `hz <= 200_000_000` (1e9 / 5).
const MAX_CADENCE_HZ: u64 = 200_000_000;

const MARKER_TIMEOUT: Duration = Duration::from_secs(120);
const CHILD_GRACE: Duration = Duration::from_secs(10);
const DETECT_TIMEOUT: Duration = Duration::from_secs(180);
/// Slack on top of the actor duration when waiting for STOPPED markers
/// (covers spawn latency, marker delivery, and cadence jitter).
const STOP_DEADLINE_SLACK_SECS: u64 = 15;

// Exit codes.
const EXIT_REPRODUCED: i32 = 0;
const EXIT_CLEAN: i32 = 1;
const EXIT_HARNESS: i32 = 2;
const EXIT_SKIP: i32 = 3;
const EXIT_BUDGET: i32 = 4;
/// Detect-child: mechanism fired (stale-index evidence) but no A/B damage.
const EXIT_EVIDENCE: i32 = 5;
/// Truncate-child only: the durability proof could not complete. Not a batch
/// exit code — the parent ignores it (the reproduction is already confirmed).
const EXIT_PROOF_SKIPPED: i32 = 6;
/// Child panic-class exit codes (parent-side classification is primary; the
/// codes make an early actor death unambiguous).
const CHILD_EXIT_A: i32 = 21;
const CHILD_EXIT_C: i32 = 22;
const CHILD_EXIT_READSIDE: i32 = 23;

// ── Oracle names / SQL ────────────────────────────────────────────────────

const ORACLE_TABLE: &str = "SELECT count(*) FROM t";
const ORACLE_INDEX: &str = "SELECT count(*) FROM t INDEXED BY idx_t_val";
const ORACLE_PK: &str = "SELECT count(*) FROM t WHERE id >= 0";

// ── Shared DB helpers ─────────────────────────────────────────────────────

fn open_db_with_result(
    db_path: &Path,
    flags: OpenFlags,
) -> Result<Arc<turso::core::Connection>, String> {
    let io: Arc<dyn turso::core::IO> =
        Arc::new(PlatformIO::new().map_err(|e| format!("platform io: {e}"))?);
    let opts = DatabaseOpts::new()
        .with_multiprocess_wal(true)
        .with_index_method(true);
    let db = Database::open_file_with_flags(
        io,
        db_path
            .to_str()
            .ok_or_else(|| "db path must be UTF-8".to_string())?,
        flags,
        opts,
        None,
    )
    .map_err(|e| format!("open database: {e}"))?;
    let conn = db
        .connect()
        .map_err(|e| format!("connect to database: {e}"))?;
    conn.set_busy_timeout(Duration::from_secs(60));
    Ok(conn)
}

fn open_db_with(db_path: &Path, flags: OpenFlags) -> Arc<turso::core::Connection> {
    open_db_with_result(db_path, flags).expect("open database")
}

fn open_db(db_path: &Path) -> Arc<turso::core::Connection> {
    open_db_with(db_path, OpenFlags::default())
}

fn open_db_readonly(db_path: &Path) -> Arc<turso::core::Connection> {
    open_db_with(db_path, OpenFlags::ReadOnly)
}

/// Execute `sql`, retrying on `Busy` (write-lock contention is expected while
/// the peer actor holds the writer lock). Callers decide whether a non-Busy
/// failure is a harness error (`expect`) or a probe outcome (`match`).
fn execute_retry(conn: &Arc<turso::core::Connection>, sql: &str) -> Result<(), String> {
    for _ in 0..10_000 {
        match conn.execute(sql) {
            Ok(()) => return Ok(()),
            Err(e) if is_busy_error(&e) => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(e) => return Err(format!("statement failed: {e}")),
        }
    }
    Err("statement stayed busy".to_string())
}

fn is_busy_error(e: &turso::core::LimboError) -> bool {
    e.to_string().to_lowercase().contains("busy")
}

/// Insert `count` padded rows (~4 KiB each), forcing every row onto its own
/// page(s) and therefore its own WAL frame(s).
fn bulk_insert(conn: &Arc<turso::core::Connection>, count: usize) -> Result<(), String> {
    let pad = "x".repeat(PAD_BYTES);
    let mut remaining = count;
    let mut offset = 0usize;
    while remaining > 0 {
        let n = remaining.min(BATCH_ROWS);
        let mut sql = String::from("INSERT INTO t (val) VALUES ");
        for i in 0..n {
            if i > 0 {
                sql.push_str(", ");
            }
            sql.push_str(&format!("('bulk-{}-{pad}')", offset + i));
        }
        execute_retry(conn, &sql)?;
        remaining -= n;
        offset += n;
    }
    Ok(())
}

/// Single-row integer scalar (a count oracle).
fn query_count(conn: &Arc<turso::core::Connection>, sql: &str) -> Result<i64, String> {
    let mut stmt = conn
        .query(sql)
        .map_err(|e| format!("query failed: {e}"))?
        .ok_or_else(|| "query returned no statement".to_string())?;
    let rows = stmt
        .run_collect_rows()
        .map_err(|e| format!("query run failed: {e}"))?;
    match rows.first().and_then(|r| r.first()) {
        Some(turso::core::Value::Numeric(turso::core::Numeric::Integer(n))) => Ok(*n),
        other => Err(format!("expected integer count, got {other:?}")),
    }
}

/// integrity_check problems: every non-"ok" row (a healthy store reports one
/// `ok` row). `Err` when the check itself cannot complete.
fn collect_integrity_problems(conn: &Arc<turso::core::Connection>) -> Result<Vec<String>, String> {
    let mut stmt = conn
        .query("PRAGMA integrity_check")
        .map_err(|e| format!("integrity_check failed: {e}"))?
        .ok_or_else(|| "integrity_check returned no statement".to_string())?;
    let rows = stmt
        .run_collect_rows()
        .map_err(|e| format!("integrity_check run failed: {e}"))?;
    Ok(rows
        .iter()
        .filter_map(|r| match r.first() {
            Some(turso::core::Value::Text(t)) if t.value != "ok" => Some(t.value.to_string()),
            _ => None,
        })
        .collect())
}

/// The three verified count oracles: table scan, `INDEXED BY` index scan,
/// and PK scan. All three must agree — a planner silently falling back to a
/// table scan would make the index oracle useless (verified by `--selfcheck`).
fn count_oracles(conn: &Arc<turso::core::Connection>) -> Result<(i64, i64, i64), String> {
    Ok((
        query_count(conn, ORACLE_TABLE)?,
        query_count(conn, ORACLE_INDEX)?,
        query_count(conn, ORACLE_PK)?,
    ))
}

/// Run `PRAGMA wal_checkpoint(<mode>)`. `Ok(())` = completed with busy == 0
/// (a successful TRUNCATE resets the shared index and starts a fresh WAL
/// generation). `Err(cause)` = could not complete — busy exhaustion, query
/// failure, or a busy result — so the caller can attribute the skipped
/// durability proof correctly. On an already-confirmed mismatch a failed
/// checkpoint must never downgrade the reproduction to a harness error.
fn wal_checkpoint(conn: &Arc<turso::core::Connection>, mode: &str) -> Result<(), String> {
    let sql = format!("PRAGMA wal_checkpoint({mode})");
    for _ in 0..10_000 {
        let mut stmt = match conn.query(&sql) {
            Ok(Some(stmt)) => stmt,
            Ok(None) => return Err("returned no result".into()),
            Err(e) if is_busy_error(&e) => {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            Err(e) => return Err(format!("failed: {e}")),
        };
        let rows = match stmt.run_collect_rows() {
            Ok(rows) => rows,
            Err(e) if is_busy_error(&e) => {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            Err(e) => return Err(format!("failed: {e}")),
        };
        let busy = rows.first().and_then(|r| r.first()).is_some_and(|v| {
            matches!(
                v,
                turso::core::Value::Numeric(turso::core::Numeric::Integer(n)) if *n != 0
            )
        });
        return if busy {
            Err("stayed busy".into())
        } else {
            Ok(())
        };
    }
    Err("stayed busy".into())
}

// ── Panic hooks / cleanup ─────────────────────────────────────────────────

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Classify a panic message into the actor exit-code classes.
fn classify_panic(msg: &str) -> i32 {
    let lower = msg.to_lowercase();
    if lower.contains(&A_SIG_MONOTONIC.to_lowercase())
        || lower.contains(&A_SIG_HIGH_WATER.to_lowercase())
    {
        CHILD_EXIT_A
    } else if lower.contains(&C_SIG_READER_SLOT.to_lowercase()) {
        CHILD_EXIT_C
    } else if READ_SIDE_HINTS.iter().any(|h| lower.contains(h)) {
        CHILD_EXIT_READSIDE
    } else {
        EXIT_HARNESS
    }
}

/// Child roles (writer / mirror / detect): a panic in turso storage code is a
/// probe outcome, not a harness bug — print the text and exit with the
/// classified code so the parent can report it loudly.
fn install_actor_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = panic_message(info.payload());
        eprintln!("PANIC: {msg}");
        std::process::exit(classify_panic(&msg));
    }));
}

fn exit_after_cleanup(tmp: &tempfile::TempDir, code: i32) -> ! {
    let _ = fs::remove_dir_all(tmp.path());
    std::process::exit(code);
}

fn harness_no_tmp(msg: &str) -> ! {
    eprintln!("harness error: {msg}");
    std::process::exit(EXIT_HARNESS);
}

// ── fcntl byte-lock probe (diagnostic only; never the detection oracle) ───

/// Probe `.tshm` byte locks from this process. `F_SETLK` works from a second
/// process against F_SETLK holders; used to confirm the close-drop window.
struct ProbeFd {
    file: fs::File,
}

impl ProbeFd {
    fn open(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            file: fs::OpenOptions::new().read(true).write(true).open(path)?,
        })
    }

    /// Try an exclusive byte-range lock; `Ok(true)` = acquired (not held).
    fn probe(&self, offset: u64) -> std::io::Result<bool> {
        let mut flock = libc::flock {
            l_type: libc::F_WRLCK as libc::c_short,
            l_whence: libc::SEEK_SET as libc::c_short,
            l_start: offset as libc::off_t,
            l_len: 1,
            l_pid: 0,
        };
        let rc = unsafe { libc::fcntl(self.file.as_raw_fd(), libc::F_SETLK, &mut flock) };
        if rc == 0 {
            return Ok(true);
        }
        let e = std::io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::EACCES) || e.raw_os_error() == Some(libc::EAGAIN) {
            return Ok(false);
        }
        Err(e)
    }
}

// ── `.tshm` / WAL inspection ──────────────────────────────────────────────

fn read_u32_at(f: &mut fs::File, off: u64) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    f.seek(std::io::SeekFrom::Start(off))?;
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_at(f: &mut fs::File, off: u64) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    f.seek(std::io::SeekFrom::Start(off))?;
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Raw pre-open `.tshm` inspection: the durable frame-index length, committed
/// high-water mark and backfill counter. `snapshot_seq` (even = stable;
/// writers flip it odd while publishing) guards max_frame/nbackfills against
/// a torn read; `frame_index_len` is published under the separate
/// `frame_index_publish_lock` (compare_exchange), so it is not covered here —
/// the u32 read is field-atomic and a slightly stale pairing is diagnostic
/// only (worst case spurious `evidence`, never a false reproduction). `None`
/// when the file is absent or too small to hold the map header (fresh
/// generation).
fn read_tshm_diag(tshm_path: &Path) -> Option<(u64, u64, u64)> {
    let mut f = fs::File::open(tshm_path).ok()?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"TSHMWAL\0" {
        return None;
    }
    for _ in 0..100 {
        let seq_before = read_u64_at(&mut f, TSHM_SNAPSHOT_SEQ_OFFSET).ok()?;
        if seq_before & 1 != 0 {
            continue; // a writer is mid-publish
        }
        let index_len = read_u32_at(&mut f, TSHM_FRAME_INDEX_LEN_OFFSET).ok()? as u64;
        let max_frame = read_u64_at(&mut f, TSHM_MAX_FRAME_OFFSET).ok()?;
        let nbackfills = read_u64_at(&mut f, TSHM_NBACKFILLS_OFFSET).ok()?;
        let seq_after = read_u64_at(&mut f, TSHM_SNAPSHOT_SEQ_OFFSET).ok()?;
        if seq_before == seq_after {
            return Some((index_len, max_frame, nbackfills));
        }
    }
    None
}

/// Number of frames physically present in the `-wal` file.
fn wal_frame_count(wal_path: &Path) -> u64 {
    match fs::metadata(wal_path) {
        Ok(m) if m.len() > WAL_HEADER_BYTES => (m.len() - WAL_HEADER_BYTES) / WAL_FRAME_BYTES,
        _ => 0,
    }
}

/// Stale-index evidence: the durable index diverged from the actual WAL
/// beyond the stop-race tolerance, or the index lost committed entries
/// (`index_len < max_frame`, impossible in a healthy append-only index).
fn is_stale_index_evidence(index_len: u64, max_frame: u64, wal_frames: u64) -> bool {
    index_len.abs_diff(wal_frames) > EVIDENCE_TOLERANCE_ABS || index_len < max_frame
}

// ── Oracle self-check ─────────────────────────────────────────────────────

/// `EXPLAIN QUERY PLAN` text for `sql` — the planner's access-path proof.
fn eqp_plan(conn: &Arc<turso::core::Connection>, sql: &str) -> String {
    let mut stmt = conn.query(sql).expect("eqp").expect("eqp stmt");
    let rows = stmt.run_collect_rows().expect("eqp run");
    rows.iter()
        .flat_map(|r| r.iter())
        .filter_map(|v| match v {
            turso::core::Value::Text(t) => Some(t.value.to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Verify the detection oracles on a disposable healthy DB: all three count
/// access paths return the same cardinality, `INDEXED BY` is honored (errors
/// after dropping the index, and `EXPLAIN QUERY PLAN` names the index), the
/// PK oracle scans the primary-key btree, and integrity_check is clean. A
/// planner silently falling back to a table scan would make the index oracle
/// useless — fail loudly instead of risking a false clean. Runs on any Unix
/// platform.
fn selfcheck_main() -> ! {
    let tmp = Arc::new(tempfile::TempDir::new().expect("temp dir for selfcheck"));
    // Panics (e.g. an oracle `.expect()` on infra failure) exit with the
    // documented EXIT_HARNESS(2) — unconditionally, unlike the actor hook
    // whose code is classify_panic-based — instead of Rust's default 101,
    // and clean up the disposable db like the success path does.
    let hook_tmp = tmp.clone();
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("PANIC: {}", panic_message(info.payload()));
        exit_after_cleanup(&hook_tmp, EXIT_HARNESS);
    }));
    let db_path = tmp.path().join("selfcheck.db");
    let conn = open_db(&db_path);
    execute_retry(&conn, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("create table");
    execute_retry(&conn, "CREATE INDEX idx_t_val ON t(val)").expect("create index");
    for i in 0..100 {
        execute_retry(&conn, &format!("INSERT INTO t (val) VALUES ('row-{i}')"))
            .expect("insert row");
    }

    let (table, index, pk) = count_oracles(&conn).expect("count oracles");
    if table != 100 || index != 100 || pk != 100 {
        eprintln!(
            "selfcheck failed: counts on a healthy 100-row db diverged \
             (table={table} index={index} pk={pk})"
        );
        exit_after_cleanup(&tmp, EXIT_HARNESS);
    }

    // The INDEXED BY oracle must actually scan the index btree.
    let plan = eqp_plan(
        &conn,
        "EXPLAIN QUERY PLAN SELECT count(*) FROM t INDEXED BY idx_t_val",
    );
    if !plan.contains("idx_t_val") {
        eprintln!("selfcheck failed: INDEXED BY did not select the index ({plan:?})");
        exit_after_cleanup(&tmp, EXIT_HARNESS);
    }
    // And the hint must be enforced, not silently ignored.
    execute_retry(&conn, "DROP INDEX idx_t_val").expect("drop index");
    if conn.query(ORACLE_INDEX).is_ok() {
        eprintln!("selfcheck failed: INDEXED BY was silently ignored after the index drop");
        exit_after_cleanup(&tmp, EXIT_HARNESS);
    }
    execute_retry(&conn, "CREATE INDEX idx_t_val ON t(val)").expect("recreate index");

    // The PK oracle must scan the primary-key btree, not silently fall back
    // to a table scan (which would make the oracle useless).
    let pk_plan = eqp_plan(
        &conn,
        "EXPLAIN QUERY PLAN SELECT count(*) FROM t WHERE id >= 0",
    );
    if !pk_plan.contains("PRIMARY KEY") {
        eprintln!("selfcheck failed: PK oracle did not scan the primary key ({pk_plan:?})");
        exit_after_cleanup(&tmp, EXIT_HARNESS);
    }

    // Healthy integrity_check must be clean.
    let problems = collect_integrity_problems(&conn).expect("integrity check");
    if !problems.is_empty() {
        eprintln!("selfcheck failed: healthy db integrity_check reported {problems:?}");
        exit_after_cleanup(&tmp, EXIT_HARNESS);
    }

    eprintln!(
        "selfcheck ok: oracles verified (table=index=pk=100, INDEXED BY + PK plans honored, \
         integrity clean)"
    );
    exit_after_cleanup(&tmp, EXIT_CLEAN);
}

// ── Actor children ────────────────────────────────────────────────────────

/// Deterministic xorshift64 for per-trial cadence jitter (decorrelates
/// consecutive trials so phases do not lock).
struct XorShift(u64);

impl XorShift {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn send_marker(marker: &str) {
    println!("{marker}");
    std::io::stdout().flush().expect("flush marker");
}

/// Writer role (owner): owns the store, bulk-inserts continuously, and — when
/// `churn_hz > 0` — runs a tight in-process open+read+drop loop over the
/// `.tshm`, which on macOS releases every fcntl lock this process holds
/// (the daemon's wal-guard probe pattern). After the deadline it stops all
/// activity but keeps its connection open so the parent's pre-open
/// inspection still sees the live (possibly stale) durable index.
fn writer_main(db_path: &Path, churn_hz: u64, duration: Duration, rate: u64, seed: u64) -> ! {
    install_actor_panic_hook();
    let tshm_path = format!("{}-tshm", db_path.display());
    let conn = open_db(db_path);
    // A large page cache keeps the btree interior pages warm across the
    // auto-checkpoint cache evictions, so the writer survives the full run
    // and the stale index keeps baking wrong pages into the main DB (the
    // durable-B form) instead of the writer dying on a re-read mid-run.
    conn.set_cache_size(20_000);
    execute_retry(&conn, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("create table");
    execute_retry(&conn, "CREATE INDEX idx_t_val ON t(val)").expect("create index");
    bulk_insert(&conn, BATCH_ROWS).expect("baseline insert");

    let stop = Arc::new(AtomicBool::new(false));
    if churn_hz > 0 {
        let stop = stop.clone();
        let tshm = tshm_path.clone();
        std::thread::spawn(move || churn_loop(&tshm, churn_hz, duration, seed, stop));
    }

    send_marker("READY");
    let deadline = Instant::now() + duration;
    let batch_period = Duration::from_secs_f64(BATCH_ROWS as f64 / rate.max(1) as f64);
    while Instant::now() < deadline && rate > 0 {
        bulk_insert(&conn, BATCH_ROWS).expect("writer insert");
        std::thread::sleep(batch_period);
    }
    stop.store(true, Ordering::Relaxed);
    send_marker("STOPPED");
    // Hold the connection (and thus the shared index) until the parent kills
    // us — the parent's inspection must run before any close-time checkpoint.
    // Bounded (duration + detect window + grace) so a SIGKILLed parent leaves
    // no hour-long orphan holding the tempdir DB.
    std::thread::sleep(duration + DETECT_TIMEOUT + CHILD_GRACE);
    std::process::exit(0);
}

/// Pace one loop iteration to `hz`: subtract the work's elapsed time and add
/// deterministic jitter (shared by the writer churn and mirror reopen loops).
fn pace_iteration(rng: &mut XorShift, period_ns: u64, t0: Instant) {
    let elapsed = t0.elapsed().as_nanos() as u64;
    let jitter = rng.next() % (period_ns / 5).max(1);
    if elapsed < period_ns - jitter {
        std::thread::sleep(Duration::from_nanos(period_ns - jitter - elapsed));
    }
}

fn churn_loop(tshm_path: &str, hz: u64, duration: Duration, seed: u64, stop: Arc<AtomicBool>) {
    let mut rng = XorShift::new(seed ^ 0x9E37_79B9);
    let deadline = Instant::now() + duration;
    let period_ns = 1_000_000_000u64 / hz.max(1);
    while Instant::now() < deadline && !stop.load(Ordering::Relaxed) {
        let t0 = Instant::now();
        // open+read+drop: the close is what releases the process-scoped
        // F_SETLK locks on macOS. A plain read-only open takes no locks.
        if let Ok(mut f) = fs::File::open(tshm_path) {
            let mut b = [0u8; 1];
            let _ = f.read_exact(&mut b);
        }
        pace_iteration(&mut rng, period_ns, t0);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Observer {
    ReadOnly,
    Writing,
}

fn observer_arg(o: Observer) -> &'static str {
    match o {
        Observer::ReadOnly => "read-only",
        Observer::Writing => "writing",
    }
}

/// Mirror role (second process): repeatedly drops and reopens the store at
/// the fast cadence. Each reopen probes byte 0 and — inside the writer's
/// lock-drop window — classifies Exclusive and runs the repair that can
/// rebuild the durable index from a stale local scan. No page reads are
/// needed: the reopen's WAL scan is what races the writer. `Writing` mode
/// also inserts rows between reopens (the class-A topology: the mirror's own
/// writes make its local scan "match" the authority, so the stale index is
/// kept and the writer's next append fires the monotonicity panic).
fn mirror_main(
    db_path: &Path,
    mirror_hz: u64,
    duration: Duration,
    seed: u64,
    observer: Observer,
) -> ! {
    install_actor_panic_hook();
    let mut rng = XorShift::new(seed ^ 0x85EB_CA6B);
    // Stop reopening MIRROR_LEAD_SECS before the writer stops inserting: the
    // last (possibly stale) rebuild is then followed by fresh writer appends
    // that keep the divergence visible, instead of being overwritten by a
    // final correct rebuild right before the parent's inspection.
    let deadline = Instant::now() + duration.saturating_sub(Duration::from_secs(MIRROR_LEAD_SECS));
    let period_ns = 1_000_000_000u64 / mirror_hz.max(1);
    while Instant::now() < deadline && mirror_hz > 0 {
        let t0 = Instant::now();
        let conn = if observer == Observer::Writing {
            // The writing observer needs write access for its own inserts.
            open_db(db_path)
        } else {
            open_db_readonly(db_path)
        };
        if observer == Observer::Writing {
            bulk_insert(&conn, MIRROR_INSERT_ROWS).expect("mirror insert");
        }
        drop(conn);
        pace_iteration(&mut rng, period_ns, t0);
    }
    send_marker("STOPPED");
    // No connection is held here (dropped every iteration); the bounded sleep
    // mirrors the writer's hold window so a SIGKILLed parent leaves no orphan.
    std::thread::sleep(duration + DETECT_TIMEOUT + CHILD_GRACE);
    std::process::exit(0);
}

// ── Detection ─────────────────────────────────────────────────────────────

struct DetectOutcome {
    problems: Vec<String>,
    /// Integrity problems carrying the durable-B signature (`wrong # of
    /// entries in index` / `row N missing from index`).
    b_class: usize,
    table: i64,
    index: i64,
    pk: i64,
}

/// Fresh-reopen detection on a quiet store: integrity_check + three verified
/// count oracles. A corrupt pager read surfaces as an `Err` or a panic
/// (classified by the actor panic hook) — both are the reproduced B-state.
fn detect_once(db_path: &Path) -> Result<DetectOutcome, String> {
    detect_once_with(&open_db(db_path))
}

/// Oracle checks against an already-open connection (the proof phase reopens
/// fallibly, so the post-TRUNCATE re-check must not panic on open).
fn detect_once_with(conn: &Arc<turso::core::Connection>) -> Result<DetectOutcome, String> {
    let problems = collect_integrity_problems(conn)?;
    let b_class = problems
        .iter()
        .filter(|p| p.contains(B_SIG_WRONG_ENTRIES) || p.contains(B_SIG_ROW_MISSING))
        .count();
    let (table, index, pk) = count_oracles(conn)?;
    Ok(DetectOutcome {
        problems,
        b_class,
        table,
        index,
        pk,
    })
}

impl DetectOutcome {
    fn reproduced(&self) -> bool {
        !self.problems.is_empty()
            || self.table != self.index
            || self.table != self.pk
            || self.index != self.pk
    }
}

/// Whether a detection failure text is the corruption itself (a stale index
/// mis-resolving pages makes the DB unreadable — the read-side pager
/// symptom, and the reason integrity/count cannot complete).
fn is_corruption_text(text: &str) -> bool {
    let lower = text.to_lowercase();
    READ_SIDE_HINTS.iter().any(|h| lower.contains(h))
}

/// Detect child: pre-open `.tshm` inspection and fresh-reopen oracles; on a
/// mismatch the parent spawns the truncate child that runs the ss2 durability
/// proof (see `run_truncate_proof`).
fn detect_main(db_path: &Path) -> ! {
    install_actor_panic_hook();
    let tshm_path = format!("{}-tshm", db_path.display());
    let wal_path = format!("{}-wal", db_path.display());

    // 1. Pre-open inspection — before ANY reopen can repair the index. The
    //    lifetime-lock probe is a mechanism diagnostic only (the close-drop
    //    fired when byte 0 is free); detection keys on the index/count checks.
    let (index_len, max_frame, nbackfills, wal_frames) = match read_tshm_diag(Path::new(&tshm_path))
    {
        Some((i, m, n)) => (i, m, n, wal_frame_count(Path::new(&wal_path))),
        None => (0, 0, 0, 0),
    };
    let evidence = is_stale_index_evidence(index_len, max_frame, wal_frames);
    let lifetime_free = ProbeFd::open(Path::new(&tshm_path))
        .and_then(|p| p.probe(0))
        .unwrap_or(false);
    println!(
        "tshm-pre: index_len={index_len} max_frame={max_frame} nbackfills={nbackfills} \
         wal_frames={wal_frames} lifetime_lock={} evidence={evidence}",
        if lifetime_free { "free" } else { "held" }
    );

    // 2. Fresh reopen + oracles. A corruption failure (the DB is unreadable
    //    through the stale index) IS the reproduced B-state — report it with
    //    the exact error instead of a harness error.
    let outcome = match detect_once(db_path) {
        Ok(o) => o,
        Err(e) if is_corruption_text(&e) => {
            eprintln!("REPRODUCED-B: corrupt DB during detection: {e}");
            std::process::exit(EXIT_REPRODUCED);
        }
        Err(e) => {
            eprintln!("harness error: detection failed: {e}");
            std::process::exit(EXIT_HARNESS);
        }
    };
    print_outcome("detect", &outcome);

    // 3. A mismatch is the reproduction — the durability proof (does the
    //    desync survive TRUNCATE, the ss2 form) runs in the truncate child
    //    the parent spawns after releasing the actors.
    if outcome.reproduced() {
        eprintln!("REPRODUCED-B: integrity/count mismatch on fresh reopen");
        std::process::exit(EXIT_REPRODUCED);
    }
    if evidence {
        eprintln!(
            "EVIDENCE: stale durable index (index_len={index_len} vs wal_frames={wal_frames}) but no integrity/count damage"
        );
        std::process::exit(EXIT_EVIDENCE);
    }
    eprintln!("CLEAN: no mechanism signal");
    std::process::exit(EXIT_CLEAN);
}

/// Print a detect/truncate outcome block (count oracles + integrity problems).
fn print_outcome(prefix: &str, o: &DetectOutcome) {
    println!(
        "{prefix}: table={} index={} pk={} problems={} b_class={}",
        o.table,
        o.index,
        o.pk,
        o.problems.len(),
        o.b_class
    );
    for p in &o.problems {
        println!("  {prefix}-integrity: {p}");
    }
}

/// ss2 durability proof, run as a separate child after the parent has killed
/// the actors (so the TRUNCATE checkpoint is not busy). Exit-code contract
/// mirrors the parent's: 0 = durable reproduction, 1 = consistent after
/// TRUNCATE, 6 = proof skipped. The parent ignores the code (the reproduction
/// is already confirmed by the detect child) — for manual/diagnostic use.
///
/// No actor panic hook: the proof is caught with `catch_unwind` so a storage
/// panic is attributed to the right stage — a panic after the checkpoint
/// completed means the stale pages are baked into the main DB (the durable
/// ss2 read-side variant); a panic before it cannot establish the verdict.
///
/// A manual `--truncate` against a nonexistent path creates the DB file —
/// turso's `OpenFlags::default` includes Create — which is fine in the
/// parent flow, where the DB always exists.
fn truncate_main(db_path: &Path) -> ! {
    // The proof reports panics itself from the catch_unwind payload; silence
    // the default hook so the message is not printed twice. Tradeoff: this
    // also silences background-thread panics (catch_unwind is per-thread) —
    // accepted: the proof runs entirely on the calling thread.
    std::panic::set_hook(Box::new(|_| {}));
    let after_checkpoint = std::cell::Cell::new(false);
    let proof = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let conn = open_db_with_result(db_path, OpenFlags::default())?;
        wal_checkpoint(&conn, "TRUNCATE")?;
        drop(conn);
        after_checkpoint.set(true);
        let conn = open_db_with_result(db_path, OpenFlags::default())?;
        detect_once_with(&conn)
    }));
    let exit = match proof {
        Ok(Ok(o)) => {
            print_outcome("truncate", &o);
            if o.reproduced() {
                println!("truncate: durable: persists through TRUNCATE (ss2 form)");
                EXIT_REPRODUCED
            } else {
                // Covers both a desync healed by the checkpoint and a store
                // that was never desynced (manual invocation).
                println!("truncate: consistent after TRUNCATE (no durable desync)");
                EXIT_CLEAN
            }
        }
        // A corruption failure after the checkpoint completed is the ss2
        // durable form (the TRUNCATE finished and the main DB still cannot be
        // read through the stale index); a failure before it is a skipped
        // proof — durable-vs-healed is unprovable without a completed TRUNCATE.
        Ok(Err(e)) if after_checkpoint.get() && is_corruption_text(&e) => {
            println!("truncate: corrupt DB after TRUNCATE: {e}");
            println!("truncate: durable: persists through TRUNCATE (ss2 form)");
            EXIT_REPRODUCED
        }
        Ok(Err(e)) => {
            eprintln!("durability proof skipped: {e}");
            EXIT_PROOF_SKIPPED
        }
        Err(payload) if after_checkpoint.get() => {
            // Same evidentiary standard as the Err arm: only a corruption
            // panic after the checkpoint establishes the durable form.
            let msg = panic_message(&*payload);
            if is_corruption_text(&msg) {
                println!("truncate: storage panic after TRUNCATE: {msg}");
                println!("truncate: durable: persists through TRUNCATE (ss2 form)");
                EXIT_REPRODUCED
            } else {
                eprintln!("durability proof skipped: storage panic after TRUNCATE: {msg}");
                EXIT_PROOF_SKIPPED
            }
        }
        Err(payload) => {
            eprintln!(
                "durability proof skipped: storage panic: {}",
                panic_message(&*payload)
            );
            EXIT_PROOF_SKIPPED
        }
    };
    std::process::exit(exit);
}

// ── Parent orchestration ──────────────────────────────────────────────────

struct ActorHandle {
    child: Child,
    stdout_rx: std::sync::mpsc::Receiver<String>,
    stderr: Arc<Mutex<String>>,
    role: &'static str,
}

/// Spawn a child role with piped stdout (markers) and piped stderr (drained
/// into a shared buffer, printed live).
fn spawn_actor(exe: &Path, role: &'static str, args: &[&str]) -> Result<ActorHandle, String> {
    let mut child = Command::new(exe)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn {role}: {e}"))?;
    let stdout = child.stdout.take().expect("actor stdout");
    let stderr = child.stderr.take().expect("actor stderr");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let l = line.trim().to_string();
                    if !l.is_empty() && tx.send(l).is_err() {
                        break;
                    }
                }
            }
        }
    });
    let captured = Arc::new(Mutex::new(String::new()));
    {
        let captured = captured.clone();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let l = line.trim_end().to_string();
                        if !l.is_empty() {
                            eprintln!("[{role}] {l}");
                            if let Ok(mut cap) = captured.lock() {
                                cap.push_str(&l);
                                cap.push('\n');
                            }
                        }
                    }
                }
            }
        });
    }
    Ok(ActorHandle {
        child,
        stdout_rx: rx,
        stderr: captured,
        role,
    })
}

impl ActorHandle {
    fn recv_marker(&mut self, label: &str, expected: &str) -> Result<(), String> {
        let deadline = Instant::now() + MARKER_TIMEOUT;
        loop {
            // A child that died instantly (e.g. DB open failure) must not
            // stall the parent a full MARKER_TIMEOUT.
            if let Some(code) = self.exited() {
                // Stderr flush grace (as in poll_actor/wait_child) so the
                // cause of an instantly-dying child is not lost.
                std::thread::sleep(Duration::from_millis(100));
                return Err(format!(
                    "{} exited (code {code}) while waiting for {expected:?} ({label}): {}",
                    self.role,
                    self.stderr_text().trim()
                ));
            }
            match self.stdout_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(m) if m == expected => return Ok(()),
                Ok(m) => {
                    return Err(format!(
                        "{} expected {expected:?} ({label}), got {m:?}",
                        self.role
                    ));
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) if Instant::now() >= deadline => {
                    return Err(format!(
                        "{} timed out waiting for {expected:?} ({label})",
                        self.role
                    ));
                }
                Err(_) => {}
            }
        }
    }

    fn exited(&mut self) -> Option<i32> {
        match self.child.try_wait().ok()? {
            // A signal death (SIGSEGV/SIGABRT on the corrupt state) is still
            // an exit — `code()` is None for it, so map it to 128+signal.
            Some(status) => status.code().or_else(|| status.signal().map(|s| 128 + s)),
            None => None,
        }
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn stderr_text(&self) -> String {
        self.stderr.lock().map(|s| s.clone()).unwrap_or_default()
    }
}

/// Classify an actor's stderr for panic signatures (belt-and-braces: the
/// child's classified exit code is primary, but a hooked panic may only be
/// visible in the captured text).
fn classify_stderr(text: &str) -> Option<i32> {
    // The last PANIC line (an OS crash message may follow the panic text).
    let last = text.lines().rev().find(|l| l.contains("PANIC:"))?;
    let msg = last.trim_start_matches("PANIC:");
    let code = classify_panic(msg);
    if code == EXIT_HARNESS {
        return None;
    }
    Some(code)
}

/// Map a writer/mirror exit to a probe-outcome class. A panic in these actors
/// is treated as a corruption outcome: turso storage code crashing on the
/// desynced state is the probe firing, and even a bench-code `.expect()`
/// failure surfaced as a PANIC line is classified as a probe outcome — the
/// churn-less control group is the harness-bug detector for the actor code.
/// Harness errors are kept for non-panic exits (code 0) only: a raw signal
/// death (SIGSEGV/SIGABRT from the corrupt pager — the exact read-side defect
/// class, no panic text to classify) and any PANIC-bearing stderr are probe
/// outcomes, so a mitigation that turns the panic into a plain signal death
/// cannot silently convert reproductions into harness aborts. The detect
/// child's classification (`finish_detect`) instead honors its own EXIT_HARNESS
/// so a harness-bug panic there is never promoted.
fn classify_actor_exit(code: i32, stderr: &str) -> Option<i32> {
    match code {
        CHILD_EXIT_A | CHILD_EXIT_C | CHILD_EXIT_READSIDE => Some(code),
        0 => None, // clean exit before STOPPED is unexpected
        // `exited()` maps signal deaths to 128+signal; on the desynced state
        // that is the read-side pager crash (storage code dying with no Rust
        // panic text to classify).
        _ if code > 128 => Some(CHILD_EXIT_READSIDE),
        _ => classify_stderr(stderr)
            .or_else(|| stderr.contains("PANIC:").then_some(CHILD_EXIT_READSIDE)),
    }
}

#[derive(Default)]
struct TrialStats {
    reproduced_b: usize,
    reproduced_a: usize,
    evidence: usize,
    panic_void: usize,
    clean: usize,
}

/// Trial outcome classification (type-safe; run_batch's match is exhaustive).
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrialKind {
    ReproducedB,
    ReproducedA,
    Evidence,
    PanicVoid,
    Clean,
    Harness,
}

/// Panic-class priority: the write-path asserts (A) are the headline symptom
/// and must never be shadowed by a lower-value concurrent panic (C or
/// read-side) in the other actor.
fn class_rank(code: i32) -> u8 {
    match code {
        CHILD_EXIT_A => 3,
        CHILD_EXIT_C => 2,
        CHILD_EXIT_READSIDE => 1,
        _ => 0,
    }
}

fn report_panic_class(code: i32, actor: &str) {
    match code {
        CHILD_EXIT_A => {
            eprintln!("REPRODUCED-A: {actor} hit a write-path monotonicity/high-water-mark panic");
        }
        CHILD_EXIT_C => {
            eprintln!("{actor} hit a class-C reader-slot panic (sibling lock_drop_repro defect)");
        }
        CHILD_EXIT_READSIDE => {
            eprintln!(
                "{actor} panicked in turso storage code (mechanism fired; detection continues \
                 on the leftover state)"
            );
        }
        _ => unreachable!("classify_actor_exit only yields class codes"),
    }
}

/// Poll one actor for an early classified exit or the STOPPED marker.
/// `Ok(true)` = done (STOPPED or classified panic), `Ok(false)` = still
/// running, `Err` = unexpected early exit (harness error). Concurrent panics
/// merge into `early_panic` by class priority (A > C > read-side).
fn poll_actor(
    actor: &mut ActorHandle,
    early_panic: &mut Option<(i32, &'static str)>,
) -> Result<bool, String> {
    if let Some(code) = actor.exited() {
        // Give the stderr drain thread a moment to flush the tail after the
        // child's exit (like the detect path does), so an unrecognized panic
        // line is not missed and misreported as a harness error.
        std::thread::sleep(Duration::from_millis(100));
        match classify_actor_exit(code, &actor.stderr_text()) {
            Some(class) => {
                if early_panic
                    .as_ref()
                    .is_none_or(|(c, _)| class_rank(class) > class_rank(*c))
                {
                    *early_panic = Some((class, actor.role));
                }
                Ok(true)
            }
            None => Err(format!("{} exited early: code {code}", actor.role)),
        }
    } else if let Ok(m) = actor.stdout_rx.recv_timeout(Duration::from_millis(100)) {
        if m == "STOPPED" {
            Ok(true)
        } else {
            Err(format!("{} emitted unexpected marker {m:?}", actor.role))
        }
    } else {
        Ok(false)
    }
}

/// Drain a child's buffered stdout diagnostics (tshm-pre / detect / truncate
/// lines) — the most useful evidence when the child dies or hangs.
fn drain_stdout(rx: &std::sync::mpsc::Receiver<String>, buf: &mut String) {
    while let Ok(m) = rx.recv_timeout(Duration::from_millis(50)) {
        buf.push_str(&m);
        buf.push('\n');
    }
}

/// Run a single trial: writer (optionally churning) + mirror, wait for both
/// to reach STOPPED, then pre-open inspection + detect child.
///
/// Returns `(TrialKind, detail)`; the detail is non-empty only for harness
/// errors (probe-outcome stderr is printed live by the drain threads).
fn run_trial(
    db_path: &Path,
    churn_hz: u64,
    duration_secs: u64,
    mirror_hz: u64,
    rate: u64,
    seed: u64,
    observer: Observer,
) -> (TrialKind, String) {
    let exe = std::env::current_exe().expect("current exe");
    let db = db_path.to_str().expect("utf8").to_string();

    let mut writer = match spawn_actor(
        &exe,
        "writer",
        &[
            "--writer",
            &db,
            "--churn-hz",
            &churn_hz.to_string(),
            "--duration",
            &duration_secs.to_string(),
            "--rate",
            &rate.to_string(),
            "--seed",
            &seed.to_string(),
        ],
    ) {
        Ok(a) => a,
        Err(e) => return (TrialKind::Harness, e),
    };

    // The mirror must not race the writer's CREATE: spawn it only once the
    // writer reports its baseline is on disk.
    if let Err(e) = writer.recv_marker("ready", "READY") {
        writer.kill();
        return (TrialKind::Harness, e);
    }
    let mut mirror = match spawn_actor(
        &exe,
        "mirror",
        &[
            "--mirror",
            &db,
            "--mirror-hz",
            &mirror_hz.to_string(),
            "--duration",
            &duration_secs.to_string(),
            "--seed",
            &seed.to_string(),
            "--observer",
            observer_arg(observer),
        ],
    ) {
        Ok(a) => a,
        Err(e) => {
            writer.kill();
            return (TrialKind::Harness, e);
        }
    };

    // Wait for both STOPPED markers (deadline ~ duration), watching for early
    // actor exits (panics). Exit-code classification is primary; the captured
    // stderr is the belt-and-braces fallback for a hooked panic. Concurrent
    // panics keep the highest-value class (A > C > read-side) so a genuine
    // monotonicity panic in one actor is never shadowed by the other.
    let wait_deadline = Instant::now()
        + Duration::from_secs(duration_secs.saturating_add(STOP_DEADLINE_SLACK_SECS))
        + CHILD_GRACE;
    let mut writer_done = false;
    let mut mirror_done = false;
    let mut early_panic: Option<(i32, &'static str)> = None;
    while Instant::now() < wait_deadline {
        if !writer_done {
            match poll_actor(&mut writer, &mut early_panic) {
                Ok(true) => writer_done = true,
                Ok(false) => {}
                Err(e) => {
                    mirror.kill();
                    return (TrialKind::Harness, e);
                }
            }
        }
        if !mirror_done {
            match poll_actor(&mut mirror, &mut early_panic) {
                Ok(true) => mirror_done = true,
                Ok(false) => {}
                Err(e) => {
                    writer.kill();
                    return (TrialKind::Harness, e);
                }
            }
        }
        if writer_done && mirror_done {
            break;
        }
        // A class-A panic is an immediate reproduction — do not wait for the
        // other actor's STOPPED (~duration) before reporting it.
        if early_panic.is_some_and(|(c, _)| c == CHILD_EXIT_A) {
            break;
        }
    }

    // If an actor panicked mid-run, classify it. A class-A write-path panic
    // is an immediate reproduction; class-C / read-side panics are reported
    // loudly and the detection phase still runs on the leftover state (the
    // writer may have baked damage before dying).
    let mut early_panic_code: Option<i32> = None;
    if let Some((code, actor)) = early_panic {
        report_panic_class(code, actor);
        early_panic_code = Some(code);
        writer.kill();
        mirror.kill();
        if code == CHILD_EXIT_A {
            return (TrialKind::ReproducedA, String::new());
        }
    } else if !writer_done || !mirror_done {
        writer.kill();
        mirror.kill();
        return (
            TrialKind::Harness,
            format!("actors did not reach STOPPED (writer={writer_done} mirror={mirror_done})"),
        );
    }

    // The writer still holds its connection in the normal path (the early-
    // panic branch above already killed both actors): run the detect child
    // while the (possibly stale) durable index is still live.
    let mut detect = match spawn_actor(&exe, "detect", &["--detect", &db]) {
        Ok(a) => a,
        Err(e) => {
            writer.kill();
            mirror.kill();
            return (TrialKind::Harness, e);
        }
    };
    // Bound the detect child; the actor panic hook exits with a class code.
    let mut detect_stdout = String::new();
    let code = match wait_child(&mut detect, &mut detect_stdout, DETECT_TIMEOUT) {
        Some(code) => code,
        None => {
            // The hung child's tshm-pre / detect lines are the most useful
            // evidence — drain them before reporting the timeout.
            if !detect_stdout.is_empty() {
                eprintln!("[detect] {detect_stdout}");
            }
            writer.kill();
            mirror.kill();
            return (TrialKind::Harness, "detect child timed out".into());
        }
    };
    let stderr = detect.stderr_text();
    if !detect_stdout.is_empty() {
        eprintln!("[detect] {detect_stdout}");
    }
    writer.kill();
    mirror.kill();
    // Only the detect child's own stderr feeds the classification — an early
    // actor's panic line must not be attributed to a harness-exiting detect
    // child (early_panic_code above already carries the actor's signal:
    // Clean → PanicVoid).
    let (mut kind, det_detail) = finish_detect(code, stderr);
    if kind == TrialKind::Clean && early_panic_code.is_some() {
        kind = TrialKind::PanicVoid;
    }
    if kind == TrialKind::ReproducedB {
        // ss2 durability proof in a fresh child once the actors' connections
        // are gone (kill() above waited): the TRUNCATE checkpoint is not
        // busy, so the proof is reachable in the primary B path, not just the
        // early-panic path.
        run_truncate_proof(&exe, &db);
    }
    (kind, det_detail)
}

/// Wait for a child's exit (bounded by `timeout`), draining its stdout
/// diagnostics into `out`; the 100ms grace before the drain lets both the
/// stderr drain thread and the stdout reader thread flush their tails before
/// the caller classifies or prints. Returns the exit code, or `None` on
/// timeout (the child is killed).
fn wait_child(actor: &mut ActorHandle, out: &mut String, timeout: Duration) -> Option<i32> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(code) = actor.exited() {
            std::thread::sleep(Duration::from_millis(100));
            drain_stdout(&actor.stdout_rx, out);
            return Some(code);
        }
        if Instant::now() > deadline {
            actor.kill();
            drain_stdout(&actor.stdout_rx, out);
            return None;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Run the ss2 durability proof in a fresh `--truncate` child (spawned after
/// the detect child confirmed a mismatch and the actors' connections are
/// gone). The proof is a bonus diagnostic — the reproduction is already
/// confirmed — so any spawn/wait failure just skips it; the child prints its
/// own verdict (durable / healed / skipped) live and its exit code is only a
/// manual-invocation contract (0 durable, 1 consistent, 6 skipped).
fn run_truncate_proof(exe: &Path, db: &str) {
    let mut trunc = match spawn_actor(exe, "truncate", &["--truncate", db]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("durability proof skipped: {e}");
            return;
        }
    };
    let mut out = String::new();
    let done = wait_child(&mut trunc, &mut out, DETECT_TIMEOUT).is_some();
    if !out.is_empty() {
        eprintln!("[truncate] {out}");
    }
    if !done {
        eprintln!("durability proof skipped: truncate child timed out");
    }
}

/// Panic-class code → TrialKind, printing the live message (shared by the
/// primary exit-code arms and the stderr-fallback classification in
/// `finish_detect`).
fn class_outcome(code: i32) -> TrialKind {
    let (kind, msg): (TrialKind, &str) = match code {
        CHILD_EXIT_A => (
            TrialKind::ReproducedA,
            "REPRODUCED-A: detect child hit a write-path monotonicity panic",
        ),
        CHILD_EXIT_C => (
            TrialKind::PanicVoid,
            "detect hit a class-C reader-slot panic (sibling lock_drop_repro defect)",
        ),
        // The detection itself could not read the store: the DB is corrupt
        // through the stale index — the read-side pager B-symptom.
        CHILD_EXIT_READSIDE => (
            TrialKind::ReproducedB,
            "REPRODUCED-B: corrupt DB during detection (read-side pager panic)",
        ),
        _ => unreachable!("class codes only"),
    };
    eprintln!("{msg}");
    kind
}

fn finish_detect(code: i32, stderr: String) -> (TrialKind, String) {
    match code {
        EXIT_REPRODUCED => {
            eprintln!("REPRODUCED-B: detect child confirmed class-B durable desync");
            (TrialKind::ReproducedB, String::new())
        }
        EXIT_EVIDENCE => {
            eprintln!("EVIDENCE: mechanism fired (stale index) without durable damage");
            (TrialKind::Evidence, String::new())
        }
        EXIT_CLEAN => (TrialKind::Clean, String::new()),
        CHILD_EXIT_A | CHILD_EXIT_C | CHILD_EXIT_READSIDE => (class_outcome(code), String::new()),
        // A signal death in the detect child can only be self-inflicted: the
        // parent kills only on timeout (which returns None above), so a raw
        // SIGSEGV/SIGABRT here is the read-side pager crashing on the
        // desynced state — a probe outcome, not a harness error (mirrors
        // classify_actor_exit's 128+signal arm).
        _ if code > 128 => (class_outcome(CHILD_EXIT_READSIDE), String::new()),
        // A panic that slipped past the child's exit code is classified from
        // the detect child's own stderr; `classify_stderr` honors the child's
        // deliberate EXIT_HARNESS (an unrecognized panic its hook classified
        // as harness is never promoted to a reproduction). Probe outcomes
        // return no detail — the stderr was printed live by the drain
        // threads; the harness detail carries the exit-code context.
        _ => match classify_stderr(&stderr) {
            Some(class) => (class_outcome(class), String::new()),
            None => (
                TrialKind::Harness,
                format!("detect child exited with code {code}: {stderr}"),
            ),
        },
    }
}

/// Batch configuration (one struct keeps run_batch's signature under the
/// clippy arg-count bar and makes `--run`/`--control` construction uniform).
struct BatchSpec {
    churn: bool,
    iterations: usize,
    duration_secs: u64,
    churn_hz: u64,
    mirror_hz: u64,
    rate: u64,
    seed: u64,
    observer: Observer,
}

fn batch_spec(args: &[String], churn: bool) -> BatchSpec {
    BatchSpec {
        churn,
        iterations: arg_usize(
            args,
            "--iterations",
            if churn {
                DEFAULT_ITERATIONS
            } else {
                CONTROL_ITERATIONS
            },
        ),
        duration_secs: arg_u64(args, "--duration", DEFAULT_DURATION_SECS),
        churn_hz: arg_u64(args, "--churn-hz", DEFAULT_CHURN_HZ),
        mirror_hz: arg_u64(args, "--mirror-hz", DEFAULT_MIRROR_HZ),
        rate: arg_u64(args, "--rate", DEFAULT_ROWS_PER_SEC),
        seed: arg_u64(args, "--seed", DEFAULT_SEED),
        observer: arg_observer(args),
    }
}

/// Batch orchestrator. `churn: false` is the control group.
fn run_batch(spec: BatchSpec) -> ! {
    if !cfg!(target_os = "macos") {
        eprintln!(
            "SKIP: the macOS F_SETLK lock-drop repro is not applicable on this platform \
             (the close-drop is a POSIX F_SETLK property; Linux uses OFD locks)"
        );
        std::process::exit(EXIT_SKIP);
    }
    let BatchSpec {
        churn,
        iterations,
        duration_secs,
        churn_hz,
        mirror_hz,
        rate,
        seed,
        observer,
    } = spec;
    // Degenerate budgets must not emit vacuous clean claims: a zero-trial
    // batch or a zero-duration run never exercises the mechanism, so neither
    // can support "fix present" / "causality proof".
    if iterations == 0 {
        harness_no_tmp("--iterations must be >= 1 (a zero-trial batch proves nothing)");
    }
    if duration_secs <= MIRROR_LEAD_SECS {
        harness_no_tmp(&format!(
            "--duration must be > {MIRROR_LEAD_SECS} (the mirror's reopen window is duration - \
             {MIRROR_LEAD_SECS}s; a shorter run degenerates into the owner-only topology, \
             where the second process never participates)"
        ));
    }
    let effective_churn = if churn { churn_hz } else { 0 };
    if churn && churn_hz == 0 {
        harness_no_tmp(
            "--run requires --churn-hz > 0 (churn off is the control topology; use --control)",
        );
    }
    // `--churn-hz` is honored by --run only (control always disables it);
    // `--mirror-hz` is honored by both modes.
    if churn && churn_hz > MAX_CADENCE_HZ {
        harness_no_tmp(&format!(
            "--churn-hz must be <= {MAX_CADENCE_HZ} (the cadence jitter divides the period)"
        ));
    }
    if mirror_hz > MAX_CADENCE_HZ {
        harness_no_tmp(&format!(
            "--mirror-hz must be <= {MAX_CADENCE_HZ} (the cadence jitter divides the period)"
        ));
    }
    // The fast-mirror cadence and a writing workload are preconditions for the
    // fix-present claim (the owner-only / no-insert topologies calibrate 0/3
    // and never fire), so `--run` rejects them; `--control` allows them — the
    // causality claim is about churn, and the actors simply do nothing.
    if churn && mirror_hz == 0 {
        harness_no_tmp(
            "--run requires --mirror-hz > 0 (mirror-hz 0 = the owner-only topology: the mirror \
             never reopens, calibrated 0/3 — a clean run there proves nothing)",
        );
    }
    if churn && rate == 0 {
        harness_no_tmp(
            "--run requires --rate > 0 (rate 0 = the writer never inserts — the mechanism never \
             fires, so a clean run proves nothing)",
        );
    }
    if effective_churn == 0 && observer == Observer::Writing {
        eprintln!(
            "note: --control --observer writing is the two-writer topology (the sibling \
             wal_race_repro defect); a violation here may be that defect, not a churn \
             causality failure — the control's causality claim holds for the read-only observer"
        );
    }
    let label = if churn { "run" } else { "control" };
    eprintln!(
        "wal_stale_index_repro --{label}: {iterations} trials x {duration_secs}s, \
         churn_hz={effective_churn} mirror_hz={mirror_hz} rate={rate} rows/s seed=0x{seed:X} \
         observer={}",
        observer_arg(observer)
    );
    let mut stats = TrialStats::default();
    // Batch-scoped temp dir: the pre-loop dir serves trial 1, later trials
    // replace it; the batch-end exits go through exit_after_cleanup so the
    // last trial's dir (with its repro DB) is removed — a bare
    // `std::process::exit` would skip TempDir's Drop.
    let mut tmp = tempfile::TempDir::new().expect("temp dir per trial");
    for i in 0..iterations {
        if i > 0 {
            tmp = tempfile::TempDir::new().expect("temp dir per trial");
        }
        let trial_seed = seed ^ ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        eprintln!("trial {}/{} ...", i + 1, iterations);
        let (kind, detail) = run_trial(
            &tmp.path().join("repro.db"),
            effective_churn,
            duration_secs,
            mirror_hz,
            rate,
            trial_seed,
            observer,
        );
        match kind {
            TrialKind::ReproducedB => stats.reproduced_b += 1,
            TrialKind::ReproducedA => stats.reproduced_a += 1,
            TrialKind::Evidence => stats.evidence += 1,
            TrialKind::PanicVoid => stats.panic_void += 1,
            TrialKind::Clean => stats.clean += 1,
            TrialKind::Harness => {
                eprintln!("harness error: {detail}");
                exit_after_cleanup(&tmp, EXIT_HARNESS);
            }
        }
        if stats.reproduced_b + stats.reproduced_a > 0 {
            // The early-stop contract applies to the churn run only: a
            // control-group reproduction is a harness bug and must exit 2
            // (CONTROL GROUP VIOLATION), never 0 (confirmed turso defect).
            if effective_churn == 0 {
                eprintln!(
                    "CONTROL GROUP VIOLATION at trial {}/{}: the churn-less control reproduced \
                     class-A/B — a harness bug, not a turso defect",
                    i + 1,
                    iterations
                );
                exit_after_cleanup(&tmp, EXIT_HARNESS);
            }
            eprintln!(
                "REPRODUCED at trial {}/{} (B={} A={} evidence={} panic-void={} clean={})",
                i + 1,
                iterations,
                stats.reproduced_b,
                stats.reproduced_a,
                stats.evidence,
                stats.panic_void,
                stats.clean
            );
            exit_after_cleanup(&tmp, EXIT_REPRODUCED);
        }
    }

    // Batch finished without A/B damage. Honest aggregation.
    eprintln!(
        "batch done: B={} A={} evidence={} panic-void={} clean={} (of {} trials)",
        stats.reproduced_b,
        stats.reproduced_a,
        stats.evidence,
        stats.panic_void,
        stats.clean,
        iterations
    );
    if effective_churn == 0 {
        if stats.reproduced_b + stats.reproduced_a + stats.evidence + stats.panic_void > 0 {
            eprintln!(
                "CONTROL GROUP VIOLATION: the churn-less control must be clean, but {} of {} \
                 trials showed a mechanism signal",
                stats.reproduced_b + stats.reproduced_a + stats.evidence + stats.panic_void,
                iterations
            );
            exit_after_cleanup(&tmp, EXIT_HARNESS);
        }
        // The causality-proof claim needs the default calibration topology: a
        // read-only observer, the mirror actually reopens, the writer keeps
        // inserting, and the run is not shorter than the calibration —
        // anything weaker is evidence, not proof.
        let full_control = observer == Observer::ReadOnly
            && iterations >= CONTROL_ITERATIONS
            && mirror_hz == DEFAULT_MIRROR_HZ
            && rate == DEFAULT_ROWS_PER_SEC
            && duration_secs >= DEFAULT_DURATION_SECS;
        if full_control {
            eprintln!("CONTROL CLEAN: the churn-less control group stayed clean (causality proof)");
        } else {
            eprintln!(
                "CONTROL CLEAN: {iterations} trial(s) clean — weak evidence (the \
                 causality-proof claim needs a read-only observer and >= {CONTROL_ITERATIONS} \
                 trials at the default fast-mirror topology: mirror {DEFAULT_MIRROR_HZ}/s, \
                 {DEFAULT_ROWS_PER_SEC} rows/s, >= {DEFAULT_DURATION_SECS}s)"
            );
        }
        exit_after_cleanup(&tmp, EXIT_CLEAN);
    }
    if stats.evidence + stats.panic_void > 0 {
        eprintln!(
            "INCONCLUSIVE: the lock-drop mechanism still fired (evidence={} panic-void={}) but no \
             class-A/B damage materialized in {} trials — not a fix",
            stats.evidence, stats.panic_void, iterations
        );
        exit_after_cleanup(&tmp, EXIT_BUDGET);
    }
    if iterations < DEFAULT_ITERATIONS {
        eprintln!(
            "BUDGET EXHAUSTED: {iterations} clean trials < {DEFAULT_ITERATIONS} — a clean result \
             at this budget cannot distinguish a fixed build from bad luck (6/11 calibration); \
             rerun with --iterations {DEFAULT_ITERATIONS}"
        );
        exit_after_cleanup(&tmp, EXIT_BUDGET);
    }
    // The confidence figure assumes the 6/11 fast-mirror calibration defaults
    // and the read-only mirror topology (the writing-observer A-path has no
    // documented hit rate); a clean run outside those is weaker evidence.
    let default_cadence = observer == Observer::ReadOnly
        && churn_hz == DEFAULT_CHURN_HZ
        && mirror_hz == DEFAULT_MIRROR_HZ
        && rate == DEFAULT_ROWS_PER_SEC
        && duration_secs == DEFAULT_DURATION_SECS;
    eprintln!(
        "CLEAN: 0/{iterations} trials showed any mechanism signal — fix present \
         (P(0/{iterations} | broken at 6/11 hit rate) ≈ {:.2}%){}",
        (5.0f64 / 11.0f64).powi(iterations as i32) * 100.0,
        if default_cadence {
            ""
        } else {
            " (note: non-default topology — the 6/11 confidence assumes the read-only \
             fast-mirror calibration defaults)"
        }
    );
    exit_after_cleanup(&tmp, EXIT_CLEAN);
}

// ── Dispatch ──────────────────────────────────────────────────────────────

/// Parse a u64 flag value; accepts decimal or `0x`/`0X`-prefixed hex (the
/// seed default is documented and printed in hex). A bare prefix without
/// digits is rejected, not silently parsed as 0.
fn parse_u64(v: &str) -> Option<u64> {
    match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        Some(hex) if !hex.is_empty() => u64::from_str_radix(hex, 16).ok(),
        Some(_) => None,
        None => v.parse().ok(),
    }
}

fn arg_u64(args: &[String], flag: &str, default: u64) -> u64 {
    match args.iter().position(|a| a == flag) {
        Some(i) => match args.get(i + 1).and_then(|v| parse_u64(v)) {
            Some(v) => v,
            None => harness_no_tmp(&format!("{flag} requires a u64 value")),
        },
        None => default,
    }
}

fn arg_usize(args: &[String], flag: &str, default: usize) -> usize {
    arg_u64(args, flag, default as u64) as usize
}

fn arg_observer(args: &[String]) -> Observer {
    match args.iter().position(|a| a == "--observer") {
        Some(i) => match args.get(i + 1).map(String::as_str) {
            Some("read-only") => Observer::ReadOnly,
            Some("writing") => Observer::Writing,
            _ => harness_no_tmp("--observer requires read-only|writing"),
        },
        None => Observer::ReadOnly,
    }
}

fn print_usage() {
    eprintln!(
        "usage: wal_stale_index_repro <mode> [flags]

modes:
  --run               primary batch repro (macOS): writer (churn) + read-only
                      mirror, default 11 trials, early stop on reproduction
  --control           control group: identical run with the churn disabled;
                      must stay clean (causality proof)
  --selfcheck         verify the detection oracles on a disposable db
  --writer DB         internal writer child (parent-spawned only)
  --mirror DB         internal mirror child (parent-spawned only)
  --detect DB         internal detect child (parent-spawned only)
  --truncate DB       internal truncate child (parent-spawned only)

flags (--run / --control):
  --iterations N      trials per batch (default 11; control default 3)
  --duration S        run seconds per trial (default 30; requires > 1 — the
                      mirror's reopen window is duration-1s, so a 1s run
                      degenerates into the owner-only topology)
  --churn-hz H        writer .tshm open+close rate /s (default 400; --run
                      requires > 0 — churn off is --control, which always
                      disables churn)
  --mirror-hz H       mirror reopen rate /s (default 15; --run requires > 0
                      — 0 = the owner-only topology: the mirror never
                      reopens, calibrated 0/3)
  --rate R            writer insert rows/s (default 1900; --run requires
                      > 0 — 0 = the writer never inserts)
  --seed S            jitter seed for trial decorrelation (default 0x5EED;
                      decimal or 0x-prefixed hex)
  --observer MODE     read-only (default) | writing (class-A topology;
                      --control is only meaningful read-only — the writing
                      observer is the two-writer wal_race_repro topology)

exit codes:
  0  bug reproduced (class-A write-path panic or class-B durable desync)
  1  clean / fix present (>=11 trials, zero mechanism signals) or control clean
  2  harness error / control-group violation
  3  skip (non-macOS)
  4  inconclusive (mechanism fired without A/B damage, or fewer than 11 clean
     trials ran — a clean result at that budget cannot distinguish a fixed
     build from bad luck)
  5  internal detect-child evidence-only code (parent-visible only)
  6  internal truncate-child proof-skipped code (parent-visible only)"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--run") => run_batch(batch_spec(&args, true)),
        Some("--control") => run_batch(batch_spec(&args, false)),
        Some("--selfcheck") => selfcheck_main(),
        Some("--writer") => {
            let Some(db) = args.get(2) else {
                harness_no_tmp("--writer requires a db path");
            };
            writer_main(
                Path::new(db),
                arg_u64(&args, "--churn-hz", DEFAULT_CHURN_HZ),
                Duration::from_secs(arg_u64(&args, "--duration", DEFAULT_DURATION_SECS)),
                arg_u64(&args, "--rate", DEFAULT_ROWS_PER_SEC),
                arg_u64(&args, "--seed", DEFAULT_SEED),
            );
        }
        Some("--mirror") => {
            let Some(db) = args.get(2) else {
                harness_no_tmp("--mirror requires a db path");
            };
            mirror_main(
                Path::new(db),
                arg_u64(&args, "--mirror-hz", DEFAULT_MIRROR_HZ),
                Duration::from_secs(arg_u64(&args, "--duration", DEFAULT_DURATION_SECS)),
                arg_u64(&args, "--seed", DEFAULT_SEED),
                arg_observer(&args),
            );
        }
        Some("--detect") => {
            let Some(db) = args.get(2) else {
                harness_no_tmp("--detect requires a db path");
            };
            detect_main(Path::new(db));
        }
        Some("--truncate") => {
            let Some(db) = args.get(2) else {
                harness_no_tmp("--truncate requires a db path");
            };
            truncate_main(Path::new(db));
        }
        _ => {
            print_usage();
            std::process::exit(EXIT_HARNESS);
        }
    }
}
