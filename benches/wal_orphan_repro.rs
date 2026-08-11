//! Deterministic reproduction of the turso multiprocess-WAL orphaned-WAL
//! defect (class A): an external standard `sqlite3` that opens and closes a
//! live turso database (`multiprocess_wal`) takes an EXCLUSIVE lock, runs a
//! PASSIVE checkpoint, and truncates the on-disk `-wal` to zero bytes (Apple
//! build; vanilla builds unlink the file), stranding the `-tshm` shared index
//! with a stale `max_frame > 0`. Escalation artifact for the turso
//! maintainers (version pin `=0.7.2`, deliberate; see "Version pin").
//!
//! **Experimental-feature disclaimer**: this exercises the experimental
//! `multiprocess_wal` engine feature (opt-in via
//! `experimental_multiprocess_wal(true)`), which the production daemon enables
//! for cross-process WAL. The defect lives in that opt-in path.
//!
//! # Defect
//!
//! With `multiprocess_wal` turso deliberately takes no OS file locks on
//! `-db`/`-wal` (NoLock) — coordination lives entirely in the `-tshm` shared
//! index. An external standard `sqlite3` therefore cannot be stopped: it sees
//! a `-wal` with no `-shm` of its own, takes an EXCLUSIVE lock, performs a
//! PASSIVE checkpoint on close, and truncates the WAL (zero bytes on Apple
//! builds, unlink on vanilla builds) while the `-tshm` keeps its stale
//! `max_frame > 0`. The two lock domains (`-tshm` vs `-shm`) are mutually
//! invisible: sqlite3's `xShmLock` PENDING/CHECKPOINT protocol has no
//! counterpart in turso, so the checkpoint proceeds against live turso
//! readers. There is no upstream fix — issue #6454 is open without a
//! maintainer response (a same-class report: stock-sqlite checkpoint sidecar
//! corrupting live turso readers).
//!
//! Empirically confirmed consequences (3/3 on temp copies):
//! 1) Silent loss of fsynced commits: after the sqlite3 visit, daemon commits
//!    silently disappear on fresh reopen — an empty state is built, no errors
//!    raised.
//! 2) Checkpoint data destruction: after truncation + sparse growth a fresh
//!    connection's `wal_checkpoint` reads zeroed frames and writes zeros into
//!    `-db` ("file is not a database"); the pragma itself looks harmless
//!    (busy=1).
//!
//! # How to run
//!
//! Build via the non-default feature gate (excluded from default
//! builds/tests/benches):
//!
//! ```text
//! cargo bench --no-default-features --features wal-orphan-repro --bench wal_orphan_repro -- --run
//! ```
//!
//! `--run` is the one-shot primary reproduction: fresh multiprocess-wal db in
//! a temp dir, baseline commits, an external sqlite3 visit (its close runs the
//! destructive path), the truncation predicate (on-disk WAL smaller than the
//! baseline while `-tshm` `max_frame` stays stale), a small post-visit commit,
//! and a fresh-reopen durability check. On turso 0.7.2 with the Apple sqlite3
//! the post-visit commit silently vanishes (observed on macOS, sqlite3 3.51.0;
//! exact sizes vary with platform):
//!
//! ```text
//! sqlite3: 3.51.0 2025-06-12 ... (64-bit)
//! baseline: wal=12392 bytes, tshm max_frame=3
//! phase baseline: 4ms
//! visit: wal 12392 -> 0 bytes, tshm max_frame=3
//! orphaned: wal truncated to 0 bytes with stale tshm max_frame=3
//! phase visit: 7ms
//! commit: post-visit row committed
//! phase commit: 1ms
//! durability: 0 post-visit rows on fresh reopen
//! phase durability: 2ms
//! REPRODUCED: post-visit commit silently lost (0 rows on fresh reopen)
//! ```
//!
//! `--torn-pre` runs the same sequence with the disputed precondition first:
//! a child TRUNCATE-checkpoints, spills mid-transaction, and self-aborts,
//! leaving a torn shared index (index entries beyond `max_frame`) before the
//! visit. The mode reports actual behavior rather than asserting an outcome.
//! Observed on turso 0.7.2 / macOS: the torn state (`max_frame=0`,
//! `frame_index_len=4096`) does not trip the two-writer monotonicity assert —
//! the first append lands past the stale entries — so the orphaned-WAL
//! sequence runs and reproduces exactly like `--run` (exit 0).
//!
//! # Exit contract
//!
//! - `0` — bug reproduced: post-visit commits silently lost, or the fresh
//!   reopen failed/panicked on a zeroed or corrupt state.
//! - `1` — clean / fix present: the post-visit commit survived the fresh
//!   reopen.
//! - `2` — harness error: sqlite3 spawn/visit failure, baseline/commit failure,
//!   the visit did not leave a valid orphaned state, or a setup-phase panic.
//! - `3` — skip: sqlite3 binary absent, the sqlite3 build does not truncate
//!   the WAL (behavioral probe of applicability, not build-version based), or
//!   the `.tshm` layout drifted from the documented 0.7.2 offsets.
//!
//! Bare invocation (no arguments) prints this usage and exits `2` — nothing
//! runs silently.
//!
//! # Version pin / upstream status
//!
//! The repo pins `turso = "=0.7.2"` deliberately so the storage layer cannot
//! drift under this artifact. The defect is unfixed upstream: issue #6454
//! (same error class) is open without a maintainer response.
//!
//! # Suggested fix direction
//!
//! Validate the content of WAL frames when classifying authority on reopen:
//! a zeroed or sparse frame 1 must not pass the salt check, forcing a full
//! rebuild instead of accepting the truncated WAL as authoritative.
//!
//! # Fix-landing caveat
//!
//! The `1` outcome only appears when the fix lands in the write path
//! (truncation detected at commit time). A reopen/authority-classification
//! only fix cannot recover the sparse post-visit frame (its header is
//! zeroed), so this bench may keep exiting `0` after such a fix.
//!
//! # Safety / standalone
//!
//! This bench only ever creates fresh disposable databases in a `tempfile`
//! temp dir. The sole path-taking mode is the internal `--build-torn` child
//! sub-arg (mirroring wal_race_repro's `--build-and-abort`): the parent
//! spawns itself against its own temp copy — internal only, never point it at
//! a live store. Every repro run operates on its own temp copy.
//! Self-contained: only the `turso` (`=0.7.2`) and
//! `tempfile` crates plus `std` are used, so a copy of this file in a fresh
//! project (`harness = false`, deps `turso = "=0.7.2"`, `tempfile`) builds and
//! runs unchanged (edition-2021 compatible; no edition-2024-only syntax).
//! Concurrent runs cannot interfere.
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use turso::core::{Database, DatabaseOpts, OpenFlags, PlatformIO};

const SQLITE3_BIN_ENV: &str = "SQLITE3_BIN";
/// The two-writer defect's panic signature (wal_race_repro): the torn
/// precondition can trip it before the orphaned-WAL sequence runs.
const WAL_RACE_SIGNATURE: &str = "shared WAL frame ids must increase monotonically";
/// Padded rows needed to force a completed page-cache spill in the torn-build
/// child (same sizing rationale as wal_race_repro).
const SPILL_ROWS: usize = 5000;

// ── Phase-aware panic classification ──────────────────────────────────────
// Durability-phase failures (zeroed/corrupt db) ARE the reproduced outcome;
// setup-phase failures are harness errors.

const PHASE_SETUP: u8 = 0;
const PHASE_DURABILITY: u8 = 1;
static PHASE: AtomicU8 = AtomicU8::new(PHASE_SETUP);
static TORN_PRE: AtomicBool = AtomicBool::new(false);

// ── Shared DB helpers ─────────────────────────────────────────────────────

fn open_db(db_path: &Path) -> Arc<turso::core::Connection> {
    let io: Arc<dyn turso::core::IO> = Arc::new(PlatformIO::new().expect("platform io"));
    let opts = DatabaseOpts::new()
        .with_multiprocess_wal(true)
        .with_index_method(true);
    let db = Database::open_file_with_flags(
        io,
        db_path.to_str().expect("db path must be UTF-8"),
        OpenFlags::default(),
        opts,
        None,
    )
    .expect("open database");
    let conn = db.connect().expect("connect to database");
    conn.set_busy_timeout(Duration::from_secs(60));
    conn
}

/// Execute `sql`, retrying on `Busy` (write-lock contention is expected after
/// a TRUNCATE). Callers decide whether a non-Busy failure is a harness error
/// (`expect`) or a probe outcome (`match`).
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

/// Run `PRAGMA wal_checkpoint(<mode>)`, returning true when it succeeded
/// (busy == 0). A successful TRUNCATE resets the shared index and starts a
/// fresh WAL generation.
fn wal_checkpoint(conn: &Arc<turso::core::Connection>, mode: &str) -> bool {
    let sql = format!("PRAGMA wal_checkpoint({mode})");
    for _ in 0..10_000 {
        let mut stmt = match conn.query(&sql) {
            Ok(Some(stmt)) => stmt,
            Ok(None) => return false,
            Err(e) if is_busy_error(&e) => {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            Err(e) => panic!("checkpoint query failed: {e}"),
        };
        let rows = match stmt.run_collect_rows() {
            Ok(rows) => rows,
            Err(e) if is_busy_error(&e) => {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            Err(e) => panic!("checkpoint failed: {e}"),
        };
        let busy = rows.first().and_then(|r| r.first()).is_some_and(|v| {
            matches!(
                v,
                turso::core::Value::Numeric(turso::core::Numeric::Integer(n)) if *n != 0
            )
        });
        return !busy;
    }
    panic!("checkpoint stayed busy");
}

/// Insert `count` padded rows (~4 KiB each), forcing every row onto its own
/// page(s) and therefore its own WAL frame(s) — the torn-build spill shape.
fn bulk_insert(conn: &Arc<turso::core::Connection>, count: usize) {
    let pad = "x".repeat(4000);
    let mut remaining = count;
    let mut offset = 0usize;
    while remaining > 0 {
        let n = remaining.min(64);
        let mut sql = String::from("INSERT INTO t (val) VALUES ");
        for i in 0..n {
            if i > 0 {
                sql.push_str(", ");
            }
            sql.push_str(&format!("('bulk-{}-{pad}')", offset + i));
        }
        execute_retry(conn, &sql).expect("bulk insert with busy retry");
        remaining -= n;
        offset += n;
    }
}

/// Single-row scalar query (the durability probe).
fn query_count(conn: &Arc<turso::core::Connection>, sql: &str) -> Result<i64, String> {
    let mut stmt = conn
        .query(sql)
        .map_err(|e| format!("query failed: {e}"))?
        .ok_or_else(|| "query returned no statement".to_string())?;
    let rows = stmt
        .run_collect_rows()
        .map_err(|e| format!("collect rows failed: {e}"))?;
    match rows.first().and_then(|r| r.first()) {
        Some(turso::core::Value::Numeric(turso::core::Numeric::Integer(n))) => Ok(*n),
        other => Err(format!("unexpected scalar result: {other:?}")),
    }
}

// ── Panic hook / cleanup ──────────────────────────────────────────────────

/// Inline replacement for the daemon's panic-payload stringifier, keeping this
/// file free of project imports.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Phase-aware panic hook: a durability-phase panic (zeroed/corrupt db on
/// fresh reopen) is the reproduced outcome — exit 0. Setup-phase panics are
/// harness errors — exit 2 (with a torn-pre interference note when the
/// two-writer signature fires).
fn install_panic_hook(cleanup: Option<PathBuf>) {
    std::panic::set_hook(Box::new(move |info| {
        let msg = panic_message(info.payload());
        eprintln!("PANIC: {msg}");
        if PHASE.load(Ordering::SeqCst) == PHASE_DURABILITY {
            eprintln!("REPRODUCED: fresh reopen panicked (zeroed or corrupt state)");
            if let Some(path) = &cleanup {
                let _ = fs::remove_dir_all(path);
            }
            std::process::exit(0);
        }
        if TORN_PRE.load(Ordering::SeqCst) && msg.contains(WAL_RACE_SIGNATURE) {
            eprintln!(
                "torn-pre: first append tripped the two-writer monotonicity defect (see \
                 wal_race_repro) — the torn precondition interferes before the orphaned-WAL \
                 sequence; orphaned-WAL verdict not reached"
            );
        }
        if let Some(path) = &cleanup {
            let _ = fs::remove_dir_all(path);
        }
        std::process::exit(2);
    }));
}

/// Remove the per-run database and exit with `code`. Exiting via
/// `process::exit` skips destructors, so the temp dir is removed explicitly
/// instead of relying on `TempDir::drop`.
fn exit_after_cleanup(tmp: &tempfile::TempDir, code: i32) -> ! {
    let _ = fs::remove_dir_all(tmp.path());
    std::process::exit(code);
}

// ── .tshm shared-index reader ─────────────────────────────────────────────
// Offsets match the durable `#[repr(C)]` map header of turso_core 0.7.2
// (`storage/shared_wal_coordination.rs`): magic `TSHMWAL\0` at 0, version at
// 8, reader_slot_count at 12, frame_index_capacity at 36, `frame_index_len`
// (u32) at 40, `max_frame` (u64) at 56. The map header is 192 bytes; the
// first frame-index block (`{page_id, frame_id}` u64 pairs) starts at
// `base_mapped_len(64)` = 4096.

const TSHM_MAGIC: &[u8; 8] = b"TSHMWAL\0";
const TSHM_VERSION: u32 = 1;
const TSHM_HEADER_BYTES: u64 = 192;
const TSHM_ENTRY_BYTES: u64 = 16;
const TSHM_MAP_ALIGNMENT: u64 = 4096;
const TSHM_VERSION_OFFSET: u64 = 8;
const TSHM_READER_SLOT_COUNT_OFFSET: u64 = 12;
const TSHM_INDEX_CAPACITY_OFFSET: u64 = 36;
const TSHM_FRAME_INDEX_LEN_OFFSET: u64 = 40;
const TSHM_MAX_FRAME_OFFSET: u64 = 56;

/// Recomputed index base — mirrors turso_core `base_mapped_len`: the map
/// header plus reader bitmap / reader frames / reader owners arrays and two
/// reserved per-reader arrays, 4096-aligned.
fn compute_index_base(reader_slot_count: u32) -> u64 {
    let words = u64::from(reader_slot_count / 64);
    let slots = u64::from(reader_slot_count);
    let raw = TSHM_HEADER_BYTES + 2 * words * 8 + 3 * slots * 8;
    raw.div_ceil(TSHM_MAP_ALIGNMENT) * TSHM_MAP_ALIGNMENT
}

struct TshmHeader {
    index_len: u32,
    max_frame: u64,
}

enum TshmReadError {
    Io(std::io::Error),
    Layout(String),
}

impl std::fmt::Display for TshmReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TshmReadError::Io(e) => write!(f, "{e}"),
            TshmReadError::Layout(detail) => write!(f, "{detail}"),
        }
    }
}

fn read_u32_at(f: &mut fs::File, off: u64) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_at(f: &mut fs::File, off: u64) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read and structurally validate the `.tshm` header: magic, version, reader
/// slot count, recomputed index base, and index-region file coverage.
fn read_tshm_header(tshm_path: &Path) -> Result<TshmHeader, TshmReadError> {
    let mut f = fs::File::open(tshm_path).map_err(TshmReadError::Io)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).map_err(TshmReadError::Io)?;
    if &magic != TSHM_MAGIC {
        return Err(TshmReadError::Layout(format!(
            "bad .tshm magic {:?}, expected TSHMWAL\\0",
            String::from_utf8_lossy(&magic)
        )));
    }
    let version = read_u32_at(&mut f, TSHM_VERSION_OFFSET).map_err(TshmReadError::Io)?;
    if version != TSHM_VERSION {
        return Err(TshmReadError::Layout(format!(
            "unsupported .tshm version {version}, expected {TSHM_VERSION}"
        )));
    }
    let reader_slot_count =
        read_u32_at(&mut f, TSHM_READER_SLOT_COUNT_OFFSET).map_err(TshmReadError::Io)?;
    if reader_slot_count < 64 || reader_slot_count % 64 != 0 {
        return Err(TshmReadError::Layout(format!(
            "invalid reader_slot_count {reader_slot_count}, expected a multiple of 64"
        )));
    }
    let index_capacity =
        read_u32_at(&mut f, TSHM_INDEX_CAPACITY_OFFSET).map_err(TshmReadError::Io)?;
    let index_len = read_u32_at(&mut f, TSHM_FRAME_INDEX_LEN_OFFSET).map_err(TshmReadError::Io)?;
    let max_frame = read_u64_at(&mut f, TSHM_MAX_FRAME_OFFSET).map_err(TshmReadError::Io)?;
    if index_len > index_capacity {
        return Err(TshmReadError::Layout(format!(
            "frame_index_len {index_len} exceeds capacity {index_capacity}"
        )));
    }
    let index_base = compute_index_base(reader_slot_count);
    let file_len = f.metadata().map_err(TshmReadError::Io)?.len();
    if file_len < index_base + u64::from(index_len) * TSHM_ENTRY_BYTES {
        return Err(TshmReadError::Layout(format!(
            "file too small for the advertised index: len {file_len} < base {index_base} + {} entries",
            index_len
        )));
    }
    Ok(TshmHeader {
        index_len,
        max_frame,
    })
}

// ── sqlite3 discovery / WAL size ──────────────────────────────────────────

fn sqlite3_exe_name() -> &'static str {
    #[cfg(windows)]
    {
        "sqlite3.exe"
    }
    #[cfg(not(windows))]
    {
        "sqlite3"
    }
}

/// Discover the external sqlite3 binary: `SQLITE3_BIN` override, then PATH,
/// then common fixed locations.
fn find_sqlite3() -> Option<PathBuf> {
    if let Ok(p) = std::env::var(SQLITE3_BIN_ENV) {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let cand = dir.join(sqlite3_exe_name());
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    for p in [
        "/usr/bin/sqlite3",
        "/usr/local/bin/sqlite3",
        "/opt/homebrew/bin/sqlite3",
    ] {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn wal_size(wal_path: &Path) -> u64 {
    fs::metadata(wal_path).map(|m| m.len()).unwrap_or(0)
}

fn phase_elapsed(t: &mut Instant, label: &str) {
    eprintln!("phase {label}: {}ms", t.elapsed().as_millis());
    *t = Instant::now();
}

// ── Torn-build child (disputed-precondition mode) ─────────────────────────
// Reached only via the internal `--build-torn` sub-arg (the parent spawns
// itself): open the db, commit a baseline, TRUNCATE to a fresh generation,
// then spill mid-transaction and self-abort — leaving a torn shared index
// (index entries beyond max_frame) plus a WAL of un-committed frames.

fn build_torn_main(db_path: &Path) -> ! {
    let conn = open_db(db_path);
    execute_retry(&conn, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("create table");
    execute_retry(&conn, "INSERT INTO t (val) VALUES ('baseline')").expect("baseline commit");
    if !wal_checkpoint(&conn, "TRUNCATE") {
        eprintln!("harness error: TRUNCATE checkpoint stayed busy");
        std::process::exit(2);
    }
    execute_retry(&conn, "BEGIN").expect("begin transaction");
    bulk_insert(&conn, SPILL_ROWS);
    eprintln!("spill complete, aborting before COMMIT (simulated daemon crash)");
    std::process::abort();
}

// ── Primary reproduction ──────────────────────────────────────────────────

fn run_orphan_repro(torn_pre: bool) -> ! {
    let sqlite3 = match find_sqlite3() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no sqlite3 binary in PATH (set {SQLITE3_BIN_ENV} to override)");
            std::process::exit(3);
        }
    };
    TORN_PRE.store(torn_pre, Ordering::SeqCst);
    if let Ok(out) = Command::new(&sqlite3).arg("--version").output() {
        let v = String::from_utf8_lossy(&out.stdout);
        let v = v.trim();
        if !v.is_empty() {
            eprintln!("sqlite3: {v}");
        }
    }
    let tmp = tempfile::TempDir::new().expect("temp dir");
    install_panic_hook(Some(tmp.path().to_path_buf()));
    let db_path = tmp.path().join("orphan.db");
    let wal_path = PathBuf::from(format!("{}-wal", db_path.display()));
    let tshm_path = PathBuf::from(format!("{}-tshm", db_path.display()));
    let mut t = Instant::now();

    if torn_pre {
        // Build the disputed precondition (TRUNCATE + crash-mid-tx) in a
        // child whose self-abort is the daemon-crash shape.
        let exe = std::env::current_exe().expect("current exe");
        let child = Command::new(exe)
            .arg("--build-torn")
            .arg(&db_path)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn torn-build child");
        let output = child.wait_with_output().expect("wait for torn-build child");
        let child_err = String::from_utf8_lossy(&output.stderr);
        let child_err = child_err.trim();
        if !child_err.is_empty() {
            eprintln!("torn-build: {child_err}");
        }
        // Contract: the child self-aborts via SIGABRT. Any explicit exit code
        // (0 clean, 2 checkpoint-busy, 101 panic) means the crash shape broke.
        if let Some(code) = output.status.code() {
            eprintln!("harness error: torn-build child exited with code {code} (expected SIGABRT)");
            exit_after_cleanup(&tmp, 2);
        }
        let header = match read_tshm_header(&tshm_path) {
            Ok(h) => h,
            Err(TshmReadError::Io(e)) => {
                eprintln!("harness error: cannot read shared index after torn build: {e}");
                exit_after_cleanup(&tmp, 2);
            }
            Err(TshmReadError::Layout(detail)) => {
                eprintln!("tshm layout mismatch: {detail}");
                exit_after_cleanup(&tmp, 3);
            }
        };
        if u64::from(header.index_len) <= header.max_frame {
            eprintln!(
                "harness error: torn build left max_frame={} frame_index_len={} — no torn state",
                header.max_frame, header.index_len
            );
            exit_after_cleanup(&tmp, 2);
        }
        eprintln!(
            "torn-pre: child aborted mid-tx, torn state max_frame={} frame_index_len={}",
            header.max_frame, header.index_len
        );
        phase_elapsed(&mut t, "torn-pre");
    }

    // Live connection — HELD OPEN through the visit. Dropping it before the
    // visit reproduces no loss (empirical) and would false-negative; turso's
    // in-process registry also makes a second open reuse the live Database,
    // so the durability check below must reopen only after this drops.
    let live = open_db(&db_path);
    execute_retry(
        &live,
        "CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY, val TEXT)",
    )
    .expect("create table");
    execute_retry(
        &live,
        "INSERT INTO t (val) VALUES ('baseline-0'),('baseline-1'),('baseline-2')",
    )
    .expect("baseline commit");

    // Baseline must have produced a non-empty WAL with a stale-able max_frame.
    let pre_wal = wal_size(&wal_path);
    let pre_tshm = match read_tshm_header(&tshm_path) {
        Ok(h) => h,
        Err(TshmReadError::Io(e)) => {
            eprintln!("harness error: cannot read shared index: {e}");
            exit_after_cleanup(&tmp, 2);
        }
        Err(TshmReadError::Layout(detail)) => {
            eprintln!("tshm layout mismatch: {detail}");
            exit_after_cleanup(&tmp, 3);
        }
    };
    if pre_wal == 0 {
        eprintln!("harness error: baseline left an empty WAL (size 0)");
        exit_after_cleanup(&tmp, 2);
    }
    if pre_tshm.max_frame == 0 {
        eprintln!("harness error: baseline left max_frame=0 in the shared index");
        exit_after_cleanup(&tmp, 2);
    }
    eprintln!(
        "baseline: wal={pre_wal} bytes, tshm max_frame={}",
        pre_tshm.max_frame
    );
    phase_elapsed(&mut t, "baseline");

    // Visit: external sqlite3 opens the live db and closes it — the close
    // takes an EXCLUSIVE lock, PASSIVE-checkpoints, and truncates the WAL
    // (Apple builds) or unlinks it (vanilla builds).
    let visit = Command::new(&sqlite3)
        .arg("-batch")
        .arg(&db_path)
        .arg("SELECT count(*) FROM t")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .expect("run sqlite3 visit");
    let visit_err = String::from_utf8_lossy(&visit.stderr);
    let visit_err = visit_err.trim();
    if !visit_err.is_empty() {
        eprintln!("visit stderr: {visit_err}");
    }
    if !visit.status.success() {
        eprintln!(
            "harness error: sqlite3 visit failed (exit {:?}) — a failed open cannot truncate \
             the WAL",
            visit.status.code()
        );
        exit_after_cleanup(&tmp, 2);
    }

    let post_wal = wal_size(&wal_path);
    let post_tshm = match read_tshm_header(&tshm_path) {
        Ok(h) => h,
        Err(TshmReadError::Io(e)) => {
            eprintln!("harness error: cannot read shared index after visit: {e}");
            exit_after_cleanup(&tmp, 2);
        }
        Err(TshmReadError::Layout(detail)) => {
            eprintln!("tshm layout mismatch: {detail}");
            exit_after_cleanup(&tmp, 3);
        }
    };
    eprintln!(
        "visit: wal {pre_wal} -> {post_wal} bytes, tshm max_frame={}",
        post_tshm.max_frame
    );
    // Behavioral applicability probe (not build-version based): the visit must
    // have truncated the WAL while the shared index keeps a stale max_frame.
    if post_wal >= pre_wal {
        eprintln!(
            "SKIP: sqlite3 build did not truncate the WAL (pre={pre_wal} post={post_wal}) — \
             non-applicable build"
        );
        exit_after_cleanup(&tmp, 3);
    }
    if post_tshm.max_frame == 0 {
        eprintln!("SKIP: shared index max_frame reset to 0 — no orphaned state left behind");
        exit_after_cleanup(&tmp, 3);
    }
    eprintln!(
        "orphaned: wal truncated to {post_wal} bytes with stale tshm max_frame={}",
        post_tshm.max_frame
    );
    phase_elapsed(&mut t, "visit");

    // Post-visit commit — small: a spill-sized commit would trigger turso's
    // write-path self-heal (full WAL rebuild) and lose nothing (0/4).
    if let Err(e) = execute_retry(&live, "INSERT INTO t (val) VALUES ('post-visit-0')") {
        eprintln!("harness error: post-visit commit failed: {e}");
        exit_after_cleanup(&tmp, 2);
    }
    eprintln!("commit: post-visit row committed");
    phase_elapsed(&mut t, "commit");
    drop(live);

    // Durability: fresh cold-cache connection (the committing connection is
    // gone, so the open re-reads the on-disk state). Only the post-visit rows
    // count — sqlite3's checkpoint legitimately folded the baseline into the
    // db, and a fresh append can reuse a lost row's id, so ids are not
    // compared.
    PHASE.store(PHASE_DURABILITY, Ordering::SeqCst);
    let fresh = open_db(&db_path); // zeroed/corrupt db panics here -> hook exits 0
    let count = match query_count(
        &fresh,
        "SELECT count(*) FROM t WHERE val LIKE 'post-visit%'",
    ) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("REPRODUCED: fresh reopen query failed (corrupt state): {e}");
            exit_after_cleanup(&tmp, 0);
        }
    };
    eprintln!("durability: {count} post-visit rows on fresh reopen");
    phase_elapsed(&mut t, "durability");
    if count > 0 {
        eprintln!("CLEAN: post-visit commit survived the fresh reopen — fix present");
        exit_after_cleanup(&tmp, 1);
    }
    eprintln!("REPRODUCED: post-visit commit silently lost (0 rows on fresh reopen)");
    exit_after_cleanup(&tmp, 0);
}

// ── Dispatch ──────────────────────────────────────────────────────────────

fn print_usage() {
    eprintln!(
        "usage: wal_orphan_repro <mode>

modes:
  --run        primary repro: baseline, external sqlite3 visit, post-visit
               commit, fresh-reopen durability check (fresh temp database)
  --torn-pre   primary repro with the disputed precondition first: a child
               TRUNCATE-checkpoints, spills mid-transaction and self-aborts,
               leaving a torn shared index before the visit
  --build-torn DB  internal child mode (parent-spawned only): open, baseline,
               TRUNCATE, spill mid-transaction, self-abort

exit codes:
  0  bug reproduced (post-visit commit silently lost / db zeroed)
  1  clean / fix present (post-visit commit survived the fresh reopen)
  2  harness error
  3  skip (sqlite3 absent or non-truncating build, or .tshm layout drift)"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--run") => run_orphan_repro(false),
        Some("--torn-pre") => run_orphan_repro(true),
        Some("--build-torn") => {
            let Some(path) = args.get(2) else {
                eprintln!("harness error: --build-torn requires a db path");
                std::process::exit(2);
            };
            build_torn_main(Path::new(path));
        }
        _ => {
            print_usage();
            std::process::exit(2);
        }
    }
}
