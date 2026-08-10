//! Deterministic reproduction of the turso multiprocess-WAL crash: a hard
//! `assert!` panic — `shared WAL frame ids must increase monotonically` — on
//! the first append after a crash mid-transaction. Escalation artifact for
//! the turso maintainers (version pin `=0.7.2`, deliberate; see "Version pin").
//!
//! **Experimental-feature disclaimer**: this exercises the experimental
//! `multiprocess_wal` engine feature (opt-in via
//! `experimental_multiprocess_wal(true)`), which the production daemon enables
//! for cross-process WAL. The defect lives in that opt-in path.
//!
//! # Defect
//!
//! turso_core's `.tshm` reopen path accepts the persisted shared frame index
//! as trusted while validating only WAL-file facts and index structure; its
//! index-content checks are bounded to `frame_id <= max_frame`, so the
//! un-published region beyond `max_frame` is never validated against the WAL
//! generation. A crash mid-transaction leaves un-published frame entries in
//! the mmap'd index: every spilled page is recorded as a `(page_id, frame_id)`
//! entry with `frame_index_len` bumped, but `max_frame` only advances at
//! COMMIT, so all entries beyond `max_frame` sit in the index unseen. The
//! first fresh append after reopen derives its next frame id from the
//! coordination snapshot's `max_frame` (`max_frame + 1` — the durable `.tshm`
//! header on the Trusted path, the rebuilt in-process WAL state on the
//! RebuildFromDisk path) and lands on a stale slot, tripping a plain `assert!`
//! that, under panic=unwind on a tokio worker thread, incapacitates the
//! process (stuck writer byte-lock, dead workers) instead of rebuilding the
//! index from the WAL:
//!
//! ```text
//! shared WAL frame ids must increase monotonically: new_frame_id={},
//! previous_frame_id={}, slot={}, shared_max_frame={}
//! ```
//!
//! (Quote the message text, not a file:line: the assert lives in
//! `record_frame`, which is `#[track_caller]`, so production panic reports
//! name the frame-caching call site — observed `turso_core-0.7.2/storage/
//! wal.rs:2357:24` — not the assert site (`shared_wal_coordination.rs:2317`
//! in 0.7.2), and those line numbers shift between versions, so the message
//! text is the only stable identifier.)
//!
//! Both reopen outcomes end in the same panic (in production: process
//! incapacitation; in this bench: a deterministic exit-code abort via the
//! panic hook):
//! - Trusted — the WAL facts match `max_frame`: the persisted index is kept
//!   as-is.
//! - RebuildFromDisk — the WAL facts mismatch: only the in-process WAL state
//!   is rebuilt from disk (`sqlite3_ondisk::build_shared_wal`); the durable
//!   shared index is left untouched.
//!
//! In either case the stale tail (entries with frame ids > `max_frame`) is
//! never inspected on reopen, and the first append panics with
//! `new_frame_id = max_frame + 1`, `previous_frame_id == slot == stale index
//! length` (observed shape).
//!
//! # Production consequence
//!
//! The production daemon hit this assert twice, each time becoming
//! incapacitated (stuck writer byte-lock, dead tokio workers) rather than
//! aborting:
//! - 2026-08-06 on turso 0.7.0 (ordinary pipeline dispatch).
//! - 2026-08-07 on turso 0.7.2 — a fatal outage on the order of tens of
//!   minutes (approximate: not independently verifiable from retained data,
//!   consistent with the surviving chat-history gap).
//!
//! These are NOT self-update handoffs: the update log is merely stderr
//! capture, the single-instance file lock forbids a second daemon, and both
//! incidents ran as the single daemon (Aug 7 with two tokio workers).
//! Read-side pager panics observed in the same incident window are consistent
//! with this defect (readers resolving pages through the stale index can hit
//! related invariants), but were not proven causally linked.
//!
//! # How to run
//!
//! Build via the non-default feature gate (excluded from default
//! builds/tests/benches):
//!
//! ```text
//! cargo bench --no-default-features --features wal-repro --bench wal_race_repro -- --natural
//! ```
//!
//! `--natural` is the one-shot primary reproduction: it spawns a child that
//! opens the database with the multiprocess-WAL option, commits a small
//! baseline, begins a transaction, inserts padded rows until the page cache
//! spills to the WAL, and self-aborts before COMMIT; the parent then reopens
//! the database and inserts one row. On turso 0.7.2 the insert hard-panics:
//!
//! ```text
//! build-and-abort: spill complete, aborting before COMMIT (simulated daemon crash)
//! torn state: shared max_frame=3, frame_index_len=4099 (4096 un-published entries)
//! PANIC: shared WAL frame ids must increase monotonically: new_frame_id=4, previous_frame_id=4099, slot=4099, shared_max_frame=3
//! ```
//!
//! (Values are live — exact frame counts vary with platform and page-cache
//! sizing; the shape is what matters: `new_frame_id = max_frame + 1`,
//! `previous_frame_id == slot == stale index length`.)
//!
//! The same run is available as a documented two-step sequence:
//!
//! ```text
//! cargo bench --no-default-features --features wal-repro --bench wal_race_repro -- --build-and-abort /tmp/wal-repro.db
//! cargo bench --no-default-features --features wal-repro --bench wal_race_repro -- --reopen-and-append /tmp/wal-repro.db
//! ```
//!
//! The secondary mode (`--probe-tshm`) manufactures the stale tail by patching
//! a freshly checkpointed `.tshm` directly (fault injection), demonstrating
//! the reopen-acceptance gap in isolation — no crash needed:
//!
//! ```text
//! cargo bench --no-default-features --features wal-repro --bench wal_race_repro -- --probe-tshm
//! ```
//!
//! # Exit contract
//!
//! - `0` — bug reproduced: the monotonicity panic fired and its signature was
//!   printed live.
//! - `1` — clean / fix present: the reopen+append succeeded without the panic
//!   (a fixed build rejects or heals the stale index on reopen). The message
//!   is deliberately fix-agnostic.
//! - `2` — harness error: missing argument, failed open/insert, the
//!   build-and-abort phase did not leave the torn index state, or a
//!   non-signature panic.
//! - `3` — structural `.tshm` layout mismatch (probe only): magic, version,
//!   slot count, recomputed index base, or the fresh-state invariant
//!   (`frame_index_len == max_frame`) deviated before patching. Reported
//!   loudly — never a silent false negative or a false clean.
//!
//! Bare invocation (no arguments) prints this usage and exits `2` — nothing
//! runs silently.
//!
//! # Version pin / upstream status
//!
//! The repo pins `turso = "=0.7.2"` deliberately so the storage layer cannot
//! drift under this artifact. The defect is unfixed upstream: `main` and
//! `0.8.0-pre.3` both carry the same assert, and no public issue or PR exists
//! for this specific defect — this artifact is the first report. Two adjacent
//! upstream changes in the same code area fix different bugs and do not cover
//! this one: PR #7674 (WAL spill frame-slot reuse — merged, present in 0.7.2)
//! and a separate proposed fix for a stale disk-scan clearing the shared
//! frame index (not merged; absent from 0.7.2, `0.8.0-pre.3`, and `main`).
//!
//! # Suggested fix
//!
//! Validate the persisted index contents against the WAL generation on the
//! trusted reopen path: after the WAL-facts check, trim or rebuild the index
//! when its length/entries exceed `max_frame` — the existing
//! `rollback_frames(max_frame)` stale-tail trim primitive is directly
//! reusable — instead of keeping un-published entries that the next append
//! collides with.
//!
//! # Escalation framing
//!
//! This is a rigorous engineering bug report — issue, deterministic
//! reproduction, and a suggested fix — not a bounty claim (Turso retired its
//! bug-bounty program on 2026-05-12).
//!
//! # Standalone
//!
//! Self-contained: only the `turso` (`=0.7.2`) and `tempfile` crates plus
//! `std` are used, so a copy of this file in a fresh project (`harness =
//! false`, deps `turso = "=0.7.2"`, `tempfile`) builds and runs unchanged.
//! Each run uses a fresh disposable database; concurrent runs cannot
//! interfere.
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use turso::core::{Database, DatabaseOpts, OpenFlags, PlatformIO};

const PANIC_SIGNATURE: &str = "shared WAL frame ids must increase monotonically";
/// Padded rows needed to force a completed page-cache spill: the default cache
/// is 2000 pages with a 90% spill threshold, so 5000 one-page rows guarantee a
/// spill on every native platform.
const SPILL_ROWS: usize = 5000;
/// Size of the stale tail manufactured by the tshm probe (~1000 entries, the
/// production panic shape).
const STALE_TAIL_ENTRIES: u32 = 1000;

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
    // Match the daemon's connection setup (1-minute busy timeout).
    conn.set_busy_timeout(Duration::from_secs(60));
    conn
}

/// Execute `sql`, retrying on `Busy` (write-lock contention is expected in the
/// probe after a TRUNCATE). Callers decide whether a non-Busy failure is a
/// harness error (`expect`) or a probe outcome (`match`).
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

/// Insert `count` rows. `padded` rows (~4 KiB each) force every row onto its
/// own page(s) and therefore its own WAL frame(s); small rows share a page and
/// collapse into a single frame, undersizing frame accounting. Padded batches
/// are chunked so a single statement stays far below any parser size limit.
fn bulk_insert(conn: &Arc<turso::core::Connection>, count: usize, padded: bool) {
    let pad = if padded {
        "x".repeat(4000)
    } else {
        String::new()
    };
    let mut remaining = count;
    let mut offset = 0usize;
    while remaining > 0 {
        let n = remaining.min(if padded { 64 } else { count });
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

/// Install the reproduction panic hook: print the panic message (the live
/// assert signature), clean up, and exit 0 on the reproduced signature / 2
/// otherwise. `cleanup` removes a temp dir (skipped for standalone phases
/// operating on user-supplied paths).
fn install_panic_hook(cleanup: Option<PathBuf>) {
    std::panic::set_hook(Box::new(move |info| {
        let msg = panic_message(info.payload());
        eprintln!("PANIC: {msg}");
        if let Some(path) = &cleanup {
            let _ = fs::remove_dir_all(path);
        }
        std::process::exit(if msg.contains(PANIC_SIGNATURE) { 0 } else { 2 });
    }));
}

/// Remove the per-run database and exit with `code`. Exiting via
/// `process::exit` skips destructors, so the temp dir is removed explicitly
/// instead of relying on `TempDir::drop`.
fn exit_after_cleanup(tmp: &tempfile::TempDir, code: i32) -> ! {
    let _ = fs::remove_dir_all(tmp.path());
    std::process::exit(code);
}

// ── Primary mode: natural crash-mid-spill reproduction ────────────────────

/// Phase 1 (child role): open the DB with the multiprocess-WAL option, commit
/// a small baseline, begin a transaction, insert padded rows until the page
/// cache spills to the WAL, then self-abort before COMMIT — the daemon crash
/// shape. The torn on-disk state (shared index entries beyond `max_frame`) is
/// what phase 2 trips over.
fn build_and_abort_main(db_path: &Path) -> ! {
    install_panic_hook(None);
    let conn = open_db(db_path);
    execute_retry(&conn, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("create table");
    execute_retry(&conn, "INSERT INTO t (val) VALUES ('baseline')").expect("baseline commit");
    execute_retry(&conn, "BEGIN").expect("begin transaction");
    bulk_insert(&conn, SPILL_ROWS, true);
    eprintln!("spill complete, aborting before COMMIT (simulated daemon crash)");
    // Self-abort: kills the process without running destructors, exactly like
    // the production crash that left the torn index behind.
    std::process::abort();
}

/// Phase 2: reopen the database and insert one row. On turso 0.7.2 the insert
/// hard-panics with the monotonicity assert (exit 0 via the hook); a fixed
/// build rejects or heals the stale index and the append succeeds (exit 1).
fn reopen_and_append_main(db_path: &Path) -> ! {
    install_panic_hook(None);
    let conn = open_db(db_path);
    match execute_retry(&conn, "INSERT INTO t (val) VALUES ('after-crash')") {
        Ok(()) => {
            eprintln!(
                "CLEAN: reopen+append succeeded without the monotonicity panic — the stale \
                 shared frame index was rejected or healed on reopen (fix present)"
            );
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("harness error: reopen append failed: {e}");
            std::process::exit(2);
        }
    }
}

/// One-shot primary reproduction: build-and-abort child, verify the torn
/// index state, then reopen-and-append in this process.
fn run_natural_main() -> ! {
    let tmp = tempfile::TempDir::new().expect("temp dir for natural repro");
    install_panic_hook(Some(tmp.path().to_path_buf()));
    let db_path = tmp.path().join("repro.db");
    let tshm_path = format!("{}-tshm", db_path.display());

    // Phase 1 — build-and-abort (child self-aborts mid-transaction).
    let exe = std::env::current_exe().expect("current exe");
    let child = Command::new(exe)
        .arg("--build-and-abort")
        .arg(&db_path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn build-and-abort child");
    let output = child
        .wait_with_output()
        .expect("wait for build-and-abort child");
    let child_stderr = String::from_utf8_lossy(&output.stderr);
    let child_stderr = child_stderr.trim();
    if !child_stderr.is_empty() {
        eprintln!("build-and-abort: {child_stderr}");
    }
    if output.status.success() {
        eprintln!(
            "harness error: build-and-abort exited cleanly — the mid-transaction crash did not fire"
        );
        exit_after_cleanup(&tmp, 2);
    }

    // Verify the crash left the torn state: index entries beyond max_frame.
    let header = match read_tshm_header(Path::new(&tshm_path)) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("harness error: cannot read torn shared index: {e}");
            exit_after_cleanup(&tmp, 2);
        }
    };
    if u64::from(header.index_len) <= header.max_frame {
        eprintln!(
            "harness error: build phase left max_frame={} frame_index_len={} — no un-published \
             entries; the spill did not complete",
            header.max_frame, header.index_len
        );
        exit_after_cleanup(&tmp, 2);
    }
    eprintln!(
        "torn state: shared max_frame={}, frame_index_len={} ({} un-published entries)",
        header.max_frame,
        header.index_len,
        u64::from(header.index_len) - header.max_frame
    );

    // Phase 2 — reopen and append; the first insert must hard-panic.
    let conn = open_db(&db_path);
    match execute_retry(&conn, "INSERT INTO t (val) VALUES ('after-crash')") {
        Ok(()) => {
            eprintln!(
                "CLEAN: reopen+append succeeded without the monotonicity panic — the stale \
                 shared frame index was rejected or healed on reopen (fix present)"
            );
            exit_after_cleanup(&tmp, 1);
        }
        Err(e) => {
            eprintln!("harness error: reopen append failed: {e}");
            exit_after_cleanup(&tmp, 2);
        }
    }
}

// ── Secondary mode: tshm fault-injection probe ────────────────────────────

/// Offsets into the `.tshm` file, matching the durable `#[repr(C)]` map header
/// of turso_core 0.7.2 (`storage/shared_wal_coordination.rs`): magic
/// `TSHMWAL\0` at 0, version at 8, reader_slot_count at 12,
/// frame_index_capacity at 36, `frame_index_len` (u32) at 40, `max_frame`
/// (u64) at 56. The map header is 192 bytes; the first frame-index block
/// (`{page_id, frame_id}` u64 pairs) starts at `base_mapped_len(64)` = 4096.
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
    index_base: u64,
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
        index_base,
    })
}

/// Append `stale_len` entries to the shared frame index and bump
/// `frame_index_len`, leaving `max_frame` at its current (small) value. The
/// stale entries are a structurally plausible crashed-transaction tail:
/// contiguous frame ids just above `max_frame`, each with an arbitrary page
/// id — all derived live from the observed header.
fn patch_tshm_stale_tail(
    header: &TshmHeader,
    tshm_path: &Path,
    stale_len: u32,
) -> std::io::Result<()> {
    let mut f = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(tshm_path)?;
    let start = header.index_base + u64::from(header.index_len) * TSHM_ENTRY_BYTES;
    for i in 0..stale_len {
        let off = start + u64::from(i) * TSHM_ENTRY_BYTES;
        f.seek(SeekFrom::Start(off))?;
        f.write_all(&u64::from(i + 1).to_le_bytes())?; // page_id
        f.write_all(&(header.max_frame + u64::from(i) + 1).to_le_bytes())?; // frame_id
    }
    f.seek(SeekFrom::Start(TSHM_FRAME_INDEX_LEN_OFFSET))?;
    f.write_all(&(header.index_len + stale_len).to_le_bytes())?;
    f.sync_all()?;
    Ok(())
}

/// Secondary reproduction (fault injection): build an old generation, TRUNCATE
/// to a fresh one, append a small committed tail, patch a stale tail into the
/// `.tshm` on disk, then reopen and append. The Trusted reopen classification
/// keeps the patched index, so the next append (id = max_frame + 1) must trip
/// the monotonicity assert — or be handled, equally informative for the
/// upstream report.
fn probe_tshm_main() -> ! {
    let tmp = tempfile::TempDir::new().expect("temp dir for tshm probe");
    install_panic_hook(Some(tmp.path().to_path_buf()));
    let db_path = tmp.path().join("probe.db");
    let tshm_path = format!("{}-tshm", db_path.display());

    // Old generation, then a TRUNCATE to a fresh generation with a small
    // committed tail — the freshly checkpointed state the probe patches.
    let conn_a = open_db(&db_path);
    execute_retry(&conn_a, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("probe create table");
    bulk_insert(&conn_a, 200, true);
    if !wal_checkpoint(&conn_a, "TRUNCATE") {
        eprintln!("harness error: TRUNCATE checkpoint stayed busy");
        exit_after_cleanup(&tmp, 2);
    }
    drop(conn_a);
    let conn_b = open_db(&db_path);
    bulk_insert(&conn_b, 12, true);
    drop(conn_b);

    // All connections dropped → tshm unmapped. Validate the layout before
    // patching; exit 3 on structural deviations, never a silent false clean.
    let header = match read_tshm_header(Path::new(&tshm_path)) {
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
    if u64::from(header.index_len) != header.max_frame {
        eprintln!(
            "tshm layout mismatch: freshly checkpointed index must have frame_index_len == \
             max_frame, got len={} max_frame={}",
            header.index_len, header.max_frame
        );
        exit_after_cleanup(&tmp, 3);
    }
    if let Err(e) = patch_tshm_stale_tail(&header, Path::new(&tshm_path), STALE_TAIL_ENTRIES) {
        eprintln!("harness error: patch failed: {e}");
        exit_after_cleanup(&tmp, 2);
    }
    eprintln!(
        "patched stale tail: max_frame={}, frame_index_len {} -> {} ({} un-published entries)",
        header.max_frame,
        header.index_len,
        header.index_len + STALE_TAIL_ENTRIES,
        STALE_TAIL_ENTRIES
    );

    // Reopen: the Trusted classification keeps the patched index; the next
    // append computes frame_id = max_frame + 1 and must panic on the stale
    // slot — or be handled (equally informative for the upstream report).
    let conn_c = open_db(&db_path);
    match execute_retry(&conn_c, "INSERT INTO t (val) VALUES ('after-crash')") {
        Ok(()) => {
            eprintln!(
                "CLEAN: patched stale index accepted fresh appends without the monotonicity \
                 panic — the index is rejected or healed on reopen (fix present)"
            );
            exit_after_cleanup(&tmp, 1);
        }
        Err(e) => {
            eprintln!("harness error: probe append failed: {e}");
            exit_after_cleanup(&tmp, 2);
        }
    }
}

// ── Dispatch ──────────────────────────────────────────────────────────────

fn print_usage() {
    eprintln!(
        "usage: wal_race_repro <mode> [db_path]

modes:
  --natural               one-shot primary repro: build-and-abort child, then
                          reopen and append (fresh temp database)
  --build-and-abort DB    phase 1: open, baseline, BEGIN, spill, self-abort
                          before COMMIT
  --reopen-and-append DB  phase 2: reopen DB and insert (panics on turso 0.7.2)
  --probe-tshm            secondary repro: patch a stale tail into a freshly
                          checkpointed .tshm, reopen and append

exit codes:
  0  bug reproduced (panic signature printed live)
  1  clean / fix present (no monotonicity panic)
  2  harness error
  3  structural .tshm layout mismatch (probe only)"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--natural") => run_natural_main(),
        Some("--build-and-abort") => {
            let Some(path) = args.get(2) else {
                eprintln!("harness error: --build-and-abort requires a db path");
                std::process::exit(2);
            };
            build_and_abort_main(Path::new(path));
        }
        Some("--reopen-and-append") => {
            let Some(path) = args.get(2) else {
                eprintln!("harness error: --reopen-and-append requires a db path");
                std::process::exit(2);
            };
            reopen_and_append_main(Path::new(path));
        }
        Some("--probe-tshm") => probe_tshm_main(),
        _ => {
            print_usage();
            std::process::exit(2);
        }
    }
}
