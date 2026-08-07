//! Cross-process reproduction harness for the two-writer WAL checkpoint race.
//!
//! The corruption vector: a TRUNCATE/RESTART checkpoint starts a new WAL
//! generation (resets the shared max_frame, bumps checkpoint_seq, clears the
//! shared frame index) while a connection holding a stale frame position can
//! still append into an old frame-index slot, panicking with "shared WAL frame
//! ids must increase monotonically". The daemon hit this when a self-update
//! handoff ran a TRUNCATE checkpoint while a second instance was already
//! writing to the same store.
//!
//! The harness plays both roles from a single executable (no `CARGO_BIN_EXE`).
//! Modes (test databases only):
//!
//! - Default race loop: the parent alternates TRUNCATE/RESTART checkpoints and
//!   reopens a fresh connection after each successful one (the post-restart
//!   reseed/reconcile path) while the child writes bulk transactions.
//! - `--chaos`: the parent SIGKILLs the child mid-write (crash / ENOSPC-adjacent
//!   writer death) and respawns it between checkpoints.
//! - `--probe-stale-tail`: deterministic RESTART -> fresh tail -> reopen
//!   construction, replaying the disk-scan reseed against the fresh generation.
//! - `--probe-tshm-patch`: fault injection — directly appends old-generation
//!   frame ids into the shared `.tshm` index (test db only), then reopens and
//!   appends fresh, synthesizing the stale-tail state without a race.
//!
//! # Result (turso_core 0.7.2, pinned build)
//!
//! The natural race is NOT reproducible through the public API (documented
//! negative result): the reopen reconcile guards with checkpoint_seq/salt
//! generation checks, the disk scan rejects salt-mismatched old frames, and
//! `record_frame` resets the index on the fresh generation's frame 1 — long
//! runs (180s race, 200s chaos, 15 stale-tail rounds) stayed clean. The one
//! unguarded path is the reopen Trusted classification, which keeps the
//! persisted index without validating its contents against the WAL generation:
//! `--probe-tshm-patch` proves it by deterministically tripping the exact
//! assert — `shared WAL frame ids must increase monotonically: new_frame_id=16,
//! previous_frame_id=1000, slot=27, shared_max_frame=15` — a fresh writer that
//! computed its id from the fresh header (16 = 15+1) facing an old-generation
//! index tail. The probe is fault injection, not a failing interleaving; hand
//! it to the Turso developers as the reproduction of the missing
//! index-generation validation on reopen.
//!
//! # Exit codes
//!
//! - `0`: panic signature reproduced (probe or observed race)
//! - `1`: not reproduced within the run window (stochastic — increase
//!   `MAHBOT_WAL_RACE_DURATION_SECS`)
//! - `2`: harness error
//!
//! Run with: `cargo bench --no-default-features --features wal-repro --bench wal_race_repro`
//! (`-- --probe-tshm-patch`, `-- --probe-stale-tail`, or `-- --chaos` select
//! the other modes).
//!
//! Mahbot-independent by design: only `turso` and `tempfile` are used, so a
//! copy of this file in a fresh project (deps `turso = "=0.7.2"`, `tempfile`;
//! `harness = false`) builds and runs unchanged — the escalation artifact.
//! Each run uses a fresh disposable database, so no single-instance lock is
//! needed; concurrent runs cannot interfere.
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use turso::core::{Database, DatabaseOpts, OpenFlags, PlatformIO};

const PANIC_SIGNATURE: &str = "shared WAL frame ids must increase monotonically";
const DEFAULT_RUN_SECS: u64 = 15;

// ── Shared DB helpers ─────────────────────────────────────────────────────

fn open_race_connection(db_path: &Path) -> Arc<turso::core::Connection> {
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
    .expect("open race database");
    let conn = db.connect().expect("connect to race database");
    // Match the daemon's connection setup (1-minute busy timeout).
    conn.set_busy_timeout(Duration::from_secs(60));
    conn
}

/// Execute `sql`, retrying on `Busy` (write-lock contention between the two
/// writers is expected — the race needs both writers active, not one blocked).
/// Callers decide whether a non-Busy failure is a harness error (`expect`) or a
/// probe outcome (`match`).
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
/// (busy == 0). A successful checkpoint resets the shared WAL frame index;
/// RESTART additionally starts a new generation (max_frame=0, checkpoint_seq+1)
/// while leaving the old frames on disk — the state that makes a later
/// disk-scan reopen replay stale entries into a fresh generation.
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

// ── Child: the "stale" second writer ──────────────────────────────────────

/// Child writer. `cycle` makes it drop and reopen its connection every 40
/// passes so the reopen reseed/reconcile path also runs in the second process;
/// otherwise it holds one connection open across the parent's checkpoints
/// (the stale-holder shape). The parent kills the child when the run ends.
fn child_main(db_dir: &Path, cycle: bool) {
    let db_path = db_dir.join("race.db");
    loop {
        let conn = open_race_connection(&db_path);
        // cycle=false: the bound is unreachable before the parent kills us.
        let passes = if cycle { 40 } else { u64::MAX };
        for _ in 0..passes {
            // Padded batches keep this writer's frame ids far ahead of the
            // parent's fresh-generation ids, widening any stale-tail window.
            bulk_insert(&conn, 24, true);
            std::thread::sleep(Duration::from_millis(1));
        }
        // cycle=true: drop conn at end of pass, reopen on the next pass.
    }
}

// ── Parent: TRUNCATE-checkpointing writer ─────────────────────────────────

/// Remove the per-run database and exit with `code`. The parent terminates via
/// `process::exit`, which skips destructors, so the temp dir is removed
/// explicitly instead of relying on `TempDir::drop`.
fn exit_after_cleanup(tmp: &tempfile::TempDir, code: i32) -> ! {
    let _ = fs::remove_dir_all(tmp.path());
    std::process::exit(code);
}

/// One race-loop iteration's checkpoint cadence: every 5th insert runs a
/// TRUNCATE/RESTART checkpoint, and every 3rd successful checkpoint reopens a
/// fresh connection (exercising the post-restart reseed/reconcile path).
fn checkpoint_cadence(
    conn: &mut Arc<turso::core::Connection>,
    db_path: &Path,
    inserts: u64,
    checkpoints: &mut u64,
    successful: &mut u64,
    reopens: &mut u64,
) {
    if !inserts.is_multiple_of(5) {
        return;
    }
    *checkpoints += 1;
    let mode = if inserts.is_multiple_of(20) {
        "RESTART"
    } else {
        "TRUNCATE"
    };
    if wal_checkpoint(conn, mode) {
        *successful += 1;
        if successful.is_multiple_of(3) {
            *conn = open_race_connection(db_path);
            execute_retry(conn, "INSERT INTO t (val) VALUES ('reopen')").expect("reopen insert");
            *reopens += 1;
        }
    }
}

/// Shared race loop for the default and chaos modes. `chaos` kills and
/// respawns the child mid-write (SIGKILL crash / ENOSPC-adjacent writer
/// death); otherwise the child cycles its own connection while the parent
/// checkpoints against it — the handoff shape that makes a stale connection's
/// next append trip the monotonicity panic. Exits 0 on the signature, 1 on a
/// clean window, 2 on a harness error.
fn run_race(chaos: bool) -> ! {
    let run_secs: u64 = std::env::var("MAHBOT_WAL_RACE_DURATION_SECS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_RUN_SECS);
    if run_secs == 0 {
        eprintln!("harness error: MAHBOT_WAL_RACE_DURATION_SECS must be >= 1");
        std::process::exit(2);
    }

    let tmp = tempfile::TempDir::new().expect("temp dir for race database");
    let db_path = tmp.path().join("race.db");

    let conn = open_race_connection(&db_path);
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("create race table");

    // Spawn the child writer before any checkpointing so it holds its
    // connection open across the parent's checkpoints. `Option` so the chaos
    // kill/respawn can move the child through `take()` without losing the
    // outer handle.
    let mut child: Option<std::process::Child> = Some(spawn_child(tmp.path(), !chaos));
    let mut child_stderr = String::new();

    let deadline = Instant::now() + Duration::from_secs(run_secs);
    let mut inserts = 0u64;
    let mut checkpoints = 0u64;
    let mut successful = 0u64;
    let mut reopens = 0u64;
    let mut kills = 0u64;
    let parent_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut conn = conn;
        while Instant::now() < deadline {
            bulk_insert(&conn, 8, false);
            inserts += 1;
            if chaos && inserts.is_multiple_of(29) {
                // Kill the child mid-write so the reopen reconcile always runs
                // against state a dying writer may have left torn.
                let mut dying = child.take().expect("child present");
                let _ = dying.kill();
                let out = dying.wait_with_output().expect("wait for killed child");
                child_stderr.push_str(&String::from_utf8_lossy(&out.stderr));
                kills += 1;
                child = Some(spawn_child(tmp.path(), false));
            }
            checkpoint_cadence(
                &mut conn,
                &db_path,
                inserts,
                &mut checkpoints,
                &mut successful,
                &mut reopens,
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }));

    // Tear down the final child and collect its stderr.
    let mut child = child.take().expect("final child present");
    let _ = child.kill();
    let child_output = child.wait_with_output().expect("wait for child");
    child_stderr.push_str(&String::from_utf8_lossy(&child_output.stderr));

    let parent_msg = parent_panic
        .err()
        .map(|payload| panic_message(payload.as_ref()));
    if let Some(msg) = &parent_msg {
        eprintln!("parent panicked: {msg}");
    }

    let reproduced = child_stderr.contains(PANIC_SIGNATURE)
        || parent_msg
            .as_deref()
            .is_some_and(|m| m.contains(PANIC_SIGNATURE));

    if reproduced {
        eprintln!(
            "REPRODUCED: two-writer checkpoint race observed ({} inserts, {} checkpoints / {} \
             successful, {} reopens, {} child kills, child exit code {:?})",
            inserts,
            checkpoints,
            successful,
            reopens,
            kills,
            child_output.status.code()
        );
        exit_after_cleanup(&tmp, 0);
    }
    if parent_msg.is_some() {
        // A non-signature parent panic is a harness error, not a negative.
        exit_after_cleanup(&tmp, 2);
    }

    eprintln!(
        "not reproduced in this run ({} inserts, {} checkpoints / {} successful, {} reopens, \
         {} child kills over {run_secs}s); increase MAHBOT_WAL_RACE_DURATION_SECS and retry",
        inserts, checkpoints, successful, reopens, kills,
    );
    exit_after_cleanup(&tmp, 1);
}

/// Inline replacement for the daemon's panic-payload stringifier, keeping this
/// file free of mahbot imports.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Install a panic hook for the deterministic probes: print the panic message,
/// clean up the temp dir, and exit 0 on the reproduced signature / 2 otherwise.
fn install_probe_panic_hook(tmp_path: std::path::PathBuf) {
    std::panic::set_hook(Box::new(move |info| {
        let msg = panic_message(info.payload());
        eprintln!("PROBE panic: {msg}");
        let _ = fs::remove_dir_all(&tmp_path);
        std::process::exit(if msg.contains(PANIC_SIGNATURE) { 0 } else { 2 });
    }));
}

// ── Deterministic stale-tail probe ────────────────────────────────────────
/// Deterministic construction of the stale-tail state (test db only):
///
/// 1. Build an old generation: one bulk insert, so the shared index and the
///    WAL file both carry many frames.
/// 2. RESTART checkpoint: new WAL generation (max_frame=0, checkpoint_seq+1,
///    index cleared) while the old frames stay on disk.
/// 3. Append a small fresh tail through a new connection.
/// 4. Reopen (fresh connection): the disk-scan reseed/reconcile path replays
///    the local frame cache into the shared index. If the reconcile lacks
///    generation validation, the index is left with an old-generation tail
///    while the header advertises a small max_frame, and the next fresh
///    append trips the monotonicity assert.
///
/// The probe exits 0 on the panic signature (reproduced) and 1 after all
/// rounds stay clean (self-healing reconcile — informative negative result).
fn probe_stale_tail_main() {
    let tmp = tempfile::TempDir::new().expect("temp dir for stale-tail probe");
    install_probe_panic_hook(tmp.path().to_path_buf());
    let old_frame_targets = [200usize, 1000, 5000];

    for round in 0..15 {
        let db_path = tmp.path().join(format!("probe-{round}.db"));
        let old_frames = old_frame_targets[round % old_frame_targets.len()];

        // Phase A: old generation (padded rows, one frame each).
        let conn_a = open_race_connection(&db_path);
        execute_retry(&conn_a, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
            .expect("probe create table");
        bulk_insert(&conn_a, old_frames, true);

        // Phase B: RESTART — new generation, old frames stay on disk.
        let restarted = wal_checkpoint(&conn_a, "RESTART");
        drop(conn_a);
        if !restarted {
            eprintln!("probe round {round}: restart busy, skipped");
            continue;
        }

        // Phase C: fresh generation tail via a new connection.
        let conn_b = open_race_connection(&db_path);
        bulk_insert(&conn_b, 12, true);

        // Phase D: reopen (reseed reconcile) and keep appending from both
        // connections, interleaving fresh ids with whatever the reseed left.
        let conn_c = open_race_connection(&db_path);
        for _ in 0..40 {
            bulk_insert(&conn_b, 8, true);
            bulk_insert(&conn_c, 8, true);
        }
        drop(conn_b);
        drop(conn_c);
        eprintln!("probe round {round} (old_frames={old_frames}): clean");
    }

    eprintln!(
        "PROBE NEGATIVE: deterministic stale-tail construction never tripped the assert; \
         the reopen reconcile (checkpoint_seq/salt generation checks + frame-1 index reset) \
         self-heals the stale state"
    );
    exit_after_cleanup(&tmp, 1);
}

// ── Deterministic tshm-patch probe ────────────────────────────────────────

/// Offsets into the `.tshm` file. The coordination header is `#[repr(C)]` and
/// part of the durable file format (turso_core 0.7.2
/// `storage/shared_wal_coordination.rs`), so the layout below is stable:
/// `frame_index_len` is a u32 at 40, `max_frame` a u64 at 56, and the first
/// frame-index block's `{page_id, frame_id}` u64 pair array starts at
/// `base_mapped_len(64)` = 4096 (header 192 B + reader arrays, 4096-aligned).
const TSHM_FRAME_INDEX_LEN_OFFSET: u64 = 40;
const TSHM_MAX_FRAME_OFFSET: u64 = 56;
const TSHM_INDEX_BASE: u64 = 4096;
const TSHM_ENTRY_BYTES: u64 = 16;

/// Directly synthesize the stale-tail state the race is believed to produce:
/// a shared index whose tail holds old-generation frame ids while the header
/// advertises a small post-restart max_frame. This isolates the
/// `record_frame` monotonicity assert from the (unreproducible) race that
/// produces the state: the reopen's Trusted path keeps the persisted index
/// without validating its contents, so a fresh append (id = max_frame + 1)
/// must trip the assert — or be handled by a reset/Busy, which is equally
/// informative for the upstream report.
///
/// Phase sequence (test db only): build an old generation, TRUNCATE to a fresh
/// generation, append a 12-frame tail, drop every connection (tshm unmapped),
/// patch the index on disk, reopen, and append. Exits 0 on the signature.
fn probe_tshm_patch_main() {
    let tmp = tempfile::TempDir::new().expect("temp dir for tshm-patch probe");
    install_probe_panic_hook(tmp.path().to_path_buf());
    let db_path = tmp.path().join("tshm.db");
    let tshm_path = format!("{}-tshm", db_path.display());

    // Old generation, then a fresh generation with a small tail.
    let conn_a = open_race_connection(&db_path);
    execute_retry(&conn_a, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("probe create table");
    bulk_insert(&conn_a, 200, true);
    if !wal_checkpoint(&conn_a, "TRUNCATE") {
        drop(conn_a);
        eprintln!("probe-tshm: truncate stayed busy — harness error");
        exit_after_cleanup(&tmp, 2);
    }
    drop(conn_a);
    let conn_b = open_race_connection(&db_path);
    bulk_insert(&conn_b, 12, true);
    drop(conn_b);

    // All connections dropped → tshm unmapped; patch the file directly.
    let stale_tail = [
        989u64, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000,
    ];
    if let Err(e) = patch_tshm_stale_tail(Path::new(&tshm_path), &stale_tail) {
        eprintln!("probe-tshm: patch failed: {e}");
        exit_after_cleanup(&tmp, 2);
    }

    // Reopen: the Trusted classification keeps the patched index. The next
    // append computes its frame id from the header (max_frame+1) and must
    // trip the monotonicity assert against the stale tail — or be handled.
    let conn_c = open_race_connection(&db_path);
    bulk_insert(&conn_c, 8, true);
    drop(conn_c);

    eprintln!(
        "PROBE NEGATIVE: patched stale-tail index accepted fresh appends; the \
         monotonicity assert was not reachable through a reopened Trusted index"
    );
    exit_after_cleanup(&tmp, 1);
}

/// Append `stale_tail` entries to the shared frame index and bump
/// `frame_index_len`, leaving `max_frame` at its current (small) value.
/// The header state is read from the file, so the probe adapts to however
/// many frames the fresh tail actually produced.
fn patch_tshm_stale_tail(tshm_path: &Path, stale_tail: &[u64]) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom, Write};

    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(tshm_path)?;

    let mut max_buf = [0u8; 8];
    f.seek(SeekFrom::Start(TSHM_MAX_FRAME_OFFSET))?;
    f.read_exact(&mut max_buf)?;
    let max_frame = u64::from_le_bytes(max_buf);
    let mut len_buf = [0u8; 4];
    f.seek(SeekFrom::Start(TSHM_FRAME_INDEX_LEN_OFFSET))?;
    f.read_exact(&mut len_buf)?;
    let index_len = u32::from_le_bytes(len_buf);
    if index_len == 0 || index_len != max_frame as u32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("tshm layout mismatch: max_frame={max_frame}, frame_index_len={index_len}"),
        ));
    }

    for (i, &frame_id) in stale_tail.iter().enumerate() {
        let off = TSHM_INDEX_BASE + (index_len as u64 + i as u64) * TSHM_ENTRY_BYTES;
        f.seek(SeekFrom::Start(off))?;
        f.write_all(&(i as u64 + 1).to_le_bytes())?;
        f.write_all(&frame_id.to_le_bytes())?;
    }
    f.seek(SeekFrom::Start(TSHM_FRAME_INDEX_LEN_OFFSET))?;
    f.write_all(&(index_len + stale_tail.len() as u32).to_le_bytes())?;
    f.sync_all()?;
    Ok(())
}

// ── Chaos mode: kill/respawn the second writer mid-write ──────────────────

fn spawn_child(db_dir: &Path, cycle: bool) -> std::process::Child {
    let mut cmd = Command::new(std::env::current_exe().expect("current exe"));
    cmd.arg("--child")
        .arg(db_dir)
        .arg(if cycle { "1" } else { "0" })
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd.spawn().expect("spawn child writer")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--child") => {
            let Some(dir) = args.get(2) else {
                eprintln!("--child requires a db dir argument");
                std::process::exit(2);
            };
            let cycle = args.get(3).map(String::as_str) == Some("1");
            child_main(Path::new(dir), cycle);
            // Child exits normally only if it survived the window unkilled.
        }
        Some("--probe-stale-tail") => probe_stale_tail_main(),
        Some("--probe-tshm-patch") => probe_tshm_patch_main(),
        Some("--chaos") => run_race(true),
        _ => run_race(false),
    }
}
