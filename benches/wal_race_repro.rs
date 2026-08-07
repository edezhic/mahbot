//! Cross-process reproduction harness for the two-writer WAL checkpoint race.
//!
//! The corruption vector: a TRUNCATE checkpoint resets the shared WAL frame
//! index while a second connection holding a stale frame index appends into an
//! old frame-index slot, panicking with "shared WAL frame ids must increase
//! monotonically". The daemon hit this when a self-update handoff ran a
//! TRUNCATE checkpoint while a second instance was already writing to the same
//! store.
//!
//! The harness plays both roles from a single executable (no `CARGO_BIN_EXE`):
//! the parent opens a temporary store and runs TRUNCATE checkpoints while the
//! child holds a second connection open across the truncates and keeps
//! appending — the exact handoff shape. Both processes' stderr is scanned for
//! the panic signature.
//!
//! # Exit codes
//!
//! - `0`: panic signature reproduced (the unpatched storage layer exhibits the
//!   two-writer bug)
//! - `1`: not reproduced within the run window (stochastic — increase
//!   `MAHBOT_WAL_RACE_DURATION_SECS`)
//! - `2`: harness error
//!
//! A reproducing run is the gate for any future vendored storage-patch
//! attempt: the patch must make the harness stop reproducing the signature.
//!
//! Run with: `cargo bench --no-default-features --features wal-repro --bench wal_race_repro`
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
fn execute_with_busy_retry(conn: &Arc<turso::core::Connection>, sql: &str) {
    for _ in 0..10_000 {
        match conn.execute(sql) {
            Ok(()) => return,
            Err(e) if is_busy_error(&e) => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(e) => panic!("statement failed: {e}"),
        }
    }
    panic!("statement stayed busy: {sql}");
}

fn is_busy_error(e: &turso::core::LimboError) -> bool {
    e.to_string().to_lowercase().contains("busy")
}

/// Run a TRUNCATE checkpoint, returning true when it succeeded (busy == 0).
/// A successful truncate is what resets the shared WAL frame index.
fn truncate_checkpoint(conn: &Arc<turso::core::Connection>) -> bool {
    for _ in 0..10_000 {
        let mut stmt = match conn.query("PRAGMA wal_checkpoint(TRUNCATE)") {
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

// ── Child: the "stale" second writer ──────────────────────────────────────

fn child_main(db_dir: &Path) {
    let conn = open_race_connection(&db_dir.join("race.db"));
    let mut i = 0u64;
    loop {
        execute_with_busy_retry(&conn, &format!("INSERT INTO t (val) VALUES ('child-{i}')"));
        i += 1;
        std::thread::sleep(Duration::from_millis(1));
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

fn parent_main() {
    let run_secs: u64 = std::env::var("MAHBOT_WAL_RACE_DURATION_SECS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(DEFAULT_RUN_SECS);

    let tmp = tempfile::TempDir::new().expect("temp dir for race database");
    let db_path = tmp.path().join("race.db");

    let conn = open_race_connection(&db_path);
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("create race table");

    // Spawn the child writer before any checkpointing so it holds its
    // connection open across the parent's TRUNCATEs.
    let mut child = Command::new(std::env::current_exe().expect("current exe"))
        .arg("--child")
        .arg(tmp.path())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn child writer");

    // Race loop: keep inserting while periodically resetting the shared WAL
    // frame index with a TRUNCATE checkpoint — the handoff shape that makes a
    // stale connection's next append trip the monotonicity panic.
    let deadline = Instant::now() + Duration::from_secs(run_secs);
    let mut inserts = 0u64;
    let mut checkpoints = 0u64;
    let mut successful_truncates = 0u64;
    let parent_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        while Instant::now() < deadline {
            execute_with_busy_retry(
                &conn,
                &format!("INSERT INTO t (val) VALUES ('parent-{inserts}')"),
            );
            inserts += 1;
            if inserts.is_multiple_of(5) {
                checkpoints += 1;
                if truncate_checkpoint(&conn) {
                    successful_truncates += 1;
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }));

    // Tear down the child and collect its stderr.
    let _ = child.kill();
    let child_output = child.wait_with_output().expect("wait for child");
    let child_stderr = String::from_utf8_lossy(&child_output.stderr).to_string();

    let parent_msg = parent_panic.err().map(|payload| panic_message(&*payload));
    if let Some(msg) = &parent_msg {
        eprintln!("parent panicked: {msg}");
    }

    let reproduced = child_stderr.contains(PANIC_SIGNATURE)
        || parent_msg
            .as_deref()
            .is_some_and(|m| m.contains(PANIC_SIGNATURE));

    if reproduced {
        eprintln!(
            "REPRODUCED: two-writer checkpoint race observed ({} inserts, {} TRUNCATE \
             attempts / {} successful, child exit code {:?})",
            inserts,
            checkpoints,
            successful_truncates,
            child_output.status.code()
        );
        exit_after_cleanup(&tmp, 0);
    }

    eprintln!(
        "not reproduced in this run ({} inserts, {} TRUNCATE attempts / {} successful over \
         {run_secs}s); increase MAHBOT_WAL_RACE_DURATION_SECS and retry",
        inserts, checkpoints, successful_truncates,
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--child") {
        let Some(dir) = args.get(2) else {
            eprintln!("--child requires a db dir argument");
            std::process::exit(2);
        };
        child_main(Path::new(dir));
        // Child exits normally only if it survived the window unkilled.
        return;
    }
    parent_main();
}
