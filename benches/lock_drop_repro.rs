//! Deterministic reproduction of the turso multiprocess-WAL macOS lock-drop
//! defect (class C): on macOS (and every non-Linux platform) turso protects
//! the `.tshm` shared-WAL coordination file with process-scoped POSIX
//! `fcntl` byte locks (`F_SETLK`), and POSIX semantics drop ALL of a
//! process's locks on a file when ANY fd of that file is closed by that
//! process. turso never re-verifies its local bookkeeping against the
//! kernel, so one stray open+close of the `.tshm` silently releases the
//! lifetime/reader locks while the owner believes it still holds them.
//! Escalation artifact for the turso maintainers (version pin `=0.7.2`,
//! deliberate; see "Version pin").
//!
//! **Experimental-feature disclaimer**: this exercises the experimental
//! `multiprocess_wal` engine feature (opt-in via
//! `experimental_multiprocess_wal(true)`), which the production daemon enables
//! for cross-process WAL. The defect lives in that opt-in path.
//!
//! # Defect
//!
//! The `.tshm` byte-lock layout (turso_core 0.7.2
//! `storage/shared_wal_coordination.rs`):
//!
//! | Offset | Purpose
//! |--------|---------------
//! | 0      | Process-lifetime shared/exclusive lock (determines Exclusive vs MultiProcess open)
//! | 1      | Writer lock
//! | 2      | Checkpoint lock
//! | 3..3+N | Reader slot locks (one byte per slot)
//!
//! On Linux the locks are open-file-description locks (`F_OFD_SETLK`) that
//! survive unrelated closes. On macOS / other Unix they are process-scoped
//! `F_SETLK` locks: closing ANY fd of the file — even a freshly opened,
//! otherwise unrelated one — releases every lock the process holds on that
//! file. turso's local bookkeeping (`local_lock_state` /
//! `process_local_ownership`) is never re-checked against the kernel, so the
//! owner keeps operating as if the locks were held.
//!
//! A second turso process then probes byte 0 at open (`detect_open_mode`):
//! the probe succeeds, the open is classified **Exclusive**, and the repair
//! path (`repair_transient_state_for_exclusive_open`) reclaims reader slots
//! whose byte locks are gone — clearing the live owner's slot. The owner's
//! next read-txn end hard-panics:
//!
//! ```text
//! PANIC: reader slot released by non-owner
//! ```
//!
//! (The assert text is the stable identifier; the assert lives in
//! `unregister_reader` in `storage/shared_wal_coordination.rs`.) Write-path
//! variants (`connection WAL position must not be behind the committed
//! high-water mark`, monotonicity) are possible; this bench pins the
//! reader-slot variant — it fires at read-txn end with no second commit.
//!
//! # Production consequence
//!
//! Any code path in one process that opens and closes the `.tshm` (a
//! read-only inspection, a sidecar tool, a re-open, a misbehaving Drop)
//! while a peer process holds a reader slot strands the peer: the next
//! process to open the store sees Exclusive, repairs the shared authority
//! underneath the live owner, and the owner panics on the next read-txn end
//! (or on the next commit, on the write path) — incapacitating it mid-flight.
//!
//! # How to run
//!
//! Build via the non-default feature gate (excluded from default
//! builds/tests/benches):
//!
//! ```text
//! cargo bench --no-default-features --features lock-drop-repro --bench lock_drop_repro -- --natural
//! ```
//!
//! `--natural` is the one-shot primary reproduction (macOS): a two-process
//! handshake over pipes (no sleeps, events are deterministic). The owner
//! child opens the store, commits a baseline, opens a write transaction,
//! spills padded rows into the WAL, and registers a reader slot; the
//! observer parent probes the `.tshm` byte locks from a second process
//! (its own process-scoped locks would be invisible), commands the trigger,
//! and watches byte 0 transition EAGAIN -> acquired. It then opens the store
//! itself (the "second turso process"): the Exclusive repair clears the
//! owner's reader slot, and the owner's ROLLBACK panics with the reader-slot
//! signature. On turso 0.7.2 / macOS:
//!
//! ```text
//! phase baseline: 41ms
//! phase probe-before: 0ms (wal=16887912 B, lifetime+reader-slot locks held)
//! phase trigger: 0ms
//! locks dropped: byte 0 (lifetime) and byte 3 (reader slot 0) both free
//! phase probe-after: 0ms
//! owner stderr:
//! PANIC: reader slot released by non-owner | slot_index=0, expected_owner=37087042600961, current_owner=0, local_reader_count=1
//! phase consequence: 6ms
//! REPRODUCED: owner's ROLLBACK panicked with the reader-slot assert after a
//! second process repaired the shared authority over its dropped locks
//! ```
//!
//! (Phase timings are live diagnostics only — the bench never asserts on
//! wall-clock durations.)
//!
//! Controls — all assert the opposite outcome (locks survive, exit 1):
//!
//! ```text
//! cargo bench --no-default-features --features lock-drop-repro --bench lock_drop_repro -- --no-trigger
//! cargo bench --no-default-features --features lock-drop-repro --bench lock_drop_repro -- --close-other
//! cargo bench --no-default-features --features lock-drop-repro --bench lock_drop_repro -- --ofd-contrast
//! ```
//!
//! - `--no-trigger` (macOS): the owner never performs a close; the locks
//!   must stay held. Proves the probe machinery itself holds nothing.
//! - `--close-other` (macOS): the owner closes a fd of the main `.db` file,
//!   not the `.tshm`; POSIX close-drop is per-file, so the `.tshm` locks
//!   must survive. Proves the drop is file-scoped, not process-wide.
//! - `--ofd-contrast` (Linux): the same trigger as `--natural`; OFD locks
//!   must survive an unrelated close. On a Linux build this is the proof
//!   that the defect is the F_SETLK backend, not the trigger itself.
//!
//! On non-macOS `--natural`/`--no-trigger`/`--close-other` skip (exit 3);
//! `--ofd-contrast` skips on non-Linux. The default invocation therefore
//! never runs the primary repro on Linux — the OFD contrast is an explicit,
//! separate mode.
//!
//! # Exit contract
//!
//! - `0` — bug reproduced: after the trigger the observer's exclusive probe
//!   of byte 0 transitioned held -> free, and the owner panicked with the
//!   reader-slot signature (printed live).
//! - `1` — clean / fix present: the locks survived the trigger (a fixed
//!   build using OFD locks on macOS behaves this way), or PARTIAL — the
//!   locks were dropped but the owner's ROLLBACK completed without the
//!   reader-slot panic (the consequence stage did not fire; reported loudly,
//!   never a silent clean).
//! - `2` — harness error: missing argument, failed open/insert/probe, the
//!   reader slot was not registered, the WAL did not spill, an unexpected
//!   panic, or a handshake timeout.
//! - `3` — skip: the mode is not applicable on this platform.
//!
//! Bare invocation (no arguments) prints this usage and exits `2` — nothing
//! runs silently.
//!
//! # Version pin / upstream status
//!
//! The repo pins `turso = "=0.7.2"` deliberately so the storage layer cannot
//! drift under this artifact (same pin as the wal_race_repro escalation).
//! The defect is unfixed upstream: no public issue or PR exists for the
//! macOS `F_SETLK` close-drop — this artifact is the first report.
//!
//! # Suggested fix
//!
//! Use OFD locks (`F_OFD_SETLK`) on macOS, as on Linux — they are per-open
//! file description and survive unrelated closes. `F_OFD_SETLK` is observed
//! to work on macOS 26+; a runtime fallback to `F_SETLK` (with the
//! close-drop caveat documented) is needed for older macOS releases. The
//! `.tshm` byte-lock layout and the ownership model are unchanged.
//!
//! # Standalone
//!
//! Self-contained: only `turso` (`=0.7.2`), `tempfile`, `libc` (unix) and
//! `std` are used, so a copy of this file in a fresh project (`harness =
//! false`, deps `turso = "=0.7.2"`, `tempfile`, `libc`) builds and runs
//! unchanged. Each run uses a fresh disposable database under a temp dir;
//! live stores are never touched and concurrent runs cannot interfere.
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::process::{Child, ChildStderr, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use turso::core::{Database, DatabaseOpts, OpenFlags, PlatformIO};

const PANIC_SIGNATURE: &str = "reader slot released by non-owner";
/// Padded rows that force a page-cache spill to the WAL inside the owner's
/// open write transaction (default cache 2000 pages, 90% spill threshold).
/// The spill is a deliberate precondition: only the resulting WAL/authority
/// length mismatch makes the intruder rebuild from disk instead of adopting
/// the authority — and only that repair branch clears the owner's reader
/// slot. The monotonicity assert cannot fire here: the owner never appends
/// after the trigger, and the intruder discards the index before rebuilding.
const SPILL_ROWS: usize = 5000;
/// WAL layout (turso_core 0.7.2 `storage/sqlite3_ondisk.rs`, default
/// 4096-byte pages): 32-byte header, 24-byte frame header + page.
const WAL_HEADER: u64 = 32;
const FRAME_BYTES: u64 = 24 + 4096;
/// Baseline commits (CREATE + INSERT) write at most this many frames; a WAL
/// larger than header + 8 frames can only be an in-transaction spill.
const MAX_BASELINE_FRAMES: u64 = 8;
/// `.tshm` byte-lock offsets (turso_core `shared_wal_coordination.rs`).
const LIFETIME_OFFSET: u64 = 0;
const READER_SLOT0_OFFSET: u64 = 3;
/// Handshake hang-guard: maps to harness error 2, never a timing assertion.
const MARKER_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone, Copy)]
enum Trigger {
    Fd,
    None,
    OtherFile,
}

fn trigger_arg(t: Trigger) -> &'static str {
    match t {
        Trigger::Fd => "fd",
        Trigger::None => "none",
        Trigger::OtherFile => "other",
    }
}

// ── Shared DB helpers (mirror wal_race_repro) ─────────────────────────────

fn open_db(db_path: &Path) -> std::sync::Arc<turso::core::Connection> {
    let io: std::sync::Arc<dyn turso::core::IO> =
        std::sync::Arc::new(PlatformIO::new().expect("platform io"));
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

fn execute_retry(conn: &std::sync::Arc<turso::core::Connection>, sql: &str) -> Result<(), String> {
    for _ in 0..10_000 {
        match conn.execute(sql) {
            Ok(()) => return Ok(()),
            Err(e) if e.to_string().to_lowercase().contains("busy") => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(e) => return Err(format!("statement failed: {e}")),
        }
    }
    Err("statement stayed busy".to_string())
}

/// Insert `count` padded rows (~4 KiB each, one page = one WAL frame each) so
/// the page cache spills to the WAL inside the open transaction, leaving
/// uncommitted frames on disk past the committed `max_frame`.
fn bulk_insert_spill(conn: &std::sync::Arc<turso::core::Connection>, count: usize) {
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
        execute_retry(conn, &sql).expect("spill insert with busy retry");
        remaining -= n;
        offset += n;
    }
}

// ── Panic hook / cleanup ──────────────────────────────────────────────────

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Owner role: exit 0 on the reproduced panic signature, 2 otherwise. The
/// DONE marker lets the parent bound the reap (the parent never waits on the
/// child's exit status directly).
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = panic_message(info.payload());
        eprintln!("PANIC: {msg}");
        send_marker("DONE");
        std::process::exit(if msg.contains(PANIC_SIGNATURE) { 0 } else { 2 });
    }));
}

/// Parent role: any panic is a harness error (the target panic fires in the
/// owner child, never here); clean up the per-run temp dir on the way out.
fn install_harness_panic_hook(tmp: &std::path::Path) {
    let tmp = tmp.to_path_buf();
    std::panic::set_hook(Box::new(move |info| {
        eprintln!("PANIC: {}", panic_message(info.payload()));
        let _ = fs::remove_dir_all(&tmp);
        std::process::exit(2);
    }));
}

fn exit_after_cleanup(tmp: &tempfile::TempDir, code: i32) -> ! {
    let _ = fs::remove_dir_all(tmp.path());
    std::process::exit(code);
}

fn harness(tmp: &tempfile::TempDir, msg: &str) -> ! {
    eprintln!("harness error: {msg}");
    exit_after_cleanup(tmp, 2);
}

// ── fcntl byte-lock probe (observer) ──────────────────────────────────────

/// Open fd used to probe `.tshm` byte locks. `F_SETLK` works from a second
/// process against both `F_SETLK` and `F_OFD_SETLK` holders, so one probe
/// implementation serves macOS and the Linux contrast mode.
struct ProbeFd {
    file: fs::File,
}

impl ProbeFd {
    fn open(path: &Path) -> std::io::Result<Self> {
        // F_WRLCK requires an fd open for writing (else EBADF).
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
        // macOS documents EACCES for a conflicting F_SETLK; modern macOS
        // returns EAGAIN. Treat both as "lock held".
        if e.raw_os_error() == Some(libc::EACCES) || e.raw_os_error() == Some(libc::EAGAIN) {
            return Ok(false);
        }
        Err(e)
    }
}

// ── Two-process handshake ─────────────────────────────────────────────────

struct Owner {
    child: Child,
    stdin: ChildStdin,
    stderr: ChildStderr,
    rx: Receiver<String>,
}

fn spawn_owner(db_path: &Path, trigger: Trigger) -> Owner {
    let exe = std::env::current_exe().expect("current exe");
    let mut child = Command::new(exe)
        .arg("--owner")
        .arg(db_path)
        .arg("--trigger")
        .arg(trigger_arg(trigger))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn owner");
    let stdin = child.stdin.take().expect("owner stdin");
    let stderr = child.stderr.take().expect("owner stderr");
    let stdout = child.stdout.take().expect("owner stdout");
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let l = line.trim();
                    if !l.is_empty() && tx.send(l.to_string()).is_err() {
                        break;
                    }
                }
            }
        }
    });
    Owner {
        child,
        stdin,
        stderr,
        rx,
    }
}

/// Piped stdout is block-buffered: markers must be flushed explicitly.
fn send_marker(marker: &str) {
    println!("{marker}");
    std::io::stdout().flush().expect("flush marker");
}

fn send_cmd(stdin: &mut ChildStdin, cmd: &str) {
    writeln!(stdin, "{cmd}").expect("write owner command");
    stdin.flush().expect("flush owner command");
}

fn recv_marker(rx: &Receiver<String>, label: &str, expected: &str) -> Result<(), String> {
    match rx.recv_timeout(MARKER_TIMEOUT) {
        Ok(m) if m == expected => Ok(()),
        Ok(m) => Err(format!("expected {expected:?} ({label}), got {m:?}")),
        Err(_) => Err(format!("timed out waiting for {expected:?} ({label})")),
    }
}

fn recv_marker_or_exit(
    tmp: &tempfile::TempDir,
    rx: &Receiver<String>,
    label: &str,
    expected: &str,
) {
    if let Err(e) = recv_marker(rx, label, expected) {
        harness(tmp, &e);
    }
}

fn wait_cmd(expected: &str) {
    let mut line = String::new();
    loop {
        line.clear();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => panic!("owner: stdin closed before {expected}"),
            Ok(_) => {
                if line.trim() == expected {
                    return;
                }
                eprintln!("owner: ignoring unexpected command {:?}", line.trim());
            }
        }
    }
}

fn read_stderr(stderr: &mut ChildStderr) -> String {
    let mut s = String::new();
    let _ = BufReader::new(stderr).read_to_string(&mut s);
    s
}

/// Send REPAIRED, then reap the owner with a bounded wait: the owner always
/// emits a DONE marker (normal exits and the panic hook alike) before
/// terminating, so a hang surfaces as a marker timeout (harness error 2)
/// instead of blocking the parent. Prints the owner's stderr; returns its
/// exit status so the caller classifies it (never a silent clean).
fn finish_owner(o: &mut Owner, tmp: &tempfile::TempDir) -> std::process::ExitStatus {
    let _ = writeln!(o.stdin, "REPAIRED");
    let _ = o.stdin.flush();
    recv_marker_or_exit(tmp, &o.rx, "done", "DONE");
    let stderr = read_stderr(&mut o.stderr);
    let status = o.child.wait().expect("wait for owner");
    if !stderr.is_empty() {
        eprintln!("owner stderr:\n{stderr}");
    }
    status
}

// ── Owner child role ──────────────────────────────────────────────────────

fn owner_main(db_path: &Path, trigger: Trigger) -> ! {
    install_panic_hook();
    let conn = open_db(db_path);
    execute_retry(&conn, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("owner create table");
    execute_retry(&conn, "INSERT INTO t (val) VALUES ('baseline')").expect("owner baseline");
    execute_retry(&conn, "BEGIN").expect("owner begin");
    // The write txn registers a reader slot (snapshot at max_frame) before
    // the first frame is written, then spills uncommitted frames to the WAL.
    bulk_insert_spill(&conn, SPILL_ROWS);
    send_marker("READY");
    wait_cmd("TRIGGER");
    match trigger {
        Trigger::Fd => {
            // POSIX F_SETLK: closing this fd drops EVERY lock this process
            // holds on the `.tshm` — the defect, triggered by open+read+drop.
            let tshm = format!("{}-tshm", db_path.display());
            let f = fs::File::open(&tshm).expect("owner open tshm for trigger");
            let mut buf = [0u8; 16];
            let _ = (&f).read(&mut buf);
            drop(f);
        }
        Trigger::None => {}
        Trigger::OtherFile => {
            // A different file: close-drop is per-file, `.tshm` locks survive.
            let f = fs::File::open(db_path).expect("owner open db for trigger");
            let mut buf = [0u8; 16];
            let _ = (&f).read(&mut buf);
            drop(f);
        }
    }
    send_marker("TRIGGERED");
    wait_cmd("REPAIRED");
    match execute_retry(&conn, "ROLLBACK") {
        Ok(()) => {
            eprintln!("owner: ROLLBACK completed without the reader-slot panic");
            send_marker("DONE");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("owner: ROLLBACK failed: {e}");
            send_marker("DONE");
            std::process::exit(2);
        }
    }
}

// ── Primary reproduction (observer parent) ────────────────────────────────

fn natural_main() -> ! {
    if !cfg!(target_os = "macos") {
        eprintln!(
            "SKIP: the macOS F_SETLK lock-drop repro is not applicable on this platform \
             (run --ofd-contrast on Linux)"
        );
        std::process::exit(3);
    }
    let tmp = tempfile::TempDir::new().expect("temp dir");
    install_harness_panic_hook(tmp.path());
    let db_path = tmp.path().join("repro.db");
    let tshm_path = format!("{}-tshm", db_path.display());
    let wal_path = format!("{}-wal", db_path.display());
    let mut owner = spawn_owner(&db_path, Trigger::Fd);

    // Phase 1 — owner baseline + reader slot (with spilled WAL frames).
    let t0 = Instant::now();
    recv_marker_or_exit(&tmp, &owner.rx, "baseline", "READY");
    eprintln!("phase baseline: {}ms", t0.elapsed().as_millis());

    // Phase 2 — pre-trigger probes from a second process: the lifetime and
    // reader-slot locks must be held, and the spill must have happened (it is
    // what forces the intruder onto the repair branch later).
    let t1 = Instant::now();
    {
        let probe = ProbeFd::open(Path::new(&tshm_path)).expect("open probe fd");
        if probe.probe(LIFETIME_OFFSET).expect("probe byte 0") {
            harness(&tmp, "byte 0 was not held before the trigger");
        }
        if probe.probe(READER_SLOT0_OFFSET).expect("probe reader slot") {
            harness(
                &tmp,
                "reader slot byte was not held — the owner's read txn did not register a slot",
            );
        }
    }
    let wal_size = fs::metadata(&wal_path).expect("stat wal").len();
    if wal_size <= WAL_HEADER + MAX_BASELINE_FRAMES * FRAME_BYTES {
        harness(
            &tmp,
            &format!("owner WAL did not spill (size {wal_size} B)"),
        );
    }
    eprintln!(
        "phase probe-before: {}ms (wal={wal_size} B, lifetime+reader-slot locks held)",
        t1.elapsed().as_millis()
    );

    // Phase 3 — trigger: the owner performs the open+read+drop pattern.
    let t2 = Instant::now();
    send_cmd(&mut owner.stdin, "TRIGGER");
    recv_marker_or_exit(&tmp, &owner.rx, "trigger", "TRIGGERED");
    eprintln!("phase trigger: {}ms", t2.elapsed().as_millis());

    // Phase 4 — post-trigger probe: the locks must be gone.
    let t3 = Instant::now();
    {
        let probe = ProbeFd::open(Path::new(&tshm_path)).expect("open probe fd");
        let dropped = probe.probe(LIFETIME_OFFSET).expect("probe byte 0");
        if !dropped {
            eprintln!(
                "CLEAN: the trigger did not drop the process-scoped fcntl locks — a fixed \
                 build using OFD locks on macOS behaves this way"
            );
            let status = finish_owner(&mut owner, &tmp);
            match status.code() {
                Some(1) => exit_after_cleanup(&tmp, 1),
                _ => harness(
                    &tmp,
                    &format!("owner exited unexpectedly after a clean trigger: {status}"),
                ),
            }
        }
        if !probe.probe(READER_SLOT0_OFFSET).expect("probe reader slot") {
            harness(
                &tmp,
                "reader slot byte survived while byte 0 dropped — inconsistent lock state",
            );
        }
    }
    eprintln!("locks dropped: byte 0 (lifetime) and byte 3 (reader slot 0) both free");
    eprintln!("phase probe-after: {}ms", t3.elapsed().as_millis());

    // Phase 5 — consequence: the observer opens the store as a second turso
    // process. It sees Exclusive (byte 0 free) and its repair clears the
    // owner's reader slot; the owner's ROLLBACK must then hard-panic. The
    // intruder connection stays alive here and its fds are never churned.
    let t4 = Instant::now();
    let _intruder = open_db(&db_path);
    let status = finish_owner(&mut owner, &tmp);
    match status.code() {
        Some(0) => {
            eprintln!("phase consequence: {}ms", t4.elapsed().as_millis());
            eprintln!(
                "REPRODUCED: owner's ROLLBACK panicked with the reader-slot assert after a \
                 second process repaired the shared authority over its dropped locks"
            );
            exit_after_cleanup(&tmp, 0);
        }
        Some(1) => {
            eprintln!(
                "PARTIAL: locks were dropped but the owner's ROLLBACK completed without the \
                 reader-slot panic"
            );
            exit_after_cleanup(&tmp, 1);
        }
        Some(2) => harness(&tmp, "owner reported a harness error"),
        _ => harness(&tmp, &format!("owner exited abnormally: {status}")),
    }
}

// ── Controls ──────────────────────────────────────────────────────────────

fn control_main(trigger: Trigger, label: &str, applicable: bool) -> ! {
    if !applicable {
        eprintln!("SKIP: {label} is not applicable on this platform");
        std::process::exit(3);
    }
    let tmp = tempfile::TempDir::new().expect("temp dir");
    install_harness_panic_hook(tmp.path());
    let db_path = tmp.path().join("repro.db");
    let tshm_path = format!("{}-tshm", db_path.display());
    let mut owner = spawn_owner(&db_path, trigger);

    let t0 = Instant::now();
    recv_marker_or_exit(&tmp, &owner.rx, "baseline", "READY");
    let probe = ProbeFd::open(Path::new(&tshm_path)).expect("open probe fd");
    if probe.probe(LIFETIME_OFFSET).expect("probe byte 0") {
        harness(&tmp, "byte 0 was not held before the trigger");
    }
    drop(probe);
    send_cmd(&mut owner.stdin, "TRIGGER");
    recv_marker_or_exit(&tmp, &owner.rx, "trigger", "TRIGGERED");
    let probe = ProbeFd::open(Path::new(&tshm_path)).expect("open probe fd");
    let held = !probe.probe(LIFETIME_OFFSET).expect("probe byte 0");
    drop(probe);
    if !held {
        harness(
            &tmp,
            &format!("locks were dropped — the {label} control did not confirm clean behavior"),
        );
    }
    eprintln!("phase control: {}ms", t0.elapsed().as_millis());
    eprintln!("CLEAN: locks still held ({label})");
    let status = finish_owner(&mut owner, &tmp);
    match status.code() {
        Some(1) => exit_after_cleanup(&tmp, 1),
        _ => harness(
            &tmp,
            &format!("owner exited unexpectedly in the {label} control: {status}"),
        ),
    }
}

// ── Dispatch ──────────────────────────────────────────────────────────────

fn print_usage() {
    eprintln!(
        "usage: lock_drop_repro <mode> [args]

modes:
  --natural              primary repro (macOS): two-process lock-drop via
                         open+read+drop of an unrelated .tshm fd, then the
                         reader-slot consequence panic
  --no-trigger           control (macOS): no close performed; locks must hold
  --close-other          control (macOS): close a fd of the .db file; locks must hold
  --ofd-contrast         control (Linux): same trigger as --natural; OFD
                         locks must survive an unrelated close
  --owner DB --trigger M owner child role (spawned by the modes above);
                         M = fd | none | other

exit codes:
  0  bug reproduced (locks dropped + reader-slot panic signature printed live)
  1  clean / fix present (locks held, or PARTIAL: dropped without the panic)
  2  harness error
  3  skip (mode not applicable on this platform)"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--natural") => natural_main(),
        Some("--no-trigger") => {
            control_main(Trigger::None, "no-trigger", cfg!(target_os = "macos"))
        }
        Some("--close-other") => {
            control_main(Trigger::OtherFile, "close-other", cfg!(target_os = "macos"))
        }
        Some("--ofd-contrast") => {
            control_main(Trigger::Fd, "ofd-contrast", cfg!(target_os = "linux"))
        }
        Some("--owner") => {
            let Some(db) = args.get(2) else {
                eprintln!("harness error: --owner requires a db path");
                std::process::exit(2);
            };
            let trigger = match (
                args.get(3).map(String::as_str),
                args.get(4).map(String::as_str),
            ) {
                (Some("--trigger"), Some("fd")) => Trigger::Fd,
                (Some("--trigger"), Some("none")) => Trigger::None,
                (Some("--trigger"), Some("other")) => Trigger::OtherFile,
                _ => {
                    eprintln!("harness error: --owner requires --trigger fd|none|other");
                    std::process::exit(2);
                }
            };
            owner_main(Path::new(db), trigger);
        }
        _ => {
            print_usage();
            std::process::exit(2);
        }
    }
}
