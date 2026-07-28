/// Voice pipeline E2E benchmark with single-instance lock and 30-minute timeout.
///
/// The lock prevents concurrent benchmark runs (which cause Metal GPU deadlocks
/// via competing Metal device submissions).  A 30-minute timeout aborts hung
/// runs via [`std::process::exit`].
///
/// # Lock mechanism
///
/// The lock functions here mirror the pattern in `src/self_update.rs` (private
/// `try_flock` / `open_lock_file`).  That module is the canonical implementation;
/// this file duplicates ~15 lines rather than extracting a shared helper for a
/// single consumer.
///
/// # Self-test mode
///
/// This binary has `harness = false` (see Cargo.toml), so `#[test]` functions
/// compile but are never discovered by libtest.  Set `MAHBOT_BENCH_SELF_TEST=1`
/// to run the internal [`run_self_tests()`] function instead of the benchmark.
///
/// ```text
/// MAHBOT_BENCH_SELF_TEST=1 cargo bench --bench voice_pipeline_e2e --features voice-tests
/// ```
///
/// # Lock release guarantee
///
/// The returned `File` keeps the kernel-level `flock` alive.  The kernel
/// releases the lock when the file descriptor is closed during process
/// teardown — this happens on normal `Drop`, on `process::exit` (Rust
/// destructors do NOT run, but the kernel closes all fds), and on SIGKILL.
/// There is no stale-lock scenario.
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::PathBuf;
use std::time::Duration;

// ── Single-instance lock ──────────────────────────────────────────────────

fn lock_file_path() -> PathBuf {
    mahbot::config::default_config_dir()
        .expect("Cannot resolve ~/.mahbot/ for lock file")
        .join("voice_pipeline_e2e.lock")
}

/// Acquire an exclusive lock on the benchmark, blocking until available.
///
/// Polls with `LOCK_EX | LOCK_NB` every 5 seconds, printing a status
/// message on the first failure and every 60 seconds thereafter.
/// The returned `File` keeps the kernel-level `flock` alive — the kernel
/// releases the lock when the file descriptor is closed during process
/// teardown (including on `process::exit` or SIGKILL).
///
/// # Panics
///
/// Panics if the lock directory cannot be created or if a non-retryable
/// OS error occurs.
fn acquire_bench_lock() -> File {
    let lock_path = lock_file_path();

    // Ensure parent ~/.mahbot/ exists.
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent).expect("failed to create ~/.mahbot/ for benchmark lock");
    }

    // Open once; reuse the same fd for all polling iterations.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .expect("failed to open benchmark lock file");

    // Throttle-printing state: print on first failure, then every 60 s.
    let mut first_fail = true;
    const STATUS_INTERVAL: Duration = Duration::from_secs(60);
    let mut last_status = std::time::Instant::now();

    loop {
        match try_flock(&file) {
            Ok(true) => return file,
            Ok(false) => {
                if first_fail {
                    eprintln!(
                        "Another voice benchmark is already running (lock: {}) — \
                         waiting for it to complete...",
                        lock_path.display()
                    );
                    first_fail = false;
                    last_status = std::time::Instant::now();
                } else if last_status.elapsed() >= STATUS_INTERVAL {
                    eprintln!("Still waiting for lock ({})...", lock_path.display());
                    last_status = std::time::Instant::now();
                }
                std::thread::sleep(Duration::from_secs(5));
            }
            Err(e) => panic!(
                "flock on benchmark lock {} failed: {e}",
                lock_path.display(),
            ),
        }
    }
}

/// Mirrors `src/self_update.rs::try_flock` — see that module for the
/// canonical implementation and test coverage (including
/// `test_lock_acquire_and_release_with_temp_dir`).
#[cfg(unix)]
fn try_flock(file: &File) -> io::Result<bool> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: `flock` operates on a valid fd opened with read+write.
    // `LOCK_NB` makes it non-blocking; `EAGAIN` means lock held.
    let result = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        Ok(true)
    } else {
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EAGAIN) => Ok(false),
            _ => Err(err),
        }
    }
}

/// Mirrors `src/self_update.rs::try_flock` — see that module for the
/// canonical implementation (including `const LOCK_VIOLATION` pattern).
#[cfg(windows)]
fn try_flock(file: &File) -> io::Result<bool> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{ERROR_LOCK_VIOLATION, HANDLE};
    use windows_sys::Win32::Storage::FileSystem::{
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
    };

    let handle = file.as_raw_handle() as HANDLE;

    // Matches self_update.rs canonical pattern: const LOCK_VIOLATION.
    const LOCK_VIOLATION: i32 = ERROR_LOCK_VIOLATION as i32;

    let mut overlapped =
        unsafe { std::mem::zeroed::<windows_sys::Win32::System::IO::OVERLAPPED>() };
    let locked = unsafe {
        LockFileEx(
            handle,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            0,
            0,
            &mut overlapped,
        )
    };
    if locked != 0 {
        Ok(true)
    } else {
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(LOCK_VIOLATION) => Ok(false),
            _ => Err(err),
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    // Self-test mode: run internal tests and exit.
    // This is the standard approach for harness=false bench binaries that
    // cannot use #[test] functions.
    if std::env::var("MAHBOT_BENCH_SELF_TEST").as_deref() == Ok("1") {
        run_self_tests();
        return;
    }

    // 1. Acquire single-instance lock.
    //    Prevents concurrent benchmark runs that cause Metal GPU deadlock.
    let _lock = acquire_bench_lock();
    // Kernel releases `flock` on process death (Drop, process::exit, SIGKILL).

    // 2. Create tokio runtime for the timeout.
    let runtime = tokio::runtime::Runtime::new()
        .expect("failed to create tokio runtime for benchmark timeout");

    // 3. Run benchmark with 30-minute timeout.
    // NOTE: spawn_blocking tasks are NOT cancelable at the Rust level.
    // When the timeout fires, tokio returns Err(Elapsed) but the kernel
    // threads (Metal GPU work, ONNX) continue executing.  We call
    // process::exit(1) to terminate the process, which kills all threads
    // and the kernel releases the flock.  This is accepted per the
    // non-goals (no GPU-level timeout guards).
    let result = runtime.block_on(async {
        tokio::time::timeout(
            Duration::from_mins(30),
            tokio::task::spawn_blocking(|| {
                mahbot::voice::run_voice_pipeline_benchmark();
            }),
        )
        .await
    });

    match result {
        Ok(Ok(())) => {
            // Normal completion — benchmark handled pass/fail internally.
        }
        Ok(Err(join_err)) => {
            eprintln!("BENCHMARK PANICKED: {join_err}");
            std::process::exit(1);
        }
        Err(_elapsed) => {
            eprintln!("BENCHMARK TIMED OUT after 30 minutes");
            std::process::exit(1);
        }
    }
}

// ── Self-tests (for harness=false binaries) ────────────────────────────────

/// Run internal self-tests for the lock functions.
///
/// The bench binary has `harness = false` (see Cargo.toml), so `#[test]`
/// functions compile but are never discovered by libtest.  This self-test
/// entry point is invoked by setting `MAHBOT_BENCH_SELF_TEST=1`.
///
/// Panics on failure — the panic message identifies the failing test.
fn run_self_tests() {
    let dir = std::env::temp_dir().join("mahbot_bench_self_test");
    let _ = fs::remove_dir_all(&dir); // clean slate
    fs::create_dir_all(&dir).expect("failed to create self-test temp dir");

    let lock_path = dir.join("test_bench.lock");

    // ── Test 1: try_flock acquires and releases ───────────────────────
    let file1 = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .expect("Test 1: failed to open lock file");

    assert!(
        try_flock(&file1).expect("Test 1: try_flock should not error"),
        "Test 1: first lock should succeed",
    );

    // ── Test 2: second fd on same file is rejected ────────────────────
    let file2 = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .expect("Test 2: failed to open lock file");

    assert!(
        !try_flock(&file2).expect("Test 2: try_flock should not error"),
        "Test 2: second lock on same file should fail",
    );

    // ── Test 3: lock is reusable after release ────────────────────────
    drop(file1); // releases flock

    assert!(
        try_flock(&file2).expect("Test 3: try_flock should not error"),
        "Test 3: lock should succeed after first holder releases",
    );

    // ── Test 4: lock file path ───────────────────────────────────────
    let path = lock_file_path();
    assert!(
        path.ends_with("voice_pipeline_e2e.lock"),
        "Test 4: lock file path must end with voice_pipeline_e2e.lock, got: {}",
        path.display(),
    );

    // Clean up.
    drop(file2);
    let _ = fs::remove_dir_all(&dir);

    eprintln!("✅ All bench lock self-tests passed");
}
