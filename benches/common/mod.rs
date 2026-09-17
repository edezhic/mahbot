//! Single-instance lock and hung-run timeout scaffolding shared by the two
//! wake-word bench targets (`wake_word`, `wake_analysis`).
//!
//! The lock file is `~/.mahbot/wake_word.lock`.  The `File` handed back by
//! [`acquire_bench_lock`] keeps the kernel-level `flock` alive, and the kernel
//! releases it when the file descriptor is closed during process teardown —
//! normal `Drop`, `process::exit` (Rust destructors do NOT run, but the kernel
//! closes all fds) and SIGKILL alike — so there is no stale-lock scenario.
//!
//! This is a plain module, not a third bench target: cargo only auto-discovers
//! `benches/*.rs` and `benches/*/main.rs`, so each bench pulls it in with
//! `mod common;`.

use mahbot::util::lock::try_flock;
use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;
use std::time::Duration;

/// The one bench lock: both wake-word targets take the writer's lock, so the
/// analyser can never read a capture that is still growing.
pub(crate) const LOCK_NAME: &str = "wake_word.lock";

/// Resolve the benchmark lock file path under `~/.mahbot/`.
pub(crate) fn lock_file_path() -> PathBuf {
    mahbot::config::default_config_dir()
        .expect("Cannot resolve ~/.mahbot/ for lock file")
        .join(LOCK_NAME)
}

/// Acquire an exclusive lock on the benchmark, blocking until available.
///
/// Polls with `LOCK_EX | LOCK_NB` every 5 seconds, printing a status message on
/// the first failure and every 60 seconds thereafter.  The returned `File`
/// keeps the kernel-level `flock` alive (see the module doc for the release
/// guarantee).
///
/// # Panics
///
/// Panics if the lock directory cannot be created or if a non-retryable OS
/// error occurs.
pub(crate) fn acquire_bench_lock() -> File {
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

/// Run `bench` (or the analyser) under the single-instance lock and a
/// hung-run timeout.
///
/// Acquires the lock, then runs `bench` on a blocking tokio
/// thread under a `timeout_mins` timeout.  Aborts the process with exit code 1
/// if `bench` panics or the timeout fires.
///
/// NOTE: `spawn_blocking` tasks are NOT cancelable at the Rust level.  When the
/// timeout fires, tokio returns `Err(Elapsed)` but the kernel threads (ONNX
/// evaluations) continue executing.  We call `process::exit(1)` to terminate
/// the process, which kills all threads and lets the kernel release the flock.
pub(crate) fn run_with_timeout(timeout_mins: u64, bench: fn()) {
    let _lock = acquire_bench_lock();

    let runtime = tokio::runtime::Runtime::new()
        .expect("failed to create tokio runtime for benchmark timeout");

    let result = runtime.block_on(async {
        tokio::time::timeout(
            Duration::from_mins(timeout_mins),
            tokio::task::spawn_blocking(bench),
        )
        .await
    });

    match result {
        Ok(Ok(())) => {}
        Ok(Err(join_err)) => {
            eprintln!("BENCHMARK PANICKED: {join_err}");
            std::process::exit(1);
        }
        Err(_elapsed) => {
            eprintln!("BENCHMARK TIMED OUT after {timeout_mins} minutes");
            std::process::exit(1);
        }
    }
}
