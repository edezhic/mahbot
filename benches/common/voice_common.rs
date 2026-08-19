//! Shared single-instance-lock + hung-run-timeout scaffolding for the voice
//! benchmark programs (`voice_pipeline_e2e` and `voice_pipeline_simple`).
//!
//! Each bench keeps its OWN lock file name so runs of the same bench
//! serialize against each other; the two benches share the local encoder, so
//! operators should not run them simultaneously.
//!
//! # Lock mechanism
//!
//! Uses [`mahbot::lock_utils::try_flock`] for the underlying file lock and a
//! lock file at `~/.mahbot/<lock_name>`.  The `lock_utils` module is the
//! canonical implementation shared with [`mahbot::self_update`].
//!
//! # Lock release guarantee
//!
//! The returned `File` keeps the kernel-level `flock` alive.  The kernel
//! releases the lock when the file descriptor is closed during process
//! teardown — this happens on normal `Drop`, on `process::exit` (Rust
//! destructors do NOT run, but the kernel closes all fds), and on SIGKILL.
//! There is no stale-lock scenario.

use mahbot::lock_utils::try_flock;
use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;
use std::time::Duration;

/// Resolve a benchmark lock file path under `~/.mahbot/`.
fn lock_file_path(lock_name: &str) -> PathBuf {
    mahbot::config::default_config_dir()
        .expect("Cannot resolve ~/.mahbot/ for lock file")
        .join(lock_name)
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
pub fn acquire_bench_lock(lock_name: &str) -> File {
    let lock_path = lock_file_path(lock_name);

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

/// Run a bench entry under a single-instance lock and a hung-run timeout.
///
/// Acquires the benchmark lock for `lock_name`, then runs `bench` on a
/// blocking tokio thread under a `timeout_mins` timeout.  Aborts the process
/// with exit code 1 if the bench panics or the timeout fires.
///
/// NOTE: `spawn_blocking` tasks are NOT cancelable at the Rust level.  When
/// the timeout fires, tokio returns `Err(Elapsed)` but the kernel threads
/// (ONNX evaluations) continue executing.  We call `process::exit(1)` to
/// terminate the process, which kills all threads and the kernel releases the
/// flock.
pub fn run_with_timeout(lock_name: &str, timeout_mins: u64, bench: fn()) {
    let _lock = acquire_bench_lock(lock_name);
    // Kernel releases `flock` on process death (Drop, process::exit, SIGKILL).

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
        Ok(Ok(())) => {
            // Normal completion within the timeout.
        }
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
