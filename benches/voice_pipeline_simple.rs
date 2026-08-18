/// Simple voice pipeline benchmark (three plain metrics) with a
/// single-instance lock and a hung-run timeout.
///
/// This is the fast baseline bench (mahbot-1804): recognition (X of 40
/// phrase utterances), false reactions (N on the 113 non-phrase set + a rate
/// per hour on real audio), and data coverage — nothing else.
///
/// # Budget
///
/// The whole WARM run is capped at ~10 minutes (the pinned real-audio subset
/// is sized so the parallel feed fits; the synthetic phases reuse the TTS PCM
/// cache).  The timeout is a fast-fail hang guard only: 15 minutes.
///
/// # Lock mechanism
///
/// Uses [`mahbot::lock_utils::try_flock`] for the underlying file lock and a
/// lock file at `~/.mahbot/voice_pipeline_simple.lock` (its own lock file,
/// distinct from the e2e bench's — each bench serializes its own runs; the
/// two benches share the encoder, so operators should not run them
/// simultaneously).  The `lock_utils` module is the canonical implementation
/// shared with [`mahbot::self_update`].
///
/// # Lock release guarantee
///
/// The returned `File` keeps the kernel-level `flock` alive.  The kernel
/// releases the lock when the file descriptor is closed during process
/// teardown — this happens on normal `Drop`, on `process::exit` (Rust
/// destructors do NOT run, but the kernel closes all fds), and on SIGKILL.
/// There is no stale-lock scenario.
use mahbot::lock_utils::try_flock;
use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;
use std::time::Duration;

// ── Single-instance lock ──────────────────────────────────────────────────

fn lock_file_path() -> PathBuf {
    mahbot::config::default_config_dir()
        .expect("Cannot resolve ~/.mahbot/ for lock file")
        .join("voice_pipeline_simple.lock")
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

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    // 1. Acquire single-instance lock.
    //    Prevents concurrent benchmark runs.
    let _lock = acquire_bench_lock();
    // Kernel releases `flock` on process death (Drop, process::exit, SIGKILL).

    // 2. Create tokio runtime for the timeout.
    let runtime = tokio::runtime::Runtime::new()
        .expect("failed to create tokio runtime for benchmark timeout");

    // 3. Run the benchmark with a hung-run timeout.
    //    The warm run is budgeted at ~10 min; 15 min is a fast-fail hang
    //    guard with headroom.
    // NOTE: spawn_blocking tasks are NOT cancelable at the Rust level.
    // When the timeout fires, tokio returns Err(Elapsed) but the kernel
    // threads (ONNX evaluations) continue executing.  We call
    // process::exit(1) to terminate the process, which kills all threads
    // and the kernel releases the flock.
    let timeout_mins: u64 = 15;
    let result = runtime.block_on(async {
        tokio::time::timeout(
            Duration::from_mins(timeout_mins),
            tokio::task::spawn_blocking(|| {
                mahbot::audio::voice::run_simple_voice_pipeline_benchmark();
            }),
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
