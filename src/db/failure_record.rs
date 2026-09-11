//! The service's ONE durable failure record: `<root>/error.log`.
//!
//! Two writers append to it — the checkpoint watchdog (a persistent checkpoint
//! failure, see [`crate::db::checkpoint`]) and the boot store-usability gate (a
//! refusal to start, see [`crate::boot`]). Both live here so there is one file,
//! one format, and one place that knows how to append a block without tearing.

use std::path::Path;

/// Append `report` as a block to `<root>/error.log`, creating it if absent.
/// Returns the log path. Pure `std::fs` — no async, never panics on the
/// caller's behalf. The report + terminator go out as a single `write_all`
/// (one O_APPEND write in practice); even if the libc layer splits a large
/// buffer, each chunk is offset-atomic, so the worst case under concurrent
/// store failures is interleaved chunks, never torn bytes.
pub(crate) fn append_failure_record(
    root: &Path,
    report: &str,
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;
    let path = root.join("error.log");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    file.write_all(format!("{report}\n").as_bytes())?;
    Ok(path)
}

/// Append a start-up refusal block to `<root>/error.log`. Best-effort at the
/// call site: the refusal is returned and shown on the start-failure screen
/// regardless.
pub(crate) fn record_startup_refusal(
    root: &Path,
    store: &str,
    db_path: &Path,
    reason: &str,
) -> std::io::Result<std::path::PathBuf> {
    let report = format!(
        "MahBot start-up refusal — {}\nstore: {store}\ndb path: {}\nreason: {reason}\n",
        chrono::Utc::now().to_rfc3339(),
        db_path.display(),
    );
    append_failure_record(root, &report)
}
