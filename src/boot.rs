//! Process start-up: the store bring-up sequence ([`open_stores`]) and the
//! diagnostics emitted before the tracing layer exists. Diagnostics are written
//! to stderr immediately (carrying a local-time timestamp for update.log
//! forensics) and buffered for replay into the logs store once tracing is live,
//! so they appear in the GUI boot log. A store that is not usable refuses the
//! start before either store is opened, and the refusal is also appended to the
//! storage root's durable failure record (`<root>/error.log`).

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;

use crate::util::UnwrapPoison;

static PRE_TRACING_DIAGNOSTICS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
static TRACING_INITIALIZED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Write a diagnostic line to stderr with a local-time timestamp prefix, so
/// lines captured in update.log (the replacement daemon's stderr) are
/// time-attributable during incident review.
pub(crate) fn timestamped_stderr(message: &str) {
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    eprintln!("[mahbot] [{ts}] {message}");
}

/// Record a boot-time diagnostic: stderr now, logs store after tracing init.
pub(crate) fn boot_diagnostic(message: String) {
    timestamped_stderr(&message);
    if TRACING_INITIALIZED.load(std::sync::atomic::Ordering::Acquire) {
        tracing::warn!("{message}");
    } else {
        PRE_TRACING_DIAGNOSTICS.lock().unwrap_poison().push(message);
    }
}

/// Mark tracing as initialized; subsequent diagnostics go straight to logs.
pub(crate) fn mark_tracing_initialized() {
    TRACING_INITIALIZED.store(true, std::sync::atomic::Ordering::Release);
}

/// Drop the pre-tracing buffer without replaying it. Called when
/// `init_tracing` fails — the messages were already written to stderr, so
/// nothing is lost; only the (failed) logs-store replay is skipped.
pub(crate) fn clear_boot_diagnostics() {
    PRE_TRACING_DIAGNOSTICS.lock().unwrap_poison().clear();
}

/// Replay buffered pre-tracing diagnostics through tracing (into the logs
/// store) and clear the buffer. Called right after `init_tracing` succeeds.
pub(crate) fn replay_boot_diagnostics() {
    let messages = {
        let mut buf = PRE_TRACING_DIAGNOSTICS.lock().unwrap_poison();
        std::mem::take(&mut *buf)
    };
    for m in messages {
        tracing::warn!("{m}");
    }
}

/// Open every store in the one supported boot order, returning the logs store
/// (the daemon keeps it for the boot log and the background tasks).
///
/// The order is load-bearing: the file-level shape check of BOTH stores runs
/// before anything opens a store (a stat plus one 18-byte header read — no
/// engine open, so a refusal here leaves both stores untouched), stale `.tshm`
/// debris is dropped, the logs store opens inside `init_tracing` (which
/// publishes the GUI log broadcast), then the process-global inits and the
/// consolidated domain store.
/// The store-lock check drives this same function, so a boot-order change cannot
/// silently weaken it.
///
/// `config::load_or_init()` must have run first: the storage root comes from
/// [`crate::config::CONFIG`], and an uninitialized root is a boot error here
/// rather than a panic.
pub async fn open_stores() -> anyhow::Result<Arc<crate::logs::LogStore>> {
    let root = crate::config::CONFIG.try_storage_root().context(
        "boot::open_stores: config::load_or_init() must run before the stores are opened",
    )?;
    // Stage 1: check BOTH stores' data files before either is opened. On an
    // unusable file, refuse here so the refusal leaves both stores untouched —
    // every unusable store is named, in the diagnostic lines and the record.
    let mut refusals: Vec<crate::db::StoreRefusal> = crate::db::wal_guard::scan_store_shapes(&root)
        .iter()
        .map(|store| {
            crate::db::StoreRefusal::new(store.store, &store.db_path, store.defect.reason())
        })
        .collect();
    if !refusals.is_empty() {
        for refusal in &refusals {
            record_refusal(&root, refusal);
        }
        // The first damaged store is also the refusal returned to the caller.
        return Err(refusals.remove(0).into());
    }
    crate::db::wal_guard::cleanup_stale_tshm(&root);
    let (log_store, log_broadcast) = crate::logs::init_tracing(&root)
        .await
        .map_err(|e| recorded_refusal(&root, "logs", e))?;
    let _ = crate::gui::LOG_BROADCAST.set(log_broadcast);
    crate::search_engine::init_global(); // sync — no I/O
    crate::pipeline::chronicle::init_global(); // sync — no I/O
    crate::agent::message_router::init_global()?;
    crate::audio::voice::init_global()?;
    crate::audio::tts::init_global()?;
    crate::db::init_all_stores()
        .await
        .map_err(|e| recorded_refusal(&root, "core", e))?;
    Ok(log_store)
}

/// Record a start-up refusal: the loud one-liner on the boot-diagnostic channel
/// (stderr now, logs store after tracing init) and the structured block in the
/// storage root's ONE durable failure record (`<root>/error.log`, the shared
/// writer).
fn record_refusal(root: &Path, refusal: &crate::db::StoreRefusal) {
    boot_diagnostic(refusal.to_string());
    write_refusal_record(root, refusal.store, &refusal.db_path, &refusal.reason);
}

/// Append the refusal block to the durable record and report where it landed.
/// Best-effort: the refusal is still returned (and shown on the start-failure
/// screen) if the record write fails.
fn write_refusal_record(root: &Path, store: &str, db_path: &Path, reason: &str) {
    match crate::db::failure_record::record_startup_refusal(root, store, db_path, reason) {
        Ok(path) => timestamped_stderr(&format!("start-up refusal recorded in {}", path.display())),
        Err(err) => timestamped_stderr(&format!("failed to record the start-up refusal: {err:#}")),
    }
}

/// Record a store bring-up refusal and pass the error through unchanged. The
/// refusal may sit under `.context()` wrappers, so the whole chain is searched.
/// A store held by another process is not a usability refusal — it is still
/// recorded, carrying the engine's own cause for the operator. Any other error
/// is returned untouched.
fn recorded_refusal(root: &Path, store: &'static str, e: anyhow::Error) -> anyhow::Error {
    if let Some(refusal) = e
        .chain()
        .find_map(|cause| cause.downcast_ref::<crate::db::StoreRefusal>())
    {
        record_refusal(root, refusal);
    } else if crate::db::is_store_lock_error(&e) {
        let reason = format!("a store is locked by another process — {e:#}");
        boot_diagnostic(format!("refusing to start: {reason}"));
        write_refusal_record(root, store, &crate::db::store_db_path(root, store), &reason);
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal is appended to the storage root's one durable failure record
    /// (`error.log`) with its store, path and reason; a refusal wrapped in
    /// `.context()` is still found; a non-refusal error passes through untouched
    /// and writes no record.
    #[test]
    fn startup_refusal_is_recorded_durably() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let root = tmp.path();
        let db_path = crate::db::store_db_path(root, "core");
        let refusal =
            crate::db::StoreRefusal::new("core", &db_path, "the data file is empty (0 bytes)");
        let err = recorded_refusal(root, "core", anyhow::Error::new(refusal));
        assert!(
            format!("{err:#}").starts_with("refusing to start: store 'core' is not usable"),
            "got: {err:#}"
        );

        let body = std::fs::read_to_string(root.join("error.log")).expect("error.log written");
        for needle in [
            "MahBot start-up refusal",
            "store: core",
            "db path:",
            "reason: the data file is empty (0 bytes)",
        ] {
            assert!(
                body.contains(needle),
                "error.log must contain {needle:?}: {body}"
            );
        }

        // A refusal under `.context()` wrappers is still found and recorded.
        let wrapped = anyhow::Error::new(crate::db::StoreRefusal::new(
            "logs",
            &crate::db::store_db_path(root, "logs"),
            "the data file is too short for a valid header (4 bytes)",
        ))
        .context("opening the logs store");
        let err = recorded_refusal(root, "logs", wrapped);
        assert!(
            format!("{err:#}").contains("refusing to start: store 'logs' is not usable"),
            "got: {err:#}"
        );
        let body = std::fs::read_to_string(root.join("error.log")).expect("error.log written");
        assert!(
            body.contains("store: logs") && body.contains("(4 bytes)"),
            "a wrapped refusal must still be recorded: {body}"
        );

        // A non-refusal error passes through and is not recorded.
        let passed = recorded_refusal(root, "core", anyhow::anyhow!("some other failure"));
        assert_eq!(format!("{passed:#}"), "some other failure");
        assert!(
            !std::fs::read_to_string(root.join("error.log"))
                .unwrap()
                .contains("some other failure"),
            "a non-refusal error must not be recorded"
        );
    }
}
