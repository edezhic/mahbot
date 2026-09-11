//! Process start-up: the store bring-up sequence ([`open_stores`]) and the
//! diagnostics emitted before the tracing layer exists. Diagnostics are written
//! to stderr immediately (carrying a local-time timestamp for update.log
//! forensics) and buffered for replay into the logs store once tracing is live,
//! so they appear in the GUI boot log. A store that is not usable refuses the
//! start before either store is opened; every store bring-up failure — the
//! refusal, the locked-store case, and any other error — is also appended to the
//! storage root's durable failure record (`<root>/error.log`), which is why
//! [`record_startup_failure`] exists for the start-up steps outside this module.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;

use crate::db::failure_record::{self, FailureKind};
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
/// engine open, so a refusal here leaves both stores untouched), the logs store
/// opens inside `init_tracing` (which publishes the GUI log broadcast), then the
/// process-global inits and the consolidated domain store.
/// The store-lock check drives this same function, so a boot-order change cannot
/// silently weaken it.
///
/// `config::load_or_init()` must have run first: the storage root comes from
/// [`crate::config::CONFIG`], and an uninitialized root is a boot error here
/// rather than a panic.
pub async fn open_stores() -> anyhow::Result<Arc<crate::logs::LogStore>> {
    let root = crate::config::CONFIG
        .try_storage_root()
        .context("config::load_or_init() must run before the stores are opened")
        .map_err(|e| record_startup_failure("boot::open_stores", e))?;
    // Stage 1: check BOTH stores' data files before either is opened. On an
    // unusable file, refuse here so the refusal leaves both stores untouched —
    // every unusable store is named, in the diagnostic lines and the record.
    let refusals: Vec<crate::db::StoreRefusal> = crate::db::wal_guard::scan_store_shapes(&root)
        .iter()
        .map(|store| {
            crate::db::StoreRefusal::new(store.store, &store.db_path, store.defect.reason())
                .with_environment(store.defect.is_environment_caused())
        })
        .collect();
    if !refusals.is_empty() {
        for refusal in &refusals {
            record_refusal(&root, refusal);
        }
        // The boot screen must name every unusable store, not only the first.
        let named = refusals
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ");
        return Err(anyhow::anyhow!(named));
    }
    let (log_store, log_broadcast) = crate::logs::init_tracing(&root)
        .await
        .map_err(|e| record_bring_up_failure(&root, "logs", e))?;
    let _ = crate::gui::LOG_BROADCAST.set(log_broadcast);
    crate::search_engine::init_global(); // sync — no I/O
    crate::pipeline::chronicle::init_global(); // sync — no I/O
    // These globals are not stores, so a failure here has no store to name —
    // but it is still a start-up failure and must be recorded, not lost.
    crate::agent::message_router::init_global()
        .map_err(|e| record_startup_failure("agent::message_router::init_global", e))?;
    crate::audio::voice::init_global()
        .map_err(|e| record_startup_failure("audio::voice::init_global", e))?;
    crate::audio::tts::init_global()
        .map_err(|e| record_startup_failure("audio::tts::init_global", e))?;
    crate::db::init_all_stores()
        .await
        .map_err(|e| record_bring_up_failure(&root, "core", e))?;
    Ok(log_store)
}

/// Point the operator at the durable record when the block was filed, on the
/// boot-diagnostic channel (stderr now, logs store after tracing init). The
/// stderr fallback already carries the full block, so a failed write adds
/// nothing here.
fn note_recorded(what: &str, path: Option<std::path::PathBuf>) {
    if let Some(pointer) = failure_record::recorded_pointer(what, path) {
        boot_diagnostic(pointer);
    }
}

/// Record a start-up refusal: the loud one-liner on the boot-diagnostic channel
/// (stderr now, logs store after tracing init) and the structured block in the
/// storage root's ONE durable failure record (`<root>/error.log`, the shared
/// writer) — plus a diagnostic line pointing the operator at the file it landed
/// in when the write succeeded.
fn record_refusal(root: &Path, refusal: &crate::db::StoreRefusal) {
    boot_diagnostic(refusal.to_string());
    note_recorded(
        "start-up refusal",
        failure_record::record_startup_refusal(
            root,
            refusal.store,
            &refusal.db_path,
            &refusal.reason,
            refusal.environment,
        ),
    );
}

/// Record a store bring-up failure and pass the error through unchanged. The
/// rule for all three branches: a refusal (possibly under `.context()`
/// wrappers, so the whole chain is searched) goes through [`record_refusal`]; a
/// store locked by another process is recorded as an environment-caused
/// refusal; every other bring-up error is a start-up failure naming the store
/// and its path, with the environment note when the cause is an external
/// condition rather than store damage.
fn record_bring_up_failure(root: &Path, store: &'static str, e: anyhow::Error) -> anyhow::Error {
    if let Some(refusal) = e
        .chain()
        .find_map(|cause| cause.downcast_ref::<crate::db::StoreRefusal>())
    {
        record_refusal(root, refusal);
    } else if crate::db::is_store_lock_error(&e) {
        let reason = format!("a store is locked by another process — {e:#}");
        boot_diagnostic(format!("refusing to start: {reason}"));
        note_recorded(
            "start-up refusal",
            failure_record::record_startup_refusal(
                root,
                store,
                &crate::db::store_db_path(root, store),
                &reason,
                true,
            ),
        );
    } else {
        let report = failure_record::start_up_report(
            FailureKind::StartUpFailure,
            Some((store, crate::db::store_db_path(root, store))),
            format!("{e:#}"),
            crate::db::is_actionable_signal(&e),
        );
        note_recorded(
            "start-up failure",
            failure_record::record(Some(root), &report.render()),
        );
    }
    e
}

/// Record a start-up failure that is not a store bring-up error (a config load,
/// a provider or another global init) and return `e` unchanged — the returned
/// error is what the start-failure screen shows, while the record's reason is
/// prefixed with `context`, so the two texts are not identical by construction.
/// Callers that need the screen to carry a prefix must put it in the error they
/// pass (see the panic arm in the binary).
///
/// The storage root is resolved best-effort; when it is not resolved yet (a
/// failure before config init) or cannot be written, [`failure_record::record`]
/// puts the full block on stderr instead.
///
/// Public (and hidden from the library's docs) for the binary's pre-store boot
/// steps, the sibling of [`open_stores`].
#[must_use]
#[doc(hidden)]
pub fn record_startup_failure(context: &str, e: anyhow::Error) -> anyhow::Error {
    record_startup_failure_at(
        crate::config::CONFIG.try_storage_root().as_deref(),
        context,
        e,
    )
}

/// The error a panic of the boot sequence is recorded and shown as: the panic
/// summary qualified as a start-up panic. The start-failure screen renders the
/// error [`record_startup_failure`] returns verbatim, so the qualifier lives in
/// the error rather than in the recorder's step context.
#[must_use]
#[doc(hidden)]
pub fn startup_panic_error(message: &str) -> anyhow::Error {
    anyhow::anyhow!("Startup panicked: {message}")
}

/// [`record_startup_failure`] with an explicit storage root (the injectable
/// seam: the production root is the process-global one).
#[must_use]
fn record_startup_failure_at(
    root: Option<&Path>,
    context: &str,
    e: anyhow::Error,
) -> anyhow::Error {
    let report = failure_record::start_up_report(
        FailureKind::StartUpFailure,
        None,
        format!("{context}: {e:#}"),
        crate::db::is_actionable_signal(&e),
    );
    note_recorded(
        "start-up failure",
        failure_record::record(root, &report.render()),
    );
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal is appended to the storage root's one durable failure record
    /// (`error.log`) with its store, path and reason; a refusal wrapped in
    /// `.context()` is still found; an environment-caused refusal carries the
    /// cause line; and a non-refusal bring-up error is recorded as a start-up
    /// failure (never silent), passing through untouched.
    #[test]
    fn startup_refusal_is_recorded_durably() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let root = tmp.path();
        let db_path = crate::db::store_db_path(root, "core");
        let refusal =
            crate::db::StoreRefusal::new("core", &db_path, "the data file is empty (0 bytes)");
        let err = record_bring_up_failure(root, "core", anyhow::Error::new(refusal));
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
        let err = record_bring_up_failure(root, "logs", wrapped);
        assert!(
            format!("{err:#}").contains("refusing to start: store 'logs' is not usable"),
            "got: {err:#}"
        );
        let body = std::fs::read_to_string(root.join("error.log")).expect("error.log written");
        assert!(
            body.contains("store: logs") && body.contains("(4 bytes)"),
            "a wrapped refusal must still be recorded: {body}"
        );

        // An environment-caused refusal (an unreadable file is not evidence of
        // damage) is recorded as such.
        let defect = crate::db::wal_guard::ShapeDefect::Unreadable;
        assert!(
            defect.is_environment_caused(),
            "a file that could not be read is not evidence of damage"
        );
        let unreadable = crate::db::StoreRefusal::new("core", &db_path, defect.reason())
            .with_environment(defect.is_environment_caused());
        record_bring_up_failure(root, "core", anyhow::Error::new(unreadable));
        let body = std::fs::read_to_string(root.join("error.log")).expect("error.log written");
        assert!(
            body.contains(failure_record::ENVIRONMENT_CAUSE),
            "an environment-caused refusal must say so: {body}"
        );
    }

    /// A store bring-up error that is not a refusal is recorded as a start-up
    /// failure naming the store and its path, and passes the error through
    /// unchanged. A resource/permission cause is marked environment-caused; a
    /// cause that says nothing about the environment is not.
    #[test]
    fn non_refusal_bring_up_failure_is_recorded() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let root = tmp.path();

        let passed = record_bring_up_failure(
            root,
            "core",
            anyhow::anyhow!("permission denied").context("opening the core store"),
        );
        assert_eq!(
            format!("{passed:#}"),
            "opening the core store: permission denied"
        );
        let body = std::fs::read_to_string(root.join("error.log")).expect("error.log written");
        for needle in [
            "MahBot start-up failure",
            "store: core",
            "db path:",
            failure_record::ENVIRONMENT_CAUSE,
            "reason: opening the core store: permission denied",
        ] {
            assert!(
                body.contains(needle),
                "error.log must contain {needle:?}: {body}"
            );
        }

        record_bring_up_failure(root, "logs", anyhow::anyhow!("the schema probe failed"));
        let body = std::fs::read_to_string(root.join("error.log")).expect("error.log written");
        assert!(
            body.contains("reason: the schema probe failed"),
            "a non-environment failure must still be recorded: {body}"
        );
        assert_eq!(
            body.matches(failure_record::ENVIRONMENT_CAUSE).count(),
            1,
            "only the environment-caused failure may carry the cause line: {body}"
        );
    }

    /// The start-up-failure writer (config, providers, the process-global
    /// inits) files a block naming the step and the reason, and passes the error
    /// through unchanged.
    #[test]
    fn non_store_startup_failure_is_recorded() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let passed = record_startup_failure_at(
            Some(tmp.path()),
            "providers::init_global",
            anyhow::anyhow!("no provider credential"),
        );
        assert_eq!(format!("{passed:#}"), "no provider credential");
        let body =
            std::fs::read_to_string(tmp.path().join("error.log")).expect("error.log written");
        for needle in [
            "MahBot start-up failure",
            failure_record::UNKNOWN_STORE,
            failure_record::UNKNOWN_DB_PATH,
            "reason: providers::init_global: no provider credential",
        ] {
            assert!(
                body.contains(needle),
                "error.log must contain {needle:?}: {body}"
            );
        }
    }

    /// The boot-panic path's screen text: the panic summary stays qualified as
    /// such (the start-failure screen renders the returned error verbatim), and
    /// the record's reason carries that same sentence under the step context.
    #[test]
    fn startup_panic_keeps_its_qualifier() {
        let tmp = tempfile::TempDir::new().expect("temp dir for test");
        let returned =
            record_startup_failure_at(Some(tmp.path()), "bootstrap", startup_panic_error("boom"));
        assert_eq!(
            format!("{returned:#}"),
            "Startup panicked: boom",
            "the start-failure screen shows the returned error verbatim"
        );
        let body =
            std::fs::read_to_string(tmp.path().join("error.log")).expect("error.log written");
        assert!(
            body.contains("reason: bootstrap: Startup panicked: boom"),
            "the record keeps the same sentence under the step context: {body}"
        );
    }
}
