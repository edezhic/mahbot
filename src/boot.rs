//! Process start-up: the store bring-up sequence ([`open_stores`]), the retirement
//! of the superseded per-user speech model tree, and the diagnostics emitted before
//! the tracing layer exists. A diagnostic goes to stderr at once, timestamped
//! locally so a line kept in `daemon.log` or `update.log` is time-attributable, and
//! is buffered for replay into the logs store (the GUI boot log) once tracing is live.
//!
//! That stderr leg is best-effort: a detached launch's stderr is [`crate::detach`]'s
//! `daemon.log`, a launch given no standard streams prints into nothing, and the
//! update hand-off's is `update.log`. An unusable store refuses the start before
//! either store is opened, and every bring-up failure — the refusal, the
//! locked-store case, any other error — is appended to the storage root's durable
//! record (`<root>/error.log`), once per condition however often the launch repeats
//! (see [`failure_record::record`]); [`record_startup_failure`] covers the start-up
//! steps outside this module.

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;

use crate::db::failure_record::{self, FailureKind};
use crate::util::UnwrapPoison;

static PRE_TRACING_DIAGNOSTICS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
static TRACING_INITIALIZED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The daemon's single stderr funnel: boot diagnostics, the failure record's
/// stderr fallback, log-store warnings and the panic hook all pass through here.
/// It prints through [`crate::util::print_stderr`], so a failed write is discarded
/// and a panic in a caller (the panic hook) cannot abort the process on its way out.
pub(crate) fn timestamped_stderr(message: &str) {
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    crate::util::print_stderr(&format!("[mahbot] [{ts}] {message}"));
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

pub(crate) fn mark_tracing_initialized() {
    TRACING_INITIALIZED.store(true, std::sync::atomic::Ordering::Release);
}

/// Drop the pre-tracing buffer without replaying it, for when `init_tracing` fails:
/// the messages already went to the best-effort stderr channel, so only the (failed)
/// logs-store replay is skipped. The failure itself is filed in `error.log` by the
/// step that hit it.
pub(crate) fn clear_boot_diagnostics() {
    PRE_TRACING_DIAGNOSTICS.lock().unwrap_poison().clear();
}

/// Replay buffered pre-tracing diagnostics into the logs store, right after
/// `init_tracing` succeeds.
pub(crate) fn replay_boot_diagnostics() {
    let messages = {
        let mut buf = PRE_TRACING_DIAGNOSTICS.lock().unwrap_poison();
        std::mem::take(&mut *buf)
    };
    for m in messages {
        tracing::warn!("{m}");
    }
}

/// The one supported boot order, returning the logs store (kept for the boot log
/// and the background tasks): the file-level shape check of BOTH stores (a stat and
/// one 18-byte header read, no engine open, so a refusal leaves both untouched),
/// then the logs store inside `init_tracing` (which publishes the GUI log
/// broadcast), then the process-global inits and the consolidated domain store. The
/// store-lock check drives this same function, so a boot-order change cannot
/// silently weaken it.
///
/// `config::load_or_init()` must have run first: the storage root comes from
/// [`crate::config::CONFIG`], and an uninitialized root is a boot error here rather
/// than a panic.
pub async fn open_stores() -> anyhow::Result<Arc<crate::logs::LogStore>> {
    let root = crate::config::CONFIG
        .try_storage_root()
        .context("config::load_or_init() must run before the stores are opened")
        .map_err(|e| record_startup_failure("boot::open_stores", e))?;
    // Both stores' data files are checked before either is opened, so the refusal
    // leaves both untouched; every unusable store is named.
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
    crate::workspace_index::init_global(); // sync — no I/O
    crate::pipeline::chronicle::init_global(); // sync — no I/O
    // Not stores, so a failure names no store — but it is still a start-up failure
    // and must be recorded, not lost.
    crate::agent::message_router::init_global()
        .map_err(|e| record_startup_failure("agent::message_router::init_global", e))?;
    // The microphone, the voice pipeline and the wake word.
    crate::audio::voice::init_global()
        .map_err(|e| record_startup_failure("audio::voice::init_global", e))?;
    // The retired per-user model tree is dead weight now: delete it once, off the boot path.
    // Nothing reads it any more, and a failure only leaves it there.
    drop(tokio::task::spawn_blocking(
        crate::audio::remove_retired_models,
    ));
    crate::db::init_all_stores()
        .await
        .map_err(|e| record_bring_up_failure(&root, "core", e))?;
    Ok(log_store)
}

/// Point the operator at the durable record when the block was filed. The stderr
/// fallback already carries the full block, so a failed write adds nothing here.
fn note_recorded(what: &str, filed: Option<failure_record::Filed>) {
    if let Some(pointer) = failure_record::recorded_pointer(what, filed) {
        boot_diagnostic(pointer);
    }
}

/// The loud one-liner on the boot-diagnostic channel plus the structured block in
/// the storage root's ONE durable record (`<root>/error.log`). Filed once per
/// condition ([`failure_record::record`]).
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

/// Passes the error through unchanged. A refusal (possibly under `.context()`
/// wrappers, so the whole chain is searched) goes through [`record_refusal`]; a
/// store locked by another process is an environment-caused refusal; any other error
/// is a start-up failure naming the store and its path, with the environment note
/// when the cause is external rather than store damage. Filed once per condition.
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

/// For start-up failures that are not store bring-up errors (a config load, a
/// provider, another global init). Returns `e` unchanged — the start-failure screen
/// shows it, while the record names `context` as the failing step. Callers needing
/// a prefix on the screen must put it in the error they pass.
///
/// The storage root is resolved best-effort; unresolved (a failure before config
/// init) or unwritable, [`failure_record::record`] puts the full block on stderr.
///
/// Public and hidden for the binary's pre-store boot steps.
#[must_use]
#[doc(hidden)]
pub fn record_startup_failure(context: &str, e: anyhow::Error) -> anyhow::Error {
    let block = start_up_block(context, &e);
    note_recorded(
        "start-up failure",
        failure_record::record(crate::config::CONFIG.try_storage_root().as_deref(), &block),
    );
    e
}

/// Like [`record_startup_failure`], under an explicitly given storage root: the
/// binary's pre-boot steps run before `config::load_or_init()` sets the global one,
/// and file under the root they already resolved. The error text is returned
/// unchanged so the caller's own report is unaffected.
///
/// Filed once per condition (step plus cause class), so a double-launch refused
/// again and again does not grow the file.
#[must_use]
#[doc(hidden)]
pub fn record_launch_failure(root: &Path, context: &str, e: anyhow::Error) -> anyhow::Error {
    let block = start_up_block(context, &e);
    note_recorded(
        "start-up failure",
        failure_record::record(Some(root), &block),
    );
    e
}

/// The failing step as its own line and the reason as the error chain alone — so
/// the step is identity (see [`failure_record`]'s condition rule) rather than a
/// prefix of a reason two occurrences may word differently.
fn start_up_block(context: &str, e: &anyhow::Error) -> String {
    failure_record::start_up_report(
        FailureKind::StartUpFailure,
        None,
        format!("{e:#}"),
        crate::db::is_actionable_signal(e),
    )
    .step(context)
    .render()
}

/// The start-failure screen renders the error [`record_startup_failure`] returns
/// verbatim, so the qualifier lives in the error, not in the recorder's step context.
#[must_use]
#[doc(hidden)]
pub fn startup_panic_error(message: &str) -> anyhow::Error {
    anyhow::anyhow!("Startup panicked: {message}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refusal lands in `error.log` with its store, path and reason, and is still
    /// found under `.context()` wrappers; an environment-caused refusal carries the
    /// cause line.
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

        // An unreadable file is not evidence of damage, so the refusal says so.
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

    /// Never silent: a non-refusal is recorded as a start-up failure naming the store
    /// and its path and passes through unchanged. Outside the stores there is neither,
    /// so the step takes that place. A resource/permission cause is marked
    /// environment-caused; a cause saying nothing about the environment is not.
    #[test]
    fn a_failure_that_is_no_refusal_is_recorded_as_a_start_up_failure() {
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

        // Outside the stores: no store, no path, the step named in their place.
        let passed = record_launch_failure(
            root,
            "providers::init_global",
            anyhow::anyhow!("no provider credential"),
        );
        assert_eq!(format!("{passed:#}"), "no provider credential");
        let body = std::fs::read_to_string(root.join("error.log")).expect("error.log written");
        for needle in [
            "MahBot start-up failure",
            failure_record::NO_STORE,
            failure_record::NO_DB_PATH,
            "step: providers::init_global",
            "reason: no provider credential",
        ] {
            assert!(
                body.contains(needle),
                "error.log must contain {needle:?}: {body}"
            );
        }
    }
}
