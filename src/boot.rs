//! Process start-up: the store bring-up sequence ([`open_stores`]) and the
//! diagnostics emitted before the tracing layer exists (the pre-flight scan and
//! the logs store's own heal run before `init_tracing`). Diagnostics are
//! written to stderr immediately (carrying a local-time timestamp for
//! update.log forensics) and buffered for replay into the logs store once
//! tracing is live, so they appear in the GUI boot log.

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
/// The order is load-bearing: the pre-flight classification runs before anything
/// opens a store (turso's own reopen would consume the evidence, and the
/// whole-file lock it takes is never re-taken), stale `.tshm` debris is dropped,
/// the logs store opens inside `init_tracing` (which publishes the GUI log
/// broadcast), then the process-global inits and the consolidated domain store.
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
    crate::db::wal_guard::diagnose_all_stores(&root);
    crate::db::wal_guard::cleanup_stale_tshm(&root);
    let (log_store, log_broadcast) = crate::logs::init_tracing(&root)
        .await
        .map_err(recorded_refusal)?;
    let _ = crate::gui::LOG_BROADCAST.set(log_broadcast);
    crate::search_engine::init_global(); // sync — no I/O
    crate::pipeline::chronicle::init_global(); // sync — no I/O
    crate::agent::message_router::init_global()?;
    crate::audio::voice::init_global()?;
    crate::audio::tts::init_global()?;
    crate::db::init_all_stores()
        .await
        .map_err(recorded_refusal)?;
    Ok(log_store)
}

/// Pass a store bring-up failure through, recording it first when it is another
/// process holding a store locked: a store the service cannot lock is a start-up
/// refusal, and this channel survives the case where nothing else can record it
/// (a refusal on the `logs.db` open happens before tracing exists, and
/// `init_tracing`'s failure path drops the pre-tracing buffer).
fn recorded_refusal(e: anyhow::Error) -> anyhow::Error {
    if crate::db::is_store_lock_error(&e) {
        boot_diagnostic(format!(
            "refusing to start: a store is locked by another process — {e:#}"
        ));
    }
    e
}
