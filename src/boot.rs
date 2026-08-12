//! Boot-time diagnostics emitted before the tracing layer exists (the
//! pre-flight scan and the logs store's own heal run before `init_tracing`).
//! Written to stderr immediately and buffered for replay into the logs store
//! once tracing is live, so they appear in the GUI boot log.

use crate::util::UnwrapPoison;

static PRE_TRACING_DIAGNOSTICS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
static TRACING_INITIALIZED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Record a boot-time diagnostic: stderr now, logs store after tracing init.
pub(crate) fn boot_diagnostic(message: String) {
    eprintln!("[mahbot] {message}");
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
