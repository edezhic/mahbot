//! Process-wide runtime-change events for the GUI dashboard.
//!
//! GUI pages read in-memory registries (running agents, non-agent LLM calls, live transcript
//! snapshots) at render time — there is no poll loop. A mutation touching any of them
//! broadcasts a [`RuntimeEvent`] here and the dashboard re-renders on delivery. Emit sites are
//! fire-and-forget; the receiver coalesces, so a burst settles into one re-render.

use std::sync::OnceLock;

/// A GUI-relevant in-memory runtime source changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeEvent {
    /// Agent/non-agent registry lifecycle, activity instrumentation, or live transcript content.
    Registries,
    /// The local Telegram Bot API server's state: health, mode, version or last error.
    TelegramLocal,
    VoiceStatus,
    /// The self-update availability cache: a release was found or is no longer offered.
    UpdateAvailability,
}

pub(crate) static RUNTIME_EVENTS: OnceLock<tokio::sync::broadcast::Sender<RuntimeEvent>> =
    OnceLock::new();

/// Before the iced application runs (same convention as [`crate::gui::LOG_BROADCAST`]), so the
/// dashboard's subscription always has a source — an uninitialized sender would end the stream on
/// its first poll. Idempotent.
pub(crate) fn init_runtime_event_tx() {
    RUNTIME_EVENTS.get_or_init(|| tokio::sync::broadcast::channel(256).0);
}

/// Best-effort: a no-op when the channel is not yet initialised (tests, headless runs).
pub(crate) fn publish(event: RuntimeEvent) {
    if let Some(tx) = RUNTIME_EVENTS.get() {
        let _ = tx.send(event);
    }
}
