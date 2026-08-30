//! Process-wide runtime-change events for the GUI dashboard.
//!
//! GUI pages read in-memory registries (running agents, non-agent LLM calls,
//! live transcript snapshots) and the voice pipeline status at render time —
//! there is no poll loop. When a mutation touches any of those in-memory
//! sources it broadcasts a [`RuntimeEvent`] here and the dashboard re-renders
//! on delivery. Emit sites are fire-and-forget; the receiver (the GUI) owns a
//! coalescing stream so a burst of mutations settles into a single re-render.

use std::sync::OnceLock;

/// A GUI-relevant in-memory runtime source changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeEvent {
    /// Agent/non-agent registry lifecycle, live tool/activity instrumentation,
    /// or live transcript content changed.
    Registries,
    /// The voice pipeline status value changed.
    VoiceStatus,
}

/// Process-wide broadcast of runtime-change events.
pub(crate) static RUNTIME_EVENTS: OnceLock<tokio::sync::broadcast::Sender<RuntimeEvent>> =
    OnceLock::new();

/// Initialise the runtime-change broadcast before the iced application runs
/// (same convention as [`crate::gui::LOG_BROADCAST`]) so the dashboard's
/// subscription always has a source — an uninitialized sender would end the
/// subscription stream on its first poll. Idempotent; called from `main`
/// via the [`crate::gui`] warm-up façade.
pub(crate) fn init_runtime_event_tx() {
    RUNTIME_EVENTS.get_or_init(|| tokio::sync::broadcast::channel(256).0);
}

/// Broadcast a runtime-change event. Best-effort: when the channel is not yet
/// initialised (tests / headless runs) this is a no-op.
pub(crate) fn publish(event: RuntimeEvent) {
    if let Some(tx) = RUNTIME_EVENTS.get() {
        let _ = tx.send(event);
    }
}
