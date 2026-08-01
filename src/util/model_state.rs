//! Shared model-loading state machine wrapper (mahbot-1043).
//!
//! Extracted from the duplicated `AtomicU8` lifecycle blocks (Uninit→Loading→
//! Ready/Failed) in `audio::tts`, `audio::local_transcriber`, and `embedder`.
//! This is an **extraction only** — each module keeps its own guard, `Drop`
//! behavior, and reset/retry paths.  Concurrency behavior is intentionally NOT
//! unified across modules (the ticket's explicit non-goal).
//!
//! Memory-ordering semantics are preserved exactly from the original copies:
//! `Acquire` loads, `Release` stores, `AcqRel`/`Acquire` compare-exchange.
//!
//! Note: `audio::voice` has its own local `ModelState`/`AtomicModelState`
//! (voice.rs) which is deliberately NOT migrated — the naming overlap is
//! intentional and scoped (mahbot-1043 manager pin).

use std::sync::atomic::{AtomicU8, Ordering};

/// Model loading state with type-safe atomic access.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub(crate) enum ModelState {
    Uninit = 0,
    Loading = 1,
    Ready = 2,
    Failed = 3,
}

impl ModelState {
    /// Decode a raw `u8` (used by test hooks that set state numerically).
    pub(crate) const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Loading,
            2 => Self::Ready,
            3 => Self::Failed,
            _ => Self::Uninit,
        }
    }
}

/// Atomic wrapper around [`ModelState`] that provides lock-free access.
pub(crate) struct AtomicModelState(AtomicU8);

impl AtomicModelState {
    pub(crate) const fn new(state: ModelState) -> Self {
        Self(AtomicU8::new(state as u8))
    }

    pub(crate) fn load(&self, order: Ordering) -> ModelState {
        ModelState::from_u8(self.0.load(order))
    }

    pub(crate) fn store(&self, state: ModelState, order: Ordering) {
        self.0.store(state as u8, order);
    }

    /// Atomically compare-and-exchange the current state.
    ///
    /// See [`AtomicU8::compare_exchange`] for ordering semantics.
    pub(crate) fn compare_exchange(
        &self,
        expected: ModelState,
        new: ModelState,
        success: Ordering,
        failure: Ordering,
    ) -> Result<ModelState, ModelState> {
        self.0
            .compare_exchange(expected as u8, new as u8, success, failure)
            .map(ModelState::from_u8)
            .map_err(ModelState::from_u8)
    }
}
