//! Shared model-loading state machine wrapper for the macOS audio subsystem.
//!
//! Extracted from the duplicated `AtomicU8` lifecycle blocks (Uninit→Loading→Ready/Failed)
//! in `audio::tts` and `audio::local_transcriber`: [`ModelLoadGuard`] unifies the byte-
//! identical Loading→Failed Drop guard that keeps a panicking background download from
//! leaving the state stuck in `Loading`.
//!
//! Deliberate non-unifications: the per-module singletons keep their own container types
//! (`OnceLock<RwLock<Option<_>>>` vs `Mutex<Option<_>>`) — unifying them would change
//! poisoning/init-once/contention semantics.
//!
//! Memory ordering is preserved exactly: `Acquire` loads, `Release` stores,
//! `AcqRel`/`Acquire` compare-exchange.
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
    /// Decode a raw `u8` back into a [`ModelState`] — the production decoder for
    /// [`AtomicModelState::load`] and `compare_exchange`.
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

    /// Returns `true` if the state is [`ModelState::Ready`].
    #[must_use]
    pub(crate) fn is_ready(&self) -> bool {
        self.load(Ordering::Acquire) == ModelState::Ready
    }

    /// Atomically compare-and-exchange the current state (see
    /// [`AtomicU8::compare_exchange`] for ordering).
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

    /// Atomically transition `from` → `to` (AcqRel success, Acquire failure), returning
    /// `true` if the CAS succeeded.
    ///
    /// A failed CAS means another task already moved the state, so the call is a no-op — the
    /// caller must not spawn duplicate work. A `Failed → Uninit` reset is normally followed by
    /// an `Uninit → Loading` site that spawns the download loop (two-step retry dance).
    #[must_use]
    pub(crate) fn transition(&self, from: ModelState, to: ModelState) -> bool {
        self.compare_exchange(from, to, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// Drop guard that transitions a model-loading state from [`ModelState::Loading`] to
/// [`ModelState::Failed`].
///
/// Tokio swallows a panic in the background download task; without this guard the state
/// would stay stuck in `Loading`, silently disabling the feature until restart. On success
/// or explicit terminal failure the CAS is a no-op.
pub(crate) struct ModelLoadGuard<'a>(&'a AtomicModelState);

impl<'a> ModelLoadGuard<'a> {
    pub(crate) const fn new(state: &'a AtomicModelState) -> Self {
        Self(state)
    }
}

impl Drop for ModelLoadGuard<'_> {
    fn drop(&mut self) {
        self.0
            .compare_exchange(
                ModelState::Loading,
                ModelState::Failed,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok();
    }
}

#[cfg(test)]
mod tests {
    use super::{AtomicModelState, ModelLoadGuard, ModelState};
    use std::sync::atomic::Ordering;

    #[test]
    fn guard_transitions_loading_to_failed_on_drop() {
        let state = AtomicModelState::new(ModelState::Loading);
        let guard = ModelLoadGuard::new(&state);
        drop(guard);
        assert_eq!(state.load(Ordering::Acquire), ModelState::Failed);
    }

    #[test]
    fn guard_drop_is_noop_on_terminal_states() {
        let ready = AtomicModelState::new(ModelState::Ready);
        drop(ModelLoadGuard::new(&ready));
        assert_eq!(ready.load(Ordering::Acquire), ModelState::Ready);

        let failed = AtomicModelState::new(ModelState::Failed);
        drop(ModelLoadGuard::new(&failed));
        assert_eq!(failed.load(Ordering::Acquire), ModelState::Failed);
    }
}
