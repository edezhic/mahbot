//! Shared model-loading state machine wrapper.
//!
//! Extracted from the duplicated `AtomicU8` lifecycle blocks (Uninit→Loading→
//! Ready/Failed) in `audio::tts`, `audio::local_transcriber`, and `embedder`.
//! `ModelState`/`AtomicModelState` provide the shared state representation;
//! [`ModelLoadGuard`] unifies the byte-identical Loading→Failed Drop guard used
//! by `embedder` and `audio::tts` for panic safety in background download tasks.
//!
//! Deliberate non-unifications (semantics-preserving scope):
//! * The per-module singletons keep their own container/value types — `embedder`
//!   uses `RwLock<Option<_>>`, `tts` `OnceLock<RwLock<Option<_>>>`, and
//!   `local_transcriber` `Mutex<Option<_>>`.  Unifying these would change
//!   synchronization semantics (poisoning, init-once, contention).
//! * `audio::local_transcriber` has no Drop guard: a panic in its download loop
//!   would otherwise flip a stuck-Loading state to Failed, an observable
//!   failure-behavior change outside this scope.
//!
//! Memory-ordering semantics are preserved exactly from the original copies:
//! `Acquire` loads, `Release` stores, `AcqRel`/`Acquire` compare-exchange.
//!
//! All consumers (`audio::voice`, `audio::tts`, `audio::local_transcriber`,
//! `embedder`) use this shared copy; no module-local duplicates remain.

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

    /// Returns `true` if the state is [`ModelState::Ready`].
    #[must_use]
    pub(crate) fn is_ready(&self) -> bool {
        self.load(Ordering::Acquire) == ModelState::Ready
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

    /// Atomically transition `from` → `to` (AcqRel success, Acquire failure
    /// ordering), returning `true` if the CAS succeeded.
    ///
    /// A failed CAS means another task already moved the state (e.g. a
    /// concurrent retry or a still-running download loop), so the call is a
    /// no-op — the caller must not spawn duplicate work.  A `Failed → Uninit`
    /// reset is normally followed by an `Uninit → Loading` site that spawns
    /// the download loop (two-step retry dance).
    #[must_use]
    pub(crate) fn transition(&self, from: ModelState, to: ModelState) -> bool {
        self.compare_exchange(from, to, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// Drop guard that transitions a model-loading state from [`ModelState::Loading`]
/// to [`ModelState::Failed`].
///
/// If the background download task panics, Tokio catches the panic and swallows
/// it; without this guard the state would remain permanently stuck in
/// [`ModelState::Loading`], silently disabling the feature until process
/// restart.  On normal success or explicit terminal failure the CAS is a no-op
/// (wrong expected value).
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
