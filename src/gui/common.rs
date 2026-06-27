//! Shared state types used across GUI pages.

/// Shared async loading state, used by GUI pages that fetch data asynchronously.
///
/// Combines the three common fields (`loading`, `has_loaded`, `error`) into a single
/// struct with helper methods for the standard lifecycle:
/// 1. [`start_loading`](AsyncLoadState::start_loading) — called before a fetch
/// 2. [`finish_loading`](AsyncLoadState::finish_loading) — on success
/// 3. [`fail`](AsyncLoadState::fail) — on error
///
/// # Behavioural note
///
/// Most pages set `has_loaded` only on success, but `ToolFailuresState` also sets it
/// on error.  That page uses [`set_has_loaded`](AsyncLoadState::set_has_loaded) to
/// preserve the divergence without exposing the fields directly.
#[derive(Debug, Clone)]
pub(crate) struct AsyncLoadState {
    loading: bool,
    has_loaded: bool,
    error: Option<String>,
}

impl AsyncLoadState {
    pub(crate) const fn new() -> Self {
        Self {
            loading: false,
            has_loaded: false,
            error: None,
        }
    }

    /// Returns `true` while an async fetch is in progress.
    pub(crate) fn loading(&self) -> bool {
        self.loading
    }

    /// Returns `true` after at least one successful fetch has completed.
    pub(crate) fn has_loaded(&self) -> bool {
        self.has_loaded
    }

    /// The last error message, if any.
    pub(crate) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Mark the start of an async load — clears any previous error.
    pub(crate) fn start_loading(&mut self) {
        self.loading = true;
        self.error = None;
    }

    /// Mark successful completion of an async load.
    pub(crate) fn finish_loading(&mut self) {
        self.loading = false;
        self.has_loaded = true;
    }

    /// Mark failure of an async load.
    ///
    /// Note: does **not** touch `has_loaded` — most pages leave it at its previous
    /// value (the initial `false`) so the view continues to show "Loading…".
    /// Pages that need to set `has_loaded = true` on error (e.g. `ToolFailuresState`)
    /// can do so via [`set_has_loaded`](Self::set_has_loaded).
    pub(crate) fn fail(&mut self, error: String) {
        self.error = Some(error);
        self.loading = false;
    }

    /// Clear the error state without starting a new load.
    ///
    /// Used after a successful operation (e.g. delete) that should dismiss any
    /// prior error banner without re-triggering the loading spinner.
    pub(crate) fn clear_error(&mut self) {
        self.error = None;
    }

    /// Mark `has_loaded` as `true` regardless of error state.
    ///
    /// Only used by `ToolFailuresState` which shows an empty state instead of
    /// "Loading…" after the first attempt, even on failure.
    pub(crate) fn set_has_loaded(&mut self) {
        self.has_loaded = true;
    }
}
