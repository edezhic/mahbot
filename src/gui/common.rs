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
/// on error.  That page accesses the field directly to preserve the divergence.
#[derive(Debug, Clone)]
pub(crate) struct AsyncLoadState {
    pub(crate) loading: bool,
    pub(crate) has_loaded: bool,
    pub(crate) error: Option<String>,
}

impl AsyncLoadState {
    pub(crate) const fn new() -> Self {
        Self {
            loading: false,
            has_loaded: false,
            error: None,
        }
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
    /// can do so via direct field access.
    pub(crate) fn fail(&mut self, error: String) {
        self.error = Some(error);
        self.loading = false;
    }
}
