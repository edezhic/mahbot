//! Shared state types used across GUI pages.

use iced::Task;

/// Pagination state shared by dashboard pages that display paginated data.
///
/// Groups `page`, `page_size`, and `total` into a single struct with helper
/// methods for common operations.  Used by [`PaginationState`] and the
/// [`pagination_bar`](super::widgets::pagination_bar) widget.
///
/// # Structural benefits (not line savings)
///
/// The struct adds a few lines of definition, but the value is:
/// - Cleaner [`pagination_bar`](super::widgets::pagination_bar) signature
///   (takes `page` / `total_pages` instead of needing the whole state object)
/// - Centralised boundary logic in [`prev_page`](Self::prev_page) /
///   [`next_page`](Self::next_page)
/// - Reusable by any future page that needs pagination
#[derive(Debug, Clone)]
pub(crate) struct PaginationState {
    pub(crate) page: usize,
    pub(crate) page_size: usize,
    pub(crate) total: usize,
}

impl PaginationState {
    pub(crate) const fn new(page_size: usize) -> Self {
        Self {
            page: 0,
            page_size,
            total: 0,
        }
    }

    /// Total number of pages given the current `total` and `page_size`.
    pub(crate) const fn total_pages(&self) -> usize {
        if self.total == 0 {
            0
        } else {
            self.total.div_ceil(self.page_size)
        }
    }

    /// Move to the previous page.  Returns `true` if the page changed.
    pub(crate) fn prev_page(&mut self) -> bool {
        if self.page > 0 {
            self.page -= 1;
            true
        } else {
            false
        }
    }

    /// Move to the next page.  Returns `true` if the page changed.
    pub(crate) fn next_page(&mut self) -> bool {
        if self.page + 1 < self.total_pages() {
            self.page += 1;
            true
        } else {
            false
        }
    }

    /// Reset to page 0 (e.g. when a filter changes).
    pub(crate) fn reset(&mut self) {
        self.page = 0;
    }

    /// Compute the offset for SQL ``LIMIT … OFFSET …`` queries.
    pub(crate) fn offset(&self) -> usize {
        self.page * self.page_size
    }
}

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

// ── Debounce state ──────────────────────────────────────────────────

/// Debounce state for search/filter text inputs.
///
/// Groups the generation counter and pending flag from the manual debounce
/// pattern into a single struct.  The caller keeps a `DebounceState` field,
/// calls [`trigger`](Self::trigger) on input changes, and calls
/// [`should_process`](Self::should_process) in the response handler.
///
/// # Pattern
///
/// ```ignore
/// // In the input handler:
/// self.debounce.trigger(300).map(MyMessage::DebouncedRefresh)
///
/// // In the response handler:
/// if self.debounce.should_process(generation) {
///     return self.refresh();
/// }
/// Task::none()
/// ```
#[derive(Debug, Clone)]
pub(crate) struct DebounceState {
    /// Monotonically increasing (modulo overflow) counter.  Each
    /// [`trigger`](Self::trigger) call bumps this; the response handler
    /// compares the incoming generation against it to reject stale tasks.
    generation: u64,
    /// `true` while a debounced refresh is pending (avoids processing
    /// stale responses after a newer trigger has been spawned).
    pending: bool,
}

impl DebounceState {
    pub(crate) const fn new() -> Self {
        Self {
            generation: 0,
            pending: false,
        }
    }

    /// Register a new debounced trigger.
    ///
    /// Increments the generation counter (wrapping on overflow), sets
    /// `pending` to `true`, and returns a [`Task`] that resolves to the
    /// new generation after `ms` milliseconds.
    ///
    /// The caller should map the returned task to their debounced-refresh
    /// message variant (e.g. `.map(MyMessage::DebouncedRefresh)`).
    pub(crate) fn trigger(&mut self, ms: u64) -> Task<u64> {
        self.generation = self.generation.wrapping_add(1);
        self.pending = true;
        let current = self.generation;
        Task::perform(
            super::widgets::debounce_sleep(ms, current),
            std::convert::identity,
        )
    }

    /// Check whether a debounced response should be processed.
    ///
    /// Returns `true` **and** clears the pending flag when `generation`
    /// matches the current generation while a response is pending.
    /// Returns `false` for stale (out-of-date) responses.
    ///
    /// After a `true` return the caller should run their refresh logic.
    #[must_use]
    pub(crate) fn should_process(&mut self, generation: u64) -> bool {
        if generation == self.generation && self.pending {
            self.pending = false;
            true
        } else {
            false
        }
    }
}

// ── String helpers ──────────────────────────────────────────────────

/// Convert a string reference to `None` if empty, otherwise
/// `Some(s.to_string())`.
///
/// Useful when building query structs where empty filters mean "no filter".
pub(crate) fn none_if_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}
