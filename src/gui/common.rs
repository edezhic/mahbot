//! Shared state types used across GUI pages.

use iced::Task;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::broadcast;

use super::editor_widget::{CursorState, EditorAction, EditorBuffer};

/// Maximum characters allowed in a chat message / comment input.
pub(crate) const MAX_INPUT_CHARS: usize = 100_000;

/// Pagination state shared by dashboard pages that display paginated data.
///
/// Groups `page`, `page_size`, and `total` into a single struct with helper
/// methods for common operations (used by the Logs and Tool Failures pages).
///
/// # Structural benefits (not line savings)
///
/// The struct adds a few lines of definition, but the value is:
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

    /// Clamp `page` to a valid range for a freshly returned `total`.
    ///
    /// Totals can shrink between refreshes (e.g. log retention purges rows),
    /// leaving a previously valid page past the end. Returns `true` when the
    /// page moved to a valid in-range page and the caller should re-query —
    /// the entries it currently holds are from a now-out-of-range offset.
    /// Returns `false` when no re-query is needed (the page was already
    /// valid, or the total is zero so the fresh empty result is correct).
    ///
    /// Note: the bound is computed from `total` (the value just returned by
    /// the query), not the stored `self.total` — the stored value is stale
    /// until the caller assigns it, so clamping against it could never fire
    /// on the refresh that observes the shrink. When a clamp fires, callers
    /// should assign `self.total = total` before re-querying so the page
    /// indicator stays consistent with the clamped page during the re-query
    /// window.
    pub(crate) fn clamp_page(&mut self, total: usize) -> bool {
        let total_pages = total.div_ceil(self.page_size);
        if total_pages == 0 {
            self.page = 0;
            false
        } else if self.page >= total_pages {
            self.page = total_pages - 1;
            true
        } else {
            false
        }
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

/// Shared state for a paginated tab that fetches typed entries.
///
/// Groups the per-tab fields (`entries`, `load_state`, `pagination`, `search`,
/// `refresh_generation`) and the common refresh bookkeeping. Each tab unit
/// still owns its own `refresh()` (query type, store path and message enum
/// differ), so callers use [`begin_refresh`](Self::begin_refresh) to prepare a
/// refresh and [`handle_refreshed`](Self::handle_refreshed)/
/// [`handle_refresh_error`](Self::handle_refresh_error) to process responses.
/// `handle_refresh_error` takes `set_has_loaded_on_error` so the Tool Failures
/// tab marks `has_loaded` on error while the Logs tabs leave it unset.
pub(crate) struct PaginatedTabState<T> {
    pub(crate) entries: Vec<T>,
    pub(crate) load_state: AsyncLoadState,
    pub(crate) pagination: PaginationState,
    pub(crate) search: String,
    pub(crate) refresh_generation: u64,
}

impl<T> PaginatedTabState<T> {
    pub(crate) fn new(page_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            load_state: AsyncLoadState::new(),
            pagination: PaginationState::new(page_size),
            search: String::new(),
            refresh_generation: 0,
        }
    }

    /// Prepare the next refresh: mark loading and bump the generation counter.
    /// Returns the new generation for tagging the async response.
    pub(crate) fn begin_refresh(&mut self) -> u64 {
        self.load_state.start_loading();
        self.refresh_generation = self.refresh_generation.wrapping_add(1);
        self.refresh_generation
    }

    /// Process a successful refresh response.
    ///
    /// Returns `true` when the fresh `total` shrank past the current page and
    /// the caller should re-query (the total is already adopted, so the page
    /// indicator stays consistent during the re-query window). Returns `false`
    /// when the response was stale and dropped, or fresh and applied — in both
    /// cases the caller does nothing further.
    pub(crate) fn handle_refreshed(
        &mut self,
        generation: u64,
        entries: Vec<T>,
        total: usize,
    ) -> bool {
        if generation != self.refresh_generation {
            return false;
        }
        if self.pagination.clamp_page(total) {
            self.pagination.total = total;
            return true;
        }
        self.entries = entries;
        self.pagination.total = total;
        self.load_state.finish_loading();
        false
    }

    /// Process a failed refresh response. `set_has_loaded_on_error` preserves
    /// the Tool-Failures-only behaviour of marking `has_loaded` on error; Logs
    /// tabs pass `false`.
    pub(crate) fn handle_refresh_error(
        &mut self,
        generation: u64,
        e: String,
        set_has_loaded_on_error: bool,
    ) {
        if generation != self.refresh_generation {
            return;
        }
        self.load_state.fail(e);
        if set_has_loaded_on_error {
            self.load_state.set_has_loaded();
        }
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

// ── Undo/Redo stack ─────────────────────────────────────────────────

/// Content accessor for the shared undo stack.
///
/// Implemented for `editor_widget::EditorBuffer` — the single text-buffer
/// engine shared by the code editor and all GUI prose fields.
///
/// `cursor` returns the unified [`CursorState`] so undo/redo snapshots can
/// restore both text and cursor position regardless of the underlying buffer.
pub(crate) trait UndoableText {
    fn text(&self) -> String;
    fn cursor(&self) -> CursorState;

    /// Replace all content with `text` and reset the cursor to (0, 0).
    fn set_text(&mut self, text: &str);

    /// Move the cursor to `(line, column)`, clearing any selection.
    fn move_to(&mut self, line: usize, col: usize);
}

/// Snapshot-based undo/redo stack for a text input editor.
///
/// Stores `(String, CursorState)` pairs because `editor_widget::EditorBuffer`
/// does not implement `Clone` in a way that preserves cursor position.
/// Restoration reconstructs via [`UndoableText::set_text`] +
/// [`UndoableText::move_to`].
#[derive(Debug, Clone)]
pub(crate) struct UndoStack {
    /// Previous states, newest last.
    undo: Vec<UndoSnapshot>,
    /// Undone states, cleared on new edit.
    redo: Vec<UndoSnapshot>,
}

/// A single undo snapshot for a text input editor.
#[derive(Debug, Clone)]
pub(crate) struct UndoSnapshot {
    pub(crate) text: String,
    pub(crate) cursor: CursorState,
}

impl UndoStack {
    const MAX_UNDO_DEPTH: usize = 100;
    const LARGE_FILE_UNDO_THRESHOLD: usize = 100_000;

    pub(crate) const fn new() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// Take a snapshot before an edit is performed. Content larger than
    /// [`Self::LARGE_FILE_UNDO_THRESHOLD`] halves the depth cap to bound memory.
    pub(crate) fn snap_before_edit(&mut self, content: &impl UndoableText) {
        let text = content.text();
        let max_depth = if text.len() > Self::LARGE_FILE_UNDO_THRESHOLD {
            Self::MAX_UNDO_DEPTH / 2
        } else {
            Self::MAX_UNDO_DEPTH
        };
        self.redo.clear();
        self.undo.push(UndoSnapshot {
            text,
            cursor: content.cursor(),
        });
        if self.undo.len() > max_depth {
            self.undo.remove(0);
        }
    }

    fn push_and_pop(
        dst: &mut Vec<UndoSnapshot>,
        src: &mut Vec<UndoSnapshot>,
        content: &impl UndoableText,
    ) -> Option<UndoSnapshot> {
        dst.push(UndoSnapshot {
            text: content.text(),
            cursor: content.cursor(),
        });
        src.pop()
    }

    /// Pop the most recent snapshot, saving current state to the redo stack.
    pub(crate) fn undo(&mut self, content: &impl UndoableText) -> Option<UndoSnapshot> {
        Self::push_and_pop(&mut self.redo, &mut self.undo, content)
    }

    /// Pop the most recent undone snapshot, saving current state to the undo stack.
    pub(crate) fn redo(&mut self, content: &impl UndoableText) -> Option<UndoSnapshot> {
        Self::push_and_pop(&mut self.undo, &mut self.redo, content)
    }

    /// Reset the stack (e.g. after sending a message).
    pub(crate) fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

/// Shared broadcast-stream producer skeleton used by the chat and logs pages.
///
/// Subscribes to `source` (skipped when it already has >100 receivers) and
/// forwards events through `emit` — called with `Some(item)` for received
/// items and `None` for lagged slots — via a direct `broadcast::Receiver`
/// recv loop. `emit` decides how to publish (awaited send vs. try_send), so
/// each page keeps its own backpressure semantics.
pub(crate) fn broadcast_stream_producer<Msg, T, E>(
    capacity: usize,
    source: &'static std::sync::OnceLock<tokio::sync::broadcast::Sender<T>>,
    mut emit: E,
) -> impl futures_util::Stream<Item = Msg>
where
    Msg: Send + 'static,
    T: Clone + Send + 'static,
    E: FnMut(
            &mut iced::futures::channel::mpsc::Sender<Msg>,
            Option<T>,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>
        + Send
        + 'static,
{
    iced::stream::channel(
        capacity,
        move |mut output: iced::futures::channel::mpsc::Sender<Msg>| async move {
            let Some(mut rx) = source.get().and_then(|tx| {
                if tx.receiver_count() > 100 {
                    None
                } else {
                    Some(tx.subscribe())
                }
            }) else {
                return;
            };

            // Direct broadcast receiver loop matching the removed
            // tokio-stream BroadcastStream's semantics: Ok(event) →
            // emit(Some(event)), Lagged → emit(None) (gap in the stream),
            // Closed → end.
            loop {
                match rx.recv().await {
                    Ok(event) => emit(&mut output, Some(event)).await,
                    Err(broadcast::error::RecvError::Lagged(_n)) => {
                        emit(&mut output, None).await;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        },
    )
}

/// Coalescing broadcast-stream producer: collapses bursts of broadcast events
/// into a single trailing message per `window`. The first event anchors a
/// deadline; every event arriving before it (including a `Lagged` slot, which
/// means events were lost and counts as "changed") is absorbed; at the
/// deadline exactly one message is emitted via `make_msg`. A `Lagged` error is
/// treated as an event, never as end-of-stream. Ends only when the channel
/// closes. Used for high-frequency GUI refresh signals (agent registry /
/// transcript content / voice status) where only "something changed" matters.
pub(crate) fn coalesced_broadcast_producer<Msg, T>(
    capacity: usize,
    source: &'static std::sync::OnceLock<tokio::sync::broadcast::Sender<T>>,
    window: Duration,
    make_msg: impl Fn() -> Msg + Send + 'static,
) -> impl futures_util::Stream<Item = Msg>
where
    Msg: Send + 'static,
    T: Clone + Send + 'static,
{
    iced::stream::channel(
        capacity,
        move |output: iced::futures::channel::mpsc::Sender<Msg>| async move {
            let Some(rx) = source.get().and_then(|tx| {
                if tx.receiver_count() > 100 {
                    None
                } else {
                    Some(tx.subscribe())
                }
            }) else {
                return;
            };
            coalesce_loop(rx, output, window, make_msg).await;
        },
    )
}

/// The coalescing body of [`coalesced_broadcast_producer`], extracted so it can
/// be driven directly in tests without an iced stream channel.
async fn coalesce_loop<Msg, T, F>(
    mut rx: broadcast::Receiver<T>,
    mut output: iced::futures::channel::mpsc::Sender<Msg>,
    window: Duration,
    make_msg: F,
) where
    Msg: Send,
    T: Clone + Send,
    F: Fn() -> Msg + Send,
{
    loop {
        // First event anchors a batch. A Lagged slot means events were lost and
        // still counts as a change, so it anchors a batch too.
        match rx.recv().await {
            Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {
                let deadline = tokio::time::Instant::now() + window;
                loop {
                    tokio::select! {
                        () = tokio::time::sleep_until(deadline) => {
                            let _ = futures_util::SinkExt::send(&mut output, make_msg()).await;
                            break;
                        }
                        recv = rx.recv() => {
                            match recv {
                                // Absorb into the current batch; keep batching
                                // until the deadline fires.
                                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                                // Source gone — no more changes will arrive.
                                Err(broadcast::error::RecvError::Closed) => return,
                            }
                        }
                    }
                }
            }
            Err(broadcast::error::RecvError::Closed) => return,
        }
    }
}

/// Apply a text-editor action to a buffer: edit actions snapshot the
/// pre-edit state for undo, then the action is performed. The editor widget
/// publishes every action (cursor movement, click positioning, edit), so
/// `perform_action` is called unconditionally; only edit actions are
/// snapshotted. Undo/Redo restore from the undo/redo stack instead.
pub(crate) fn apply_editor_action(
    content: &mut EditorBuffer,
    undo_stack: &mut UndoStack,
    action: EditorAction,
) {
    match action {
        EditorAction::Undo => {
            if let Some(s) = undo_stack.undo(content) {
                restore_undo_snapshot(content, Some(s));
            }
        }
        EditorAction::Redo => {
            if let Some(s) = undo_stack.redo(content) {
                restore_undo_snapshot(content, Some(s));
            }
        }
        other => {
            if other.is_edit_action() {
                undo_stack.snap_before_edit(content);
            }
            content.perform_action(other);
        }
    }
}

/// Restore an undo/redo snapshot into a buffer (`None`, i.e. an empty stack,
/// is a no-op).
pub(crate) fn restore_undo_snapshot(content: &mut EditorBuffer, snapshot: Option<UndoSnapshot>) {
    if let Some(snapshot) = snapshot {
        content.set_text(&snapshot.text);
        content.move_to(snapshot.cursor.line, snapshot.cursor.column);
    }
}

/// Map a single-line Tab / Shift+Tab action to Iced focus traversal.
///
/// Returns `Some(task)` for [`EditorAction::FocusNext`] / [`FocusPrevious`]
/// and `None` for every other action, so a page can intercept focus navigation
/// at the top of any single-line field's handler without disturbing the
/// buffer/undo logic below.
#[must_use]
pub(crate) fn focus_navigation_task<Message>(action: &EditorAction) -> Option<iced::Task<Message>> {
    match action {
        EditorAction::FocusNext => Some(iced::widget::operation::focus_next()),
        EditorAction::FocusPrevious => Some(iced::widget::operation::focus_previous()),
        _ => None,
    }
}

/// Per-field state for a single-line shared-editor field: an [`EditorBuffer`]
/// plus its own undo stack. Each input owns its own undo/redo, matching the
/// per-field undo requirement.
pub(crate) struct SingleLineEditorState {
    pub(crate) buffer: EditorBuffer,
    undo: UndoStack,
}

impl SingleLineEditorState {
    /// Create a single-line field pre-populated with `text`.
    pub(crate) fn new(text: &str) -> Self {
        let buffer = EditorBuffer::with_text(text, None);
        buffer.set_single_line(true);
        Self {
            buffer,
            undo: UndoStack::new(),
        }
    }

    /// Re-populate the field from an external source (e.g. config reload),
    /// resetting the undo stack so stale snapshots cannot be restored.
    pub(crate) fn set_text(&mut self, text: &str) {
        self.buffer.set_text(text);
        self.undo.clear();
    }

    /// Current field text.
    pub(crate) fn text(&self) -> String {
        self.buffer.text()
    }

    /// Reset the field to empty (clears undo too).
    pub(crate) fn clear(&mut self) {
        self.buffer.clear();
        self.undo.clear();
    }

    /// Apply an [`EditorAction`] emitted by the shared widget.
    pub(crate) fn apply_action(&mut self, action: EditorAction) {
        // `apply_editor_action` uniformly handles undo/redo (restoring from the
        // stack) and edit actions (snapshotting then performing), so there is
        // no separate undo/redo dispatch here.
        apply_editor_action(&mut self.buffer, &mut self.undo, action);
    }
}

/// Guard a chat send: trim-empty noop, then the in-flight noop and the
/// over-limit toast in per-page order (`in_flight_first` — home checks
/// in-flight before the limit, board after). Returns the trimmed text when
/// the send may proceed.
pub(crate) fn send_guard<M: 'static>(
    text: &str,
    sending: bool,
    in_flight_first: bool,
    over_limit: impl Fn(usize) -> Task<M>,
) -> Result<&str, Task<M>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(Task::none());
    }
    let over_limit_task = || {
        let count = trimmed.chars().count();
        if count > MAX_INPUT_CHARS {
            Some(over_limit(count))
        } else {
            None
        }
    };
    if in_flight_first {
        if sending {
            return Err(Task::none());
        }
        if let Some(task) = over_limit_task() {
            return Err(task);
        }
    } else {
        if let Some(task) = over_limit_task() {
            return Err(task);
        }
        if sending {
            return Err(Task::none());
        }
    }
    Ok(trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Stack with one snapshot of `text` taken before any edit.
    fn stack_with_snapshot(text: &str) -> UndoStack {
        let mut stack = UndoStack::new();
        let content = EditorBuffer::with_text(text, None);
        stack.snap_before_edit(&content);
        stack
    }

    #[test]
    fn undo_restores_snapshot() {
        let mut stack = stack_with_snapshot("original");
        let modified = EditorBuffer::with_text("modified", None);
        let snapshot = stack.undo(&modified).unwrap();
        assert_eq!(snapshot.text, "original");
    }

    #[test]
    fn redo_restores_undone_state() {
        let mut stack = stack_with_snapshot("original");
        let modified = EditorBuffer::with_text("modified", None);
        let _ = stack.undo(&modified);

        let snapshot = stack.redo(&modified).unwrap();
        assert_eq!(snapshot.text, "modified");
    }

    #[test]
    fn new_edit_clears_redo() {
        let mut stack = stack_with_snapshot("v1");
        let v2 = EditorBuffer::with_text("v2", None);
        let _ = stack.undo(&v2);

        // New edit after undo should clear redo.
        let v3 = EditorBuffer::with_text("v3", None);
        stack.snap_before_edit(&v3);

        assert!(stack.redo(&v3).is_none());
    }

    #[test]
    fn snapshot_preserves_cursor() {
        let content = EditorBuffer::with_text("line1\nline2\nline3", None);
        content.move_to(1, 2);
        let mut stack = UndoStack::new();
        stack.snap_before_edit(&content);

        let modified = EditorBuffer::with_text("changed", None);
        let snapshot = stack.undo(&modified).unwrap();
        assert_eq!(snapshot.cursor.line, 1);
        assert_eq!(snapshot.cursor.column, 2);
    }

    #[test]
    fn apply_editor_action_snapshots_edits() {
        let mut buffer = EditorBuffer::with_text("hello", None);
        buffer.move_to(0, 5);
        let mut stack = UndoStack::new();
        apply_editor_action(&mut buffer, &mut stack, EditorAction::Insert('!'));
        assert_eq!(buffer.text(), "hello!");
        // The edit was snapshotted before it happened.
        assert_eq!(stack.undo(&buffer).unwrap().text, "hello");
    }

    #[test]
    fn restore_undo_snapshot_restores_text_and_cursor() {
        let mut buffer = EditorBuffer::with_text("changed", None);
        let snapshot = Some(UndoSnapshot {
            text: "line1\nline2\nline3".to_string(),
            cursor: CursorState {
                line: 1,
                column: 3,
                selection: None,
            },
        });
        restore_undo_snapshot(&mut buffer, snapshot);
        assert_eq!(buffer.text(), "line1\nline2\nline3");
        let cursor = buffer.cursor();
        assert_eq!(cursor.line, 1);
        assert_eq!(cursor.column, 3);
    }

    /// Run `send_guard`, recording whether `over_limit` fired (task never run).
    fn run_guard(
        text: &str,
        sending: bool,
        in_flight_first: bool,
    ) -> (bool, Result<&str, Task<()>>) {
        let fired = Cell::new(false);
        let result = send_guard(text, sending, in_flight_first, |_| {
            fired.set(true);
            Task::none()
        });
        (fired.get(), result)
    }

    #[test]
    fn send_guard_rejects_empty_and_trims() {
        let (fired, result) = run_guard(" \t ", false, true);
        assert!(result.is_err() && !fired, "empty input is a silent noop");
        let (_, result) = run_guard("  hello  ", false, true);
        assert_eq!(result.unwrap(), "hello");
    }

    #[test]
    fn send_guard_limit_boundary() {
        let at_limit = "a".repeat(MAX_INPUT_CHARS);
        let (fired, result) = run_guard(&at_limit, false, true);
        assert!(result.is_ok() && !fired, "at-limit text is accepted");
        let over_limit = "a".repeat(MAX_INPUT_CHARS + 1);
        let (fired, result) = run_guard(&over_limit, false, true);
        assert!(result.is_err() && fired, "over-limit text is rejected");
    }

    #[test]
    fn send_guard_combined_in_flight_and_over_limit() {
        let text = "a".repeat(MAX_INPUT_CHARS + 1);
        let (fired, result) = run_guard(&text, true, true);
        assert!(result.is_err() && !fired, "in-flight first: silent noop");
        let (fired, result) = run_guard(&text, true, false);
        assert!(result.is_err() && fired, "toast fires over-limit first");
    }

    #[tokio::test]
    async fn coalesce_loop_batches_burst() {
        use futures_util::StreamExt;

        let (tx, rx) = broadcast::channel::<()>(16);
        let (out_tx, mut out_rx) = iced::futures::channel::mpsc::channel::<u32>(16);
        let window = Duration::from_millis(50);
        let handle = tokio::spawn(coalesce_loop(rx, out_tx, window, || 1u32));

        // A burst of 5 events within one window collapses to a single message.
        for _ in 0..5 {
            let _ = tx.send(());
        }
        let first = tokio::time::timeout(Duration::from_secs(1), out_rx.next())
            .await
            .expect("burst should flush within the timeout");
        assert_eq!(first, Some(1));

        // A single event after a flush is still delivered (a deregister
        // coalesced with transcript traffic must not be lost).
        let _ = tx.send(());
        let second = tokio::time::timeout(Duration::from_secs(1), out_rx.next())
            .await
            .expect("trailing event should flush within the timeout");
        assert_eq!(second, Some(1));

        // Drop the only source to close the channel; the loop then ends.
        drop(tx);
        let _ = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop should end when the source closes");
    }
}
