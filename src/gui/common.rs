//! Shared state types used across GUI pages.

use iced::Task;
use iced::widget::text_editor;
use std::future::Future;
use std::pin::Pin;

/// Maximum characters allowed in a chat message / comment input.
pub(crate) const MAX_INPUT_CHARS: usize = 100_000;

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

// ── Undo/Redo stack ─────────────────────────────────────────────────

/// Content accessor for the shared undo stack. Implemented for
/// [`text_editor::Content`] (chat composers) and `editor_widget::EditorBuffer`
/// (the code editor); `cursor` returns a full [`text_editor::Cursor`] so
/// selection anchors survive undo/redo where the underlying buffer has them.
pub(crate) trait UndoableText {
    fn text(&self) -> String;
    fn cursor(&self) -> text_editor::Cursor;
}

impl UndoableText for text_editor::Content {
    fn text(&self) -> String {
        text_editor::Content::text(self)
    }

    fn cursor(&self) -> text_editor::Cursor {
        text_editor::Content::cursor(self)
    }
}

/// Snapshot-based undo/redo stack for a text input editor.
///
/// Stores `(String, Cursor)` pairs because [`text_editor::Content`] does not
/// implement `Clone` in a way that preserves cursor position.  Restoration
/// reconstructs via [`text_editor::Content::with_text`] +
/// [`text_editor::Content::move_to`].
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
    pub(crate) cursor: text_editor::Cursor,
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
/// Subscribes to `source` (skipped when it already has >100 receivers), wraps
/// the receiver in a [`BroadcastStream`], and forwards events through `emit` —
/// called with `Some(item)` for received items and `None` for lagged slots.
/// `emit` decides how to publish (awaited send vs. try_send), so each page
/// keeps its own backpressure semantics.
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
            let Some(rx) = source.get().and_then(|tx| {
                if tx.receiver_count() > 100 {
                    None
                } else {
                    Some(tx.subscribe())
                }
            }) else {
                return;
            };

            let mut stream = tokio_stream::wrappers::BroadcastStream::new(rx);
            loop {
                match tokio_stream::StreamExt::next(&mut stream).await {
                    Some(Ok(event)) => emit(&mut output, Some(event)).await,
                    Some(Err(
                        tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(_n),
                    )) => {
                        emit(&mut output, None).await;
                    }
                    None => break,
                }
            }
        },
    )
}

/// Map a keyboard event for a chat composer: modifier changes plus
/// Cmd+Z / Cmd+Shift+Z undo/redo. `mods_changed`, `undo`, `redo` are the
/// consuming page's message constructors (passed as `fn` pointers so the
/// per-page `filter_map` closures stay non-capturing).
pub(crate) fn composer_keyboard_event<M>(
    event: iced::keyboard::Event,
    mods_changed: fn(iced::keyboard::Modifiers) -> M,
    undo: fn() -> M,
    redo: fn() -> M,
) -> Option<M> {
    match event {
        iced::keyboard::Event::ModifiersChanged(modifiers) => Some(mods_changed(modifiers)),
        iced::keyboard::Event::KeyPressed {
            key,
            modifiers,
            physical_key,
            ..
        } => {
            let km = super::detect_keyboard_mods(modifiers);
            // Cmd+Z / Ctrl+Z → undo. Check shift first so Cmd+Shift+Z → redo.
            if km.is_shortcut_platform_mod() && key.to_latin(physical_key) == Some('z') {
                if modifiers.shift() {
                    return Some(redo());
                }
                return Some(undo());
            }
            None
        }
        iced::keyboard::Event::KeyReleased { .. } => None,
    }
}

/// Apply a text-editor action to the composer content: shift+click becomes a
/// drag (extending the selection), and edit actions snapshot the pre-edit
/// state for undo. `perform()` must be called unconditionally for every
/// action (cursor movement, click positioning, edit) — the Iced text_editor
/// widget does not call it itself.
pub(crate) fn apply_editor_action(
    content: &mut text_editor::Content,
    undo_stack: &mut UndoStack,
    action: text_editor::Action,
    shift: bool,
) {
    let action = match action {
        text_editor::Action::Click(pos) if shift => text_editor::Action::Drag(pos),
        other => other,
    };
    if action.is_edit() {
        undo_stack.snap_before_edit(content);
    }
    content.perform(action);
}

/// Restore an undo/redo snapshot into the composer content (`None`, i.e. an
/// empty stack, is a no-op).
pub(crate) fn restore_undo_snapshot(
    content: &mut text_editor::Content,
    snapshot: Option<UndoSnapshot>,
) {
    if let Some(snapshot) = snapshot {
        *content = text_editor::Content::with_text(&snapshot.text);
        content.move_to(snapshot.cursor);
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

    /// Stack with one snapshot of `text` taken before any edit.
    fn stack_with_snapshot(text: &str) -> UndoStack {
        let mut stack = UndoStack::new();
        let content = text_editor::Content::with_text(text);
        stack.snap_before_edit(&content);
        stack
    }

    #[test]
    fn undo_restores_snapshot() {
        let mut stack = stack_with_snapshot("original");
        let modified = text_editor::Content::with_text("modified");
        let snapshot = stack.undo(&modified).unwrap();
        assert_eq!(snapshot.text, "original");
    }

    #[test]
    fn redo_restores_undone_state() {
        let mut stack = stack_with_snapshot("original");
        let modified = text_editor::Content::with_text("modified");
        let _ = stack.undo(&modified);

        let snapshot = stack.redo(&modified).unwrap();
        assert_eq!(snapshot.text, "modified");
    }

    #[test]
    fn new_edit_clears_redo() {
        let mut stack = stack_with_snapshot("v1");
        let v2 = text_editor::Content::with_text("v2");
        let _ = stack.undo(&v2);

        // New edit after undo should clear redo.
        let v3 = text_editor::Content::with_text("v3");
        stack.snap_before_edit(&v3);

        assert!(stack.redo(&v3).is_none());
    }

    #[test]
    fn snapshot_preserves_cursor() {
        let mut content = text_editor::Content::with_text("line1\nline2\nline3");
        content.move_to(text_editor::Cursor {
            position: text_editor::Position { line: 1, column: 2 },
            selection: None,
        });
        let mut stack = UndoStack::new();
        stack.snap_before_edit(&content);

        let modified = text_editor::Content::with_text("changed");
        let snapshot = stack.undo(&modified).unwrap();
        assert_eq!(snapshot.cursor.position.line, 1);
        assert_eq!(snapshot.cursor.position.column, 2);
    }
}
