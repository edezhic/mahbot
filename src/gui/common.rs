//! Shared state types used across GUI pages.

use iced::Task;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::broadcast;

use super::editor_widget::{CursorState, EditorAction, EditorBuffer};

pub(crate) const MAX_INPUT_CHARS: usize = 100_000;

/// Pagination state shared by dashboard pages that display paginated data.
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

    pub(crate) const fn total_pages(&self) -> usize {
        if self.total == 0 {
            0
        } else {
            self.total.div_ceil(self.page_size)
        }
    }

    pub(crate) fn prev_page(&mut self) -> bool {
        if self.page > 0 {
            self.page -= 1;
            true
        } else {
            false
        }
    }

    pub(crate) fn next_page(&mut self) -> bool {
        if self.page + 1 < self.total_pages() {
            self.page += 1;
            true
        } else {
            false
        }
    }

    pub(crate) fn reset(&mut self) {
        self.page = 0;
    }

    /// Totals can shrink between refreshes (e.g. log retention purges rows),
    /// leaving a valid page past the end; `true` means the page moved and the caller
    /// should re-query — its entries are from a now-out-of-range offset.
    ///
    /// The bound comes from `total`, not the stale stored `self.total` (clamping
    /// against that could never fire on the refresh that observes the shrink).
    /// A caller whose clamp fires should assign `self.total = total` before
    /// re-querying, so the page indicator matches the clamped page meanwhile.
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

    pub(crate) fn offset(&self) -> usize {
        self.page * self.page_size
    }
}

/// Loading state of one async fetch: before it, in flight, succeeded or failed.
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

    pub(crate) fn loading(&self) -> bool {
        self.loading
    }

    pub(crate) fn has_loaded(&self) -> bool {
        self.has_loaded
    }

    pub(crate) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub(crate) fn start_loading(&mut self) {
        self.loading = true;
        self.error = None;
    }

    pub(crate) fn finish_loading(&mut self) {
        self.loading = false;
        self.has_loaded = true;
    }

    /// Mark failure of an async load.
    ///
    /// Leaves `has_loaded` untouched, so a failed read never counts as a completed
    /// one — each view decides what an unloaded failure renders.
    pub(crate) fn fail(&mut self, error: String) {
        self.error = Some(error);
        self.loading = false;
    }

    /// Dismiss a prior error banner without re-triggering the loading spinner.
    pub(crate) fn clear_error(&mut self) {
        self.error = None;
    }
}

/// Interval a shown polled list re-reads its source at. Deliberately not gated on a
/// read being in flight — the render that follows moves the automations countdown,
/// so suppressing it would freeze the digits; not starting a second read is
/// [`PolledList`]'s job.
pub(crate) const POLLED_LIST_REFRESH_INTERVAL: Duration = Duration::from_secs(1);

/// A list a surface re-reads on a timer. The shared rules, so no consumer restates
/// them: [`begin`](Self::begin) starts a read only when none is in flight; the last
/// successful entries stay on screen during the next read and after a failed one, so
/// a failure never blanks the list; [`loaded`](Self::loaded) flips on the first
/// outcome either way, so "Loading…" belongs to the first read only; a failure is
/// recorded as data, once per read, and nothing is logged here.
///
/// Asks come from a timer (Automations, custom tools) or from events (Sessions
/// re-reads on every runtime tick: a session only grows by a running agent).
pub(crate) struct PolledList<T> {
    entries: Vec<T>,
    in_flight: bool,
    loaded: bool,
    error: Option<String>,
}

impl<T> PolledList<T> {
    pub(crate) const fn new() -> Self {
        Self {
            entries: Vec::new(),
            in_flight: false,
            loaded: false,
            error: None,
        }
    }

    /// `false` when a read already is in flight (start no second read).
    pub(crate) fn begin(&mut self) -> bool {
        if self.in_flight {
            return false;
        }
        self.in_flight = true;
        true
    }

    /// Keeps the previous entries when the read failed.
    pub(crate) fn settle(&mut self, result: Result<Vec<T>, String>) {
        self.in_flight = false;
        self.loaded = true;
        match result {
            Ok(entries) => {
                self.entries = entries;
                self.error = None;
            }
            Err(error) => self.error = Some(error),
        }
    }

    pub(crate) fn entries(&self) -> &[T] {
        &self.entries
    }

    /// For in-place bookkeeping after a mutation this page performed.
    pub(crate) fn entries_mut(&mut self) -> &mut Vec<T> {
        &mut self.entries
    }

    pub(crate) const fn loaded(&self) -> bool {
        self.loaded
    }

    pub(crate) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

/// Shared state for one paginated view of a dashboard page. Each view owns its own
/// `refresh()` (query type, store path and message enum differ) and brackets it with
/// [`begin_refresh`](Self::begin_refresh) and the two response handlers.
pub(crate) struct PaginatedViewState<T> {
    pub(crate) entries: Vec<T>,
    pub(crate) load_state: AsyncLoadState,
    pub(crate) pagination: PaginationState,
    pub(crate) search: String,
    pub(crate) refresh_generation: u64,
}

impl<T> PaginatedViewState<T> {
    pub(crate) fn new(page_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            load_state: AsyncLoadState::new(),
            pagination: PaginationState::new(page_size),
            search: String::new(),
            refresh_generation: 0,
        }
    }

    /// Marks loading, bumps the generation, and returns it for tagging the response.
    pub(crate) fn begin_refresh(&mut self) -> u64 {
        self.load_state.start_loading();
        self.refresh_generation = self.refresh_generation.wrapping_add(1);
        self.refresh_generation
    }

    /// `true` when the fresh `total` shrank past the current page and the caller
    /// should re-query — the total is already adopted, so the page indicator stays
    /// consistent meanwhile. `false` covers both a stale, dropped response and a
    /// fresh, applied one: the caller does nothing further.
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

    pub(crate) fn handle_refresh_error(&mut self, generation: u64, e: String) {
        if generation != self.refresh_generation {
            return;
        }
        self.load_state.fail(e);
    }

    pub(crate) fn clear_entries(&mut self) {
        self.entries.clear();
        self.pagination.reset();
    }
}

/// Debounce state for text inputs that persist or refresh only after the user stops
/// typing: [`trigger`](Self::trigger) on input, [`should_process`](Self::should_process)
/// in the response handler.
#[derive(Debug, Clone)]
pub(crate) struct DebounceState {
    /// Bumped by every trigger; the response handler compares against it to reject
    /// stale tasks.
    generation: u64,
    /// Avoids processing stale responses after a newer trigger was spawned.
    pending: bool,
}

impl DebounceState {
    pub(crate) const fn new() -> Self {
        Self {
            generation: 0,
            pending: false,
        }
    }

    /// Bumps the generation (wrapping on overflow), sets `pending`, and returns a
    /// [`Task`] resolving to the new generation after `ms`. Map it to the caller's own
    /// debounced-refresh message variant.
    pub(crate) fn trigger(&mut self, ms: u64) -> Task<u64> {
        self.generation = self.generation.wrapping_add(1);
        self.pending = true;
        let current = self.generation;
        Task::perform(
            super::widgets::debounce_sleep(ms, current),
            std::convert::identity,
        )
    }

    /// `true` **and** clears the pending flag when `generation` matches while a
    /// response is pending; `false` for stale responses.
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

/// Content accessor for the shared undo stack, implemented for
/// `editor_widget::EditorBuffer`. `cursor` returns the unified [`CursorState`] so a
/// snapshot can restore text and cursor position whatever the buffer.
pub(crate) trait UndoableText {
    fn text(&self) -> String;
    fn cursor(&self) -> CursorState;

    fn set_text(&mut self, text: &str);

    fn move_to(&mut self, line: usize, col: usize);
}

/// Snapshot-based undo/redo stack for a text input editor. Stores `(String,
/// CursorState)` pairs because `EditorBuffer` has no `Clone` that preserves the
/// cursor; restoration replays [`UndoableText::set_text`] + `move_to`.
#[derive(Debug, Clone)]
pub(crate) struct UndoStack {
    /// Previous states, newest last.
    undo: Vec<UndoSnapshot>,
    /// Undone states, cleared on new edit.
    redo: Vec<UndoSnapshot>,
}

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

    /// Saves the current state to the redo stack.
    pub(crate) fn undo(&mut self, content: &impl UndoableText) -> Option<UndoSnapshot> {
        Self::push_and_pop(&mut self.redo, &mut self.undo, content)
    }

    /// Saves the current state to the undo stack.
    pub(crate) fn redo(&mut self, content: &impl UndoableText) -> Option<UndoSnapshot> {
        Self::push_and_pop(&mut self.undo, &mut self.redo, content)
    }

    /// Reset the stack (e.g. after sending a message).
    pub(crate) fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

/// Broadcast-stream producer skeleton used by the chat and logs pages. Subscribes to
/// `source` — skipped when it already has >100 receivers — and forwards events through
/// `emit`, `None` for lagged slots. `emit` decides how to publish (awaited send vs.
/// try_send), so each page keeps its own backpressure semantics.
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

            // A lagged slot is a gap in the stream, emitted as `None`; Closed ends.
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

/// Collapses a burst of broadcast events into one trailing message per `window`: the
/// first event anchors a deadline, everything arriving before it (a `Lagged` slot,
/// i.e. lost events, counts as "changed") is absorbed, and at the deadline exactly one
/// `make_msg` is emitted. A `Lagged` error is never end-of-stream; only channel
/// closure ends the stream. For high-frequency refresh signals where only "something
/// changed" matters.
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

/// The body of [`coalesced_broadcast_producer`], extracted for tests.
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
        // First event anchors a batch. A Lagged slot means events were lost and still counts as a
        // change.
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
                                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
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

/// Edit actions snapshot the pre-edit state for undo and then run; every other
/// action (cursor movement, click positioning) is applied without a snapshot.
/// Undo/Redo restore from the stack instead.
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

/// `None` (an empty stack) is a no-op.
fn restore_undo_snapshot(content: &mut EditorBuffer, snapshot: Option<UndoSnapshot>) {
    if let Some(snapshot) = snapshot {
        content.set_text(&snapshot.text);
        content.move_to(snapshot.cursor.line, snapshot.cursor.column);
    }
}

/// `Some` for [`EditorAction::FocusNext`] / `FocusPrevious`, so a page can intercept
/// focus navigation at the top of a single-line field's handler.
#[must_use]
pub(crate) fn focus_navigation_task<Message>(action: &EditorAction) -> Option<iced::Task<Message>> {
    match action {
        EditorAction::FocusNext => Some(iced::widget::operation::focus_next()),
        EditorAction::FocusPrevious => Some(iced::widget::operation::focus_previous()),
        _ => None,
    }
}

/// Per-field state for a single-line shared-editor field: an [`EditorBuffer`] plus
/// its own undo stack, so undo/redo is per field.
pub(crate) struct SingleLineEditorState {
    pub(crate) buffer: EditorBuffer,
    undo: UndoStack,
}

impl SingleLineEditorState {
    pub(crate) fn new(text: &str) -> Self {
        let buffer = EditorBuffer::with_text(text, None);
        buffer.set_single_line(true);
        Self {
            buffer,
            undo: UndoStack::new(),
        }
    }

    /// E.g. a config reload; resets the undo stack so stale snapshots cannot be restored.
    pub(crate) fn set_text(&mut self, text: &str) {
        self.buffer.set_text(text);
        self.undo.clear();
    }

    pub(crate) fn text(&self) -> String {
        self.buffer.text()
    }

    /// Reset the field to empty (clears undo too).
    pub(crate) fn clear(&mut self) {
        self.buffer.clear();
        self.undo.clear();
    }

    pub(crate) fn apply_action(&mut self, action: EditorAction) {
        // Undo/redo (stack restore) and edit actions (snapshot then perform) share
        // one dispatch, so there is nothing undo-specific here.
        apply_editor_action(&mut self.buffer, &mut self.undo, action);
    }
}

/// Which page's send rules apply, beyond the shared length limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SendRules {
    /// The field is content even with no text (an attachment, a recording the send
    /// stops), and an in-flight send is refused before the length limit is reported.
    Chat { has_content: bool },
    /// The board comment box: text is required, and the length limit is reported before an
    /// in-flight send.
    Board,
}

/// A silent noop for empty text (unless the chat composer holds content of its own),
/// then the page's own pair of checks. Returns the trimmed text when a send may go.
pub(crate) fn send_guard<M: 'static>(
    text: &str,
    sending: bool,
    rules: SendRules,
    over_limit: impl Fn(usize) -> Task<M>,
) -> Result<&str, Task<M>> {
    let trimmed = text.trim();
    let allows_empty = matches!(rules, SendRules::Chat { has_content: true });
    if trimmed.is_empty() && !allows_empty {
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
    if matches!(rules, SendRules::Chat { .. }) {
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
    use SendRules::Board;
    use Step::{Insert, Move, Redo, RedoSnapshot, Restore, Set, Snap, Text, Undo};
    use std::cell::Cell;

    #[derive(Clone, Copy)]
    enum Step {
        Set(&'static str),
        Move(usize, usize),
        Snap,
        Insert(char),
        Undo,
        Redo,
        /// The snapshot's text and cursor `(line, column)`.
        Restore(&'static str, usize, usize),
        /// Call [`UndoStack::redo`]: the snapshot it hands back, `None` when empty.
        RedoSnapshot(Option<&'static str>),
        /// The buffer must hold exactly this text here.
        Text(&'static str),
    }

    const OLD: &str = "original";
    const NEW: &str = "modified";
    const LINES: &str = "line1\nline2\nline3";

    fn restore(buffer: &mut EditorBuffer, text: &str, line: usize, column: usize) {
        let source = EditorBuffer::with_text(text, None);
        source.move_to(line, column);
        let mut stack = UndoStack::new();
        stack.snap_before_edit(&source);
        restore_undo_snapshot(buffer, stack.undo(&source));
    }

    type UndoCase<'a> = (&'a str, &'a [Step], (&'a str, usize, usize));
    /// The stack preserves the cursor, restores text, and drops invalidated redo.
    #[test]
    fn undo_stack_cases() {
        const EDITED: &[Step] = &[Set(OLD), Snap, Set(NEW), Undo, Redo];
        const NEW_EDIT: &[Step] = &[Set("v1"), Snap, Set("v2"), Undo, Set("v3"), Snap, Redo];
        const CURSOR: &[Step] = &[Set(LINES), Move(1, 2), Snap, Undo];
        const ACTION: &[Step] = &[Set("hi"), Move(0, 2), Insert('!'), Text("hi!"), Undo];
        const REDO_RESTORES: &[Step] = &[Set(OLD), Snap, Set(NEW), Undo, RedoSnapshot(Some(NEW))];
        const REDO_CLEARED: &[Step] = &[
            Set("v1"),
            Snap,
            Set("v2"),
            Undo,
            Set("v3"),
            Snap,
            RedoSnapshot(None),
        ];
        let cases: &[UndoCase] = &[
            ("undo", &[Set(OLD), Snap, Set(NEW), Undo], (OLD, 0, 0)),
            ("redo", EDITED, (NEW, 0, 0)),
            ("new edit", NEW_EDIT, ("v3", 0, 0)),
            (
                "redo returns the undone snapshot and keeps the buffer",
                REDO_RESTORES,
                (OLD, 0, 0),
            ),
            (
                "a new edit after an undo clears redo",
                REDO_CLEARED,
                ("v3", 0, 0),
            ),
            ("cursor", CURSOR, (LINES, 1, 2)),
            ("action", ACTION, ("hi", 0, 2)),
            ("restore", &[Set("x"), Restore(LINES, 1, 3)], (LINES, 1, 3)),
        ];

        for &(name, steps, (text, line, column)) in cases {
            let mut buf = EditorBuffer::with_text("", None);
            let mut stack = UndoStack::new();
            for &step in steps {
                match step {
                    Step::Set(text) => buf.set_text(text),
                    Step::Move(line, column) => buf.move_to(line, column),
                    Step::Snap => stack.snap_before_edit(&buf),
                    Step::Insert(c) => {
                        apply_editor_action(&mut buf, &mut stack, EditorAction::Insert(c));
                    }
                    Step::Undo => apply_editor_action(&mut buf, &mut stack, EditorAction::Undo),
                    Step::Redo => apply_editor_action(&mut buf, &mut stack, EditorAction::Redo),
                    Step::Restore(text, line, column) => restore(&mut buf, text, line, column),
                    Step::Text(expected) => {
                        assert_eq!(buf.text(), *expected, "case {name}: text");
                    }
                    Step::RedoSnapshot(expected) => {
                        let snapshot = stack.redo(&buf).map(|undone| undone.text);
                        assert_eq!(snapshot.as_deref(), expected, "case {name}: redo");
                    }
                }
            }
            assert_eq!(buf.text(), text, "case {name}: text");
            let cursor = buf.cursor();
            let at = (cursor.line, cursor.column);
            assert_eq!(at, (line, column), "case {name}: cursor");
        }
    }

    const OVER: usize = MAX_INPUT_CHARS + 1;
    const CHAT: SendRules = SendRules::Chat { has_content: true };
    const CHAT_IDLE: SendRules = SendRules::Chat { has_content: false };

    type GuardCase<'a> = (
        &'a str,
        &'a str,
        bool,
        SendRules,
        Result<Option<&'a str>, bool>,
    );
    #[test]
    fn send_guard_cases() {
        let at_limit = "a".repeat(MAX_INPUT_CHARS);
        let over_limit = "a".repeat(OVER);
        let cases: &[GuardCase] = &[
            ("empty noop", " \t ", false, Board, Err(false)),
            ("trimmed", " hello ", false, Board, Ok(Some("hello"))),
            ("attachment", "  ", false, CHAT, Ok(Some(""))),
            ("at limit", &at_limit, false, Board, Ok(None)),
            ("over limit", &over_limit, false, Board, Err(true)),
            ("chat inflight", &over_limit, true, CHAT_IDLE, Err(false)),
            ("board inflight", &over_limit, true, Board, Err(true)),
        ];

        for (name, input, sending, rules, expect) in cases {
            let fired = Cell::new(false);
            let result: Result<&str, Task<()>> = send_guard(input, *sending, *rules, |_| {
                fired.set(true);
                Task::none()
            });
            match *expect {
                Ok(trimmed) => {
                    let expected = trimmed.unwrap_or(input);
                    assert_eq!(result.ok(), Some(expected), "case {name}: accepted");
                }
                Err(toast) => {
                    assert!(result.is_err(), "case {name}: refused");
                    assert_eq!(fired.get(), toast, "case {name}: over-limit toast");
                }
            }
        }
    }

    #[tokio::test]
    async fn coalesce_loop_batches_burst() {
        use futures_util::StreamExt;

        let (tx, rx) = broadcast::channel::<()>(16);
        let (out_tx, mut out_rx) = iced::futures::channel::mpsc::channel::<u32>(16);
        let window = Duration::from_millis(50);
        let handle = tokio::spawn(coalesce_loop(rx, out_tx, window, || 1u32));

        for _ in 0..5 {
            let _ = tx.send(());
        }
        let first = tokio::time::timeout(Duration::from_secs(1), out_rx.next())
            .await
            .expect("burst should flush within the timeout");
        assert_eq!(first, Some(1));

        // A lone event after a flush is still delivered: a deregister coalesced with
        // transcript traffic must not be lost.
        let _ = tx.send(());
        let second = tokio::time::timeout(Duration::from_secs(1), out_rx.next())
            .await
            .expect("trailing event should flush within the timeout");
        assert_eq!(second, Some(1));

        drop(tx);
        let _ = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("loop should end when the source closes");
    }
}
