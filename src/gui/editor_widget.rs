//! A [`cosmic_text::Buffer`]-backed text buffer with cursor and selection
//! management. Serves as the core text editing buffer for the editor.rs codebase.

use std::cell::{Cell, RefCell};

use cosmic_text::Scroll;
use iced::advanced::graphics::text::cosmic_text;
use iced::advanced::input_method;
use iced::mouse::ScrollDelta;

use super::highlight::{self, FileHighlights, HighlightLanguage};
use super::text_rendering::{
    GUTTER_FONT_SIZE, MAX_HIGHLIGHT_SIZE, compute_total_height, draw_highlight_background,
    font_metrics, gutter_clip_rect, iced_color_to_cosmic, push_or_merge, reshape_and_shape,
    text_area_rect, with_font_system,
};
use crate::util::UnwrapPoison;

// ── Constants ───────────────────────────────────────────────────────

/// Number of lines to scroll when paging up or down. Both directions use
/// the same value so that page-up and page-down undo each other.
pub(crate) const PAGE_SCROLL_LINES: usize = 40;

// ── CursorState ──────────────────────────────────────────────────────

/// Represents a cursor position (line and character-based column) together
/// with an optional selection anchor.
#[derive(Debug, Clone)]
pub struct CursorState {
    /// Zero-based line index.
    pub line: usize,
    /// Character-based column offset (not byte index).
    pub column: usize,
    /// The other end of a selection range, if any.
    pub selection: Option<Box<CursorState>>,
}

impl CursorState {
    /// Create a new cursor state at the given position with no selection.
    #[must_use]
    pub const fn new(line: usize, column: usize) -> Self {
        Self {
            line,
            column,
            selection: None,
        }
    }
}

// ── CursorMove ───────────────────────────────────────────────────────

/// Direction for [`EditorAction::Move`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CursorMove {
    /// Move cursor one character left.
    Left,
    /// Move cursor one character right.
    Right,
    /// Move cursor one line up.
    Up,
    /// Move cursor one line down.
    Down,
    /// Move cursor to start of current line.
    Home,
    /// Move cursor to end of current line.
    End,
    /// Move cursor one word left.
    WordLeft,
    /// Move cursor one word right.
    WordRight,
    /// Move cursor to start of document.
    DocStart,
    /// Move cursor to end of document.
    DocEnd,
    /// Move cursor one page up.
    PageUp,
    /// Move cursor one page down.
    PageDown,
}

// ── EditorAction ────────────────────────────────────────────────────

/// An action to perform on an [`EditorBuffer`].
#[derive(Debug, Clone)]
pub enum EditorAction {
    /// Insert a single character at the cursor.
    Insert(char),
    /// Insert a newline at the cursor.
    Enter,
    /// Delete the character behind the cursor.
    Backspace,
    /// Delete the character in front of the cursor.
    Delete,
    /// Insert a string at the cursor (paste).
    Paste(String),
    /// Move the cursor to an absolute (line, column).
    MoveTo {
        /// Target line.
        line: usize,
        /// Target character-based column.
        col: usize,
    },
    /// Extend the selection to an absolute (line, column).
    SelectTo {
        /// Target line.
        line: usize,
        /// Target character-based column.
        col: usize,
    },
    /// Move cursor (or extend selection) in a given direction.
    /// See [`CursorMove`] for the available directions.
    Move {
        /// Direction to move the cursor.
        direction: CursorMove,
        /// `true` extends the selection instead of moving the cursor.
        select: bool,
    },
    /// Delete from cursor to start of previous word.
    DeleteWordBack,
    /// Delete from cursor to start of next word.
    DeleteWordForward,
    /// Select all text.
    SelectAll,
    /// Select the word at the given position (double-click).
    SelectWordAt {
        /// Line of the clicked word.
        line: usize,
        /// Character-based column of the click.
        col: usize,
    },
    /// Insert a literal tab character at the cursor.
    Indent,
    /// Remove leading whitespace from the current line.
    Unindent,
    /// Toggle line comment on the current line or selection.
    ToggleLineComment,
    /// Jump cursor to the matching bracket.
    JumpToMatchingBracket,
    /// Delete the current line (or selected lines).
    DeleteLine,
    /// Duplicate the current line (or selected lines).
    DuplicateLine,
    /// Move the current line (or selected lines) up by one.
    MoveLineUp,
    /// Move the current line (or selected lines) down by one.
    MoveLineDown,
}

impl EditorAction {
    /// Returns `true` if this action modifies the editor content.
    #[must_use]
    pub const fn is_edit_action(&self) -> bool {
        matches!(
            self,
            Self::Insert(_)
                | Self::Enter
                | Self::Backspace
                | Self::Delete
                | Self::Paste(_)
                | Self::Indent
                | Self::Unindent
                | Self::DeleteWordBack
                | Self::DeleteWordForward
                | Self::ToggleLineComment
                | Self::DeleteLine
                | Self::DuplicateLine
                | Self::MoveLineUp
                | Self::MoveLineDown
        )
    }

    /// Returns `true` if this action moves the cursor (including extending selection).
    #[must_use]
    pub const fn is_cursor_movement(&self) -> bool {
        matches!(
            self,
            Self::Move { .. }
                | Self::SelectWordAt { .. }
                | Self::JumpToMatchingBracket
                | Self::MoveLineUp
                | Self::MoveLineDown
        )
    }
}

// ── EditorBuffer ────────────────────────────────────────────────────

/// A text buffer backed by [`cosmic_text::Buffer`] with manual cursor and
/// selection tracking. All mutating methods take `&self` using interior
/// mutability (`Cell` / `RefCell`) so the buffer can be used throughout
/// Iced's widget tree without requiring `&mut` access.
pub struct EditorBuffer {
    buffer: RefCell<cosmic_text::Buffer>,
    cursor_line: Cell<usize>,
    cursor_col: Cell<usize>,
    sel_line: Cell<usize>,
    sel_col: Cell<usize>,
    has_selection: Cell<bool>,
    /// Language for syntax highlighting. When `Some` and text is within
    /// [`MAX_HIGHLIGHT_SIZE`], tree-sitter highlighting is applied when
    /// setting text.
    language: Option<HighlightLanguage>,
    /// File extension for fallback comment prefix lookup (e.g., `"yaml"`,
    /// `"yml"`, `"dockerfile"`) when `language` is `None` or has no line
    /// comment syntax.
    file_extension: RefCell<Option<String>>,
}

impl EditorBuffer {
    /// Create an empty buffer with no language (no syntax highlighting).
    #[must_use]
    pub fn new() -> Self {
        Self::with_text("", None)
    }

    /// Create a buffer pre-populated with the given text.
    ///
    /// When `language` is `Some` and the text is within
    /// `MAX_HIGHLIGHT_SIZE`, syntax highlighting is applied via tree-sitter.
    /// Otherwise text is rendered with default attributes.
    #[must_use]
    pub fn with_text(text: &str, language: Option<HighlightLanguage>) -> Self {
        let buffer = with_font_system(|font_sys| {
            let mut buffer = cosmic_text::Buffer::new(font_sys, font_metrics());
            Self::set_buffer_text_highlighted(&mut buffer, font_sys, text, language);
            buffer
        });
        Self {
            buffer: RefCell::new(buffer),
            cursor_line: Cell::new(0),
            cursor_col: Cell::new(0),
            sel_line: Cell::new(0),
            sel_col: Cell::new(0),
            has_selection: Cell::new(false),
            language,
            file_extension: RefCell::new(None),
        }
    }

    /// Create a buffer pre-populated with the given text and infer the
    /// language and file extension from the given file path.
    pub fn from_file(text: &str, path: impl AsRef<std::path::Path>) -> Self {
        let path_ref = path.as_ref();
        let language = path_ref.to_str().and_then(HighlightLanguage::from_path);
        let content = Self::with_text(text, language);
        content.set_file_extension(path_ref.extension().and_then(|e| e.to_str()));
        content
    }

    /// Set the file extension for fallback comment prefix lookup.
    /// The extension should be the file extension without the dot
    /// (e.g., `"yaml"`, `"py"`, `"rs"`).
    pub fn set_file_extension(&self, ext: Option<&str>) {
        *self.file_extension.borrow_mut() = ext.map(String::from);
    }

    /// Return the file extension, if any.
    #[must_use]
    pub fn file_extension(&self) -> Option<String> {
        self.file_extension.borrow().clone()
    }

    /// Return the full text content.
    pub fn text(&self) -> String {
        buffer_text(&self.buffer.borrow())
    }

    /// Return the text of a specific line (0-based), or `None` if the line
    /// index is out of range.
    pub fn line(&self, index: usize) -> Option<String> {
        self.buffer
            .borrow()
            .lines
            .get(index)
            .map(|l| l.text().to_string())
    }

    /// Return the number of lines in the buffer.
    pub fn line_count(&self) -> usize {
        self.buffer.borrow().lines.len()
    }

    /// Return the associated highlight language, if any.
    #[must_use]
    pub const fn language(&self) -> Option<HighlightLanguage> {
        self.language
    }

    // ── Cursor ────────────────────────────────────────────────────

    /// Return the current cursor state, including selection anchor if any.
    pub fn cursor(&self) -> CursorState {
        let has_real_selection = self.has_selection.get()
            && (self.sel_line.get() != self.cursor_line.get()
                || self.sel_col.get() != self.cursor_col.get());
        let selection = if has_real_selection {
            Some(Box::new(CursorState {
                line: self.sel_line.get(),
                column: self.sel_col.get(),
                selection: None,
            }))
        } else {
            None
        };
        CursorState {
            line: self.cursor_line.get(),
            column: self.cursor_col.get(),
            selection,
        }
    }

    /// Move the cursor to a given (line, column) and clear any selection.
    pub fn move_to(&self, line: usize, col: usize) {
        self.set_cursor_pos(line, col);
        self.has_selection.set(false);
    }

    /// Set cursor position without affecting selection state.
    /// Clamps `line` and `col` to valid ranges.
    fn set_cursor_pos(&self, line: usize, col: usize) {
        let max_line = self.line_count().saturating_sub(1);
        let line = line.min(max_line);
        let col = self.clamp_col_to_line(line, col);
        self.cursor_line.set(line);
        self.cursor_col.set(col);
    }

    // ── Selection ─────────────────────────────────────────────────

    /// Return the currently selected text, if any.
    pub fn selection(&self) -> Option<String> {
        if !self.has_selection.get() {
            return None;
        }
        let (start_line, start_col, end_line, end_col) = self.selection_range();
        let text_buf = self.text();
        let start_offset = line_col_to_byte_offset(&text_buf, start_line, start_col);
        let end_offset = line_col_to_byte_offset(&text_buf, end_line, end_col);
        match start_offset.cmp(&end_offset) {
            std::cmp::Ordering::Less => Some(text_buf[start_offset..end_offset].to_string()),
            std::cmp::Ordering::Greater => Some(text_buf[end_offset..start_offset].to_string()),
            std::cmp::Ordering::Equal => None,
        }
    }

    /// Select all text in the buffer.
    pub fn select_all(&self) {
        let line_count = self.line_count();
        if line_count == 0 {
            return;
        }
        // Set cursor to end of last line
        let last_line = line_count - 1;
        let last_line_len = self
            .buffer
            .borrow()
            .lines
            .last()
            .map_or(0, |l| l.text().chars().count());
        self.cursor_line.set(last_line);
        self.cursor_col.set(last_line_len);
        // Set selection anchor to start
        self.sel_line.set(0);
        self.sel_col.set(0);
        self.has_selection.set(true);
    }

    // ── Content manipulation ──────────────────────────────────────

    /// Apply an [`EditorAction`] to the buffer, modifying text and cursor
    /// as appropriate.
    pub fn perform_action(&self, action: EditorAction) {
        match action {
            EditorAction::Insert(c) => self.do_insert(c),
            EditorAction::Enter => self.do_enter(),
            EditorAction::Backspace => self.do_backspace(),
            EditorAction::Delete => self.do_delete(),
            EditorAction::Paste(s) => self.do_paste(&s),
            EditorAction::MoveTo { line, col } => self.move_to(line, col),
            EditorAction::SelectTo { line, col } => {
                if !self.has_selection.get() {
                    self.sel_line.set(self.cursor_line.get());
                    self.sel_col.set(self.cursor_col.get());
                }
                let max_line = self.line_count().saturating_sub(1);
                let line = line.min(max_line);
                let col = self.clamp_col_to_line(line, col);
                if line == self.cursor_line.get() && col == self.cursor_col.get() {
                    // Duplicate SelectTo at the current endpoint (e.g. repeated
                    // CursorMoved during drag) must not clear an existing
                    // non-empty selection.
                    if self.has_selection.get()
                        && (self.sel_line.get() != self.cursor_line.get()
                            || self.sel_col.get() != self.cursor_col.get())
                    {
                        return;
                    }
                    self.has_selection.set(false);
                    return;
                }
                self.cursor_line.set(line);
                self.cursor_col.set(col);
                self.has_selection.set(true);
                self.normalize_selection();
            }
            EditorAction::SelectAll => self.select_all(),
            EditorAction::SelectWordAt { line, col } => {
                let text_buf = self.text();
                let byte_offset = line_col_to_byte_offset(&text_buf, line, col);
                let (word_start, word_end) = word_bounds_at(&text_buf, byte_offset);
                if word_start == word_end {
                    // Zero-width (whitespace, newline, end-of-text) → fall through to MoveTo.
                    self.move_to(line, col);
                } else {
                    let (anchor_line, anchor_col) = byte_offset_to_line_col(&text_buf, word_start);
                    let (cursor_line, cursor_col) = byte_offset_to_line_col(&text_buf, word_end);
                    self.sel_line.set(anchor_line);
                    self.sel_col.set(anchor_col);
                    self.cursor_line.set(cursor_line);
                    self.cursor_col.set(cursor_col);
                    self.has_selection.set(true);
                    self.normalize_selection();
                }
            }
            EditorAction::Indent => self.do_indent(),
            EditorAction::Unindent => self.do_unindent(),
            EditorAction::Move { direction, select } => match direction {
                CursorMove::Left => self.do_move_left(select),
                CursorMove::Right => self.do_move_right(select),
                CursorMove::Up => self.do_move_up(select),
                CursorMove::Down => self.do_move_down(select),
                CursorMove::Home => self.do_move_home(select),
                CursorMove::End => self.do_move_end(select),
                CursorMove::WordLeft => self.do_move_word_left(select),
                CursorMove::WordRight => self.do_move_word_right(select),
                CursorMove::DocStart => self.do_move_doc_start(select),
                CursorMove::DocEnd => self.do_move_doc_end(select),
                CursorMove::PageUp => self.do_move_page_up(select),
                CursorMove::PageDown => self.do_move_page_down(select),
            },
            EditorAction::DeleteWordBack => self.do_delete_word_back(),
            EditorAction::DeleteWordForward => self.do_delete_word_forward(),
            EditorAction::ToggleLineComment => self.do_toggle_line_comment(),
            EditorAction::JumpToMatchingBracket => self.do_jump_to_matching_bracket(),
            EditorAction::DeleteLine => self.do_delete_line(),
            EditorAction::DuplicateLine => self.do_duplicate_line(),
            EditorAction::MoveLineUp => self.do_move_line_up(),
            EditorAction::MoveLineDown => self.do_move_line_down(),
        }
    }

    /// Replace the entire buffer content with new text. Resets cursor and
    /// selection to the start. Re-applies syntax highlighting if a language
    /// is configured.
    pub fn set_text(&self, new_text: &str) {
        let language = self.language;
        with_font_system(|font_sys| {
            let mut buffer = self.buffer.borrow_mut();
            Self::set_buffer_text_highlighted(&mut buffer, font_sys, new_text, language);
        });
        self.cursor_line.set(0);
        self.cursor_col.set(0);
        self.has_selection.set(false);
    }

    // ── Expose inner buffer for widget rendering ──────────────────

    /// Borrow the underlying [`cosmic_text::Buffer`] for drawing.
    pub fn borrow_buffer(&self) -> std::cell::Ref<'_, cosmic_text::Buffer> {
        self.buffer.borrow()
    }

    /// Borrow the underlying [`cosmic_text::Buffer`] mutably for shaping.
    pub fn borrow_buffer_mut(&self) -> std::cell::RefMut<'_, cosmic_text::Buffer> {
        self.buffer.borrow_mut()
    }

    // ── Private helpers ───────────────────────────────────────────

    /// Clamp a column value to the number of characters on the given line.
    fn clamp_col_to_line(&self, line: usize, col: usize) -> usize {
        self.buffer
            .borrow()
            .lines
            .get(line)
            .map_or(0, |l| l.text().chars().count().min(col))
    }

    /// Set the inner [`cosmic_text::Buffer`] content with optional
    /// syntax highlighting.
    ///
    /// When `language` is `Some` and `text` is within
    /// [`MAX_HIGHLIGHT_SIZE`], parses the text with tree-sitter and
    /// uses [`cosmic_text::Buffer::set_rich_text`] to apply per-span
    /// colors. Falls back to plain `set_text` otherwise.
    fn set_buffer_text_highlighted(
        buffer: &mut cosmic_text::Buffer,
        font_sys: &mut cosmic_text::FontSystem,
        text: &str,
        language: Option<HighlightLanguage>,
    ) {
        if let Some(lang) = language {
            if text.len() <= MAX_HIGHLIGHT_SIZE {
                let highlights = Self::compute_highlights(text, lang);
                let base_attrs =
                    cosmic_text::Attrs::new().family(cosmic_text::Family::Name("JetBrains Mono"));
                let spans = build_rich_spans(text, &highlights, &base_attrs);
                buffer.set_rich_text(
                    font_sys,
                    spans,
                    &base_attrs,
                    cosmic_text::Shaping::Advanced,
                    None,
                );
                // `set_rich_text` (unlike `set_text`) does NOT append a trailing
                // empty line for trailing newlines because `BidiParagraphs` does
                // not produce an empty trailing paragraph.  This causes a
                // mismatch: buffer line count differs from what `set_text` would
                // produce, leading to cursor clamping in `edit_text()` that
                // makes Enter at end-of-file appear to insert the newline
                // *before* the last line instead of after it.
                //
                // Mirror `set_text` behaviour: if the original text ends with a
                // newline and the last buffer line has a non-None line ending,
                // append an empty trailing line with `LineEnding::None`.
                if text.ends_with('\n')
                    && buffer
                        .lines
                        .last()
                        .is_some_and(|l| l.ending() != cosmic_text::LineEnding::None)
                {
                    buffer.lines.push(cosmic_text::BufferLine::new(
                        "",
                        cosmic_text::LineEnding::None,
                        cosmic_text::AttrsList::new(&base_attrs),
                        cosmic_text::Shaping::Advanced,
                    ));
                }
                return;
            }
        }
        // Fallback for large files or no language
        buffer.set_text(
            font_sys,
            text,
            &cosmic_text::Attrs::new().family(cosmic_text::Family::Name("JetBrains Mono")),
            cosmic_text::Shaping::Advanced,
            None,
        );
    }

    /// Run tree-sitter highlighting on the given text.
    /// Returns per-line highlight spans.
    /// Delegates to [`highlight::parse_file_highlights`] which handles
    /// both standard languages and Markdown's dual-grammar approach.
    fn compute_highlights(text: &str, lang: HighlightLanguage) -> FileHighlights {
        let mut parser = tree_sitter::Parser::new();
        highlight::parse_file_highlights(&mut parser, text, lang)
    }

    /// Return the normalised selection range (start before end).
    const fn selection_range(&self) -> (usize, usize, usize, usize) {
        let cl = self.cursor_line.get();
        let cc = self.cursor_col.get();
        let sl = self.sel_line.get();
        let sc = self.sel_col.get();
        // Normalise: compare by (line, col) lexicographically
        if cl < sl || (cl == sl && cc < sc) {
            (cl, cc, sl, sc)
        } else {
            (sl, sc, cl, cc)
        }
    }

    /// Clamp the selection range to valid line boundaries.
    /// Returns `None` when the start line is past the end of the buffer.
    fn clamped_selection_range(&self) -> Option<(usize, usize, usize, usize)> {
        let (sl, sc, el, ec) = self.selection_range();
        let line_count = self.line_count();
        if sl >= line_count {
            return None;
        }
        let el = el.min(line_count.saturating_sub(1));
        Some((sl, sc, el, ec))
    }

    /// Returns `(start_line, end_line)` from the current selection, or
    /// `(cursor_line, cursor_line)` if there is no selection.
    /// Returns `None` when the buffer is empty or the selection is invalid.
    fn selected_line_range(&self) -> Option<(usize, usize)> {
        if self.line_count() == 0 {
            return None;
        }

        if self.has_selection.get() {
            let (sl, _sc, el, _ec) = self.clamped_selection_range()?;
            Some((sl, el))
        } else {
            let line = self.cursor_line.get();
            Some((line, line))
        }
    }

    /// Delete the currently selected text and return the byte range that
    /// was removed. If there is no selection, returns `None`.
    fn delete_selection_get_range(&self) -> Option<(usize, usize)> {
        if !self.has_selection.get() {
            return None;
        }
        let text_buf = self.text();
        let (sl, sc, el, ec) = self.selection_range();
        let start_off = line_col_to_byte_offset(&text_buf, sl, sc);
        let end_off = line_col_to_byte_offset(&text_buf, el, ec);
        self.has_selection.set(false);
        self.cursor_line.set(sl);
        self.cursor_col.set(sc);
        Some((start_off, end_off))
    }

    /// If a selection exists, delete it and return `true`.
    fn delete_if_selected(&self) -> bool {
        if let Some((start, end)) = self.delete_selection_get_range() {
            self.edit_text(|text| {
                let mut new_text = text.to_string();
                new_text.replace_range(start..end, "");
                let (line, col) = byte_offset_to_line_col(&new_text, start);
                (new_text, Some((line, col)))
            });
            true
        } else {
            false
        }
    }

    /// Helper: apply a text edit that replaces a byte range. The closure
    /// receives the byte offset and can return an optional new cursor
    /// (line, col). If the closure returns `None`, cursor is not adjusted.
    fn edit_text(&self, f: impl FnOnce(&str) -> (String, Option<(usize, usize)>)) {
        let text_buf = self.text();
        let (new_text, new_cursor) = f(&text_buf);
        if new_text == text_buf {
            return;
        }
        // Save cursor position before re-highlighting (set_rich_text may
        // reshape the buffer internals).
        let saved_line = self.cursor_line.get();
        let saved_col = self.cursor_col.get();
        let saved_has_sel = self.has_selection.get();
        let saved_sel_line = self.sel_line.get();
        let saved_sel_col = self.sel_col.get();

        let language = self.language;
        with_font_system(|font_sys| {
            let mut buffer = self.buffer.borrow_mut();
            Self::set_buffer_text_highlighted(&mut buffer, font_sys, &new_text, language);
        });

        // Restore cursor/selection after re-highlighting.
        self.cursor_line.set(saved_line);
        self.cursor_col.set(saved_col);
        self.has_selection.set(saved_has_sel);
        self.sel_line.set(saved_sel_line);
        self.sel_col.set(saved_sel_col);

        if let Some((line, col)) = new_cursor {
            let max_line = self.line_count().saturating_sub(1);
            let line = line.min(max_line);
            let col = self.clamp_col_to_line(line, col);
            self.cursor_line.set(line);
            self.cursor_col.set(col);
            self.has_selection.set(false);
        }
    }

    /// Insert a single character at cursor.
    fn do_insert(&self, c: char) {
        // If there's a selection, replace it.
        let sel_range = self.delete_selection_get_range();
        self.edit_text(|text| {
            let mut new_text = text.to_string();
            if let Some((start, end)) = sel_range {
                new_text.replace_range(start..end, &c.to_string());
            } else {
                let offset =
                    line_col_to_byte_offset(text, self.cursor_line.get(), self.cursor_col.get());
                new_text.insert(offset, c);
            }
            let (new_line, new_col) = if c == '\n' {
                (self.cursor_line.get() + 1, 0)
            } else {
                (self.cursor_line.get(), self.cursor_col.get() + 1)
            };
            (new_text, Some((new_line, new_col)))
        });
    }

    /// Insert a newline at cursor, preserving the leading whitespace of the
    /// current line (auto-indent).
    fn do_enter(&self) {
        let sel_range = self.delete_selection_get_range();

        // Determine the "current line" after selection removal: if a selection
        // existed, `delete_selection_get_range` resets cursor to its start.
        let current_line = self.cursor_line.get();
        let leading_ws = self
            .buffer
            .borrow()
            .lines
            .get(current_line)
            .map(|l| {
                let line_text = l.text();
                let ws_len = line_text
                    .chars()
                    .take_while(|c| *c == ' ' || *c == '\t')
                    .count();
                line_text[..ws_len].to_string()
            })
            .unwrap_or_default();

        let ws_len = leading_ws.chars().count();
        self.edit_text(|text| {
            let mut new_text = text.to_string();
            if let Some((start, end)) = sel_range {
                new_text.replace_range(start..end, &format!("\n{leading_ws}"));
            } else {
                let offset = line_col_to_byte_offset(text, current_line, self.cursor_col.get());
                new_text.insert_str(offset, &format!("\n{leading_ws}"));
            }
            (new_text, Some((current_line + 1, ws_len)))
        });
    }

    /// Delete the character behind the cursor.
    fn do_backspace(&self) {
        if self.delete_if_selected() {
            return;
        }

        let (cl, cc) = (self.cursor_line.get(), self.cursor_col.get());
        if cl == 0 && cc == 0 {
            return; // Nothing to delete
        }
        self.edit_text(|text| {
            let offset = line_col_to_byte_offset(text, cl, cc);
            if offset == 0 {
                return (text.to_string(), None);
            }
            let prev_boundary = text.floor_char_boundary(offset.saturating_sub(1));
            let deleted = &text[prev_boundary..offset];
            let mut new_text = text.to_string();
            new_text.replace_range(prev_boundary..offset, "");

            let (new_line, new_col) = if deleted == "\n" {
                // Moved up from start of line: go to end of previous line
                let new_cl = cl.saturating_sub(1);
                let prev_line_text = self
                    .buffer
                    .borrow()
                    .lines
                    .get(new_cl)
                    .map_or(0, |l| l.text().chars().count());
                (new_cl, prev_line_text)
            } else {
                (cl, cc.saturating_sub(1))
            };
            (new_text, Some((new_line, new_col)))
        });
    }

    /// Delete the character in front of the cursor.
    fn do_delete(&self) {
        if self.delete_if_selected() {
            return;
        }

        let (cl, cc) = (self.cursor_line.get(), self.cursor_col.get());
        self.edit_text(|text| {
            let offset = line_col_to_byte_offset(text, cl, cc);
            if offset >= text.len() {
                return (text.to_string(), None);
            }
            let next_boundary = text[offset..]
                .chars()
                .next()
                .map_or(offset, |c| offset + c.len_utf8());
            let mut new_text = text.to_string();
            new_text.replace_range(offset..next_boundary, "");
            (new_text, None) // cursor stays the same
        });
    }

    /// Paste a string at cursor.
    fn do_paste(&self, s: &str) {
        let sel_range = self.delete_selection_get_range();
        let s = s.to_string();
        self.edit_text(move |text| {
            let mut new_text = text.to_string();
            let new_offset = if let Some((start, end)) = sel_range {
                new_text.replace_range(start..end, &s);
                start + s.len()
            } else {
                let offset =
                    line_col_to_byte_offset(text, self.cursor_line.get(), self.cursor_col.get());
                new_text.insert_str(offset, &s);
                offset + s.len()
            };
            let (line, col) = byte_offset_to_line_col(&new_text, new_offset);
            (new_text, Some((line, col)))
        });
    }

    /// Insert a tab at cursor, or indent each line in the selection.
    fn do_indent(&self) {
        if self.has_selection.get() {
            // Multi-line indent: prepend a tab to each line in the selection.
            let Some((sl, _sc, el, _ec)) = self.clamped_selection_range() else {
                return;
            };

            self.edit_text(|text| {
                let mut new_text = text.to_string();
                // Insert tabs at the start of each line in [sl, el].
                for line_idx in sl..=el {
                    let offset = line_col_to_byte_offset(&new_text, line_idx, 0);
                    new_text.insert(offset, '\t');
                }
                (new_text, None)
            });

            // Shift cursor and anchor columns right on indented lines (col 0
            // stays at line start, which becomes the new tab).
            let cl = self.cursor_line.get();
            let cc = self.cursor_col.get();
            if (sl..=el).contains(&cl) && cc > 0 {
                self.cursor_col.set(cc + 1);
            }
            if self.has_selection.get() {
                let anchor_line = self.sel_line.get();
                let anchor_col = self.sel_col.get();
                if (sl..=el).contains(&anchor_line) && anchor_col > 0 {
                    self.sel_col.set(anchor_col + 1);
                }
            }
        } else {
            let offset = line_col_to_byte_offset(
                &self.text(),
                self.cursor_line.get(),
                self.cursor_col.get(),
            );
            self.edit_text(|text| {
                let mut new_text = text.to_string();
                new_text.insert(offset, '\t');
                (
                    new_text,
                    Some((self.cursor_line.get(), self.cursor_col.get() + 1)),
                )
            });
        }
    }

    /// Remove leading whitespace from the current line, or outdent each
    /// line in the selection.
    fn do_unindent(&self) {
        if self.has_selection.get() {
            let Some((sl, _sc, el, _ec)) = self.clamped_selection_range() else {
                return;
            };

            let mut modified_lines: Vec<usize> = Vec::new();
            self.edit_text(|text| {
                let mut new_text = text.to_string();

                // Process lines bottom-up so byte offsets stay valid for
                // earlier lines when we remove from later lines.
                for line_idx in (sl..=el).rev() {
                    let line_text = new_text.lines().nth(line_idx).unwrap_or("").to_string();
                    let leading_count = line_text
                        .chars()
                        .take_while(|c| *c == ' ' || *c == '\t')
                        .count();
                    if leading_count == 0 {
                        continue;
                    }
                    modified_lines.push(line_idx);
                    let remove_count = 1.min(leading_count);
                    let remove_bytes = line_text[..remove_count].len();
                    let line_start = line_col_to_byte_offset(&new_text, line_idx, 0);
                    new_text.replace_range(line_start..line_start + remove_bytes, "");
                }

                (new_text, None)
            });

            let adjust_col = |line: usize, col: usize| {
                if modified_lines.contains(&line) && col > 0 {
                    col - 1
                } else {
                    col
                }
            };
            let cl = self.cursor_line.get();
            self.cursor_col.set(adjust_col(cl, self.cursor_col.get()));
            if self.has_selection.get() {
                let anchor_line = self.sel_line.get();
                self.sel_col
                    .set(adjust_col(anchor_line, self.sel_col.get()));
            }
        } else {
            let cl = self.cursor_line.get();
            let line_text = self
                .buffer
                .borrow()
                .lines
                .get(cl)
                .map(|l| l.text().to_string())
                .unwrap_or_default();

            let leading_spaces = line_text
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .count();
            if leading_spaces == 0 {
                return;
            }

            let remove_count = 1.min(leading_spaces);
            let remove_bytes = line_text[..remove_count].len();

            self.edit_text(|text| {
                let mut new_text = text.to_string();
                let line_start = line_col_to_byte_offset(text, cl, 0);
                new_text.replace_range(line_start..line_start + remove_bytes, "");
                let new_col = self.cursor_col.get().saturating_sub(1);
                (new_text, Some((cl, new_col)))
            });
        }
    }

    // ── Cursor movement helpers ───────────────────────────────────

    /// Execute a cursor movement operation, handling the
    /// `extend_selection` scaffolding automatically.
    ///
    /// If `extend_selection` is true, the current cursor position is
    /// recorded as the selection anchor before moving. The `compute`
    /// closure returns the target `(line, col)` or `None` to abort
    /// (e.g. at document boundaries).
    fn with_cursor_movement(
        &self,
        extend_selection: bool,
        compute: impl FnOnce() -> Option<(usize, usize)>,
    ) {
        if extend_selection {
            let anchor = (self.cursor_line.get(), self.cursor_col.get());
            if let Some((line, col)) = compute() {
                if !self.has_selection.get() {
                    self.sel_line.set(anchor.0);
                    self.sel_col.set(anchor.1);
                    self.has_selection.set(true);
                }
                self.set_cursor_pos(line, col);
                self.normalize_selection();
            }
        } else if let Some((line, col)) = compute() {
            self.move_to(line, col);
        }
    }

    /// Move cursor one character left.
    fn do_move_left(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let (line, col) = (self.cursor_line.get(), self.cursor_col.get());
            if col > 0 {
                Some((line, col - 1))
            } else if line > 0 {
                let prev_line = line - 1;
                let prev_len = self
                    .buffer
                    .borrow()
                    .lines
                    .get(prev_line)
                    .map_or(0, |l| l.text().chars().count());
                Some((prev_line, prev_len))
            } else {
                None
            }
        });
    }

    /// Move cursor one character right.
    fn do_move_right(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let (line, col) = (self.cursor_line.get(), self.cursor_col.get());
            let max_line = self.line_count().saturating_sub(1);
            let line_len = self
                .buffer
                .borrow()
                .lines
                .get(line)
                .map_or(0, |l| l.text().chars().count());
            if col < line_len {
                Some((line, col + 1))
            } else if line < max_line {
                Some((line + 1, 0))
            } else {
                None
            }
        });
    }

    /// Move cursor one line up.
    fn do_move_up(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let (line, col) = (self.cursor_line.get(), self.cursor_col.get());
            if line > 0 {
                Some((line - 1, col))
            } else {
                None
            }
        });
    }

    /// Move cursor one line down.
    fn do_move_down(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let (line, col) = (self.cursor_line.get(), self.cursor_col.get());
            let max_line = self.line_count().saturating_sub(1);
            if line < max_line {
                Some((line + 1, col))
            } else {
                None
            }
        });
    }

    /// Move cursor to start of current line.
    fn do_move_home(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let line = self.cursor_line.get();
            Some((line, 0))
        });
    }

    /// Move cursor to end of current line.
    fn do_move_end(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let line = self.cursor_line.get();
            let line_len = self
                .buffer
                .borrow()
                .lines
                .get(line)
                .map_or(0, |l| l.text().chars().count());
            Some((line, line_len))
        });
    }

    /// Move cursor one word left.
    fn do_move_word_left(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let text = self.text();
            let offset =
                line_col_to_byte_offset(&text, self.cursor_line.get(), self.cursor_col.get());
            if offset == 0 {
                None
            } else {
                Some(byte_offset_to_line_col(
                    &text,
                    find_word_start(&text, offset),
                ))
            }
        });
    }

    /// Move cursor one word right.
    fn do_move_word_right(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let text = self.text();
            let offset =
                line_col_to_byte_offset(&text, self.cursor_line.get(), self.cursor_col.get());
            if offset >= text.len() {
                None
            } else {
                Some(byte_offset_to_line_col(&text, find_word_end(&text, offset)))
            }
        });
    }

    /// Move cursor to start of document.
    fn do_move_doc_start(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || Some((0, 0)));
    }

    /// Move cursor to end of document.
    fn do_move_doc_end(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let max_line = self.line_count().saturating_sub(1);
            let line_len = self
                .buffer
                .borrow()
                .lines
                .get(max_line)
                .map_or(0, |l| l.text().chars().count());
            Some((max_line, line_len))
        });
    }

    /// Move cursor one page up (by viewport height lines).
    fn do_move_page_up(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let line = self.cursor_line.get();
            let col = self.cursor_col.get();
            let page_lines = PAGE_SCROLL_LINES;
            Some(if line > page_lines {
                (line - page_lines, col)
            } else {
                (0, col)
            })
        });
    }

    /// Move cursor one page down.
    fn do_move_page_down(&self, extend_selection: bool) {
        self.with_cursor_movement(extend_selection, || {
            let line = self.cursor_line.get();
            let max_line = self.line_count().saturating_sub(1);
            let col = self.cursor_col.get();
            let page_lines = PAGE_SCROLL_LINES;
            Some((line.saturating_add(page_lines).min(max_line), col))
        });
    }

    /// Clear selection when anchor and cursor coincide (zero-width range).
    fn normalize_selection(&self) {
        if self.has_selection.get()
            && self.sel_line.get() == self.cursor_line.get()
            && self.sel_col.get() == self.cursor_col.get()
        {
            self.has_selection.set(false);
        }
    }

    // ── Delete word helpers ───────────────────────────────────────

    /// Delete from cursor backward to start of previous word.
    fn do_delete_word_back(&self) {
        if self.delete_if_selected() {
            return;
        }

        // Early guard: cursor at document start → no text before cursor, no-op.
        // Equivalent to `offset == 0` but avoids allocating `self.text()`.
        if self.cursor_line.get() == 0 && self.cursor_col.get() == 0 {
            return;
        }

        let (cl, cc) = (self.cursor_line.get(), self.cursor_col.get());
        self.edit_text(|text| {
            let offset = line_col_to_byte_offset(text, cl, cc);
            let word_start = find_word_start(text, offset);
            let mut new_text = text.to_string();
            new_text.replace_range(word_start..offset, "");
            let (line, col) = byte_offset_to_line_col(&new_text, word_start);
            (new_text, Some((line, col)))
        });
    }

    /// Delete from cursor forward to start of next word.
    fn do_delete_word_forward(&self) {
        if self.delete_if_selected() {
            return;
        }

        let (cl, cc) = (self.cursor_line.get(), self.cursor_col.get());
        self.edit_text(|text| {
            let offset = line_col_to_byte_offset(text, cl, cc);
            if offset >= text.len() {
                return (text.to_string(), None);
            }
            let word_end = find_word_end(text, offset);
            let mut new_text = text.to_string();
            new_text.replace_range(offset..word_end, "");
            (new_text, None)
        });
    }

    // ── New editor actions ────────────────────────────────────────

    /// Toggle line comments on the current line or selection.
    ///
    /// Uses the comment prefix determined by the file language. When a
    /// selection exists, operates on every line touched by the selection.
    /// Leading whitespace is preserved.
    fn do_toggle_line_comment(&self) {
        let ext = self.file_extension();
        let Some(prefix) = line_comment_prefix(self.language, ext.as_deref()) else {
            return; // No comment syntax for this language.
        };

        let Some((start_line, end_line)) = self.selected_line_range() else {
            return;
        };

        let text = self.text();
        let mut replacements: Vec<(usize, usize, String)> = Vec::new();
        let mut first_toggled_col = None;

        for line_idx in start_line..=end_line {
            let Some((ls, le)) = line_byte_range(&text, line_idx) else {
                continue;
            };
            let line_slice = &text[ls..le];
            let (body, ending) = split_line_body_and_ending(line_slice);
            let trimmed = body.trim_start();
            let leading_ws_len = body.len() - trimmed.len();
            let leading_ws = &body[..leading_ws_len];

            let (new_body, toggled_col) = if let Some(stripped) = trimmed.strip_prefix(prefix) {
                let after_comment = stripped.strip_prefix(' ').unwrap_or(stripped);
                (format!("{leading_ws}{after_comment}"), Some(leading_ws_len))
            } else {
                (
                    format!("{leading_ws}{prefix} {trimmed}"),
                    Some(leading_ws_len + prefix.len() + 1),
                )
            };

            if line_idx == start_line {
                first_toggled_col = toggled_col;
            }
            replacements.push((ls, le, format!("{new_body}{ending}")));
        }

        let target_line = start_line;
        let target_col = first_toggled_col.unwrap_or(0);

        self.edit_text(|text| {
            let mut new_text = text.to_string();
            for (ls, le, replacement) in replacements.into_iter().rev() {
                new_text.replace_range(ls..le, &replacement);
            }
            (new_text, Some((target_line, target_col)))
        });
    }

    /// Jump cursor to the matching bracket (parentheses, brackets, braces).
    fn do_jump_to_matching_bracket(&self) {
        let text = self.text();
        let (cl, cc) = (self.cursor_line.get(), self.cursor_col.get());
        #[allow(clippy::similar_names)]
        if let Some(pair) = find_matching_bracket(&text, cl, cc) {
            // pair is ((open_line, open_col), (close_line, close_col)).
            // If cursor is on/near the opening bracket → jump to close.
            // If cursor is on/near the closing bracket → jump to open.
            let ((ol, oc), (close_l, close_c)) = pair;
            let at_open = cl == ol && (cc == oc || cc == oc + 1 || (cc > 0 && cc - 1 == oc));
            let at_close = cl == close_l && cc == close_c;

            if at_open {
                self.move_to(close_l, close_c);
            } else if at_close {
                self.move_to(ol, oc + 1);
            }
            // If neither, still try: jump to the matching position that is
            // farther from the current cursor (prefer the other end).
            else {
                // Determine which end of the pair is farther using
                // saturating absolute difference (no signed cast needed).
                let open_dist = cl.abs_diff(ol) + cc.abs_diff(oc);
                let close_dist = cl.abs_diff(close_l) + cc.abs_diff(close_c);
                if open_dist <= close_dist {
                    self.move_to(close_l, close_c);
                } else {
                    self.move_to(ol, oc + 1);
                }
            }
        }
    }

    /// Delete the current line (or selected lines).
    ///
    /// After deletion, the cursor is placed at the start of the line that
    /// followed the deleted content. If the last line was deleted, the cursor
    /// moves to the new last line.
    fn do_delete_line(&self) {
        let line_count = self.line_count();
        if line_count == 0 {
            return;
        }

        let Some((start_line, end_line)) = self.selected_line_range() else {
            return;
        };

        // We'll replace the range from start of start_line to start of
        // the line after end_line (or end of file). This effectively removes
        // the entire line(s).
        self.edit_text(|text| {
            let mut new_text = text.to_string();
            let start_off = line_col_to_byte_offset(text, start_line, 0);
            let end_off = if end_line + 1 < line_count {
                line_col_to_byte_offset(text, end_line + 1, 0)
            } else {
                text.len()
            };

            // If this is the last line and there's a preceding newline,
            // remove the preceding newline too (preserve file structure).
            let adjusted_end = if end_line + 1 >= line_count && start_line > 0 {
                // Remove the newline before the last line.
                let prev_line_end = line_col_to_byte_offset(text, start_line, 0).saturating_sub(1);
                // Check if it's a \r\n situation
                if prev_line_end > 0
                    && text.as_bytes().get(prev_line_end) == Some(&b'\n')
                    && text.as_bytes().get(prev_line_end.saturating_sub(1)) == Some(&b'\r')
                {
                    start_off - 2
                } else if text.as_bytes().get(prev_line_end) == Some(&b'\n') {
                    start_off.saturating_sub(1)
                } else {
                    start_off
                }
            } else {
                start_off
            };

            // If deleting the only line, just clear it.
            if line_count == 1 {
                new_text.clear();
                return (new_text, Some((0, 0)));
            }

            new_text.replace_range(adjusted_end..end_off, "");

            // Determine new cursor: same line index as start_line if possible.
            let new_line_count = new_text.lines().count().max(1);
            let target_line = start_line.min(new_line_count.saturating_sub(1));
            (new_text, Some((target_line, 0)))
        });
    }

    /// Duplicate the current line (or selected lines).
    ///
    /// The duplicated content is inserted after the last selected line (or
    /// after the current line). The cursor is placed at the start of the
    /// first duplicated line.
    fn do_duplicate_line(&self) {
        let line_count = self.line_count();
        if line_count == 0 {
            return;
        }

        let Some((start_line, end_line)) = self.selected_line_range() else {
            return;
        };

        self.edit_text(|text| {
            let mut new_text = text.to_string();
            let line_ending = detect_line_ending(text).as_str();

            // Get the text of the lines to duplicate.
            let duplicated: String = if end_line + 1 < line_count {
                // Lines start at start_line, end at end_line (inclusive).
                let start_off = line_col_to_byte_offset(text, start_line, 0);
                let end_off = line_col_to_byte_offset(text, end_line + 1, 0);
                text[start_off..end_off].to_string()
            } else {
                // Last line(s) — may not have a trailing newline.
                let start_off = line_col_to_byte_offset(text, start_line, 0);
                let dup_text = text[start_off..].to_string();
                // Prepend a newline so the duplicated content appears on a new
                // line after the original last line.
                if !text.ends_with(line_ending) {
                    format!("{line_ending}{dup_text}")
                } else {
                    dup_text
                }
            };

            // Insert after end_line.
            let insert_off = if end_line + 1 < line_count {
                line_col_to_byte_offset(text, end_line + 1, 0)
            } else {
                text.len()
            };

            new_text.insert_str(insert_off, &duplicated);

            let target_line = end_line + 1;
            (new_text, Some((target_line, 0)))
        });
    }

    /// Move the current line (or selected lines) up by one.
    /// At the first line boundary, this is a no-op.
    fn do_move_line_up(&self) {
        let line_count = self.line_count();
        if line_count <= 1 {
            return;
        }

        let Some((start_line, end_line)) = self.selected_line_range() else {
            return;
        };

        if start_line == 0 {
            return; // Already at top.
        }

        let swap_line = start_line.saturating_sub(1);

        self.edit_text(|text| {
            let default_ending = detect_line_ending(text);
            let had_trailing = has_trailing_newline(text);
            let mut lines = logical_lines(text);
            if swap_line >= lines.len() || end_line >= lines.len() {
                return (text.to_string(), None);
            }
            if start_line == end_line {
                swap_lines_with_endings(&mut lines, swap_line, start_line);
            } else {
                let block: Vec<_> = lines.drain(start_line..=end_line).collect();
                lines.splice(swap_line..swap_line, block);
            }
            fix_line_endings(&mut lines, had_trailing, default_ending);
            (reassemble_lines(&lines), Some((swap_line, 0)))
        });
    }

    /// Move the current line (or selected lines) down by one.
    /// At the last line boundary, this is a no-op.
    fn do_move_line_down(&self) {
        let line_count = self.line_count();
        if line_count <= 1 {
            return;
        }

        let Some((start_line, end_line)) = self.selected_line_range() else {
            return;
        };

        if end_line + 1 >= line_count {
            return; // Already at bottom.
        }

        let swap_line = end_line + 1;

        self.edit_text(|text| {
            let default_ending = detect_line_ending(text);
            let had_trailing = has_trailing_newline(text);
            let mut lines = logical_lines(text);
            if swap_line >= lines.len() || end_line >= lines.len() {
                return (text.to_string(), None);
            }
            if start_line == end_line {
                swap_lines_with_endings(&mut lines, start_line, swap_line);
            } else {
                let below = lines.remove(swap_line);
                let block: Vec<_> = lines.drain(start_line..=end_line).collect();
                lines.splice(start_line..start_line, std::iter::once(below));
                let block_insert_at = start_line + 1;
                lines.splice(block_insert_at..block_insert_at, block);
            }
            fix_line_endings(&mut lines, had_trailing, default_ending);
            (reassemble_lines(&lines), Some((start_line + 1, 0)))
        });
    }
}

impl Default for EditorBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum number of characters to scan in each direction when finding
/// matching brackets. Prevents frame drops on large files.
const BRACKET_SCAN_LIMIT: usize = 20_000;

/// Result of bracket matching: the (line, col) of the opening bracket and
/// the (line, col) of the closing bracket.
pub type BracketPair = ((usize, usize), (usize, usize));

/// Find a matching bracket adjacent to the cursor.
///
/// Checks the character *before* the cursor for opening brackets `(`, `[`, `{`
/// and the character *at* the cursor for closing brackets `)`, `]`, `}`.
/// Scans forward/backward with a depth counter, bounded by `BRACKET_SCAN_LIMIT`
/// characters to avoid performance issues on large files.
#[must_use]
pub fn find_matching_bracket(
    text: &str,
    cursor_line: usize,
    cursor_col: usize,
) -> Option<BracketPair> {
    let offset = line_col_to_byte_offset(text, cursor_line, cursor_col);
    let bytes = text.as_bytes();

    // Check character *before* cursor — cursor is right after an opening bracket.
    if offset > 0 {
        let c = bytes[offset - 1];
        let (open, close) = match c {
            b'(' => (b'(', b')'),
            b'[' => (b'[', b']'),
            b'{' => (b'{', b'}'),
            _ => (0, 0),
        };

        if open != 0 {
            // Scan forward for matching close.
            let mut depth = 1u32;
            let search_end = (offset + BRACKET_SCAN_LIMIT).min(bytes.len());
            for (i, &b) in bytes[offset..search_end].iter().enumerate() {
                let abs_i = offset + i;
                if b == open {
                    depth += 1;
                } else if b == close {
                    depth -= 1;
                    if depth == 0 {
                        let (line, col) = byte_offset_to_line_col(text, abs_i);
                        return Some(((cursor_line, cursor_col.saturating_sub(1)), (line, col)));
                    }
                }
            }
            return None;
        }
    }

    // Check character *at* cursor — cursor is right before a closing bracket.
    if let Some(&c) = bytes.get(offset) {
        let (open, close) = match c {
            b')' => (b'(', b')'),
            b']' => (b'[', b']'),
            b'}' => (b'{', b'}'),
            _ => return None,
        };
        let mut depth = 1u32;
        let search_start = offset.saturating_sub(BRACKET_SCAN_LIMIT);
        for (rev_i, &b) in bytes[search_start..offset].iter().rev().enumerate() {
            let abs_i = offset - 1 - rev_i;
            if b == close {
                depth += 1;
            } else if b == open {
                depth -= 1;
                if depth == 0 {
                    let (line, col) = byte_offset_to_line_col(text, abs_i);
                    return Some(((line, col), (cursor_line, cursor_col)));
                }
            }
        }
    }

    None
}

/// Determine the line comment prefix for a given language and file extension.
///
/// Returns `None` for languages that have no standard line comment syntax
/// (JSON, HTML, Markdown) or when the language is unknown and no extension
/// hint was provided.
///
/// Uses the `HighlightLanguage` first; falls back to extension-based matching
/// for YAML (which has no `HighlightLanguage` variant yet). Returns `None`
/// for any unrecognised extension.
#[must_use]
pub fn line_comment_prefix(
    lang: Option<HighlightLanguage>,
    ext: Option<&str>,
) -> Option<&'static str> {
    // Prefer the HighlightLanguage match.
    if let Some(lang) = lang {
        return match lang {
            HighlightLanguage::Rust
            | HighlightLanguage::JavaScript
            | HighlightLanguage::TypeScript
            | HighlightLanguage::TSX
            | HighlightLanguage::Go
            | HighlightLanguage::C
            | HighlightLanguage::Css => Some("//"),
            HighlightLanguage::Python
            | HighlightLanguage::Ruby
            | HighlightLanguage::Bash
            | HighlightLanguage::Toml => Some("#"),
            HighlightLanguage::Sql => Some("--"),
            // JSON, HTML, Markdown have no standard line comment.
            HighlightLanguage::Json | HighlightLanguage::Html | HighlightLanguage::Markdown => {
                return None;
            }
        };
    }

    // Fallback: match by extension for languages not in HighlightLanguage.
    if let Some(ext) = ext {
        return match ext {
            "yaml" | "yml" | "dockerfile" | "makefile" | "mak" | "cmake" => Some("#"),
            _ => None,
        };
    }

    None
}

/// Build a list of `(&str, Attrs)` pairs from [`FileHighlights`] that
/// covers every byte of `text` with no gaps. Each span gets a color from
/// its [`super::highlight::HighlightClass`]; gaps between tree-sitter captures get the
/// default color (from `base_attrs`).
///
/// The returned spans borrow from `text`. The caller must ensure `text`
/// outlives the returned spans.
fn build_rich_spans<'a>(
    text: &'a str,
    highlights: &FileHighlights,
    base_attrs: &cosmic_text::Attrs<'a>,
) -> Vec<(&'a str, cosmic_text::Attrs<'a>)> {
    let mut result: Vec<(&str, cosmic_text::Attrs)> = Vec::new();
    let mut byte_pos = 0usize;

    for line_spans in &highlights.spans {
        // Find the start of this line in text
        let line_start = byte_pos;
        // Find the end of this line (including \n if present)
        let line_end = text[byte_pos..]
            .find('\n')
            .map_or(text.len(), |nl| byte_pos + nl + 1);

        let mut cursor = line_start;

        for span in line_spans {
            let s = line_start + span.start;
            let e = (line_start + span.end).min(line_end);

            if s > cursor {
                // Gap before this span — fill with base attrs
                push_or_merge(text, &mut result, &text[cursor..s], base_attrs.clone());
            }

            if e > s {
                let color = iced_color_to_cosmic(span.highlight_class.color());
                let attrs = base_attrs.clone().color(color);
                push_or_merge(text, &mut result, &text[s..e], attrs);
                cursor = e;
            }
        }

        // Fill any remaining text on this line with base attrs
        if cursor < line_end {
            push_or_merge(
                text,
                &mut result,
                &text[cursor..line_end],
                base_attrs.clone(),
            );
        }

        byte_pos = line_end;
    }

    result
}

/// Extract full text from a [`cosmic_text::Buffer`] by joining lines with
/// newline characters.
fn buffer_text(buffer: &cosmic_text::Buffer) -> String {
    let mut result = String::with_capacity(buffer.lines.iter().map(|l| l.text().len() + 1).sum());
    for (i, line) in buffer.lines.iter().enumerate() {
        if i > 0 {
            result.push('\n');
        }
        result.push_str(line.text());
    }
    result
}

/// Convert a character-based (line, column) pair to a byte offset into
/// the given text.
fn line_col_to_byte_offset(text: &str, line: usize, col: usize) -> usize {
    let mut current_line = 0;
    let mut byte_offset = 0;
    for c in text.chars() {
        if current_line == line {
            break;
        }
        if c == '\n' {
            current_line += 1;
        }
        byte_offset += c.len_utf8();
    }
    for (current_col, c) in text[byte_offset..].chars().enumerate() {
        if current_col == col || c == '\n' {
            break;
        }
        byte_offset += c.len_utf8();
    }
    byte_offset
}

/// Byte range `[start, end)` for a logical line, including its line ending.
fn line_byte_range(text: &str, line_idx: usize) -> Option<(usize, usize)> {
    if text.is_empty() {
        return (line_idx == 0).then_some((0, 0));
    }
    let mut current = 0usize;
    let mut start = 0usize;
    for (i, b) in text.bytes().enumerate() {
        if b == b'\n' {
            if current == line_idx {
                return Some((start, i + 1));
            }
            current += 1;
            start = i + 1;
        }
    }
    if current == line_idx {
        return Some((start, text.len()));
    }
    None
}

/// Split a line slice into body text and trailing `\n` / `\r\n`.
fn split_line_body_and_ending(line: &str) -> (&str, &str) {
    if let Some(body) = line.strip_suffix("\r\n") {
        (body, "\r\n")
    } else if let Some(body) = line.strip_suffix('\n') {
        (body, "\n")
    } else {
        (line, "")
    }
}

/// Split buffer text into logical lines preserving each line's ending.
fn logical_lines(text: &str) -> Vec<(String, String)> {
    let mut lines = Vec::new();
    let mut idx = 0;
    while let Some((ls, le)) = line_byte_range(text, idx) {
        let slice = &text[ls..le];
        let (body, ending) = split_line_body_and_ending(slice);
        lines.push((body.to_string(), ending.to_string()));
        idx += 1;
    }
    if lines.is_empty() {
        lines.push((String::new(), String::new()));
    }
    lines
}

fn reassemble_lines(lines: &[(String, String)]) -> String {
    let mut out = String::new();
    for (body, ending) in lines {
        out.push_str(body);
        out.push_str(ending);
    }
    out
}

/// Line ending convention detected for a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LineEnding {
    Lf,
    Crlf,
}

impl LineEnding {
    /// Return the string representation of this line ending.
    #[must_use]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            LineEnding::Lf => "\n",
            LineEnding::Crlf => "\r\n",
        }
    }
}

/// Check whether a string has a trailing newline.
#[must_use]
pub(crate) fn has_trailing_newline(text: &str) -> bool {
    text.ends_with('\n')
}

/// Detect the line ending convention (LF vs CRLF) by scanning the first 64 KiB.
#[must_use]
pub(crate) fn detect_line_ending(text: &str) -> LineEnding {
    let bytes = text.as_bytes();
    let limit = bytes.len().min(65536);
    let has_crlf = bytes[..limit].windows(2).any(|w| w == b"\r\n");
    if has_crlf {
        LineEnding::Crlf
    } else {
        LineEnding::Lf
    }
}

fn swap_lines_with_endings(lines: &mut [(String, String)], i: usize, j: usize) {
    let end_i = lines[i].1.clone();
    let end_j = lines[j].1.clone();
    lines.swap(i, j);
    lines[i].1 = end_i;
    lines[j].1 = end_j;
}

fn fix_line_endings(
    lines: &mut [(String, String)],
    had_trailing: bool,
    default_ending: LineEnding,
) {
    if lines.is_empty() {
        return;
    }
    let default_str = default_ending.as_str();
    let last_idx = lines.len() - 1;
    for line in &mut lines[..last_idx] {
        if line.1.is_empty() {
            line.1 = default_str.to_string();
        }
    }
    if had_trailing {
        if lines[last_idx].1.is_empty() {
            lines[last_idx].1 = default_str.to_string();
        }
    } else {
        lines[last_idx].1.clear();
    }
}

/// Convert a byte offset into a (line, column) pair, where column is
/// character-based (not byte-based).
pub(crate) fn byte_offset_to_line_col(text: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(text.len());
    let prefix = &text[..offset];
    let line = prefix.bytes().filter(|&b| b == b'\n').count();
    let last_newline = prefix.rfind('\n').map_or(0, |p| p + 1);
    let col = prefix[last_newline..].chars().count();
    (line, col)
}

/// Convert a character-based column on a single line to a byte offset within
/// that line's text. Used when passing indices to `cosmic_text::Cursor`.
pub(crate) fn char_col_to_byte_offset_in_line(line_text: &str, char_col: usize) -> usize {
    line_text.chars().take(char_col).map(char::len_utf8).sum()
}

/// Byte range `[start, end)` covering the single character at `char_col` on
/// `line_text`, or an empty range at the line end if `char_col` is past EOF.
pub(crate) fn char_col_to_byte_range_in_line(line_text: &str, char_col: usize) -> (usize, usize) {
    let start = char_col_to_byte_offset_in_line(line_text, char_col);
    let end = line_text[start..]
        .chars()
        .next()
        .map_or(start, |c| start + c.len_utf8());
    (start, end)
}

/// Classify a character for word boundary detection.
/// Returns `true` if the character is alphanumeric or underscore.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Find the byte offset of the start of the word before the given offset.
fn find_word_start(text: &str, offset: usize) -> usize {
    let offset = offset.min(text.len());
    // Skip non-word chars and whitespace immediately before offset
    let mut pos = offset;

    // Skip whitespace/non-word chars going backwards
    while pos > 0 {
        let c = text[..pos].chars().last();
        match c {
            Some(ch) if !is_word_char(ch) && !ch.is_whitespace() => {
                pos -= ch.len_utf8();
            }
            Some(ch) if ch.is_whitespace() => {
                pos -= ch.len_utf8();
            }
            _ => break,
        }
    }

    // Now skip alphanumeric chars going backwards to find word start
    while pos > 0 {
        let c = text[..pos].chars().last();
        match c {
            Some(ch) if is_word_char(ch) => {
                pos -= ch.len_utf8();
            }
            _ => break,
        }
    }

    pos
}

/// Find the byte offset of the start of the next word after the given offset.
fn find_word_end(text: &str, offset: usize) -> usize {
    let len = text.len();
    if offset >= len {
        return len;
    }

    let mut pos = offset;

    // Helper: get char at current position
    let char_at = |p: usize| text[p..].chars().next().map(|c| (c, c.len_utf8()));

    // Skip word chars
    while let Some((ch, ch_len)) = char_at(pos) {
        if !is_word_char(ch) {
            break;
        }
        pos += ch_len;
        if pos >= len {
            return len;
        }
    }

    // Skip non-word, non-whitespace
    while let Some((ch, ch_len)) = char_at(pos) {
        if is_word_char(ch) || ch.is_whitespace() {
            break;
        }
        pos += ch_len;
        if pos >= len {
            return len;
        }
    }

    // Skip whitespace to find start of next word
    while let Some((ch, ch_len)) = char_at(pos) {
        if !ch.is_whitespace() {
            break;
        }
        pos += ch_len;
        if pos >= len {
            return len;
        }
    }

    pos
}

/// Find the bounds of the word at the given byte offset by expanding
/// outward from the click point. Returns (start_byte, end_byte).
///
/// - On a word character: expands backward and forward over word chars,
///   respecting newline boundaries.
/// - On punctuation: returns a single-character range.
/// - On whitespace or at end of text: returns (offset, offset) — zero-width.
fn word_bounds_at(text: &str, byte_offset: usize) -> (usize, usize) {
    let len = text.len();

    // Past end of text → zero-width.
    if byte_offset >= len {
        return (len, len);
    }

    let first_char = text[byte_offset..].chars().next().unwrap();

    // Newline acts as a boundary — zero-width.
    if first_char == '\n' {
        return (byte_offset, byte_offset);
    }

    if is_word_char(first_char) {
        // Expand backward over word chars (stop at newlines or non-word chars).
        let mut start = byte_offset;
        loop {
            if start == 0 {
                break;
            }
            let c = text[..start].chars().last().unwrap();
            if c == '\n' || !is_word_char(c) {
                break;
            }
            start -= c.len_utf8();
        }

        // Expand forward over word chars.
        let mut end = byte_offset + first_char.len_utf8();
        while end < len {
            let c = text[end..].chars().next().unwrap();
            if c == '\n' || !is_word_char(c) {
                break;
            }
            end += c.len_utf8();
        }

        (start, end)
    } else if first_char.is_whitespace() {
        // Whitespace → zero-width (caller falls through to MoveTo).
        (byte_offset, byte_offset)
    } else {
        // Punctuation → single character.
        (byte_offset, byte_offset + first_char.len_utf8())
    }
}

// ── EditorWidget ────────────────────────────────────────────────────

use std::sync::Arc;

use iced::advanced::graphics::text::{self as graphics_text, Raw as TextRaw};
use iced::advanced::layout::{self, Layout};
use iced::advanced::mouse;
use iced::advanced::renderer;
use iced::advanced::widget::{self, Tree, Widget};
use iced::advanced::{Shell, graphics};
use iced::keyboard::{self, key};
use iced::window;
use iced::{Event, Length, Point, Rectangle, Size};

use std::time::Duration;

use super::theme;

/// Convert a screen-space mouse position to a (line, col) pair using the
/// cosmic_text Buffer hit-test. Returns `None` if the mouse is outside the
/// text area (in the gutter/padding area).
fn hit_test(
    buffer: &EditorBuffer,
    layout: Layout<'_>,
    cursor: mouse::Cursor,
    gutter_width: f32,
    padding: f32,
) -> Option<(usize, usize)> {
    let bounds = layout.bounds();
    let pos = cursor.position_in(bounds)?;

    // Text area origin within the widget
    let text_x = padding + gutter_width + 4.0; // 4 px gap between gutter and text
    let text_y = padding;

    let buf_x = pos.x - text_x;
    let buf_y = pos.y - text_y;

    // Ignore clicks/drags in the gutter/padding area
    if buf_x < 0.0 || buf_y < 0.0 {
        return None;
    }

    let buf = buffer.borrow_buffer();
    let hit = buf.hit(buf_x, buf_y)?;
    let line_text = buf.lines.get(hit.line).map_or("", |l| l.text());
    let byte_offset = hit.index.min(line_text.len());
    let col = line_text[..byte_offset].chars().count();
    Some((hit.line, col))
}

/// Find the layout run that contains the cursor position, accounting for
/// line wrapping. When a logical line wraps across multiple visual rows,
/// returns the visual row whose glyph character range contains `cursor_col`.
/// Falls back to the last run for the logical line if the cursor is at or
/// past the end of all glyphs (e.g. cursor is just past the last character,
/// or clicking in the margin area after a wrapped line end).
fn find_cursor_run<'a>(
    runs: impl Iterator<Item = cosmic_text::LayoutRun<'a>>,
    cursor_line: usize,
    cursor_col: usize,
) -> Option<cosmic_text::LayoutRun<'a>> {
    let mut last_for_line: Option<cosmic_text::LayoutRun<'a>> = None;
    for run in runs {
        if run.line_i != cursor_line {
            if last_for_line.is_some() {
                break; // Past the target line's runs
            }
            continue;
        }
        // Check if cursor column falls within this run's glyph range.
        // LayoutGlyph.start / .end are byte offsets into the line text;
        // cursor_col is a character count, so convert for comparison.
        // Use <= for last.end so the cursor at exact visual row
        // boundaries is associated with the previous run (where the
        // cursor is visually at the end), not the next one — this
        // preserves horizontal position when navigating Up/Down
        // between wrapped visual rows.
        if let (Some(first), Some(last)) = (run.glyphs.first(), run.glyphs.last()) {
            let first_char = run.text[..first.start].chars().count();
            let last_char = run.text[..last.end.min(run.text.len())].chars().count();
            if cursor_col >= first_char && cursor_col <= last_char {
                return Some(run);
            }
        }
        last_for_line = Some(run);
    }
    // Fallback: cursor at or past end of line — use last run
    last_for_line
}

/// Persistent state stored in `widget::Tree::State`.
struct EditorWidgetState {
    /// The `Arc<Buffer>` must live across frames for fill_raw to work.
    buffer_for_render: Option<Arc<cosmic_text::Buffer>>,
    /// Timestamp of last cursor blink toggle.
    last_blink: std::time::Instant,
    /// Current vertical scroll offset in pixels.
    scroll_y: f32,
    /// Maximum allowed vertical scroll offset.
    max_scroll_y: f32,
    /// Cached gutter width in pixels (computed per frame in layout).
    gutter_width: f32,
    /// Whether the left mouse button is currently held (for drag-selection).
    mouse_held: bool,
    /// Whether auto-scroll-to-cursor is active. Disabled by manual wheel
    /// scrolling, re-enabled by keyboard cursor movement.
    auto_scroll_enabled: bool,
    /// Instant of the last mouse click, used for double-click detection.
    last_click_time: Option<std::time::Instant>,
    /// (line, col) of the last mouse click, used for double-click proximity check.
    last_click_pos: Option<(usize, usize)>,
    /// Text from the most recent IME Commit, if any. Used to suppress the
    /// duplicate `KeyPressed.text` that follows on some platforms (Linux/IBus).
    /// Cleared after suppression or when a non-matching key event arrives.
    ime_commit_suppress: Option<String>,
    /// Last active buffer key (typically file path) — used to reset scroll
    /// and interaction state when switching tabs.
    last_buffer_key: Option<String>,
    /// Tracks the current keyboard modifiers (shift, ctrl, alt, etc.).
    /// Updated from `ModifiersChanged` events. Used to detect shift+click
    /// for extending the selection.
    modifiers: keyboard::Modifiers,
}

impl Default for EditorWidgetState {
    fn default() -> Self {
        Self {
            buffer_for_render: None,
            last_blink: std::time::Instant::now(),
            scroll_y: 0.0,
            max_scroll_y: 0.0,
            gutter_width: 0.0,
            mouse_held: false,
            auto_scroll_enabled: true,
            last_click_time: None,
            last_click_pos: None,
            ime_commit_suppress: None,
            last_buffer_key: None,
            modifiers: keyboard::Modifiers::empty(),
        }
    }
}

/// A custom Iced widget that renders an [`EditorBuffer`] using `fill_raw`,
/// with line numbers, syntax highlighting, selection, and cursor.
pub struct EditorWidget<'a> {
    buffer: &'a EditorBuffer,
    font_size: f32,
    padding: f32,
    /// When `true`, skip all keyboard event processing (e.g. when the
    /// file-tree panel or find/replace bar has focus).
    ignore_keyboard: bool,
    /// Find match highlights: `Vec<(line, byte_col_start, byte_col_end)>`.
    /// Set fresh each frame from the editor page's find/replace state.
    /// Empty/none when no find bar is open or no matches exist.
    matches: Option<Vec<(usize, usize, usize)>>,
    /// Index of the currently-focused match within `matches`.
    /// Used to render the current match with a stronger highlight color.
    match_current_idx: usize,
    /// Blink tick counter from the editor state.
    /// Incremented on each `BlinkTick` subscription event to force Iced
    /// to redraw the widget even when no other state has changed.
    blink_tick: u64,
    /// Matching bracket pair to highlight, if any.
    /// Each element is `(line, col)`.
    bracket_pair: Option<((usize, usize), (usize, usize))>,
    /// Identity of the active buffer (typically file path). When this
    /// changes, widget scroll and interaction state are reset.
    buffer_key: Option<&'a str>,
}

impl<'a> EditorWidget<'a> {
    /// Create a new [`EditorWidget`].
    pub const fn new(buffer: &'a EditorBuffer) -> Self {
        Self {
            buffer,
            font_size: 13.0,
            padding: 8.0,
            ignore_keyboard: false,
            matches: None,
            match_current_idx: 0,
            blink_tick: 0,
            bracket_pair: None,
            buffer_key: None,
        }
    }

    /// Set the font size.
    #[must_use]
    pub const fn font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Set the padding.
    #[must_use]
    pub const fn padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    /// When `true`, skip all keyboard event processing.
    /// Set this when another UI element (tree panel, find/replace bar)
    /// has keyboard focus and should receive the events instead.
    #[must_use]
    pub const fn ignore_keyboard(mut self, ignore: bool) -> Self {
        self.ignore_keyboard = ignore;
        self
    }

    /// Set find match highlights for the current frame.
    /// Each tuple is `(line, byte_col_start, byte_col_end)`.
    /// `current_idx` is the index of the currently-focused match
    /// (rendered with a stronger highlight color).
    /// Pass an empty `Vec` to hide all highlights.
    #[must_use]
    pub fn matches(mut self, matches: Vec<(usize, usize, usize)>, current_idx: usize) -> Self {
        self.matches = if matches.is_empty() {
            None
        } else {
            Some(matches)
        };
        self.match_current_idx = current_idx;
        self
    }

    /// Set the blink tick counter.
    /// This is passed from the editor state's `BlinkTick` handler to force
    /// Iced to detect a widget change and schedule a redraw on each tick.
    #[must_use]
    pub const fn blink_tick(mut self, blink_tick: u64) -> Self {
        self.blink_tick = blink_tick;
        self
    }

    /// Set the matching bracket pair to highlight.
    /// `pair` is `((open_line, open_col), (close_line, close_col))`.
    /// Pass `None` to hide bracket highlighting.
    #[must_use]
    pub const fn bracket_pair(mut self, pair: Option<((usize, usize), (usize, usize))>) -> Self {
        self.bracket_pair = pair;
        self
    }

    /// Set the buffer identity key used to detect tab switches.
    #[must_use]
    pub const fn buffer_key(mut self, key: Option<&'a str>) -> Self {
        self.buffer_key = key;
        self
    }
}

impl<Theme, Renderer> Widget<EditorAction, Theme, Renderer> for EditorWidget<'_>
where
    Renderer: iced::advanced::Renderer + graphics::text::Renderer + iced::advanced::text::Renderer,
{
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn state(&self) -> widget::tree::State {
        widget::tree::State::Some(Box::new(EditorWidgetState::default()))
    }

    fn tag(&self) -> widget::tree::Tag {
        widget::tree::Tag::of::<EditorWidgetState>()
    }

    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    fn layout(
        &mut self,
        tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let bounds = limits.max();

        let state = tree.state.downcast_mut::<EditorWidgetState>();

        // Reset scroll/interaction state when the active buffer changes (tab switch).
        let current_key = self.buffer_key.map(str::to_string);
        if state.last_buffer_key != current_key {
            state.scroll_y = 0.0;
            state.auto_scroll_enabled = true;
            state.mouse_held = false;
            state.last_click_time = None;
            state.last_click_pos = None;
            state.last_buffer_key = current_key;
        }

        // ── Gutter width ───────────────────────────────────────────────
        let line_count = self.buffer.line_count();
        let gutter_width = {
            let digits = (line_count.max(1).ilog10() + 1).min(6) as f32;
            digits * 5.0 + 10.0
        };
        state.gutter_width = gutter_width;

        // ── Compute text area after gutter ─────────────────────────────
        let text_x = self.padding + gutter_width + 4.0; // 4px gap between gutter and text
        let text_area_width = (bounds.width - text_x - self.padding).max(0.0);
        let text_area_height = (bounds.height - self.padding * 2.0).max(0.0);

        // ── Shape the buffer with current scroll ───────────────────────
        let mut guard = graphics_text::font_system().write().unwrap_poison();
        let font_sys = guard.raw();
        let mut buffer = self.buffer.borrow_buffer_mut();

        reshape_and_shape(
            &mut buffer,
            font_sys,
            Some(state.scroll_y),
            text_area_width,
            text_area_height,
        );

        // ── Auto-scroll: keep cursor in viewport ───────────────────────
        let cursor = self.buffer.cursor();
        let old_scroll_y = state.scroll_y;
        let metrics = font_metrics();
        if state.auto_scroll_enabled {
            let mut cursor_in_view = false;
            if let Some(run) = find_cursor_run(buffer.layout_runs(), cursor.line, cursor.column) {
                let cursor_top = run.line_top;
                let cursor_bottom = run.line_top + run.line_height;
                // Cursor above viewport → scroll up
                if cursor_top < 0.0 {
                    state.scroll_y = (state.scroll_y + cursor_top).max(0.0);
                }
                // Cursor below viewport → scroll down
                if cursor_bottom > text_area_height {
                    state.scroll_y =
                        (state.scroll_y + cursor_bottom - text_area_height).min(state.max_scroll_y);
                }
                cursor_in_view = true;
            }
            // Fallback: cursor is far outside the shaped range — estimate position
            if !cursor_in_view {
                let est_y = cursor.line as f32 * metrics.line_height;
                if est_y < state.scroll_y {
                    state.scroll_y = est_y;
                } else if est_y >= state.scroll_y + text_area_height {
                    state.scroll_y = (est_y - text_area_height + metrics.line_height).max(0.0);
                }
            }
            // Re-enable for subsequent frames so cursor tracking resumes
            // after this one-shot adjustment
            state.auto_scroll_enabled = true;
        }

        // ── Compute max scroll range ───────────────────────────────────
        // Count visual lines (including wrapped lines) using line_layout().
        // Previously used source line count which prevented scrolling to the
        // end of files with wrapped lines beyond the viewport.
        // line_layout() returns cached data for already-shaped lines at
        // near-zero cost. First frame after load/resize shapes all lines.
        // Cap each source line at MAX_VISUAL_LINES_PER_SOURCE visual lines
        // as a safety limit against pathological single lines (e.g.
        // no-whitespace megabyte).
        let total_height = compute_total_height(&mut buffer, font_sys, metrics);
        state.max_scroll_y = (total_height - text_area_height).max(0.0);
        state.scroll_y = state.scroll_y.clamp(0.0, state.max_scroll_y);

        // Re-apply scroll if auto-scroll changed it
        if (state.scroll_y - old_scroll_y).abs() > f32::EPSILON {
            buffer.set_scroll(Scroll {
                line: 0,
                vertical: state.scroll_y,
                horizontal: 0.0,
            });
            buffer.shape_until_scroll(font_sys, false);
        }

        // Clone the shaped buffer into an Arc and store for fill_raw
        // This keeps the buffer alive across frames during rendering
        let arc = Arc::new(buffer.clone());
        state.buffer_for_render = Some(arc);
        drop(buffer);
        drop(guard);

        layout::Node::new(bounds)
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_ref::<EditorWidgetState>();
        let bounds = layout.bounds();
        let gutter_width = state.gutter_width;

        let text_rect = text_area_rect(bounds, self.padding, gutter_width);
        let text_x = text_rect.x;
        let text_y = text_rect.y;
        let text_area_width = text_rect.width;
        let text_area_height = text_rect.height;

        // Use the buffer Arc that was prepared in layout()
        let buffer_for_draw = state.buffer_for_render.clone().unwrap_or_else(|| {
            // Fallback: create a fresh buffer if layout wasn't called
            with_font_system(|font_sys| {
                let mut buffer = self.buffer.borrow_buffer_mut();
                reshape_and_shape(
                    &mut buffer,
                    font_sys,
                    None,
                    text_area_width,
                    text_area_height,
                );
                Arc::new(buffer.clone())
            })
        });

        let text_geo = TextGeometry {
            clip: text_rect,
            x: text_x,
            y: text_y,
        };

        draw_background(renderer, bounds);
        draw_line_numbers(
            renderer,
            &buffer_for_draw,
            bounds,
            self.padding,
            text_y,
            gutter_width,
            text_area_height,
        );
        draw_find_match_highlights(
            renderer,
            &buffer_for_draw,
            &text_geo,
            self.matches.as_ref(),
            self.match_current_idx,
        );
        draw_bracket_match_highlights(renderer, &buffer_for_draw, &text_geo, self.bracket_pair);
        draw_selection(renderer, &buffer_for_draw, &text_geo, self.buffer);
        draw_text(renderer, &buffer_for_draw, &text_geo);
        draw_cursor(renderer, &buffer_for_draw, &text_geo, state, self.buffer);
    }

    #[allow(clippy::too_many_lines)]
    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        clipboard: &mut dyn iced::advanced::Clipboard,
        shell: &mut Shell<'_, EditorAction>,
        _viewport: &Rectangle,
    ) {
        let state = tree.state.downcast_mut::<EditorWidgetState>();

        match event {
            // ── Mouse wheel scrolling ───────────────────────────────
            Event::Mouse(iced::mouse::Event::WheelScrolled { delta }) => {
                if cursor.position_in(layout.bounds()).is_none() {
                    return;
                }
                let line_height = font_metrics().line_height;
                let pixel_delta = match delta {
                    ScrollDelta::Lines { y, .. } => y * line_height,
                    ScrollDelta::Pixels { y, .. } => *y,
                };
                state.scroll_y = (state.scroll_y - pixel_delta).clamp(0.0, state.max_scroll_y);
                state.auto_scroll_enabled = false;
                shell.invalidate_layout();
                shell.request_redraw();
            }

            // ── Mouse button press: move cursor to click point ───────
            Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Left)) => {
                // Re-shape the buffer with the current scroll BEFORE hit-test.
                // layout() runs after events, but WheelScrolled only calls
                // invalidate_layout() — the buffer still has the pre-scroll
                // shape when click arrives.  Without this reshape, hit_test
                // returns a stale line number, then auto-scroll in the next
                // layout() jumps the viewport to that wrong line.
                {
                    let bounds = layout.bounds();
                    let text_x = self.padding + state.gutter_width + 4.0;
                    let text_area_width = (bounds.width - text_x - self.padding).max(0.0);
                    let text_area_height = (bounds.height - self.padding * 2.0).max(0.0);

                    let scroll_y = state.scroll_y;
                    with_font_system(|font_sys| {
                        let mut buffer = self.buffer.borrow_buffer_mut();
                        reshape_and_shape(
                            &mut buffer,
                            font_sys,
                            Some(scroll_y),
                            text_area_width,
                            text_area_height,
                        );
                    });
                }

                if let Some((line, col)) = hit_test(
                    self.buffer,
                    layout,
                    cursor,
                    state.gutter_width,
                    self.padding,
                ) {
                    state.mouse_held = true;
                    // Reset blink so cursor becomes visible immediately after click.
                    state.last_blink = std::time::Instant::now();
                    state.auto_scroll_enabled = true;

                    let now = state.last_blink;
                    let is_double_click = match (state.last_click_time, state.last_click_pos) {
                        (Some(last_time), Some((last_line, last_col))) => {
                            now.duration_since(last_time).as_millis() < 500
                                && line == last_line
                                && col.abs_diff(last_col) <= 2
                        }
                        _ => false,
                    };

                    // Always update tracking for double-click detection.
                    state.last_click_time = Some(now);
                    state.last_click_pos = Some((line, col));

                    if is_double_click {
                        if state.modifiers.shift() {
                            // Shift+double-click: extend existing selection to include the
                            // word at the click position (word-boundary selection).
                            let text_buf = self.buffer.text();
                            let byte_offset = line_col_to_byte_offset(&text_buf, line, col);
                            let (word_start, word_end) = word_bounds_at(&text_buf, byte_offset);
                            if word_start != word_end {
                                let (start_line, start_col) =
                                    byte_offset_to_line_col(&text_buf, word_start);
                                let (end_line, end_col) =
                                    byte_offset_to_line_col(&text_buf, word_end);

                                // Determine which word boundary to extend to based on
                                // the anchor position relative to the word.
                                let cur = self.buffer.cursor();
                                let anchor_byte = cur.selection.as_ref().map_or_else(
                                    || line_col_to_byte_offset(&text_buf, cur.line, cur.column),
                                    |a| line_col_to_byte_offset(&text_buf, a.line, a.column),
                                );

                                if anchor_byte < word_start {
                                    // Anchor is before the word — extend to word
                                    // end to include the full word.
                                    shell.publish(EditorAction::SelectTo {
                                        line: end_line,
                                        col: end_col,
                                    });
                                } else if anchor_byte >= word_end {
                                    // Anchor is after the word — extend to word
                                    // start to include the full word.
                                    shell.publish(EditorAction::SelectTo {
                                        line: start_line,
                                        col: start_col,
                                    });
                                } else {
                                    // Anchor is inside the word — select the full
                                    // word by first resetting the cursor to the
                                    // word start, then extending to word end.
                                    //
                                    // NOTE: This relies on Iced draining queued
                                    // shell.publish() messages in order between
                                    // platform event dispatches (same assumption
                                    // documented in ButtonReleased below at the
                                    // zero-width-selection fix).
                                    shell.publish(EditorAction::MoveTo {
                                        line: start_line,
                                        col: start_col,
                                    });
                                    shell.publish(EditorAction::SelectTo {
                                        line: end_line,
                                        col: end_col,
                                    });
                                }
                            } else {
                                // Zero-width word (whitespace) — fall back to regular
                                // shift+click behaviour.
                                shell.publish(EditorAction::SelectTo { line, col });
                            }
                        } else {
                            shell.publish(EditorAction::SelectWordAt { line, col });
                        }
                        // Clear mouse_held so intermediate CursorMoved events
                        // don't trigger SelectTo and truncate the word/word-boundary
                        // selection.
                        state.mouse_held = false;
                    } else if state.modifiers.shift() {
                        shell.publish(EditorAction::SelectTo { line, col });
                    } else {
                        shell.publish(EditorAction::MoveTo { line, col });
                    }
                } else {
                    state.mouse_held = false;
                    // Gutter/padding click — clear double-click tracking.
                    state.last_click_time = None;
                    state.last_click_pos = None;
                }
                // Request redraw to keep the cursor blinking.
                shell.request_redraw();
            }

            // ── Mouse button release: end selection drag ────────────
            Event::Mouse(iced::mouse::Event::ButtonReleased(iced::mouse::Button::Left)) => {
                state.mouse_held = false;

                // Fix 1: Clear zero-width selection from sub-pixel mouse jitter.
                //
                // Between ButtonPressed (which publishes MoveTo, clearing selection) and
                // ButtonReleased, Iced dispatches CursorMoved events.  Even sub-pixel
                // mouse movement can fire CursorMoved → SelectTo, creating a spurious
                // zero-width selection (anchor == cursor).
                //
                // For real drag-selects, anchor != cursor so the selection is preserved.
                // For jitter (no actual movement), anchor == cursor so we detect and
                // clear it here, otherwise the blink loop (RedrawRequested) and draw()
                // would both suppress the cursor because has_selection is true.
                //
                // ASSUMPTION: Iced 0.14 drains shell.publish() messages between
                // platform event dispatches, so cursor() reflects MoveTo/SelectTo
                // actions already applied by ButtonPressed/CursorMoved.  If Iced ever
                // batches events, cursor() would be stale (MoveTo only) and this
                // check becomes a harmless no-op — Fix 2 (RedrawRequested guard
                // removal) takes over as sole recovery path.
                let cursor_state = self.buffer.cursor();
                if let Some(ref anchor) = cursor_state.selection {
                    if anchor.line == cursor_state.line && anchor.column == cursor_state.column {
                        shell.publish(EditorAction::MoveTo {
                            line: cursor_state.line,
                            col: cursor_state.column,
                        });
                    }
                }

                // Kickstart the blink cycle now that selection has ended.
                shell.request_redraw();
            }

            // ── Mouse move while button held: extend selection ──────
            Event::Mouse(iced::mouse::Event::CursorMoved { .. }) if state.mouse_held => {
                if let Some((line, col)) = hit_test(
                    self.buffer,
                    layout,
                    cursor,
                    state.gutter_width,
                    self.padding,
                ) {
                    shell.publish(EditorAction::SelectTo { line, col });
                    // Keep the redraw cycle alive while dragging to select.
                    shell.request_redraw();
                }
            }

            // ── Keyboard modifiers changed ─────────────────────────
            Event::Keyboard(keyboard::Event::ModifiersChanged(modifiers)) => {
                state.modifiers = *modifiers;
            }

            // ── Keyboard handling ───────────────────────────────────
            Event::Keyboard(keyboard::Event::KeyPressed {
                key: key_press,
                modifiers,
                physical_key,
                text,
                modified_key: _,
                ..
            }) => {
                // Skip keyboard processing when another UI element has
                // keyboard focus (tree panel, quick-open, etc.).
                if self.ignore_keyboard {
                    // Find/replace inputs own keyboard focus — block all editor
                    // keys (including arrows) so navigation does not move the
                    // code cursor while editing the search/replace fields.
                    // Tree panel and modal overlays block everything too.
                    return;
                }
                // Any keyboard cursor movement re-enables auto-scroll
                if is_cursor_movement_key(key_press) {
                    state.auto_scroll_enabled = true;
                    state.last_blink = std::time::Instant::now();
                }

                // ── Clipboard shortcuts (Cmd/Ctrl+C/X/V) ──────────────
                // On macOS, only Cmd (not Ctrl) triggers clipboard shortcuts;
                // Ctrl+C/X/V are terminal control characters.
                if super::detect_keyboard_mods(*modifiers).is_text_platform_mod() {
                    if let Some(latin) = key_press.to_latin(*physical_key) {
                        match latin {
                            'c' | 'x' => {
                                if let Some(text) = self.buffer.selection() {
                                    clipboard
                                        .write(iced::advanced::clipboard::Kind::Standard, text);
                                    if latin == 'x' {
                                        shell.publish(EditorAction::Delete);
                                    }
                                }
                                return;
                            }
                            'v' => {
                                if let Some(text) =
                                    clipboard.read(iced::advanced::clipboard::Kind::Standard)
                                {
                                    shell.publish(EditorAction::Paste(text));
                                    shell.invalidate_layout();
                                    shell.request_redraw();
                                }
                                return;
                            }
                            _ => {}
                        }
                    }
                }

                // ── Visual row navigation for Up/Down arrows ──────────
                // On wrapped lines, ArrowUp/Down should move one visual
                // row at a time, not jump over entire logical lines.
                // Plain arrows (no cmd/ctrl, no alt) use visual navigation;
                // Shift+Up/Down extends selection visually.
                {
                    let platform_mod =
                        super::detect_keyboard_mods(*modifiers).is_nav_platform_mod();
                    let alt = modifiers.alt();
                    let shift = modifiers.shift();
                    let is_arrow_up = matches!(key_press, key::Key::Named(key::Named::ArrowUp));
                    let is_arrow_down = matches!(key_press, key::Key::Named(key::Named::ArrowDown));

                    if (is_arrow_up || is_arrow_down) && !platform_mod && !alt {
                        // Shape the buffer with current scroll so layout runs
                        // reflect the viewport (same as mouse click handler).
                        let bounds = layout.bounds();
                        let text_x_offset = self.padding + state.gutter_width + 4.0;
                        let text_area_width =
                            (bounds.width - text_x_offset - self.padding).max(0.0);
                        let text_area_height = (bounds.height - self.padding * 2.0).max(0.0);

                        let scroll_y = state.scroll_y;
                        let result = with_font_system(|font_sys| {
                            let mut buffer = self.buffer.borrow_buffer_mut();
                            reshape_and_shape(
                                &mut buffer,
                                font_sys,
                                Some(scroll_y),
                                text_area_width,
                                text_area_height,
                            );

                            let cursor = self.buffer.cursor();
                            let metrics = font_metrics();

                            // Find the cursor's current visual row
                            let cursor_run =
                                find_cursor_run(buffer.layout_runs(), cursor.line, cursor.column);

                            if let Some(run) = cursor_run {
                                // Compute cursor x-offset within the text area
                                let cursor_x = run
                                    .glyphs
                                    .iter()
                                    .find(|g| {
                                        cursor.column
                                            < run.text[..g.end.min(run.text.len())].chars().count()
                                    })
                                    .map_or_else(
                                        || run.glyphs.last().map_or(0.0, |last| last.x + last.w),
                                        |g| g.x,
                                    );

                                // Target y is one line above/below the
                                // current visual row
                                let target_y = if is_arrow_up {
                                    run.line_top - 1.0
                                } else {
                                    run.line_top + run.line_height + 1.0
                                };

                                buffer.hit(cursor_x, target_y).map(|hit| {
                                    let line_text =
                                        buffer.lines.get(hit.line).map_or("", |l| l.text());
                                    let col =
                                        line_text[..hit.index.min(line_text.len())].chars().count();
                                    (hit.line, col)
                                })
                            } else {
                                // Cursor not in any shaped run — use
                                // estimated y position
                                #[allow(clippy::cast_precision_loss)]
                                let est_y =
                                    cursor.line as f32 * metrics.line_height - state.scroll_y;
                                let run_h = metrics.line_height;
                                let target_y = if is_arrow_up {
                                    est_y - 1.0
                                } else {
                                    est_y + run_h + 1.0
                                };
                                buffer.hit(0.0, target_y).map(|hit| {
                                    let line_text =
                                        buffer.lines.get(hit.line).map_or("", |l| l.text());
                                    let col =
                                        line_text[..hit.index.min(line_text.len())].chars().count();
                                    (hit.line, col)
                                })
                            }
                        }); // with_font_system drops guard

                        if let Some((target_line, target_col)) = result {
                            if shift {
                                shell.publish(EditorAction::SelectTo {
                                    line: target_line,
                                    col: target_col,
                                });
                            } else {
                                shell.publish(EditorAction::MoveTo {
                                    line: target_line,
                                    col: target_col,
                                });
                            }
                            // Invalidate layout so the auto-scroll logic in
                            // layout() runs and adjusts the scroll offset to
                            // bring the cursor into view.
                            shell.invalidate_layout();
                            shell.request_redraw();
                            return;
                        }
                        // If hit() returned None (target outside shaped area),
                        // fall through to map_key_to_action which does logical
                        // line movement — this triggers auto-scroll.
                    }
                }

                // ── Visual row navigation for Cmd+Left/Right ─────────
                // On wrapped lines, Cmd+Left/Right implement a two-press
                // behavior: first press goes to the visual row boundary,
                // second press goes to the logical line start/end.
                {
                    let platform_mod =
                        super::detect_keyboard_mods(*modifiers).is_nav_platform_mod();
                    let alt = modifiers.alt();
                    let shift = modifiers.shift();
                    let is_cmd_left = platform_mod
                        && !alt
                        && matches!(key_press, key::Key::Named(key::Named::ArrowLeft));
                    let is_cmd_right = platform_mod
                        && !alt
                        && matches!(key_press, key::Key::Named(key::Named::ArrowRight));

                    if is_cmd_left || is_cmd_right {
                        // Shape the buffer so layout runs reflect the viewport.
                        let bounds = layout.bounds();
                        let text_x_offset = self.padding + state.gutter_width + 4.0;
                        let text_area_width =
                            (bounds.width - text_x_offset - self.padding).max(0.0);
                        let text_area_height = (bounds.height - self.padding * 2.0).max(0.0);

                        let scroll_y = state.scroll_y;
                        let result = with_font_system(|font_sys| {
                            let mut buffer = self.buffer.borrow_buffer_mut();
                            reshape_and_shape(
                                &mut buffer,
                                font_sys,
                                Some(scroll_y),
                                text_area_width,
                                text_area_height,
                            );

                            let cursor = self.buffer.cursor();
                            let cursor_run =
                                find_cursor_run(buffer.layout_runs(), cursor.line, cursor.column);

                            cursor_run.map(|run| {
                                let first = run.glyphs.first();
                                let last = run.glyphs.last();
                                let line_text =
                                    buffer.lines.get(cursor.line).map_or("", |l| l.text());
                                let line_len = line_text.chars().count();

                                // Visual row boundary columns (character-based).
                                let visual_start: usize = first.map_or(0, |g| {
                                    line_text[..g.start.min(line_text.len())].chars().count()
                                });
                                let visual_end: usize = last.map_or(line_len, |g| {
                                    line_text[..g.end.min(line_text.len())].chars().count()
                                });

                                if is_cmd_left {
                                    // Two-press behavior via positional comparison:
                                    // - If cursor is at the visual row start, go to
                                    //   logical line start (column 0) — "second press".
                                    // - Otherwise, go to visual row start — "first press".
                                    if cursor.column == visual_start {
                                        (cursor.line, 0)
                                    } else {
                                        (cursor.line, visual_start)
                                    }
                                } else {
                                    // is_cmd_right
                                    // Two-press behavior via positional comparison:
                                    // - If cursor is at the visual row end, go to
                                    //   logical line end — "second press".
                                    // - Otherwise, go to visual row end — "first press".
                                    if cursor.column == visual_end {
                                        (cursor.line, line_len)
                                    } else {
                                        (cursor.line, visual_end)
                                    }
                                }
                            })
                        }); // with_font_system drops guard

                        if let Some((target_line, target_col)) = result {
                            if shift {
                                shell.publish(EditorAction::SelectTo {
                                    line: target_line,
                                    col: target_col,
                                });
                            } else {
                                shell.publish(EditorAction::MoveTo {
                                    line: target_line,
                                    col: target_col,
                                });
                            }
                            shell.invalidate_layout();
                            shell.request_redraw();
                            return;
                        }
                        // If find_cursor_run returned None (cursor outside
                        // shaped area), fall through to map_key_to_action which
                        // does logical line movement — this triggers auto-scroll.
                    }
                }

                // ── Text-first character input ──────────────────────────
                // Use the `text` field as primary source for character
                // insertion (handles dead keys, IME, AltGr, etc.).
                // Skip when an IME Commit was just processed to prevent
                // double-insertion on platforms that send both
                // InputMethod::Commit and KeyPressed.text for the same
                // composition.

                // When a platform modifier is held, this is a keyboard
                // shortcut — skip character insertion so the event reaches
                // both map_key_to_action() (SelectAll, ToggleLineComment,
                // DuplicateLine, DeleteLine, JumpToMatchingBracket) and
                // the subscription handler in editor.rs (Save, Undo, Find,
                // etc.). Without this guard, the shortcut letter leaks into
                // the text buffer while also firing the action.
                //
                // On macOS, only Cmd (not Ctrl) is the platform modifier;
                // Ctrl triggers emacs-style shortcuts (Ctrl+B/F/A/E etc.)
                // and terminal control characters.
                // On non-macOS, both Ctrl and Cmd are platform modifiers;
                // Ctrl+Alt (AltGr) is excluded because it produces text
                // characters for international layouts.
                if !super::detect_keyboard_mods(*modifiers).is_text_platform_mod() {
                    if let Some(committed) = text {
                        if !committed.is_empty() {
                            if let Some(ref suppress) = state.ime_commit_suppress {
                                if committed.as_ref() == suppress {
                                    state.ime_commit_suppress = None;
                                    return;
                                }
                                state.ime_commit_suppress = None;
                            }
                            let committed: &str = committed.as_ref();
                            if committed.chars().count() == 1 {
                                let c = committed.chars().next().unwrap();
                                if !c.is_control() {
                                    shell.publish(EditorAction::Insert(c));
                                    shell.invalidate_layout();
                                    shell.request_redraw();
                                    return;
                                }
                                // Control character (Backspace, Enter, Tab, Escape,
                                // and potentially Delete) — fall through to
                                // map_key_to_action which maps these Named keys to
                                // the corresponding EditorAction. These keys produce
                                // control characters via winit's Key::to_text()
                                // (e.g. Backspace → '\x08', Enter → '\r', Tab →
                                // '\t', Escape → '\x1b'). On some platforms Delete
                                // may produce '\x7f' (DEL) which is also a control
                                // character and needs the same fallthrough.
                            } else {
                                // Multi-codepoint grapheme clusters (emoji
                                // ZWJ sequences, flags, etc.) use Paste.
                                shell.publish(EditorAction::Paste(committed.to_string()));
                                shell.invalidate_layout();
                                shell.request_redraw();
                                return;
                            }
                        }
                    }
                }

                let action =
                    map_key_to_action(key_press, *modifiers, *physical_key, text.is_some());
                if let Some(ref action) = action {
                    // Cursor-movement actions re-enable auto-scroll (catches emacs
                    // shortcuts that aren't Named keys).
                    let is_cursor_move = action.is_cursor_movement();
                    if is_cursor_move {
                        state.auto_scroll_enabled = true;
                        state.last_blink = std::time::Instant::now();
                    }
                    shell.publish(action.clone());
                    // Invalidate layout after any action so the auto-scroll
                    // logic in layout() runs and adjusts the scroll offset to
                    // bring the cursor into view if it moved off-screen.
                    shell.invalidate_layout();
                    // Request redraw so the cursor blink state is refreshed
                    // after any keystroke (cursor moves and blink resets).
                    shell.request_redraw();
                }
            }

            // ── Input Method (IME) events ──────────────────────────
            Event::InputMethod(ime_event) => {
                if self.ignore_keyboard {
                    return;
                }
                match ime_event {
                    input_method::Event::Commit(committed) => {
                        if committed.is_empty() {
                            return;
                        }
                        state.ime_commit_suppress = Some(committed.clone());
                        if committed.chars().count() == 1 {
                            let c = committed.chars().next().unwrap();
                            if !c.is_control() {
                                shell.publish(EditorAction::Insert(c));
                            }
                        } else {
                            shell.publish(EditorAction::Paste(committed.clone()));
                        }
                        shell.invalidate_layout();
                        shell.request_redraw();
                    }
                    // Preedit rendering is deferred (no on-the-spot display).
                    input_method::Event::Preedit(_, _)
                    | input_method::Event::Opened
                    | input_method::Event::Closed => {}
                }
            }

            // ── Redraw requested — schedule next blink toggle ─────
            Event::Window(window::Event::RedrawRequested(_)) => {
                let now = std::time::Instant::now();
                let elapsed_ms =
                    u64::try_from(now.duration_since(state.last_blink).as_millis()).unwrap_or(0);
                let ms_into_cycle = elapsed_ms % 1000;
                // 500 ms visible / 500 ms hidden — toggle every 500 ms.
                let ms_until_toggle = if ms_into_cycle < 500 {
                    500 - ms_into_cycle
                } else {
                    1000 - ms_into_cycle
                };
                let next = now + Duration::from_millis(ms_until_toggle + 1);
                shell.request_redraw_at(window::RedrawRequest::At(next));
                // Also request an immediate redraw so draw() is called
                // in the current frame. Iced 0.14's reactive rendering
                // may skip draw() for undamaged widgets when only
                // request_redraw_at was used; request_redraw ensures
                // the cursor blink state is evaluated now.
                shell.request_redraw();
            }

            _ => {}
        }
    }

    fn mouse_interaction(
        &self,
        _tree: &Tree,
        _layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        mouse::Interaction::Text
    }
}

// ── Draw layer helpers ──────────────────────────────────────────────

/// Geometry parameters for text-drawing functions, derived from the
/// computed `text_rect` in [`Widget::draw`].
///
/// All five `draw_*` functions that render text content share these three
/// values, which are always computed together from a single `text_rect`.
struct TextGeometry {
    /// The clipping rectangle for text content (same as `text_rect`).
    clip: Rectangle,
    /// Absolute x-coordinate of the text area origin.
    x: f32,
    /// Absolute y-coordinate of the text area origin.
    y: f32,
}

/// Fill the widget background.
fn draw_background<Renderer>(renderer: &mut Renderer, bounds: Rectangle)
where
    Renderer: iced::advanced::Renderer,
{
    renderer.fill_quad(
        renderer::Quad {
            bounds,
            border: iced::Border::default(),
            ..renderer::Quad::default()
        },
        theme::BG_BASE,
    );
}

/// Draw line numbers in the gutter area.
fn draw_line_numbers<Renderer>(
    renderer: &mut Renderer,
    buffer: &cosmic_text::Buffer,
    bounds: Rectangle,
    padding: f32,
    text_y: f32,
    gutter_width: f32,
    text_area_height: f32,
) where
    Renderer: iced::advanced::text::Renderer,
{
    let number_color = theme::TEXT_MUTED;
    let number_clip = gutter_clip_rect(bounds, padding, gutter_width, text_area_height);

    let mut last_line_i = usize::MAX;
    for run in buffer.layout_runs() {
        // Only draw the first run per line (avoid duplicates for wrapped lines)
        if run.line_i == last_line_i {
            continue;
        }
        last_line_i = run.line_i;
        let num = run.line_i + 1;
        let num_str = num.to_string();
        let num_text = iced::advanced::text::Text {
            content: num_str,
            bounds: Size::new(gutter_width, run.line_height),
            size: iced::Pixels(GUTTER_FONT_SIZE),
            line_height: iced::advanced::text::LineHeight::Relative(1.3),
            font: renderer.default_font(),
            align_x: iced::alignment::Horizontal::Right.into(),
            align_y: iced::alignment::Vertical::Center,
            shaping: iced::advanced::text::Shaping::Advanced,
            wrapping: iced::advanced::text::Wrapping::None,
        };
        renderer.fill_text(
            num_text,
            Point::new(
                bounds.x + padding + gutter_width,
                text_y + run.line_top + run.line_height / 2.0,
            ),
            number_color,
            number_clip,
        );
    }
}

/// Draw find match highlight backgrounds behind text.
/// Rendered before selection so selection teal appears on top.
fn draw_find_match_highlights<Renderer>(
    renderer: &mut Renderer,
    buffer: &cosmic_text::Buffer,
    geo: &TextGeometry,
    matches: Option<&Vec<(usize, usize, usize)>>,
    match_current_idx: usize,
) where
    Renderer: iced::advanced::Renderer,
{
    // Match highlights are drawn BEFORE selection so selection
    // (ACCENT_DIM teal) renders on top of match highlights.
    // Text is drawn AFTER both via fill_raw, so highlights
    // appear as background tints behind the glyphs.
    if let Some(matches) = matches {
        for (i, &(match_line, col_start, col_end)) in matches.iter().enumerate() {
            let color = if i == match_current_idx {
                theme::FIND_MATCH_CURRENT
            } else {
                theme::FIND_MATCH_DIM
            };
            for run in buffer.layout_runs() {
                if run.line_i != match_line {
                    continue;
                }
                // Use cosmic_text::Cursor with byte-offset indices to
                // compute the pixel span of this match within the line.
                if let Some(hl) = run.highlight(
                    cosmic_text::Cursor {
                        line: match_line,
                        index: col_start,
                        ..cosmic_text::Cursor::default()
                    },
                    cosmic_text::Cursor {
                        line: match_line,
                        index: col_end,
                        ..cosmic_text::Cursor::default()
                    },
                ) {
                    draw_highlight_background(
                        renderer, geo.clip, geo.x, geo.y, &run, hl.0, hl.1, color,
                    );
                }
                // Match may span multiple visual runs on soft-wrapped
                // lines — don't break, continue checking all runs for
                // this logical line.
            }
        }
    }
}

/// Draw bracket matching highlight backgrounds behind the open/close bracket.
fn draw_bracket_match_highlights<Renderer>(
    renderer: &mut Renderer,
    buffer: &cosmic_text::Buffer,
    geo: &TextGeometry,
    bracket_pair: Option<((usize, usize), (usize, usize))>,
) where
    Renderer: iced::advanced::Renderer,
{
    // Draw a subtle background under both the opening and closing bracket.
    if let Some(((open_line, open_col), (close_line, close_col))) = bracket_pair {
        let bracket_color = theme::BRACKET_MATCH;
        for &(b_line, b_col) in &[(open_line, open_col), (close_line, close_col)] {
            let line_text = buffer.lines.get(b_line).map_or("", |l| l.text());
            let (byte_start, byte_end) = char_col_to_byte_range_in_line(line_text, b_col);
            for run in buffer.layout_runs() {
                if run.line_i != b_line {
                    continue;
                }
                // Highlight one character at the bracket position.
                if let Some(hl) = run.highlight(
                    cosmic_text::Cursor {
                        line: b_line,
                        index: byte_start,
                        ..cosmic_text::Cursor::default()
                    },
                    cosmic_text::Cursor {
                        line: b_line,
                        index: byte_end,
                        ..cosmic_text::Cursor::default()
                    },
                ) {
                    draw_highlight_background(
                        renderer,
                        geo.clip,
                        geo.x,
                        geo.y,
                        &run,
                        hl.0,
                        hl.1,
                        bracket_color,
                    );
                }
                break;
            }
        }
    }
}

/// Draw selection highlight rectangles.
fn draw_selection<Renderer>(
    renderer: &mut Renderer,
    buffer: &cosmic_text::Buffer,
    geo: &TextGeometry,
    editor_buffer: &EditorBuffer,
) where
    Renderer: iced::advanced::Renderer,
{
    let cursor_state = editor_buffer.cursor();

    if let Some(ref anchor) = cursor_state.selection {
        let start = (cursor_state.line, cursor_state.column);
        let end = (anchor.line, anchor.column);
        let (sel_start, sel_end) = if start < end {
            (start, end)
        } else {
            (end, start)
        };

        let sel_start_byte = buffer.lines.get(sel_start.0).map_or(0, |l| {
            char_col_to_byte_offset_in_line(l.text(), sel_start.1)
        });
        let sel_end_byte = buffer
            .lines
            .get(sel_end.0)
            .map_or(0, |l| char_col_to_byte_offset_in_line(l.text(), sel_end.1));

        for run in buffer.layout_runs() {
            if let Some(highlight) = run.highlight(
                cosmic_text::Cursor {
                    line: sel_start.0,
                    index: sel_start_byte,
                    ..cosmic_text::Cursor::default()
                },
                cosmic_text::Cursor {
                    line: sel_end.0,
                    index: sel_end_byte,
                    ..cosmic_text::Cursor::default()
                },
            ) {
                draw_highlight_background(
                    renderer,
                    geo.clip,
                    geo.x,
                    geo.y,
                    &run,
                    highlight.0,
                    highlight.1,
                    theme::ACCENT_DIM,
                );
            }
        }
    }
}

/// Draw the text glyphs via `fill_raw` for syntax-coloured output.
fn draw_text<Renderer>(
    renderer: &mut Renderer,
    buffer: &Arc<cosmic_text::Buffer>,
    geo: &TextGeometry,
) where
    Renderer: iced::advanced::graphics::text::Renderer,
{
    renderer.fill_raw(TextRaw {
        buffer: Arc::downgrade(buffer),
        position: Point::new(geo.x, geo.y),
        color: iced::Color::WHITE, // neutral multiplier preserves per-glyph colors
        clip_bounds: geo.clip,
    });
}

/// Draw the blinking cursor caret when no selection is active.
fn draw_cursor<Renderer>(
    renderer: &mut Renderer,
    buffer: &cosmic_text::Buffer,
    geo: &TextGeometry,
    state: &EditorWidgetState,
    editor_buffer: &EditorBuffer,
) where
    Renderer: iced::advanced::Renderer,
{
    let now = std::time::Instant::now();
    let blink_on = now.duration_since(state.last_blink).as_millis() % 1000 < 500;
    let cursor_state = editor_buffer.cursor();
    let has_selection = cursor_state.selection.is_some();

    if blink_on && !has_selection {
        let cursor_x;
        let cursor_y;
        let cursor_height;

        if let Some(run) =
            find_cursor_run(buffer.layout_runs(), cursor_state.line, cursor_state.column)
        {
            cursor_y = geo.y + run.line_top;
            cursor_height = run.line_height;
            let found_x = run
                .glyphs
                .iter()
                .find(|g| {
                    cursor_state.column < run.text[..g.end.min(run.text.len())].chars().count()
                })
                .map(|g| g.x);
            cursor_x = geo.x
                + found_x.unwrap_or_else(|| run.glyphs.last().map_or(0.0, |last| last.x + last.w));
        } else {
            cursor_x = 0.0;
            cursor_y = geo.y;
            cursor_height = font_metrics().line_height;
        }

        let cursor_rect = Rectangle {
            x: cursor_x,
            y: cursor_y,
            width: 1.5,
            height: cursor_height,
        };

        if let Some(clipped) = geo.clip.intersection(&cursor_rect) {
            renderer.fill_quad(
                renderer::Quad {
                    bounds: clipped,
                    border: iced::Border::default(),
                    ..renderer::Quad::default()
                },
                theme::TEXT_PRIMARY,
            );
        }
    }
}

// ── Keybinding mapping ──────────────────────────────────────────────

/// Returns `true` if the key is a cursor-movement key (arrow keys,
/// Home, End, PageUp, PageDown).
const fn is_cursor_movement_key(key: &key::Key) -> bool {
    matches!(
        key,
        key::Key::Named(
            key::Named::ArrowLeft
                | key::Named::ArrowRight
                | key::Named::ArrowUp
                | key::Named::ArrowDown
                | key::Named::Home
                | key::Named::End
                | key::Named::PageUp
                | key::Named::PageDown
        )
    )
}

/// Map a keyboard key + modifiers to an [`EditorAction`].
///
/// `has_text` is `true` when the `KeyPressed` event carried a non-empty
/// `text` field (dead key / IME / AltGr composition). Only used for
/// distinguishing AltGr from shortcuts on non-macOS — character insertion
/// itself is handled in `on_event`.
#[allow(clippy::too_many_lines)]
fn map_key_to_action(
    key: &key::Key,
    modifiers: keyboard::Modifiers,
    physical_key: key::Physical,
    _has_text: bool,
) -> Option<EditorAction> {
    // On macOS, platform shortcuts use Cmd only, not Ctrl.
    // Ctrl is reserved for emacs-style shortcuts (Ctrl+F/B/A/E/etc.)
    // and terminal conventions. On other platforms, Ctrl triggers
    // platform shortcuts alongside the Windows/Super key.
    let platform_mod = super::detect_keyboard_mods(modifiers).is_nav_platform_mod();
    let shift = modifiers.shift();
    let alt = modifiers.alt();

    // On non-macOS, AltGr (Ctrl+Alt) produces text — treat as character
    // input, not a shortcut. When `has_text` is true, AltGr is active.
    #[cfg(not(target_os = "macos"))]
    let altgr_active = alt && modifiers.control() && _has_text;
    #[cfg(target_os = "macos")]
    let altgr_active = false;

    let mv = |dir, sel| {
        Some(EditorAction::Move {
            direction: dir,
            select: sel,
        })
    };

    match key {
        key::Key::Named(named) => {
            match (named, platform_mod, shift, alt) {
                // Platform mod + Up/Home → document start
                (key::Named::ArrowUp | key::Named::Home, true, s, false) => {
                    mv(CursorMove::DocStart, s)
                }
                // Platform mod + Down/End → document end
                (key::Named::ArrowDown | key::Named::End, true, s, false) => {
                    mv(CursorMove::DocEnd, s)
                }
                (key::Named::ArrowLeft, false, s, false) => mv(CursorMove::Left, s),
                (key::Named::ArrowRight, false, s, false) => mv(CursorMove::Right, s),
                (key::Named::ArrowUp, false, s, false) => mv(CursorMove::Up, s),
                (key::Named::ArrowDown, false, s, false) => mv(CursorMove::Down, s),
                (key::Named::Home, false, s, false) | (key::Named::ArrowLeft, true, s, false) => {
                    mv(CursorMove::Home, s)
                }
                (key::Named::End, false, s, false) | (key::Named::ArrowRight, true, s, false) => {
                    mv(CursorMove::End, s)
                }

                // Page keys
                (key::Named::PageUp, false, s, false) => mv(CursorMove::PageUp, s),
                (key::Named::PageDown, false, s, false) => mv(CursorMove::PageDown, s),

                // Alt+Left/Right → word boundary navigation
                (key::Named::ArrowLeft, false, s, true) => mv(CursorMove::WordLeft, s),
                (key::Named::ArrowRight, false, s, true) => mv(CursorMove::WordRight, s),

                // Alt+Up/Down → move line up/down (no selection variant — always moves)
                (key::Named::ArrowUp, false, false, true) => Some(EditorAction::MoveLineUp),
                (key::Named::ArrowDown, false, false, true) => Some(EditorAction::MoveLineDown),

                // Delete/Backspace
                (key::Named::Backspace, false, false, false) => Some(EditorAction::Backspace),
                (key::Named::Delete, false, false, false) => Some(EditorAction::Delete),
                // Platform mod + Backspace/Delete → delete word
                (key::Named::Backspace, true, false, false) => Some(EditorAction::DeleteWordBack),
                (key::Named::Delete, true, false, false) => Some(EditorAction::DeleteWordForward),

                // Enter
                (key::Named::Enter, false, false, false) => Some(EditorAction::Enter),

                // Space — defensive: some platforms may deliver space as Named
                (key::Named::Space, false, false, false) => Some(EditorAction::Insert(' ')),

                // Tab / Shift+Tab
                // NOTE: On macOS, Ctrl+Tab and Ctrl+Shift+Tab should NOT trigger
                // Indent/Unindent since the editor.rs subscription uses Ctrl+Tab
                // for tab switching and Ctrl+Shift+Tab for reverse tab switching.
                // The control() guard prevents that conflict.
                (key::Named::Tab, false, false, false) if !modifiers.control() => {
                    Some(EditorAction::Indent)
                }
                (key::Named::Tab, false, true, false) if !modifiers.control() => {
                    Some(EditorAction::Unindent)
                }

                _ => None,
            }
        }
        key::Key::Unidentified => None,
        key::Key::Character(ch) => {
            // All shortcut matching uses to_latin() to work with any
            // keyboard layout (Cyrillic, Greek, etc.).
            let latin = key.to_latin(physical_key);

            // Mac-specific emacs shortcuts (Ctrl, not Cmd)
            #[cfg(target_os = "macos")]
            {
                let ctrl = modifiers.control();
                if ctrl && !modifiers.command() {
                    match latin {
                        Some('f') => {
                            return Some(EditorAction::Move {
                                direction: CursorMove::Right,
                                select: false,
                            });
                        }
                        Some('b') => {
                            return Some(EditorAction::Move {
                                direction: CursorMove::Left,
                                select: false,
                            });
                        }
                        Some('a') => {
                            return Some(EditorAction::Move {
                                direction: CursorMove::Home,
                                select: false,
                            });
                        }
                        Some('e') => {
                            return Some(EditorAction::Move {
                                direction: CursorMove::End,
                                select: false,
                            });
                        }
                        Some('h') => return Some(EditorAction::Backspace),
                        Some('d') => return Some(EditorAction::Delete),
                        Some('n') => {
                            return Some(EditorAction::Move {
                                direction: CursorMove::Down,
                                select: false,
                            });
                        }
                        Some('p') => {
                            return Some(EditorAction::Move {
                                direction: CursorMove::Up,
                                select: false,
                            });
                        }
                        _ => {}
                    }
                }
            }

            // SelectAll via platform mod + A
            // Guard: !altgr_active prevents AltGr+A (on non-macOS) from
            // triggering SelectAll.
            if !altgr_active && latin == Some('a') && platform_mod && !shift {
                return Some(EditorAction::SelectAll);
            }

            // Toggle line comment via platform mod + /
            if !altgr_active && latin == Some('/') && platform_mod && !shift {
                return Some(EditorAction::ToggleLineComment);
            }

            // Jump to matching bracket via platform mod + Shift + \
            if !altgr_active && latin == Some('\\') && platform_mod && shift {
                return Some(EditorAction::JumpToMatchingBracket);
            }

            // Delete line via platform mod + Shift + K
            if !altgr_active && latin == Some('k') && platform_mod && shift {
                return Some(EditorAction::DeleteLine);
            }

            // Duplicate line via platform mod + Shift + D
            if !altgr_active && latin == Some('d') && platform_mod && shift {
                return Some(EditorAction::DuplicateLine);
            }

            // Fallback: character input when no non-Shift modifiers are held.
            // Use the actual character from the key event rather than
            // key.to_latin() which maps non-Latin characters (Cyrillic,
            // Greek, CJK) to their physical-key Latin equivalents.
            // Note: `!modifiers.control()` guards against macOS Ctrl-only
            // combos falling through to character insertion after the emacs
            // shortcut block above handled the ones we recognize. On other
            // platforms, platform_mod already includes control() so this is
            // redundant but harmless.
            if !platform_mod && !modifiers.control() {
                if let Some(c) = ch.chars().next() {
                    if !c.is_control() {
                        return Some(EditorAction::Insert(c));
                    }
                }
            }

            None
        }
    }
}

#[cfg(test)]
#[path = "editor_widget_tests.rs"]
mod tests;
