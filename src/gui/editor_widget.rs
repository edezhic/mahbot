//! A [`cosmic_text::Buffer`]-backed text buffer with cursor and selection
//! management. Intended as a drop-in replacement for `text_editor::Content`
//! in the editor.rs codebase.

use std::cell::{Cell, RefCell};

use cosmic_text::Scroll;
use iced::advanced::graphics::text::cosmic_text;
use iced::advanced::input_method;
use iced::mouse::ScrollDelta;

use super::highlight::{self, FileHighlights, HighlightLanguage};
use crate::util::UnwrapPoison;

// ── Constants ───────────────────────────────────────────────────────

/// Font metrics used for the editor buffer.
#[must_use]
pub fn font_metrics() -> cosmic_text::Metrics {
    cosmic_text::Metrics::relative(14.0, 1.3)
}

/// Maximum file size in bytes for which to apply syntax highlighting via
/// tree-sitter. Files larger than this are not parsed for highlighting to
/// avoid blocking the UI thread during parsing.
///
/// Both the editor widget and the diff viewer enforce this limit, sharing
/// the same value to prevent accidental drift.
pub(crate) const MAX_HIGHLIGHT_SIZE: usize = 2 * 1024 * 1024; // 2 MB

/// Font size for line numbers in the editor gutter.
/// Matches the diff page styling (JetBrains Mono 11px).
pub(crate) const GUTTER_FONT_SIZE: f32 = 11.0;

/// Maximum visual lines per source line as a safety limit against
/// pathological single lines (e.g. no-whitespace megabyte).
pub(crate) const MAX_VISUAL_LINES_PER_SOURCE: usize = 10_000;

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
    /// Move cursor one character left.
    MoveLeft,
    /// Move cursor one character right.
    MoveRight,
    /// Move cursor one line up.
    MoveUp,
    /// Move cursor one line down.
    MoveDown,
    /// Move cursor to start of current line.
    MoveHome,
    /// Move cursor to end of current line.
    MoveEnd,
    /// Move cursor one word left.
    MoveWordLeft,
    /// Move cursor one word right.
    MoveWordRight,
    /// Move cursor to start of document.
    MoveDocStart,
    /// Move cursor to end of document.
    MoveDocEnd,
    /// Move cursor one page up.
    MovePageUp,
    /// Move cursor one page down.
    MovePageDown,
    /// Extend selection one character left.
    SelectLeft,
    /// Extend selection one character right.
    SelectRight,
    /// Extend selection one line up.
    SelectUp,
    /// Extend selection one line down.
    SelectDown,
    /// Extend selection to start of current line.
    SelectHome,
    /// Extend selection to end of current line.
    SelectEnd,
    /// Extend selection one word left.
    SelectWordLeft,
    /// Extend selection one word right.
    SelectWordRight,
    /// Extend selection to start of document.
    SelectDocStart,
    /// Extend selection to end of document.
    SelectDocEnd,
    /// Extend selection one page up.
    SelectPageUp,
    /// Extend selection one page down.
    SelectPageDown,
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

// ── EditorBuffer ────────────────────────────────────────────────────

/// A text buffer backed by [`cosmic_text::Buffer`] with manual cursor and
/// selection tracking. All mutating methods take `&self` using interior
/// mutability (`Cell` / `RefCell`) so the buffer can be used from Iced's
/// `view()` without a `&mut` reference.
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
    pub fn new() -> Self {
        let mut guard = iced::advanced::graphics::text::font_system()
            .write()
            .unwrap_poison();
        let font_sys = guard.raw();
        let mut buffer = cosmic_text::Buffer::new(font_sys, font_metrics());
        buffer.set_text(
            font_sys,
            "",
            &cosmic_text::Attrs::new().family(cosmic_text::Family::Name("JetBrains Mono")),
            cosmic_text::Shaping::Advanced,
            None,
        );
        drop(guard);
        Self {
            buffer: RefCell::new(buffer),
            cursor_line: Cell::new(0),
            cursor_col: Cell::new(0),
            sel_line: Cell::new(0),
            sel_col: Cell::new(0),
            has_selection: Cell::new(false),
            language: None,
            file_extension: RefCell::new(None),
        }
    }

    /// Create a buffer pre-populated with the given text.
    ///
    /// When `language` is `Some` and the text is within
    /// `MAX_HIGHLIGHT_SIZE`, syntax highlighting is applied via tree-sitter.
    /// Otherwise text is rendered with default attributes.
    pub fn with_text(text: &str, language: Option<HighlightLanguage>) -> Self {
        let mut guard = iced::advanced::graphics::text::font_system()
            .write()
            .unwrap_poison();
        let font_sys = guard.raw();
        let mut buffer = cosmic_text::Buffer::new(font_sys, font_metrics());
        Self::set_buffer_text_highlighted(&mut buffer, font_sys, text, language);
        drop(guard);
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

    // ── Text access ───────────────────────────────────────────────

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
    pub fn language(&self) -> Option<HighlightLanguage> {
        self.language
    }

    // ── Cursor ────────────────────────────────────────────────────

    /// Return the current cursor state, including selection anchor if any.
    pub fn cursor(&self) -> CursorState {
        let selection = if self.has_selection.get() {
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
                self.cursor_line.set(line);
                self.cursor_col.set(col);
                self.has_selection.set(true);
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
                }
            }
            EditorAction::Indent => self.do_indent(),
            EditorAction::Unindent => self.do_unindent(),
            EditorAction::MoveLeft => self.do_move_left(false),
            EditorAction::MoveRight => self.do_move_right(false),
            EditorAction::MoveUp => self.do_move_up(false),
            EditorAction::MoveDown => self.do_move_down(false),
            EditorAction::MoveHome => self.do_move_home(false),
            EditorAction::MoveEnd => self.do_move_end(false),
            EditorAction::MoveWordLeft => self.do_move_word_left(false),
            EditorAction::MoveWordRight => self.do_move_word_right(false),
            EditorAction::MoveDocStart => self.do_move_doc_start(false),
            EditorAction::MoveDocEnd => self.do_move_doc_end(false),
            EditorAction::MovePageUp => self.do_move_page_up(false),
            EditorAction::MovePageDown => self.do_move_page_down(false),
            EditorAction::SelectLeft => self.do_move_left(true),
            EditorAction::SelectRight => self.do_move_right(true),
            EditorAction::SelectUp => self.do_move_up(true),
            EditorAction::SelectDown => self.do_move_down(true),
            EditorAction::SelectHome => self.do_move_home(true),
            EditorAction::SelectEnd => self.do_move_end(true),
            EditorAction::SelectWordLeft => self.do_move_word_left(true),
            EditorAction::SelectWordRight => self.do_move_word_right(true),
            EditorAction::SelectDocStart => self.do_move_doc_start(true),
            EditorAction::SelectDocEnd => self.do_move_doc_end(true),
            EditorAction::SelectPageUp => self.do_move_page_up(true),
            EditorAction::SelectPageDown => self.do_move_page_down(true),
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
        let mut guard = iced::advanced::graphics::text::font_system()
            .write()
            .unwrap_poison();
        let font_sys = guard.raw();
        let mut buffer = self.buffer.borrow_mut();
        Self::set_buffer_text_highlighted(&mut buffer, font_sys, new_text, self.language);
        drop(buffer);
        drop(guard);
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
    fn selection_range(&self) -> (usize, usize, usize, usize) {
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

        let mut guard = iced::advanced::graphics::text::font_system()
            .write()
            .unwrap_poison();
        let font_sys = guard.raw();
        let mut buffer = self.buffer.borrow_mut();
        Self::set_buffer_text_highlighted(&mut buffer, font_sys, &new_text, self.language);
        drop(buffer);
        drop(guard);

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
        // If selection exists, delete it and we're done.
        if let Some((start, end)) = self.delete_selection_get_range() {
            self.edit_text(|text| {
                let mut new_text = text.to_string();
                new_text.replace_range(start..end, "");
                let (line, col) = byte_offset_to_line_col(&new_text, start);
                (new_text, Some((line, col)))
            });
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
        // If selection exists, delete it.
        if let Some((start, end)) = self.delete_selection_get_range() {
            self.edit_text(|text| {
                let mut new_text = text.to_string();
                new_text.replace_range(start..end, "");
                let (line, col) = byte_offset_to_line_col(&new_text, start);
                (new_text, Some((line, col)))
            });
            return;
        }

        let (cl, cc) = (self.cursor_line.get(), self.cursor_col.get());
        self.edit_text(|text| {
            let offset = line_col_to_byte_offset(text, cl, cc);
            if offset >= text.len() {
                return (text.to_string(), None);
            }
            let next_boundary = if text.is_char_boundary(offset + 1) {
                offset + 1
            } else {
                text.floor_char_boundary(offset + 1)
            };
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
            let (sl, _sc, el, _ec) = self.selection_range();
            let line_count = self.line_count();
            if sl >= line_count {
                return;
            }
            let el = el.min(line_count.saturating_sub(1));

            self.edit_text(|text| {
                let mut new_text = text.to_string();
                // Insert tabs at the start of each line in [sl, el].
                for line_idx in sl..=el {
                    let offset = line_col_to_byte_offset(&new_text, line_idx, 0);
                    new_text.insert(offset, '\t');
                }

                // Cursor: move to start of the first indented line (column 0)
                // and stretch the selection to cover the same lines.
                (new_text, Some((sl, 0)))
            });
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
            let (sl, _sc, el, _ec) = self.selection_range();
            let line_count = self.line_count();
            if sl >= line_count {
                return;
            }
            let el = el.min(line_count.saturating_sub(1));

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
                    let remove_count = 1.min(leading_count);
                    let remove_bytes = line_text[..remove_count].len();
                    let line_start = line_col_to_byte_offset(&new_text, line_idx, 0);
                    new_text.replace_range(line_start..line_start + remove_bytes, "");
                }

                (new_text, Some((sl, 0)))
            });
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
            self.ensure_selection_anchor();
        }
        if let Some((line, col)) = compute() {
            if extend_selection {
                self.set_cursor_pos(line, col);
            } else {
                self.move_to(line, col);
            }
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

    fn ensure_selection_anchor(&self) {
        if !self.has_selection.get() {
            self.sel_line.set(self.cursor_line.get());
            self.sel_col.set(self.cursor_col.get());
            self.has_selection.set(true);
        }
    }

    // ── Delete word helpers ───────────────────────────────────────

    /// Delete from cursor backward to start of previous word.
    fn do_delete_word_back(&self) {
        // If selection exists, delete it.
        if let Some((start, end)) = self.delete_selection_get_range() {
            self.edit_text(|text| {
                let mut new_text = text.to_string();
                new_text.replace_range(start..end, "");
                let (line, col) = byte_offset_to_line_col(&new_text, start);
                (new_text, Some((line, col)))
            });
            return;
        }

        let text = self.text();
        let offset = line_col_to_byte_offset(&text, self.cursor_line.get(), self.cursor_col.get());
        if offset == 0 {
            return;
        }
        let word_start = find_word_start(&text, offset);
        self.edit_text(|_| {
            let mut new_text = text.clone();
            new_text.replace_range(word_start..offset, "");
            let (line, col) = byte_offset_to_line_col(&new_text, word_start);
            (new_text, Some((line, col)))
        });
    }

    /// Delete from cursor forward to start of next word.
    fn do_delete_word_forward(&self) {
        // If selection exists, delete it.
        if let Some((start, end)) = self.delete_selection_get_range() {
            self.edit_text(|text| {
                let mut new_text = text.to_string();
                new_text.replace_range(start..end, "");
                let (line, col) = byte_offset_to_line_col(&new_text, start);
                (new_text, Some((line, col)))
            });
            return;
        }

        let text = self.text();
        let offset = line_col_to_byte_offset(&text, self.cursor_line.get(), self.cursor_col.get());
        if offset >= text.len() {
            return;
        }
        let word_end = find_word_end(&text, offset);
        self.edit_text(|_| {
            let mut new_text = text.clone();
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

        // Determine which lines to operate on.
        let (start_line, end_line) = if self.has_selection.get() {
            let (sl, _sc, el, _ec) = self.selection_range();
            (sl, el)
        } else {
            let line = self.cursor_line.get();
            (line, line)
        };

        let line_count = self.line_count();
        if start_line >= line_count {
            return;
        }
        let end_line = end_line.min(line_count.saturating_sub(1));

        let text = self.text();
        // Normalise \r\n → \n so str::lines() doesn't preserve stray \r chars.
        let text = text.replace("\r\n", "\n");
        let lines: Vec<&str> = text.lines().collect();

        // Build new lines with toggled comments.
        let mut new_lines: Vec<String> = Vec::with_capacity(lines.len());
        let mut first_toggled_col = None;

        for (i, line_text) in lines.iter().enumerate() {
            if i >= start_line && i <= end_line {
                let trimmed = line_text.trim_start();
                let leading_ws_len = line_text.len() - trimmed.len();
                let leading_ws = &line_text[..leading_ws_len];

                if let Some(stripped) = trimmed.strip_prefix(prefix) {
                    // Commented → uncomment. Preserve one space after prefix if present.
                    let after_comment = stripped.strip_prefix(' ').unwrap_or(stripped);
                    new_lines.push(format!("{leading_ws}{after_comment}"));
                    if i == start_line {
                        first_toggled_col = Some(leading_ws_len);
                    }
                } else {
                    // Not commented → add comment prefix.
                    new_lines.push(format!("{leading_ws}{prefix} {trimmed}"));
                    if i == start_line {
                        first_toggled_col = Some(leading_ws_len + prefix.len() + 1);
                    }
                }
            } else {
                new_lines.push(line_text.to_string());
            }
        }

        let new_text = new_lines.join("\n");
        let target_line = start_line.min(new_lines.len().saturating_sub(1));
        let target_col = first_toggled_col.unwrap_or(0);

        self.edit_text(|_| (new_text, Some((target_line, target_col))));
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
            // Check if cursor is at or just after the opening bracket.
            let at_open = cl == ol && (cc == oc || cc == oc + 1 || (cc > 0 && cc - 1 == oc));
            // Check if cursor is at the closing bracket position.
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

        let (start_line, end_line) = if self.has_selection.get() {
            let (sl, _sc, el, _ec) = self.selection_range();
            (sl, el)
        } else {
            let line = self.cursor_line.get();
            (line, line)
        };

        let start_line = start_line.min(line_count.saturating_sub(1));
        let end_line = end_line.min(line_count.saturating_sub(1));

        if start_line > end_line {
            return;
        }

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

        let (start_line, end_line) = if self.has_selection.get() {
            let (sl, _sc, el, _ec) = self.selection_range();
            (sl, el)
        } else {
            let line = self.cursor_line.get();
            (line, line)
        };

        let start_line = start_line.min(line_count.saturating_sub(1));
        let end_line = end_line.min(line_count.saturating_sub(1));

        if start_line > end_line {
            return;
        }

        self.edit_text(|text| {
            let mut new_text = text.to_string();
            let line_ending = if text.contains("\r\n") { "\r\n" } else { "\n" };

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

        let (start_line, end_line) = if self.has_selection.get() {
            let (sl, _sc, el, _ec) = self.selection_range();
            (sl, el)
        } else {
            let line = self.cursor_line.get();
            (line, line)
        };

        if start_line == 0 {
            return; // Already at top.
        }

        let end_line = end_line.min(line_count.saturating_sub(1));
        let swap_line = start_line.saturating_sub(1);

        self.edit_text(|text| {
            let text = text.replace("\r\n", "\n");
            let lines: Vec<&str> = text.lines().collect();

            // Swap the block of lines [swap_line..=end_line] so that
            // the block [start_line..=end_line] moves up, and line
            // swap_line moves down.
            let mut new_lines = lines.clone();
            // Extract the line(s) to move up.
            let block: Vec<&str> = lines[start_line..=end_line].to_vec();
            // Remove the block from its current position.
            new_lines.splice(start_line..=end_line, std::iter::empty());
            // Insert the block before swap_line (which is now at the
            // same position since we removed the block above it).
            new_lines.splice(swap_line..swap_line, block);

            let new_text = new_lines.join("\n");

            // Cursor goes to the moved block's start.
            let target_line = swap_line;
            (new_text, Some((target_line, 0)))
        });
    }

    /// Move the current line (or selected lines) down by one.
    /// At the last line boundary, this is a no-op.
    fn do_move_line_down(&self) {
        let line_count = self.line_count();
        if line_count <= 1 {
            return;
        }

        let (start_line, end_line) = if self.has_selection.get() {
            let (sl, _sc, el, _ec) = self.selection_range();
            (sl, el)
        } else {
            let line = self.cursor_line.get();
            (line, line)
        };

        if end_line + 1 >= line_count {
            return; // Already at bottom.
        }

        let end_line = end_line.min(line_count.saturating_sub(1));
        let swap_line = end_line + 1;

        self.edit_text(|text| {
            let text = text.replace("\r\n", "\n");
            let lines: Vec<&str> = text.lines().collect();

            // Swap the block of lines [start_line..=swap_line] so that
            // line swap_line moves up, and the block [start_line..=end_line]
            // moves down.
            let mut new_lines = lines.clone();
            // Extract the line below.
            let below_line = lines[swap_line];
            // Remove the line below.
            new_lines.remove(swap_line);
            // Insert it before the block (at start_line).
            new_lines.insert(start_line, below_line);

            let new_text = new_lines.join("\n");

            // Cursor goes to the moved block's start.
            let target_line = start_line + 1;
            (new_text, Some((target_line, 0)))
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
            "yaml" | "yml" => Some("#"),
            "dockerfile" | "makefile" | "mak" | "cmake" => Some("#"),
            _ => None,
        };
    }

    None
}

// ── Helper functions ────────────────────────────────────────────────

/// Compute the total height of a cosmic_text buffer, accounting for wrapped
/// lines. Each source line is capped at [`MAX_VISUAL_LINES_PER_SOURCE`]
/// visual lines as a safety limit against pathological single lines
/// (e.g. no-whitespace megabyte lines).
#[allow(clippy::cast_precision_loss)]
pub(crate) fn compute_total_height(
    buffer: &mut cosmic_text::Buffer,
    font_sys: &mut cosmic_text::FontSystem,
    metrics: cosmic_text::Metrics,
) -> f32 {
    let mut total_visual_lines: f32 = 0.0;
    for i in 0..buffer.lines.len() {
        let visual_count = buffer
            .line_layout(font_sys, i)
            .map_or(1, |ll| ll.len().min(MAX_VISUAL_LINES_PER_SOURCE));
        total_visual_lines += visual_count as f32;
    }
    total_visual_lines * metrics.line_height
}

/// Compute the text area rectangle (position and size) inside the given
/// `bounds`, accounting for `padding` and `gutter_width`.
///
/// The returned rectangle has:
/// - `x`: `bounds.x + padding + gutter_width + 4px` gap
/// - `y`: `bounds.y + padding`
/// - `width`: remainder of `bounds.width` after gutter, gap, and padding
/// - `height`: `bounds.height` minus `padding` on both sides
pub(crate) fn text_area_rect(
    bounds: iced::Rectangle,
    padding: f32,
    gutter_width: f32,
) -> iced::Rectangle {
    let x = bounds.x + padding + gutter_width + 4.0; // 4px gap
    let y = bounds.y + padding;
    let width = (bounds.width - (x - bounds.x) - padding).max(0.0);
    let height = (bounds.height - padding * 2.0).max(0.0);
    iced::Rectangle {
        x,
        y,
        width,
        height,
    }
}

/// Compute the gutter clip rectangle for line numbers.
pub(crate) fn gutter_clip_rect(
    bounds: iced::Rectangle,
    padding: f32,
    gutter_width: f32,
    text_area_height: f32,
) -> iced::Rectangle {
    iced::Rectangle {
        x: bounds.x + padding,
        y: bounds.y + padding,
        width: gutter_width,
        height: text_area_height,
    }
}

/// Convert an [`iced::Color`] (f32 RGBA components, 0.0–1.0) to
/// [`cosmic_text::Color`] (u8 RGB).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn iced_color_to_cosmic(c: iced::Color) -> cosmic_text::Color {
    let r = (c.r * 255.0).round() as u8;
    let g = (c.g * 255.0).round() as u8;
    let b = (c.b * 255.0).round() as u8;
    cosmic_text::Color::rgb(r, g, b)
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

/// Push a span to `result`, merging it with the previous span if their
/// `Attrs` match. Both slices must be contiguous views into `text`, so
/// merging is a simple span extension into the source string.
pub(crate) fn push_or_merge<'a>(
    text: &'a str,
    result: &mut Vec<(&'a str, cosmic_text::Attrs<'a>)>,
    new_text: &'a str,
    new_attrs: cosmic_text::Attrs<'a>,
) {
    if let Some(last) = result.last_mut() {
        if last.1 == new_attrs {
            // Both slices are contiguous in `text`. Extend the last span.
            let start = (last.0.as_ptr() as usize) - (text.as_ptr() as usize);
            let end = (new_text.as_ptr() as usize + new_text.len()) - (text.as_ptr() as usize);
            last.0 = &text[start..end];
            return;
        }
    }
    result.push((new_text, new_attrs));
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

/// Convert a byte offset into a (line, column) pair, where column is
/// character-based (not byte-based).
fn byte_offset_to_line_col(text: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(text.len());
    let prefix = &text[..offset];
    let line = prefix.bytes().filter(|&b| b == b'\n').count();
    let last_newline = prefix.rfind('\n').map_or(0, |p| p + 1);
    let col = prefix[last_newline..].chars().count();
    (line, col)
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
    /// Whether an IME Commit event was just handled. Used to suppress the
    /// duplicate `KeyPressed.text` that follows on some platforms (Linux/IBus).
    ime_commit_pending: bool,
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
            ime_commit_pending: false,
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
    /// When `true` and `ignore_keyboard` is also `true`, only blocks editing
    /// keys (Insert, Backspace, Delete, Enter, char typing) while allowing
    /// cursor movement (arrows, Home, End, PageUp/Down). Used when the
    /// find/replace bar is open — the text_input handles its own editing,
    /// but the editor should still respond to cursor navigation.
    block_editing: bool,
    /// Find match highlights: `Vec<(line, byte_col_start, byte_col_end)>`.
    /// Set fresh each frame from the editor page's find/replace state.
    /// Empty/none when no find bar is open or no matches exist.
    matches: Option<Vec<(usize, usize, usize)>>,
    /// Index of the currently-focused match within `matches`.
    /// Used to render the current match with a stronger highlight color.
    match_current_idx: usize,
    /// Blink generation counter from the editor state.
    /// Incremented on each `BlinkTick` subscription event to force Iced
    /// to redraw the widget even when no other state has changed.
    blink_gen: u64,
    /// Matching bracket pair to highlight, if any.
    /// Each element is `(line, col)`.
    bracket_pair: Option<((usize, usize), (usize, usize))>,
}

impl<'a> EditorWidget<'a> {
    /// Create a new [`EditorWidget`].
    pub fn new(buffer: &'a EditorBuffer) -> Self {
        Self {
            buffer,
            font_size: 13.0,
            padding: 8.0,
            ignore_keyboard: false,
            block_editing: false,
            matches: None,
            match_current_idx: 0,
            blink_gen: 0,
            bracket_pair: None,
        }
    }

    /// Set the font size.
    #[must_use]
    pub fn font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    /// Set the padding.
    #[must_use]
    pub fn padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    /// When `true`, skip all keyboard event processing.
    /// Set this when another UI element (tree panel, find/replace bar)
    /// has keyboard focus and should receive the events instead.
    #[must_use]
    pub fn ignore_keyboard(mut self, ignore: bool) -> Self {
        self.ignore_keyboard = ignore;
        self
    }

    /// When `true` (and `ignore_keyboard` is also `true`), only block
    /// text-editing keys (Insert, Backspace, Delete, Enter, char input)
    /// while allowing cursor movement keys (arrows, Home, End, PgUp/Down).
    /// Used when the find/replace bar is open so the editor can still
    /// scroll and navigate while the text_input handles its own editing.
    #[must_use]
    pub fn block_editing(mut self, block: bool) -> Self {
        self.block_editing = block;
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

    /// Set the blink generation counter.
    /// This is passed from the editor state's `BlinkTick` handler to force
    /// Iced to detect a widget change and schedule a redraw on each tick.
    #[must_use]
    pub fn blink_gen(mut self, blink_gen: u64) -> Self {
        self.blink_gen = blink_gen;
        self
    }

    /// Set the matching bracket pair to highlight.
    /// `pair` is `((open_line, open_col), (close_line, close_col))`.
    /// Pass `None` to hide bracket highlighting.
    #[must_use]
    pub fn bracket_pair(mut self, pair: Option<((usize, usize), (usize, usize))>) -> Self {
        self.bracket_pair = pair;
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

    #[allow(clippy::cast_precision_loss)]
    fn layout(
        &mut self,
        tree: &mut Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let bounds = limits.max();

        let state = tree.state.downcast_mut::<EditorWidgetState>();

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

        // set_scroll MUST be called before shape_until_scroll / set_size
        buffer.set_scroll(Scroll {
            line: 0,
            vertical: state.scroll_y,
            horizontal: 0.0,
        });
        buffer.set_size(font_sys, Some(text_area_width), Some(text_area_height));
        // Ensure shaping runs even if set_size was a no-op (size unchanged)
        buffer.shape_until_scroll(font_sys, false);

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
                } else if est_y > state.scroll_y + text_area_height {
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

    #[allow(clippy::too_many_lines)]
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

        // ── 1. Fill background ──
        renderer.fill_quad(
            renderer::Quad {
                bounds,
                border: iced::Border::default(),
                ..renderer::Quad::default()
            },
            theme::BG_BASE,
        );

        // Use the buffer Arc that was prepared in layout()
        let buffer_for_draw = state.buffer_for_render.clone().unwrap_or_else(|| {
            // Fallback: create a fresh buffer if layout wasn't called
            let mut guard = graphics_text::font_system().write().unwrap_poison();
            let font_sys = guard.raw();
            let mut buffer = self.buffer.borrow_buffer_mut();
            buffer.set_size(font_sys, Some(text_area_width), Some(text_area_height));
            buffer.shape_until_scroll(font_sys, false);
            let cloned = Arc::new(buffer.clone());
            drop(buffer);
            drop(guard);
            cloned
        });

        let text_clip = text_rect;

        // ── Draw line numbers ──
        let number_color = theme::TEXT_MUTED;
        let number_clip = gutter_clip_rect(bounds, self.padding, gutter_width, text_area_height);

        let mut last_line_i = usize::MAX;
        for run in buffer_for_draw.layout_runs() {
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
                    bounds.x + self.padding + gutter_width,
                    text_y + run.line_top + run.line_height / 2.0,
                ),
                number_color,
                number_clip,
            );
        }

        // ── Draw find match highlights ──
        // Match highlights are drawn BEFORE selection so selection
        // (ACCENT_DIM teal) renders on top of match highlights.
        // Text is drawn AFTER both via fill_raw, so highlights
        // appear as background tints behind the glyphs.
        if let Some(ref matches) = self.matches {
            for (i, &(match_line, col_start, col_end)) in matches.iter().enumerate() {
                let color = if i == self.match_current_idx {
                    theme::FIND_MATCH_CURRENT
                } else {
                    theme::FIND_MATCH_DIM
                };
                for run in buffer_for_draw.layout_runs() {
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
                        let rect = Rectangle {
                            x: text_x + hl.0,
                            y: text_y + run.line_top,
                            width: hl.1,
                            height: run.line_height,
                        };
                        if let Some(clipped) = text_clip.intersection(&rect) {
                            renderer.fill_quad(
                                renderer::Quad {
                                    bounds: clipped,
                                    border: iced::Border::default(),
                                    ..renderer::Quad::default()
                                },
                                color,
                            );
                        }
                    }
                    // Match may span multiple visual runs on soft-wrapped
                    // lines — don't break, continue checking all runs for
                    // this logical line.
                }
            }
        }

        // ── Draw bracket matching highlights ──
        // Draw a subtle background under both the opening and closing bracket.
        if let Some(((open_line, open_col), (close_line, close_col))) = self.bracket_pair {
            let bracket_color = theme::BRACKET_MATCH;
            for &(b_line, b_col) in &[(open_line, open_col), (close_line, close_col)] {
                for run in buffer_for_draw.layout_runs() {
                    if run.line_i != b_line {
                        continue;
                    }
                    // Highlight one character at the bracket position.
                    if let Some(hl) = run.highlight(
                        cosmic_text::Cursor {
                            line: b_line,
                            index: b_col,
                            ..cosmic_text::Cursor::default()
                        },
                        cosmic_text::Cursor {
                            line: b_line,
                            index: b_col + 1,
                            ..cosmic_text::Cursor::default()
                        },
                    ) {
                        let rect = Rectangle {
                            x: text_x + hl.0,
                            y: text_y + run.line_top,
                            width: hl.1,
                            height: run.line_height,
                        };
                        if let Some(clipped) = text_clip.intersection(&rect) {
                            renderer.fill_quad(
                                renderer::Quad {
                                    bounds: clipped,
                                    border: iced::Border::default(),
                                    ..renderer::Quad::default()
                                },
                                bracket_color,
                            );
                        }
                    }
                    break;
                }
            }
        }

        // ── Draw selection rectangles ──
        let cursor_state = self.buffer.cursor();
        let has_selection = cursor_state.selection.is_some();

        if let Some(ref anchor) = cursor_state.selection {
            let start = (cursor_state.line, cursor_state.column);
            let end = (anchor.line, anchor.column);
            let (sel_start, sel_end) = if start < end {
                (start, end)
            } else {
                (end, start)
            };

            for run in buffer_for_draw.layout_runs() {
                if let Some(highlight) = run.highlight(
                    cosmic_text::Cursor {
                        line: sel_start.0,
                        index: sel_start.1,
                        ..cosmic_text::Cursor::default()
                    },
                    cosmic_text::Cursor {
                        line: sel_end.0,
                        index: sel_end.1,
                        ..cosmic_text::Cursor::default()
                    },
                ) {
                    let sel_rect = Rectangle {
                        x: text_x + highlight.0,
                        y: text_y + run.line_top,
                        width: highlight.1,
                        height: run.line_height,
                    };
                    if let Some(clipped) = text_clip.intersection(&sel_rect) {
                        renderer.fill_quad(
                            renderer::Quad {
                                bounds: clipped,
                                border: iced::Border::default(),
                                ..renderer::Quad::default()
                            },
                            theme::ACCENT_DIM,
                        );
                    }
                }
            }
        }

        // ── 5. Draw text via fill_raw ──
        renderer.fill_raw(TextRaw {
            buffer: Arc::downgrade(&buffer_for_draw),
            position: Point::new(text_x, text_y),
            color: iced::Color::WHITE, // neutral multiplier preserves per-glyph colors
            clip_bounds: text_clip,
        });

        // ── 6. Draw cursor (blinking vertical line) ──
        let now = std::time::Instant::now();
        let blink_on = now.duration_since(state.last_blink).as_millis() % 1000 < 500;

        if blink_on && !has_selection {
            let cursor_x;
            let cursor_y;
            let cursor_height;

            if let Some(run) = find_cursor_run(
                buffer_for_draw.layout_runs(),
                cursor_state.line,
                cursor_state.column,
            ) {
                cursor_y = text_y + run.line_top;
                cursor_height = run.line_height;
                let found_x = run
                    .glyphs
                    .iter()
                    .find(|g| {
                        cursor_state.column < run.text[..g.end.min(run.text.len())].chars().count()
                    })
                    .map(|g| g.x);
                cursor_x = text_x
                    + found_x
                        .unwrap_or_else(|| run.glyphs.last().map_or(0.0, |last| last.x + last.w));
            } else {
                cursor_x = 0.0;
                cursor_y = text_y;
                cursor_height = font_metrics().line_height;
            }

            let cursor_rect = Rectangle {
                x: cursor_x,
                y: cursor_y,
                width: 1.5,
                height: cursor_height,
            };

            if let Some(clipped) = text_clip.intersection(&cursor_rect) {
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
                state.mouse_held = true;

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

                    let mut guard = graphics_text::font_system().write().unwrap_poison();
                    let font_sys = guard.raw();
                    let mut buffer = self.buffer.borrow_buffer_mut();
                    buffer.set_scroll(Scroll {
                        line: 0,
                        vertical: state.scroll_y,
                        horizontal: 0.0,
                    });
                    buffer.set_size(font_sys, Some(text_area_width), Some(text_area_height));
                    buffer.shape_until_scroll(font_sys, false);
                }

                if let Some((line, col)) = hit_test(
                    self.buffer,
                    layout,
                    cursor,
                    state.gutter_width,
                    self.padding,
                ) {
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

                    // Always update tracking (enables triple-click to also select).
                    state.last_click_time = Some(now);
                    state.last_click_pos = Some((line, col));

                    if is_double_click {
                        shell.publish(EditorAction::SelectWordAt { line, col });
                        // Clear mouse_held so intermediate CursorMoved events
                        // don't trigger SelectTo and truncate the word selection.
                        state.mouse_held = false;
                    } else {
                        shell.publish(EditorAction::MoveTo { line, col });
                    }

                    // Request redraw to keep the cursor blinking.
                    shell.request_redraw();
                } else {
                    // Gutter/padding click — clear double-click tracking.
                    state.last_click_time = None;
                    state.last_click_pos = None;
                    // Keep blinking after gutter click too.
                    shell.request_redraw();
                }
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
                    if self.block_editing {
                        // Find/replace bar is open — allow cursor movement
                        // (arrows, Home, End, PgUp/PgDown) through so the
                        // editor can still navigate while the text_input
                        // handles its own editing keys.
                        match key_press {
                            key::Key::Named(
                                key::Named::ArrowLeft
                                | key::Named::ArrowRight
                                | key::Named::ArrowUp
                                | key::Named::ArrowDown
                                | key::Named::Home
                                | key::Named::End
                                | key::Named::PageUp
                                | key::Named::PageDown,
                            ) => {
                                // Fall through to cursor movement handling below.
                            }
                            _ => return,
                        }
                    } else {
                        // Tree panel or quick-open — block ALL keyboard.
                        return;
                    }
                }
                // Any keyboard cursor movement re-enables auto-scroll
                if is_cursor_movement_key(key_press) {
                    state.auto_scroll_enabled = true;
                    state.last_blink = std::time::Instant::now();
                }

                // ── Clipboard shortcuts (Cmd/Ctrl+C/X/V) ──────────────
                // On macOS, only Cmd (not Ctrl) triggers clipboard shortcuts;
                // Ctrl+C/X/V are terminal control characters.
                #[cfg(target_os = "macos")]
                let is_clipboard_mod = modifiers.command() && !modifiers.control();
                #[cfg(not(target_os = "macos"))]
                let is_clipboard_mod =
                    (modifiers.command() || modifiers.control()) && !modifiers.alt();
                if is_clipboard_mod {
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
                    // On macOS, platform shortcuts use Cmd only.
                    #[cfg(target_os = "macos")]
                    let platform_mod = modifiers.command();
                    #[cfg(not(target_os = "macos"))]
                    let platform_mod = modifiers.command() || modifiers.control();
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

                        let result = {
                            let mut guard = graphics_text::font_system().write().unwrap_poison();
                            let font_sys = guard.raw();
                            let mut buffer = self.buffer.borrow_buffer_mut();
                            buffer.set_scroll(Scroll {
                                line: 0,
                                vertical: state.scroll_y,
                                horizontal: 0.0,
                            });
                            buffer.set_size(
                                font_sys,
                                Some(text_area_width),
                                Some(text_area_height),
                            );
                            buffer.shape_until_scroll(font_sys, false);

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
                        }; // drop RefMut and font_system guard

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
                    #[cfg(target_os = "macos")]
                    let platform_mod = modifiers.command();
                    #[cfg(not(target_os = "macos"))]
                    let platform_mod = modifiers.command() || modifiers.control();
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

                        let result = {
                            let mut guard = graphics_text::font_system().write().unwrap_poison();
                            let font_sys = guard.raw();
                            let mut buffer = self.buffer.borrow_buffer_mut();
                            buffer.set_scroll(Scroll {
                                line: 0,
                                vertical: state.scroll_y,
                                horizontal: 0.0,
                            });
                            buffer.set_size(
                                font_sys,
                                Some(text_area_width),
                                Some(text_area_height),
                            );
                            buffer.shape_until_scroll(font_sys, false);

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
                        }; // drop RefMut and font_system guard

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
                #[cfg(target_os = "macos")]
                let is_platform_mod = modifiers.command() && !modifiers.control();
                #[cfg(not(target_os = "macos"))]
                let is_platform_mod =
                    (modifiers.command() || modifiers.control()) && !modifiers.alt();

                if !is_platform_mod {
                    if let Some(committed) = text {
                        if !committed.is_empty() {
                            if state.ime_commit_pending {
                                // IME Commit already inserted this text.
                                state.ime_commit_pending = false;
                                return;
                            }
                            let committed: &str = committed.as_ref();
                            if committed.chars().count() == 1 {
                                let c = committed.chars().next().unwrap();
                                if !c.is_control() {
                                    shell.publish(EditorAction::Insert(c));
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
                    let is_cursor_move = is_action_cursor_movement(action);
                    if is_cursor_move {
                        state.auto_scroll_enabled = true;
                        state.last_blink = std::time::Instant::now();
                    }
                    shell.publish(action.clone());
                    // After a cursor-movement action, invalidate layout so the
                    // auto-scroll logic in layout() runs and adjusts the scroll
                    // offset to bring the cursor into view.
                    if is_cursor_move {
                        shell.invalidate_layout();
                    }
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
                        state.ime_commit_pending = true;
                        if committed.chars().count() == 1 {
                            let c = committed.chars().next().unwrap();
                            if !c.is_control() {
                                shell.publish(EditorAction::Insert(c));
                            }
                        } else {
                            shell.publish(EditorAction::Paste(committed.clone()));
                        }
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

// ── Keybinding mapping ──────────────────────────────────────────────

/// Returns `true` if the key is a cursor-movement key (arrow keys,
/// Home, End, PageUp, PageDown).
fn is_cursor_movement_key(key: &key::Key) -> bool {
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

/// Returns `true` when the action moves the cursor (including extending selection).
fn is_action_cursor_movement(action: &EditorAction) -> bool {
    matches!(
        action,
        EditorAction::MoveLeft
            | EditorAction::MoveRight
            | EditorAction::MoveUp
            | EditorAction::MoveDown
            | EditorAction::MoveHome
            | EditorAction::MoveEnd
            | EditorAction::MoveWordLeft
            | EditorAction::MoveWordRight
            | EditorAction::MoveDocStart
            | EditorAction::MoveDocEnd
            | EditorAction::MovePageUp
            | EditorAction::MovePageDown
            | EditorAction::SelectLeft
            | EditorAction::SelectRight
            | EditorAction::SelectUp
            | EditorAction::SelectDown
            | EditorAction::SelectHome
            | EditorAction::SelectEnd
            | EditorAction::SelectWordLeft
            | EditorAction::SelectWordRight
            | EditorAction::SelectDocStart
            | EditorAction::SelectDocEnd
            | EditorAction::SelectPageUp
            | EditorAction::SelectPageDown
            | EditorAction::SelectWordAt { .. }
            | EditorAction::JumpToMatchingBracket
            | EditorAction::MoveLineUp
            | EditorAction::MoveLineDown
    )
}

/// Map a keyboard key + modifiers to an [`EditorAction`].
///
/// `has_text` is `true` when the `KeyPressed` event carried a non-empty
/// `text` field (dead key / IME / AltGr composition). Only used for
/// distinguishing AltGr from shortcuts on non-macOS — character insertion
/// itself is handled in `on_event`.
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
    #[cfg(target_os = "macos")]
    let platform_mod = modifiers.command();
    #[cfg(not(target_os = "macos"))]
    let platform_mod = modifiers.command() || modifiers.control();
    let shift = modifiers.shift();
    let alt = modifiers.alt();

    // On non-macOS, AltGr (Ctrl+Alt) produces text — treat as character
    // input, not a shortcut. When `has_text` is true, AltGr is active.
    #[cfg(not(target_os = "macos"))]
    let altgr_active = alt && modifiers.control() && _has_text;
    #[cfg(target_os = "macos")]
    let altgr_active = false;

    match key {
        key::Key::Named(named) => {
            match (named, platform_mod, shift, alt) {
                // Platform mod + Up/Down → jump to document start/end
                (key::Named::ArrowUp, true, false, false) => Some(EditorAction::MoveDocStart),
                (key::Named::ArrowDown, true, false, false) => Some(EditorAction::MoveDocEnd),
                (key::Named::ArrowUp, true, true, false) => Some(EditorAction::SelectDocStart),
                (key::Named::ArrowDown, true, true, false) => Some(EditorAction::SelectDocEnd),

                // Navigation (no modifiers)
                (key::Named::ArrowLeft, false, false, false) => Some(EditorAction::MoveLeft),
                (key::Named::ArrowRight, false, false, false) => Some(EditorAction::MoveRight),
                (key::Named::ArrowUp, false, false, false) => Some(EditorAction::MoveUp),
                (key::Named::ArrowDown, false, false, false) => Some(EditorAction::MoveDown),
                (key::Named::Home, false, false, false) => Some(EditorAction::MoveHome),
                (key::Named::End, false, false, false) => Some(EditorAction::MoveEnd),

                // Navigation + Shift
                (key::Named::ArrowLeft, false, true, false) => Some(EditorAction::SelectLeft),
                (key::Named::ArrowRight, false, true, false) => Some(EditorAction::SelectRight),
                (key::Named::ArrowUp, false, true, false) => Some(EditorAction::SelectUp),
                (key::Named::ArrowDown, false, true, false) => Some(EditorAction::SelectDown),
                (key::Named::Home, false, true, false) => Some(EditorAction::SelectHome),
                (key::Named::End, false, true, false) => Some(EditorAction::SelectEnd),

                // Page keys
                (key::Named::PageUp, false, false, false) => Some(EditorAction::MovePageUp),
                (key::Named::PageDown, false, false, false) => Some(EditorAction::MovePageDown),
                (key::Named::PageUp, false, true, false) => Some(EditorAction::SelectPageUp),
                (key::Named::PageDown, false, true, false) => Some(EditorAction::SelectPageDown),

                // Alt+Left/Right → word boundary navigation
                (key::Named::ArrowLeft, false, false, true) => Some(EditorAction::MoveWordLeft),
                (key::Named::ArrowRight, false, false, true) => Some(EditorAction::MoveWordRight),
                (key::Named::ArrowLeft, false, true, true) => Some(EditorAction::SelectWordLeft),
                (key::Named::ArrowRight, false, true, true) => Some(EditorAction::SelectWordRight),

                // Alt+Up/Down → move line up/down
                (key::Named::ArrowUp, false, false, true) => Some(EditorAction::MoveLineUp),
                (key::Named::ArrowDown, false, false, true) => Some(EditorAction::MoveLineDown),

                // Platform mod + Left/Right → jump to line start/end
                (key::Named::ArrowLeft, true, false, false) => Some(EditorAction::MoveHome),
                (key::Named::ArrowRight, true, false, false) => Some(EditorAction::MoveEnd),
                (key::Named::ArrowLeft, true, true, false) => Some(EditorAction::SelectHome),
                (key::Named::ArrowRight, true, true, false) => Some(EditorAction::SelectEnd),
                // Platform mod + Home/End → document start/end
                (key::Named::Home, true, false, false) => Some(EditorAction::MoveDocStart),
                (key::Named::End, true, false, false) => Some(EditorAction::MoveDocEnd),
                (key::Named::Home, true, true, false) => Some(EditorAction::SelectDocStart),
                (key::Named::End, true, true, false) => Some(EditorAction::SelectDocEnd),

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
                        Some('f') => return Some(EditorAction::MoveRight),
                        Some('b') => return Some(EditorAction::MoveLeft),
                        Some('a') => return Some(EditorAction::MoveHome),
                        Some('e') => return Some(EditorAction::MoveEnd),
                        Some('h') => return Some(EditorAction::Backspace),
                        Some('d') => return Some(EditorAction::Delete),
                        Some('n') => return Some(EditorAction::MoveDown),
                        Some('p') => return Some(EditorAction::MoveUp),
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

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let buf = EditorBuffer::new();
        assert_eq!(buf.text(), "");
        assert_eq!(buf.line_count(), 1);
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 0);
        assert!(cursor.selection.is_none());
    }

    #[test]
    fn test_with_text() {
        let buf = EditorBuffer::with_text("hello\nworld", None);
        assert_eq!(buf.text(), "hello\nworld");
        assert_eq!(buf.line_count(), 2);
        assert_eq!(buf.line(0).as_deref(), Some("hello"));
        assert_eq!(buf.line(1).as_deref(), Some("world"));
    }

    #[test]
    fn test_line_out_of_bounds() {
        let buf = EditorBuffer::with_text("hello", None);
        assert!(buf.line(1).is_none());
        assert!(buf.line(usize::MAX).is_none());
    }

    #[test]
    fn test_cursor_move_to() {
        let buf = EditorBuffer::with_text("hello\nworld", None);
        buf.move_to(1, 2);
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 1);
        assert_eq!(cursor.column, 2);
        assert!(cursor.selection.is_none());
    }

    #[test]
    fn test_cursor_move_to_beyond_end() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.move_to(999, 999);
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 5);
    }

    #[test]
    fn test_select_all_and_selection() {
        let buf = EditorBuffer::with_text("hello\nworld", None);
        buf.select_all();
        let cursor = buf.cursor();
        assert!(cursor.selection.is_some());
        let sel = buf.selection();
        assert_eq!(sel, Some("hello\nworld".to_string()));
    }

    #[test]
    fn test_no_selection() {
        let buf = EditorBuffer::with_text("hello", None);
        assert!(buf.selection().is_none());
    }

    #[test]
    fn test_set_text() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.set_text("world");
        assert_eq!(buf.text(), "world");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 0);
    }

    #[test]
    fn test_insert_character() {
        let buf = EditorBuffer::with_text("helo", None);
        buf.move_to(0, 2);
        buf.perform_action(EditorAction::Insert('l'));
        assert_eq!(buf.text(), "hello");
    }

    #[test]
    fn test_insert_at_end() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.move_to(0, 5);
        buf.perform_action(EditorAction::Insert('!'));
        assert_eq!(buf.text(), "hello!");
    }

    #[test]
    fn test_enter() {
        let buf = EditorBuffer::with_text("hello world", None);
        buf.move_to(0, 5);
        buf.perform_action(EditorAction::Enter);
        assert_eq!(buf.text(), "hello\n world");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 1);
        // "hello world" has no leading whitespace, so auto-indent produces
        // column 0 (empty indent on the new line).
        assert_eq!(cursor.column, 0);
    }

    #[test]
    fn test_backspace() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.move_to(0, 5);
        buf.perform_action(EditorAction::Backspace);
        assert_eq!(buf.text(), "hell");
        let cursor = buf.cursor();
        assert_eq!(cursor.column, 4);
    }

    #[test]
    fn test_backspace_at_start() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.perform_action(EditorAction::Backspace);
        assert_eq!(buf.text(), "hello");
    }

    #[test]
    fn test_backspace_newline() {
        let buf = EditorBuffer::with_text("hello\nworld", None);
        buf.move_to(1, 0);
        buf.perform_action(EditorAction::Backspace);
        assert_eq!(buf.text(), "helloworld");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 5);
    }

    #[test]
    fn test_delete() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.perform_action(EditorAction::Delete);
        assert_eq!(buf.text(), "ello");
    }

    #[test]
    fn test_delete_at_end() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.move_to(0, 5);
        buf.perform_action(EditorAction::Delete);
        assert_eq!(buf.text(), "hello");
    }

    #[test]
    fn test_paste() {
        let buf = EditorBuffer::with_text("heo", None);
        buf.move_to(0, 2);
        buf.perform_action(EditorAction::Paste("ll".to_string()));
        assert_eq!(buf.text(), "hello");
    }

    #[test]
    fn test_select_to() {
        let buf = EditorBuffer::with_text("hello world", None);
        buf.perform_action(EditorAction::SelectTo { line: 0, col: 5 });
        let sel = buf.selection();
        assert_eq!(sel, Some("hello".to_string()));
    }

    #[test]
    fn test_indent() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.perform_action(EditorAction::Indent);
        assert_eq!(buf.text(), "\thello");
    }

    #[test]
    fn test_insert_with_selection_replaces() {
        let buf = EditorBuffer::with_text("hello world", None);
        buf.select_all();
        buf.perform_action(EditorAction::Insert('X'));
        assert_eq!(buf.text(), "X");
    }

    #[test]
    fn test_enter_with_selection_replaces() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.move_to(0, 3);
        buf.perform_action(EditorAction::SelectTo { line: 0, col: 5 });
        buf.perform_action(EditorAction::Enter);
        assert_eq!(buf.text(), "hel\n");
    }

    #[test]
    fn test_paste_with_selection_replaces() {
        let buf = EditorBuffer::with_text("hello world", None);
        buf.move_to(0, 6);
        buf.perform_action(EditorAction::SelectTo { line: 0, col: 11 });
        buf.perform_action(EditorAction::Paste("there".to_string()));
        assert_eq!(buf.text(), "hello there");
    }

    #[test]
    fn test_line_col_roundtrip() {
        let text = "hello\nworld\nfoo";
        for (line, line_text) in text.lines().enumerate() {
            for (col, _) in line_text.chars().enumerate() {
                let offset = line_col_to_byte_offset(text, line, col);
                let (rl, rc) = byte_offset_to_line_col(text, offset);
                assert_eq!(rl, line, "line mismatch at ({line},{col})");
                assert_eq!(rc, col, "col mismatch at ({line},{col})");
            }
            // End of line
            let col = line_text.chars().count();
            let offset = line_col_to_byte_offset(text, line, col);
            let (rl, rc) = byte_offset_to_line_col(text, offset);
            assert_eq!(rl, line, "line mismatch at end of line {line}");
            assert_eq!(rc, col, "col mismatch at end of line {line}");
        }
    }

    #[test]
    fn test_selection_after_select_to_then_move() {
        let buf = EditorBuffer::with_text("abcdef", None);
        // Select "bcd"
        buf.move_to(0, 1);
        buf.perform_action(EditorAction::SelectTo { line: 0, col: 4 });
        assert_eq!(buf.selection(), Some("bcd".to_string()));
        // Performing a move clears selection
        buf.move_to(0, 0);
        assert!(buf.selection().is_none());
    }

    #[test]
    fn test_unindent() {
        let buf = EditorBuffer::with_text("    hello", None);
        buf.perform_action(EditorAction::Unindent);
        assert_eq!(buf.text(), "   hello");
    }

    #[test]
    fn test_multi_byte_character() {
        let buf = EditorBuffer::with_text("héllo", None);
        buf.move_to(0, 1);
        buf.perform_action(EditorAction::Insert('é'));
        assert_eq!(buf.text(), "hééllo");
        let cursor = buf.cursor();
        assert_eq!(cursor.column, 2);
    }

    #[test]
    fn test_select_all_empty() {
        let buf = EditorBuffer::new();
        buf.select_all();
        assert!(buf.selection().is_none());
    }

    // ── Line comment prefix ───────────────────────────────────────

    #[test]
    fn test_line_comment_prefix_by_language() {
        use super::highlight::HighlightLanguage::*;
        assert_eq!(line_comment_prefix(Some(Rust), None), Some("//"));
        assert_eq!(line_comment_prefix(Some(Python), None), Some("#"));
        assert_eq!(line_comment_prefix(Some(Sql), None), Some("--"));
        assert_eq!(line_comment_prefix(Some(Json), None), None);
        assert_eq!(line_comment_prefix(Some(Html), None), None);
        assert_eq!(line_comment_prefix(Some(Markdown), None), None);
    }

    #[test]
    fn test_line_comment_prefix_by_extension() {
        assert_eq!(line_comment_prefix(None, Some("yaml")), Some("#"));
        assert_eq!(line_comment_prefix(None, Some("yml")), Some("#"));
        assert_eq!(line_comment_prefix(None, Some("rs")), None); // falls to ext-only
    }

    #[test]
    fn test_line_comment_prefix_none() {
        assert_eq!(line_comment_prefix(None, None), None);
        assert_eq!(line_comment_prefix(None, Some("xyz")), None);
    }

    // ── Toggle line comment ───────────────────────────────────────

    #[test]
    fn test_toggle_line_comment_add() {
        let buf = EditorBuffer::with_text("hello", Some(HighlightLanguage::Rust));
        buf.perform_action(EditorAction::ToggleLineComment);
        assert_eq!(buf.text(), "// hello");
    }

    #[test]
    fn test_toggle_line_comment_remove() {
        let buf = EditorBuffer::with_text("// hello", Some(HighlightLanguage::Rust));
        buf.perform_action(EditorAction::ToggleLineComment);
        assert_eq!(buf.text(), "hello");
    }

    #[test]
    fn test_toggle_line_comment_remove_with_space() {
        let buf = EditorBuffer::with_text("//  hello", Some(HighlightLanguage::Rust));
        buf.perform_action(EditorAction::ToggleLineComment);
        assert_eq!(buf.text(), " hello");
    }

    #[test]
    fn test_toggle_line_comment_preserves_whitespace() {
        let buf = EditorBuffer::with_text("    hello", Some(HighlightLanguage::Rust));
        buf.perform_action(EditorAction::ToggleLineComment);
        assert_eq!(buf.text(), "    // hello");
    }

    #[test]
    fn test_toggle_line_comment_noop_for_unknown() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.perform_action(EditorAction::ToggleLineComment);
        assert_eq!(buf.text(), "hello");
    }

    #[test]
    fn test_toggle_line_comment_rust_hash() {
        let buf = EditorBuffer::with_text("hello", Some(HighlightLanguage::Python));
        buf.perform_action(EditorAction::ToggleLineComment);
        assert_eq!(buf.text(), "# hello");
    }

    #[test]
    fn test_toggle_line_comment_yaml_via_extension() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.set_file_extension(Some("yaml"));
        buf.perform_action(EditorAction::ToggleLineComment);
        assert_eq!(buf.text(), "# hello");
    }

    #[test]
    fn test_toggle_line_comment_unknown_extension_noop() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.set_file_extension(Some("xyz"));
        buf.perform_action(EditorAction::ToggleLineComment);
        assert_eq!(buf.text(), "hello");
    }

    // ── Jump to matching bracket ───────────────────────────────────

    #[test]
    fn test_jump_to_matching_bracket_forward() {
        let buf = EditorBuffer::with_text("(hello)", None);
        buf.move_to(0, 1); // Cursor right after '('
        buf.perform_action(EditorAction::JumpToMatchingBracket);
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 6); // On ')'
    }

    #[test]
    fn test_jump_to_matching_bracket_backward() {
        let buf = EditorBuffer::with_text("(hello)", None);
        buf.move_to(0, 6); // Cursor at ')'
        buf.perform_action(EditorAction::JumpToMatchingBracket);
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 1); // Right after '('
    }

    #[test]
    fn test_jump_to_matching_bracket_square() {
        let buf = EditorBuffer::with_text("[hello]", None);
        buf.move_to(0, 1);
        buf.perform_action(EditorAction::JumpToMatchingBracket);
        let cursor = buf.cursor();
        assert_eq!(cursor.column, 6);
    }

    #[test]
    fn test_jump_to_matching_bracket_brace() {
        let buf = EditorBuffer::with_text("{hello}", None);
        buf.move_to(0, 1);
        buf.perform_action(EditorAction::JumpToMatchingBracket);
        let cursor = buf.cursor();
        assert_eq!(cursor.column, 6);
    }

    #[test]
    fn test_jump_to_matching_bracket_none() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.move_to(0, 3);
        buf.perform_action(EditorAction::JumpToMatchingBracket);
        let cursor = buf.cursor();
        assert_eq!(cursor.column, 3); // No movement
    }

    // ── Delete line ────────────────────────────────────────────────

    #[test]
    fn test_delete_current_line() {
        let buf = EditorBuffer::with_text("line1\nline2\nline3", None);
        buf.move_to(1, 0);
        buf.perform_action(EditorAction::DeleteLine);
        assert_eq!(buf.text(), "line1\nline3");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 1); // Stayed at index 1 (now "line3")
        assert_eq!(cursor.column, 0);
    }

    #[test]
    fn test_delete_first_line() {
        let buf = EditorBuffer::with_text("line1\nline2\nline3", None);
        buf.perform_action(EditorAction::DeleteLine);
        assert_eq!(buf.text(), "line2\nline3");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 0);
    }

    #[test]
    fn test_delete_last_line() {
        let buf = EditorBuffer::with_text("line1\nline2\nline3", None);
        buf.move_to(2, 0);
        buf.perform_action(EditorAction::DeleteLine);
        assert_eq!(buf.text(), "line1\nline2");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 1);
    }

    #[test]
    fn test_delete_single_line() {
        let buf = EditorBuffer::with_text("hello", None);
        buf.perform_action(EditorAction::DeleteLine);
        assert_eq!(buf.text(), "");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 0);
    }

    #[test]
    fn test_delete_selected_lines() {
        let buf = EditorBuffer::with_text("a\nb\nc\nd\ne", None);
        buf.move_to(1, 0);
        buf.perform_action(EditorAction::SelectTo { line: 3, col: 0 });
        buf.perform_action(EditorAction::DeleteLine);
        assert_eq!(buf.text(), "a\ne");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 1);
    }

    // ── Duplicate line ─────────────────────────────────────────────

    #[test]
    fn test_duplicate_current_line() {
        let buf = EditorBuffer::with_text("hello\nworld", None);
        buf.move_to(0, 0);
        buf.perform_action(EditorAction::DuplicateLine);
        assert_eq!(buf.text(), "hello\nhello\nworld");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 1); // Cursor on duplicated line
        assert_eq!(cursor.column, 0);
    }

    #[test]
    fn test_duplicate_last_line() {
        let buf = EditorBuffer::with_text("hello\nworld", None);
        buf.move_to(1, 0);
        buf.perform_action(EditorAction::DuplicateLine);
        assert_eq!(buf.text(), "hello\nworld\nworld");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 2);
    }

    #[test]
    fn test_duplicate_selected_lines() {
        let buf = EditorBuffer::with_text("a\nb\nc\nd", None);
        buf.move_to(1, 0);
        buf.perform_action(EditorAction::SelectTo { line: 2, col: 0 });
        buf.perform_action(EditorAction::DuplicateLine);
        assert_eq!(buf.text(), "a\nb\nc\nb\nc\nd");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 3);
    }

    // ── Move line up/down ──────────────────────────────────────────

    #[test]
    fn test_move_line_up() {
        let buf = EditorBuffer::with_text("a\nb\nc", None);
        buf.move_to(1, 0);
        buf.perform_action(EditorAction::MoveLineUp);
        assert_eq!(buf.text(), "b\na\nc");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 0);
    }

    #[test]
    fn test_move_line_down() {
        let buf = EditorBuffer::with_text("a\nb\nc", None);
        buf.move_to(1, 0);
        buf.perform_action(EditorAction::MoveLineDown);
        assert_eq!(buf.text(), "a\nc\nb");
        let cursor = buf.cursor();
        assert_eq!(cursor.line, 2);
        assert_eq!(cursor.column, 0);
    }

    #[test]
    fn test_move_line_up_at_top() {
        let buf = EditorBuffer::with_text("a\nb", None);
        buf.perform_action(EditorAction::MoveLineUp);
        assert_eq!(buf.text(), "a\nb"); // No change
    }

    #[test]
    fn test_move_line_down_at_bottom() {
        let buf = EditorBuffer::with_text("a\nb", None);
        buf.move_to(1, 0);
        buf.perform_action(EditorAction::MoveLineDown);
        assert_eq!(buf.text(), "a\nb"); // No change
    }

    // ── Multi-line indent/outdent ──────────────────────────────────

    #[test]
    fn test_indent_with_selection() {
        let buf = EditorBuffer::with_text("hello\nworld\nfoo", None);
        buf.move_to(0, 0);
        buf.perform_action(EditorAction::SelectTo { line: 1, col: 0 });
        buf.perform_action(EditorAction::Indent);
        assert_eq!(buf.text(), "\thello\n\tworld\nfoo");
    }

    #[test]
    fn test_unindent_with_selection() {
        let buf = EditorBuffer::with_text("\thello\n\tworld\nfoo", None);
        buf.move_to(0, 0);
        buf.perform_action(EditorAction::SelectTo { line: 1, col: 0 });
        buf.perform_action(EditorAction::Unindent);
        assert_eq!(buf.text(), "hello\nworld\nfoo");
    }
}
