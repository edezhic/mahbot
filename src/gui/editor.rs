//! Code editor dashboard page — tabbed code editor with file tree, syntax-aware
//! editing, and workspace-backed tab persistence.
//!
//! Layout: split view of file tree (left, 25%) and tabbed editor (right, 75%).
//! Workspace selection is handled by the Home page picker. Tabs persist
//! to the workspace database and are restored on workspace selection.
//! Key bindings: Ctrl+S/Cmd+S to save, Tab/Shift+Tab for indent/outdent,
//! Ctrl+B for tree focus toggle.
//!
//! Tree keyboard navigation: when tree is focused, Arrow Up/Down navigate
//! entries, Enter opens files or expands directories, Escape exits focus.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use iced::widget::{Space, button, column, container, row, scrollable, text, text_input};
use iced::{
    Alignment, Element, Length, Subscription, Task,
    keyboard::{self},
    widget::Id,
};

use iced_fonts::lucide;

use fff_search::grep::{GrepMode, GrepSearchOptions};
use fff_search::parse_grep_query;

use super::context_menu::ContextMenu;

use crate::git_commands::{is_git_repo, run_git_check_ignore, run_git_output, run_git_status};
use crate::util::unquote_c_style;

use super::editor_widget::{LineEnding, detect_line_ending, has_trailing_newline};
use crate::tools::MAX_FILE_SIZE_BYTES as MAX_FILE_SIZE;

use super::editor_widget::EditorBuffer;
use super::theme;
use super::widgets::{self, FileTree};

mod editor_dialog;

// ── Constants ─────────────────────────────────────────────────────

/// Estimated width per tab in the tab bar for programmatic scrolling.
const ESTIMATED_TAB_WIDTH: f32 = 140.0;

/// Tick interval (keeps consistency with other dashboard pages).
const TICK_INTERVAL_SECS: u64 = 5;

/// Interval for re-reading expanded directory entries from disk.
const DIR_REFRESH_INTERVAL_SECS: u64 = 30;

/// Base font size for the editor.
const EDITOR_FONT_SIZE: f32 = 13.0;

/// Widget IDs for find/replace text inputs (used for auto-focus).
const FIND_SEARCH_ID: &str = "find_search_input";
const FIND_REPLACE_ID: &str = "find_replace_input";

/// Widget ID for the global search input.
const GLOBAL_SEARCH_INPUT_ID: &str = "global_search_input";

/// Widget ID for the go-to-line input.
const GOTO_LINE_INPUT_ID: &str = "goto_line_input";

/// Widget ID for the quick-open filter input.
const QUICK_OPEN_INPUT_ID: &str = "quick_open_input";

/// Widget ID for the new file/directory name input.
const NEW_ITEM_INPUT_ID: &str = "new_item_input";

/// Maximum number of global search results to display.
const MAX_GLOBAL_SEARCH_RESULTS: usize = 200;

/// Maximum matches per file for global search — spread results across files.
const GLOBAL_SEARCH_MATCHES_PER_FILE: usize = 20;

/// Debounce delay for global search query input (milliseconds).
const GLOBAL_SEARCH_DEBOUNCE_MS: u64 = 300;

/// Check whether a file name is an OS-generated metadata file that should
/// be hidden from the file tree.
#[must_use]
fn is_os_file(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower == ".ds_store" || lower == "thumbs.db" || lower == "desktop.ini"
}

/// Render a centered empty-state placeholder with text content.
///
/// The caller passes a fully-configured `text()` widget (with size, color,
/// optional font, etc.) and this helper wraps it in the standard centered
/// container pattern used throughout the editor panel.
fn empty_placeholder(
    text: iced::widget::Text<'_, iced::Theme, iced::Renderer>,
) -> Element<'_, EditorMessage> {
    container(text)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .into()
}

// ── Types ─────────────────────────────────────────────────────────

/// File-system entry for the directory tree.
#[derive(Debug, Clone)]
pub struct FsEntry {
    pub name: String,
    /// Path relative to the workspace root.
    pub full_path: String,
    pub is_dir: bool,
    /// Error message if this entry couldn't be properly inspected
    /// (broken symlink, permission denied, etc.).
    pub error: Option<String>,
}

/// Git file status for coloring the file tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitFileStatus {
    /// File has uncommitted modifications (M in porcelain output).
    Modified,
    /// File is untracked (?? in porcelain output) or newly added (A).
    Added,
}

/// A single editor tab (metadata, no content).
#[derive(Debug, Clone)]
struct Tab {
    /// Full filesystem path to the file.
    path: String,
    /// Display name (file name component only).
    file_name: String,
    /// Whether the file has unsaved changes.
    is_dirty: bool,
    /// Whether the file ends with a newline.
    has_trailing_newline: bool,
    /// Detected line ending convention.
    line_ending: LineEnding,
}

/// Content data for a tab, keyed by full path.
struct TabData {
    content: super::editor_widget::EditorBuffer,
    /// Undo/redo stack for this tab.
    undo_stack: RefCell<UndoStack>,
    /// Find/replace state (None when bar is hidden).
    find_replace_state: Option<FindReplaceState>,
    /// Hash of the last saved (or loaded) text. Used by undo/redo to
    /// detect when the editor returns to the saved state.
    saved_text_hash: u64,
}

/// Fast non-crypto hash of a string for dirty-state comparison.
fn hash_text(text: &str) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(text.as_bytes());
    h.finish()
}

/// Shared helper to construct a `Tab` + `TabData` pair from file text
/// and metadata.  Returns the pair together with the file's mtime (if
/// readable) so the caller can record it in `file_mtimes`.
fn make_tab_and_data(
    path: &str,
    text: &str,
    has_trailing_newline: bool,
    line_ending: LineEnding,
    is_dirty: bool,
    saved_text_hash: u64,
) -> (Tab, TabData, Option<SystemTime>) {
    let content = EditorBuffer::from_file(text, path);
    let file_name = Path::new(path)
        .file_name()
        .map_or_else(|| path.to_string(), |n| n.to_string_lossy().to_string());

    let tab = Tab {
        path: path.to_string(),
        file_name,
        is_dirty,
        has_trailing_newline,
        line_ending,
    };

    let tab_data = TabData {
        content,
        undo_stack: RefCell::new(UndoStack::new()),
        find_replace_state: None,
        saved_text_hash,
    };

    let mtime = std::fs::metadata(path)
        .ok()
        .and_then(|meta| meta.modified().ok());

    (tab, tab_data, mtime)
}

// ── Undo/Redo ──────────────────────────────────────────────────────

/// Snapshot-based undo/redo stack. Stores full-content snapshots
/// with cursor positions. Bounded to [`MAX_UNDO_DEPTH`] entries.
#[derive(Debug, Clone)]
struct UndoStack {
    /// Previous states, newest last.
    undo: Vec<UndoSnapshot>,
    /// Undone states, cleared on new edit.
    redo: Vec<UndoSnapshot>,
    /// Prevents snapping during composite operations (auto-close pairs).
    batch_depth: usize,
}

/// A single undo snapshot.
#[derive(Debug, Clone)]
struct UndoSnapshot {
    text: String,
    cursor_line: usize,
    cursor_col: usize,
}

/// Maximum undo/redo entries per tab. Reduced for large files.
const MAX_UNDO_DEPTH: usize = 100;
/// Threshold (bytes) above which max undo depth is halved.
const LARGE_FILE_UNDO_THRESHOLD: usize = 100_000;

impl UndoStack {
    #[must_use]
    const fn new() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            batch_depth: 0,
        }
    }

    /// Take a snapshot before an edit is performed.
    fn snap_before_edit(&mut self, content: &super::editor_widget::EditorBuffer) {
        if self.batch_depth > 0 {
            return;
        }
        let text = content.text();
        let cursor = content.cursor();
        let max_depth = if text.len() > LARGE_FILE_UNDO_THRESHOLD {
            MAX_UNDO_DEPTH / 2
        } else {
            MAX_UNDO_DEPTH
        };
        self.redo.clear();
        self.undo.push(UndoSnapshot {
            text,
            cursor_line: cursor.line,
            cursor_col: cursor.column,
        });
        if self.undo.len() > max_depth {
            self.undo.remove(0);
        }
    }

    fn push_and_pop(
        dst: &mut Vec<UndoSnapshot>,
        src: &mut Vec<UndoSnapshot>,
        content: &super::editor_widget::EditorBuffer,
    ) -> Option<UndoSnapshot> {
        let cursor = content.cursor();
        dst.push(UndoSnapshot {
            text: content.text(),
            cursor_line: cursor.line,
            cursor_col: cursor.column,
        });
        src.pop()
    }

    #[must_use]
    fn undo(&mut self, content: &super::editor_widget::EditorBuffer) -> Option<UndoSnapshot> {
        if self.batch_depth > 0 {
            return None;
        }
        Self::push_and_pop(&mut self.redo, &mut self.undo, content)
    }

    #[must_use]
    fn redo(&mut self, content: &super::editor_widget::EditorBuffer) -> Option<UndoSnapshot> {
        if self.batch_depth > 0 {
            return None;
        }
        Self::push_and_pop(&mut self.undo, &mut self.redo, content)
    }
}

// ── Find/Replace ───────────────────────────────────────────────────

/// State for the find/replace search bar.
#[derive(Debug, Clone)]
struct FindReplaceState {
    /// Current search query string.
    query: String,
    /// Replace-with string.
    replace: String,
    /// Byte ranges of all matches in the file.
    matches: Vec<std::ops::Range<usize>>,
    /// Index of the currently focused match.
    current_match_idx: usize,
    /// Whether matching is case-sensitive (default: false).
    case_sensitive: bool,
}

// ── Global Search ──────────────────────────────────────────────────

/// Status of the global (find-in-files) search.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GlobalSearchStatus {
    /// Search panel is open but no query entered yet.
    Idle,
    /// Search is in progress.
    Searching,
    /// Search completed with results.
    Done,
    /// Search completed with no results.
    NoResults,
    /// Search encountered an error.
    Error(String),
}

/// Owned representation of a single grep match, extracted from
/// `fff_search::GrepResult` so it can cross async boundaries.
#[derive(Debug, Clone)]
pub struct OwnedGrepMatch {
    /// Absolute filesystem path to the matched file.
    abs_path: String,
    /// Relative path (for display).
    rel_path: String,
    /// 1-based line number.
    line_number: u64,
    /// Content of the matching line.
    line_content: String,
    /// Byte offsets of the matched portion within `line_content`,
    /// as `(start, end)` pairs (for highlighting).
    match_byte_offsets: Vec<(u32, u32)>,
}

/// State for the global search (Cmd+Shift+F) panel.
#[derive(Debug, Clone)]
struct GlobalSearchState {
    /// Current query text in the search input.
    query: String,
    /// Search results (empty when no search has been performed).
    results: Vec<OwnedGrepMatch>,
    /// Index of the currently selected result in the list.
    selected_index: usize,
    /// Current search status.
    status: GlobalSearchStatus,
    /// Generation counter for stale-result prevention.
    search_gen: u64,
}

use std::ops::Range;

/// Compute byte-range matches of `query` in `text`. Returns empty
/// when query is shorter than 2 characters.
///
/// When `case_sensitive` is `false`, matching uses ASCII-only case
/// folding via [`str::to_ascii_lowercase`] — this is length-preserving
/// so returned byte ranges are valid for the original `text`.
/// Non-ASCII queries are matched literally in case-insensitive mode
/// (standard editor convention).
#[must_use]
fn compute_text_matches(text: &str, query: &str, case_sensitive: bool) -> Vec<Range<usize>> {
    if query.len() < 2 {
        return Vec::new();
    }
    let mut matches = Vec::new();

    if case_sensitive {
        let mut offset = 0;
        while let Some(pos) = text[offset..].find(query) {
            let abs_start = offset + pos;
            let abs_end = abs_start + query.len();
            matches.push(abs_start..abs_end);
            offset = abs_end;
        }
    } else {
        // Case-insensitive: lowercase both strings (ASCII-only, length-preserving).
        let text_lower = text.to_ascii_lowercase();
        let query_lower = query.to_ascii_lowercase();
        let mut offset = 0;
        while let Some(pos) = text_lower[offset..].find(&query_lower) {
            let abs_start = offset + pos;
            let abs_end = abs_start + query.len();
            matches.push(abs_start..abs_end);
            offset = abs_end;
        }
    }

    matches
}

/// Direction for navigating between find matches.
enum FindDirection {
    Next,
    Prev,
}

/// Direction for navigating the file tree, global search results, or quick-open list.
enum TreeNavDirection {
    Up,
    Down,
}

/// Direction for switching tabs.
enum TabDirection {
    Next,
    Prev,
}

/// Data returned from the async file load operation.
#[derive(Debug, Clone)]
pub struct FileLoadData {
    path: String,
    text: String,
    has_trailing_newline: bool,
    line_ending: LineEnding,
}

/// What to do with a dirty tab when closing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseAction {
    Save,
    Discard,
    Cancel,
}

/// Raw data loaded from a saved tab entry (string content, not Content).
#[derive(Debug, Clone)]
pub struct SavedTabData {
    file_path: String,
    text: String,
    was_dirty: bool,
    has_trailing_newline: bool,
    line_ending: LineEnding,
    /// Whether this tab was the active one when saved.
    is_active: bool,
}

// ── Messages ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum EditorMessage {
    /// Workspace selected via the Home page picker (name, optional filesystem path).
    WorkspaceSelected(String, Option<String>),
    /// A directory's listing was loaded from the filesystem.
    DirExpanded {
        dir_path: String,
        r#gen: u64,
        entries: Result<Vec<FsEntry>, String>,
        /// When `true`, errors are silently logged instead of shown as a toast.
        /// Used by background (periodic/manual) refresh to avoid noise.
        quiet: bool,
    },
    /// User toggled a directory in the file tree.
    ToggleDir(String),
    /// User selected a file in the file tree.
    SelectFile(String),
    /// Ctrl+B toggled tree keyboard focus on/off.
    TreeFocusToggled,
    /// Arrow Up in tree keyboard navigation.
    TreeNavUp,
    /// Arrow Down in tree keyboard navigation.
    TreeNavDown,
    /// Enter key in tree navigation — open file or expand directory.
    TreeNavEnter,
    /// Arrow Left in tree navigation — collapse directory or go to parent.
    TreeNavLeft,
    /// Arrow Right in tree navigation — expand directory or go to first child.
    TreeNavRight,
    /// Scroll position changed in the tree panel. First element is the
    /// absolute vertical scroll offset, second is the visible viewport height.
    TreeScrolled(f32, f32),
    /// Escape key — dismiss find bar, go-to-line, quick open, tree focus, or close dialog.
    Escape,
    /// A file's contents were loaded from disk.
    FileLoaded {
        path: String,
        r#gen: u64,
        result: Result<FileLoadData, String>,
    },
    /// Saved tabs were loaded from the database with file contents.
    SavedTabsLoaded {
        tabs_data: Vec<SavedTabData>,
        r#gen: u64,
    },
    /// User selected an existing tab.
    TabSelected(usize),
    /// User closed a tab.
    TabClosed(usize),
    /// User performed an editing action in the text editor.
    EditorAction(super::editor_widget::EditorAction),
    /// User requested to save the active tab.
    SaveActiveTab,
    /// Result of a save operation.
    SaveResult {
        path: String,
        result: Result<(), String>,
        /// Hash of the content that was written to disk, so we can
        /// update `TabData::saved_text_hash` for undo/redo comparison.
        saved_hash: u64,
    },
    /// User interacted with the close-dirty-tab dialog.
    CloseDialog {
        tab_index: usize,
        action: CloseAction,
    },
    /// User interacted with the close-others dirty-tab dialog.
    CloseOthersDialog {
        keep_idx: usize,
        action: CloseAction,
    },
    /// Periodic tick — refreshes git status and gitignore for file tree coloring.
    Tick,
    /// Fast tick (100 ms) — keeps the editor cursor blinking.
    BlinkTick,
    /// Git status has been loaded for the current workspace's file tree.
    GitStatusLoaded(Result<HashMap<String, GitFileStatus>, String>),
    /// Git ignore status has been loaded for the current workspace's file tree.
    GitIgnoredLoaded(Result<HashSet<String>, String>),
    /// Toast message to show.
    Toast(super::ToastMessage),
    /// Undo the last edit.
    Undo,
    /// Redo a previously undone edit.
    Redo,
    /// Open/toggle the find/replace bar.
    FindToggle,
    /// Search query text changed.
    FindQueryInput(String),
    /// Replace text changed.
    FindReplaceInput(String),
    /// Navigate to the next match.
    FindNext,
    /// Navigate to the previous match.
    FindPrev,
    /// Replace the current match with the replace text.
    FindReplace,
    /// Replace all matches.
    FindReplaceAll,
    /// Toggle case-sensitive matching.
    FindToggleCaseSensitivity,
    /// Manual or periodic refresh of all expanded directory listings from disk.
    /// Also triggers a git status refresh so newly discovered files get colors.
    RefreshFileTree,
    /// Close all tabs except the given index.
    CloseOtherTabs(usize),
    /// Periodic check (every 300 ms) for external file changes on the active tab.
    /// Only fires when a workspace is selected.
    CheckFileChanges,
    /// A file was reloaded after being detected as changed on disk.
    /// The cursor position was captured *before* the async read and should
    /// be restored (clamped to new file bounds) on success.
    FileReloaded {
        /// Path of the reloaded file.
        path: String,
        /// Ok(text) on success, Err(msg) on failure.
        result: Result<String, String>,
        /// Cursor line before reload (preserved, clamped to new bounds).
        cursor_line: usize,
        /// Cursor column before reload (preserved, clamped to new bounds).
        cursor_col: usize,
    },
    /// Toggle the go-to-line input bar.
    GoToLineToggle,
    /// Input text for the go-to-line bar.
    GoToLineInput(String),
    /// Jump to the entered line number.
    GoToLineGo,
    /// Toggle the quick-open file picker.
    QuickOpenToggle,
    /// Filter text for the quick-open file picker.
    QuickOpenInput(String),
    /// Select a file from the quick-open list by index.
    QuickOpenSelect(usize),
    /// Switch to the next tab (Ctrl+Tab).
    TabSwitchNext,
    /// Switch to the previous tab (Ctrl+Shift+Tab).
    TabSwitchPrev,
    /// Close the active tab (Ctrl+W).
    CloseActiveTab,
    /// Toggle the global search panel (Cmd+Shift+F / Ctrl+Shift+F).
    GlobalSearchToggle,
    /// Query text changed in the global search input.
    GlobalSearchInput(String),
    /// Results returned from the async global search.
    GlobalSearchResults {
        /// Generation counter for stale-result prevention.
        r#gen: u64,
        /// Owned grep match results.
        results: Vec<OwnedGrepMatch>,
        /// Error message if the search failed.
        error: Option<String>,
    },
    /// A result was clicked or selected in the global search list.
    GlobalSearchSelect(usize),
    /// Close the global search panel.
    GlobalSearchClose,
    // ── Context menu actions ────────────────────────────────────────
    /// Context menu: delete a file (shows confirmation dialog).
    DeleteFileRequested(String),
    /// Context menu: delete a directory (shows confirmation dialog).
    DeleteDirectoryRequested(String),
    /// Context menu: create a new file in the given parent directory.
    NewFileRequested(String),
    /// Context menu: create a new directory in the given parent directory.
    NewDirectoryRequested(String),
    /// Context menu: reveal the path in the system file manager.
    RevealInFinder(String),
    /// Context menu: copy a relative path to clipboard.
    CopyRelativePath(String),
    /// Context menu: copy an absolute path to clipboard.
    CopyAbsolutePath(String),
    /// User confirmed the delete operation.
    ConfirmDelete,
    /// User cancelled the delete dialog.
    CancelDelete,
    /// User submitted a name for a new file or directory.
    NewItemSubmit(String),
    /// User changed the new-item name input.
    NewItemInput(String),
    /// Internal: reveal-in-finder operation completed (no-op).
    RevealDone,
    // ── Inline rename ───────────────────────────────────────────
    /// Context menu: rename a file or directory (starts inline rename).
    RenameRequested(String),
    /// User changed the rename input text.
    RenameInput(String),
    /// User submitted the rename (Enter pressed in inline input).
    RenameSubmit,
    /// User cancelled the inline rename.
    RenameCancel,
    /// Async rename operation completed.
    RenameCompleted {
        /// Old relative path (workspace-relative).
        old_path: String,
        /// New relative path (workspace-relative).
        new_path: String,
        /// Whether the renamed item was a directory.
        is_dir: bool,
        /// Result of the filesystem rename.
        result: Result<(), String>,
        /// Re-read parent directory entries.
        dir_entries: Result<Vec<FsEntry>, String>,
        /// Generation counter for the parent directory's `dir_generations`
        /// slot.  Used for stale-result prevention via the standard
        /// generation invalidation protocol (see `dir_expanded`).
        rename_gen: u64,
    },
}

// ── Context menu types ──────────────────────────────────────────

/// Target for the delete confirmation dialog.
#[derive(Debug, Clone)]
struct DeleteConfirmTarget {
    /// Full path (relative to workspace root).
    path: String,
    /// Whether this is a directory.
    is_dir: bool,
    /// Number of dirty tabs that would be affected (directory deletes only).
    dirty_tab_count: usize,
    /// Absolute path for filesystem operations.
    abs_path: String,
}

/// Target for the new file/directory name input.
#[derive(Debug, Clone)]
struct NewItemTarget {
    /// Parent directory path (relative to workspace root; empty = root).
    parent_dir: String,
    /// Whether to create a directory (vs a file).
    is_dir: bool,
    /// Absolute path of the parent directory.
    abs_parent: String,
    /// Absolute path of the workspace root.
    ws_root: String,
    /// Current input text.
    input_text: String,
}

/// Target for the inline rename operation.
#[derive(Debug, Clone)]
struct RenameTarget {
    /// Full path (relative to workspace root) of the item being renamed.
    path: String,
    /// Absolute path of the item being renamed.
    abs_path: String,
    /// Whether this is a directory.
    is_dir: bool,
    /// Absolute path of the workspace root.
    ws_root: String,
    /// Current input text (the new name being edited).
    input_text: String,
    /// Optional inline error message (e.g., "File already exists").
    error: Option<String>,
}

/// Style for the inline rename text input — transparent background, no border,
/// matching the appearance of the tree node label it replaces.
#[must_use]
fn rename_input_style(_theme: &iced::Theme, _status: text_input::Status) -> text_input::Style {
    text_input::Style {
        background: iced::Background::Color(iced::Color::TRANSPARENT),
        border: iced::Border {
            radius: 0.0.into(),
            width: 0.0,
            color: iced::Color::TRANSPARENT,
        },
        icon: theme::TEXT_MUTED,
        placeholder: theme::TEXT_MUTED,
        value: theme::TEXT_PRIMARY,
        selection: theme::ACCENT_DIM,
    }
}

/// Validate a user-supplied file/directory name for new-item or rename operations.
///
/// Returns `Some(error_message)` when the name is invalid, `None` when it passes
/// all checks.  Used by both [`NewItemSubmit`] and [`RenameSubmit`] to avoid
/// duplicating the common validation rules.
///
/// Checks performed:
/// - Empty (or all-whitespace) name
/// - Path separators (`/`, `\`, NUL)
/// - Reserved path components (`.`, `..`)
/// - OS-reserved names (CON, NUL, PRN, AUX, COM1–COM9, LPT1–LPT9) — Windows only
#[must_use]
fn validate_item_name(name: &str) -> Option<&'static str> {
    if name.is_empty() {
        return Some("Name cannot be empty");
    }
    if name.contains('/') || name.contains('\\') || name.contains('\0') {
        return Some("Name cannot contain path separators");
    }
    if name == "." || name == ".." {
        return Some("Invalid name");
    }
    #[cfg(target_os = "windows")]
    {
        let reserved = [
            "con", "nul", "prn", "aux", "com1", "com2", "com3", "com4", "com5", "com6", "com7",
            "com8", "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
        ];
        let stem = name.split('.').next().unwrap_or(name);
        if reserved.contains(&stem.to_lowercase().as_str()) {
            return Some("Name is reserved by the operating system");
        }
    }
    None
}

// ── Helpers — prefix-based collection re-keying ───────────────────

/// Given a key starting with `old_prefix`, compute the new key with
/// `new_prefix` substituted for the prefix portion.  The caller must
/// already have verified that the key starts with `old_prefix` (via
/// `starts_with` or equivalent) — this function uses `unwrap_or("")`
/// as a safe fallback but assumes the prefix match.
fn rekey_compute_new_key(key: &str, old_prefix: &str, new_prefix: &str) -> String {
    let rest = key.strip_prefix(old_prefix).unwrap_or("");
    if rest.is_empty() {
        new_prefix.to_string()
    } else {
        format!("{new_prefix}/{rest}")
    }
}

/// Collect all keys matching `old_prefix` and compute their new key with
/// `new_prefix` substituted.  Returns a vec of `(old_key, new_key)` pairs.
/// Used by [`rekey_map_prefix`] and [`rekey_set_prefix`] to avoid
/// duplicating the filter-and-collect logic.
fn rekey_keys(
    old_prefix: &str,
    new_prefix: &str,
    keys: impl IntoIterator<Item = String>,
) -> Vec<(String, String)> {
    keys.into_iter()
        .filter(|k| k.starts_with(old_prefix))
        .map(|k| {
            let new_key = rekey_compute_new_key(&k, old_prefix, new_prefix);
            (k, new_key)
        })
        .collect()
}

/// Re-key entries in a `HashMap<String, V>` whose keys start with
/// `old_prefix` to use `new_prefix` instead.  Each value passes through
/// `modify` before re-insertion (use `|_| {}` when no modification is
/// needed).  The old prefix should include a trailing separator (e.g.
/// `"old_dir/"`), and `rest` is the portion of the key after it; the
/// new key is `"{new_prefix}/{rest}"` (or just `new_prefix` when
/// `rest` is empty — i.e. when the key exactly equals `old_prefix`).
fn rekey_map_prefix<V>(
    map: &mut HashMap<String, V>,
    old_prefix: &str,
    new_prefix: &str,
    modify: impl Fn(&mut V),
) {
    let key_pairs = rekey_keys(old_prefix, new_prefix, map.keys().cloned());
    for (old_key, new_key) in key_pairs {
        if let Some(mut v) = map.remove(&old_key) {
            modify(&mut v);
            map.insert(new_key, v);
        }
    }
}

/// Re-key entries in a `HashSet<String>` whose keys start with
/// `old_prefix` to use `new_prefix` instead.  Same prefix conventions
/// as [`rekey_map_prefix`].
fn rekey_set_prefix(set: &mut HashSet<String>, old_prefix: &str, new_prefix: &str) {
    let key_pairs = rekey_keys(old_prefix, new_prefix, set.iter().cloned());
    for (old_key, new_key) in key_pairs {
        set.remove(&old_key);
        set.insert(new_key);
    }
}

/// Update the `full_path` of a single [`FsEntry`] by replacing `old_prefix`
/// with `new_prefix` when the path starts with `old_prefix`.  Used during
/// directory-rename migrations to keep `FsEntry` paths in sync with their
/// new directory key.
fn update_entry_path(entry: &mut FsEntry, old_prefix: &str, new_prefix: &str) {
    if entry.full_path.starts_with(old_prefix) {
        entry.full_path = rekey_compute_new_key(&entry.full_path, old_prefix, new_prefix);
    }
}

// ── Helpers — async I/O ──────────────────────────────────────────

/// Read a flat list of directory entries for a given path relative to the
/// workspace root. The `root` is the workspace's filesystem path; `rel_path`
/// is the subdirectory relative to root (empty string for root).
async fn read_directory_entries(root: &str, rel_path: &str) -> Result<Vec<FsEntry>, String> {
    let dir_path = if rel_path.is_empty() {
        root.to_string()
    } else {
        let p = Path::new(root).join(rel_path);
        p.to_string_lossy().to_string()
    };
    let mut entries = match tokio::fs::read_dir(&dir_path).await {
        Ok(rd) => rd,
        Err(e) => return Err(format!("Failed to read directory '{rel_path}': {e}")),
    };

    let mut result: Vec<FsEntry> = Vec::new();
    let mut dirs: Vec<FsEntry> = Vec::new();
    let mut files: Vec<FsEntry> = Vec::new();

    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().to_string();
        // Filter out .git directory — it's not a user-editable file.
        if name == ".git" {
            continue;
        }
        // Filter out OS-generated metadata files.
        if is_os_file(&name) {
            continue;
        }
        let full_path = if rel_path.is_empty() {
            name.clone()
        } else {
            format!("{rel_path}/{name}")
        };
        // Use tokio::fs::metadata() on the absolute path to follow symlinks.
        // DirEntry::file_type() does NOT traverse symlinks (per Rust docs).
        let abs_path = Path::new(&dir_path).join(&name);
        let (is_dir, err) = match tokio::fs::metadata(&abs_path).await {
            Ok(m) => (m.is_dir(), None),
            Err(e) => (false, Some(format!("{e}"))),
        };
        let fs_entry = FsEntry {
            name,
            full_path,
            is_dir: is_dir && err.is_none(),
            error: err,
        };
        if fs_entry.is_dir {
            dirs.push(fs_entry);
        } else {
            files.push(fs_entry);
        }
    }

    // Sort: directories first, then files, alphabetical within each group.
    dirs.sort_by_key(|e| e.name.to_lowercase());
    files.sort_by_key(|e| e.name.to_lowercase());
    result.extend(dirs);
    result.extend(files);
    Ok(result)
}

/// Validate file content for size and binary content.
///
/// Returns `Ok(())` if the bytes pass size and null-byte checks,
/// or `Err` with a user-facing error message.
fn validate_file_content(bytes: &[u8]) -> Result<(), String> {
    if (bytes.len() as u64) > MAX_FILE_SIZE {
        return Err(format!(
            "File too large ({} bytes, max {MAX_FILE_SIZE})",
            bytes.len()
        ));
    }
    if bytes.contains(&0) {
        return Err("Binary file detected (contains null bytes)".to_string());
    }
    Ok(())
}

/// Load a file's contents from disk with detection of indent style, line
/// ending, and trailing newline.
async fn load_file_data(full_path: String, r#gen: u64) -> EditorMessage {
    // Check file size first.
    let metadata = match tokio::fs::metadata(&full_path).await {
        Ok(m) => m,
        Err(e) => {
            return EditorMessage::FileLoaded {
                path: full_path,
                r#gen,
                result: Err(format!("Cannot read file metadata: {e}")),
            };
        }
    };
    if metadata.len() > MAX_FILE_SIZE {
        return EditorMessage::FileLoaded {
            path: full_path,
            r#gen,
            result: Err(format!(
                "File too large ({} bytes, max {MAX_FILE_SIZE})",
                metadata.len()
            )),
        };
    }

    let bytes = match tokio::fs::read(&full_path).await {
        Ok(b) => b,
        Err(e) => {
            return EditorMessage::FileLoaded {
                path: full_path,
                r#gen,
                result: Err(format!("Cannot read file: {e}")),
            };
        }
    };

    // Size and binary content validation.
    if let Err(e) = validate_file_content(&bytes) {
        return EditorMessage::FileLoaded {
            path: full_path,
            r#gen,
            result: Err(e),
        };
    }
    // UTF-8 validation for binary detection.
    let Ok(text) = String::from_utf8(bytes) else {
        return EditorMessage::FileLoaded {
            path: full_path,
            r#gen,
            result: Err("Binary file detected (invalid UTF-8)".to_string()),
        };
    };

    let data = FileLoadData {
        path: full_path,
        has_trailing_newline: has_trailing_newline(&text),
        line_ending: detect_line_ending(&text),
        text,
    };
    EditorMessage::FileLoaded {
        path: data.path.clone(),
        r#gen,
        result: Ok(data),
    }
}

/// Spawn a file load from disk and return a `Task` that produces
/// `EditorMessage::FileLoaded` when the data is ready.
///
/// This is extracted as a free function because the two callers
/// (`open_file_in_editor` and `select_file`) have different
/// generation-bumping strategies but share the same spawn logic.
fn spawn_file_load(abs_path: String, file_gen: u64) -> Task<EditorMessage> {
    Task::perform(load_file_data(abs_path, file_gen), |msg| msg)
}

/// Build the `EditorTabRecord` list from the current tab state.
#[must_use]
fn build_tab_records(
    tabs: &[Tab],
    active_index: usize,
    tab_contents: &HashMap<String, TabData>,
) -> Vec<crate::workspace::EditorTabRecord> {
    tabs.iter()
        .enumerate()
        .map(|(i, t)| crate::workspace::EditorTabRecord {
            file_path: t.path.clone(),
            tab_order: i,
            is_active: i == active_index,
            is_dirty: t.is_dirty,
            dirty_content: if t.is_dirty {
                tab_contents.get(&t.path).map(|d| d.content.text())
            } else {
                None
            },
        })
        .collect()
}

/// Save current tabs to the database for a workspace.
///
/// Checks `gen_counter` for staleness before writing (pre-write guard): if a
/// newer save has superseded this one, the DB write is skipped.  The write is
/// fire-and-forget — completion is not tracked since the pre-write guard is the
/// only staleness protection needed.
fn save_tabs_to_db(
    workspace_name: String,
    records: Vec<crate::workspace::EditorTabRecord>,
    save_gen: u64,
    gen_counter: Arc<AtomicU64>,
) -> Task<EditorMessage> {
    tokio::spawn(async move {
        if gen_counter.load(Ordering::Acquire) != save_gen {
            return;
        }
        let store = crate::workspace::store();
        if let Err(e) = store.save_editor_tabs(&workspace_name, &records).await {
            tracing::warn!("Failed to save editor tabs: {e}");
        }
    });
    Task::none()
}

/// Save a single file to disk (async).
async fn save_file_to_disk(
    path: String,
    content: String,
    line_ending: LineEnding,
) -> Result<(), String> {
    // Normalize to LF first to handle mixed line endings safely.
    let lf = content.replace("\r\n", "\n");
    let normalized = if line_ending == LineEnding::Crlf {
        lf.replace('\n', "\r\n")
    } else {
        lf
    };

    tokio::fs::write(&path, &normalized)
        .await
        .map_err(|e| format!("Failed to write file: {e}"))
}

/// Build a `Task` that saves the tab at `idx` to disk asynchronously.
///
/// # Panics
///
/// Panics in debug builds if `idx` is out of bounds (`idx >= tabs.len()`).
/// All current callers validate the index before calling.
fn build_save_task(
    tabs: &[Tab],
    tab_contents: &HashMap<String, TabData>,
    idx: usize,
) -> Task<EditorMessage> {
    debug_assert!(
        idx < tabs.len(),
        "idx {idx} out of bounds (len {})",
        tabs.len()
    );
    let path = tabs[idx].path.clone();
    let line_ending = tabs[idx].line_ending;
    let Some(content) = tab_contents.get(&path).map(|d| d.content.text()) else {
        tracing::error!(
            ?path,
            idx,
            "build_save_task: path not found in tab_contents — invariant violation"
        );
        return Task::perform(
            async move {
                EditorMessage::SaveResult {
                    path,
                    result: Err("Internal error: file content not found for save".into()),
                    saved_hash: 0,
                }
            },
            |msg| msg,
        );
    };
    let saved_hash = hash_text(&content);
    Task::perform(
        async move {
            let result = save_file_to_disk(path.clone(), content, line_ending).await;
            EditorMessage::SaveResult {
                path,
                result,
                saved_hash,
            }
        },
        |msg| msg,
    )
}

/// Update the dirty flag for a tab by comparing current text hash against
/// the saved hash.  A free function (not a `&mut self` method) so callers
/// can avoid borrow-checker conflicts from simultaneous mutable borrows
/// of `self.tabs` and immutable borrows of `self.tab_contents`.
fn update_dirty_flag(
    tabs: &mut [Tab],
    tab_contents: &HashMap<String, TabData>,
    idx: usize,
    path: &str,
) {
    if let (Some(tab), Some(tab_data)) = (tabs.get_mut(idx), tab_contents.get(path)) {
        let current_hash = hash_text(&tab_data.content.text());
        tab.is_dirty = current_hash != tab_data.saved_text_hash;
    }
}

// ── Helpers — tree building ──────────────────────────────────────

/// Recursively build a hierarchical tree from flat directory entries.
/// Only expanded directories have their children populated.
fn build_hierarchical_tree(
    dir_entries: &HashMap<String, Vec<FsEntry>>,
    expanded_dirs: &HashSet<String>,
    parent_path: &str,
) -> Vec<widgets::TreeNode> {
    let Some(entries) = dir_entries.get(parent_path) else {
        return Vec::new();
    };

    let mut nodes: Vec<widgets::TreeNode> = entries
        .iter()
        .map(|entry| {
            let mut node = widgets::TreeNode {
                name: entry.name.clone(),
                full_path: entry.full_path.clone(),
                is_dir: entry.is_dir,
                children: Vec::new(),
                error: entry.error.clone(),
            };
            if node.is_dir && expanded_dirs.contains(&node.full_path) {
                node.children =
                    build_hierarchical_tree(dir_entries, expanded_dirs, &node.full_path);
            }
            node
        })
        .collect();

    FileTree::sort_nodes(&mut nodes);
    nodes
}

// ── Git status utilities ──────────────────────────────────────────

/// Parse `git status --porcelain` output into a map of file path → `GitFileStatus`.
///
/// The porcelain format uses a two-column status (index + worktree).
/// Precedence: modified > added > untracked. Rename entries (`R`) extract
/// the new path after ` -> `. Deleted files (`D`) are ignored. Handles
/// git's C-style quoting for paths with special characters.
fn parse_git_status_porcelain(output: &str) -> HashMap<String, GitFileStatus> {
    let mut map: HashMap<String, GitFileStatus> = HashMap::new();

    for line in output.lines() {
        let trimmed = line.trim_end();
        if trimmed.len() < 2 {
            continue;
        }

        let chars: Vec<char> = trimmed.chars().take(2).collect();
        if chars.len() < 2 {
            continue;
        }

        let ix = chars[0];
        let wt = chars[1];

        // Skip deleted files — they don't appear in the working tree.
        if ix == 'D' || wt == 'D' {
            continue;
        }

        // Rename entries: "R  old_path -> new_path"
        // Both paths may be individually C-quoted by git. The separator
        // ` -> ` appears at the boundary between the two paths.
        if ix == 'R' {
            let rest = &trimmed[2..];
            let rest = rest.trim_start();
            let new_path: String = if rest.starts_with('"') {
                // Both paths are quoted. The boundary is `" -> "`.
                // rsplit_once on `" -> "` strips the opening quote of the
                // new path — add it back so `unquote_c_style` can strip both.
                if let Some((_, tail)) = rest.rsplit_once("\" -> \"") {
                    format!("\"{tail}")
                } else {
                    // Malformed — skip.
                    continue;
                }
            } else {
                // Unquoted paths: split on ` -> `.
                if let Some((_, tail)) = rest.rsplit_once(" -> ") {
                    tail.to_string()
                } else {
                    continue;
                }
            };
            if let Some(unquoted) = unquote_c_style(&new_path) {
                map.insert(unquoted, GitFileStatus::Modified);
            }
            continue;
        }

        // Extract path: strip first 2 chars (status columns) and leading space.
        let path = trimmed[2..].trim_start();
        if path.is_empty() {
            continue;
        }

        let status = if ix == 'M' || wt == 'M' {
            GitFileStatus::Modified
        } else if ix == 'A' || wt == 'A' || (ix == '?' && wt == '?') {
            GitFileStatus::Added
        } else {
            // Clean or ignored — don't store.
            continue;
        };

        let Some(path) = unquote_c_style(path) else {
            continue;
        };
        // Strip trailing slash — git appends '/' for untracked directories
        // (e.g., `?? new_dir/`), but tree node full_path has no trailing slash.
        let path = path.strip_suffix('/').unwrap_or(&path).to_string();

        // For entries with multiple lines referencing the same file (e.g., staged +
        // unstaged), keep the most "interesting" status: Modified > Added.
        let entry = map.entry(path).or_insert(status);
        if status == GitFileStatus::Modified && *entry == GitFileStatus::Added {
            *entry = GitFileStatus::Modified;
        }
    }

    map
}

/// Load git status for a workspace. Returns an empty map if the workspace
/// is not a git repo or if git is not installed.
async fn load_git_status(workspace_path: String) -> Result<HashMap<String, GitFileStatus>, String> {
    let ws_path = Path::new(&workspace_path);
    if !is_git_repo(ws_path) {
        tracing::debug!("Workspace '{workspace_path}' is not a git repo — skipping git status");
        return Ok(HashMap::new());
    }

    let output = run_git_status(ws_path).await?;
    Ok(parse_git_status_porcelain(&output))
}

/// Collect all file and directory paths from the tree recursively.
fn collect_tree_paths(nodes: &[widgets::TreeNode]) -> Vec<String> {
    let mut paths = Vec::new();
    for node in nodes {
        paths.push(node.full_path.clone());
        if node.is_dir {
            paths.extend(collect_tree_paths(&node.children));
        }
    }
    paths
}

/// Load git ignore status for the given tree paths.
/// Handles workspaces that are subdirectories of a git repo by detecting
/// the repo root and adjusting paths accordingly.
/// Returns an empty set if the workspace is not in a git repo.
async fn load_git_ignore(
    workspace_path: String,
    tree_paths: Vec<String>,
) -> Result<HashSet<String>, String> {
    if tree_paths.is_empty() {
        return Ok(HashSet::new());
    }

    let ws_path = Path::new(&workspace_path);

    // Find the git repo root (handles subdirectory-of-repo workspaces).
    let output = run_git_output(ws_path, &["rev-parse", "--show-toplevel"])
        .await
        .map_err(|e| format!("Failed to run git rev-parse: {e}"))?;

    if !output.status.success() {
        tracing::debug!(
            "Workspace '{workspace_path}' is not in a git repo — skipping git ignore check"
        );
        return Ok(HashSet::new());
    }

    let repo_root = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let repo_path = Path::new(&repo_root);

    // Compute the relative prefix from repo root to workspace.
    // When workspace is the repo root itself, prefix is empty.
    let ws_canonical = ws_path
        .canonicalize()
        .map_err(|e| format!("Failed to canonicalize workspace path: {e}"))?;

    let prefix = ws_canonical
        .strip_prefix(repo_path)
        .map_err(|e| format!("Workspace is not inside git repo: {e}"))?;

    let prefix_empty = prefix.as_os_str().is_empty();

    // Adjust tree paths to be repo-root-relative for git check-ignore.
    let adjusted_paths: Vec<String> = if prefix_empty {
        tree_paths
    } else {
        let prefix_str = prefix.to_string_lossy();
        tree_paths
            .iter()
            .map(|p| format!("{prefix_str}/{p}"))
            .collect()
    };

    let raw_ignored = run_git_check_ignore(repo_path, &adjusted_paths).await?;

    // Strip the prefix back to get workspace-relative paths for the cache.
    let ignored: HashSet<String> = if prefix_empty {
        raw_ignored
    } else {
        let prefix_str = prefix.to_string_lossy();
        let prefix_slash = format!("{prefix_str}/");
        raw_ignored
            .into_iter()
            .filter_map(|p| {
                p.strip_prefix(&prefix_slash)
                    .or_else(|| (p == prefix_str).then_some(""))
                    .map(ToString::to_string)
            })
            .collect()
    };

    Ok(ignored)
}

// ── Global search helpers ──────────────────────────────────────────

/// Run a global (find-in-files) grep search with debounce.
///
/// 1. Debounce: waits 300ms (cancelled if a newer generation supersedes).
/// 2. Initialises the search engine if needed.
/// 3. Runs `picker.grep()` on the blocking thread pool.
/// 4. Extracts owned data from `GrepResult` while holding the picker lock.
#[allow(clippy::too_many_lines)]
async fn run_global_search(
    ws_path: String,
    ws_name: String,
    query: String,
    gs_gen: u64,
) -> EditorMessage {
    // Step 1: Debounce.
    tokio::time::sleep(Duration::from_millis(GLOBAL_SEARCH_DEBOUNCE_MS)).await;

    // Step 2: Get or init the search engine.
    let entry =
        match crate::search_engine::get_or_init_engine(&ws_name, std::path::Path::new(&ws_path)) {
            Ok(e) => e,
            Err(e) => {
                return EditorMessage::GlobalSearchResults {
                    r#gen: gs_gen,
                    results: Vec::new(),
                    error: Some(e),
                };
            }
        };
    if let Err(e) = crate::search_engine::ensure_scanned(&entry).await {
        return EditorMessage::GlobalSearchResults {
            r#gen: gs_gen,
            results: Vec::new(),
            error: Some(format!("Search engine not ready: {e}")),
        };
    }

    // Step 3: Run grep on the blocking thread pool.
    let entry_for_blocking = Arc::clone(&entry);
    let query_for_blocking = query.clone();
    let base_path = ws_path.clone();

    let result = tokio::task::spawn_blocking(move || {
        let guard = entry_for_blocking.picker.read().unwrap();
        let Some(picker) = guard.as_ref() else {
            return (
                Vec::new(),
                Some("Search engine not initialized.".to_string()),
            );
        };

        let fff_query = parse_grep_query(&query_for_blocking);
        let grep_opts = GrepSearchOptions {
            mode: GrepMode::PlainText,
            smart_case: true,
            max_file_size: MAX_FILE_SIZE,
            max_matches_per_file: GLOBAL_SEARCH_MATCHES_PER_FILE,
            file_offset: 0,
            page_limit: MAX_GLOBAL_SEARCH_RESULTS,
            time_budget_ms: 3_000,
            before_context: 0,
            after_context: 0,
            classify_definitions: false,
            trim_whitespace: true,
            abort_signal: None,
        };

        let grep_result = picker.grep(&fff_query, &grep_opts);

        if grep_result.matches.is_empty() {
            return (Vec::new(), None);
        }

        // Step 4: Extract owned data while holding the picker lock.
        let base = Path::new(&base_path);
        let mut owned: Vec<OwnedGrepMatch> = Vec::with_capacity(grep_result.matches.len());

        for m in &grep_result.matches {
            let file = grep_result.files[m.file_index];
            let rel_path = file.relative_path(picker);
            let abs_path = base.join(&rel_path).to_string_lossy().to_string();
            let offsets: Vec<(u32, u32)> =
                m.match_byte_offsets.iter().map(|&(s, e)| (s, e)).collect();

            owned.push(OwnedGrepMatch {
                abs_path,
                rel_path,
                line_number: m.line_number,
                line_content: m.line_content.clone(),
                match_byte_offsets: offsets,
            });
        }

        (owned, None)
    })
    .await
    .unwrap_or((Vec::new(), Some("spawn_blocking join error".to_string())));

    let (owned_matches, error) = result;

    EditorMessage::GlobalSearchResults {
        r#gen: gs_gen,
        results: owned_matches,
        error,
    }
}

// ── Editor State ──────────────────────────────────────────────────

pub struct EditorState {
    /// Currently selected workspace name (set by the Home page picker via Dashboard).
    selected_workspace_name: Option<String>,
    /// Filesystem path for the currently selected workspace.
    selected_workspace_path: Option<String>,
    /// Monotonically increasing generation counter for stale-result prevention.
    generation: u64,
    /// Monotonically increasing generation counter for saved-tabs restoration,
    /// kept separate from `generation` to avoid collision with file loads and
    /// directory expansions that are dispatched concurrently.
    saved_tabs_gen: u64,
    /// Per-directory generation counters.
    dir_generations: HashMap<String, u64>,
    /// Per-file generation counters (prevents stale FileLoaded results).
    file_generations: HashMap<String, u64>,
    /// Directories currently being loaded.
    loading_dirs: HashSet<String>,
    /// Directory entries loaded from the filesystem (keyed by full path).
    dir_entries: HashMap<String, Vec<FsEntry>>,
    /// Shared file tree state (nodes, expanded dirs, focus, visible nodes, scroll ID).
    file_tree: FileTree,
    /// Currently selected file in the tree (full path).
    selected_file: Option<String>,
    /// Open tabs in order.
    tabs: Vec<Tab>,
    /// Index of the active tab.
    active_tab_index: usize,
    /// Content per tab (keyed by full filesystem path).
    tab_contents: HashMap<String, TabData>,
    /// Scrollable ID for the tab bar.
    tab_scroll_id: Id,
    /// When set, after the next successful save, close this tab (used by Save-from-close-dialog).
    pending_save_close: Option<usize>,
    /// Save queue for close-others: (keep_idx, remaining dirty indices to save).
    pending_close_others: Option<(usize, Vec<usize>)>,
    /// Whether the workspace tabs have been loaded at least once this session.
    session_initialized: bool,
    /// When Enter expands a directory that needs async loading, this holds
    /// the directory path so DirExpanded can advance focus to the first child.
    pending_enter_dir: Option<String>,
    /// Cached git status per file path (relative to workspace root).
    git_status_cache: HashMap<String, GitFileStatus>,
    /// Guard against concurrent git status refresh operations.
    git_status_loading: bool,
    /// Cached gitignored file/directory paths (relative to workspace root).
    git_ignore_cache: HashSet<String>,
    /// Guard against concurrent git ignore refresh operations.
    git_ignore_loading: bool,
    /// Monotonically incrementing blink generation counter.
    /// Incremented on each `BlinkTick` to force Iced to redraw the editor
    /// widget, keeping the cursor blink alive even if the `RedrawRequested`
    /// chain breaks.
    blink_gen: u64,
    /// Shared atomic counter used by async save tasks for pre-write staleness
    /// checking.  Written to on every save initiation; read by in-flight tasks
    /// to determine if a newer save has superseded them.
    tab_save_counter: Arc<AtomicU64>,
    /// Last-known modification time per open file (keyed by full path).
    /// Used by the auto-refresh poll to detect external file changes.
    file_mtimes: HashMap<String, SystemTime>,
    /// Paths for which a "file deleted" toast has already been shown.
    /// Prevents spamming the toast every 300 ms.
    deleted_file_toasted: HashSet<String>,
    /// Which modal overlay is currently open (None when no overlay is active).
    /// Enforced by the type system to be mutually exclusive — only one overlay
    /// may be open at a time.
    active_modal: Option<ModalKind>,
    /// Cached list of all workspace files for quick-open filtering.
    /// Populated on each quick-open toggle from currently expanded dirs.
    all_workspace_files: Vec<String>,
    /// Generation counter for global search stale-result prevention.
    global_search_gen: u64,
    /// When set, the next file load for this path+generation should jump to this line.
    /// The tuple is (abs_path, 1-based line_number, expected_file_gen). Consumed by the
    /// `FileLoaded` handler only when both path and generation match.
    pending_goto: Option<(String, usize, u64)>,
}

/// Identifies which modal overlay is currently open, in Escape-dismissal
/// priority order (GlobalSearch highest, CloseOthers lowest).
///
/// Each variant carries the state data for that overlay.
#[derive(Debug, Clone)]
enum ModalKind {
    GlobalSearch(GlobalSearchState),
    GotoLine(String),
    QuickOpen(QuickOpenState),
    Rename(RenameTarget),
    NewItem(NewItemTarget),
    DeleteConfirm(DeleteConfirmTarget),
    CloseDialog(usize),
    CloseOthers(usize),
}

/// State for the quick-open file picker.
#[derive(Debug, Clone)]
struct QuickOpenState {
    /// Current filter text.
    filter: String,
    /// Currently highlighted result index.
    selected_index: usize,
    /// Filtered file list (matching the filter text).
    results: Vec<String>,
}

impl EditorState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            selected_workspace_name: None,
            selected_workspace_path: None,
            generation: 0,
            saved_tabs_gen: 0,
            dir_generations: HashMap::new(),
            file_generations: HashMap::new(),
            loading_dirs: HashSet::new(),
            dir_entries: HashMap::new(),
            file_tree: FileTree::new(Id::new("editor_tree_panel")),
            selected_file: None,
            tabs: Vec::new(),
            active_tab_index: 0,
            tab_contents: HashMap::new(),
            tab_scroll_id: Id::new("editor_tabs_bar"),
            pending_save_close: None,
            pending_close_others: None,
            session_initialized: false,
            pending_enter_dir: None,
            git_status_cache: HashMap::new(),
            git_status_loading: false,
            git_ignore_cache: HashSet::new(),
            git_ignore_loading: false,
            blink_gen: 0,
            tab_save_counter: Arc::new(AtomicU64::new(0)),
            file_mtimes: HashMap::new(),
            deleted_file_toasted: HashSet::new(),
            active_modal: None,
            all_workspace_files: Vec::new(),
            global_search_gen: 0,
            pending_goto: None,
        }
    }

    /// The filesystem root of the currently selected workspace, if any.
    /// Always absolute — validated & canonicalized at creation time.
    #[inline]
    fn workspace_root(&self) -> Option<&str> {
        self.selected_workspace_path.as_deref()
    }

    /// Resolve a relative tree path to an absolute filesystem path.
    fn abs_path(&self, rel_path: &str) -> Option<String> {
        self.selected_workspace_path
            .as_ref()
            .map(|ws| Path::new(ws).join(rel_path).to_string_lossy().to_string())
    }

    /// Rebuild both the hierarchical tree and visible node list from `dir_entries`
    /// and `expanded_dirs`.  Callers that also need `tree_focused = true` must set
    /// it separately after this call.
    fn rebuild_tree(&mut self) {
        self.file_tree.nodes =
            build_hierarchical_tree(&self.dir_entries, &self.file_tree.expanded_dirs, "");
        self.file_tree.rebuild_visible();
    }

    /// Start an async load of a directory's entries.
    ///
    /// Returns `Some(Task)` with the async load if a workspace is selected and
    /// the directory needs loading (the caller is responsible for checking
    /// `!self.dir_entries.contains_key(dir_path)` before calling this).
    /// Returns `None` if no workspace is selected (caller should
    /// early-return `Task::none()`).
    ///
    /// After calling this, the caller MUST call [`Self::rebuild_tree`] (or
    /// equivalent) to reflect the expanded state.  If the caller needs focus
    /// advancement after the load completes (Enter/Right navigation), it should
    /// set `self.pending_enter_dir` after this call.
    fn load_dir_async(&mut self, dir_path: &str, label: &str) -> Option<Task<EditorMessage>> {
        debug_assert!(
            !self.dir_entries.contains_key(dir_path),
            "load_dir_async: caller must check !dir_entries.contains_key(dir_path) first"
        );
        let ws_path = if let Some(p) = self.selected_workspace_path.as_ref() {
            p.clone()
        } else {
            tracing::error!("{label} without workspace selected");
            return None;
        };
        let dir_gen = self.generation.wrapping_add(1);
        self.generation = dir_gen;
        self.dir_generations.insert(dir_path.to_string(), dir_gen);
        self.loading_dirs.insert(dir_path.to_string());
        let d_path = dir_path.to_string();
        Some(Task::perform(
            async move {
                let entries = read_directory_entries(&ws_path, &d_path).await;
                EditorMessage::DirExpanded {
                    dir_path: d_path,
                    r#gen: dir_gen,
                    entries,
                    quiet: false,
                }
            },
            |msg| msg,
        ))
    }

    /// Expand a directory and either start an async load or focus the first child.
    ///
    /// If the directory's entries are not yet cached, starts an async load and
    /// sets [`Self::pending_enter_dir`] so [`Self::dir_expanded`] can advance
    /// focus when data arrives.  If the entries are already cached (sync path),
    /// expands and immediately focuses the first child.
    ///
    /// Returns `Task::none()` if no workspace is selected (caller should
    /// propagate this return).
    fn expand_dir_and_focus(&mut self, path: &str, label: &str) -> Task<EditorMessage> {
        self.selected_file = None;
        self.file_tree.expanded_dirs.insert(path.to_string());

        let needs_async_load = !self.dir_entries.contains_key(path);

        if needs_async_load {
            let Some(task) = self.load_dir_async(path, label) else {
                return Task::none();
            };
            self.pending_enter_dir = Some(path.to_string());
            // Rebuild tree for the expanded-but-still-loading state.
            self.rebuild_tree();
            return task;
        }

        // Sync path — children are already cached.
        self.rebuild_tree();
        self.file_tree
            .expand_dir_and_focus_first_child::<EditorMessage>(path)
    }

    /// Collapse an expanded directory and keep keyboard focus on it.
    ///
    /// Removes `path` from [`expanded_dirs`], rebuilds the tree, and delegates
    /// focus-and-scroll management to [`FileTree::collapse_dir_and_keep_focus`].
    /// The caller is responsible for ensuring `expanded_dirs` contains `path`
    /// before calling.
    ///
    /// Note: [`FileTree::collapse_dir_and_keep_focus`] calls [`rebuild_visible`]
    /// internally, so this helper uses raw [`build_hierarchical_tree`] (not
    /// [`rebuild_tree`]) to avoid rebuilding the visible list twice.
    fn collapse_dir(&mut self, path: &str) -> Task<EditorMessage> {
        self.file_tree.expanded_dirs.remove(path);
        self.file_tree.nodes =
            build_hierarchical_tree(&self.dir_entries, &self.file_tree.expanded_dirs, "");
        self.file_tree
            .collapse_dir_and_keep_focus::<EditorMessage>(path)
    }

    /// Build an inline rename [`TextInput`] element for a tree node that is
    /// currently being renamed.  Returns [`None`] when the node is not the
    /// rename target, so callers can fall through to their normal label rendering.
    fn build_rename_input<'a>(
        &'a self,
        node: &'a widgets::TreeNode,
    ) -> Option<Element<'a, EditorMessage>> {
        let ModalKind::Rename(rt) = self.active_modal.as_ref()? else {
            return None;
        };
        if rt.path != node.full_path {
            return None;
        }
        let input: Element<'a, EditorMessage> = text_input("", &rt.input_text)
            .id(Id::from(format!("rename_input_{}", node.full_path)))
            .on_input(EditorMessage::RenameInput)
            .on_submit(EditorMessage::RenameSubmit)
            .size(12)
            .padding([0, 2])
            .style(rename_input_style)
            .into();
        // Only wrap in a Column when an inline error needs to be shown
        // below the input.  The common (no-error) case returns a bare
        // TextInput to keep widget nesting shallow.
        if let Some(ref err) = rt.error {
            Some(
                column![input, text(err).size(10).color(theme::STATUS_ERROR)]
                    .spacing(0)
                    .into(),
            )
        } else {
            Some(input)
        }
    }

    #[allow(clippy::too_many_lines)]
    pub fn subscription(&self) -> Subscription<EditorMessage> {
        let mut subs: Vec<Subscription<EditorMessage>> = Vec::new();
        if self.selected_workspace_name.is_some() {
            subs.push(
                iced::time::every(Duration::from_secs(TICK_INTERVAL_SECS))
                    .map(|_| EditorMessage::Tick),
            );
            // Periodic directory refresh — re-reads all expanded directories
            // to pick up external filesystem changes (git checkout, build
            // scripts, other editors).
            subs.push(
                iced::time::every(Duration::from_secs(DIR_REFRESH_INTERVAL_SECS))
                    .map(|_| EditorMessage::RefreshFileTree),
            );
            // Fast tick for cursor blinking — 100 ms ensures smooth blink.
            subs.push(
                iced::time::every(Duration::from_millis(100)).map(|_| EditorMessage::BlinkTick),
            );
            // Auto-refresh tick for detecting external file changes on the
            // active tab.  Only the active (visible) tab is polled; dirty
            // tabs (unsaved edits) are never auto-reloaded.
            subs.push(
                iced::time::every(Duration::from_millis(300))
                    .map(|_| EditorMessage::CheckFileChanges),
            );
        }
        // Always listen for keyboard events — tree navigation may be active.
        subs.push(keyboard::listen().filter_map(|event| {
            use keyboard::{Event, Key};
            let Event::KeyPressed {
                key,
                modifiers,
                physical_key,
                ..
            } = event
            else {
                return None;
            };
            let km = super::detect_keyboard_mods(modifiers);

            // Helper: match a Character key by its Latin equivalent.
            let latin = |target: char| -> bool { key.to_latin(physical_key) == Some(target) };

            // Ctrl+B / Cmd+B → toggle tree focus.
            if km.is_shortcut_platform_mod() && latin('b') {
                return Some(EditorMessage::TreeFocusToggled);
            }
            // Cmd+Shift+F / Ctrl+Shift+F → global search (find-in-files).
            // Must appear BEFORE the Cmd+F / Ctrl+F check so Cmd+Shift+F
            // doesn't also trigger FindToggle.
            if km.is_platform_mod && !km.altgr_active && modifiers.shift() && latin('f') {
                return Some(EditorMessage::GlobalSearchToggle);
            }
            // Cmd+F / Ctrl+F → toggle find/replace bar.
            // Guard: Cmd+Shift+F handled above, so !modifiers.shift() prevents
            // Cmd+Shift+F from also triggering FindToggle.
            if km.is_shortcut_platform_mod() && !modifiers.shift() && latin('f') {
                return Some(EditorMessage::FindToggle);
            }
            // Cmd+Z / Ctrl+Z → undo.  Check shift first so Cmd+Shift+Z / Ctrl+Shift+Z → redo.
            if km.is_shortcut_platform_mod() && latin('z') {
                if modifiers.shift() {
                    return Some(EditorMessage::Redo);
                }
                return Some(EditorMessage::Undo);
            }
            // Cmd+S / Ctrl+S → save.
            if km.is_shortcut_platform_mod() && latin('s') {
                return Some(EditorMessage::SaveActiveTab);
            }
            // Ctrl+Tab / Ctrl+Shift+Tab → switch tabs.
            // On macOS, modifiers.control() is used directly (not is_platform_mod)
            // since Cmd+Tab is captured by the OS for app switching.
            if modifiers.control() && matches!(key, Key::Named(keyboard::key::Named::Tab)) {
                return if modifiers.shift() {
                    Some(EditorMessage::TabSwitchPrev)
                } else {
                    Some(EditorMessage::TabSwitchNext)
                };
            }
            // Ctrl+W → close tab (all platforms). Cmd+W on macOS is typically
            // captured by the window manager to close the window, so we use
            // Ctrl+W consistently.
            if !km.altgr_active && modifiers.control() && latin('w') {
                return Some(EditorMessage::CloseActiveTab);
            }
            // Go-to-line: Cmd+L on macOS, Ctrl+G on other platforms.
            #[cfg(target_os = "macos")]
            {
                if modifiers.command() && !modifiers.control() && latin('l') {
                    return Some(EditorMessage::GoToLineToggle);
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                if !km.altgr_active && modifiers.control() && latin('g') {
                    return Some(EditorMessage::GoToLineToggle);
                }
            }
            // Quick open: Cmd+P / Ctrl+P
            if km.is_shortcut_platform_mod() && latin('p') {
                return Some(EditorMessage::QuickOpenToggle);
            }
            // Refresh file tree: Cmd+R / Ctrl+R
            if km.is_shortcut_platform_mod() && latin('r') {
                return Some(EditorMessage::RefreshFileTree);
            }
            // Find next/prev: Cmd+G / F3 → FindNext, Cmd+Shift+G / Shift+F3 → FindPrev
            // macOS uses Cmd+G; non-macOS uses Ctrl+G for go-to-line (already mapped),
            // so F3 and Shift+F3 serve as the cross-platform find shortcuts.
            #[cfg(target_os = "macos")]
            if modifiers.command() && !modifiers.control() && latin('g') {
                return if modifiers.shift() {
                    Some(EditorMessage::FindPrev)
                } else {
                    Some(EditorMessage::FindNext)
                };
            }
            // F3 / Shift+F3 (all platforms)
            if matches!(key, Key::Named(keyboard::key::Named::F3)) {
                return if modifiers.shift() {
                    Some(EditorMessage::FindPrev)
                } else {
                    Some(EditorMessage::FindNext)
                };
            }
            // Shift+Enter → previous match (for use in the find/replace bar;
            // no-op when find bar is closed — handler checks state).
            if modifiers.shift() && matches!(key, Key::Named(keyboard::key::Named::Enter)) {
                return Some(EditorMessage::FindPrev);
            }
            // Arrow key navigation: when quick-open is active, arrow keys
            // navigate the results list (handled in the update method by
            // checking quick_open state before tree focus).
            match &key {
                Key::Named(named) => match named {
                    keyboard::key::Named::ArrowUp => Some(EditorMessage::TreeNavUp),
                    keyboard::key::Named::ArrowDown => Some(EditorMessage::TreeNavDown),
                    keyboard::key::Named::ArrowLeft => Some(EditorMessage::TreeNavLeft),
                    keyboard::key::Named::ArrowRight => Some(EditorMessage::TreeNavRight),
                    keyboard::key::Named::Enter => Some(EditorMessage::TreeNavEnter),
                    _ => None,
                },
                _ => None,
            }
        }));
        Subscription::batch(subs)
    }

    /// Returns the active tab index, or `None` if there are no tabs open.
    const fn active_tab_idx(&self) -> Option<usize> {
        let idx = self.active_tab_index;
        if idx >= self.tabs.len() {
            None
        } else {
            Some(idx)
        }
    }

    /// Returns the `(index, path)` of the active tab, or `None` if no tab is open.
    fn active_tab(&self) -> Option<(usize, String)> {
        let idx = self.active_tab_idx()?;
        Some((idx, self.tabs[idx].path.clone()))
    }

    /// Returns `true` when the find/replace bar is open on the active tab.
    fn is_find_bar_open(&self) -> bool {
        self.active_tab_idx()
            .and_then(|idx| self.tabs.get(idx))
            .and_then(|tab| self.tab_contents.get(&tab.path))
            .and_then(|data| data.find_replace_state.as_ref())
            .is_some()
    }

    /// Returns the active modal overlay, if any.
    fn active_modal(&self) -> Option<&ModalKind> {
        self.active_modal.as_ref()
    }

    /// Returns `true` when a modal overlay owns keyboard focus
    /// and editor-wide shortcuts (undo, save, tab switch, etc.) must not run.
    fn modal_overlay_blocks_editor_shortcuts(&self) -> bool {
        self.active_modal().is_some()
    }

    /// Save editor tabs to the database for the currently selected workspace.
    ///
    /// Returns a task that performs the async DB write, or [`None`] if:
    /// - Tabs haven't been initialized yet this session
    /// - No workspace is selected
    ///
    /// Uses the shared atomic counter for stale-result prevention: the pre-write
    /// guard inside the async task checks whether a newer save has superseded
    /// this one before writing.
    pub(crate) fn try_save_current_tabs(&self) -> Option<Task<EditorMessage>> {
        if !self.session_initialized {
            return None;
        }
        let workspace_name = self.selected_workspace_name.as_ref()?;

        // Increment shared counter to invalidate any in-flight stale saves.
        let save_gen = self
            .tab_save_counter
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);

        let records = build_tab_records(&self.tabs, self.active_tab_index, &self.tab_contents);
        Some(save_tabs_to_db(
            workspace_name.clone(),
            records,
            save_gen,
            self.tab_save_counter.clone(),
        ))
    }

    /// Save editor tabs to the database, returning a fallback [`Task::none`] if
    /// the session isn't initialized or no workspace is selected.
    ///
    /// Most callers should use this wrapper; only use
    /// [`try_save_current_tabs`](Self::try_save_current_tabs) directly when you
    /// need to inspect whether a save was actually dispatched.
    pub(crate) fn save_current_tabs(&mut self) -> Task<EditorMessage> {
        self.try_save_current_tabs().unwrap_or(Task::none())
    }

    /// Scroll the tab bar to keep the active tab visible.
    #[allow(clippy::cast_precision_loss)]
    fn scroll_to_active_tab(&self) -> Task<EditorMessage> {
        let index = self.active_tab_index;
        let offset_x = index as f32 * ESTIMATED_TAB_WIDTH;
        iced::widget::operation::scroll_to(
            self.tab_scroll_id.clone(),
            iced::widget::operation::AbsoluteOffset {
                x: offset_x,
                y: 0.0,
            },
        )
    }

    /// Scroll to the tab at `new_idx` without saving tabs.
    /// Sets the active index and scrolls the tab bar, but does not persist.
    fn scroll_to_tab(&mut self, new_idx: usize) -> Task<EditorMessage> {
        if new_idx >= self.tabs.len() {
            return Task::none();
        }
        self.active_tab_index = new_idx;
        self.scroll_to_active_tab()
    }

    /// Switch to the tab at `idx`, updating active index, scrolling, and
    /// persisting the tab list to the database.
    fn switch_to_tab(&mut self, idx: usize) -> Task<EditorMessage> {
        if idx >= self.tabs.len() {
            return Task::none();
        }
        self.active_tab_index = idx;
        Task::batch(vec![self.scroll_to_active_tab(), self.save_current_tabs()])
    }

    /// Switch to the tab one step in the given direction, wrapping around.
    /// Returns `Task::none()` if a modal overlay is active or if there is
    /// only one tab.
    fn switch_tab_relative(&mut self, direction: &TabDirection) -> Task<EditorMessage> {
        if self.modal_overlay_blocks_editor_shortcuts() || self.tabs.len() <= 1 {
            return Task::none();
        }
        let delta: isize = match direction {
            TabDirection::Next => 1,
            TabDirection::Prev => -1,
        };
        let new_idx = ((self.active_tab_index.cast_signed() + delta)
            .rem_euclid(self.tabs.len().cast_signed())) as usize;
        self.scroll_to_tab(new_idx)
    }

    /// Collect all file paths from the workspace's directory entries for
    /// quick-open filtering. Walks all expanded and known directories.
    /// Called each time QuickOpen is opened to pick up newly expanded dirs.
    fn scan_all_workspace_files(&mut self) {
        self.all_workspace_files.clear();
        let mut paths: Vec<String> = Vec::new();
        for entries in self.dir_entries.values() {
            for entry in entries {
                if !entry.is_dir {
                    paths.push(entry.full_path.clone());
                }
            }
        }
        paths.sort();
        self.all_workspace_files = paths;
    }

    /// Filter workspace file paths by a fuzzy query string.
    /// Returns paths that contain the filter text (case-insensitive).
    fn filter_workspace_files(&self, filter: &str) -> Vec<String> {
        if filter.is_empty() {
            return self.all_workspace_files.iter().take(200).cloned().collect();
        }
        let lower_filter = filter.to_ascii_lowercase();
        let mut scored: Vec<(usize, &String)> = self
            .all_workspace_files
            .iter()
            .filter(|path| path.to_ascii_lowercase().contains(&lower_filter))
            .map(|path| {
                // Score: prefer matches on file name (after last /).
                let name = path.rsplit('/').next().unwrap_or(path);
                let name_lower = name.to_ascii_lowercase();
                let name_score = if name_lower.starts_with(&lower_filter) {
                    0 // highest priority: file name starts with query
                } else if name_lower.contains(&lower_filter) {
                    1 // file name contains query
                } else {
                    2 // path segment match only
                };
                (name_score, path)
            })
            .collect();
        scored.sort_by_key(|(score, _)| *score);
        scored
            .into_iter()
            .take(200)
            .map(|(_, p)| p.clone())
            .collect()
    }

    /// Open a file by its workspace-relative path.
    /// If the file is already open in a tab, switches to that tab.
    /// Otherwise loads the file and adds a new tab.
    fn open_file_in_editor(&mut self, path: &str) -> Task<EditorMessage> {
        let Some(ws) = self.workspace_root() else {
            return Task::none();
        };
        let abs_path = if path.starts_with('/') {
            path.to_string()
        } else {
            std::path::Path::new(&ws)
                .join(path)
                .to_string_lossy()
                .to_string()
        };

        // If already open, just switch to that tab.
        if let Some(existing_idx) = self.tabs.iter().position(|t| t.path == abs_path) {
            return self.scroll_to_tab(existing_idx);
        }

        // Mark tree as not focused when a file is opened.
        self.file_tree.tree_focused = false;
        self.pending_enter_dir = None;

        let file_path = abs_path.clone();
        let file_gen = self.generation.wrapping_add(1);
        self.generation = file_gen;
        self.file_generations.insert(file_path.clone(), file_gen);
        self.selected_file = Some(file_path.clone());

        spawn_file_load(file_path, file_gen)
    }

    /// Remove the tab at `idx`, cleaning up `tab_contents` and adjusting
    /// `active_tab_index`.
    fn remove_tab_at(&mut self, idx: usize) {
        let closed_path = self.tabs[idx].path.clone();
        self.tab_contents.remove(&closed_path);
        self.tabs.remove(idx);
        let len = self.tabs.len();
        if len == 0 {
            self.active_tab_index = 0;
        } else if idx < self.active_tab_index {
            self.active_tab_index = self.active_tab_index.saturating_sub(1);
        } else {
            self.active_tab_index = self.active_tab_index.min(len.saturating_sub(1));
        }
    }

    /// Close all tabs except `keep_idx`, discarding changes.
    ///
    /// Does not update `active_tab_index` — callers that need scroll handling
    /// should do so after the call.
    fn remove_all_tabs_except(&mut self, keep_idx: usize) {
        let mut to_remove: Vec<usize> = (0..self.tabs.len()).collect();
        to_remove.retain(|&i| i != keep_idx);
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for i in to_remove {
            self.remove_tab_at(i);
        }
    }

    /// Close the tab at `idx`. If the tab is dirty, shows the close dialog.
    /// If clean, immediately removes the tab and persists.
    /// Returns the task for saving to DB.
    fn close_tab_at(&mut self, idx: usize) -> Task<EditorMessage> {
        if idx >= self.tabs.len() {
            return Task::none();
        }
        if self.tabs[idx].is_dirty {
            self.active_modal = Some(ModalKind::CloseDialog(idx));
            return Task::none();
        }
        self.active_modal = None;
        self.remove_tab_at(idx);
        self.save_current_tabs()
    }

    /// Apply an undo or redo snapshot to the tab at `idx`.
    ///
    /// The snapshot is an owned value so there is no borrow entanglement
    /// with the undo stack.  The helper does a fresh (O(1)) lookup of
    /// `tab_contents` by path — the caller must have already resolved
    /// the path from `self.tabs[idx]`.
    ///
    /// # Panics
    /// Panics if `idx` is out of bounds for `self.tabs`.
    fn apply_undo_snapshot(&mut self, idx: usize, snapshot: Option<UndoSnapshot>) {
        let Some(snapshot) = snapshot else {
            return;
        };
        let path = self.tabs[idx].path.clone();
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            // Clear find/replace state — match byte ranges are now stale.
            tab_data.find_replace_state = None;
            tab_data.content = EditorBuffer::from_file(&snapshot.text, &path);
            tab_data
                .content
                .move_to(snapshot.cursor_line, snapshot.cursor_col);
        }
        update_dirty_flag(&mut self.tabs, &self.tab_contents, idx, &path);
    }

    /// Apply an undo or redo operation to the active tab.
    ///
    /// `is_redo` selects which operation: `false` for undo,
    /// `true` for redo.
    fn apply_undo_or_redo(&mut self, is_redo: bool) -> Task<EditorMessage> {
        let Some((idx, path)) = self.active_tab() else {
            return Task::none();
        };
        let snapshot = self.tab_contents.get_mut(&path).and_then(|tab_data| {
            let mut stack = tab_data.undo_stack.borrow_mut();
            if is_redo {
                stack.redo(&tab_data.content)
            } else {
                stack.undo(&tab_data.content)
            }
        });
        self.apply_undo_snapshot(idx, snapshot);
        Task::none()
    }

    /// Handle Undo or Redo after checking that the find bar or modal overlay
    /// won't intercept the keyboard shortcut.
    ///
    /// When the find bar is open, Cmd+Z / Cmd+Shift+Z should undo/redo within
    /// the find bar's text input (handled natively by Iced's text widget), not
    /// undo the editor content. Bail out early so the text input handles the
    /// shortcut internally.
    fn handle_undo_or_redo(&mut self, is_redo: bool) -> Task<EditorMessage> {
        if self.is_find_bar_open() || self.modal_overlay_blocks_editor_shortcuts() {
            Task::none()
        } else {
            self.apply_undo_or_redo(is_redo)
        }
    }

    /// Clear all workspace-scoped editor state when switching workspaces.
    /// Does not touch `selected_workspace_name`, `selected_workspace_path`,
    /// `generation`, or `saved_tabs_gen` — those are managed at the call site.
    fn clear_workspace_editor_state(&mut self) {
        self.file_tree.nodes.clear();
        self.file_tree.expanded_dirs.clear();
        self.selected_file = None;
        self.tabs.clear();
        self.tab_contents.clear();
        self.active_tab_index = 0;
        self.dir_entries.clear();
        self.loading_dirs.clear();
        self.dir_generations.clear();
        self.file_generations.clear();
        self.git_status_cache.clear();
        self.git_status_loading = false;
        self.git_ignore_cache.clear();
        self.git_ignore_loading = false;
        self.session_initialized = false;
        self.active_modal = None;
        self.pending_save_close = None;
        self.pending_close_others = None;
        self.file_tree.visible_tree_nodes.clear();
        self.file_tree.tree_focused = false;
        self.file_tree.tree_focus_index = 0;
        self.pending_enter_dir = None;
        self.tab_save_counter.store(0, Ordering::Release);
        self.file_mtimes.clear();
        self.deleted_file_toasted.clear();
        self.all_workspace_files.clear();
        self.global_search_gen = 0;
        self.pending_goto = None;
    }

    /// Start creating a new item (file or directory) in the given parent directory.
    ///
    /// Resolves the absolute parent path and sets up the `new_item_input` state
    /// so the user can type a name.  Any previously active modal is implicitly
    /// replaced since `active_modal` enforces mutual exclusion.
    fn start_new_item_creation(&mut self, parent_dir: String, is_dir: bool) -> Task<EditorMessage> {
        let Some(ref ws) = self.selected_workspace_path else {
            return Task::none();
        };
        let abs_parent = if parent_dir.is_empty() {
            ws.clone()
        } else {
            Path::new(ws)
                .join(&parent_dir)
                .to_string_lossy()
                .to_string()
        };
        self.active_modal = Some(ModalKind::NewItem(NewItemTarget {
            parent_dir,
            is_dir,
            abs_parent,
            ws_root: ws.clone(),
            input_text: String::new(),
        }));
        iced::widget::operation::focus::<EditorMessage>(Id::new(NEW_ITEM_INPUT_ID))
    }

    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, msg: EditorMessage) -> Task<EditorMessage> {
        match msg {
            EditorMessage::WorkspaceSelected(ref name, ref path) => {
                self.workspace_selected(name, path.as_deref())
            }

            EditorMessage::SavedTabsLoaded { tabs_data, r#gen } => {
                self.saved_tabs_loaded(tabs_data, r#gen)
            }

            EditorMessage::DirExpanded {
                dir_path,
                r#gen,
                entries,
                quiet,
            } => self.dir_expanded(&dir_path, r#gen, entries, quiet),

            EditorMessage::ToggleDir(dir_path) => self.toggle_dir(&dir_path),

            EditorMessage::SelectFile(path) => self.select_file(&path),

            EditorMessage::FileLoaded {
                path,
                r#gen,
                result,
            } => self.file_loaded(&path, r#gen, result),

            EditorMessage::TabSelected(idx) => self.switch_to_tab(idx),

            EditorMessage::TabClosed(idx) => self.close_tab_at(idx),

            EditorMessage::EditorAction(action) => self.editor_action(action),

            EditorMessage::SaveActiveTab => self.save_active_tab(),

            EditorMessage::SaveResult {
                path,
                result,
                saved_hash,
            } => self.save_result(&path, result, saved_hash),

            EditorMessage::CloseDialog { tab_index, action } => {
                self.close_dialog(tab_index, action)
            }

            EditorMessage::CloseOthersDialog { keep_idx, action } => {
                self.close_others_dialog(keep_idx, action)
            }

            EditorMessage::CloseOtherTabs(idx) => self.close_other_tabs(idx),

            EditorMessage::Escape => self.escape(),

            // ── Go-to-line ────────────────────────────────────────────
            EditorMessage::GoToLineToggle => self.go_to_line_toggle(),

            EditorMessage::GoToLineInput(input) => self.go_to_line_input(&input),

            EditorMessage::GoToLineGo => self.go_to_line_go(),

            // ── Global search (find-in-files) ──────────────────────────
            EditorMessage::GlobalSearchToggle => self.global_search_toggle(),

            EditorMessage::GlobalSearchInput(query) => self.global_search_input(query),

            EditorMessage::GlobalSearchResults {
                r#gen,
                results,
                error,
            } => self.global_search_results(r#gen, results, error),

            EditorMessage::GlobalSearchSelect(idx) => self.global_search_select(idx),

            EditorMessage::GlobalSearchClose => self.global_search_close(),

            // ── Context menu actions ─────────────────────────────────
            EditorMessage::DeleteFileRequested(path) => self.delete_file_requested(path),

            EditorMessage::DeleteDirectoryRequested(path) => self.delete_directory_requested(path),

            EditorMessage::NewFileRequested(parent_dir) => {
                self.start_new_item_creation(parent_dir, false)
            }

            EditorMessage::NewDirectoryRequested(parent_dir) => {
                self.start_new_item_creation(parent_dir, true)
            }

            EditorMessage::RevealInFinder(path) => Self::perform_reveal_in_finder(path),

            EditorMessage::CopyRelativePath(path) | EditorMessage::CopyAbsolutePath(path) => {
                iced::clipboard::write(path)
            }

            EditorMessage::ConfirmDelete => self.confirm_delete(),

            EditorMessage::CancelDelete => {
                self.active_modal = None;
                Task::none()
            }

            EditorMessage::NewItemSubmit(name) => self.new_item_submit(&name),

            EditorMessage::NewItemInput(new_text) => self.new_item_input(new_text),

            // ── Inline rename ────────────────────────────────────────────
            EditorMessage::RenameRequested(path) => self.rename_requested(&path),

            EditorMessage::RenameInput(new_text) => self.rename_input(new_text),

            EditorMessage::RenameSubmit => self.rename_submit(),

            EditorMessage::RenameCancel => self.rename_cancel(),

            EditorMessage::RenameCompleted {
                old_path,
                new_path,
                is_dir,
                result,
                dir_entries,
                rename_gen,
            } => self.rename_completed(
                &old_path,
                &new_path,
                is_dir,
                result,
                dir_entries,
                rename_gen,
            ),

            // ── Quick-open file picker ────────────────────────────────
            EditorMessage::QuickOpenToggle => self.quick_open_toggle(),

            EditorMessage::QuickOpenInput(filter) => self.quick_open_input(filter),

            EditorMessage::QuickOpenSelect(idx) => self.quick_open_select(idx),

            // ── Tab switching ─────────────────────────────────────────
            EditorMessage::TabSwitchNext => self.switch_tab_relative(&TabDirection::Next),
            EditorMessage::TabSwitchPrev => self.switch_tab_relative(&TabDirection::Prev),

            EditorMessage::CloseActiveTab => self.close_active_tab(),

            // ── Tree keyboard navigation ─────────────────────────────
            EditorMessage::TreeFocusToggled => self.tree_focus_toggled(),

            EditorMessage::TreeScrolled(scroll_y, viewport_h) => {
                self.tree_scrolled(scroll_y, viewport_h)
            }

            EditorMessage::TreeNavUp => self.navigate_tree_vertical(&TreeNavDirection::Up),

            EditorMessage::TreeNavDown => self.navigate_tree_vertical(&TreeNavDirection::Down),

            EditorMessage::TreeNavEnter => self.tree_nav_enter(),

            EditorMessage::TreeNavLeft => self.tree_nav_left(),

            EditorMessage::TreeNavRight => self.tree_nav_right(),

            EditorMessage::Undo => self.handle_undo_or_redo(false),

            EditorMessage::Redo => self.handle_undo_or_redo(true),

            EditorMessage::FindToggle => self.find_toggle(),

            EditorMessage::FindQueryInput(query) => self.find_query_input(query),

            EditorMessage::FindReplaceInput(replace) => self.find_replace_input(replace),

            EditorMessage::FindNext => self.navigate_find_match(&FindDirection::Next),

            EditorMessage::FindPrev => self.navigate_find_match(&FindDirection::Prev),

            EditorMessage::FindReplace => self.find_replace(),

            EditorMessage::FindReplaceAll => self.find_replace_all(),

            EditorMessage::FindToggleCaseSensitivity => self.find_toggle_case_sensitivity(),

            EditorMessage::RefreshFileTree => self.refresh_file_tree(),

            EditorMessage::Tick => self.tick(),

            EditorMessage::BlinkTick => self.blink_tick(),

            EditorMessage::GitStatusLoaded(result) => self.git_status_loaded(result),

            EditorMessage::GitIgnoredLoaded(result) => self.git_ignored_loaded(result),

            EditorMessage::CheckFileChanges => self.check_file_changes(),

            EditorMessage::FileReloaded {
                path,
                result,
                cursor_line,
                cursor_col,
            } => self.file_reloaded(path, result, cursor_line, cursor_col),

            EditorMessage::RevealDone | EditorMessage::Toast(_) => Task::none(),
        }
    }

    // ── Extracted handler methods ────────────────────────────────────

    /// Handle workspace selection — initializes file tree, loads tabs, sets up workspace.
    #[allow(clippy::too_many_lines)]
    fn workspace_selected(&mut self, name: &str, path: Option<&str>) -> Task<EditorMessage> {
        // Accept personal workspaces when a path is provided.
        if name.is_empty() && path.is_none() {
            self.selected_workspace_name = None;
            self.selected_workspace_path = None;
            self.clear_workspace_editor_state();
            return Task::none();
        }

        let mut tasks: Vec<Task<EditorMessage>> = Vec::new();

        // Update selected workspace.
        self.selected_workspace_name = Some(name.to_string());
        self.selected_workspace_path = path.map(std::string::ToString::to_string);

        // Clear previous state and bump both generation counters.
        let r#gen = self.generation.wrapping_add(1);
        self.generation = r#gen;
        let saved_gen = self.saved_tabs_gen.wrapping_add(1);
        self.saved_tabs_gen = saved_gen;
        self.clear_workspace_editor_state();

        // Register the root generation so DirExpanded can validate it.
        self.dir_generations.insert(String::new(), r#gen);

        // ── Task 1: read root directory ───────────────────────
        let root_path = path.unwrap_or_default().to_string();
        let root_gen = r#gen;
        let read_root_task = Task::perform(
            async move {
                let entries = read_directory_entries(&root_path, "").await;
                EditorMessage::DirExpanded {
                    dir_path: String::new(),
                    r#gen: root_gen,
                    entries,
                    quiet: false,
                }
            },
            |msg| msg,
        );
        tasks.push(read_root_task);

        // ── Task 2: load tabs from DB + file contents ────────
        let tab_ws = name.to_string();
        let tab_path = path.unwrap_or_default().to_string();
        let tab_gen = saved_gen;
        let load_tabs_task = Task::perform(
            async move {
                let store = crate::workspace::store();
                let records = store.load_editor_tabs(&tab_ws).await.unwrap_or_else(|e| {
                    tracing::warn!(?e, workspace = %tab_ws, "Failed to load editor tabs");
                    Vec::new()
                });
                let ws_path = tab_path;

                let mut loaded: Vec<SavedTabData> = Vec::new();
                for record in &records {
                    // Belt-and-suspenders: skip tabs with empty file_path —
                    // load_editor_tabs already filters these, but guard anyway.
                    if record.file_path.is_empty() || record.file_path.trim().is_empty() {
                        tracing::warn!(
                            workspace = %tab_ws,
                            tab_order = record.tab_order,
                            "Skipping editor tab with empty file_path in GUI loader"
                        );
                        continue;
                    }
                    let file_path = if ws_path.is_empty() {
                        record.file_path.clone()
                    } else {
                        Path::new(&ws_path)
                            .join(&record.file_path)
                            .to_string_lossy()
                            .to_string()
                    };

                    let loaded_text = if let Some(dirty) = record.dirty_content.clone() {
                        Some(dirty)
                    } else if let Ok(bytes) = tokio::fs::read(&file_path).await {
                        if validate_file_content(&bytes).is_ok() {
                            String::from_utf8(bytes).ok()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(text) = loaded_text {
                        let has_trailing = has_trailing_newline(&text);
                        let line_ending = detect_line_ending(&text);
                        loaded.push(SavedTabData {
                            file_path,
                            text,
                            was_dirty: record.is_dirty,
                            has_trailing_newline: has_trailing,
                            line_ending,
                            is_active: record.is_active,
                        });
                    }
                }
                EditorMessage::SavedTabsLoaded {
                    tabs_data: loaded,
                    r#gen: tab_gen,
                }
            },
            |msg| msg,
        );
        tasks.push(load_tabs_task);

        // ── Task 3: refresh git status for file tree coloring ──
        let git_path = path.unwrap_or_default().to_string();
        let git_task = Task::perform(
            async move { load_git_status(git_path).await },
            EditorMessage::GitStatusLoaded,
        );
        tasks.push(git_task);

        Task::batch(tasks)
    }

    /// Insert a newly created tab into the editor state: push the tab,
    /// store its content, and record the file mtime (if available).
    fn insert_tab(&mut self, path: String, tab: Tab, tab_data: TabData, mtime: Option<SystemTime>) {
        self.tabs.push(tab);
        self.tab_contents.insert(path.clone(), tab_data);
        if let Some(mtime) = mtime {
            self.file_mtimes.insert(path, mtime);
        }
    }

    /// Handle saved tabs loaded from the database — deserializes tab data,
    /// builds Tab/TabData structures.
    fn saved_tabs_loaded(
        &mut self,
        tabs_data: Vec<SavedTabData>,
        r#gen: u64,
    ) -> Task<EditorMessage> {
        if r#gen != self.saved_tabs_gen {
            return Task::none();
        }

        // Track which tab was active when persisted.
        let mut active_idx = 0;

        for (i, saved) in tabs_data.into_iter().enumerate() {
            if saved.is_active {
                active_idx = i;
            }

            let saved_hash = if saved.was_dirty {
                // Tab was dirty when persisted — the text in DB
                // differs from what's on disk.  Try to read the
                // on-disk version for an accurate saved hash;
                // fall back to the in-memory text if the file
                // is gone or unreadable.
                std::fs::read_to_string(&saved.file_path)
                    .as_ref()
                    .map_or_else(|_| hash_text(&saved.text), |disk| hash_text(disk))
            } else {
                hash_text(&saved.text)
            };

            let (tab, td, mtime) = make_tab_and_data(
                &saved.file_path,
                &saved.text,
                saved.has_trailing_newline,
                saved.line_ending,
                saved.was_dirty,
                saved_hash,
            );
            self.insert_tab(saved.file_path, tab, td, mtime);
        }

        if !self.tabs.is_empty() {
            self.active_tab_index = active_idx.min(self.tabs.len().saturating_sub(1));
        }
        self.session_initialized = true;

        if !self.tabs.is_empty() {
            self.scroll_to_active_tab()
        } else {
            Task::none()
        }
    }

    /// Handle directory expansion — populates dir_entries, rebuilds tree,
    /// advances focus to first child if requested.
    fn dir_expanded(
        &mut self,
        dir_path: &str,
        r#gen: u64,
        entries: Result<Vec<FsEntry>, String>,
        quiet: bool,
    ) -> Task<EditorMessage> {
        if self.dir_generations.get(dir_path) != Some(&r#gen) {
            return Task::none();
        }
        // Consume the generation slot (mirroring the pattern in rename_completed).
        // The entry is no longer needed once the matching result has been accepted.
        self.dir_generations.remove(dir_path);

        self.loading_dirs.remove(dir_path);

        match entries {
            Ok(entries) => {
                self.dir_entries.insert(dir_path.to_string(), entries);
                self.rebuild_tree();
                // If this was triggered by Enter-on-directory, advance
                // focus to the first child now that children are loaded.
                if self.pending_enter_dir.as_deref() == Some(dir_path) {
                    self.pending_enter_dir = None;
                    return self
                        .file_tree
                        .expand_dir_and_focus_first_child::<EditorMessage>(dir_path);
                }
            }
            Err(e) => {
                if quiet {
                    tracing::warn!("Failed to read directory (refresh): {e}");
                    return Task::none();
                }
                return Task::done(EditorMessage::Toast(super::ToastMessage::Warning(format!(
                    "Failed to read directory: {e}"
                ))));
            }
        }
        Task::none()
    }

    /// Handle a file being loaded from disk — opens a tab, initializes
    /// EditorBuffer, sets hash/mtime, and handles pending goto from
    /// global search.
    fn file_loaded(
        &mut self,
        path: &str,
        r#gen: u64,
        result: Result<FileLoadData, String>,
    ) -> Task<EditorMessage> {
        // Check per-file generation to prevent stale loads.
        if self.file_generations.get(path).copied() != Some(r#gen) {
            return Task::none();
        }
        // Consume the generation slot — it has served its purpose.
        // This prevents unbounded accumulation in the map without
        // requiring removal code at every close/delete/rename path.
        self.file_generations.remove(path);

        match result {
            Ok(data) => {
                let saved_hash = hash_text(&data.text);
                let (tab, tab_data, mtime) = make_tab_and_data(
                    &data.path,
                    &data.text,
                    data.has_trailing_newline,
                    data.line_ending,
                    false,
                    saved_hash,
                );
                self.insert_tab(data.path, tab, tab_data, mtime);
                self.active_tab_index = self.tabs.len().saturating_sub(1);
                self.session_initialized = true;

                // ── Pending goto from global search ────────────
                // If this file was loaded for a global-search result click,
                // jump to the matching line. Only consume when both path and
                // generation match to avoid stealing from a different file load.
                if self
                    .pending_goto
                    .as_ref()
                    .is_some_and(|(gp, _, gg)| *gp == path && *gg == r#gen)
                {
                    if let Some((_, goto_line_1based, _)) = self.pending_goto.take() {
                        let cursor_line = goto_line_1based.saturating_sub(1);
                        let tab_path = self.tabs[self.active_tab_index].path.clone();
                        if let Some(tab_data) = self.tab_contents.get_mut(&tab_path) {
                            let max_line = tab_data.content.line_count();
                            let line = cursor_line.min(max_line.saturating_sub(1));
                            tab_data.content.move_to(line, 0);
                        }
                    }
                }

                let tasks = vec![self.scroll_to_active_tab(), self.save_current_tabs()];
                Task::batch(tasks)
            }
            Err(e) => {
                let toast =
                    if e.starts_with("File too large") || e.starts_with("Binary file detected") {
                        super::ToastMessage::Warning(e)
                    } else {
                        super::ToastMessage::Error(e)
                    };
                Task::done(EditorMessage::Toast(toast))
            }
        }
    }

    /// Handle the result of a save operation — updates dirty flags,
    /// handles CloseDialog→close-tab and CloseOthers→save-queue flows.
    fn save_result(
        &mut self,
        path: &str,
        result: Result<(), String>,
        saved_hash: u64,
    ) -> Task<EditorMessage> {
        match result {
            Ok(()) => {
                let still_matches_saved = self
                    .tab_contents
                    .get(path)
                    .is_some_and(|tab_data| hash_text(&tab_data.content.text()) == saved_hash);
                if !still_matches_saved {
                    // A newer edit arrived while the save was in flight — keep dirty state.
                    return Task::none();
                }
                if let Some(tab) = self.tabs.iter_mut().find(|t| t.path == path) {
                    tab.is_dirty = false;
                    if let Some(tab_data) = self.tab_contents.get(path) {
                        let text = tab_data.content.text();
                        tab.has_trailing_newline = has_trailing_newline(&text);
                    }
                }
                if let Some(tab_data) = self.tab_contents.get_mut(path) {
                    tab_data.saved_text_hash = saved_hash;
                }
                // Update stored mtime so the next auto-refresh tick won't
                // detect the save-time mtime change as an external edit
                // and re-read the file, destroying the undo stack.
                if let Ok(meta) = std::fs::metadata(path) {
                    if let Ok(mtime) = meta.modified() {
                        self.file_mtimes.insert(path.to_string(), mtime);
                    }
                }

                // If this save was triggered by CloseDialog::Save, close the tab now.
                if let Some(close_idx) = self.pending_save_close.take() {
                    if close_idx < self.tabs.len() {
                        self.remove_tab_at(close_idx);
                        // Save after removal + scroll.
                        let mut tasks: Vec<Task<EditorMessage>> = Vec::new();
                        if !self.tabs.is_empty() {
                            tasks.push(self.scroll_to_active_tab());
                        }
                        tasks.push(self.save_current_tabs());
                        return Task::batch(tasks);
                    }
                }
                // If this save is part of a close-others save queue, continue.
                if let Some((keep_idx, mut remaining)) = self.pending_close_others.take() {
                    return if remaining.is_empty() {
                        // All dirty tabs saved — close everything except keep_idx.
                        self.remove_all_tabs_except(keep_idx);
                        // Save after removal.
                        self.try_save_current_tabs().map_or_else(Task::none, |t| {
                            Task::batch([
                                t,
                                Task::done(EditorMessage::Toast(super::ToastMessage::Saved)),
                            ])
                        })
                    } else {
                        // Save the next dirty tab.
                        let next = remaining.remove(0);
                        self.pending_close_others = Some((keep_idx, remaining));
                        build_save_task(&self.tabs, &self.tab_contents, next)
                    };
                }

                // Regular save (not from close dialog) — persist clean state.
                if let Some(save_task) = self.try_save_current_tabs() {
                    save_task
                } else {
                    Task::done(EditorMessage::Toast(super::ToastMessage::Saved))
                }
            }
            Err(e) => {
                self.pending_save_close = None;
                self.pending_close_others = None;
                let toast = super::ToastMessage::Error(e);
                Task::done(EditorMessage::Toast(toast))
            }
        }
    }

    /// Handle Escape key — dismisses modal overlays, find bar, tree focus,
    /// and residual close-dialog auxiliary state in priority order.
    ///
    /// Priority:
    /// 1. Active modal overlay (closed via [`ModalKind`] match on
    ///    [`active_modal`] — clears auxiliary state for `CloseDialog`
    ///    and `CloseOthers`).
    /// 2. Find/replace bar on the active tab.
    /// 3. File-tree focus.
    /// 4. Residual [`pending_save_close`] / [`pending_close_others`] state.
    fn escape(&mut self) -> Task<EditorMessage> {
        // Close modal overlays first.
        if let Some(modal) = self.active_modal.take() {
            match modal {
                ModalKind::CloseDialog(..) => {
                    self.pending_save_close = None;
                }
                ModalKind::CloseOthers(..) => {
                    self.pending_close_others = None;
                }
                _ => {}
            }
            return Task::none();
        }

        // Close find bar on active tab next, if open.
        if let Some((_, path)) = self.active_tab() {
            if let Some(tab_data) = self.tab_contents.get_mut(&path) {
                if tab_data.find_replace_state.is_some() {
                    tab_data.find_replace_state = None;
                    return Task::none();
                }
            }
        }

        // Unfocus the file tree, or clear residual close-dialog state.
        if self.file_tree.tree_focused {
            self.file_tree.tree_focused = false;
            self.pending_enter_dir = None;
            return Task::none();
        }
        self.pending_save_close = None;
        self.pending_close_others = None;
        Task::none()
    }

    /// Handle global search toggle — opens/closes the search overlay,
    /// spawns search engine initialization.
    fn global_search_toggle(&mut self) -> Task<EditorMessage> {
        if matches!(self.active_modal, Some(ModalKind::GlobalSearch(_))) {
            // Close if already open.
            self.active_modal = None;
            return Task::none();
        }
        if self.active_modal.is_some() {
            return Task::none();
        }
        // Close find bar when opening global search.
        if let Some((_, path)) = self.active_tab() {
            if let Some(tab_data) = self.tab_contents.get_mut(&path) {
                tab_data.find_replace_state = None;
            }
        }

        let ws_path = match self.selected_workspace_path.as_ref() {
            Some(p) => p.clone(),
            None => return Task::none(),
        };
        let ws_name = match self.selected_workspace_name.as_ref() {
            Some(n) => n.clone(),
            None => return Task::none(),
        };

        self.global_search_gen = self.global_search_gen.wrapping_add(1);
        let gs_gen = self.global_search_gen;

        self.active_modal = Some(ModalKind::GlobalSearch(GlobalSearchState {
            query: String::new(),
            results: Vec::new(),
            selected_index: 0,
            status: GlobalSearchStatus::Idle,
            search_gen: gs_gen,
        }));

        // Start scanning the search engine and show readiness status.
        let engine_task = Task::perform(
            async move {
                match crate::search_engine::get_or_init_engine(
                    &ws_name,
                    std::path::Path::new(&ws_path),
                ) {
                    Ok(entry) => match crate::search_engine::ensure_scanned(&entry).await {
                        Ok(()) => EditorMessage::GlobalSearchResults {
                            r#gen: gs_gen,
                            results: Vec::new(),
                            error: None,
                        },
                        Err(e) => EditorMessage::GlobalSearchResults {
                            r#gen: gs_gen,
                            results: Vec::new(),
                            error: Some(e),
                        },
                    },
                    Err(e) => EditorMessage::GlobalSearchResults {
                        r#gen: gs_gen,
                        results: Vec::new(),
                        error: Some(e),
                    },
                }
            },
            |msg| msg,
        );

        // Auto-focus the search input when the panel opens.
        let focus_task =
            iced::widget::operation::focus::<EditorMessage>(Id::new(GLOBAL_SEARCH_INPUT_ID));

        Task::batch([engine_task, focus_task])
    }

    /// Handle global search results — populates search results, handles
    /// stale results, error states, and empty results.
    fn global_search_results(
        &mut self,
        r#gen: u64,
        results: Vec<OwnedGrepMatch>,
        error: Option<String>,
    ) -> Task<EditorMessage> {
        // Stale result? Discard (r#gen is never 0 from the async helper).
        if r#gen != self.global_search_gen {
            return Task::none();
        }

        let Some(ModalKind::GlobalSearch(state)) = &mut self.active_modal else {
            return Task::none();
        };

        if let Some(err) = error {
            state.status = GlobalSearchStatus::Error(err);
            state.results.clear();
            return Task::none();
        }

        if results.is_empty() && state.query.is_empty() {
            state.status = GlobalSearchStatus::Idle;
            return Task::none();
        }

        if results.is_empty() {
            state.status = GlobalSearchStatus::NoResults;
            state.results.clear();
            state.selected_index = 0;
            return Task::none();
        }

        state.results = results;
        state.selected_index = 0;
        state.status = GlobalSearchStatus::Done;
        Task::none()
    }

    /// Handle FindReplaceAll — replaces all matches in the active buffer.
    fn find_replace_all(&mut self) -> Task<EditorMessage> {
        let Some((idx, path)) = self.active_tab() else {
            return Task::none();
        };
        let mut toast = None;
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if let Some(ref state) = tab_data.find_replace_state {
                if !state.matches.is_empty() {
                    // Take undo snapshot.
                    tab_data
                        .undo_stack
                        .borrow_mut()
                        .snap_before_edit(&tab_data.content);
                    let cursor_before = tab_data.content.cursor();
                    let text = tab_data.content.text();
                    let replace = &state.replace;
                    // Replace all in reverse order to preserve positions.
                    let mut new_text = text.clone();
                    for range in state.matches.iter().rev() {
                        new_text.replace_range(range.start..range.end, replace);
                    }
                    tab_data.content = EditorBuffer::from_file(&new_text, &path);
                    let max_line = tab_data.content.line_count().saturating_sub(1);
                    let line = cursor_before.line.min(max_line);
                    tab_data.content.move_to(line, cursor_before.column);
                    // Clear matches since they're all replaced.
                    if let Some(ref mut state) = tab_data.find_replace_state {
                        state.matches.clear();
                        state.current_match_idx = 0;
                    }
                    toast = Some(EditorMessage::Toast(super::ToastMessage::SuccessMsg(
                        "All matches replaced".to_string(),
                    )));
                }
            }
        }
        update_dirty_flag(&mut self.tabs, &self.tab_contents, idx, &path);
        if let Some(t) = toast {
            Task::done(t)
        } else {
            Task::none()
        }
    }

    /// Toggle directory expansion in the file tree — collapses if already expanded,
    /// otherwise loads and expands.
    fn toggle_dir(&mut self, dir_path: &str) -> Task<EditorMessage> {
        // Clicking a tree row while renaming means the user is dismissing the rename.
        self.dismiss_rename();
        // Clear any previously-selected file highlight — navigating
        // to a directory should visually show the directory as focused,
        // not the previously-selected file.
        self.selected_file = None;
        if self.file_tree.expanded_dirs.contains(dir_path) {
            self.file_tree.tree_focused = true;
            return self.collapse_dir(dir_path);
        }
        self.file_tree.expanded_dirs.insert(dir_path.to_string());

        let read_task = if !self.dir_entries.contains_key(dir_path) {
            match self.load_dir_async(dir_path, "ToggleDir") {
                Some(t) => t,
                None => return Task::none(),
            }
        } else {
            Task::none()
        };

        self.rebuild_tree();
        self.file_tree.tree_focused = true;
        // Place focus on the expanding directory.
        self.file_tree.focus_path(dir_path);
        read_task
    }

    /// Handle file selection in the tree — opens or switches to the selected file.
    fn select_file(&mut self, path: &str) -> Task<EditorMessage> {
        // Clicking a file tree row transfers keyboard focus to the tree
        // so that arrow keys navigate the tree instead of the editor.
        self.file_tree.tree_focused = true;
        // Clicking a tree row while renaming means the user is dismissing the rename.
        self.dismiss_rename();
        self.pending_enter_dir = None;
        // Remember the clicked file's position for Ctrl+B re-focus.
        self.file_tree.focus_path(path);
        self.selected_file = Some(path.to_string());

        // Resolve tree-relative path against workspace root so that
        // file operations and tab paths are absolute (matching restored
        // tabs) and work regardless of MahBot's CWD.
        let Some(abs_path) = self.abs_path(path) else {
            return Task::none();
        };

        if let Some(pos) = self.tabs.iter().position(|t| t.path == abs_path) {
            return self.switch_to_tab(pos);
        }

        // Per-file generation: keyed by absolute path.
        let file_gen = self
            .file_generations
            .get(&abs_path)
            .copied()
            .unwrap_or(0)
            .wrapping_add(1);
        self.file_generations.insert(abs_path.clone(), file_gen);
        spawn_file_load(abs_path, file_gen)
    }

    /// Handle an editor action — performs the action, tracks undo state.
    fn editor_action(&mut self, action: super::editor_widget::EditorAction) -> Task<EditorMessage> {
        // Clicking in the editor content transfers focus from the file
        // tree to the editor, matching Escape handler behavior.
        self.file_tree.tree_focused = false;
        self.pending_enter_dir = None;
        self.dismiss_rename();

        let Some((idx, path)) = self.active_tab() else {
            return Task::none();
        };
        let is_edit = action.is_edit_action();
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if is_edit {
                tab_data
                    .undo_stack
                    .borrow_mut()
                    .snap_before_edit(&tab_data.content);
            }
            tab_data.content.perform_action(action);
        }
        if is_edit {
            update_dirty_flag(&mut self.tabs, &self.tab_contents, idx, &path);
        }
        Task::none()
    }

    /// Handle save-active-tab — builds a save task for the active tab.
    fn save_active_tab(&self) -> Task<EditorMessage> {
        if self.modal_overlay_blocks_editor_shortcuts() {
            return Task::none();
        }
        let Some(idx) = self.active_tab_idx() else {
            return Task::none();
        };
        build_save_task(&self.tabs, &self.tab_contents, idx)
    }

    /// Handle close-dialog actions (Save, Discard, Cancel) for a single tab.
    fn close_dialog(&mut self, tab_index: usize, action: CloseAction) -> Task<EditorMessage> {
        match action {
            CloseAction::Save => {
                if tab_index < self.tabs.len() {
                    // Clear dialog immediately; close tab after save completes.
                    self.active_modal = None;
                    self.pending_save_close = Some(tab_index);

                    build_save_task(&self.tabs, &self.tab_contents, tab_index)
                } else {
                    self.active_modal = None;
                    Task::none()
                }
            }
            CloseAction::Discard => {
                self.active_modal = None;
                self.pending_save_close = None;
                if tab_index < self.tabs.len() {
                    self.remove_tab_at(tab_index);
                }
                self.save_current_tabs()
            }
            CloseAction::Cancel => {
                self.active_modal = None;
                self.pending_save_close = None;
                Task::none()
            }
        }
    }

    /// Handle close-others-dialog — saves dirty tabs then closes all but keep_idx.
    fn close_others_dialog(&mut self, keep_idx: usize, action: CloseAction) -> Task<EditorMessage> {
        match action {
            CloseAction::Save => {
                self.active_modal = None;
                // Collect all dirty tabs (excluding keep_idx) to save sequentially.
                let mut dirty: Vec<usize> = (0..self.tabs.len())
                    .filter(|&i| i != keep_idx && self.tabs[i].is_dirty)
                    .collect();
                if dirty.is_empty() {
                    // Nothing to save — just close the rest and persist.
                    self.remove_all_tabs_except(keep_idx);
                    return self.save_current_tabs();
                }
                // Start saving the first dirty tab in the queue.
                let first = dirty.remove(0);
                self.pending_close_others = Some((keep_idx, dirty));
                build_save_task(&self.tabs, &self.tab_contents, first)
            }
            CloseAction::Discard => {
                self.active_modal = None;
                self.pending_close_others = None;
                // Close all tabs except keep_idx, discarding unsaved changes.
                self.remove_all_tabs_except(keep_idx);
                self.save_current_tabs()
            }
            CloseAction::Cancel => {
                self.active_modal = None;
                self.pending_close_others = None;
                Task::none()
            }
        }
    }

    /// Handle close-other-tabs — shows a dialog if there are dirty tabs.
    fn close_other_tabs(&mut self, idx: usize) -> Task<EditorMessage> {
        if idx >= self.tabs.len() {
            return Task::none();
        }
        // Collect indices of dirty tabs (excluding the kept tab).
        let dirty: Vec<usize> = (0..self.tabs.len())
            .filter(|&i| i != idx && self.tabs[i].is_dirty)
            .collect();
        if dirty.is_empty() {
            // No unsaved changes — close immediately and persist.
            self.remove_all_tabs_except(idx);
            return self.save_current_tabs();
        }
        self.active_modal = Some(ModalKind::CloseOthers(idx));
        Task::none()
    }

    /// Handle go-to-line toggle — opens/closes the go-to-line input bar.
    fn go_to_line_toggle(&mut self) -> Task<EditorMessage> {
        // Allow toggle-to-close when GotoLine is already open, but
        // block if any other modal is active.
        if let Some(modal) = &self.active_modal {
            if !matches!(modal, ModalKind::GotoLine(_)) {
                return Task::none();
            }
        }
        if let Some((_, path)) = self.active_tab() {
            if matches!(self.active_modal, Some(ModalKind::GotoLine(_))) {
                self.active_modal = None;
                return Task::none();
            }
            // Close find bar when opening go-to-line.
            if let Some(tab_data) = self.tab_contents.get_mut(&path) {
                tab_data.find_replace_state = None;
            }
            self.active_modal = Some(ModalKind::GotoLine(String::new()));
            return iced::widget::operation::focus::<EditorMessage>(Id::new(GOTO_LINE_INPUT_ID));
        }
        Task::none()
    }

    /// Handle go-to-line input — filters to digits only.
    fn go_to_line_input(&mut self, input: &str) -> Task<EditorMessage> {
        // Only keep digits in the input.
        let digits: String = input.chars().filter(char::is_ascii_digit).collect();
        if matches!(self.active_modal, Some(ModalKind::GotoLine(_))) {
            self.active_modal = Some(ModalKind::GotoLine(digits));
        }
        Task::none()
    }

    /// Handle go-to-line go — jumps to the entered line number.
    fn go_to_line_go(&mut self) -> Task<EditorMessage> {
        let input = match &self.active_modal {
            Some(ModalKind::GotoLine(v)) => v.clone(),
            _ => return Task::none(),
        };
        let line_num: usize = match input.parse::<usize>() {
            Ok(n) if n > 0 => n.saturating_sub(1), // convert 1-based to 0-based
            _ => return Task::none(),
        };
        let Some((_, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            let max_line = tab_data.content.line_count();
            let line = line_num.min(max_line.saturating_sub(1));
            tab_data.content.move_to(line, 0);
        }
        self.active_modal = None;
        Task::none()
    }

    /// Handle global search input — updates query and triggers async search.
    fn global_search_input(&mut self, query: String) -> Task<EditorMessage> {
        let Some(ModalKind::GlobalSearch(state)) = &mut self.active_modal else {
            return Task::none();
        };
        state.query.clone_from(&query);

        if query.is_empty() {
            state.status = GlobalSearchStatus::Idle;
            state.results.clear();
            state.selected_index = 0;
            // Increment generation to cancel any in-flight searches.
            self.global_search_gen = self.global_search_gen.wrapping_add(1);
            state.search_gen = self.global_search_gen;
            return Task::none();
        }

        state.status = GlobalSearchStatus::Searching;

        let ws_path = match self.selected_workspace_path.as_ref() {
            Some(p) => p.clone(),
            None => return Task::none(),
        };
        let ws_name = match self.selected_workspace_name.as_ref() {
            Some(n) => n.clone(),
            None => return Task::none(),
        };

        self.global_search_gen = self.global_search_gen.wrapping_add(1);
        let gs_gen = self.global_search_gen;
        state.search_gen = gs_gen;

        Task::perform(run_global_search(ws_path, ws_name, query, gs_gen), |msg| {
            msg
        })
    }

    /// Handle global search select — opens the selected file at the matching line.
    fn global_search_select(&mut self, idx: usize) -> Task<EditorMessage> {
        let Some(ModalKind::GlobalSearch(state)) = &self.active_modal else {
            return Task::none();
        };
        let Some(match_result) = state.results.get(idx) else {
            return Task::none();
        };
        let abs_path = match_result.abs_path.clone();
        #[allow(clippy::cast_possible_truncation)]
        let line_number = match_result.line_number as usize;

        // Close the search panel.
        self.active_modal = None;

        // Open the file and move to the matching line.
        // Convert from 1-based (grep) to 0-based (editor).
        let cursor_line = line_number.saturating_sub(1);

        // Check if already open in a tab.
        if let Some(existing_idx) = self.tabs.iter().position(|t| t.path == abs_path) {
            self.active_tab_index = existing_idx;
            if let Some(tab_data) = self.tab_contents.get_mut(&abs_path) {
                let max_line = tab_data.content.line_count();
                let line = cursor_line.min(max_line.saturating_sub(1));
                tab_data.content.move_to(line, 0);
            }
            return self.scroll_to_active_tab();
        }

        // File not open — load it, then jump after loading.
        // Set pending_goto so FileLoaded handler moves the cursor.
        // Use self.generation.wrapping_add(1) to match the generation
        // that open_file_in_editor will assign (line 2103).
        let file_gen = self.generation.wrapping_add(1);
        self.pending_goto = Some((abs_path.clone(), line_number, file_gen));
        self.open_file_in_editor(&abs_path)
    }

    /// Handle global search close — closes the search panel.
    fn global_search_close(&mut self) -> Task<EditorMessage> {
        self.active_modal = None;
        Task::none()
    }

    /// Handle delete-file-requested — shows the delete confirmation dialog.
    fn delete_file_requested(&mut self, path: String) -> Task<EditorMessage> {
        let Some(abs_path) = self.abs_path(&path) else {
            return Task::none();
        };
        self.active_modal = Some(ModalKind::DeleteConfirm(DeleteConfirmTarget {
            path,
            is_dir: false,
            dirty_tab_count: 0,
            abs_path,
        }));
        Task::none()
    }

    /// Handle delete-directory-requested — shows the delete confirmation dialog.
    fn delete_directory_requested(&mut self, path: String) -> Task<EditorMessage> {
        // Guard: don't allow deleting the root directory.
        if path.is_empty() {
            return Task::done(EditorMessage::Toast(super::ToastMessage::Warning(
                "Cannot delete root directory".into(),
            )));
        }
        let Some(abs_path) = self.abs_path(&path) else {
            return Task::none();
        };
        let abs_prefix = format!("{abs_path}/");

        // Count open tabs that are inside this directory.
        let mut dirty_count = 0;
        for tab in &self.tabs {
            if tab.path.starts_with(&abs_prefix) {
                if tab.is_dirty {
                    dirty_count += 1;
                }
            }
        }

        self.active_modal = Some(ModalKind::DeleteConfirm(DeleteConfirmTarget {
            path,
            is_dir: true,
            dirty_tab_count: dirty_count,
            abs_path,
        }));
        Task::none()
    }

    /// Handle confirm-delete — performs the actual file/directory deletion.
    fn confirm_delete(&mut self) -> Task<EditorMessage> {
        let Some(ModalKind::DeleteConfirm(target)) = self.active_modal.clone() else {
            return Task::none();
        };
        self.active_modal = None;
        if target.is_dir {
            self.perform_dir_delete(&target)
        } else {
            self.perform_file_delete(&target)
        }
    }

    /// Handle new-item-submit — validates and creates the new file/directory.
    fn new_item_submit(&mut self, name: &str) -> Task<EditorMessage> {
        let Some(ModalKind::NewItem(target)) = self.active_modal.clone() else {
            return Task::none();
        };
        let trimmed = name.trim();
        if let Some(msg) = validate_item_name(trimmed) {
            return Task::done(EditorMessage::Toast(super::ToastMessage::Warning(
                msg.into(),
            )));
        }
        self.active_modal = None;
        self.perform_create_item(&target, trimmed)
    }

    /// Handle new-item-input — updates the input text as the user types.
    fn new_item_input(&mut self, new_text: String) -> Task<EditorMessage> {
        if let Some(ModalKind::NewItem(ref mut target)) = self.active_modal {
            target.input_text = new_text;
        }
        Task::none()
    }

    /// Handle rename-requested — starts the inline rename modal.
    fn rename_requested(&mut self, path: &str) -> Task<EditorMessage> {
        let Some(ref ws) = self.selected_workspace_path else {
            return Task::none();
        };
        // Guard: don't allow renaming root directory.
        if path.is_empty() {
            return Task::done(EditorMessage::Toast(super::ToastMessage::Warning(
                "Cannot rename root directory".into(),
            )));
        }
        let abs_path = self
            .abs_path(path)
            .expect("RenameRequested: selected_workspace_path already guarded above");
        let file_name = Path::new(&abs_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        // Determine if it's a directory by checking the actual filesystem.
        let is_dir = Path::new(&abs_path).is_dir();

        self.active_modal = Some(ModalKind::Rename(RenameTarget {
            abs_path,
            ws_root: ws.clone(),
            path: path.to_string(),
            is_dir,
            input_text: file_name,
            error: None,
        }));
        iced::widget::operation::focus::<EditorMessage>(Id::from(format!("rename_input_{path}")))
    }

    /// Handle rename-input — updates the inline rename text as the user types.
    fn rename_input(&mut self, new_text: String) -> Task<EditorMessage> {
        if let Some(ModalKind::Rename(ref mut target)) = self.active_modal {
            target.input_text = new_text;
            // Clear error when user starts typing again.
            if target.error.is_some() {
                target.error = None;
            }
        }
        Task::none()
    }

    /// Handle rename-submit — validates and performs the async rename operation.
    #[allow(clippy::too_many_lines)]
    fn rename_submit(&mut self) -> Task<EditorMessage> {
        let Some(ModalKind::Rename(target)) = self.active_modal.clone() else {
            return Task::none();
        };
        // All-space names fall through to the empty-name check below.
        let trimmed = target.input_text.trim().to_string();

        // ── Validation ────────────────────────────────────────
        // validate_item_name covers empty name, path separators,
        // dot/dotdot, and OS-reserved names.
        let error_msg = validate_item_name(&trimmed);
        if let Some(msg) = error_msg {
            if let Some(ModalKind::Rename(ref mut rt)) = self.active_modal {
                rt.error = Some(msg.into());
            }
            return Task::none();
        }

        // Compute the new absolute and relative paths.
        let parent_dir = Path::new(&target.path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        let new_rel_path = if parent_dir.is_empty() {
            trimmed.clone()
        } else {
            format!("{parent_dir}/{trimmed}")
        };
        let new_abs_path = Path::new(&target.ws_root)
            .join(&new_rel_path)
            .to_string_lossy()
            .to_string();

        // Check if target already exists.
        if Path::new(&new_abs_path).exists() {
            if let Some(ModalKind::Rename(ref mut rt)) = self.active_modal {
                rt.error = Some("A file or directory with that name already exists".into());
            }
            return Task::none();
        }

        // All validations passed — clear the inline rename state
        // and fire the async rename task.
        self.active_modal = None;

        let old_abs = target.abs_path.clone();
        let old_rel = target.path.clone();
        let is_dir = target.is_dir;
        let parent_dir_clone = parent_dir;
        let ws_root = target.ws_root.clone();
        // Follow the same generation-based invalidation protocol as
        // every other async directory operation (ToggleDir, TreeNavEnter,
        // perform_create_item, etc.): bump self.generation and register
        // it in dir_generations so that any in-flight DirExpanded for
        // this directory is invalidated (its generation won't match).
        let dir_gen = self.generation.wrapping_add(1);
        self.generation = dir_gen;
        self.dir_generations
            .insert(parent_dir_clone.clone(), dir_gen);

        Task::perform(
            async move {
                // Handle case-only rename on case-insensitive filesystems
                // via a two-step rename through a temporary name.
                let old_lower = old_rel.to_lowercase();
                let new_lower = new_rel_path.to_lowercase();
                let result = if old_lower == new_lower && old_rel != new_rel_path {
                    // Case-only rename: rename to a temp name first, then to the target.
                    let temp_name = format!(
                        "{}_{}",
                        &trimmed,
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map_or(0, |d| d.as_nanos())
                    );
                    let temp_abs = Path::new(&ws_root)
                        .join(&parent_dir_clone)
                        .join(&temp_name)
                        .to_string_lossy()
                        .to_string();
                    if let Err(e) = tokio::fs::rename(&old_abs, &temp_abs).await {
                        Err(format!("Rename failed: {e}"))
                    } else {
                        tokio::fs::rename(&temp_abs, &new_abs_path)
                            .await
                            .map_err(|e| format!("Rename failed: {e}"))
                    }
                } else {
                    tokio::fs::rename(&old_abs, &new_abs_path)
                        .await
                        .map_err(|e| format!("Rename failed: {e}"))
                };

                // Re-read parent directory regardless of success/failure
                // so the tree reflects the current filesystem state.
                let entries = read_directory_entries(&ws_root, &parent_dir_clone).await;

                EditorMessage::RenameCompleted {
                    old_path: old_rel,
                    new_path: new_rel_path,
                    is_dir,
                    result,
                    dir_entries: entries,
                    rename_gen: dir_gen,
                }
            },
            |msg| msg,
        )
    }

    /// Cancel inline rename — clicking any other UI element while a rename
    /// input is active dismisses the rename.
    fn dismiss_rename(&mut self) {
        if matches!(self.active_modal, Some(ModalKind::Rename(_))) {
            self.active_modal = None;
        }
    }

    /// Handle rename-cancel — dismisses the inline rename modal.
    fn rename_cancel(&mut self) -> Task<EditorMessage> {
        self.active_modal = None;
        Task::none()
    }

    /// Handle rename-completed — updates paths, tab data, tree, and filesystem
    /// state after an async rename operation.
    #[allow(clippy::too_many_lines)]
    fn rename_completed(
        &mut self,
        old_path: &str,
        new_path: &str,
        is_dir: bool,
        result: Result<(), String>,
        dir_entries: Result<Vec<FsEntry>, String>,
        rename_gen: u64,
    ) -> Task<EditorMessage> {
        // Workspace could have been cleared mid-rename.  abs_path()
        // returns None when workspace_root() returns None, so we
        // handle both cases in the Ok arm below.

        // Stale-result prevention via the standard dir_generations
        // protocol (same as dir_expanded).  Compute the parent dir
        // and check if we still own the generation slot.
        let re_path = Path::new(old_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        if self.dir_generations.get(&re_path) != Some(&rename_gen) {
            return Task::none();
        }
        // Own the generation — consume it so a future operation can
        // take the slot.
        self.dir_generations.remove(&re_path);

        match result {
            Ok(()) => {
                // ── Update selected_file if it matches ────
                if self.selected_file.as_deref() == Some(old_path) {
                    self.selected_file = Some(new_path.to_string());
                }

                // ── Update open tab paths ────────────────
                let Some(old_abs) = self.abs_path(old_path) else {
                    return Task::none();
                };
                let Some(new_abs) = self.abs_path(new_path) else {
                    return Task::none();
                };

                // Build a prefix-based replacement for directory renames.
                if is_dir {
                    let old_prefix = format!("{old_abs}/");
                    for tab in &mut self.tabs {
                        if tab.path.starts_with(&old_prefix) {
                            let rest = tab.path.strip_prefix(&old_prefix).unwrap_or("");
                            tab.path = format!("{new_abs}/{rest}");
                            tab.file_name = Path::new(&tab.path)
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default();
                        }
                    }
                    // Re-key tab_contents for affected files.
                    rekey_map_prefix(
                        &mut self.tab_contents,
                        &format!("{}/", &old_abs),
                        &new_abs,
                        |_| {},
                    );

                    // Update expanded_dirs to replace old_path with new_path.
                    if self.file_tree.expanded_dirs.remove(old_path) {
                        self.file_tree.expanded_dirs.insert(new_path.to_string());
                    }
                    // Also update any child expanded dirs (e.g., dir/subdir → newdir/subdir).
                    rekey_set_prefix(
                        &mut self.file_tree.expanded_dirs,
                        &format!("{old_path}/"),
                        new_path,
                    );

                    // Migrate dir_entries for child paths so expanded children
                    // don't vanish on rebuild.  build_hierarchical_tree looks up
                    // each expanded directory in dir_entries by full_path, so we
                    // must re-key those entries under the new prefix and update
                    // each entry's full_path to reflect the new path.
                    let old_entries_prefix = format!("{old_path}/");
                    let new_path_clone = new_path.to_string();
                    rekey_map_prefix(
                        &mut self.dir_entries,
                        &old_entries_prefix,
                        new_path,
                        |entries: &mut Vec<FsEntry>| {
                            for entry in entries.iter_mut() {
                                update_entry_path(entry, &old_entries_prefix, &new_path_clone);
                            }
                        },
                    );

                    // Also migrate the renamed directory's own dir_entries entry
                    // so it doesn't vanish from the tree on rebuild.
                    // Must update the child entries' full_path to reflect the
                    // new path prefix (same as the child-entries loop above).
                    if let Some(mut own_entries) = self.dir_entries.remove(old_path) {
                        for entry in &mut own_entries {
                            update_entry_path(entry, &old_entries_prefix, new_path);
                        }
                        self.dir_entries.insert(new_path.to_string(), own_entries);
                    }
                } else {
                    // File rename: update single tab.
                    for tab in &mut self.tabs {
                        if tab.path == old_abs {
                            tab.path.clone_from(&new_abs);
                            tab.file_name = Path::new(&new_abs)
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default();
                            break;
                        }
                    }
                    if let Some(data) = self.tab_contents.remove(&old_abs) {
                        self.tab_contents.insert(new_abs.clone(), data);
                    }
                }

                // ── Migrate file_mtimes, deleted_file_toasted, and file_generations ──
                // Re-key entries from old absolute path to new absolute
                // path so auto-refresh doesn't spuriously stat the old path
                // and in-flight FileLoaded results are properly validated.
                if is_dir {
                    let old_abs_prefix = format!("{old_abs}/");
                    rekey_map_prefix(&mut self.file_mtimes, &old_abs_prefix, &new_abs, |_| {});
                    rekey_set_prefix(&mut self.deleted_file_toasted, &old_abs_prefix, &new_abs);
                    rekey_map_prefix(
                        &mut self.file_generations,
                        &old_abs_prefix,
                        &new_abs,
                        |_| {},
                    );
                } else {
                    // File rename — migrate single entry.
                    if let Some(mtime) = self.file_mtimes.remove(&old_abs) {
                        self.file_mtimes.insert(new_abs.clone(), mtime);
                    }
                    if self.deleted_file_toasted.remove(&old_abs) {
                        self.deleted_file_toasted.insert(new_abs.clone());
                    }
                    if let Some(file_gen) = self.file_generations.remove(&old_abs) {
                        self.file_generations.insert(new_abs.clone(), file_gen);
                    }
                }

                // ── Update dir entries and rebuild tree ───
                match dir_entries {
                    Ok(entries) => {
                        // re_path was computed at the top of the
                        // handler for the staleness check; we own
                        // the generation slot, so insert unconditionally.
                        self.dir_entries.insert(re_path, entries);
                        self.rebuild_tree();
                        // Focus on the renamed entry.
                        self.file_tree.focus_path(new_path);
                    }
                    Err(e) => {
                        self.rebuild_tree();
                        return Task::done(EditorMessage::Toast(super::ToastMessage::Error(
                            format!("Rename succeeded but failed to refresh tree: {e}"),
                        )));
                    }
                }

                Task::batch([
                    self.save_current_tabs(),
                    Task::done(EditorMessage::Toast(super::ToastMessage::SuccessMsg(
                        format!("Renamed \"{old_path}\" → \"{new_path}\""),
                    ))),
                ])
            }
            Err(e) => {
                // re_path is already computed at the top of the
                // handler; we own the generation slot.
                match dir_entries {
                    Ok(entries) => {
                        self.dir_entries.insert(re_path, entries);
                        self.rebuild_tree();
                    }
                    Err(_) => {
                        self.rebuild_tree();
                    }
                }
                Task::done(EditorMessage::Toast(super::ToastMessage::Error(e)))
            }
        }
    }

    /// Handle quick-open toggle — opens/closes the quick-open file picker.
    fn quick_open_toggle(&mut self) -> Task<EditorMessage> {
        if matches!(self.active_modal, Some(ModalKind::QuickOpen(_))) {
            self.active_modal = None;
            return Task::none();
        }
        if self.active_modal.is_some() {
            return Task::none();
        }

        // Refresh file list from all currently expanded directories.
        self.scan_all_workspace_files();

        self.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
            filter: String::new(),
            selected_index: 0,
            results: Vec::new(),
        }));
        iced::widget::operation::focus::<EditorMessage>(Id::new(QUICK_OPEN_INPUT_ID))
    }

    /// Handle quick-open input — filters the file list.
    fn quick_open_input(&mut self, filter: String) -> Task<EditorMessage> {
        let results = self.filter_workspace_files(&filter);
        if let Some(ModalKind::QuickOpen(ref mut qo)) = self.active_modal {
            qo.filter = filter;
            qo.results = results;
            qo.selected_index = 0;
        }
        Task::none()
    }

    /// Handle quick-open select — opens the selected file.
    fn quick_open_select(&mut self, idx: usize) -> Task<EditorMessage> {
        let result_path = match &self.active_modal {
            Some(ModalKind::QuickOpen(qo)) => qo.results.get(idx).cloned(),
            _ => None,
        };
        self.active_modal = None;
        if let Some(path) = result_path {
            return self.open_file_in_editor(&path);
        }
        Task::none()
    }

    /// Handle close-active-tab — closes the currently active tab.
    fn close_active_tab(&mut self) -> Task<EditorMessage> {
        if self.modal_overlay_blocks_editor_shortcuts() {
            return Task::none();
        }
        let idx = self.active_tab_index;
        if idx < self.tabs.len() {
            return self.close_tab_at(idx);
        }
        Task::none()
    }

    /// Handle tree-focus-toggled — toggles keyboard focus between tree and editor.
    fn tree_focus_toggled(&mut self) -> Task<EditorMessage> {
        // Suppress during any modal overlay (QuickOpen, GlobalSearch,
        // GotoLine, Rename, etc.) — the overlay owns keyboard focus
        // and the single-field `active_modal` covers all variants.
        if self.modal_overlay_blocks_editor_shortcuts() {
            return Task::none();
        }
        self.file_tree.tree_focused = !self.file_tree.tree_focused;
        if self.file_tree.tree_focused && self.file_tree.visible_tree_nodes.is_empty() {
            self.file_tree.rebuild_visible();
        }
        if !self.file_tree.tree_focused || self.file_tree.visible_tree_nodes.is_empty() {
            self.file_tree.tree_focused = false;
            self.pending_enter_dir = None;
        }
        Task::none()
    }

    /// Handle tree-scrolled — updates scroll state of the file tree.
    fn tree_scrolled(&mut self, scroll_y: f32, viewport_h: f32) -> Task<EditorMessage> {
        self.file_tree.scroll_y = scroll_y;
        self.file_tree.viewport_h = Some(viewport_h);
        Task::none()
    }

    /// Handle tree-nav-enter — opens file or expands/collapses directory.
    fn tree_nav_enter(&mut self) -> Task<EditorMessage> {
        // When global search or quick-open is active, Enter selects the
        // highlighted result / file.  Borrow to extract the index without
        // cloning the entire state.
        match &self.active_modal {
            Some(ModalKind::GlobalSearch(gs)) => {
                let idx = gs.selected_index.min(gs.results.len().saturating_sub(1));
                return Task::done(EditorMessage::GlobalSearchSelect(idx));
            }
            Some(ModalKind::QuickOpen(qo)) => {
                let idx = qo.selected_index.min(qo.results.len().saturating_sub(1));
                return Task::done(EditorMessage::QuickOpenSelect(idx));
            }
            _ => {}
        }
        // When any modal overlay (Rename, GotoLine, NewItem, DeleteConfirm,
        // CloseDialog, etc.) is active, suppress tree navigation — the
        // overlay handles its own Enter key handling.  Must be placed
        // AFTER the search redirects above so Enter-to-select still works
        // in GlobalSearch and QuickOpen.
        if self.modal_overlay_blocks_editor_shortcuts() {
            return Task::none();
        }
        let Some((_idx, path, is_dir)) = self.file_tree.focused_tree_node() else {
            return Task::none();
        };
        if self.file_tree.focused_is_expanded_dir() {
            // Collapse: rebuild and keep focus on the collapsed directory.
            return self.collapse_dir(&path);
        }
        if is_dir {
            // Expand: insert, rebuild, jump to first child.
            return self.expand_dir_and_focus(&path, "TreeNavEnter");
        }
        // Open file.
        Task::done(EditorMessage::SelectFile(path))
    }

    /// Handle tree-nav-left — collapses expanded directory or navigates to parent.
    fn tree_nav_left(&mut self) -> Task<EditorMessage> {
        // Suppress during active modal overlays — the overlay handles
        // its own keyboard navigation (covers Rename, GotoLine, etc.).
        if self.modal_overlay_blocks_editor_shortcuts() {
            return Task::none();
        }
        let Some((_idx, path, _)) = self.file_tree.focused_tree_node() else {
            return Task::none();
        };

        if self.file_tree.focused_is_expanded_dir() {
            // Collapse expanded directory and keep focus on it.
            return self.collapse_dir(&path);
        }

        // ArrowLeft on collapsed directory or file — navigate to parent.
        match self.file_tree.focused_parent_path() {
            Some(ref p) if self.file_tree.focus_path(p).is_some() => {
                return widgets::scroll_to_tree_focus(
                    &mut self.file_tree,
                    widgets::ScrollMode::SnapToTop,
                );
            }
            _ => {} // Root-level item has no parent — no-op.
        }
        Task::none()
    }

    /// Handle tree-nav-right — expands directory or navigates to first child.
    fn tree_nav_right(&mut self) -> Task<EditorMessage> {
        // Suppress during active modal overlays — the overlay handles
        // its own keyboard navigation (covers Rename, GotoLine, etc.).
        if self.modal_overlay_blocks_editor_shortcuts() {
            return Task::none();
        }
        let Some((idx, path, is_dir)) = self.file_tree.focused_tree_node() else {
            return Task::none();
        };

        if !is_dir {
            // ArrowRight on a file does nothing.
            return Task::none();
        }

        if !self.file_tree.expanded_dirs.contains(&path) {
            // Expand directory and move focus to first child.
            return self.expand_dir_and_focus(&path, "TreeNavRight");
        }

        // Already expanded directory — move focus to first child (if any).
        if idx + 1 < self.file_tree.visible_tree_nodes.len() {
            self.file_tree.tree_focus_index = idx + 1;
            return widgets::scroll_to_tree_focus(
                &mut self.file_tree,
                widgets::ScrollMode::SnapToTop,
            );
        }
        Task::none()
    }

    /// Handle find-toggle — opens/closes the find/replace bar.
    fn find_toggle(&mut self) -> Task<EditorMessage> {
        if self.modal_overlay_blocks_editor_shortcuts() {
            return Task::none();
        }
        let Some((_, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if tab_data.find_replace_state.is_none() {
                // Open find bar with current selection as default query.
                let default_query = tab_data.content.selection().unwrap_or_default();
                let mut state = FindReplaceState {
                    query: default_query,
                    replace: String::new(),
                    matches: Vec::new(),
                    current_match_idx: 0,
                    case_sensitive: false,
                };
                // Compute matches if query is non-empty.
                if !state.query.is_empty() {
                    let text = tab_data.content.text();
                    state.matches = compute_text_matches(&text, &state.query, state.case_sensitive);
                    // Auto-jump to first match.
                    auto_jump_to_first_match(&tab_data.content, &mut state);
                }
                // Close go-to-line when opening find bar (mutually exclusive).
                if matches!(self.active_modal, Some(ModalKind::GotoLine(_))) {
                    self.active_modal = None;
                }
                tab_data.find_replace_state = Some(state);
            }
            // Already open — re-focus the search input (no state change needed).
        }
        // Always focus the search input when FindToggle is pressed.
        iced::widget::operation::focus::<EditorMessage>(Id::new(FIND_SEARCH_ID))
    }

    /// Handle find-query-input — updates the search query and recomputes matches.
    fn find_query_input(&mut self, query: String) -> Task<EditorMessage> {
        let Some((_, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if let Some(ref mut state) = tab_data.find_replace_state {
                state.query = query;
                let text = tab_data.content.text();
                state.matches = compute_text_matches(&text, &state.query, state.case_sensitive);
                auto_jump_to_first_match(&tab_data.content, state);
            }
        }
        Task::none()
    }

    /// Handle find-replace-input — updates the replace text.
    fn find_replace_input(&mut self, replace: String) -> Task<EditorMessage> {
        let Some((_, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if let Some(ref mut state) = tab_data.find_replace_state {
                state.replace = replace;
            }
        }
        Task::none()
    }

    /// Handle find-replace — replaces the current match and advances to the next.
    fn find_replace(&mut self) -> Task<EditorMessage> {
        let Some((idx, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if let Some(ref state) = tab_data.find_replace_state {
                if let Some(range) = state.matches.get(state.current_match_idx) {
                    let replace_text = state.replace.clone();
                    let replace_end = range.start + replace_text.len();
                    if !replace_text.is_empty() || range.start < range.end {
                        // Take undo snapshot.
                        tab_data
                            .undo_stack
                            .borrow_mut()
                            .snap_before_edit(&tab_data.content);
                        let text = tab_data.content.text();
                        let new_text = format!(
                            "{}{}{}",
                            &text[..range.start],
                            replace_text,
                            &text[range.end..]
                        );
                        tab_data.content = EditorBuffer::from_file(&new_text, &path);
                        // Recompute matches and auto-advance to next match.
                        if let Some(ref mut state) = tab_data.find_replace_state {
                            state.matches =
                                compute_text_matches(&new_text, &state.query, state.case_sensitive);
                            if !state.matches.is_empty() {
                                // Advance to the next match starting at or
                                // after the end of the replacement in the
                                // new text (position = range.start + len(replace_text)).
                                // Using a position in the old text (range.end)
                                // would be wrong when replacement length differs
                                // from the original match length.
                                let next_idx = state
                                    .matches
                                    .iter()
                                    .position(|m| m.start >= replace_end)
                                    .unwrap_or(0)
                                    .min(state.matches.len() - 1);
                                state.current_match_idx = next_idx;
                                // Position cursor at the new match.
                                if let Some(r) = state.matches.get(next_idx) {
                                    if let Some((line, col)) =
                                        byte_offset_to_cursor_pos(&tab_data.content, r.start)
                                    {
                                        tab_data.content.move_to(line, col);
                                    }
                                }
                            } else {
                                state.current_match_idx = 0;
                                // No remaining matches — place cursor at end
                                // of the replacement, not at buffer start.
                                if let Some((line, col)) =
                                    byte_offset_to_cursor_pos(&tab_data.content, replace_end)
                                {
                                    tab_data.content.move_to(line, col);
                                }
                            }
                        }
                    }
                }
            }
        }
        update_dirty_flag(&mut self.tabs, &self.tab_contents, idx, &path);
        Task::none()
    }

    /// Handle find-toggle-case-sensitivity — toggles case-sensitive search.
    fn find_toggle_case_sensitivity(&mut self) -> Task<EditorMessage> {
        let Some((_, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if let Some(ref mut state) = tab_data.find_replace_state {
                state.case_sensitive = !state.case_sensitive;
                // Recompute matches with new case sensitivity.
                let text = tab_data.content.text();
                state.matches = compute_text_matches(&text, &state.query, state.case_sensitive);
                auto_jump_to_first_match(&tab_data.content, state);
            }
        }
        Task::none()
    }

    /// Handle refresh-file-tree — re-reads all expanded directories from disk.
    fn refresh_file_tree(&mut self) -> Task<EditorMessage> {
        // Suppress during active modal overlays — the file tree should
        // not refresh behind an active overlay.  This covers both the
        // Cmd+R / Ctrl+R keyboard shortcut AND the periodic 30-second
        // timer subscription.
        if self.modal_overlay_blocks_editor_shortcuts() {
            return Task::none();
        }
        let Some(ref ws_path) = self.selected_workspace_path else {
            return Task::none();
        };

        // Collect directories to refresh: root + all expanded dirs.
        let mut dirs_to_refresh: Vec<String> = Vec::new();

        // Root directory (empty string) is always included — it's
        // implicitly expanded and not tracked in expanded_dirs.
        dirs_to_refresh.push(String::new());

        // All manually expanded directories.
        dirs_to_refresh.extend(self.file_tree.expanded_dirs.iter().cloned());

        // Filter out directories currently being loaded by the user
        // (e.g., from a ToggleDir or TreeNavEnter action). This avoids
        // racing user-initiated async loads. Generation counters also
        // protect against races, but skipping in-flight dirs avoids
        // wasted I/O.
        dirs_to_refresh.retain(|d| !self.loading_dirs.contains(d));

        if dirs_to_refresh.is_empty() {
            return Task::none();
        }

        let mut tasks: Vec<Task<EditorMessage>> = Vec::new();
        let root_path = ws_path.clone();

        for dir_path in dirs_to_refresh {
            let dir_gen = self.generation.wrapping_add(1);
            self.generation = dir_gen;
            self.dir_generations.insert(dir_path.clone(), dir_gen);
            // NOTE: deliberately NOT adding to `loading_dirs` — this
            // avoids a "Loading…" flicker for every expanded directory
            // on every background refresh. The tree silently updates
            // when results arrive via DirExpanded.

            let d_path = dir_path.clone();
            let r_path = root_path.clone();
            tasks.push(Task::perform(
                async move {
                    let entries = read_directory_entries(&r_path, &d_path).await;
                    EditorMessage::DirExpanded {
                        dir_path: d_path,
                        r#gen: dir_gen,
                        entries,
                        quiet: true,
                    }
                },
                |msg| msg,
            ));
        }

        // Kick off a git status refresh so newly discovered files
        // get their git status colors without waiting for the next Tick.
        if !self.git_status_loading {
            self.git_status_loading = true;
            let path = root_path.clone();
            tasks.push(Task::perform(
                async move { load_git_status(path).await },
                EditorMessage::GitStatusLoaded,
            ));
        }

        Task::batch(tasks)
    }

    /// Handle tick — refreshes git status and gitignore for file tree coloring.
    fn tick(&mut self) -> Task<EditorMessage> {
        // Refresh git status and gitignore for file tree coloring.
        if let Some(ref ws_path) = self.selected_workspace_path {
            let mut tasks: Vec<Task<EditorMessage>> = Vec::new();

            if !self.git_status_loading {
                self.git_status_loading = true;
                let path = ws_path.clone();
                tasks.push(Task::perform(
                    async move { load_git_status(path).await },
                    EditorMessage::GitStatusLoaded,
                ));
            }

            if !self.git_ignore_loading {
                self.git_ignore_loading = true;
                let path = ws_path.clone();
                let tree_paths = collect_tree_paths(&self.file_tree.nodes);
                tasks.push(Task::perform(
                    async move { load_git_ignore(path, tree_paths).await },
                    EditorMessage::GitIgnoredLoaded,
                ));
            }

            Task::batch(tasks)
        } else {
            Task::none()
        }
    }

    /// Handle blink-tick — increments the blink generation counter.
    fn blink_tick(&mut self) -> Task<EditorMessage> {
        // Increment the blink generation counter to force Iced
        // to redraw the editor widget. Iced 0.14 may skip redrawing
        // unchanged widgets when only request_redraw_at is used;
        // this counter ensures the widget is re-evaluated on each
        // BlinkTick (every 100 ms), keeping the cursor blink alive
        // even if the RedrawRequested chain breaks.
        self.blink_gen = self.blink_gen.wrapping_add(1);
        Task::none()
    }

    /// Handle git-status-loaded — updates the git status cache.
    fn git_status_loaded(
        &mut self,
        result: Result<HashMap<String, GitFileStatus>, String>,
    ) -> Task<EditorMessage> {
        self.git_status_loading = false;
        match result {
            Ok(cache) => self.git_status_cache = cache,
            Err(e) => {
                tracing::warn!("Failed to load git status: {e}");
                self.git_status_cache.clear();
            }
        }
        Task::none()
    }

    /// Handle git-ignored-loaded — updates the git ignore cache.
    fn git_ignored_loaded(
        &mut self,
        result: Result<HashSet<String>, String>,
    ) -> Task<EditorMessage> {
        self.git_ignore_loading = false;
        match result {
            Ok(cache) => self.git_ignore_cache = cache,
            Err(e) => {
                tracing::warn!("Failed to load git ignore status: {e}");
                self.git_ignore_cache.clear();
            }
        }
        Task::none()
    }

    /// Handle check-file-changes — detects external file modifications and reloads.
    fn check_file_changes(&mut self) -> Task<EditorMessage> {
        let Some((idx, path)) = self.active_tab() else {
            return Task::none();
        };
        // Only auto-refresh tabs that are not dirty.
        if self.tabs[idx].is_dirty {
            return Task::none();
        }

        let current_mtime = if let Ok(meta) = std::fs::metadata(&path) {
            meta.modified().ok()
        } else {
            // File doesn't exist (deleted or moved).
            if !self.deleted_file_toasted.contains(&path) {
                self.deleted_file_toasted.insert(path);
                return Task::done(EditorMessage::Toast(super::ToastMessage::Warning(format!(
                    "File was deleted: {}",
                    self.tabs[idx].file_name
                ))));
            }
            return Task::none();
        };

        // File exists — if it was previously reported as deleted,
        // clear that flag (file has been recreated).
        self.deleted_file_toasted.remove(&path);

        let Some(current_mtime) = current_mtime else {
            // Cannot determine mtime on this platform — skip.
            return Task::none();
        };

        let stored_mtime = if let Some(m) = self.file_mtimes.get(&path) {
            *m
        } else {
            // No stored mtime yet — record it now and skip.
            self.file_mtimes.insert(path, current_mtime);
            return Task::none();
        };

        // Only re-read if mtime actually changed.
        if current_mtime == stored_mtime {
            return Task::none();
        }

        // Mtime changed — capture cursor position and reload async.
        let cursor = if let Some(tab_data) = self.tab_contents.get(&path) {
            tab_data.content.cursor()
        } else {
            return Task::none();
        };

        // Start the async read.
        Task::perform(
            async move {
                let result = match tokio::fs::read_to_string(&path).await {
                    Ok(text) => validate_file_content(text.as_bytes()).map(|()| text),
                    Err(e) => Err(format!("Cannot read file: {e}")),
                };
                EditorMessage::FileReloaded {
                    path,
                    result,
                    cursor_line: cursor.line,
                    cursor_col: cursor.column,
                }
            },
            |msg| msg,
        )
    }

    /// Handle file-reloaded — replaces tab content with the reloaded file data.
    fn file_reloaded(
        &mut self,
        path: String,
        result: Result<String, String>,
        cursor_line: usize,
        cursor_col: usize,
    ) -> Task<EditorMessage> {
        // Guard: the tab must still be the active one and not dirty.
        let Some(idx) = self.active_tab_idx() else {
            return Task::none();
        };
        if self.tabs[idx].path != path || self.tabs[idx].is_dirty {
            return Task::none();
        }

        let task = match result {
            Ok(text) => {
                let has_trailing = has_trailing_newline(&text);
                let line_ending = detect_line_ending(&text);

                // Update tab metadata.
                if let Some(tab) = self.tabs.get_mut(idx) {
                    tab.is_dirty = false;
                    tab.has_trailing_newline = has_trailing;
                    tab.line_ending = line_ending;
                }

                // Replace content, preserving cursor position (clamped).
                if let Some(tab_data) = self.tab_contents.get_mut(&path) {
                    // Clear find/replace state — match byte ranges are now stale.
                    tab_data.find_replace_state = None;
                    tab_data.content = EditorBuffer::from_file(&text, &path);
                    // Restore cursor, clamped to new file bounds.
                    tab_data.content.move_to(cursor_line, cursor_col);
                    // Clear undo stack — new content didn't come from user edits.
                    *tab_data.undo_stack.borrow_mut() = UndoStack::new();
                    tab_data.saved_text_hash = hash_text(&text);
                }

                Task::none()
            }
            Err(e) => Task::done(EditorMessage::Toast(super::ToastMessage::Warning(e))),
        };

        // Update the stored mtime so the next tick matches
        // and won't retry every 300 ms — even on failure,
        // this prevents repeated read attempts.
        if let Ok(meta) = std::fs::metadata(&path) {
            if let Ok(mtime) = meta.modified() {
                self.file_mtimes.insert(path, mtime);
            }
        }

        task
    }

    /// Navigate to the next or previous find match in the active tab.
    /// Returns silently if there is no active tab, no find state, or no matches.
    fn navigate_find_match(&mut self, direction: &FindDirection) -> Task<EditorMessage> {
        let Some((_, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if let Some(ref mut state) = tab_data.find_replace_state {
                if !state.matches.is_empty() {
                    let new_idx = match direction {
                        FindDirection::Next => (state.current_match_idx + 1) % state.matches.len(),
                        FindDirection::Prev => {
                            if state.current_match_idx == 0 {
                                state.matches.len().saturating_sub(1)
                            } else {
                                state.current_match_idx - 1
                            }
                        }
                    };
                    state.current_match_idx = new_idx;
                    if let Some(range) = state.matches.get(new_idx) {
                        if let Some((line, col)) =
                            byte_offset_to_cursor_pos(&tab_data.content, range.start)
                        {
                            tab_data.content.move_to(line, col);
                        }
                    }
                }
            }
        }
        Task::none()
    }

    /// Shared helper for navigating search results — adjusts the selected index
    /// based on the direction, staying within bounds.
    fn navigate_search_results(
        selected_index: &mut usize,
        results_len: usize,
        direction: &TreeNavDirection,
    ) {
        match *direction {
            TreeNavDirection::Up if *selected_index > 0 => *selected_index -= 1,
            TreeNavDirection::Down if *selected_index + 1 < results_len => *selected_index += 1,
            _ => {}
        }
    }

    /// Navigate vertically in the active overlay or file tree.
    ///
    /// Handles global-search results, quick-open results, and file-tree focus
    /// in priority order. Only the file-tree path returns a scroll-to-focus
    /// task; the overlay paths return `Task::none()`.
    fn navigate_tree_vertical(&mut self, direction: &TreeNavDirection) -> Task<EditorMessage> {
        // When global search is active, navigate the results list.
        if let Some(ModalKind::GlobalSearch(ref mut gs)) = self.active_modal {
            Self::navigate_search_results(&mut gs.selected_index, gs.results.len(), direction);
            return Task::none();
        }
        // When quick-open is active, navigate the results list.
        if let Some(ModalKind::QuickOpen(ref mut qo)) = self.active_modal {
            Self::navigate_search_results(&mut qo.selected_index, qo.results.len(), direction);
            return Task::none();
        }
        // When another modal overlay (GotoLine, NewItem, DeleteConfirm,
        // CloseDialog, etc.) is active, suppress tree navigation.  The search
        // overlay redirects above have already returned, so only non-search
        // overlays reach this guard.
        if self.modal_overlay_blocks_editor_shortcuts() {
            return Task::none();
        }
        // Navigate the file tree focus index.
        if self.file_tree.tree_focused {
            match *direction {
                TreeNavDirection::Up if self.file_tree.tree_focus_index > 0 => {
                    self.file_tree.tree_focus_index -= 1;
                    return widgets::scroll_to_tree_focus(
                        &mut self.file_tree,
                        widgets::ScrollMode::ScrollIntoView,
                    );
                }
                TreeNavDirection::Down
                    if self.file_tree.tree_focus_index + 1
                        < self.file_tree.visible_tree_nodes.len() =>
                {
                    self.file_tree.tree_focus_index += 1;
                    return widgets::scroll_to_tree_focus(
                        &mut self.file_tree,
                        widgets::ScrollMode::ScrollIntoView,
                    );
                }
                _ => {}
            }
        }
        Task::none()
    }

    #[allow(clippy::too_many_lines)]
    pub fn view(&self) -> Element<'_, EditorMessage> {
        // ── No workspace selected — placeholder ──────────────────────
        if self.selected_workspace_name.is_none() {
            return empty_placeholder(
                text("No workspace selected")
                    .size(24)
                    .color(theme::TEXT_MUTED)
                    .font(theme::FONT_BOLD),
            );
        }

        // ── Split layout ─────────────────────────────────────────────
        let tree_panel = self.build_tree_panel();
        let editor_panel = self.build_editor_panel();

        let split = row![tree_panel, editor_panel]
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fill);

        // ── Overlay (single match on active_modal) ────────────────────
        let col_children: Vec<Element<'_, EditorMessage>> = vec![split.into()];

        let body = column(col_children)
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fill);

        // Keep Stack widget type stable — a Column→Stack type change between frames
        // destroys widget state (scroll positions, ContextMenu overlay states),
        // causing stale overlay-to-tab associations. Always return a Stack
        // with a zero-size placeholder when no overlay is present.
        let placeholder: Element<'_, EditorMessage> = container(text(""))
            .width(Length::Shrink)
            .height(Length::Shrink)
            .into();

        let overlay: Element<'_, EditorMessage> = match &self.active_modal {
            Some(ModalKind::CloseDialog(tab_idx)) => Self::build_close_modal(
                EditorMessage::CloseDialog {
                    tab_index: *tab_idx,
                    action: CloseAction::Save,
                },
                EditorMessage::CloseDialog {
                    tab_index: *tab_idx,
                    action: CloseAction::Discard,
                },
                EditorMessage::CloseDialog {
                    tab_index: *tab_idx,
                    action: CloseAction::Cancel,
                },
                "This file has unsaved changes. What would you like to do?".to_string(),
            ),
            Some(ModalKind::CloseOthers(keep_idx)) => {
                let dirty_count = self
                    .tabs
                    .iter()
                    .enumerate()
                    .filter(|(i, t)| *i != *keep_idx && t.is_dirty)
                    .count();
                let desc = if dirty_count == 1 {
                    "1 file has unsaved changes. What would you like to do?".to_string()
                } else {
                    format!("{dirty_count} files have unsaved changes. What would you like to do?")
                };
                Self::build_close_modal(
                    EditorMessage::CloseOthersDialog {
                        keep_idx: *keep_idx,
                        action: CloseAction::Save,
                    },
                    EditorMessage::CloseOthersDialog {
                        keep_idx: *keep_idx,
                        action: CloseAction::Discard,
                    },
                    EditorMessage::CloseOthersDialog {
                        keep_idx: *keep_idx,
                        action: CloseAction::Cancel,
                    },
                    desc,
                )
            }
            Some(ModalKind::GlobalSearch(gs)) => editor_dialog::overlay_dialog(
                Self::build_global_search_overlay(gs),
                EditorMessage::GlobalSearchClose,
                0.3,
            ),
            Some(ModalKind::QuickOpen(qo)) => editor_dialog::overlay_dialog(
                editor_dialog::build_quick_open_overlay(qo),
                EditorMessage::Escape,
                0.3,
            ),
            Some(ModalKind::NewItem(target)) => editor_dialog::wrap_dialog(
                editor_dialog::build_new_item_input(target),
                400,
                EditorMessage::Escape,
                0.5,
            ),
            Some(ModalKind::DeleteConfirm(target)) => editor_dialog::wrap_dialog(
                editor_dialog::build_delete_confirm_dialog(target),
                400,
                EditorMessage::CancelDelete,
                0.5,
            ),
            // GotoLine and Rename are rendered inline (not as stack overlays).
            Some(ModalKind::GotoLine(_) | ModalKind::Rename(_)) | None => placeholder,
        };

        iced::widget::stack([body.into(), overlay]).into()
    }

    /// Build a close confirmation modal with consistent sizing and escape behavior.
    fn build_close_modal(
        save_msg: EditorMessage,
        discard_msg: EditorMessage,
        cancel_msg: EditorMessage,
        desc: String,
    ) -> Element<'static, EditorMessage> {
        editor_dialog::wrap_dialog(
            editor_dialog::build_close_dialog(save_msg, discard_msg, cancel_msg, desc),
            420,
            EditorMessage::Escape,
            0.5,
        )
    }

    // ── Tree panel ────────────────────────────────────────────────

    fn build_tree_panel(&self) -> Element<'_, EditorMessage> {
        let elements: Vec<Element<'_, EditorMessage>> = self
            .file_tree
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| self.render_tree_node(n, 0, 0, i == self.file_tree.nodes.len() - 1))
            .collect();
        let panel = widgets::build_tree_panel(&self.file_tree, elements, |viewport| {
            EditorMessage::TreeScrolled(viewport.absolute_offset().y, viewport.bounds().height)
        });

        // Wrap the tree panel with a context menu that fires on empty-space
        // right-clicks. When the user right-clicks on a tree node, the inner
        // node-level ContextMenu captures the event, so this outer fallback
        // does not fire. When right-clicking on empty space below the nodes,
        // no inner ContextMenu captures it, so this one shows the menu.
        ContextMenu::new(
            panel,
            vec![
                (
                    "New File".into(),
                    EditorMessage::NewFileRequested(String::new()),
                ),
                (
                    "New Directory".into(),
                    EditorMessage::NewDirectoryRequested(String::new()),
                ),
            ],
        )
        .into()
    }

    /// Check if any child (file or expanded dir) in the node has a git status.
    /// Returns the most "interesting" status: Modified > Added. Only meaningful
    /// for expanded directories (which have children populated).
    fn dir_git_status(&self, node: &widgets::TreeNode) -> Option<GitFileStatus> {
        let mut best: Option<GitFileStatus> = None;
        for child in &node.children {
            if child.is_dir {
                // For expanded subdirectories, recurse into their children.
                if let Some(status) = self.dir_git_status(child) {
                    if best != Some(GitFileStatus::Modified) {
                        best = Some(status);
                    }
                }
            } else {
                match self.git_status_cache.get(&child.full_path) {
                    Some(&GitFileStatus::Modified) => return Some(GitFileStatus::Modified),
                    Some(&GitFileStatus::Added) => {
                        best = Some(GitFileStatus::Added);
                    }
                    None => {}
                }
            }
        }
        best
    }

    /// Check whether a path is gitignored, either directly or because an
    /// ancestor directory is in the gitignore cache (directory inheritance).
    #[must_use]
    fn is_path_ignored(&self, full_path: &str) -> bool {
        if self.git_ignore_cache.is_empty() {
            return false;
        }
        if self.git_ignore_cache.contains(full_path) {
            return true;
        }
        // Walk up the path tree: if any parent directory is ignored,
        // the child inherits that status.
        let mut path = full_path;
        while let Some(pos) = path.rfind('/') {
            path = &path[..pos];
            if self.git_ignore_cache.contains(path) {
                return true;
            }
        }
        false
    }

    /// Shared tree-node row helper.  Builds the guide-lines + icon + name row,
    /// wraps it in a `tree_node_button`, then wraps that in a `ContextMenu`
    /// with caller-specific items prepended before the common items
    /// (Copy Relative Path, Copy Absolute Path, Reveal in Finder).
    ///
    /// # Parameters
    ///
    /// * `guide` — pre-computed tree guide-line prefix string (empty for root-level
    ///   nodes, otherwise contains box-drawing characters for hierarchy lines).
    /// * `icon` — pre-built icon element (size and colour already set).
    /// * `name` — pre-built name element (text content and style already set).
    /// * `highlight` — whether the row should show the highlight style.
    /// * `message` — message to fire when the row is clicked.
    /// * `extra_context_items` — caller-specific context menu items; they are
    ///   placed *before* the three shared items listed above.
    /// * `full_path` — workspace-relative path used to compute absolute/relative
    ///   paths for the shared context menu items.
    #[allow(clippy::too_many_arguments)]
    fn render_tree_node_row<'a>(
        &'a self,
        guide: String,
        icon: Element<'a, EditorMessage>,
        name: Element<'a, EditorMessage>,
        highlight: bool,
        message: EditorMessage,
        extra_context_items: Vec<(String, EditorMessage)>,
        full_path: &str,
    ) -> Element<'a, EditorMessage> {
        let guide_text: Element<'a, EditorMessage> = text(guide)
            .size(widgets::TREE_FONT_SIZE)
            .color(theme::TEXT_MUTED)
            .into();

        let row = row![
            guide_text,
            icon,
            Space::new().width(4),
            name,
            Space::new().width(Length::Fill),
        ]
        .align_y(Alignment::Center)
        .padding([0, 8]);

        let btn = widgets::tree_node_button(row, highlight, Some(message));

        let rel_path = full_path.to_string();

        let mut menu_items: Vec<(String, EditorMessage)> = extra_context_items;
        menu_items.push((
            "Copy Relative Path".into(),
            EditorMessage::CopyRelativePath(rel_path),
        ));

        if let Some(abs_path) = self.abs_path(full_path) {
            menu_items.push((
                "Copy Absolute Path".into(),
                EditorMessage::CopyAbsolutePath(abs_path.clone()),
            ));
            menu_items.push((
                "Reveal in Finder".into(),
                EditorMessage::RevealInFinder(abs_path),
            ));
        }

        ContextMenu::new(btn, menu_items).into()
    }

    fn render_tree_node<'a>(
        &'a self,
        node: &'a widgets::TreeNode,
        depth: usize,
        ancestor_mask: u64,
        is_last: bool,
    ) -> Element<'a, EditorMessage> {
        if node.is_dir {
            self.render_dir_node(node, depth, ancestor_mask, is_last)
        } else {
            self.render_file_node(node, depth, ancestor_mask, is_last)
        }
    }

    #[allow(clippy::too_many_lines)]
    fn render_dir_node<'a>(
        &'a self,
        node: &'a widgets::TreeNode,
        depth: usize,
        ancestor_mask: u64,
        is_last: bool,
    ) -> Element<'a, EditorMessage> {
        let is_expanded = self.file_tree.expanded_dirs.contains(&node.full_path);
        let is_loading = self.loading_dirs.contains(&node.full_path);
        let is_ignored = self.is_path_ignored(&node.full_path);
        let icon = if is_expanded {
            lucide::folder_open()
        } else {
            lucide::folder()
        };
        let dir_status = if is_expanded && !is_loading {
            self.dir_git_status(node)
        } else {
            None
        };
        let icon_color = if is_ignored {
            theme::TEXT_MUTED
        } else if is_expanded && dir_status.is_some() {
            match dir_status {
                Some(GitFileStatus::Modified) => theme::STATUS_WARNING,
                Some(GitFileStatus::Added) => theme::STATUS_SUCCESS,
                _ => theme::ACCENT_LIGHT,
            }
        } else if is_expanded {
            theme::ACCENT_LIGHT
        } else {
            theme::TEXT_MUTED
        };

        let (label_text, label_color) = if is_loading {
            (format!("{}  Loading…", node.name), theme::TEXT_MUTED)
        } else if let Some(ref err) = node.error {
            (format!("{} [⚠ {err}]", node.name), theme::STATUS_ERROR)
        } else if is_ignored {
            (node.name.clone(), theme::TEXT_MUTED)
        } else if dir_status.is_some() {
            let color = match dir_status {
                Some(GitFileStatus::Modified) => theme::STATUS_WARNING,
                Some(GitFileStatus::Added) => theme::STATUS_SUCCESS,
                _ => theme::TEXT_SECONDARY,
            };
            (node.name.clone(), color)
        } else {
            (node.name.clone(), theme::TEXT_SECONDARY)
        };

        let is_focused = widgets::tree_node_focused(&self.file_tree, &node.full_path);

        let icon_element: Element<'_, EditorMessage> =
            icon.size(widgets::TREE_ICON_SIZE).color(icon_color).into();
        let name_element: Element<'_, EditorMessage> =
            self.build_rename_input(node).unwrap_or_else(|| {
                text(label_text)
                    .size(widgets::TREE_FONT_SIZE)
                    .color(label_color)
                    .into()
            });

        let guide = widgets::tree_guide_prefix(ancestor_mask, depth, is_last);
        let ctx_menu = self.render_tree_node_row(
            guide,
            icon_element,
            name_element,
            is_focused,
            EditorMessage::ToggleDir(node.full_path.clone()),
            vec![
                (
                    "New File".into(),
                    EditorMessage::NewFileRequested(node.full_path.clone()),
                ),
                (
                    "New Directory".into(),
                    EditorMessage::NewDirectoryRequested(node.full_path.clone()),
                ),
                (
                    "Rename".into(),
                    EditorMessage::RenameRequested(node.full_path.clone()),
                ),
                (
                    "Delete".into(),
                    EditorMessage::DeleteDirectoryRequested(node.full_path.clone()),
                ),
            ],
            &node.full_path,
        );

        let mut col = column![ctx_menu].spacing(0);
        if is_expanded {
            for elem in widgets::render_tree_children(
                &node.children,
                depth,
                ancestor_mask,
                is_last,
                |child, d, mask, last| self.render_tree_node(child, d, mask, last),
            ) {
                col = col.push(elem);
            }
        }
        col.into()
    }

    fn render_file_node<'a>(
        &'a self,
        node: &'a widgets::TreeNode,
        depth: usize,
        ancestor_mask: u64,
        is_last: bool,
    ) -> Element<'a, EditorMessage> {
        let is_selected = self.selected_file.as_deref() == Some(&node.full_path);

        let guide = widgets::tree_guide_prefix(ancestor_mask, depth, is_last);

        let icon = lucide::file::<iced::Theme, iced::Renderer>();
        let is_ignored = self.is_path_ignored(&node.full_path);
        let icon_color = if is_selected {
            theme::ACCENT
        } else if is_ignored {
            theme::TEXT_FAINT
        } else {
            theme::TEXT_MUTED
        };

        let git_status = self.git_status_cache.get(&node.full_path);
        let name_color = if is_selected {
            theme::TEXT_PRIMARY
        } else if node.error.is_some() {
            theme::STATUS_ERROR
        } else if is_ignored {
            theme::TEXT_MUTED
        } else if git_status == Some(&GitFileStatus::Modified) {
            theme::STATUS_WARNING
        } else if git_status == Some(&GitFileStatus::Added) {
            theme::STATUS_SUCCESS
        } else {
            theme::TEXT_SECONDARY
        };
        let name_weight = if is_selected {
            iced::font::Weight::Bold
        } else {
            iced::font::Weight::Normal
        };

        let name_text: Element<'a, EditorMessage> =
            self.build_rename_input(node).unwrap_or_else(|| {
                if node.error.is_some() {
                    row![
                        text(&node.name)
                            .size(widgets::TREE_FONT_SIZE)
                            .color(name_color)
                            .font(iced::Font {
                                weight: name_weight,
                                ..theme::FONT_REGULAR
                            }),
                        Space::new().width(4),
                        text("[⚠]").size(11).color(theme::STATUS_ERROR),
                    ]
                    .align_y(Alignment::Center)
                    .into()
                } else {
                    text(&node.name)
                        .size(widgets::TREE_FONT_SIZE)
                        .color(name_color)
                        .font(iced::Font {
                            weight: name_weight,
                            ..theme::FONT_REGULAR
                        })
                        .into()
                }
            });

        let is_focused = widgets::tree_node_focused(&self.file_tree, &node.full_path);

        let icon_element: Element<'_, EditorMessage> =
            icon.size(widgets::TREE_FONT_SIZE).color(icon_color).into();

        self.render_tree_node_row(
            guide,
            icon_element,
            name_text,
            is_selected || is_focused,
            EditorMessage::SelectFile(node.full_path.clone()),
            vec![
                (
                    "Rename".into(),
                    EditorMessage::RenameRequested(node.full_path.clone()),
                ),
                (
                    "Delete".into(),
                    EditorMessage::DeleteFileRequested(node.full_path.clone()),
                ),
            ],
            &node.full_path,
        )
    }

    // ── Editor panel ──────────────────────────────────────────────

    fn build_editor_panel(&self) -> Element<'_, EditorMessage> {
        if self.tabs.is_empty() {
            return empty_placeholder(
                text("Select a file to edit")
                    .size(18)
                    .color(theme::TEXT_MUTED),
            );
        }

        let tab_bar = self.build_tab_bar();
        let find_bar = self.build_find_replace_bar();
        let go_to_line = self.build_go_to_line_bar();
        let editor_widget = self.build_editor_widget();

        let mut col = column![tab_bar].spacing(0).width(Length::Fill);
        if let Some(bar) = find_bar {
            col = col.push(bar);
        } else if let Some(bar) = go_to_line {
            // Go-to-line uses the same UI slot; only one bar visible at a time.
            col = col.push(bar);
        }
        col = col.push(editor_widget);

        col.height(Length::Fill).into()
    }

    #[allow(clippy::too_many_lines)]
    fn build_tab_bar(&self) -> Element<'_, EditorMessage> {
        let mut tab_buttons: Vec<Element<'_, EditorMessage>> = Vec::new();

        for (i, tab) in self.tabs.iter().enumerate() {
            let is_active = i == self.active_tab_index;

            // Dirty indicator dot.
            let dirty_dot: Option<Element<'_, EditorMessage>> = if tab.is_dirty {
                Some(
                    lucide::circle::<iced::Theme, iced::Renderer>()
                        .size(8)
                        .color(theme::STATUS_WARNING)
                        .into(),
                )
            } else {
                None
            };

            let name_color = if is_active {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            };
            let name_text = text(&tab.file_name).size(12).color(name_color);

            let close_btn = button(lucide::x::<iced::Theme, iced::Renderer>().size(12).color(
                if is_active {
                    theme::TEXT_SECONDARY
                } else {
                    theme::TEXT_FAINT
                },
            ))
            .on_press(EditorMessage::TabClosed(i))
            .style(theme::button_transparent)
            .padding(0);

            let mut tab_row = row![].spacing(2).align_y(Alignment::Center);
            if let Some(dot) = dirty_dot {
                tab_row = tab_row.push(dot);
            }
            tab_row = tab_row.push(name_text).push(close_btn);

            let tab_btn = button(tab_row.padding([8, 8]))
                .on_press(EditorMessage::TabSelected(i))
                .style(move |_t: &iced::Theme, status| {
                    let bg = if is_active {
                        theme::BG_ELEVATED
                    } else if status == button::Status::Hovered {
                        theme::HOVER
                    } else {
                        theme::BG_SURFACE
                    };
                    button::Style {
                        background: Some(iced::Background::Color(bg)),
                        border: iced::Border {
                            radius: 0.0.into(),
                            width: 0.0,
                            color: iced::Color::TRANSPARENT,
                        },
                        ..Default::default()
                    }
                })
                .padding(0);

            let tab_abs_path = tab.path.clone();
            let tab_rel_path = self
                .selected_workspace_path
                .as_ref()
                .and_then(|ws| {
                    Path::new(&tab_abs_path)
                        .strip_prefix(ws)
                        .ok()
                        .map(|p| p.to_string_lossy().to_string())
                })
                .unwrap_or_else(|| tab_abs_path.clone());

            let ctx_menu = ContextMenu::new(
                tab_btn,
                vec![
                    ("Close".into(), EditorMessage::TabClosed(i)),
                    ("Close Others".into(), EditorMessage::CloseOtherTabs(i)),
                    (
                        "Copy Relative Path".into(),
                        EditorMessage::CopyRelativePath(tab_rel_path),
                    ),
                    (
                        "Copy Absolute Path".into(),
                        EditorMessage::CopyAbsolutePath(tab_abs_path),
                    ),
                ],
            );

            tab_buttons.push(ctx_menu.into());
        }

        let scrollable_content = row(tab_buttons).spacing(0).width(Length::Fill);

        container(
            scrollable(scrollable_content)
                .id(self.tab_scroll_id.clone())
                .direction(theme::horizontal_scrollbar())
                .style(theme::scrollbar_style)
                .width(Length::Fill)
                .height(Length::Shrink),
        )
        .style(|_t: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(theme::BG_SURFACE)),
            border: iced::Border {
                radius: 0.0.into(),
                width: 0.0,
                color: iced::Color::TRANSPARENT,
            },
            ..Default::default()
        })
        .width(Length::Fill)
        .into()
    }

    fn build_find_replace_bar(&self) -> Option<Element<'_, EditorMessage>> {
        let idx = self.active_tab_idx()?;
        let path = &self.tabs[idx].path;
        let state = self.tab_contents.get(path)?.find_replace_state.as_ref()?;

        let search_input = text_input("Find…", &state.query)
            .on_input(EditorMessage::FindQueryInput)
            .on_submit(EditorMessage::FindNext)
            .id(Id::new(FIND_SEARCH_ID))
            .style(widgets::text_input_style)
            .width(Length::Fixed(200.0))
            .size(13);

        let replace_input = text_input("Replace…", &state.replace)
            .on_input(EditorMessage::FindReplaceInput)
            .on_submit(EditorMessage::FindNext)
            .id(Id::new(FIND_REPLACE_ID))
            .style(widgets::text_input_style)
            .width(Length::Fixed(160.0))
            .size(13);

        let total = state.matches.len();
        let match_label = if !state.query.is_empty() && total > 0 {
            format!("{}/{}", state.current_match_idx.saturating_add(1), total)
        } else if !state.query.is_empty() {
            "0/0".to_string()
        } else {
            String::new()
        };

        let prev_btn = button(text("‹").size(14).color(theme::TEXT_SECONDARY))
            .on_press(EditorMessage::FindPrev)
            .style(theme::button_transparent)
            .padding([2, 8]);

        let next_btn = button(text("›").size(14).color(theme::TEXT_SECONDARY))
            .on_press(EditorMessage::FindNext)
            .style(theme::button_transparent)
            .padding([2, 8]);

        let replace_btn = button(text("Replace").size(11).color(theme::TEXT_SECONDARY))
            .on_press(EditorMessage::FindReplace)
            .style(theme::button_transparent)
            .padding([2, 6]);

        let replace_all_btn = button(text("All").size(11).color(theme::TEXT_SECONDARY))
            .on_press(EditorMessage::FindReplaceAll)
            .style(theme::button_transparent)
            .padding([2, 6]);

        // Case sensitivity toggle: "Aa" label, highlighted when active.
        let case_label_color = if state.case_sensitive {
            theme::ACCENT_LIGHT
        } else {
            theme::TEXT_SECONDARY
        };
        let case_btn = button(text("Aa").size(11).color(case_label_color))
            .on_press(EditorMessage::FindToggleCaseSensitivity)
            .style(theme::button_transparent)
            .padding([2, 6]);

        let bar = row![
            search_input,
            replace_input,
            prev_btn,
            text(match_label).size(12).color(theme::TEXT_MUTED),
            next_btn,
            Space::new().width(Length::Fixed(4.0)),
            case_btn,
            replace_btn,
            replace_all_btn,
        ]
        .spacing(4)
        .align_y(Alignment::Center)
        .padding([4, 8]);

        Some(
            container(bar)
                .style(theme::container_bar)
                .width(Length::Fill)
                .into(),
        )
    }

    /// Build the go-to-line input bar. Appears in the same slot as the find
    /// bar (below the tab bar) and is mutually exclusive with it.
    fn build_go_to_line_bar(&self) -> Option<Element<'_, EditorMessage>> {
        let ModalKind::GotoLine(input_text) = self.active_modal.as_ref()? else {
            return None;
        };

        let line_input = text_input("Line #", input_text)
            .on_input(EditorMessage::GoToLineInput)
            .on_submit(EditorMessage::GoToLineGo)
            .id(Id::new(GOTO_LINE_INPUT_ID))
            .style(widgets::text_input_style)
            .width(Length::Fixed(120.0))
            .size(13);

        let go_btn = button(text("Go").size(12).color(theme::TEXT_SECONDARY))
            .on_press(EditorMessage::GoToLineGo)
            .style(theme::button_transparent)
            .padding([2, 8]);

        let bar = row![
            text("Go to line:").size(12).color(theme::TEXT_MUTED),
            Space::new().width(4),
            line_input,
            go_btn,
        ]
        .spacing(4)
        .align_y(Alignment::Center)
        .padding([4, 8]);

        Some(
            container(bar)
                .style(theme::container_bar)
                .width(Length::Fill)
                .into(),
        )
    }

    #[allow(clippy::too_many_lines)]
    fn build_global_search_overlay(gs: &GlobalSearchState) -> Element<'static, EditorMessage> {
        let search_input: iced::widget::TextInput<'_, EditorMessage> =
            text_input("Search across workspace…", &gs.query)
                .on_input(EditorMessage::GlobalSearchInput)
                .on_submit(if gs.results.is_empty() {
                    EditorMessage::GlobalSearchClose
                } else {
                    EditorMessage::GlobalSearchSelect(gs.selected_index)
                })
                .id(Id::new(GLOBAL_SEARCH_INPUT_ID))
                .style(widgets::text_input_style)
                .size(14)
                .width(Length::Fill)
                .padding([8, 12]);

        let search_elem: Element<'static, EditorMessage> = search_input.into();

        // Status/hint line below the input.
        let status_elem: Element<'static, EditorMessage> = match &gs.status {
            GlobalSearchStatus::Idle => text("Type to search across workspace files")
                .size(12)
                .color(theme::TEXT_FAINT)
                .into(),
            GlobalSearchStatus::Searching => row![
                text("Searching…").size(12).color(theme::TEXT_MUTED),
                Space::new().width(Length::Fill),
            ]
            .into(),
            GlobalSearchStatus::NoResults => text("No matches found")
                .size(12)
                .color(theme::TEXT_MUTED)
                .into(),
            GlobalSearchStatus::Error(e) => text(format!("Search error: {e}"))
                .size(12)
                .color(theme::STATUS_ERROR)
                .into(),
            GlobalSearchStatus::Done => {
                let count = gs.results.len();
                text(format!(
                    "{count} result{}",
                    if count == 1 { "" } else { "s" }
                ))
                .size(12)
                .color(theme::TEXT_FAINT)
                .into()
            }
        };

        // Build results list with owned data to satisfy 'static lifetime.
        let results_owned: Vec<OwnedGrepMatch> = gs.results.clone();
        let selected_index = gs.selected_index;

        let results: Vec<Element<'static, EditorMessage>> = results_owned
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let is_selected = i == selected_index;
                let bg = if is_selected {
                    theme::HOVER_STRONG
                } else {
                    iced::Color::TRANSPARENT
                };

                // Clone owned data into the closure to satisfy 'static.
                let path_text = format!("{}:{}", m.rel_path, m.line_number);
                let line_content = m.line_content.clone();
                let offsets = m.match_byte_offsets.clone();
                let accent = if is_selected {
                    theme::ACCENT
                } else {
                    theme::TEXT_SECONDARY
                };

                let path_label = text(path_text.clone())
                    .size(12)
                    .color(accent)
                    .font(iced::Font {
                        weight: iced::font::Weight::Bold,
                        ..theme::FONT_REGULAR
                    });

                // Build the snippet with match portion highlighted in bold.
                let trimmed = line_content.trim().to_string();
                let snippet_label: Element<'static, EditorMessage> =
                    if let Some(&(start, end)) = offsets.first() {
                        let start_us = start as usize;
                        let end_us = end as usize;
                        let content = &line_content;
                        let pre = if start_us <= content.len() {
                            content[..start_us.min(content.len())]
                                .trim_start()
                                .to_string()
                        } else {
                            String::new()
                        };
                        let matched = if start_us < content.len() && end_us <= content.len() {
                            content[start_us..end_us].to_string()
                        } else {
                            String::new()
                        };
                        let post = if end_us < content.len() {
                            content[end_us..].trim_end().to_string()
                        } else {
                            String::new()
                        };

                        let text_color = if is_selected {
                            theme::TEXT_PRIMARY
                        } else {
                            theme::TEXT_MUTED
                        };

                        row![
                            text(pre).size(12).color(text_color),
                            text(matched)
                                .size(12)
                                .color(theme::ACCENT_LIGHT)
                                .font(iced::Font {
                                    weight: iced::font::Weight::Bold,
                                    ..theme::FONT_REGULAR
                                }),
                            text(post).size(12).color(text_color),
                        ]
                        .spacing(0)
                        .into()
                    } else {
                        text(trimmed)
                            .size(12)
                            .color(if is_selected {
                                theme::TEXT_PRIMARY
                            } else {
                                theme::TEXT_MUTED
                            })
                            .into()
                    };

                let entry_content = column![path_label, snippet_label].spacing(1);

                let entry = container(entry_content)
                    .padding([4, 12])
                    .width(Length::Fill)
                    .style(move |_t: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(bg)),
                        ..Default::default()
                    });

                button(entry)
                    .on_press(EditorMessage::GlobalSearchSelect(i))
                    .style(theme::button_transparent)
                    .width(Length::Fill)
                    .padding(0)
                    .into()
            })
            .collect();

        let results_column = column(results).spacing(0).width(Length::Fill);

        let has_results = !gs.results.is_empty();

        let content: Element<'static, EditorMessage> = if !has_results {
            column![search_elem, status_elem].spacing(4).into()
        } else {
            column![
                search_elem,
                status_elem,
                scrollable(results_column)
                    .height(Length::Fixed(400.0))
                    .style(theme::scrollbar_style),
            ]
            .spacing(4)
            .into()
        };

        let dialog = container(content)
            .width(Length::Fixed(600.0))
            .padding(12)
            .style(theme::dialog_container_style);

        editor_dialog::overlay_dialog(dialog, EditorMessage::GlobalSearchClose, 0.4)
    }

    fn build_editor_widget(&self) -> Element<'_, EditorMessage> {
        let Some(idx) = self.active_tab_idx() else {
            return empty_placeholder(
                text("No file selected")
                    .size(EDITOR_FONT_SIZE)
                    .color(theme::TEXT_MUTED),
            );
        };

        let path = &self.tabs[idx].path;
        let Some(tab_data) = self.tab_contents.get(path) else {
            return empty_placeholder(
                text("Error: tab content missing")
                    .size(EDITOR_FONT_SIZE)
                    .color(theme::STATUS_ERROR),
            );
        };

        // ── Build editor widget ────────────────────────────────────────
        let content = &tab_data.content;
        let tree_focused = self.file_tree.tree_focused;
        let find_bar_open = tab_data.find_replace_state.is_some();
        // Modal overlays own keyboard input entirely — block all editor keys.
        let modal_overlay_open = self.active_modal().is_some();
        // Find/replace allows cursor navigation while its text inputs are focused.
        let ignore_keyboard = tree_focused || modal_overlay_open || find_bar_open;

        // Compute match highlight tuples from find/replace state.
        // Each tuple is (line, byte_col_start, byte_col_end) for
        // cosmic_text::Cursor-based highlight rendering.
        let (match_tuples, match_current_idx) = tab_data
            .find_replace_state
            .as_ref()
            .map(|state| {
                let tuples: Vec<(usize, usize, usize)> = state
                    .matches
                    .iter()
                    .filter_map(|range| {
                        let text = tab_data.content.text();
                        let (line, byte_col_start, line_start) =
                            byte_offset_to_line_byte_col(&text, range.start)?;
                        let byte_col_end = range.end.saturating_sub(line_start);
                        Some((line, byte_col_start, byte_col_end))
                    })
                    .collect();
                (tuples, state.current_match_idx)
            })
            .unwrap_or_default();

        // ── Bracket matching ───────────────────────────────────────────
        // Compute matching bracket pair from cursor position (if any).
        let cursor = content.cursor();
        let bracket_pair = if !ignore_keyboard && cursor.selection.is_none() {
            let text = content.text();
            super::editor_widget::find_matching_bracket(&text, cursor.line, cursor.column)
        } else {
            None
        };

        Self::build_highlighted_editor(
            content,
            Some(path.as_str()),
            ignore_keyboard,
            match_tuples,
            match_current_idx,
            self.blink_gen,
            bracket_pair,
        )
    }

    /// Build an [`Element`] from an editor content reference.
    #[allow(clippy::too_many_arguments)]
    fn build_highlighted_editor<'a>(
        content: &'a super::editor_widget::EditorBuffer,
        buffer_key: Option<&'a str>,
        ignore_keyboard: bool,
        matches: Vec<(usize, usize, usize)>,
        match_current_idx: usize,
        blink_gen: u64,
        bracket_pair: Option<super::editor_widget::BracketPair>,
    ) -> Element<'a, EditorMessage> {
        let editor = super::editor_widget::EditorWidget::new(content)
            .font_size(EDITOR_FONT_SIZE)
            .padding(8.0)
            .ignore_keyboard(ignore_keyboard)
            .matches(matches, match_current_idx)
            .blink_gen(blink_gen)
            .bracket_pair(bracket_pair)
            .buffer_key(buffer_key);
        let element = iced::Element::new(editor);
        let mapped = element.map(EditorMessage::EditorAction);

        container(mapped)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_t: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..Default::default()
            })
            .into()
    }

    // ── Context menu action handlers ───────────────────────────────

    /// Perform file deletion: close tab, delete file, re-read parent directory.
    fn perform_file_delete(&mut self, target: &DeleteConfirmTarget) -> Task<EditorMessage> {
        // Close tab if open.
        if let Some(tab_idx) = self.tabs.iter().position(|t| t.path == target.abs_path) {
            self.remove_tab_at(tab_idx);
        }
        // Clear selection if it matches the deleted file.
        // selected_file stores relative paths (set by SelectFile handler).
        if self.selected_file.as_deref() == Some(&target.path) {
            self.selected_file = None;
        }
        // Clean up mtime and toast guard.
        self.file_mtimes.remove(&target.abs_path);
        self.deleted_file_toasted.remove(&target.abs_path);

        self.perform_delete_with_refresh(
            target.abs_path.clone(),
            &target.path,
            "file",
            |abs_path| async move {
                tokio::fs::remove_file(&abs_path)
                    .await
                    .map_err(|e| e.to_string())
            },
        )
    }

    /// Perform directory deletion: remove dir, close affected tabs, re-read parent.
    fn perform_dir_delete(&mut self, target: &DeleteConfirmTarget) -> Task<EditorMessage> {
        let abs_prefix = format!("{}/", target.abs_path);
        let rel_prefix = format!("{}/", target.path);

        // Collect open tabs inside this directory (close in reverse order).
        let mut affected_indices: Vec<usize> = self
            .tabs
            .iter()
            .enumerate()
            .filter(|(_, t)| t.path.starts_with(&abs_prefix))
            .map(|(i, _)| i)
            .collect();
        affected_indices.sort_unstable_by(|a, b| b.cmp(a));

        for &idx in &affected_indices {
            self.remove_tab_at(idx);
        }

        // Clear selection if it was inside the deleted directory.
        // selected_file stores relative paths.
        if let Some(ref sel) = self.selected_file {
            if sel == &target.path || sel.starts_with(&rel_prefix) {
                self.selected_file = None;
            }
        }

        // Clean up mtimes and toast guards for affected paths (absolute).
        self.file_mtimes
            .retain(|path, _| path != &target.abs_path && !path.starts_with(&abs_prefix));
        self.deleted_file_toasted
            .retain(|path| path != &target.abs_path && !path.starts_with(&abs_prefix));

        self.perform_delete_with_refresh(
            target.abs_path.clone(),
            &target.path,
            "directory",
            |abs_path| async move {
                tokio::fs::remove_dir_all(&abs_path)
                    .await
                    .map_err(|e| e.to_string())
            },
        )
    }

    /// Shared preamble for deleting a file or directory: compute parent directory,
    /// bump generation, then run the async delete operation, re-read the parent
    /// directory, and emit a [`DirExpanded`] message.
    ///
    /// `delete_op` receives the absolute path and returns `Result<(), String>`.
    /// `error_label` is used in the toast message on failure (e.g. "file" or
    /// "directory").
    fn perform_delete_with_refresh<D, F>(
        &mut self,
        abs_path: String,
        rel_path: &str,
        error_label: &'static str,
        delete_op: D,
    ) -> Task<EditorMessage>
    where
        D: FnOnce(String) -> F + 'static + Send,
        F: Future<Output = Result<(), String>> + 'static + Send,
    {
        let parent_dir = {
            let path = Path::new(rel_path);
            path.parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default()
        };
        let ws_path = self.selected_workspace_path.clone().unwrap_or_default();
        let r#gen = self.generation.wrapping_add(1);
        self.generation = r#gen;
        // Register the generation so DirExpanded handler accepts the result.
        self.dir_generations.insert(parent_dir.clone(), r#gen);

        Task::perform(
            async move {
                if let Err(e) = delete_op(abs_path).await {
                    return EditorMessage::Toast(super::ToastMessage::Error(format!(
                        "Failed to delete {error_label}: {e}"
                    )));
                }
                // Re-read parent directory.
                let entries = read_directory_entries(&ws_path, &parent_dir).await;
                EditorMessage::DirExpanded {
                    dir_path: parent_dir,
                    r#gen,
                    entries,
                    quiet: false,
                }
            },
            |msg| msg,
        )
    }

    /// Perform new file/directory creation, then re-read parent directory.
    fn perform_create_item(&mut self, target: &NewItemTarget, name: &str) -> Task<EditorMessage> {
        let abs_parent = target.abs_parent.clone();
        let parent_dir = target.parent_dir.clone();
        let is_dir = target.is_dir;
        let ws_root = target.ws_root.clone();

        let abs_new_path_str = Path::new(&abs_parent)
            .join(name)
            .to_string_lossy()
            .to_string();
        let r#gen = self.generation.wrapping_add(1);
        self.generation = r#gen;
        // Register the generation so DirExpanded handler accepts the result.
        self.dir_generations.insert(parent_dir.clone(), r#gen);

        Task::perform(
            async move {
                if is_dir {
                    if let Err(e) = tokio::fs::create_dir(&abs_new_path_str).await {
                        return EditorMessage::Toast(super::ToastMessage::Error(format!(
                            "Failed to create directory: {e}"
                        )));
                    }
                } else if let Err(e) = tokio::fs::write(&abs_new_path_str, "").await {
                    return EditorMessage::Toast(super::ToastMessage::Error(format!(
                        "Failed to create file: {e}"
                    )));
                }
                let entries = read_directory_entries(&ws_root, &parent_dir).await;
                EditorMessage::DirExpanded {
                    dir_path: parent_dir,
                    r#gen,
                    entries,
                    quiet: false,
                }
            },
            |msg| msg,
        )
    }

    /// Fire-and-forget reveal in system file manager.
    fn perform_reveal_in_finder(path: String) -> Task<EditorMessage> {
        Task::perform(
            async move {
                #[cfg(target_os = "macos")]
                {
                    if let Err(e) = std::process::Command::new("open")
                        .arg("-R")
                        .arg(&path)
                        .spawn()
                    {
                        tracing::warn!("Failed to open Finder for {path}: {e}");
                    }
                }
                #[cfg(target_os = "windows")]
                {
                    if let Err(e) = std::process::Command::new("explorer")
                        .arg("/select,")
                        .arg(&path)
                        .spawn()
                    {
                        tracing::warn!("Failed to open Explorer for {path}: {e}");
                    }
                }
                #[cfg(not(any(target_os = "macos", target_os = "windows")))]
                {
                    if let Some(parent) = std::path::Path::new(&path).parent() {
                        if let Err(e) = std::process::Command::new("xdg-open").arg(parent).spawn() {
                            tracing::warn!("Failed to open file manager for {path}: {e}");
                        }
                    }
                }
            },
            |()| EditorMessage::RevealDone,
        )
    }
}

// ── Find/Replace helpers ───────────────────────────────────────────

/// Convert a byte offset in the editor content to a (line, character column) pair.
/// Returns `None` if the offset is out of range.
#[must_use]
fn byte_offset_to_cursor_pos(
    content: &super::editor_widget::EditorBuffer,
    offset: usize,
) -> Option<(usize, usize)> {
    let text = content.text();
    if offset > text.len() {
        return None;
    }
    Some(super::editor_widget::byte_offset_to_line_col(&text, offset))
}

/// Convert a byte offset to (line, byte column within line, line byte start).
#[must_use]
fn byte_offset_to_line_byte_col(text: &str, offset: usize) -> Option<(usize, usize, usize)> {
    if offset > text.len() {
        return None;
    }
    let prefix = &text[..offset];
    let line = prefix.bytes().filter(|&b| b == b'\n').count();
    let line_start = prefix.rfind('\n').map_or(0, |p| p + 1);
    let byte_col = offset - line_start;
    Some((line, byte_col, line_start))
}

/// Auto-jump the cursor to the first find match and reset the match index to 0.
fn auto_jump_to_first_match(
    content: &super::editor_widget::EditorBuffer,
    state: &mut FindReplaceState,
) {
    state.current_match_idx = 0;
    if let Some(range) = state.matches.first() {
        if let Some((line, col)) = byte_offset_to_cursor_pos(content, range.start) {
            content.move_to(line, col);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::editor_widget::{EditorAction, EditorBuffer};

    // ── compute_text_matches ────────────────────────────────────

    #[test]
    fn test_compute_text_matches() {
        struct Case {
            text: &'static str,
            query: &'static str,
            sensitive: bool,
            expected: &'static [(usize, usize)],
        }
        let cases: &[Case] = &[
            // Empty query
            Case {
                text: "hello",
                query: "",
                sensitive: true,
                expected: &[],
            },
            // Basic match
            Case {
                text: "hello world hello",
                query: "hello",
                sensitive: true,
                expected: &[(0, 5), (12, 17)],
            },
            // No match
            Case {
                text: "hello world",
                query: "xyz",
                sensitive: true,
                expected: &[],
            },
            // Non-overlapping
            Case {
                text: "aaaaa",
                query: "aa",
                sensitive: true,
                expected: &[(0, 2), (2, 4)],
            },
            // Case-insensitive
            Case {
                text: "Hello World hello",
                query: "hello",
                sensitive: false,
                expected: &[(0, 5), (12, 17)],
            },
            // Case-insensitive no match
            Case {
                text: "Hello World",
                query: "xyz",
                sensitive: false,
                expected: &[],
            },
            // Single-char queries return empty (2-char min enforcement)
            Case {
                text: "hello",
                query: "h",
                sensitive: true,
                expected: &[],
            },
            // Boundary: shortest possible match
            Case {
                text: "ab",
                query: "ab",
                sensitive: true,
                expected: &[(0, 2)],
            },
            // Boundary: consecutive matches
            Case {
                text: "abab",
                query: "ab",
                sensitive: true,
                expected: &[(0, 2), (2, 4)],
            },
        ];
        for case in cases {
            let result = compute_text_matches(case.text, case.query, case.sensitive);
            assert_eq!(
                result.len(),
                case.expected.len(),
                "text={:?} query={:?} sensitive={}",
                case.text,
                case.query,
                case.sensitive
            );
            for (i, &(start, end)) in case.expected.iter().enumerate() {
                assert_eq!(
                    result[i],
                    start..end,
                    "match[{i}] text={:?} query={:?} sensitive={}",
                    case.text,
                    case.query,
                    case.sensitive
                );
            }
        }
    }

    // ── validate_file_content ─────────────────────────────────────

    #[test]
    fn test_validate_file_content_accepts_valid_input() {
        assert!(validate_file_content(b"").is_ok());
        assert!(validate_file_content(b"hello world").is_ok());
        assert!(validate_file_content("Привет мир 👋".as_bytes()).is_ok());
    }

    #[test]
    fn test_validate_file_content_rejects_invalid_input() {
        let big = vec![b'a'; usize::try_from(MAX_FILE_SIZE).unwrap() + 1];
        let err = validate_file_content(&big).unwrap_err();
        assert!(err.starts_with("File too large"), "unexpected error: {err}");

        let bytes = b"hello\0world";
        let err = validate_file_content(bytes).unwrap_err();
        assert!(
            err.starts_with("Binary file detected"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_validate_file_content_both_conditions_reports_size_first() {
        let mut big_with_null = vec![b'a'; usize::try_from(MAX_FILE_SIZE).unwrap() + 1];
        big_with_null.push(0);
        let err = validate_file_content(&big_with_null).unwrap_err();
        assert!(
            err.starts_with("File too large"),
            "size check should be reported before null-byte check: {err}"
        );
    }

    #[test]
    fn test_byte_offset_to_line_byte_col_unicode() {
        let text = "Привет **мир**";
        let (line, byte_col, line_start) = byte_offset_to_line_byte_col(text, 13).unwrap();
        assert_eq!(line, 0);
        assert_eq!(byte_col, 13);
        assert_eq!(line_start, 0);
        // End of match on "мир" — byte offset 21.
        let (_, byte_end_col, line_start) = byte_offset_to_line_byte_col(text, 21).unwrap();
        assert_eq!(byte_end_col, 21 - line_start);
    }

    #[test]
    fn test_build_tab_records_persists_dirty_content() {
        let tabs = vec![Tab {
            path: "/tmp/foo.md".to_string(),
            file_name: "foo.md".to_string(),
            is_dirty: true,
            has_trailing_newline: true,
            line_ending: LineEnding::Lf,
        }];
        let mut tab_contents = HashMap::new();
        let buffer = EditorBuffer::with_text("unsaved edits", None);
        tab_contents.insert(
            "/tmp/foo.md".to_string(),
            TabData {
                content: buffer,
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: None,
                saved_text_hash: 0,
            },
        );
        let records = build_tab_records(&tabs, 0, &tab_contents);
        assert_eq!(records.len(), 1);
        assert!(records[0].is_dirty);
        assert_eq!(records[0].dirty_content.as_deref(), Some("unsaved edits"));
    }

    #[test]
    fn test_build_tab_records_clears_dirty_content_when_clean() {
        let tabs = vec![Tab {
            path: "/tmp/foo.md".to_string(),
            file_name: "foo.md".to_string(),
            is_dirty: false,
            has_trailing_newline: false,
            line_ending: LineEnding::Lf,
        }];
        let records = build_tab_records(&tabs, 0, &HashMap::new());
        assert!(records[0].dirty_content.is_none());
    }

    #[test]
    fn test_save_result_ignores_stale_save() {
        let mut state = EditorState::new();
        let path = "/tmp/stale.md".to_string();
        state.tabs.push(Tab {
            path: path.clone(),
            file_name: "stale.md".to_string(),
            is_dirty: true,
            has_trailing_newline: false,
            line_ending: LineEnding::Lf,
        });
        state.tab_contents.insert(
            path.clone(),
            TabData {
                content: EditorBuffer::with_text("edited after save started", None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: None,
                saved_text_hash: hash_text("on disk"),
            },
        );
        let saved_hash = hash_text("saved snapshot");
        let _ = state.save_result(&path, Ok(()), saved_hash);
        assert!(
            state.tabs[0].is_dirty,
            "stale save must not clear dirty flag"
        );
    }

    // ── byte_offset_to_cursor_pos ───────────────────────────────

    #[test]
    fn test_byte_offset_to_cursor_pos() {
        struct Case {
            text: &'static str,
            byte_offset: usize,
            expected: Option<(usize, usize)>,
        }
        let cases: &[Case] = &[
            // Unicode multi-byte chars
            Case {
                text: "Привет мир",
                byte_offset: 13, // start of "м"
                expected: Some((0, 7)),
            },
            // Start of content
            Case {
                text: "hello\nworld",
                byte_offset: 0,
                expected: Some((0, 0)),
            },
            // Second line
            Case {
                text: "hello\nworld",
                byte_offset: 6, // after "hello\n"
                expected: Some((1, 0)),
            },
            // Middle of a line
            Case {
                text: "hello\nworld",
                byte_offset: 8, // "wo"
                expected: Some((1, 2)),
            },
            // Beyond text length
            Case {
                text: "hello",
                byte_offset: 100,
                expected: None,
            },
            // Empty content
            Case {
                text: "",
                byte_offset: 0,
                expected: Some((0, 0)),
            },
        ];
        for case in cases {
            let content = EditorBuffer::with_text(case.text, None);
            let pos = byte_offset_to_cursor_pos(&content, case.byte_offset);
            assert_eq!(
                pos, case.expected,
                "text={:?} offset={}",
                case.text, case.byte_offset
            );
        }
    }

    // ── UndoStack ───────────────────────────────────────────────

    fn setup_undo_stack(text: &str) -> (EditorBuffer, UndoStack) {
        (EditorBuffer::with_text(text, None), UndoStack::new())
    }

    #[test]
    fn test_undo_stack_snap_and_undo() {
        let (content, mut stack) = setup_undo_stack("original");
        stack.snap_before_edit(&content);

        // Simulate edit
        let modified = EditorBuffer::with_text("modified", None);
        let snapshot = stack.undo(&modified).unwrap();
        assert_eq!(snapshot.text, "original");
    }

    #[test]
    fn test_undo_stack_redo() {
        let (content, mut stack) = setup_undo_stack("original");
        stack.snap_before_edit(&content);

        let modified = EditorBuffer::with_text("modified", None);
        let _ = stack.undo(&modified);

        let snapshot = stack.redo(&modified).unwrap();
        assert_eq!(snapshot.text, "modified");
    }

    #[test]
    fn test_undo_stack_new_edit_clears_redo() {
        let (content, mut stack) = setup_undo_stack("v1");
        stack.snap_before_edit(&content);

        let v2 = EditorBuffer::with_text("v2", None);
        let _ = stack.undo(&v2);

        // New edit after undo should clear redo.
        let v3 = EditorBuffer::with_text("v3", None);
        stack.snap_before_edit(&v3);

        assert!(stack.redo(&v3).is_none());
    }

    #[test]
    fn test_undo_stack_cursor_restoration() {
        let (content, mut stack) = setup_undo_stack("line1\nline2\nline3");
        // Move cursor to (1, 2) — line 1, column 2 ("ne2")
        content.move_to(1, 2);
        stack.snap_before_edit(&content);

        let modified = EditorBuffer::with_text("changed", None);
        let snapshot = stack.undo(&modified).unwrap();
        assert_eq!(snapshot.cursor_line, 1);
        assert_eq!(snapshot.cursor_col, 2);
    }

    // ── Tree keyboard navigation focus state tests ──────────────────

    /// Helper to create a minimal EditorState with a simple tree.
    fn make_editor_with_tree() -> EditorState {
        let mut state = EditorState::new();
        state.selected_workspace_path = Some("/tmp".to_string());
        // Populate root dir_entries so build_hierarchical_tree works.
        state.dir_entries.insert(
            String::new(),
            vec![
                FsEntry {
                    name: "src".to_string(),
                    full_path: "src".to_string(),
                    is_dir: true,
                    error: None,
                },
                FsEntry {
                    name: "Cargo.toml".to_string(),
                    full_path: "Cargo.toml".to_string(),
                    is_dir: false,
                    error: None,
                },
            ],
        );
        // Populate "src" dir_entries so children show when expanded.
        state.dir_entries.insert(
            "src".to_string(),
            vec![FsEntry {
                name: "main.rs".to_string(),
                full_path: "src/main.rs".to_string(),
                is_dir: false,
                error: None,
            }],
        );
        // Build the tree from dir_entries (consistent with real behavior).
        state.rebuild_tree();
        state
    }

    #[test]
    fn test_rebuild_visible_tree_flattens_nodes() {
        let state = make_editor_with_tree();
        assert_eq!(state.file_tree.visible_tree_nodes.len(), 2);
        assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
        assert!(state.file_tree.visible_tree_nodes[0].1); // is_dir
        assert_eq!(state.file_tree.visible_tree_nodes[1].0, "Cargo.toml");
        assert!(!state.file_tree.visible_tree_nodes[1].1); // not is_dir
    }

    #[test]
    fn test_rebuild_visible_tree_with_expanded_dir() {
        let mut state = make_editor_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        // Rebuild tree from dir_entries with expanded state, then flatten.
        state.file_tree.nodes =
            build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
        state.file_tree.rebuild_visible();
        assert_eq!(state.file_tree.visible_tree_nodes.len(), 3);
        assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
        assert_eq!(state.file_tree.visible_tree_nodes[1].0, "src/main.rs");
        assert_eq!(state.file_tree.visible_tree_nodes[2].0, "Cargo.toml");
    }

    #[test]
    fn test_tree_focus_toggled_sets_focus() {
        let mut state = make_editor_with_tree();
        assert!(!state.file_tree.tree_focused);

        // Toggle on
        let _ = state.update(EditorMessage::TreeFocusToggled);
        assert!(state.file_tree.tree_focused);

        // Toggle off
        let _ = state.update(EditorMessage::TreeFocusToggled);
        assert!(!state.file_tree.tree_focused);
    }

    #[test]
    fn test_tree_focus_toggled_empty_tree_stays_off() {
        let mut state = EditorState::new();
        assert!(!state.file_tree.tree_focused);

        let _ = state.update(EditorMessage::TreeFocusToggled);
        assert!(!state.file_tree.tree_focused); // No visible nodes, focus rejected
    }

    #[test]
    fn test_tree_nav_up_down() {
        struct Case {
            name: &'static str,
            focused: bool,
            start_idx: usize,
            msg: EditorMessage,
            expected_idx: usize,
        }
        let last_idx = make_editor_with_tree().file_tree.visible_tree_nodes.len() - 1;
        let cases: &[Case] = &[
            Case {
                name: "up_at_top_clamped",
                focused: true,
                start_idx: 0,
                msg: EditorMessage::TreeNavUp,
                expected_idx: 0,
            },
            Case {
                name: "down_at_bottom_clamped",
                focused: true,
                start_idx: last_idx,
                msg: EditorMessage::TreeNavDown,
                expected_idx: last_idx,
            },
            Case {
                name: "up_moves_focus",
                focused: true,
                start_idx: 1,
                msg: EditorMessage::TreeNavUp,
                expected_idx: 0,
            },
            Case {
                name: "down_moves_focus",
                focused: true,
                start_idx: 0,
                msg: EditorMessage::TreeNavDown,
                expected_idx: 1,
            },
            Case {
                name: "ignored_when_not_focused",
                focused: false,
                start_idx: 0,
                msg: EditorMessage::TreeNavDown,
                expected_idx: 0,
            },
        ];
        for case in cases {
            let mut state = make_editor_with_tree();
            state.file_tree.tree_focused = case.focused;
            state.file_tree.tree_focus_index = case.start_idx;
            let _ = state.update(case.msg.clone());
            assert_eq!(
                state.file_tree.tree_focus_index, case.expected_idx,
                "case: {}",
                case.name
            );
        }
    }

    #[test]
    fn test_misc_focus_actions() {
        struct Case {
            name: &'static str,
            msg: EditorMessage,
            setup: fn(&mut EditorState),
            check: fn(&EditorState, name: &str),
        }
        let cases: &[Case] = &[
            Case {
                name: "escape_clears_tree_focus",
                msg: EditorMessage::Escape,
                setup: |s| s.file_tree.tree_focused = true,
                check: |s, name| assert!(!s.file_tree.tree_focused, "case: {name}"),
            },
            Case {
                name: "toggle_dir_sets_tree_focus",
                msg: EditorMessage::ToggleDir("src".to_string()),
                setup: |s| {
                    s.selected_file = Some("Cargo.toml".to_string());
                },
                check: |s, name| {
                    assert!(s.file_tree.tree_focused, "case: {name}");
                    assert!(s.selected_file.is_none(), "case: {name}");
                },
            },
            Case {
                name: "select_file_keeps_tree_focus",
                msg: EditorMessage::SelectFile("src/main.rs".to_string()),
                setup: |s| s.file_tree.tree_focused = true,
                check: |s, name| assert!(s.file_tree.tree_focused, "case: {name}"),
            },
            // A mouse-originated EditorAction (like MoveTo from a click)
            // should transfer focus from the file tree to the editor.
            Case {
                name: "editor_action_clears_tree_focus",
                msg: EditorMessage::EditorAction(EditorAction::MoveTo { line: 0, col: 0 }),
                setup: |s| {
                    s.file_tree.tree_focused = true;
                    s.pending_enter_dir = Some("src".to_string());
                    s.active_modal = Some(ModalKind::Rename(RenameTarget {
                        path: "src/main.rs".to_string(),
                        abs_path: String::new(),
                        is_dir: false,
                        ws_root: String::new(),
                        input_text: "main.rs".to_string(),
                        error: None,
                    }));
                },
                check: |s, name| {
                    assert!(!s.file_tree.tree_focused, "case: {name}");
                    assert_eq!(s.pending_enter_dir, None, "case: {name}");
                    assert!(s.active_modal.is_none(), "case: {name}");
                },
            },
        ];
        for case in cases {
            let mut state = make_editor_with_tree();
            (case.setup)(&mut state);
            let _ = state.update(case.msg.clone());
            (case.check)(&state, case.name);
        }
    }

    #[test]
    fn test_tree_nav_enter() {
        struct Case {
            name: &'static str,
            focused: bool,
            start_idx: usize,
            /// Set selected_file before the message
            pre_select_file: bool,
            /// Expected tree_focused after
            expect_focused: bool,
            /// Expected focus index after (None = skip check)
            expected_idx: Option<usize>,
            /// Additional per-case assertions
            check: Option<fn(&EditorState, name: &str)>,
        }
        let cases: &[Case] = &[
            // TreeNavEnter on a file dispatches an async load task, but
            // tree_focused stays true in the same-turn state update.
            Case {
                name: "on_file_dispatches_task",
                focused: true,
                start_idx: 1,
                pre_select_file: false,
                expect_focused: true,
                expected_idx: None,
                check: None,
            },
            Case {
                name: "not_focused_ignored",
                focused: false,
                start_idx: 1,
                pre_select_file: false,
                expect_focused: false,
                expected_idx: Some(1),
                check: None,
            },
            Case {
                name: "on_dir_expands_and_advances",
                focused: true,
                start_idx: 0,
                pre_select_file: true,
                expect_focused: true,
                expected_idx: Some(1),
                check: Some(|s, name| {
                    assert!(s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                    assert!(s.selected_file.is_none(), "case: {name}");
                    assert_eq!(s.file_tree.visible_tree_nodes[1].0, "src/main.rs");
                }),
            },
        ];
        for case in cases {
            let mut state = make_editor_with_tree();
            state.file_tree.tree_focused = case.focused;
            state.file_tree.tree_focus_index = case.start_idx;
            if case.pre_select_file {
                state.selected_file = Some("Cargo.toml".to_string());
            }
            let _ = state.update(EditorMessage::TreeNavEnter);
            assert_eq!(
                state.file_tree.tree_focused, case.expect_focused,
                "case: {}",
                case.name
            );
            if let Some(idx) = case.expected_idx {
                assert_eq!(state.file_tree.tree_focus_index, idx, "case: {}", case.name);
            }
            if let Some(check) = case.check {
                check(&state, case.name);
            }
        }
    }

    #[test]
    fn test_visible_tree_clamps_focus_on_rebuild() {
        let mut state = make_editor_with_tree();
        state.file_tree.tree_focus_index = 999; // Way out of range
        state.file_tree.rebuild_visible();
        assert_eq!(
            state.file_tree.tree_focus_index,
            state.file_tree.visible_tree_nodes.len() - 1
        );
    }

    #[test]
    fn test_async_enter_dir_sets_pending_then_advances() {
        let mut state = EditorState::new();
        state.selected_workspace_path = Some("/tmp".to_string());
        // Set up a tree where "src" dir_entries are empty (needs async load).
        state.dir_entries.insert(
            String::new(),
            vec![FsEntry {
                name: "src".to_string(),
                full_path: "src".to_string(),
                is_dir: true,
                error: None,
            }],
        );
        state.rebuild_tree();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 0; // "src"

        let _task = state.update(EditorMessage::TreeNavEnter);
        // "src" needs async loading — pending_enter_dir is set.
        assert_eq!(state.pending_enter_dir.as_deref(), Some("src"));
        assert!(state.file_tree.expanded_dirs.contains("src"));
        // Focus stays on "src" until children load.
        assert_eq!(state.file_tree.tree_focus_index, 0);

        // Simulate DirExpanded completing with children.
        let entries = vec![FsEntry {
            name: "main.rs".to_string(),
            full_path: "src/main.rs".to_string(),
            is_dir: false,
            error: None,
        }];
        let dir_gen = state.generation;
        let _task = state.update(EditorMessage::DirExpanded {
            dir_path: "src".to_string(),
            r#gen: dir_gen,
            entries: Ok(entries),
            quiet: false,
        });
        // Focus should have advanced to the first child.
        assert_eq!(state.pending_enter_dir, None);
        assert_eq!(state.file_tree.tree_focus_index, 1); // "src/main.rs"
    }

    #[test]
    fn test_toggle_dir_async_load_and_complete() {
        let mut state = EditorState::new();
        state.selected_workspace_path = Some("/tmp".to_string());
        // "src" dir has no cached entries → needs async load.
        state.dir_entries.insert(
            String::new(),
            vec![FsEntry {
                name: "src".to_string(),
                full_path: "src".to_string(),
                is_dir: true,
                error: None,
            }],
        );
        state.rebuild_tree();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 0; // "src"

        let _task = state.update(EditorMessage::ToggleDir("src".to_string()));
        // ToggleDir sets loading_dirs and dir_generations.
        assert!(state.loading_dirs.contains("src"));
        assert!(state.dir_generations.contains_key("src"));
        // ToggleDir does NOT set pending_enter_dir.
        assert_eq!(state.pending_enter_dir, None);
        // Focus is on "src".
        assert!(state.file_tree.tree_focused);
        assert_eq!(state.file_tree.tree_focus_index, 0);

        // Simulate DirExpanded completing with children.
        let dir_gen = *state.dir_generations.get("src").unwrap();
        let entries = vec![FsEntry {
            name: "main.rs".to_string(),
            full_path: "src/main.rs".to_string(),
            is_dir: false,
            error: None,
        }];
        let _task = state.update(EditorMessage::DirExpanded {
            dir_path: "src".to_string(),
            r#gen: dir_gen,
            entries: Ok(entries),
            quiet: false,
        });
        // Entries are now cached.
        assert!(state.dir_entries.contains_key("src"));
        assert_eq!(state.dir_entries["src"].len(), 1);
        // loading_dirs is cleared.
        assert!(!state.loading_dirs.contains("src"));
        // pending_enter_dir was never set.
        assert_eq!(state.pending_enter_dir, None);
        // visible_tree_nodes is correctly rebuilt (rebuild_tree was called).
        assert!(state.file_tree.visible_tree_nodes.len() >= 2);
        assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
        assert_eq!(state.file_tree.visible_tree_nodes[1].0, "src/main.rs");
    }

    #[test]
    fn test_toggle_dir_no_workspace_returns_none() {
        let mut state = EditorState::new();
        // Precondition: "src" is not yet in expanded_dirs before the call.
        assert!(!state.file_tree.expanded_dirs.contains("src"));
        // No workspace set — async load should return None (early return).
        let _task = state.update(EditorMessage::ToggleDir("src".to_string()));
        // expanded_dirs is modified (insert happens before the workspace guard),
        // but no async load was spawned since there's no workspace path.
        assert!(state.file_tree.expanded_dirs.contains("src"));
        assert_eq!(state.generation, 0);
        assert!(state.loading_dirs.is_empty());
        assert!(state.dir_generations.is_empty());
    }

    // ── Git status porcelain parsing tests ─────────────────────────

    #[allow(clippy::too_many_lines)]
    #[test]
    fn test_parse_git_status_porcelain() {
        struct Case {
            /// Short label for failure messages.
            name: &'static str,
            /// Raw git status --porcelain output.
            input: &'static str,
            /// Expected entries: (path, Some(status)) asserts the file has that
            /// status; (path, None) asserts the file is absent from the map.
            /// An empty slice asserts the entire map is empty.
            expected: &'static [(&'static str, Option<GitFileStatus>)],
        }
        let cases: &[Case] = &[
            Case {
                name: "unstaged modified file",
                input: " M src/main.rs\n",
                expected: &[("src/main.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "staged added file",
                input: "A  new_file.rs\n",
                expected: &[("new_file.rs", Some(GitFileStatus::Added))],
            },
            Case {
                name: "untracked file",
                input: "?? new_file.rs\n",
                expected: &[("new_file.rs", Some(GitFileStatus::Added))],
            },
            Case {
                name: "staged and unstaged modified (MM)",
                input: "MM both.rs\n",
                expected: &[("both.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "staged added + unstaged modified (AM)",
                input: "AM partial.rs\n",
                expected: &[("partial.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "rename (old -> new)",
                input: "R  old.rs -> new.rs\n",
                expected: &[("new.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "rename with arrow in old path",
                input: "R  \"old -> name.rs\" -> \"new -> name.rs\"\n",
                expected: &[("new -> name.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "untracked directory (trailing slash stripped)",
                input: "?? new_dir/\n",
                expected: &[("new_dir", Some(GitFileStatus::Added))],
            },
            Case {
                name: "quoted path with spaces",
                input: " M \"path with spaces.rs\"\n",
                expected: &[("path with spaces.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "deleted file (unstaged) skipped",
                input: " D gone.rs\n",
                expected: &[],
            },
            Case {
                name: "deleted file (staged) skipped",
                input: "D  gone.rs\n",
                expected: &[],
            },
            Case {
                name: "clean (unrecognized status) not present",
                input: "   clean.rs\n",
                expected: &[("clean.rs", None)],
            },
            Case {
                name: "multiple entries same file — modified wins over added",
                input: "A  dup.rs\n M dup.rs\n",
                expected: &[("dup.rs", Some(GitFileStatus::Modified))],
            },
            Case {
                name: "multiple entries same file — added sticks",
                input: "?? dup.rs\nA  dup.rs\n",
                expected: &[("dup.rs", Some(GitFileStatus::Added))],
            },
            Case {
                name: "empty output",
                input: "",
                expected: &[],
            },
            Case {
                name: "mixed statuses",
                input: concat!(
                    " M src/main.rs\n",
                    "?? new_file.rs\n",
                    "A  staged.rs\n",
                    " D deleted.rs\n",
                ),
                expected: &[
                    ("src/main.rs", Some(GitFileStatus::Modified)),
                    ("new_file.rs", Some(GitFileStatus::Added)),
                    ("staged.rs", Some(GitFileStatus::Added)),
                    ("deleted.rs", None),
                ],
            },
        ];

        for case in cases {
            let map = parse_git_status_porcelain(case.input);

            if case.expected.is_empty() {
                assert!(
                    map.is_empty(),
                    "case '{}' (input={:?}): expected empty map, got {:#?}",
                    case.name,
                    case.input,
                    map
                );
            } else {
                let expected_count = case.expected.iter().filter(|(_, s)| s.is_some()).count();
                assert_eq!(
                    map.len(),
                    expected_count,
                    "case '{}' (input={:?}): map has unexpected entries",
                    case.name,
                    case.input,
                );
                for &(path, expected_status) in case.expected {
                    match expected_status {
                        Some(status) => {
                            assert_eq!(
                                map.get(path),
                                Some(&status),
                                "case '{}' (input={:?}): path={:?}",
                                case.name,
                                case.input,
                                path
                            );
                        }
                        None => {
                            assert!(
                                !map.contains_key(path),
                                "case '{}' (input={:?}): path={:?} should be absent, got {:?}",
                                case.name,
                                case.input,
                                path,
                                map.get(path)
                            );
                        }
                    }
                }
            }
        }
    }

    // ── Find/Replace tests ───────────────────────────────────────────

    #[test]
    fn test_is_find_bar_open_true_when_active() {
        let mut state = EditorState::new();
        // Must have an active tab with find_replace_state.
        state.tabs.push(Tab {
            path: "/test.rs".to_string(),
            file_name: "test.rs".to_string(),
            is_dirty: false,
            has_trailing_newline: true,
            line_ending: LineEnding::Lf,
        });
        state.active_tab_index = 0;
        state.tab_contents.insert(
            "/test.rs".to_string(),
            TabData {
                content: EditorBuffer::with_text("fn hello() {}", None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: Some(FindReplaceState {
                    query: "hello".to_string(),
                    replace: String::new(),
                    matches: std::iter::once(4..9).collect(),
                    current_match_idx: 0,
                    case_sensitive: false,
                }),
                saved_text_hash: 0,
            },
        );
        assert!(state.is_find_bar_open());
    }

    #[test]
    fn test_is_find_bar_open_false_when_closed() {
        let mut state = EditorState::new();
        state.tabs.push(Tab {
            path: "/test.rs".to_string(),
            file_name: "test.rs".to_string(),
            is_dirty: false,
            has_trailing_newline: true,
            line_ending: LineEnding::Lf,
        });
        state.active_tab_index = 0;
        state.tab_contents.insert(
            "/test.rs".to_string(),
            TabData {
                content: EditorBuffer::with_text("fn hello() {}", None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: None,
                saved_text_hash: 0,
            },
        );
        assert!(!state.is_find_bar_open());
    }

    #[test]
    fn test_is_find_bar_open_no_tabs() {
        let state = EditorState::new();
        assert!(!state.is_find_bar_open());
    }

    #[test]
    fn test_find_replace_auto_advance_same_length() {
        // Replace "ab" with "xy" in "ab cd ab" → "xy cd ab".
        // After replacement, one "ab" remains at byte 7.
        // Auto-advance should skip past the replacement (replace_end = 0 + 2 = 2)
        // and find the remaining match at position 7.
        let mut tab_data = TabData {
            content: EditorBuffer::with_text("ab cd ab", None),
            undo_stack: RefCell::new(UndoStack::new()),
            find_replace_state: Some(FindReplaceState {
                query: "ab".to_string(),
                replace: "xy".to_string(),
                matches: vec![0..2, 6..8],
                current_match_idx: 0,
                case_sensitive: true,
            }),
            saved_text_hash: 0,
        };

        // Simulate FindReplace on match 0.
        let range = 0..2;
        let replace_text = "xy".to_string();
        let replace_end = range.start + replace_text.len(); // = 2
        let text = tab_data.content.text();
        let new_text = format!(
            "{}{}{}",
            &text[..range.start],
            replace_text,
            &text[range.end..]
        );
        assert_eq!(new_text, "xy cd ab");
        tab_data.content = EditorBuffer::with_text(&new_text, None);
        if let Some(ref mut state) = tab_data.find_replace_state {
            state.matches = compute_text_matches(&new_text, &state.query, state.case_sensitive);
            assert_eq!(state.matches.len(), 1, "one remaining match");
            let next_idx = state
                .matches
                .iter()
                .position(|m| m.start >= replace_end)
                .unwrap_or(0)
                .min(state.matches.len() - 1);
            assert_eq!(next_idx, 0, "should advance to the remaining match");
            if let Some(r) = state.matches.get(next_idx) {
                assert_eq!(r.start, 6, "remaining match should start at byte 6");
                assert_eq!(r.end, 8, "remaining match should end at byte 8");
            }
        }
    }

    #[test]
    fn test_find_replace_auto_advance_shorter_replacement() {
        // Replace "aaa" with "a" in "aaa bbb aaa" → "a bbb aaa".
        // After replacing first match (0..3) → "a" (at pos 0), the new text
        // is "a bbb aaa". Position after replacement = 0 + 1 = 1.
        // The remaining "aaa" starts at byte 6 in the new text.
        // replace_end = 1 correctly finds it; old_end = 3 would also work here
        // because 6 >= 3, but the real bug manifests with adjacent matches.
        let mut tab_data = TabData {
            content: EditorBuffer::with_text("aaa bbb aaa", None),
            undo_stack: RefCell::new(UndoStack::new()),
            find_replace_state: Some(FindReplaceState {
                query: "aaa".to_string(),
                replace: "a".to_string(),
                matches: vec![0..3, 8..11],
                current_match_idx: 0,
                case_sensitive: true,
            }),
            saved_text_hash: 0,
        };

        let range = 0..3;
        let replace_text = "a".to_string();
        // CORRECT: position after replacement in NEW text
        let replace_end = range.start + replace_text.len(); // = 1
        let text = tab_data.content.text();
        let new_text = format!(
            "{}{}{}",
            &text[..range.start],
            replace_text,
            &text[range.end..]
        );
        assert_eq!(new_text, "a bbb aaa");
        tab_data.content = EditorBuffer::with_text(&new_text, None);
        if let Some(ref mut state) = tab_data.find_replace_state {
            state.matches = compute_text_matches(&new_text, &state.query, state.case_sensitive);
            // Only the second "aaa" (now at 6..9) remains
            assert_eq!(state.matches.len(), 1, "one remaining match");
            assert_eq!(state.matches[0].start, 6);
            assert_eq!(state.matches[0].end, 9);

            // Using replace_end (= 1): finds next match at position 6
            let correct_idx = state
                .matches
                .iter()
                .position(|m| m.start >= replace_end)
                .unwrap_or(0);
            assert_eq!(
                correct_idx, 0,
                "replace_end should find the remaining match"
            );
        }
    }

    #[test]
    fn test_find_replace_auto_advance_adjacent_matches() {
        // Replace "aa" with "x" in "aaaa" → "xx".
        // Original matches: 0..2, 2..4 (adjacent overlapping prohibited).
        // After replacing match 0 (0..2) with "x": new text = "xaa".
        // Position after replacement = 0 + 1 = 1.
        // The old_end bug: old_end = 2. In new text, the remaining "aa" is at
        // byte 1..3. Using old_end (= 2) would skip it because 2 >= 2 is true!
        // But replace_end (= 1) correctly finds it because 1 >= 1.
        let mut tab_data = TabData {
            content: EditorBuffer::with_text("aaaa", None),
            undo_stack: RefCell::new(UndoStack::new()),
            find_replace_state: Some(FindReplaceState {
                query: "aa".to_string(),
                replace: "x".to_string(),
                matches: vec![0..2, 2..4],
                current_match_idx: 0,
                case_sensitive: true,
            }),
            saved_text_hash: 0,
        };

        let range = 0..2;
        let replace_text = "x".to_string();
        let replace_end = range.start + replace_text.len(); // = 1
        let old_end = range.end; // = 2 (the bug value)
        let text = tab_data.content.text();
        let new_text = format!(
            "{}{}{}",
            &text[..range.start],
            replace_text,
            &text[range.end..]
        );
        assert_eq!(new_text, "xaa", "after replacing first aa with x");
        tab_data.content = EditorBuffer::with_text(&new_text, None);
        if let Some(ref mut state) = tab_data.find_replace_state {
            state.matches = compute_text_matches(&new_text, &state.query, state.case_sensitive);
            assert_eq!(state.matches.len(), 1, "one remaining match in xaa");

            // replace_end (= 1): m.start >= 1 → finds match at 1..3
            let correct_idx = state
                .matches
                .iter()
                .position(|m| m.start >= replace_end)
                .unwrap_or(0);
            assert_eq!(
                correct_idx, 0,
                "replace_end should find the remaining match"
            );

            // old_end (= 2): m.start >= 2 → finds match at... 1..3 has start=1,
            // so 1 >= 2 is false, and 0 is returned (no match found).
            // This means the match would be SKIPPED!
            let bug_idx = state.matches.iter().position(|m| m.start >= old_end);
            assert_eq!(bug_idx, None, "old_end skips the remaining match!");
        }
    }

    #[test]
    fn test_find_replace_auto_advance_longer_replacement() {
        // Replace "ab" with "abc" in "ab" → "abc".
        // After replacement, the new match for "ab" is at 0..2.
        // replace_end = 0 + 3 = 3. The match at 0..2 has start=0 < 3,
        // so position() returns None → unwrap_or(0) wraps to index 0.
        let mut tab_data = TabData {
            content: EditorBuffer::with_text("ab", None),
            undo_stack: RefCell::new(UndoStack::new()),
            find_replace_state: Some(FindReplaceState {
                query: "ab".to_string(),
                replace: "abc".to_string(),
                matches: std::iter::once(0..2).collect(),
                current_match_idx: 0,
                case_sensitive: true,
            }),
            saved_text_hash: 0,
        };

        let range = 0..2;
        let replace_text = "abc".to_string();
        let replace_end = range.start + replace_text.len(); // = 3
        let text = tab_data.content.text();
        let new_text = format!(
            "{}{}{}",
            &text[..range.start],
            replace_text,
            &text[range.end..]
        );
        assert_eq!(new_text, "abc");
        tab_data.content = EditorBuffer::with_text(&new_text, None);
        if let Some(ref mut state) = tab_data.find_replace_state {
            state.matches = compute_text_matches(&new_text, &state.query, state.case_sensitive);
            // "abc" still contains "ab" at 0..2
            assert_eq!(state.matches.len(), 1, "one match in abc");

            // replace_end = 3: m.start >= 3 → match 0..2 has start=0, 0 >= 3 is false,
            // so position() returns None, unwrap_or(0) gives 0.
            let next_idx = state
                .matches
                .iter()
                .position(|m| m.start >= replace_end)
                .unwrap_or(0)
                .min(state.matches.len() - 1);
            assert_eq!(next_idx, 0, "wraps back to the same match");
        }
    }

    #[test]
    fn test_find_replace_auto_advance_no_more_matches() {
        // Replace "ab" with "xy" in "ab" → "xy". No more matches.
        let mut tab_data = TabData {
            content: EditorBuffer::with_text("ab", None),
            undo_stack: RefCell::new(UndoStack::new()),
            find_replace_state: Some(FindReplaceState {
                query: "ab".to_string(),
                replace: "xy".to_string(),
                matches: std::iter::once(0..2).collect(),
                current_match_idx: 0,
                case_sensitive: true,
            }),
            saved_text_hash: 0,
        };

        let range = 0..2;
        let replace_text = "xy".to_string();
        let text = tab_data.content.text();
        let new_text = format!(
            "{}{}{}",
            &text[..range.start],
            replace_text,
            &text[range.end..]
        );
        tab_data.content = EditorBuffer::with_text(&new_text, None);
        if let Some(ref mut state) = tab_data.find_replace_state {
            state.matches = compute_text_matches(&new_text, &state.query, state.case_sensitive);
            assert!(state.matches.is_empty(), "no matches in xy");
            if state.matches.is_empty() {
                state.current_match_idx = 0;
            }
            assert_eq!(state.current_match_idx, 0, "reset to 0");
        }
    }

    #[test]
    fn test_navigate_find_match_wraps_next() {
        let mut state = EditorState::new();
        state.tabs.push(Tab {
            path: "/test.rs".to_string(),
            file_name: "test.rs".to_string(),
            is_dirty: false,
            has_trailing_newline: true,
            line_ending: LineEnding::Lf,
        });
        state.active_tab_index = 0;
        state.tab_contents.insert(
            "/test.rs".to_string(),
            TabData {
                content: EditorBuffer::with_text("a b c", None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: Some(FindReplaceState {
                    query: " ".to_string(),
                    replace: String::new(),
                    matches: vec![1..2, 3..4],
                    current_match_idx: 0,
                    case_sensitive: true,
                }),
                saved_text_hash: 0,
            },
        );

        // Navigate next from index 0 → 1.
        let _ = state.navigate_find_match(&FindDirection::Next);
        let frs = state.tab_contents.get("/test.rs").unwrap();
        let s = frs.find_replace_state.as_ref().unwrap();
        assert_eq!(s.current_match_idx, 1);

        // Navigate next from index 1 → wraps to 0.
        let _ = state.navigate_find_match(&FindDirection::Next);
        let s = state.tab_contents.get("/test.rs").unwrap();
        let s = s.find_replace_state.as_ref().unwrap();
        assert_eq!(s.current_match_idx, 0);
    }

    #[test]
    fn test_navigate_find_match_wraps_prev() {
        let mut state = EditorState::new();
        state.tabs.push(Tab {
            path: "/test.rs".to_string(),
            file_name: "test.rs".to_string(),
            is_dirty: false,
            has_trailing_newline: true,
            line_ending: LineEnding::Lf,
        });
        state.active_tab_index = 0;
        state.tab_contents.insert(
            "/test.rs".to_string(),
            TabData {
                content: EditorBuffer::with_text("a b c", None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: Some(FindReplaceState {
                    query: " ".to_string(),
                    replace: String::new(),
                    matches: vec![1..2, 3..4],
                    current_match_idx: 0,
                    case_sensitive: true,
                }),
                saved_text_hash: 0,
            },
        );

        // Navigate prev from index 0 → wraps to 1 (last).
        let _ = state.navigate_find_match(&FindDirection::Prev);
        let s = state.tab_contents.get("/test.rs").unwrap();
        let s = s.find_replace_state.as_ref().unwrap();
        assert_eq!(s.current_match_idx, 1);

        // Navigate prev from index 1 → 0.
        let _ = state.navigate_find_match(&FindDirection::Prev);
        let s = state.tab_contents.get("/test.rs").unwrap();
        let s = s.find_replace_state.as_ref().unwrap();
        assert_eq!(s.current_match_idx, 0);
    }

    #[test]
    fn test_navigate_find_match_no_matches() {
        let mut state = EditorState::new();
        state.tabs.push(Tab {
            path: "/test.rs".to_string(),
            file_name: "test.rs".to_string(),
            is_dirty: false,
            has_trailing_newline: true,
            line_ending: LineEnding::Lf,
        });
        state.active_tab_index = 0;
        state.tab_contents.insert(
            "/test.rs".to_string(),
            TabData {
                content: EditorBuffer::with_text("no matches", None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: Some(FindReplaceState {
                    query: "zzz".to_string(),
                    replace: String::new(),
                    matches: vec![],
                    current_match_idx: 0,
                    case_sensitive: true,
                }),
                saved_text_hash: 0,
            },
        );

        // Navigating with no matches should not crash.
        let _ = state.navigate_find_match(&FindDirection::Next);
        let _ = state.navigate_find_match(&FindDirection::Prev);
        let s = state.tab_contents.get("/test.rs").unwrap();
        let s = s.find_replace_state.as_ref().unwrap();
        assert_eq!(s.current_match_idx, 0);
    }

    #[test]
    fn test_navigate_find_match_only_affects_find_tab() {
        // Tab without find state should not be affected.
        let mut state = EditorState::new();
        state.tabs.push(Tab {
            path: "/a.rs".to_string(),
            file_name: "a.rs".to_string(),
            is_dirty: false,
            has_trailing_newline: true,
            line_ending: LineEnding::Lf,
        });
        state.active_tab_index = 0;
        state.tab_contents.insert(
            "/a.rs".to_string(),
            TabData {
                content: EditorBuffer::with_text("hello", None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: None,
                saved_text_hash: 0,
            },
        );

        // Should not panic.
        let _ = state.navigate_find_match(&FindDirection::Next);
        let _ = state.navigate_find_match(&FindDirection::Prev);
    }

    // ── Tree arrow-key navigation tests ─────────────────────────────

    #[allow(clippy::too_many_lines)]
    #[test]
    fn test_tree_nav_left_right() {
        struct Case {
            name: &'static str,
            msg: EditorMessage,
            start_idx: usize,
            /// Pre-expand "src" before sending the message
            pre_expand_src: bool,
            /// Set selected_file to Some("Cargo.toml") before sending the message
            pre_select_file: bool,
            /// Expected focus index after the message
            expected_idx: usize,
            /// Additional per-case assertions beyond focus index
            check: Option<fn(&EditorState, name: &str)>,
        }
        let cases: &[Case] = &[
            Case {
                name: "left_on_expanded_dir_collapses",
                msg: EditorMessage::TreeNavLeft,
                start_idx: 0,
                pre_expand_src: true,
                pre_select_file: false,
                expected_idx: 0,
                check: Some(|s, name| {
                    assert!(!s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                }),
            },
            Case {
                name: "left_on_file_navigates_to_parent",
                msg: EditorMessage::TreeNavLeft,
                start_idx: 1,
                pre_expand_src: true,
                pre_select_file: false,
                expected_idx: 0,
                check: Some(|s, name| {
                    assert_eq!(s.file_tree.visible_tree_nodes[0].0, "src", "case: {name}");
                }),
            },
            Case {
                name: "left_on_root_collapsed_dir_noop",
                msg: EditorMessage::TreeNavLeft,
                start_idx: 0,
                pre_expand_src: false,
                pre_select_file: false,
                expected_idx: 0,
                check: None,
            },
            Case {
                name: "left_on_root_file_noop",
                msg: EditorMessage::TreeNavLeft,
                start_idx: 1,
                pre_expand_src: false,
                pre_select_file: false,
                expected_idx: 1,
                check: None,
            },
            Case {
                name: "right_on_collapsed_dir_expands_and_advances",
                msg: EditorMessage::TreeNavRight,
                start_idx: 0,
                pre_expand_src: false,
                pre_select_file: true,
                expected_idx: 1,
                check: Some(|s, name| {
                    assert!(s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                    assert!(s.selected_file.is_none(), "case: {name}");
                    assert_eq!(s.file_tree.visible_tree_nodes[1].0, "src/main.rs");
                }),
            },
            Case {
                name: "right_on_expanded_dir_moves_to_first_child",
                msg: EditorMessage::TreeNavRight,
                start_idx: 0,
                pre_expand_src: true,
                pre_select_file: false,
                expected_idx: 1,
                check: Some(|s, name| {
                    assert_eq!(
                        s.file_tree.visible_tree_nodes[1].0, "src/main.rs",
                        "case: {name}"
                    );
                }),
            },
            Case {
                name: "right_on_file_noop",
                msg: EditorMessage::TreeNavRight,
                start_idx: 1,
                pre_expand_src: false,
                pre_select_file: false,
                expected_idx: 1,
                check: None,
            },
        ];
        for case in cases {
            let mut state = make_editor_with_tree();
            if case.pre_expand_src {
                state.file_tree.expanded_dirs.insert("src".to_string());
                state.file_tree.nodes =
                    build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
                state.file_tree.rebuild_visible();
            }
            state.file_tree.tree_focused = true;
            state.file_tree.tree_focus_index = case.start_idx;
            if case.pre_select_file {
                state.selected_file = Some("Cargo.toml".to_string());
            }
            let _ = state.update(case.msg.clone());
            assert_eq!(
                state.file_tree.tree_focus_index, case.expected_idx,
                "case: {}",
                case.name
            );
            if let Some(check) = case.check {
                check(&state, case.name);
            }
        }
    }

    // ── Click-to-select focus index tests ────────────────────────────

    #[test]
    fn test_toggle_dir_sets_tree_focus_index() {
        let mut state = make_editor_with_tree();
        // Select a file first so we can verify it gets cleared.
        state.selected_file = Some("Cargo.toml".to_string());
        let _ = state.update(EditorMessage::ToggleDir("src".to_string()));
        // ToggleDir should set tree_focus_index to "src"'s position
        assert!(state.file_tree.tree_focused);
        assert_eq!(state.file_tree.tree_focus_index, 0);
        assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
        assert!(
            state.selected_file.is_none(),
            "ToggleDir should clear selected_file"
        );
    }

    #[test]
    fn test_select_file_sets_tree_focus_index() {
        let mut state = make_editor_with_tree();
        // Expand "src" so "src/main.rs" is visible in the flat list.
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes =
            build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        let _ = state.update(EditorMessage::SelectFile("src/main.rs".to_string()));
        // SelectFile keeps tree_focused and remembers focus index.
        assert!(state.file_tree.tree_focused);
        // tree_focus_index should point to "src/main.rs" for Ctrl+B re-focus.
        assert_eq!(
            state.file_tree.visible_tree_nodes[state.file_tree.tree_focus_index].0,
            "src/main.rs"
        );
    }

    #[test]
    fn test_select_file_sets_tree_focused_when_not_focused() {
        // When tree_focused starts false, clicking a file should set it true.
        let mut state = make_editor_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes =
            build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = false;
        let _ = state.update(EditorMessage::SelectFile("src/main.rs".to_string()));
        assert!(
            state.file_tree.tree_focused,
            "SelectFile should set tree_focused to true"
        );
    }

    // ── Focus gating and find/replace cursor tests ───────────────────

    fn make_editor_with_single_tab(text: &str) -> EditorState {
        let mut state = EditorState::new();
        state.tabs.push(Tab {
            path: "/test.rs".to_string(),
            file_name: "test.rs".to_string(),
            is_dirty: false,
            has_trailing_newline: true,
            line_ending: LineEnding::Lf,
        });
        state.active_tab_index = 0;
        state.tab_contents.insert(
            "/test.rs".to_string(),
            TabData {
                content: EditorBuffer::with_text(text, None),
                undo_stack: RefCell::new(UndoStack::new()),
                find_replace_state: None,
                saved_text_hash: hash_text(text),
            },
        );
        state
    }

    /// Each [`ModalKind`] variant must be returned by [`EditorState::active_modal()`]
    /// when the corresponding modal is open.  This catches the case where a new
    /// variant is added to the enum but `escape()`'s match is not updated (the
    /// compiler enforces exhaustiveness).
    #[test]
    fn test_active_modal_returns_correct_variant() {
        let mut state = EditorState::new();

        // None when no modal is open.
        assert!(state.active_modal().is_none());

        // Each variant — set it, verify it returns the expected variant,
        // then clear and test the next.
        state.active_modal = Some(ModalKind::GlobalSearch(GlobalSearchState {
            query: String::new(),
            results: Vec::new(),
            selected_index: 0,
            status: GlobalSearchStatus::Idle,
            search_gen: 0,
        }));
        assert!(matches!(
            state.active_modal(),
            Some(ModalKind::GlobalSearch(_))
        ));
        state.active_modal = None;

        state.active_modal = Some(ModalKind::GotoLine(String::new()));
        assert!(matches!(state.active_modal(), Some(ModalKind::GotoLine(_))));
        state.active_modal = None;

        state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
            filter: String::new(),
            selected_index: 0,
            results: Vec::new(),
        }));
        assert!(matches!(
            state.active_modal(),
            Some(ModalKind::QuickOpen(_))
        ));
        state.active_modal = None;

        state.active_modal = Some(ModalKind::Rename(RenameTarget {
            path: "foo".into(),
            abs_path: String::new(),
            is_dir: false,
            ws_root: String::new(),
            input_text: "foo".into(),
            error: None,
        }));
        assert!(matches!(state.active_modal(), Some(ModalKind::Rename(_))));
        state.active_modal = None;

        state.active_modal = Some(ModalKind::NewItem(NewItemTarget {
            parent_dir: String::new(),
            is_dir: false,
            abs_parent: String::new(),
            ws_root: String::new(),
            input_text: String::new(),
        }));
        assert!(matches!(state.active_modal(), Some(ModalKind::NewItem(_))));
        state.active_modal = None;

        state.active_modal = Some(ModalKind::DeleteConfirm(DeleteConfirmTarget {
            path: "foo".into(),
            is_dir: false,
            dirty_tab_count: 0,
            abs_path: String::new(),
        }));
        assert!(matches!(
            state.active_modal(),
            Some(ModalKind::DeleteConfirm(_))
        ));
        state.active_modal = None;

        state.active_modal = Some(ModalKind::CloseDialog(0));
        assert!(matches!(
            state.active_modal(),
            Some(ModalKind::CloseDialog(..))
        ));
        state.active_modal = None;

        state.active_modal = Some(ModalKind::CloseOthers(0));
        assert!(matches!(
            state.active_modal(),
            Some(ModalKind::CloseOthers(_))
        ));
    }

    #[test]
    fn test_undo_noop_when_quick_open_active() {
        let mut state = make_editor_with_single_tab("hello");
        let path = "/test.rs".to_string();
        if let Some(tab_data) = state.tab_contents.get_mut(&path) {
            tab_data
                .undo_stack
                .borrow_mut()
                .snap_before_edit(&tab_data.content);
            tab_data.content.perform_action(EditorAction::Insert('!'));
        }
        state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
            filter: String::new(),
            selected_index: 0,
            results: Vec::new(),
        }));
        let _ = state.update(EditorMessage::Undo);
        assert_eq!(
            state.tab_contents.get(&path).unwrap().content.text(),
            "!hello"
        );
    }

    #[test]
    fn test_refresh_file_tree_noop_when_quick_open_active() {
        let mut state = make_editor_with_tree();
        // Pre-populate dir_generations so we can detect new entries.
        let initial_gen_count = state.dir_generations.len();
        assert!(state.selected_workspace_path.is_some());

        // Activate a modal overlay (QuickOpen).
        state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
            filter: String::new(),
            selected_index: 0,
            results: Vec::new(),
        }));

        // RefreshFileTree should be suppressed — no new dir generations added.
        let _ = state.update(EditorMessage::RefreshFileTree);
        assert_eq!(
            state.dir_generations.len(),
            initial_gen_count,
            "RefreshFileTree must not spawn directory refreshes when a modal overlay is active"
        );
    }

    #[test]
    fn test_tree_focus_toggled_noop_during_modal_overlay() {
        let mut state = make_editor_with_tree();
        // First toggle tree focus ON.
        let _ = state.update(EditorMessage::TreeFocusToggled);
        assert!(state.file_tree.tree_focused);

        // Activate a modal overlay (QuickOpen).
        state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
            filter: String::new(),
            selected_index: 0,
            results: Vec::new(),
        }));

        // TreeFocusToggled should be suppressed — focus stays ON.
        let _ = state.update(EditorMessage::TreeFocusToggled);
        assert!(
            state.file_tree.tree_focused,
            "TreeFocusToggled must not toggle focus when a modal overlay is active"
        );
    }

    #[test]
    fn test_tree_nav_suppressed_during_goto_line_overlay() {
        let mut state = make_editor_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes =
            build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 0; // "src"

        // Activate a non-search modal overlay (GotoLine).
        state.active_modal = Some(ModalKind::GotoLine(String::new()));

        let prev_focus = state.file_tree.tree_focus_index;
        // Up/Down/Enter/Left/Right — assert tree_focus_index unchanged.
        let nav_msgs: &[EditorMessage] = &[
            EditorMessage::TreeNavUp,
            EditorMessage::TreeNavDown,
            EditorMessage::TreeNavEnter,
            EditorMessage::TreeNavLeft,
            EditorMessage::TreeNavRight,
        ];
        for msg in nav_msgs {
            let _ = state.update(msg.clone());
            assert_eq!(
                state.file_tree.tree_focus_index, prev_focus,
                "{msg:?} should be suppressed during GotoLine overlay"
            );
        }

        // TreeFocusToggled is handled separately because it toggles
        // tree_focused, not tree_focus_index.
        let _ = state.update(EditorMessage::TreeFocusToggled);
        assert!(
            state.file_tree.tree_focused,
            "TreeFocusToggled should be suppressed during GotoLine overlay"
        );
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn test_find_replace_cursor_after_last_match_removed() {
        let mut state = make_editor_with_single_tab("ab");
        let path = "/test.rs".to_string();
        if let Some(tab_data) = state.tab_contents.get_mut(&path) {
            tab_data.find_replace_state = Some(FindReplaceState {
                query: "ab".to_string(),
                replace: "xy".to_string(),
                matches: vec![0..2],
                current_match_idx: 0,
                case_sensitive: true,
            });
        }
        let _ = state.update(EditorMessage::FindReplace);
        let cursor = state.tab_contents.get(&path).unwrap().content.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 2);
        assert_eq!(state.tab_contents.get(&path).unwrap().content.text(), "xy");
    }

    #[test]
    fn test_find_replace_all_preserves_cursor() {
        let mut state = make_editor_with_single_tab("ab cd ab");
        let path = "/test.rs".to_string();
        if let Some(tab_data) = state.tab_contents.get_mut(&path) {
            tab_data.content.move_to(0, 5);
            tab_data.find_replace_state = Some(FindReplaceState {
                query: "ab".to_string(),
                replace: "xy".to_string(),
                matches: vec![0..2, 6..8],
                current_match_idx: 0,
                case_sensitive: true,
            });
        }
        let _ = state.update(EditorMessage::FindReplaceAll);
        let cursor = state.tab_contents.get(&path).unwrap().content.cursor();
        assert_eq!(cursor.line, 0);
        assert_eq!(cursor.column, 5);
    }

    #[test]
    fn test_quick_open_toggle_blocked_when_goto_line_open() {
        let mut state = make_editor_with_single_tab("hello");
        state.active_modal = Some(ModalKind::GotoLine(String::new()));
        let _ = state.update(EditorMessage::QuickOpenToggle);
        assert!(!matches!(state.active_modal, Some(ModalKind::QuickOpen(_))));
    }

    #[test]
    fn test_quick_open_toggle_closes_when_already_open() {
        let mut state = make_editor_with_single_tab("hello");
        state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
            filter: "foo".to_string(),
            selected_index: 0,
            results: Vec::new(),
        }));
        let _ = state.update(EditorMessage::QuickOpenToggle);
        assert!(state.active_modal.is_none());
    }

    #[test]
    fn test_global_search_toggle_blocked_when_quick_open_open() {
        let mut state = make_editor_with_single_tab("hello");
        state.selected_workspace_name = Some("ws".to_string());
        state.selected_workspace_path = Some("/tmp/ws".to_string());
        state.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
            filter: String::new(),
            selected_index: 0,
            results: Vec::new(),
        }));
        let _ = state.update(EditorMessage::GlobalSearchToggle);
        assert!(!matches!(
            state.active_modal,
            Some(ModalKind::GlobalSearch(_))
        ));
    }

    // ── Inline rename tests ────────────────────────────────────

    #[test]
    fn test_rename_request_sets_target() {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
        let rt = match state.active_modal {
            Some(ModalKind::Rename(ref rt)) => rt.clone(),
            _ => panic!("expected Rename modal"),
        };
        assert_eq!(rt.path, "Cargo.toml");
        assert_eq!(rt.input_text, "Cargo.toml");
        assert!(!rt.is_dir);
    }

    #[test]
    fn test_rename_request_on_directory_sets_is_dir() {
        // Use a real temp directory so Path::is_dir() returns true.
        let tmp_dir = tempfile::tempdir().unwrap();
        let dir_path = tmp_dir.path().join("src");
        std::fs::create_dir(&dir_path).unwrap();
        let mut state = EditorState::new();
        state.selected_workspace_path = Some(tmp_dir.path().to_string_lossy().to_string());
        state.dir_entries.insert(
            String::new(),
            vec![FsEntry {
                name: "src".to_string(),
                full_path: "src".to_string(),
                is_dir: true,
                error: None,
            }],
        );
        let _ = state.update(EditorMessage::RenameRequested("src".into()));
        assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
        let rt = match state.active_modal {
            Some(ModalKind::Rename(ref rt)) => rt.clone(),
            _ => panic!("expected Rename modal"),
        };
        assert_eq!(rt.path, "src");
        assert_eq!(rt.input_text, "src");
        assert!(rt.is_dir);
    }

    #[test]
    fn test_rename_request_on_root_dir_rejected() {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        let _ = state.update(EditorMessage::RenameRequested(String::new()));
        assert!(
            state.active_modal.is_none()
                || !matches!(state.active_modal, Some(ModalKind::Rename(_)))
        );
    }

    #[test]
    fn test_rename_input_updates_text_and_clears_error() {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        // Simulate a validation error
        if let Some(ModalKind::Rename(ref mut rt)) = state.active_modal {
            rt.error = Some("bad".into());
        }
        // Type new text
        let _ = state.update(EditorMessage::RenameInput("new_name".into()));
        if let Some(ModalKind::Rename(ref rt)) = state.active_modal {
            assert_eq!(rt.input_text, "new_name");
            // Error should be cleared when user types
            assert!(rt.error.is_none());
        } else {
            panic!("expected Rename modal");
        }
    }

    #[test]
    fn test_rename_cancel_clears_target() {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
        let _ = state.update(EditorMessage::RenameCancel);
        assert!(state.active_modal.is_none());
    }

    #[test]
    fn test_escape_cancels_rename() {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
        let _ = state.update(EditorMessage::Escape);
        assert!(state.active_modal.is_none());
    }

    #[test]
    fn test_tree_nav_suppressed_during_rename() {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        // Expand "src" so TreeNavEnter/TreeNavLeft/TreeNavRight have targets.
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes =
            build_hierarchical_tree(&state.dir_entries, &state.file_tree.expanded_dirs, "");
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        // Focus on "src" so TreeNavLeft (collapse) and TreeNavRight (expand)
        // have an effect when not suppressed.
        state.file_tree.tree_focus_index = 0; // "src"

        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        let prev_focus = state.file_tree.tree_focus_index;
        // All 6 tree-navigation messages must be suppressed during rename.
        let nav_msgs: &[EditorMessage] = &[
            EditorMessage::TreeNavUp,
            EditorMessage::TreeNavDown,
            EditorMessage::TreeNavEnter,
            EditorMessage::TreeNavLeft,
            EditorMessage::TreeNavRight,
            EditorMessage::TreeFocusToggled,
        ];
        for msg in nav_msgs {
            let _ = state.update(msg.clone());
            assert_eq!(
                state.file_tree.tree_focus_index, prev_focus,
                "{msg:?} should be suppressed during rename"
            );
        }
        // After the rename is cancelled, navigation should work again.
        let _ = state.update(EditorMessage::RenameCancel);
        let _ = state.update(EditorMessage::TreeNavDown);
        // Focus should have moved now that rename is gone.
        assert_ne!(state.file_tree.tree_focus_index, prev_focus);
    }

    #[test]
    fn test_rename_mutual_exclusion_with_new_item() {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());

        // Start rename, then NewFileRequested should cancel it.
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
        let _ = state.update(EditorMessage::NewFileRequested("src".into()));
        assert!(
            state.active_modal.is_none()
                || !matches!(state.active_modal, Some(ModalKind::Rename(_)))
        );
        assert!(matches!(state.active_modal, Some(ModalKind::NewItem(_))));

        // Start new item again — and confirm rename cancels new_item.
        let _ = state.update(EditorMessage::NewFileRequested(String::new()));
        assert!(matches!(state.active_modal, Some(ModalKind::NewItem(_))));
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        assert!(matches!(state.active_modal, Some(ModalKind::Rename(_))));
    }

    // ── Rename validation tests ────────────────────────────────

    /// Helper: set up state for rename validation tests.
    fn setup_rename_state(state: &mut EditorState, input_text: &str) {
        state.selected_workspace_path = Some("/tmp".to_string());
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        if let Some(ModalKind::Rename(ref mut rt)) = state.active_modal {
            rt.input_text = input_text.to_string();
        }
    }

    /// Helper: set up a rename with `input`, submit it, and assert that
    /// the resulting error equals `expected`.
    fn assert_rename_rejects(input: &str, expected: Option<&'static str>) {
        let mut state = make_editor_with_tree();
        setup_rename_state(&mut state, input);
        let _ = state.update(EditorMessage::RenameSubmit);
        let err = match &state.active_modal {
            Some(ModalKind::Rename(rt)) => rt.error.as_deref(),
            _ => None,
        };
        assert_eq!(err, expected, "rejection of {input:?}");
    }

    #[test]
    fn test_rename_validation() {
        struct Case {
            input: &'static str,
            expected: Option<&'static str>,
        }
        let cases: &[Case] = &[
            // Empty / whitespace-only
            Case {
                input: "   ",
                expected: Some("Name cannot be empty"),
            },
            // Path separators
            Case {
                input: "foo/bar.rs",
                expected: Some("Name cannot contain path separators"),
            },
            Case {
                input: "foo\\bar.rs",
                expected: Some("Name cannot contain path separators"),
            },
            Case {
                input: "foo\0bar.rs",
                expected: Some("Name cannot contain path separators"),
            },
            // Dot / dot-dot
            Case {
                input: ".",
                expected: Some("Invalid name"),
            },
            Case {
                input: "..",
                expected: Some("Invalid name"),
            },
        ];
        for case in cases {
            assert_rename_rejects(case.input, case.expected);
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_rename_validation_os_reserved_names() {
        let reserved = ["con", "NUL", "prn", "AUX", "com1", "lpt3"];
        for name in &reserved {
            assert_rename_rejects(name, Some("Name is reserved by the operating system"));
        }
    }

    #[test]
    fn test_rename_validation_target_already_exists() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let ws = tmp_dir.path().to_string_lossy().to_string();
        // Create a file that would conflict.
        let existing = tmp_dir.path().join("existing.txt");
        std::fs::write(&existing, "").unwrap();

        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some(ws.clone());
        let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
        if let Some(ModalKind::Rename(ref mut rt)) = state.active_modal {
            rt.input_text = "existing.txt".to_string();
        }
        let _ = state.update(EditorMessage::RenameSubmit);
        let err = match &state.active_modal {
            Some(ModalKind::Rename(rt)) => rt.error.as_deref(),
            _ => None,
        };
        assert_eq!(
            err,
            Some("A file or directory with that name already exists")
        );
    }

    #[test]
    fn test_rename_stale_generation_discarded() {
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());
        // Expand src so it's visible for RenameRequested.
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.rebuild_tree();

        // Dispatch a rename for a non-root path (src/main.rs) so that the
        // staleness check in RenameCompleted (which only applies when the
        // parent dir is non-empty) is actually exercised.
        let _ = state.update(EditorMessage::RenameRequested("src/main.rs".into()));
        if let Some(ModalKind::Rename(ref mut rt)) = state.active_modal {
            rt.input_text = "lib.rs".to_string();
        }
        let _ = state.update(EditorMessage::RenameSubmit);

        // Simulate a stale RenameCompleted whose rename_gen does not
        // match the current dir_generations entry for the parent dir ("src").
        // It passes dir_entries: Ok(vec![]) — if the staleness guard fails and
        // this result is applied, it would overwrite dir_entries["src"] with
        // an empty vec, losing the original children.
        let task = state.update(EditorMessage::RenameCompleted {
            old_path: "src/main.rs".into(),
            new_path: "src/lib.rs".into(),
            is_dir: false,
            result: Ok(()),
            dir_entries: Ok(vec![]),
            rename_gen: 0, // stale — doesn't match dir_generations["src"]
        });
        // The stale result should be a no-op (discarded silently).
        let _ = task;
        // dir_entries["src"] must still contain its original entries — if
        // the stale result were applied, the empty vec would have replaced them.
        let src_entries = state.dir_entries.get("src");
        assert!(
            src_entries.is_some(),
            "dir_entries[\"src\"] should still exist"
        );
        if let Some(entries) = src_entries {
            assert_eq!(entries.len(), 1, "should still have one entry");
            assert_eq!(entries[0].name, "main.rs");
            assert_eq!(entries[0].full_path, "src/main.rs");
        }
        // selected_file should not have been updated.
        assert_eq!(state.selected_file, None);
    }

    // ── Click-outside cancel tests (consolidated) ───────────────

    #[test]
    fn test_rename_cancelled_by_tree_click() {
        // Both ToggleDir and SelectFile should cancel a pending rename.
        let triggers: &[EditorMessage] = &[
            EditorMessage::ToggleDir("src".into()),
            EditorMessage::SelectFile("src/main.rs".into()),
        ];
        for trigger in triggers {
            let mut state = make_editor_with_tree();
            state.selected_workspace_path = Some("/tmp".to_string());
            let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
            assert!(
                matches!(state.active_modal, Some(ModalKind::Rename(_))),
                "rename should be active before {trigger:?}"
            );
            let _ = state.update(trigger.clone());
            assert!(
                state.active_modal.is_none()
                    || !matches!(state.active_modal, Some(ModalKind::Rename(_))),
                "rename should be cancelled by {trigger:?}"
            );
        }
    }

    #[test]
    fn test_rename_mutual_exclusion_cancelled_by_other_modals() {
        // Starting a different modal operation should cancel an active rename.
        // Each test case carries a message to dispatch and a check closure
        // that verifies the expected modal state after the message fires.
        struct Case {
            msg: EditorMessage,
            /// Assert the expected modal state after the message fires.
            check: fn(&EditorState),
        }
        let cases: &[Case] = &[
            Case {
                msg: EditorMessage::NewFileRequested("src".into()),
                check: |s| assert!(matches!(s.active_modal, Some(ModalKind::NewItem(_)))),
            },
            Case {
                msg: EditorMessage::NewDirectoryRequested("src".into()),
                check: |s| assert!(matches!(s.active_modal, Some(ModalKind::NewItem(_)))),
            },
            Case {
                msg: EditorMessage::DeleteFileRequested("other.rs".into()),
                check: |s| {
                    assert!(matches!(s.active_modal, Some(ModalKind::DeleteConfirm(_))));
                    if let Some(ModalKind::DeleteConfirm(ref target)) = s.active_modal {
                        assert_eq!(target.path, "other.rs");
                    }
                },
            },
            Case {
                msg: EditorMessage::DeleteDirectoryRequested("src".into()),
                check: |s| {
                    assert!(matches!(s.active_modal, Some(ModalKind::DeleteConfirm(_))));
                    if let Some(ModalKind::DeleteConfirm(ref target)) = s.active_modal {
                        assert_eq!(target.path, "src");
                    }
                },
            },
        ];
        for case in cases {
            let mut state = make_editor_with_tree();
            state.selected_workspace_path = Some("/tmp".to_string());

            // Start rename.
            let _ = state.update(EditorMessage::RenameRequested("Cargo.toml".into()));
            assert!(
                matches!(state.active_modal, Some(ModalKind::Rename(_))),
                "case {:?}",
                case.msg
            );

            // Fire the competing modal message.
            let _ = state.update(case.msg.clone());
            assert!(
                !matches!(state.active_modal, Some(ModalKind::Rename(_))),
                "rename should be cancelled by {:?}",
                case.msg
            );
            (case.check)(&state);
        }
    }

    // ── rekey helpers ──────────────────────────────────────────

    #[test]
    fn test_rekey_keys_empty() {
        let pairs = rekey_keys("old/", "new/", Vec::<String>::new());
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_rekey_keys_no_match() {
        let keys = vec!["a".to_string(), "b".to_string()];
        let pairs = rekey_keys("old/", "new/", keys);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_rekey_keys_some_match() {
        let keys = vec![
            "old/foo".to_string(),
            "other".to_string(),
            "old/bar/baz".to_string(),
        ];
        let mut pairs = rekey_keys("old/", "new", keys);
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(pairs.len(), 2);
        assert_eq!(
            pairs[0],
            ("old/bar/baz".to_string(), "new/bar/baz".to_string())
        );
        assert_eq!(pairs[1], ("old/foo".to_string(), "new/foo".to_string()));
    }

    #[test]
    fn test_rekey_keys_exact_prefix() {
        let keys = vec!["dir".to_string()];
        let pairs = rekey_keys("dir", "newdir", keys);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], ("dir".to_string(), "newdir".to_string()));
    }

    #[test]
    fn test_rekey_map_prefix_no_modify() {
        let mut map = HashMap::from([
            ("dir/file.rs".to_string(), "content_a".to_string()),
            ("dir/sub/file.rs".to_string(), "content_b".to_string()),
            ("other".to_string(), "content_c".to_string()),
        ]);
        rekey_map_prefix(&mut map, "dir/", "newdir", |_| {});
        assert_eq!(map.len(), 3);
        assert_eq!(map.get("newdir/file.rs"), Some(&"content_a".to_string()));
        assert_eq!(
            map.get("newdir/sub/file.rs"),
            Some(&"content_b".to_string())
        );
        assert_eq!(map.get("other"), Some(&"content_c".to_string()));
        assert!(!map.contains_key("dir/file.rs"));
    }

    #[test]
    fn test_rekey_map_prefix_with_modify() {
        let mut map = HashMap::from([
            ("old/key".to_string(), vec![1, 2]),
            ("old/other".to_string(), vec![3]),
            ("keep".to_string(), vec![4]),
        ]);
        rekey_map_prefix(&mut map, "old/", "new", |v: &mut Vec<i32>| v.push(99));
        assert_eq!(map.len(), 3);
        assert_eq!(map.get("new/key"), Some(&vec![1, 2, 99]));
        assert_eq!(map.get("new/other"), Some(&vec![3, 99]));
        assert_eq!(map.get("keep"), Some(&vec![4]));
    }

    #[test]
    fn test_rekey_set_prefix_basic() {
        let mut set = HashSet::from(["a/x".to_string(), "a/y".to_string(), "b/z".to_string()]);
        rekey_set_prefix(&mut set, "a/", "b");
        assert_eq!(set.len(), 3);
        assert!(set.contains("b/x"));
        assert!(set.contains("b/y"));
        assert!(set.contains("b/z"));
    }

    #[test]
    fn test_rekey_set_prefix_exact() {
        let mut set = HashSet::from(["dir".to_string()]);
        rekey_set_prefix(&mut set, "dir", "newdir");
        assert_eq!(set.len(), 1);
        assert!(set.contains("newdir"));
        assert!(!set.contains("dir"));
    }

    #[test]
    fn test_rename_dir_entries_migration_own_entry_and_full_path() {
        // Verify that after a directory rename completes, the renamed
        // directory's own dir_entries key is migrated (old_path -> new_path)
        // and child entries have their full_path fields updated.
        let mut state = make_editor_with_tree();
        state.selected_workspace_path = Some("/tmp".to_string());

        // Set up state as if the user expanded "src" and we have its children.
        state.file_tree.expanded_dirs.insert("src".to_string());
        // Add a subdirectory entry for recursive testing.
        state.dir_entries.insert(
            "src/subdir".to_string(),
            vec![FsEntry {
                name: "helper.rs".to_string(),
                full_path: "src/subdir/helper.rs".to_string(),
                is_dir: false,
                error: None,
            }],
        );

        // Simulate a rename of "src" -> "lib" completing successfully.
        // Pre-populate dir_generations so the staleness guard passes
        // (rename_submit would have registered this generation before
        // firing the async operation).
        state.dir_generations.insert(String::new(), 0);
        let _ = state.update(EditorMessage::RenameCompleted {
            old_path: "src".into(),
            new_path: "lib".into(),
            is_dir: true,
            result: Ok(()),
            dir_entries: Ok(vec![FsEntry {
                name: "lib".to_string(),
                full_path: "lib".to_string(),
                is_dir: true,
                error: None,
            }]),
            rename_gen: 0,
        });

        // The directory's own dir_entries entry should be migrated.
        assert!(
            !state.dir_entries.contains_key("src"),
            "old path key should be removed"
        );
        let own_entries = state.dir_entries.get("lib");
        assert!(
            own_entries.is_some(),
            "new path key should exist for the renamed directory"
        );
        // The own-key entry's children must have their full_path updated.
        if let Some(entries) = own_entries {
            assert_eq!(entries.len(), 1, "src had one child (main.rs)");
            assert_eq!(entries[0].full_path, "lib/main.rs");
        }

        // The child directory entry should be migrated with updated full_path.
        let child_entries = state.dir_entries.get("lib/subdir");
        assert!(
            child_entries.is_some(),
            "child dir_entries key should be migrated"
        );
        if let Some(entries) = child_entries {
            if let Some(entry) = entries.first() {
                assert_eq!(
                    entry.full_path, "lib/subdir/helper.rs",
                    "entry full_path should be updated to new prefix"
                );
            }
        }

        // The expanded_dirs should have been migrated.
        assert!(
            !state.file_tree.expanded_dirs.contains("src"),
            "old expanded_dir should be removed"
        );
        assert!(
            state.file_tree.expanded_dirs.contains("lib"),
            "new expanded_dir should exist"
        );
    }
}
