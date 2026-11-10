//! Code editor dashboard page — tabbed editor with a file tree and workspace-backed tab
//! persistence. Key bindings: Ctrl/Cmd+S save, Tab/Shift+Tab indent, Ctrl+B tree focus;
//! with the tree focused, arrows navigate, Enter opens/expands, Escape leaves.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use iced::widget::{Space, button, column, container, row, scrollable, text};
use iced::{
    Alignment, Color, Element, Length, Subscription, Task,
    keyboard::{self},
    widget::Id,
};

use iced_fonts::lucide;

use fff_search::grep::{GrepMode, GrepSearchOptions};
use fff_search::parse_grep_query;

use super::menus::{ContextMenu, MenuItem};

use crate::git::commands::GitFileStatus;

use super::common::{SingleLineEditorState, UndoSnapshot, UndoStack, focus_navigation_task};
use super::editor_widget::{LineEnding, detect_line_ending};
use crate::tools::MAX_FILE_SIZE_BYTES as MAX_FILE_SIZE;

use super::editor_widget::EnterBehavior;
use super::editor_widget::{EditorBuffer, byte_offset_to_line_col};
use super::theme;
use super::widgets::{self, FileTree, TreeNavDirection};

mod editor_dialog;

/// Advance width of one ASCII glyph in the tab label's 12px JetBrains Mono (0.6em).
const TAB_CHAR_ADVANCE: f32 = 7.2;

/// Non-text chrome per tab: button padding, 2px row spacing, 12px close button.
const TAB_FIXED_CHROME: f32 = 16.0 + 2.0 + 12.0;

/// Extra width for the dirty-indicator dot (8px) plus its 2px row spacing.
const TAB_DIRTY_EXTRA: f32 = 8.0 + 2.0;

/// Estimate a tab's rendered width in px: char count × [`TAB_CHAR_ADVANCE`], plus
/// [`TAB_FIXED_CHROME`] and [`TAB_DIRTY_EXTRA`]. Monospace keeps the per-tab error at a few
/// px, so the delta reveal in [`EditorState::scroll_to_active_tab`] never amplifies it into a
/// right-edge clamp (the old flat 140px constant accumulated that error).
#[expect(clippy::cast_precision_loss)]
fn estimate_tab_width(tab: &Tab) -> f32 {
    let name_w = tab.file_name.chars().count() as f32 * TAB_CHAR_ADVANCE;
    let dirty = if tab.is_dirty { TAB_DIRTY_EXTRA } else { 0.0 };
    name_w + TAB_FIXED_CHROME + dirty
}

/// Estimated content-space left edge of tab `idx` (the widths of all tabs before it).
fn estimated_tab_left(tabs: &[Tab], idx: usize) -> f32 {
    tabs.iter().take(idx).map(estimate_tab_width).sum()
}

fn estimated_content_width(tabs: &[Tab]) -> f32 {
    tabs.iter().map(estimate_tab_width).sum()
}

/// Widget IDs for find/replace text inputs (used for auto-focus).
const FIND_SEARCH_ID: &str = "find_search_input";
const FIND_REPLACE_ID: &str = "find_replace_input";

const GLOBAL_SEARCH_INPUT_ID: &str = "global_search_input";

const GOTO_LINE_INPUT_ID: &str = "goto_line_input";

const QUICK_OPEN_INPUT_ID: &str = "quick_open_input";

const NEW_ITEM_INPUT_ID: &str = "new_item_input";

const MAX_GLOBAL_SEARCH_RESULTS: usize = 200;

const GLOBAL_SEARCH_MATCHES_PER_FILE: usize = 20;

const GLOBAL_SEARCH_DEBOUNCE_MS: u64 = 300;

/// Shown with a failed saved-tabs restore: tab persistence stays blocked until it resolves.
const TABS_SAVE_PAUSED: &str = "tab saving is paused until this is resolved";

#[must_use]
fn is_os_file(name: &str) -> bool {
    name.eq_ignore_ascii_case(".ds_store")
        || name.eq_ignore_ascii_case("thumbs.db")
        || name.eq_ignore_ascii_case("desktop.ini")
}

/// Wrap a bar row in the full-width container so the bar background spans the row.
fn bar_harness<'a>(bar: impl Into<Element<'a, EditorMessage>>) -> Element<'a, EditorMessage> {
    container(bar)
        .style(theme::container_bar)
        .width(Length::Fill)
        .into()
}

#[derive(Debug, Clone)]
struct FsEntry {
    name: String,
    full_path: String,
    is_dir: bool,
    /// Set when the entry could not be inspected (broken symlink, permission denied).
    error: Option<String>,
}

/// A single editor tab (metadata, no content).
#[derive(Debug, Clone)]
struct Tab {
    path: String,
    file_name: String,
    is_dirty: bool,
    line_ending: LineEnding,
}

struct TabData {
    content: super::editor_widget::EditorBuffer,
    undo_stack: RefCell<UndoStack>,
    /// Find/replace state (None when bar is hidden).
    find_replace_state: Option<FindReplaceState>,
    /// Hash of the last saved (or loaded) text; undo/redo compares against it.
    saved_text_hash: u64,
}

/// Fast non-crypto hash of a string for dirty-state comparison.
fn hash_text(text: &str) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(text.as_bytes());
    h.finish()
}

/// Build the `Tab` + `TabData` pair from file text and metadata, plus the file's mtime.
fn make_tab_and_data(
    path: &str,
    text: &str,
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

struct FindReplaceState {
    query: SingleLineEditorState,
    replace: SingleLineEditorState,
    matches: Vec<std::ops::Range<usize>>,
    current_match_idx: usize,
    case_sensitive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GlobalSearchStatus {
    Idle,
    Searching,
    Done,
    NoResults,
    Error(String),
}

/// A grep match extracted from `fff_search::GrepResult` so it can cross async boundaries.
#[derive(Debug, Clone)]
struct OwnedGrepMatch {
    abs_path: String,
    rel_path: String,
    line_number: u64,
    line_content: String,
    /// Byte offsets of the match within `line_content`, for highlighting.
    match_byte_offsets: Vec<(u32, u32)>,
}

/// State for the global search (Cmd+Shift+F) panel.
struct GlobalSearchState {
    query: SingleLineEditorState,
    results: Vec<OwnedGrepMatch>,
    selected_index: usize,
    status: GlobalSearchStatus,
}

use std::ops::Range;

/// Byte-range matches of `query` in `text`; empty when the query is under 2 chars.
/// Case-insensitive matching folds ASCII only, which is length-preserving so the ranges
/// stay valid for `text`; non-ASCII queries match literally.
#[must_use]
fn compute_text_matches(text: &str, query: &str, case_sensitive: bool) -> Vec<Range<usize>> {
    if query.len() < 2 {
        return Vec::new();
    }
    let mut matches = Vec::new();

    if case_sensitive {
        for (start, _) in text.match_indices(query) {
            matches.push(start..start + query.len());
        }
    } else {
        let text_lower = text.to_ascii_lowercase();
        let query_lower = query.to_ascii_lowercase();
        for (start, _) in text_lower.match_indices(&query_lower) {
            matches.push(start..start + query.len());
        }
    }

    matches
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FindDirection {
    Next,
    Prev,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TabDirection {
    Next,
    Prev,
}

#[derive(Debug, Clone)]
struct FileLoadData {
    path: String,
    text: String,
    line_ending: LineEnding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CloseAction {
    Save,
    Discard,
    Cancel,
}

/// Pending close action to run after the next successful save; at most one at a time.
#[derive(Debug, Clone)]
enum PendingCloseAction {
    CloseTab(usize),
    /// Close all tabs except `keep_idx`; these dirty tab indices
    /// still need to be saved first.
    CloseOthers {
        keep_idx: usize,
        remaining_dirty: Vec<usize>,
    },
}

/// Raw data loaded from a saved tab entry (string content, not Content).
#[derive(Debug, Clone)]
struct SavedTabData {
    file_path: String,
    text: String,
    was_dirty: bool,
    line_ending: LineEnding,
    is_active: bool,
}

// EditorMessage's public variants carry editor-internal payload types (FsEntry,
// OwnedGrepMatch, FileLoadData, CloseAction, SavedTabData, ReadDirError); re-pub any of them
// only together with dropping this expect.
#[derive(Debug, Clone)]
#[expect(private_interfaces)]
pub enum EditorMessage {
    /// The workspace resolved for the current selection (name + path), pushed only by the
    /// dashboard's `sync_workspace_surfaces` — never a substitution for an unresolved one.
    WorkspaceSelected(String, String),
    /// The editor page became visible or hidden. `true` triggers a refresh-on-entry batch;
    /// `false` closes any open modal (modals do not survive a page switch).
    PageVisibilityChanged(bool),
    /// A directory's listing was loaded from the filesystem.
    DirExpanded {
        dir_path: String,
        r#gen: u64,
        entries: Result<Vec<FsEntry>, ReadDirError>,
        /// `true` logs read errors instead of showing a toast (background refreshes); missing
        /// directories are forgotten silently either way.
        quiet: bool,
    },
    /// A GUI-initiated directory deletion succeeded in `workspace_path`: prune the deleted path
    /// and its descendants from the tree state, then re-read the parent.
    ///
    /// `workspace_path` is a staleness guard — the user can switch workspaces while the async
    /// delete is in flight, and only the workspace it was started in may be pruned.
    DirDeleted {
        dir_path: String,
        workspace_path: String,
    },
    ToggleDir(String),
    SelectFile(String),
    TreeFocusToggled,
    TreeNavUp,
    TreeNavDown,
    TreeNavEnter,
    TreeNavLeft,
    TreeNavRight,
    /// Tree scroll: absolute vertical offset and visible viewport height.
    TreeScrolled(f32, f32),
    /// Tab-bar scroll: absolute horizontal offset and visible viewport width.
    TabBarScrolled(f32, f32),
    /// Escape key — dismiss find bar, go-to-line, quick open, tree focus, or close dialog.
    Escape,
    FileLoaded {
        path: String,
        r#gen: u64,
        result: Result<FileLoadData, String>,
    },
    /// Saved tabs loaded from the database with their contents. `Err` carries a restore failure
    /// (DB read or a non-`NotFound` per-file read); the persisted tabs stay untouched and saving
    /// is blocked until a successful reload.
    SavedTabsLoaded {
        tabs_data: Result<Vec<SavedTabData>, String>,
        r#gen: u64,
    },
    TabSelected(usize),
    TabClosed(usize),
    EditorAction(super::editor_widget::EditorAction),
    SaveActiveTab,
    /// Result of a save operation.
    SaveResult {
        path: String,
        result: Result<(), String>,
        /// Hash of the written content, stored as `TabData::saved_text_hash`.
        saved_hash: u64,
    },
    CloseDialog {
        tab_index: usize,
        action: CloseAction,
    },
    CloseOthersDialog {
        keep_idx: usize,
        action: CloseAction,
    },
    /// The workspace gitignore matcher was rebuilt. `workspace` is the path it was built for; a
    /// matcher built for a previously-selected workspace is dropped on arrival. `matcher` is
    /// `None` for no dimming.
    IgnoreMatcherReady {
        workspace: String,
        matcher: Option<ignore::IncrementalIgnore>,
    },
    /// Per-file git statuses forwarded from the Git state (single owner of
    /// `git status --porcelain`); a forward from a workspace other than the selected is dropped.
    GitStatusUpdated {
        workspace: Option<String>,
        statuses: HashMap<String, GitFileStatus>,
    },
    Toast(super::ToastMessage),
    Undo,
    Redo,
    FindToggle,
    FindQueryInput(super::editor_widget::EditorAction),
    FindReplaceInput(super::editor_widget::EditorAction),
    FindNext,
    FindPrev,
    FindReplace,
    FindReplaceAll,
    FindToggleCaseSensitivity,
    /// Manual refresh (Cmd+R / Ctrl+R) of all expanded directory listings from disk.
    RefreshFileTree,
    /// File-tree change event from the global FILE_CHANGE_TX broadcast (coalesced
    /// to at most one per window). Gated on page visibility + a selected workspace.
    FileTreePinged,
    /// Periodic git-dirty probe (1s): stats `.git/index` and `.git/logs/HEAD` and requests a
    /// git-status refresh when they change.
    CheckGitDirty,
    /// Ask the Git state to run its throttled file-change refresh; the Dashboard routes it to
    /// `GitState::update(GitMessage::FileChanged)`.
    RequestGitStatusRefresh,
    CloseOtherTabs(usize),
    /// Periodic (300 ms) check for external changes on the active tab, only with a workspace.
    CheckFileChanges,
    /// The file was reloaded after an on-disk change; the cursor captured *before* the async
    /// read is restored, clamped to the new bounds.
    FileReloaded {
        path: String,
        result: Result<String, String>,
        cursor_line: usize,
        cursor_col: usize,
    },
    GoToLineToggle,
    GoToLineInput(super::editor_widget::EditorAction),
    GoToLineGo,
    QuickOpenToggle,
    QuickOpenInput(super::editor_widget::EditorAction),
    QuickOpenSelect(usize),
    TabSwitchNext,
    TabSwitchPrev,
    CloseActiveTab,
    GlobalSearchToggle,
    GlobalSearchInput(super::editor_widget::EditorAction),
    GlobalSearchResults {
        r#gen: u64,
        results: Vec<OwnedGrepMatch>,
        error: Option<String>,
    },
    GlobalSearchSelect(usize),
    DeleteFileRequested(String),
    DeleteDirectoryRequested(String),
    NewFileRequested(String),
    NewDirectoryRequested(String),
    RevealInFinder(String),
    CopyRelativePath(String),
    CopyAbsolutePath(String),
    ConfirmDelete,
    CancelDelete,
    NewItemSubmit(String),
    NewItemInput(super::editor_widget::EditorAction),
    RenameRequested(String),
    RenameInput(super::editor_widget::EditorAction),
    RenameSubmit,
    RenameCancel,
    RenameCompleted {
        old_path: String,
        new_path: String,
        is_dir: bool,
        result: Result<(), String>,
        dir_entries: Result<Vec<FsEntry>, ReadDirError>,
        /// Generation slot in the parent's `dir_generations`, for stale-result prevention (see
        /// `dir_expanded`).
        rename_gen: u64,
    },
}

#[derive(Debug, Clone)]
struct DeleteConfirmTarget {
    path: String,
    is_dir: bool,
    dirty_tab_count: usize,
    abs_path: String,
}

struct NewItemTarget {
    /// Parent directory path (relative to workspace root; empty = root).
    parent_dir: String,
    is_dir: bool,
    abs_parent: String,
    ws_root: String,
    input_text: SingleLineEditorState,
}

struct RenameTarget {
    path: String,
    abs_path: String,
    is_dir: bool,
    ws_root: String,
    input_text: SingleLineEditorState,
    error: Option<String>,
}

/// The user-facing reason `name` is invalid, or `None` when it passes: empty/whitespace,
/// path separators (`/`, `\`, NUL), `.`/`..`, and Windows OS-reserved names (CON, NUL, PRN,
/// AUX, COM1–COM9, LPT1–LPT9). Shared by [`NewItemSubmit`] and [`RenameSubmit`].
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

// One applier pair (`rekey_map`, `rekey_set`) serves both key families: the `/`-joined
// relative tree keys (`rebase_tree_key`) and the absolute paths of tabs and caches
// (`rebase_abs_path`). Only how a key is rebased differs.

/// The new key for a *relative* tree key under `old_prefix`, or `None`. `old_prefix` carries
/// the trailing separator; a key equal to it maps to `new_prefix`.
fn rebase_tree_key(key: &str, old_prefix: &str, new_prefix: &str) -> Option<String> {
    let rest = key.strip_prefix(old_prefix)?;
    Some(if rest.is_empty() {
        new_prefix.to_string()
    } else {
        format!("{new_prefix}/{rest}")
    })
}

/// Re-key every entry `rebase` accepts, running `modify` on the value. The keys are collected
/// first because the map cannot be mutated while iterated.
fn rekey_map<V>(
    map: &mut HashMap<String, V>,
    rebase: impl Fn(&str) -> Option<String>,
    modify: impl Fn(&mut V),
) {
    for old_key in map.keys().cloned().collect::<Vec<_>>() {
        let Some(new_key) = rebase(&old_key) else {
            continue;
        };
        if let Some(mut value) = map.remove(&old_key) {
            modify(&mut value);
            map.insert(new_key, value);
        }
    }
}

fn rekey_set(set: &mut HashSet<String>, rebase: impl Fn(&str) -> Option<String>) {
    for old_key in set.iter().cloned().collect::<Vec<_>>() {
        let Some(new_key) = rebase(&old_key) else {
            continue;
        };
        set.remove(&old_key);
        set.insert(new_key);
    }
}

/// Re-key the relative file-tree keys of `map` that live under `old_prefix` to
/// live under `new_prefix` instead, running `modify` on each moved value (use
/// `|_| {}` when no modification is needed).
fn rekey_map_prefix<V>(
    map: &mut HashMap<String, V>,
    old_prefix: &str,
    new_prefix: &str,
    modify: impl Fn(&mut V),
) {
    rekey_map(map, |k| rebase_tree_key(k, old_prefix, new_prefix), modify);
}

fn rekey_set_prefix(set: &mut HashSet<String>, old_prefix: &str, new_prefix: &str) {
    rekey_set(set, |k| rebase_tree_key(k, old_prefix, new_prefix));
}

/// Re-key one [`FsEntry`]'s `full_path`, the way [`rebase_tree_key`] re-keys its collections.
fn update_entry_path(entry: &mut FsEntry, old_prefix: &str, new_prefix: &str) {
    if let Some(new_path) = rebase_tree_key(&entry.full_path, old_prefix, new_prefix) {
        entry.full_path = new_path;
    }
}

/// Rebase an absolute workspace path from `old_dir` into `new_dir`, or `None` when it does not
/// live below `old_dir`. Compared through [`Path`], so the platform separator and a trailing
/// separator on either side cannot make the match miss.
fn rebase_abs_path(path: &str, old_dir: &str, new_dir: &str) -> Option<String> {
    let rest = Path::new(path).strip_prefix(old_dir).ok()?;
    Some(if rest.as_os_str().is_empty() {
        new_dir.to_string()
    } else {
        Path::new(new_dir).join(rest).to_string_lossy().into_owned()
    })
}

/// Re-key the absolute paths of `map` under `old_dir`; values carry over unchanged.
fn rekey_abs_map<V>(map: &mut HashMap<String, V>, old_dir: &str, new_dir: &str) {
    rekey_map(map, |k| rebase_abs_path(k, old_dir, new_dir), |_| {});
}

fn rekey_abs_set(set: &mut HashSet<String>, old_dir: &str, new_dir: &str) {
    rekey_set(set, |k| rebase_abs_path(k, old_dir, new_dir));
}

/// `path` is `dir` itself or lies within it: [`crate::util::is_within`], component-wise, so
/// the trailing separator a stored workspace root carries is tolerated.
fn is_within_abs(path: &str, dir: &str) -> bool {
    crate::util::is_within(Path::new(path), Path::new(dir))
}

/// Error classifying a failed directory read, so callers can tell the routine "directory is
/// gone" case (forgotten silently) from real problems that must be surfaced.
#[derive(Debug, Clone)]
enum ReadDirError {
    /// The directory does not exist, or is not a directory — callers silently forget the path.
    NotFound,
    /// The directory exists but could not be read.
    Other(String),
}

/// ENOENT/ENOTDIR mean the directory is gone (a normal, forgettable case). Everything else
/// means it exists but cannot be read — a real problem that must keep warning.
fn classify_read_dir_error(e: &std::io::Error) -> ReadDirError {
    if matches!(
        e.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
    ) {
        ReadDirError::NotFound
    } else {
        ReadDirError::Other(format!("{e}"))
    }
}

/// Read the entries of `rel_path` under the workspace `root` (`root` is the workspace's
/// filesystem path, `rel_path` the subdirectory, empty for root). A missing directory or a
/// non-directory yields [`ReadDirError::NotFound`]; other failures yield [`ReadDirError::Other`].
async fn read_directory_entries(root: &str, rel_path: &str) -> Result<Vec<FsEntry>, ReadDirError> {
    let dir_path = if rel_path.is_empty() {
        root.to_string()
    } else {
        let p = Path::new(root).join(rel_path);
        p.to_string_lossy().to_string()
    };
    let mut entries = match tokio::fs::read_dir(&dir_path).await {
        Ok(rd) => rd,
        Err(e) => return Err(classify_read_dir_error(&e)),
    };

    let mut result: Vec<FsEntry> = Vec::new();
    let mut dirs: Vec<FsEntry> = Vec::new();
    let mut files: Vec<FsEntry> = Vec::new();

    loop {
        let entry = match entries.next_entry().await {
            Ok(Some(entry)) => entry,
            Ok(None) => break,
            // The directory can vanish mid-read; classify it as not-found instead of caching a
            // partial listing.
            Err(e) => return Err(classify_read_dir_error(&e)),
        };
        let name = entry.file_name().to_string_lossy().to_string();
        // Filter out .git directory — it's not a user-editable file.
        if name == ".git" {
            continue;
        }
        if is_os_file(&name) {
            continue;
        }
        let full_path = if rel_path.is_empty() {
            name.clone()
        } else {
            format!("{rel_path}/{name}")
        };
        // Absolute metadata follows symlinks; `DirEntry::file_type()` does not (per Rust docs).
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

    dirs.sort_by_key(|e| e.name.to_lowercase());
    files.sort_by_key(|e| e.name.to_lowercase());
    result.extend(dirs);
    result.extend(files);
    Ok(result)
}

async fn dir_expanded_msg(
    ws_path: String,
    dir_path: String,
    r#gen: u64,
    quiet: bool,
) -> EditorMessage {
    let entries = read_directory_entries(&ws_path, &dir_path).await;
    EditorMessage::DirExpanded {
        dir_path,
        r#gen,
        entries,
        quiet,
    }
}

fn dir_expanded_task(
    ws_path: String,
    dir_path: String,
    r#gen: u64,
    quiet: bool,
) -> Task<EditorMessage> {
    Task::perform(dir_expanded_msg(ws_path, dir_path, r#gen, quiet), |msg| msg)
}

/// Resolve the `.git` entry covering `start` via the shared repo-root walk
/// (`commands::resolve_git_top_level`), so a nested workspace still detects the enclosing
/// repo's staging dir. For a `.git` file (worktree) the detector's stats do not exist, so
/// detection no-ops (accepted edge; the watcher or Cmd+R recovers).
fn find_git_dir(start: &Path) -> Option<PathBuf> {
    crate::git::commands::resolve_git_top_level(start).map(|root| root.join(".git"))
}

/// Stat the detector's two targets as `(index mtime, logs/HEAD mtime)`; each is `None` when
/// the file is absent (unborn repo has no logs/HEAD, a fresh init no index yet).
fn stat_git_dirty_mtimes(git_dir: &Path) -> (Option<SystemTime>, Option<SystemTime>) {
    let index = git_dir.join("index");
    let logs_head = git_dir.join("logs").join("HEAD");
    (
        std::fs::metadata(&index)
            .ok()
            .and_then(|m| m.modified().ok()),
        std::fs::metadata(&logs_head)
            .ok()
            .and_then(|m| m.modified().ok()),
    )
}

/// Coalesced file-tree change pings from the global [`super::git::FILE_CHANGE_TX`] broadcast.
/// A free fn because [`Subscription::run`] takes a `fn() -> Stream` builder.
fn file_tree_change_stream() -> impl futures_util::Stream<Item = EditorMessage> {
    super::common::coalesced_broadcast_producer(
        64,
        &super::git::FILE_CHANGE_TX,
        Duration::from_millis(300),
        || EditorMessage::FileTreePinged,
    )
}

/// Reject oversized or binary content (null bytes, invalid UTF-8) with a user-facing message.
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

fn file_loaded_msg(
    path: String,
    r#gen: u64,
    result: Result<FileLoadData, String>,
) -> EditorMessage {
    EditorMessage::FileLoaded {
        path,
        r#gen,
        result,
    }
}

/// Load a file from disk, detecting indent style, line ending and trailing newline.
async fn load_file_data(full_path: String, r#gen: u64) -> EditorMessage {
    let bytes = match tokio::fs::read(&full_path).await {
        Ok(b) => b,
        Err(e) => {
            return file_loaded_msg(full_path, r#gen, Err(format!("Cannot read file: {e}")));
        }
    };

    // Validate size and detect binary content (null bytes).
    if let Err(e) = validate_file_content(&bytes) {
        return file_loaded_msg(full_path, r#gen, Err(e));
    }
    // Binary data can be valid UTF-8 with no null bytes (encoded payloads, structured formats),
    // so reject invalid UTF-8 on top of the null-byte check above.
    let Ok(text) = String::from_utf8(bytes) else {
        return file_loaded_msg(
            full_path,
            r#gen,
            Err("Binary file detected (invalid UTF-8)".to_string()),
        );
    };

    let data = FileLoadData {
        path: full_path,
        line_ending: detect_line_ending(&text),
        text,
    };
    file_loaded_msg(data.path.clone(), r#gen, Ok(data))
}

/// Spawn a file load that answers with [`EditorMessage::FileLoaded`]. A free fn because its
/// callers (`open_file_in_editor`, `select_file`) bump generations differently but share the spawn.
fn spawn_file_load(abs_path: String, file_gen: u64) -> Task<EditorMessage> {
    Task::perform(load_file_data(abs_path, file_gen), |msg| msg)
}

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

/// Save the current tabs for a workspace. `gen_counter` is a pre-write guard: a newer save
/// supersedes this one and the write is skipped. The write is fire-and-forget — that guard is
/// the only staleness protection needed.
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

/// Build the async save task for tab `idx`. Panics in debug builds if `idx` is out of bounds;
/// every current caller validates the index first.
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

/// Set the dirty flag by comparing the current text hash with the saved hash. A free fn (not
/// `&mut self`) so callers avoid borrowing `self.tabs` and `self.tab_contents` at the same time.
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
/// Transparent text button used by the find/replace and go-to-line bars.
fn bar_text_button(
    label: &str,
    message: EditorMessage,
    size: f32,
    color: Color,
    padding: [f32; 2],
) -> Element<'_, EditorMessage> {
    button(text(label).size(size).color(color))
        .on_press(message)
        .style(theme::button_transparent)
        .padding(padding)
        .into()
}

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
                ignored: false,
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

/// Build the in-process gitignore matcher for `workspace_path`.
///
/// Crate defaults are kept for `require_git`/`parents` — they give the
/// subdir-of-repo behavior (parent repo .gitignore applies) and the
/// no-`.git`-no-dimming behavior for non-git workspaces. The ripgrep
/// defaults are explicitly disabled: dotfiles are NOT treated as ignored
/// (`hidden(false)`) and `.ignore` files are NOT read (`ignore(false)`).
/// `current_dir` lets global git excludes resolve against the workspace.
///
/// `None` (no matcher could be built) means "no dimming" — never an error.
#[must_use]
fn build_ignore_matcher(workspace_path: &Path) -> Option<ignore::IncrementalIgnore> {
    let mut builder = ignore::WalkBuilder::new(workspace_path);
    builder.hidden(false);
    builder.ignore(false);
    builder.current_dir(workspace_path);
    builder.build_matchers().into_iter().next()
}

/// Stamp `ignored` on every node by querying the matcher with the node's
/// workspace-relative path and `is_dir` flag. Only an IGNORE verdict dims —
/// negated/whitelisted patterns stay undimmed. Runs as a post-build pass so
/// BOTH the rebuild_tree path and the collapse_dir raw-rebuild path stamp,
/// and so `build_hierarchical_tree`'s signature stays unchanged.
fn stamp_ignored_nodes(
    nodes: &mut [widgets::TreeNode],
    matcher: &mut Option<ignore::IncrementalIgnore>,
) {
    for node in nodes {
        if let Some(m) = matcher.as_mut() {
            node.ignored = m.matched(node.full_path.as_str(), node.is_dir).is_ignore();
        }
        stamp_ignored_nodes(&mut node.children, matcher);
    }
}

/// Run a global (find-in-files) grep search: a 300 ms debounce (cancelled when a newer generation
/// supersedes), engine init, `picker.grep()` on the blocking pool, then owned extraction.
async fn run_global_search(
    ws_path: String,
    ws_name: String,
    query: String,
    gs_gen: u64,
) -> EditorMessage {
    tokio::time::sleep(Duration::from_millis(GLOBAL_SEARCH_DEBOUNCE_MS)).await;

    let entry = match crate::workspace_index::resolve_engine(
        &ws_name,
        &ws_path,
        "Search engine not ready: ",
    )
    .await
    {
        Ok(e) => e,
        Err(e) => {
            return EditorMessage::GlobalSearchResults {
                r#gen: gs_gen,
                results: Vec::new(),
                error: Some(e),
            };
        }
    };

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
            // Overrides `smart_case` when set; `None` keeps the smart-case default.
            casing: None,
            max_file_size: MAX_FILE_SIZE,
            max_matches_per_file: GLOBAL_SEARCH_MATCHES_PER_FILE,
            file_offset: 0,
            page_limit: MAX_GLOBAL_SEARCH_RESULTS,
            time_budget_ms: 3_000,
            // As in the search tool: the budget only counts once something matched.
            enforce_time_budget: false,
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

        // Extract owned data while holding the picker lock.
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

pub struct EditorState {
    selected_workspace_name: Option<String>,
    selected_workspace_path: Option<String>,
    /// Whether the editor page is visible — single source of truth for page-visible gating;
    /// set before any refresh-on-entry task is dispatched.
    page_visible: bool,
    /// Monotonically increasing generation counter for stale-result prevention.
    generation: u64,
    /// Generation for saved-tabs restoration, kept separate from `generation` so concurrent
    /// file loads and directory expansions cannot collide with it.
    saved_tabs_gen: u64,
    dir_generations: HashMap<String, u64>,
    /// Per-file generation counters (stale `FileLoaded` results).
    file_generations: HashMap<String, u64>,
    loading_dirs: HashSet<String>,
    /// Directory entries loaded from the filesystem (keyed by full path).
    dir_entries: HashMap<String, Vec<FsEntry>>,
    file_tree: FileTree,
    selected_file: Option<String>,
    tabs: Vec<Tab>,
    active_tab_index: usize,
    tab_contents: HashMap<String, TabData>,
    tab_scroll_id: Id,
    /// Horizontal scroll offset of the tab bar (px), updated via `on_scroll` so scroll-into-view
    /// needs no layout pass.
    tab_scroll_x: f32,
    /// Visible width of the tab-bar viewport (px). `None` until the first `on_scroll`, which only
    /// fires while the content overflows; tab reveal then falls back to aligning the estimated
    /// left edge, which iced ignores while the content fits.
    tab_viewport_w: Option<f32>,
    /// Pending close action to execute after the next successful save.
    pending_close: Option<PendingCloseAction>,
    /// Whether the workspace tabs were loaded at least once this session (a successful restore or
    /// a direct open; a failed restore leaves it false). Independent of [`Self::tabs_error`].
    session_initialized: bool,
    /// Persistent failure of the last saved-tabs restore, as shown: the read error plus its
    /// consequence. While set, tab persistence stays blocked so an incomplete in-memory set can
    /// never overwrite the still-intact persisted tabs.
    tabs_error: Option<String>,
    /// Directory whose async expansion should advance focus to the first child.
    pending_enter_dir: Option<String>,
    /// Cached git status per file path (relative to workspace root).
    git_status_cache: HashMap<String, GitFileStatus>,
    /// In-flight guard for a git-status refresh that resolves via a forwarded `GitStatusUpdated`:
    /// set when one is awaited (`workspace_selected`), cleared only by that handler or a clear.
    git_status_loading: bool,
    /// The `.git` directory (or `.git` file for worktrees/submodules) covering the selected
    /// workspace, resolved at `workspace_selected` by walking up from the workspace path; `None`
    /// outside a repo. The mtime detector stats `.git/index` and `.git/logs/HEAD` under it.
    git_dir: Option<PathBuf>,
    /// Latch of `(index mtime, logs/HEAD mtime)` from the last git-status resolution: `None`
    /// before the first probe, an inner `None` when that file is absent (fresh/unborn repo). A
    /// change between probes means staging changed and a refresh is owed; re-latched on every
    /// `GitStatusUpdated` so a git-status write to `.git/index` does not re-trigger the detector.
    git_dirty_mtimes: Option<(Option<SystemTime>, Option<SystemTime>)>,
    /// The in-process gitignore matcher for the current workspace; `None` means no dimming (no
    /// matcher could be built, or not a repo). Rebuilt on selection and tree refresh.
    ignore_matcher: Option<ignore::IncrementalIgnore>,
    /// Shared atomic counter for save staleness: written on every save initiation, read by
    /// in-flight tasks to see whether a newer save superseded them.
    tab_save_counter: Arc<AtomicU64>,
    /// Last-known mtime per open file, for the external-change poll.
    file_mtimes: HashMap<String, SystemTime>,
    /// Paths already toasted as deleted (prevents a toast every 300 ms).
    deleted_file_toasted: HashSet<String>,
    /// The open modal overlay. [`ModalKind`] makes them mutually exclusive: at most one at a time.
    active_modal: Option<ModalKind>,
    /// Workspace files cached for quick-open, rebuilt on each toggle from the expanded dirs.
    all_workspace_files: Vec<String>,
    global_search_gen: u64,
    /// Pending jump `(abs_path, 1-based line, expected_file_gen)`, consumed by the `FileLoaded`
    /// handler only when both path and generation match.
    pending_goto: Option<(String, usize, u64)>,
    /// Go-to-line input buffer, owned here so it survives the modal's open/close cycles.
    goto_line_input: SingleLineEditorState,
}

/// The open modal overlay, in Escape-dismissal priority order (GlobalSearch highest,
/// CloseOthers lowest); each variant carries that overlay's state.
enum ModalKind {
    GlobalSearch(GlobalSearchState),
    GotoLine,
    QuickOpen(QuickOpenState),
    Rename(RenameTarget),
    NewItem(NewItemTarget),
    DeleteConfirm(DeleteConfirmTarget),
    CloseDialog(usize),
    CloseOthers(usize),
}

struct QuickOpenState {
    filter: SingleLineEditorState,
    selected_index: usize,
    results: Vec<String>,
}

impl EditorState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            selected_workspace_name: None,
            selected_workspace_path: None,
            page_visible: false,
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
            tab_scroll_x: 0.0,
            tab_viewport_w: None,
            pending_close: None,
            session_initialized: false,
            tabs_error: None,
            pending_enter_dir: None,
            git_status_cache: HashMap::new(),
            git_status_loading: false,
            git_dir: None,
            git_dirty_mtimes: None,
            ignore_matcher: None,
            tab_save_counter: Arc::new(AtomicU64::new(0)),
            file_mtimes: HashMap::new(),
            deleted_file_toasted: HashSet::new(),
            active_modal: None,
            all_workspace_files: Vec::new(),
            global_search_gen: 0,
            pending_goto: None,
            goto_line_input: SingleLineEditorState::new(""),
        }
    }

    /// The filesystem root of the currently selected workspace, if any.
    /// Always absolute — validated & canonicalized at creation time.
    #[inline]
    fn workspace_root(&self) -> Option<&str> {
        self.selected_workspace_path.as_deref()
    }

    fn abs_path(&self, rel_path: &str) -> Option<String> {
        self.selected_workspace_path
            .as_ref()
            .map(|ws| Path::new(ws).join(rel_path).to_string_lossy().to_string())
    }

    /// Rebuild the hierarchical tree and visible node list from `dir_entries` and `expanded_dirs`.
    /// Callers that also need `tree_focused = true` set it separately after this call.
    fn rebuild_tree(&mut self) {
        let mut nodes =
            build_hierarchical_tree(&self.dir_entries, &self.file_tree.expanded_dirs, "");
        stamp_ignored_nodes(&mut nodes, &mut self.ignore_matcher);
        self.file_tree.nodes = nodes;
        self.file_tree.rebuild_visible();
    }

    /// Forget `dir_path` and its descendants from the tree refresh/cache state — expanded dirs,
    /// cached listings, in-flight loads, generations, mtimes, toast guards, quick-open cache and
    /// selection — and drop the stale [`FsEntry`] from the parent's listing so the node goes at
    /// once. Call it whenever a directory no longer exists, then call [`Self::rebuild_tree`]. Open
    /// tabs stay deliberately untouched: externally deleted files keep unsaved work, while GUI
    /// deletes close their own tabs first.
    fn forget_directory(&mut self, dir_path: &str) {
        let is_root = dir_path.is_empty();
        let rel_prefix = format!("{dir_path}/");
        let within_rel = |p: &str| is_root || p == dir_path || p.starts_with(&rel_prefix);

        if !is_root {
            if let Some(parent) = Path::new(dir_path).parent() {
                let parent = parent.to_string_lossy();
                if let Some(parent_entries) = self.dir_entries.get_mut(parent.as_ref()) {
                    parent_entries.retain(|e| !within_rel(&e.full_path));
                }
            }
        }

        self.file_tree.expanded_dirs.retain(|p| !within_rel(p));
        self.dir_entries.retain(|p, _| !within_rel(p));
        self.loading_dirs.retain(|p| !within_rel(p));
        self.dir_generations.retain(|p, _| !within_rel(p));
        self.all_workspace_files.retain(|p| !within_rel(p));

        // Absolute-path-keyed caches. For the root the directory is the workspace root itself, so
        // every file is pruned ("all descendants"); membership is component-wise, so the root's
        // spelling (Windows separators, any trailing slash) need not be guessed at.
        let abs_dir = if is_root {
            self.selected_workspace_path.clone()
        } else {
            self.abs_path(dir_path)
        };
        let within_abs = |p: &str| abs_dir.as_deref().is_some_and(|dir| is_within_abs(p, dir));
        self.file_generations.retain(|p, _| !within_abs(p));
        self.file_mtimes.retain(|p, _| !within_abs(p));
        self.deleted_file_toasted.retain(|p| !within_abs(p));

        if let Some(ref sel) = self.selected_file {
            if within_rel(sel) || within_abs(sel) {
                self.selected_file = None;
            }
        }
        if self.pending_enter_dir.as_deref().is_some_and(within_rel) {
            self.pending_enter_dir = None;
        }
    }

    fn bump_generation(&mut self) -> u64 {
        let g = self.generation.wrapping_add(1);
        self.generation = g;
        g
    }

    /// Start an async load of a directory's entries. `Some` only when a workspace is selected and
    /// the caller already checked `!dir_entries.contains_key(dir_path)`; `None` means the caller
    /// should early-return `Task::none()`. The caller MUST then rebuild the tree, setting
    /// `pending_enter_dir` first if focus should advance when the load completes.
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
        let dir_gen = self.bump_generation();
        self.dir_generations.insert(dir_path.to_string(), dir_gen);
        self.loading_dirs.insert(dir_path.to_string());
        let d_path = dir_path.to_string();
        Some(dir_expanded_task(ws_path, d_path, dir_gen, false))
    }

    /// Expand `path`, then either async-load it (setting [`Self::pending_enter_dir`] so
    /// [`Self::dir_expanded`] advances focus on arrival) or, when its entries are cached, expand
    /// and focus the first child now. `Task::none()` means no workspace is selected.
    fn expand_dir_and_focus(&mut self, path: &str, label: &str) -> Task<EditorMessage> {
        self.selected_file = None;
        self.file_tree.expanded_dirs.insert(path.to_string());

        let needs_async_load = !self.dir_entries.contains_key(path);

        if needs_async_load {
            let Some(task) = self.load_dir_async(path, label) else {
                return Task::none();
            };
            self.pending_enter_dir = Some(path.to_string());
            self.rebuild_tree();
            return task;
        }

        self.rebuild_tree();
        self.file_tree
            .expand_dir_and_focus_first_child::<EditorMessage>(path)
    }

    /// Collapse `path`, keep focus on it, and let [`FileTree::collapse_dir_and_keep_focus`] manage
    /// focus and scroll (the caller guarantees `expanded_dirs` holds `path`). Raw
    /// [`build_hierarchical_tree`] rather than [`rebuild_tree`] because that helper calls
    /// [`rebuild_visible`] itself; the fresh nodes are still stamped so dimming survives.
    fn collapse_dir(&mut self, path: &str) -> Task<EditorMessage> {
        self.file_tree.expanded_dirs.remove(path);
        let mut nodes =
            build_hierarchical_tree(&self.dir_entries, &self.file_tree.expanded_dirs, "");
        stamp_ignored_nodes(&mut nodes, &mut self.ignore_matcher);
        self.file_tree.nodes = nodes;
        self.file_tree
            .collapse_dir_and_keep_focus::<EditorMessage>(path)
    }

    /// The inline rename editor for this tree node, or `None` when the node is not the rename
    /// target (callers then render their normal label).
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
        let editor = super::editor_widget::EditorWidget::new(&rt.input_text.buffer)
            .single_line(true)
            .show_gutter(false)
            .code_mode(false)
            .enter(EnterBehavior::Submit)
            .placeholder("")
            .padding(theme::PAD_2)
            .background(Some(iced::Color::TRANSPARENT))
            .id(Id::from(format!("rename_input_{}", node.full_path)))
            .keyboard_passthrough(super::is_dashboard_shortcut);
        let input: Element<'a, EditorMessage> =
            iced::Element::new(editor).map(|action| match action {
                super::editor_widget::EditorAction::Submit => EditorMessage::RenameSubmit,
                other => EditorMessage::RenameInput(other),
            });
        if let Some(ref err) = rt.error {
            Some(
                column![
                    input,
                    text(err).size(theme::TEXT_10).color(theme::STATUS_ERROR)
                ]
                .spacing(0)
                .into(),
            )
        } else {
            Some(input)
        }
    }

    /// The CheckFileChanges/CheckGitDirty timers run only while the page is visible; the
    /// FILE_CHANGE_TX subscription is always on (cheap, and it drops pings while hidden).
    pub fn subscription(&self, dashboard_modal_open: bool) -> Subscription<EditorMessage> {
        let mut subs: Vec<Subscription<EditorMessage>> = Vec::new();
        // Always on: file-tree change pings, coalesced to one per 300 ms. The handler drops them
        // while hidden; refresh-on-entry catches up.
        subs.push(Subscription::run(file_tree_change_stream));
        if self.page_visible && self.selected_workspace_name.is_some() {
            // Auto-refresh tick for external changes: only the active tab is polled, and a dirty
            // tab is never auto-reloaded.
            subs.push(
                iced::time::every(Duration::from_millis(300))
                    .map(|_| EditorMessage::CheckFileChanges),
            );
        }
        if self.page_visible && self.git_dir.is_some() {
            // Commit/staging mtime detector; a change requests a git-status refresh.
            subs.push(
                iced::time::every(Duration::from_secs(1)).map(|_| EditorMessage::CheckGitDirty),
            );
        }
        // Keyboard events are always listened for — tree navigation may be active. The modal flag
        // is threaded through so global shortcuts (Cmd+S, Cmd+F, …) still work while
        // tree-navigation keys are swallowed behind a dashboard modal. `filter_map` needs a
        // non-capturing closure, so the flag rides along via `with`.
        subs.push(
            keyboard::listen()
                .with(dashboard_modal_open)
                .filter_map(|(modal_open, event)| map_editor_shortcut(event, modal_open)),
        );
        Subscription::batch(subs)
    }

    const fn active_tab_idx(&self) -> Option<usize> {
        let idx = self.active_tab_index;
        if idx >= self.tabs.len() {
            None
        } else {
            Some(idx)
        }
    }

    fn active_tab(&self) -> Option<(usize, String)> {
        let idx = self.active_tab_idx()?;
        Some((idx, self.tabs[idx].path.clone()))
    }

    fn is_find_bar_open(&self) -> bool {
        self.active_tab_idx()
            .and_then(|idx| self.tabs.get(idx))
            .and_then(|tab| self.tab_contents.get(&tab.path))
            .and_then(|data| data.find_replace_state.as_ref())
            .is_some()
    }

    /// Save the tabs for the selected workspace, or `None` when tabs are not initialized yet, the
    /// saved-tabs restore failed (saving would delete the persisted set), or no workspace is
    /// selected. The pre-write guard inside the task skips a save a newer one has superseded.
    fn try_save_current_tabs(&self) -> Option<Task<EditorMessage>> {
        if !self.session_initialized || self.tabs_error.is_some() {
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

    /// [`Self::try_save_current_tabs`] with a `Task::none()` fallback. Call that one directly only
    /// to inspect whether a save was dispatched.
    fn save_current_tabs(&mut self) -> Task<EditorMessage> {
        self.try_save_current_tabs().unwrap_or(Task::none())
    }

    /// Scroll the tab bar so the active tab is visible — deliberately stricter than the file
    /// tree's `widgets::ScrollMode::ScrollIntoView`, which tolerates partial visibility during
    /// wheel scrolling: a tab is visible only when both estimated edges are inside the viewport,
    /// so selecting an already-visible tab never scrolls. A clipped tab moves by exactly the
    /// overflow — left-clipped flush to the left edge, right-clipped via `scroll_by` (clamped to
    /// the max offset; the old absolute `index × width` target was only clamped at draw time and
    /// could land at the far right). An unknown viewport width (no `on_scroll` yet, which only
    /// fires while the content overflows) falls back to aligning the estimated left edge — a
    /// visual no-op while the content fits, a genuine reveal when it does not.
    ///
    /// `tab_scroll_x` is updated synchronously so consecutive selections within one frame (held
    /// Ctrl+Tab) judge against the offset this call produced, before `on_scroll` fires.
    fn scroll_to_active_tab(&mut self) -> Task<EditorMessage> {
        if self.tabs.is_empty() {
            return Task::none();
        }
        let idx = self.active_tab_index.min(self.tabs.len() - 1);

        let left = estimated_tab_left(&self.tabs, idx);
        let Some(viewport_w) = self.tab_viewport_w else {
            self.tab_scroll_x = left;
            return iced::widget::operation::scroll_to(
                self.tab_scroll_id.clone(),
                iced::widget::operation::AbsoluteOffset { x: left, y: 0.0 },
            );
        };

        if estimated_content_width(&self.tabs) <= viewport_w {
            return Task::none();
        }

        let right = left + estimate_tab_width(&self.tabs[idx]);
        let viewport_right = self.tab_scroll_x + viewport_w;

        if left < self.tab_scroll_x {
            // Left-clipped: bring it flush with the viewport's left edge.
            let delta = self.tab_scroll_x - left;
            self.tab_scroll_x = left;
            iced::widget::operation::scroll_by(
                self.tab_scroll_id.clone(),
                iced::widget::operation::AbsoluteOffset { x: -delta, y: 0.0 },
            )
        } else if right > viewport_right {
            // Right-clipped: advance by exactly the overflow; `scroll_by` clamps to the max offset.
            let delta = right - viewport_right;
            let max_scroll = (estimated_content_width(&self.tabs) - viewport_w).max(0.0);
            self.tab_scroll_x = (self.tab_scroll_x + delta).clamp(0.0, max_scroll);
            iced::widget::operation::scroll_by(
                self.tab_scroll_id.clone(),
                iced::widget::operation::AbsoluteOffset { x: delta, y: 0.0 },
            )
        } else {
            Task::none()
        }
    }

    /// Switch to tab `new_idx` and scroll it into view, without persisting.
    fn scroll_to_tab(&mut self, new_idx: usize) -> Task<EditorMessage> {
        if new_idx >= self.tabs.len() {
            return Task::none();
        }
        self.active_tab_index = new_idx;
        self.scroll_to_active_tab()
    }

    /// Switch to `idx` and persist the tab list.
    fn switch_to_tab(&mut self, idx: usize) -> Task<EditorMessage> {
        if idx >= self.tabs.len() {
            return Task::none();
        }
        self.active_tab_index = idx;
        Task::batch(vec![self.scroll_to_active_tab(), self.save_current_tabs()])
    }

    /// Switch one tab in `direction`, wrapping. `Task::none()` when a modal overlay is open or
    /// there is only one tab.
    fn switch_tab_relative(&mut self, direction: TabDirection) -> Task<EditorMessage> {
        if self.active_modal.is_some() || self.tabs.len() <= 1 {
            return Task::none();
        }
        let len = self.tabs.len();
        let new_idx = match direction {
            TabDirection::Next => (self.active_tab_index + 1) % len,
            TabDirection::Prev => (self.active_tab_index + len - 1) % len,
        };
        self.scroll_to_tab(new_idx)
    }

    /// Collect the workspace files known from the cached directory entries, for quick-open.
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

    /// Case-insensitive substring filter over the cached workspace file list.
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

    /// Open a file by workspace-relative (or already absolute) path, reusing its tab when open.
    fn open_file_in_editor(&mut self, path: &str) -> Task<EditorMessage> {
        let Some(ws) = self.workspace_root() else {
            return Task::none();
        };
        // `Path::is_absolute` is the platform's own predicate (Unix: leading `/`; Windows: a
        // drive/UNC/verbatim prefix), so `C:\p` or `\\?\C:\p` is kept as-is rather than joined
        // onto the workspace root. Both branches yield the same string for every path the editor
        // can receive, but staying explicit avoids resting on `PathBuf::push` discarding the base
        // for an absolute path.
        let abs_path = if Path::new(path).is_absolute() {
            path.to_string()
        } else {
            Path::new(&ws).join(path).to_string_lossy().to_string()
        };

        if let Some(existing_idx) = self.tabs.iter().position(|t| t.path == abs_path) {
            return self.scroll_to_tab(existing_idx);
        }

        self.file_tree.tree_focused = false;
        self.pending_enter_dir = None;

        let file_path = abs_path;
        let file_gen = self.bump_generation();
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

    /// Close tab `idx`: a dirty tab opens the close dialog, a clean one is removed and persisted.
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

    /// Apply an owned undo/redo snapshot to tab `idx`, doing a fresh O(1) lookup of
    /// `tab_contents` (the caller resolved the path). Panics if `idx` is out of bounds for `tabs`.
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
                .move_to(snapshot.cursor.line, snapshot.cursor.column);
        }
        update_dirty_flag(&mut self.tabs, &self.tab_contents, idx, &path);
    }

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

    /// Undo/redo, bailing out while the find bar or a modal overlay is open: that overlay's own
    /// editor widget handles Cmd+Z/Cmd+Shift+Z per field, so the code editor must not undo.
    fn handle_undo_or_redo(&mut self, is_redo: bool) -> Task<EditorMessage> {
        if self.is_find_bar_open() || self.active_modal.is_some() {
            Task::none()
        } else {
            self.apply_undo_or_redo(is_redo)
        }
    }

    /// Clear all workspace-scoped editor state. `selected_workspace_name`,
    /// `selected_workspace_path`, `generation` and `saved_tabs_gen` are left alone — the call
    /// site manages those.
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
        self.git_dir = None;
        self.git_dirty_mtimes = None;
        self.ignore_matcher = None;
        self.session_initialized = false;
        self.tabs_error = None;
        self.active_modal = None;
        self.pending_close = None;
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
        // Per-workspace scroll state: reset so the next selection re-learns the viewport.
        self.tab_scroll_x = 0.0;
        self.tab_viewport_w = None;
    }

    /// Start creating a file/directory in `parent_dir`: resolve its absolute path and open the name
    /// input (any active modal is replaced — `active_modal` enforces mutual exclusion).
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
            input_text: SingleLineEditorState::new(""),
        }));
        iced::widget::operation::focus::<EditorMessage>(Id::new(NEW_ITEM_INPUT_ID))
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: EditorMessage) -> Task<EditorMessage> {
        match msg {
            EditorMessage::WorkspaceSelected(ref name, ref path) => {
                self.workspace_selected(name, path)
            }

            EditorMessage::PageVisibilityChanged(visible) => self.page_visibility_changed(visible),

            EditorMessage::SavedTabsLoaded { tabs_data, r#gen } => {
                self.saved_tabs_loaded(tabs_data, r#gen)
            }

            EditorMessage::DirExpanded {
                dir_path,
                r#gen,
                entries,
                quiet,
            } => self.dir_expanded(&dir_path, r#gen, entries, quiet),

            EditorMessage::DirDeleted {
                dir_path,
                workspace_path,
            } => self.dir_deleted(&dir_path, &workspace_path),

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

            EditorMessage::GoToLineToggle => self.go_to_line_toggle(),

            EditorMessage::GoToLineInput(action) => {
                focus_navigation_task(&action).unwrap_or_else(|| self.go_to_line_input(action))
            }

            EditorMessage::GoToLineGo => self.go_to_line_go(),

            EditorMessage::GlobalSearchToggle => self.global_search_toggle(),

            EditorMessage::GlobalSearchInput(action) => {
                focus_navigation_task(&action).unwrap_or_else(|| self.global_search_input(action))
            }

            EditorMessage::GlobalSearchResults {
                r#gen,
                results,
                error,
            } => self.global_search_results(r#gen, results, error),

            EditorMessage::GlobalSearchSelect(idx) => self.global_search_select(idx),

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

            EditorMessage::NewItemInput(action) => {
                focus_navigation_task(&action).unwrap_or_else(|| self.new_item_input(action))
            }

            EditorMessage::RenameRequested(path) => self.rename_requested(&path),

            EditorMessage::RenameInput(action) => {
                focus_navigation_task(&action).unwrap_or_else(|| self.rename_input(action))
            }

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

            EditorMessage::QuickOpenToggle => self.quick_open_toggle(),

            EditorMessage::QuickOpenInput(action) => {
                focus_navigation_task(&action).unwrap_or_else(|| self.quick_open_input(action))
            }

            EditorMessage::QuickOpenSelect(idx) => self.quick_open_select(idx),

            EditorMessage::TabSwitchNext => self.switch_tab_relative(TabDirection::Next),
            EditorMessage::TabSwitchPrev => self.switch_tab_relative(TabDirection::Prev),

            EditorMessage::CloseActiveTab => self.close_active_tab(),

            EditorMessage::TreeFocusToggled => self.tree_focus_toggled(),

            EditorMessage::TreeScrolled(scroll_y, viewport_h) => {
                self.tree_scrolled(scroll_y, viewport_h)
            }

            EditorMessage::TabBarScrolled(scroll_x, viewport_w) => {
                self.tab_bar_scrolled(scroll_x, viewport_w)
            }

            EditorMessage::TreeNavUp => self.navigate_tree_vertical(TreeNavDirection::Up),

            EditorMessage::TreeNavDown => self.navigate_tree_vertical(TreeNavDirection::Down),

            EditorMessage::TreeNavEnter => self.tree_nav_enter(),

            EditorMessage::TreeNavLeft => self.tree_nav_left(),

            EditorMessage::TreeNavRight => self.tree_nav_right(),

            EditorMessage::Undo => self.handle_undo_or_redo(false),

            EditorMessage::Redo => self.handle_undo_or_redo(true),

            EditorMessage::FindToggle => self.find_toggle(),

            EditorMessage::FindQueryInput(action) => {
                focus_navigation_task(&action).unwrap_or_else(|| self.find_query_input(action))
            }

            EditorMessage::FindReplaceInput(action) => {
                focus_navigation_task(&action).unwrap_or_else(|| self.find_replace_input(action))
            }

            EditorMessage::FindNext => self.navigate_find_match(FindDirection::Next),

            EditorMessage::FindPrev => self.navigate_find_match(FindDirection::Prev),

            EditorMessage::FindReplace => self.find_replace(),

            EditorMessage::FindReplaceAll => self.find_replace_all(),

            EditorMessage::FindToggleCaseSensitivity => self.find_toggle_case_sensitivity(),

            EditorMessage::RefreshFileTree => self.refresh_file_tree(),

            EditorMessage::FileTreePinged => self.file_tree_pinged(),

            EditorMessage::CheckGitDirty => self.check_git_dirty(),

            EditorMessage::RequestGitStatusRefresh => {
                // Only reachable by direct/test routing; production hits the Dashboard route.
                Task::none()
            }

            EditorMessage::Toast(_) => Task::none(),

            EditorMessage::GitStatusUpdated {
                workspace,
                statuses,
            } => self.git_status_updated(workspace.as_deref(), statuses),

            EditorMessage::IgnoreMatcherReady { workspace, matcher } => {
                self.ignore_matcher_ready(&workspace, matcher)
            }

            EditorMessage::CheckFileChanges => self.check_file_changes(),

            EditorMessage::FileReloaded {
                path,
                result,
                cursor_line,
                cursor_col,
            } => self.file_reloaded(path, result, cursor_line, cursor_col),
        }
    }

    /// Adopt a resolved workspace: reset state, read the root, restore saved tabs, resolve git.
    fn workspace_selected(&mut self, name: &str, path: &str) -> Task<EditorMessage> {
        let mut tasks: Vec<Task<EditorMessage>> = Vec::new();

        self.selected_workspace_name = Some(name.to_string());
        self.selected_workspace_path = Some(path.to_string());

        let r#gen = self.bump_generation();
        let saved_gen = self.saved_tabs_gen.wrapping_add(1);
        self.saved_tabs_gen = saved_gen;
        self.clear_workspace_editor_state();
        // Resolve the covering `.git` and drop the mtime latch so the first probe re-latches.
        self.git_dir = find_git_dir(Path::new(path));
        self.git_dirty_mtimes = None;
        // GitState eagerly refreshes on a workspace switch and forwards the statuses via
        // GitStatusUpdated, so mark the in-flight guard: colors stay absent rather than stale until
        // then. Set only when a forward can happen — a non-git workspace produces no snapshot.
        self.git_status_loading = self.git_dir.is_some();

        self.dir_generations.insert(String::new(), r#gen);

        tasks.push(dir_expanded_task(
            path.to_string(),
            String::new(),
            r#gen,
            false,
        ));

        let tab_ws = name.to_string();
        let tab_path = path.to_string();
        let tab_gen = saved_gen;
        let load_tabs_task = Task::perform(
            async move {
                let store = crate::workspace::store();
                let records = match store.load_editor_tabs(&tab_ws).await {
                    Ok(records) => records,
                    Err(e) => {
                        tracing::warn!(?e, workspace = %tab_ws, "Failed to load editor tabs");
                        return EditorMessage::SavedTabsLoaded {
                            tabs_data: Err(format!("failed to load saved editor tabs: {e}")),
                            r#gen: tab_gen,
                        };
                    }
                };
                let mut loaded: Vec<SavedTabData> = Vec::new();
                for record in &records {
                    // Belt-and-suspenders: `load_editor_tabs` filters these, but guard anyway.
                    if record.file_path.is_empty() || record.file_path.trim().is_empty() {
                        tracing::warn!(
                            workspace = %tab_ws,
                            tab_order = record.tab_order,
                            "Skipping editor tab with empty file_path in GUI loader"
                        );
                        continue;
                    }
                    let file_path = Path::new(&tab_path)
                        .join(&record.file_path)
                        .to_string_lossy()
                        .to_string();

                    let loaded_text = if let Some(dirty) = record.dirty_content.clone() {
                        Some(dirty)
                    } else {
                        match tokio::fs::read(&file_path).await {
                            Ok(bytes) => {
                                if validate_file_content(&bytes).is_ok() {
                                    String::from_utf8(bytes).ok()
                                } else {
                                    None
                                }
                            }
                            // A missing file is a legitimate absence — the tab
                            // is dropped. Any other read failure leaves the
                            // restore set incomplete, so it must surface.
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                            Err(e) => {
                                tracing::warn!(?e, path = %file_path, "Failed to read saved editor tab");
                                return EditorMessage::SavedTabsLoaded {
                                    tabs_data: Err(format!(
                                        "failed to read saved tab {file_path}: {e}"
                                    )),
                                    r#gen: tab_gen,
                                };
                            }
                        }
                    };

                    if let Some(text) = loaded_text {
                        let line_ending = detect_line_ending(&text);
                        loaded.push(SavedTabData {
                            file_path,
                            text,
                            was_dirty: record.is_dirty,
                            line_ending,
                            is_active: record.is_active,
                        });
                    }
                }
                EditorMessage::SavedTabsLoaded {
                    tabs_data: Ok(loaded),
                    r#gen: tab_gen,
                }
            },
            |msg| msg,
        );
        tasks.push(load_tabs_task);

        tasks.push(Self::spawn_ignore_matcher(path));

        Task::batch(tasks)
    }

    /// Handle editor-page visibility change (single source of truth for whether the editor is the
    /// active page). `false` closes any open modal (modals do not survive a page switch) and leaves
    /// unsaved changes in place without warning; `true` refreshes on entry, replaying tree changes
    /// that arrived while hidden (and rebuilding the ignore matcher with them).
    fn page_visibility_changed(&mut self, visible: bool) -> Task<EditorMessage> {
        self.page_visible = visible;
        if !visible {
            self.active_modal = None;
            return Task::none();
        }
        if self.selected_workspace_path.is_some() {
            self.refresh_file_tree()
        } else {
            Task::none()
        }
    }

    /// Handle a coalesced file-tree change ping. Dropped while the page is hidden or no workspace
    /// is selected — refresh-on-entry catches up; when visible, the expanded directories and the
    /// ignore matcher are re-read.
    ///
    /// Accepted consequences: a `.gitignore` edit normally pings the watcher, so dimming
    /// refreshes; a rare ignored-`.gitignore` setup goes stale until the next event or Cmd+R, and
    /// unwatched or watch-failed sessions refresh only on Cmd+R or a workspace switch.
    fn file_tree_pinged(&mut self) -> Task<EditorMessage> {
        if !self.page_visible || self.selected_workspace_path.is_none() {
            return Task::none();
        }
        self.refresh_file_tree()
    }

    /// Push a new tab, store its content and record the file mtime (if the file has one).
    fn insert_tab(&mut self, path: String, tab: Tab, tab_data: TabData, mtime: Option<SystemTime>) {
        self.tabs.push(tab);
        self.tab_contents.insert(path.clone(), tab_data);
        if let Some(mtime) = mtime {
            self.file_mtimes.insert(path, mtime);
        }
    }

    fn saved_tabs_loaded(
        &mut self,
        tabs_data: Result<Vec<SavedTabData>, String>,
        r#gen: u64,
    ) -> Task<EditorMessage> {
        if r#gen != self.saved_tabs_gen {
            return Task::none();
        }

        let tabs_data = match tabs_data {
            Ok(tabs_data) => {
                self.tabs_error = None;
                tabs_data
            }
            Err(e) => {
                // Leave `session_initialized` false and block writes: the persisted tabs are
                // still intact and must not be overwritten by the incomplete in-memory set. The
                // shown message carries that consequence and stays until a successful reload.
                tracing::warn!(error = %e, "Editor: failed to restore saved tabs");
                self.tabs_error = Some(format!("{e} — {TABS_SAVE_PAUSED}"));
                return Task::none();
            }
        };

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

    /// Directory expansion: populate `dir_entries`, rebuild the tree, advance focus if asked.
    fn dir_expanded(
        &mut self,
        dir_path: &str,
        r#gen: u64,
        entries: Result<Vec<FsEntry>, ReadDirError>,
        quiet: bool,
    ) -> Task<EditorMessage> {
        if self.dir_generations.get(dir_path) != Some(&r#gen) {
            return Task::none();
        }
        // Consume the generation slot once the result is accepted (as `rename_completed` does).
        self.dir_generations.remove(dir_path);

        self.loading_dirs.remove(dir_path);

        match entries {
            Ok(entries) => {
                self.dir_entries.insert(dir_path.to_string(), entries);
                self.rebuild_tree();
                if self.pending_enter_dir.as_deref() == Some(dir_path) {
                    self.pending_enter_dir = None;
                    return self
                        .file_tree
                        .expand_dir_and_focus_first_child::<EditorMessage>(dir_path);
                }
            }
            Err(ReadDirError::NotFound) => {
                // Gone (deleted externally, by a script or the GUI): forget it silently — no
                // toast, and at most one low-level log per path since the refresh state is pruned.
                self.forget_directory(dir_path);
                self.rebuild_tree();
                if !dir_path.is_empty() {
                    tracing::debug!(
                        "File tree: forgotten directory that no longer exists: {dir_path}"
                    );
                }
            }
            Err(ReadDirError::Other(err)) => {
                if quiet {
                    tracing::warn!("Failed to read directory '{dir_path}' (refresh): {err}");
                    return Task::none();
                }
                return Task::done(EditorMessage::Toast(super::ToastMessage::Warning(format!(
                    "Failed to read directory '{dir_path}': {err}"
                ))));
            }
        }
        Task::none()
    }

    /// A GUI directory deletion succeeded: prune the deleted path and its descendants from the tree
    /// state, then re-read the parent so the tree reflects it. Deferred to the success path so a
    /// failed delete leaves a still-existing directory in the tree.
    fn dir_deleted(&mut self, dir_path: &str, workspace_path: &str) -> Task<EditorMessage> {
        // Staleness guard: a completion from a workspace the user switched away from must not prune
        // the current workspace's state at the same relative path.
        if self.selected_workspace_path.as_deref() != Some(workspace_path) {
            return Task::none();
        }
        self.forget_directory(dir_path);
        self.rebuild_tree();

        let Some(ws_path) = self.selected_workspace_path.clone() else {
            return Task::none();
        };
        let parent_dir = Path::new(dir_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        let r#gen = self.bump_generation();
        self.dir_generations.insert(parent_dir.clone(), r#gen);
        dir_expanded_task(ws_path, parent_dir, r#gen, false)
    }

    /// A file was loaded from disk: open its tab and honour a pending global-search goto.
    fn file_loaded(
        &mut self,
        path: &str,
        r#gen: u64,
        result: Result<FileLoadData, String>,
    ) -> Task<EditorMessage> {
        if self.file_generations.get(path).copied() != Some(r#gen) {
            return Task::none();
        }
        // Consume the slot: bounds the map without removal code at every close/delete/rename path.
        self.file_generations.remove(path);

        match result {
            Ok(data) => {
                let saved_hash = hash_text(&data.text);
                let (tab, tab_data, mtime) =
                    make_tab_and_data(&data.path, &data.text, data.line_ending, false, saved_hash);
                self.insert_tab(data.path, tab, tab_data, mtime);
                self.active_tab_index = self.tabs.len().saturating_sub(1);
                self.session_initialized = true;

                // A global-search result click: jump to the matched line, consuming the pending
                // goto only when path and generation both match, so another load cannot steal it.
                if self
                    .pending_goto
                    .as_ref()
                    .is_some_and(|(gp, _, gg)| *gp == path && *gg == r#gen)
                {
                    if let Some((_, goto_line_1based, _)) = self.pending_goto.take() {
                        let cursor_line = goto_line_1based.saturating_sub(1);
                        let tab_path = self.tabs[self.active_tab_index].path.clone();
                        if let Some(tab_data) = self.tab_contents.get_mut(&tab_path) {
                            tab_data.content.move_to(cursor_line, 0);
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

    /// A save finished: update the dirty flag and run any pending close action.
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
                }
                if let Some(tab_data) = self.tab_contents.get_mut(path) {
                    tab_data.saved_text_hash = saved_hash;
                }
                // Store the new mtime so the auto-refresh tick does not mistake the save for an
                // external edit and re-read the file, destroying the undo stack.
                if let Ok(meta) = std::fs::metadata(path) {
                    if let Ok(mtime) = meta.modified() {
                        self.file_mtimes.insert(path.to_string(), mtime);
                    }
                }

                // If this save is part of a pending close action, handle it now.
                match self.pending_close.take() {
                    Some(PendingCloseAction::CloseTab(close_idx)) => {
                        if close_idx < self.tabs.len() {
                            self.remove_tab_at(close_idx);
                            let mut tasks: Vec<Task<EditorMessage>> = Vec::new();
                            if !self.tabs.is_empty() {
                                tasks.push(self.scroll_to_active_tab());
                            }
                            tasks.push(self.save_current_tabs());
                            return Task::batch(tasks);
                        }
                    }
                    Some(PendingCloseAction::CloseOthers {
                        keep_idx,
                        remaining_dirty: mut remaining,
                    }) => {
                        return if remaining.is_empty() {
                            self.remove_all_tabs_except(keep_idx);
                            self.try_save_current_tabs().map_or_else(Task::none, |t| {
                                Task::batch([
                                    t,
                                    Task::done(EditorMessage::Toast(super::ToastMessage::Saved)),
                                ])
                            })
                        } else {
                            let next = remaining.remove(0);
                            self.pending_close = Some(PendingCloseAction::CloseOthers {
                                keep_idx,
                                remaining_dirty: remaining,
                            });
                            build_save_task(&self.tabs, &self.tab_contents, next)
                        };
                    }
                    None => {}
                }

                if let Some(save_task) = self.try_save_current_tabs() {
                    save_task
                } else {
                    Task::done(EditorMessage::Toast(super::ToastMessage::Saved))
                }
            }
            Err(e) => {
                self.pending_close = None;
                let toast = super::ToastMessage::Error(e);
                Task::done(EditorMessage::Toast(toast))
            }
        }
    }

    /// Escape dismisses, in priority order: the active modal overlay (clearing the `CloseDialog`/
    /// `CloseOthers` state), the find/replace bar, tree focus, residual [`pending_close`].
    fn escape(&mut self) -> Task<EditorMessage> {
        // Close modal overlays first.
        if let Some(modal) = self.active_modal.take() {
            match modal {
                ModalKind::CloseDialog(..) | ModalKind::CloseOthers(..) => {
                    self.pending_close = None;
                }
                _ => {}
            }
            return Task::none();
        }

        if let Some((_, path)) = self.active_tab() {
            if let Some(tab_data) = self.tab_contents.get_mut(&path) {
                if tab_data.find_replace_state.is_some() {
                    tab_data.find_replace_state = None;
                    return Task::none();
                }
            }
        }

        if self.file_tree.tree_focused {
            self.file_tree.tree_focused = false;
            self.pending_enter_dir = None;
            return Task::none();
        }
        self.pending_close = None;
        Task::none()
    }

    /// Open or close the global search overlay, initialising the search engine.
    fn global_search_toggle(&mut self) -> Task<EditorMessage> {
        if matches!(self.active_modal, Some(ModalKind::GlobalSearch(_))) {
            self.active_modal = None;
            return Task::none();
        }
        if self.active_modal.is_some() {
            return Task::none();
        }
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
            query: SingleLineEditorState::new(""),
            results: Vec::new(),
            selected_index: 0,
            status: GlobalSearchStatus::Idle,
        }));

        let engine_task = Task::perform(
            async move {
                match crate::workspace_index::resolve_engine(&ws_name, &ws_path, "").await {
                    Ok(_) => EditorMessage::GlobalSearchResults {
                        r#gen: gs_gen,
                        results: Vec::new(),
                        error: None,
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

        let focus_task =
            iced::widget::operation::focus::<EditorMessage>(Id::new(GLOBAL_SEARCH_INPUT_ID));

        Task::batch([engine_task, focus_task])
    }

    /// Handle global search results: drop stale generations, record errors and empty results.
    fn global_search_results(
        &mut self,
        r#gen: u64,
        results: Vec<OwnedGrepMatch>,
        error: Option<String>,
    ) -> Task<EditorMessage> {
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

        if results.is_empty() && state.query.text().is_empty() {
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

    fn find_replace_all(&mut self) -> Task<EditorMessage> {
        let Some((idx, path)) = self.active_tab() else {
            return Task::none();
        };
        let mut toast = None;
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if let Some(ref state) = tab_data.find_replace_state {
                if !state.matches.is_empty() {
                    tab_data
                        .undo_stack
                        .borrow_mut()
                        .snap_before_edit(&tab_data.content);
                    let cursor_before = tab_data.content.cursor();
                    let text = tab_data.content.text();
                    let replace = state.replace.text();
                    // Replace all in reverse order to preserve positions.
                    let mut new_text = text;
                    for range in state.matches.iter().rev() {
                        new_text.replace_range(range.start..range.end, replace.as_str());
                    }
                    tab_data.content = EditorBuffer::from_file(&new_text, &path);
                    tab_data
                        .content
                        .move_to(cursor_before.line, cursor_before.column);
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
        // Recompute the dirty flag only when the buffer actually changed.
        if toast.is_some() {
            update_dirty_flag(&mut self.tabs, &self.tab_contents, idx, &path);
        }
        if let Some(t) = toast {
            Task::done(t)
        } else {
            Task::none()
        }
    }

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
        self.file_tree.focus_path(dir_path);
        read_task
    }

    fn select_file(&mut self, path: &str) -> Task<EditorMessage> {
        // Clicking a file tree row transfers keyboard focus to the tree
        // so that arrow keys navigate the tree instead of the editor.
        self.file_tree.tree_focused = true;
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

    fn save_active_tab(&self) -> Task<EditorMessage> {
        if self.active_modal.is_some() {
            return Task::none();
        }
        let Some(idx) = self.active_tab_idx() else {
            return Task::none();
        };
        build_save_task(&self.tabs, &self.tab_contents, idx)
    }

    fn close_dialog(&mut self, tab_index: usize, action: CloseAction) -> Task<EditorMessage> {
        match action {
            CloseAction::Save => {
                if tab_index < self.tabs.len() {
                    // Clear dialog immediately; close tab after save completes.
                    self.active_modal = None;
                    self.pending_close = Some(PendingCloseAction::CloseTab(tab_index));

                    build_save_task(&self.tabs, &self.tab_contents, tab_index)
                } else {
                    self.active_modal = None;
                    Task::none()
                }
            }
            CloseAction::Discard => {
                self.active_modal = None;
                self.pending_close = None;
                if tab_index < self.tabs.len() {
                    self.remove_tab_at(tab_index);
                }
                self.save_current_tabs()
            }
            CloseAction::Cancel => self.cancel_close_dialog(),
        }
    }

    fn cancel_close_dialog(&mut self) -> Task<EditorMessage> {
        self.active_modal = None;
        self.pending_close = None;
        Task::none()
    }

    fn close_all_tabs_except_and_save(&mut self, keep_idx: usize) -> Task<EditorMessage> {
        self.remove_all_tabs_except(keep_idx);
        self.save_current_tabs()
    }

    fn close_others_dialog(&mut self, keep_idx: usize, action: CloseAction) -> Task<EditorMessage> {
        match action {
            CloseAction::Save => {
                self.active_modal = None;
                let mut dirty: Vec<usize> = (0..self.tabs.len())
                    .filter(|&i| i != keep_idx && self.tabs[i].is_dirty)
                    .collect();
                if dirty.is_empty() {
                    return self.close_all_tabs_except_and_save(keep_idx);
                }
                let first = dirty.remove(0);
                self.pending_close = Some(PendingCloseAction::CloseOthers {
                    keep_idx,
                    remaining_dirty: dirty,
                });
                build_save_task(&self.tabs, &self.tab_contents, first)
            }
            CloseAction::Discard => {
                self.active_modal = None;
                self.pending_close = None;
                self.close_all_tabs_except_and_save(keep_idx)
            }
            CloseAction::Cancel => self.cancel_close_dialog(),
        }
    }

    fn close_other_tabs(&mut self, idx: usize) -> Task<EditorMessage> {
        if idx >= self.tabs.len() {
            return Task::none();
        }
        let dirty: Vec<usize> = (0..self.tabs.len())
            .filter(|&i| i != idx && self.tabs[i].is_dirty)
            .collect();
        if dirty.is_empty() {
            return self.close_all_tabs_except_and_save(idx);
        }
        self.active_modal = Some(ModalKind::CloseOthers(idx));
        Task::none()
    }

    /// Handle go-to-line toggle — opens/closes the go-to-line input bar.
    ///
    /// Only the page on screen may be armed: the chord reaches this listener
    /// from every page, and a bar armed behind another page would render the
    /// moment the editor is next shown.
    fn go_to_line_toggle(&mut self) -> Task<EditorMessage> {
        if !self.page_visible {
            return Task::none();
        }
        // Allow toggle-to-close when GotoLine is already open, but
        // block if any other modal is active.
        if let Some(modal) = &self.active_modal {
            if !matches!(modal, ModalKind::GotoLine) {
                return Task::none();
            }
        }
        if let Some((_, path)) = self.active_tab() {
            if matches!(self.active_modal, Some(ModalKind::GotoLine)) {
                self.active_modal = None;
                return Task::none();
            }
            // Close find bar when opening go-to-line.
            if let Some(tab_data) = self.tab_contents.get_mut(&path) {
                tab_data.find_replace_state = None;
            }
            self.active_modal = Some(ModalKind::GotoLine);
            self.goto_line_input.clear();
            return iced::widget::operation::focus::<EditorMessage>(Id::new(GOTO_LINE_INPUT_ID));
        }
        Task::none()
    }

    fn go_to_line_input(
        &mut self,
        action: super::editor_widget::EditorAction,
    ) -> Task<EditorMessage> {
        if matches!(self.active_modal, Some(ModalKind::GotoLine)) {
            self.goto_line_input.apply_action(action);
        }
        Task::none()
    }

    fn go_to_line_go(&mut self) -> Task<EditorMessage> {
        if !matches!(self.active_modal, Some(ModalKind::GotoLine)) {
            return Task::none();
        }
        let line_num: usize = match self.goto_line_input.text().parse::<usize>() {
            Ok(n) if n > 0 => n.saturating_sub(1), // convert 1-based to 0-based
            _ => return Task::none(),
        };
        let Some((_, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            tab_data.content.move_to(line_num, 0);
        }
        self.active_modal = None;
        Task::none()
    }

    fn global_search_input(
        &mut self,
        action: super::editor_widget::EditorAction,
    ) -> Task<EditorMessage> {
        let changes_text = action.changes_text();
        let Some(ModalKind::GlobalSearch(state)) = &mut self.active_modal else {
            return Task::none();
        };
        state.query.apply_action(action);
        let query = state.query.text();

        // Only text changes re-trigger the search/debounce pipeline; cursor
        // movement just repositions the caret.
        if !changes_text {
            return Task::none();
        }

        if query.is_empty() {
            state.status = GlobalSearchStatus::Idle;
            state.results.clear();
            state.selected_index = 0;
            // Increment generation to cancel any in-flight searches.
            self.global_search_gen = self.global_search_gen.wrapping_add(1);
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

        Task::perform(run_global_search(ws_path, ws_name, query, gs_gen), |msg| {
            msg
        })
    }

    fn global_search_select(&mut self, idx: usize) -> Task<EditorMessage> {
        let Some(ModalKind::GlobalSearch(state)) = &self.active_modal else {
            return Task::none();
        };
        let Some(match_result) = state.results.get(idx) else {
            return Task::none();
        };
        let abs_path = match_result.abs_path.clone();
        #[expect(clippy::cast_possible_truncation)]
        let line_number = match_result.line_number as usize;

        self.active_modal = None;

        // Convert from 1-based (grep) to 0-based (editor).
        let cursor_line = line_number.saturating_sub(1);

        if let Some(existing_idx) = self.tabs.iter().position(|t| t.path == abs_path) {
            self.active_tab_index = existing_idx;
            if let Some(tab_data) = self.tab_contents.get_mut(&abs_path) {
                tab_data.content.move_to(cursor_line, 0);
            }
            return self.scroll_to_active_tab();
        }

        // File not open — load it, then jump after loading.
        // Set pending_goto so FileLoaded handler moves the cursor.
        // Use self.generation.wrapping_add(1) to predict the generation
        // that open_file_in_editor will produce (it calls bump_generation).
        let file_gen = self.generation.wrapping_add(1);
        self.pending_goto = Some((abs_path.clone(), line_number, file_gen));
        self.open_file_in_editor(&abs_path)
    }

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

        let mut dirty_count = 0;
        for tab in &self.tabs {
            if is_within_abs(&tab.path, &abs_path) {
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

    fn confirm_delete(&mut self) -> Task<EditorMessage> {
        let Some(ModalKind::DeleteConfirm(target)) = self.active_modal.take() else {
            return Task::none();
        };
        if target.is_dir {
            self.perform_dir_delete(&target)
        } else {
            self.perform_file_delete(&target)
        }
    }

    fn new_item_submit(&mut self, name: &str) -> Task<EditorMessage> {
        let Some(ModalKind::NewItem(target)) = self.active_modal.take() else {
            return Task::none();
        };
        let trimmed = name.trim();
        if let Some(msg) = validate_item_name(trimmed) {
            return Task::done(EditorMessage::Toast(super::ToastMessage::Warning(
                msg.into(),
            )));
        }
        self.perform_create_item(&target, trimmed)
    }

    fn new_item_input(
        &mut self,
        action: super::editor_widget::EditorAction,
    ) -> Task<EditorMessage> {
        if let Some(ModalKind::NewItem(ref mut target)) = self.active_modal {
            target.input_text.apply_action(action);
        }
        Task::none()
    }

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
        let is_dir = Path::new(&abs_path).is_dir();

        self.active_modal = Some(ModalKind::Rename(RenameTarget {
            abs_path,
            ws_root: ws.clone(),
            path: path.to_string(),
            is_dir,
            input_text: SingleLineEditorState::new(&file_name),
            error: None,
        }));
        iced::widget::operation::focus::<EditorMessage>(Id::from(format!("rename_input_{path}")))
    }

    fn rename_input(&mut self, action: super::editor_widget::EditorAction) -> Task<EditorMessage> {
        if let Some(ModalKind::Rename(ref mut target)) = self.active_modal {
            target.input_text.apply_action(action);
            // Clear error when user starts typing again.
            if target.error.is_some() {
                target.error = None;
            }
        }
        Task::none()
    }

    fn rename_submit(&mut self) -> Task<EditorMessage> {
        let Some(ModalKind::Rename(target)) = self.active_modal.take() else {
            return Task::none();
        };
        // All-space names fall through to the empty-name check below.
        let trimmed = target.input_text.text().trim().to_string();

        let error_msg = validate_item_name(&trimmed);
        if let Some(msg) = error_msg {
            let mut rt = target;
            rt.error = Some(msg.into());
            self.active_modal = Some(ModalKind::Rename(rt));
            return Task::none();
        }

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

        if Path::new(&new_abs_path).exists() {
            let mut rt = target;
            rt.error = Some("A file or directory with that name already exists".into());
            self.active_modal = Some(ModalKind::Rename(rt));
            return Task::none();
        }

        let old_abs = target.abs_path;
        let old_rel = target.path;
        let is_dir = target.is_dir;
        let parent_dir_clone = parent_dir;
        let ws_root = target.ws_root;
        // Follow the same generation-based invalidation protocol as
        // every other async directory operation (ToggleDir, TreeNavEnter,
        // perform_create_item, etc.): bump self.generation and register
        // it in dir_generations so that any in-flight DirExpanded for
        // this directory is invalidated (its generation won't match).
        let dir_gen = self.bump_generation();
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
                        trimmed,
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

    fn rename_cancel(&mut self) -> Task<EditorMessage> {
        self.active_modal = None;
        Task::none()
    }

    #[expect(clippy::too_many_lines)]
    fn rename_completed(
        &mut self,
        old_path: &str,
        new_path: &str,
        is_dir: bool,
        result: Result<(), String>,
        dir_entries: Result<Vec<FsEntry>, ReadDirError>,
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
                if self.selected_file.as_deref() == Some(old_path) {
                    self.selected_file = Some(new_path.to_string());
                }

                let Some(old_abs) = self.abs_path(old_path) else {
                    return Task::none();
                };
                let Some(new_abs) = self.abs_path(new_path) else {
                    return Task::none();
                };

                // Rebase absolute paths under the renamed directory.
                if is_dir {
                    for tab in &mut self.tabs {
                        if let Some(new_tab_path) = rebase_abs_path(&tab.path, &old_abs, &new_abs) {
                            tab.path = new_tab_path;
                            tab.file_name = Path::new(&tab.path)
                                .file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default();
                        }
                    }
                    // Re-key tab_contents for affected files.
                    rekey_abs_map(&mut self.tab_contents, &old_abs, &new_abs);

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

                // Re-key from the old absolute path so auto-refresh does not stat it and in-flight
                // FileLoaded results are still validated.
                if is_dir {
                    rekey_abs_map(&mut self.file_mtimes, &old_abs, &new_abs);
                    rekey_abs_set(&mut self.deleted_file_toasted, &old_abs, &new_abs);
                    rekey_abs_map(&mut self.file_generations, &old_abs, &new_abs);
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

                match dir_entries {
                    Ok(entries) => {
                        // re_path was computed at the top of the
                        // handler for the staleness check; we own
                        // the generation slot, so insert unconditionally.
                        self.dir_entries.insert(re_path, entries);
                        self.rebuild_tree();
                        self.file_tree.focus_path(new_path);
                    }
                    Err(ReadDirError::NotFound) => {
                        // Parent directory vanished mid-rename — forget it
                        // silently like any other missing directory.
                        self.forget_directory(&re_path);
                        self.rebuild_tree();
                    }
                    Err(ReadDirError::Other(err)) => {
                        self.rebuild_tree();
                        return Task::done(EditorMessage::Toast(super::ToastMessage::Error(
                            format!(
                                "Rename succeeded but failed to refresh tree '{re_path}': {err}"
                            ),
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
                    Err(ReadDirError::NotFound) => {
                        // Parent directory vanished — forget it silently.
                        self.forget_directory(&re_path);
                        self.rebuild_tree();
                    }
                    Err(ReadDirError::Other(_)) => {
                        self.rebuild_tree();
                    }
                }
                Task::done(EditorMessage::Toast(super::ToastMessage::Error(e)))
            }
        }
    }

    fn quick_open_toggle(&mut self) -> Task<EditorMessage> {
        if matches!(self.active_modal, Some(ModalKind::QuickOpen(_))) {
            self.active_modal = None;
            return Task::none();
        }
        if self.active_modal.is_some() {
            return Task::none();
        }

        self.scan_all_workspace_files();

        self.active_modal = Some(ModalKind::QuickOpen(QuickOpenState {
            filter: SingleLineEditorState::new(""),
            selected_index: 0,
            results: Vec::new(),
        }));
        iced::widget::operation::focus::<EditorMessage>(Id::new(QUICK_OPEN_INPUT_ID))
    }

    fn quick_open_input(
        &mut self,
        action: super::editor_widget::EditorAction,
    ) -> Task<EditorMessage> {
        let changes_text = action.changes_text();
        let filter = if let Some(ModalKind::QuickOpen(ref mut qo)) = self.active_modal {
            qo.filter.apply_action(action);
            qo.filter.text()
        } else {
            return Task::none();
        };
        // The shared widget emits cursor movements too; only re-filter on
        // text changes (edits and undo/redo).
        if changes_text {
            let results = self.filter_workspace_files(&filter);
            if let Some(ModalKind::QuickOpen(ref mut qo)) = self.active_modal {
                qo.results = results;
                qo.selected_index = 0;
            }
        }
        Task::none()
    }

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

    fn close_active_tab(&mut self) -> Task<EditorMessage> {
        if self.active_modal.is_some() {
            return Task::none();
        }
        let idx = self.active_tab_index;
        if idx < self.tabs.len() {
            return self.close_tab_at(idx);
        }
        Task::none()
    }

    fn tree_focus_toggled(&mut self) -> Task<EditorMessage> {
        // Suppress during any modal overlay (QuickOpen, GlobalSearch,
        // GotoLine, Rename, etc.) — the overlay owns keyboard focus
        // and the single-field `active_modal` covers all variants.
        if self.active_modal.is_some() {
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

    fn tree_scrolled(&mut self, scroll_y: f32, viewport_h: f32) -> Task<EditorMessage> {
        self.file_tree.scroll_y = scroll_y;
        self.file_tree.viewport_h = Some(viewport_h);
        Task::none()
    }

    fn tab_bar_scrolled(&mut self, scroll_x: f32, viewport_w: f32) -> Task<EditorMessage> {
        self.tab_scroll_x = scroll_x;
        self.tab_viewport_w = Some(viewport_w);
        Task::none()
    }

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
        if self.active_modal.is_some() {
            return Task::none();
        }
        let Some((_idx, path, is_dir)) = self.file_tree.focused_tree_node() else {
            return Task::none();
        };
        if self.file_tree.focused_is_expanded_dir() {
            return self.collapse_dir(&path);
        }
        if is_dir {
            return self.expand_dir_and_focus(&path, "TreeNavEnter");
        }
        Task::done(EditorMessage::SelectFile(path))
    }

    fn tree_nav_left(&mut self) -> Task<EditorMessage> {
        // Suppress during active modal overlays — the overlay handles
        // its own keyboard navigation (covers Rename, GotoLine, etc.).
        if self.active_modal.is_some() {
            return Task::none();
        }
        let Some((_idx, path, _)) = self.file_tree.focused_tree_node() else {
            return Task::none();
        };

        if self.file_tree.focused_is_expanded_dir() {
            return self.collapse_dir(&path);
        }

        self.file_tree.focus_parent::<EditorMessage>()
    }

    fn tree_nav_right(&mut self) -> Task<EditorMessage> {
        // Suppress during active modal overlays — the overlay handles
        // its own keyboard navigation (covers Rename, GotoLine, etc.).
        if self.active_modal.is_some() {
            return Task::none();
        }
        let Some((idx, path, is_dir)) = self.file_tree.focused_tree_node() else {
            return Task::none();
        };

        if !is_dir {
            return Task::none();
        }

        if !self.file_tree.expanded_dirs.contains(&path) {
            return self.expand_dir_and_focus(&path, "TreeNavRight");
        }

        self.file_tree.focus_next_row::<EditorMessage>(idx)
    }

    fn find_toggle(&mut self) -> Task<EditorMessage> {
        if self.active_modal.is_some() {
            return Task::none();
        }
        let Some((_, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
            if tab_data.find_replace_state.is_none() {
                let default_query = tab_data.content.selection().unwrap_or_default();
                let mut state = FindReplaceState {
                    query: SingleLineEditorState::new(&default_query),
                    replace: SingleLineEditorState::new(""),
                    matches: Vec::new(),
                    current_match_idx: 0,
                    case_sensitive: false,
                };
                state.recompute(&tab_data.content);
                tab_data.find_replace_state = Some(state);
            }
        }
        iced::widget::operation::focus::<EditorMessage>(Id::new(FIND_SEARCH_ID))
    }

    /// Run a mutation against the active tab's find state, if any.
    fn with_active_find_state(
        &mut self,
        f: impl FnOnce(&mut FindReplaceState, &EditorBuffer),
    ) -> Task<EditorMessage> {
        let Some((_, path)) = self.active_tab() else {
            return Task::none();
        };
        if let Some(tab_data) = self.tab_contents.get_mut(&path)
            && let Some(state) = tab_data.find_replace_state.as_mut()
        {
            f(state, &tab_data.content);
        }
        Task::none()
    }

    fn find_query_input(
        &mut self,
        action: super::editor_widget::EditorAction,
    ) -> Task<EditorMessage> {
        let changes_text = action.changes_text();
        self.with_active_find_state(|state, content| {
            state.query.apply_action(action);
            // The shared widget emits cursor movements too; only recompute
            // matches on text changes (edits and undo/redo).
            if changes_text {
                state.recompute(content);
            }
        })
    }

    fn find_replace_input(
        &mut self,
        action: super::editor_widget::EditorAction,
    ) -> Task<EditorMessage> {
        self.with_active_find_state(|state, _| {
            state.replace.apply_action(action);
        })
    }

    fn find_replace(&mut self) -> Task<EditorMessage> {
        let Some((idx, path)) = self.active_tab() else {
            return Task::none();
        };

        // Clone the range/replace/query up front so immutable reads and the later mutable borrow do
        // not interleave in one scope.
        let Some((range, replace_text, case_sensitive, query)) = self
            .tab_contents
            .get(&path)
            .and_then(|tab_data| tab_data.find_replace_state.as_ref())
            .and_then(|state| {
                let range = state.matches.get(state.current_match_idx)?;
                Some((
                    range.clone(),
                    state.replace.text(),
                    state.case_sensitive,
                    state.query.text(),
                ))
            })
        else {
            return Task::none();
        };

        let replace_end = range.start + replace_text.len();

        // Guard: no-op replacement (empty text on zero-width range).
        if replace_text.is_empty() && range.start >= range.end {
            return Task::none();
        }

        if let Some(tab_data) = self.tab_contents.get_mut(&path) {
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
                state.matches = compute_text_matches(&new_text, &query, case_sensitive);

                if !state.matches.is_empty() {
                    // Advance to the next match starting at or after the
                    // end of the replacement in the new text
                    // (position = range.start + len(replace_text)).
                    // Using a position in the old text (range.end) would
                    // be wrong when replacement length differs from the
                    // original match length.
                    let next_idx = state
                        .matches
                        .iter()
                        .position(|m| m.start >= replace_end)
                        .unwrap_or(0)
                        .min(state.matches.len() - 1);
                    state.current_match_idx = next_idx;
                    // Position cursor at the new match.
                    if let Some(r) = state.matches.get(next_idx) {
                        let (line, col) = byte_offset_to_line_col(&new_text, r.start);
                        tab_data.content.move_to(line, col);
                    }
                } else {
                    state.current_match_idx = 0;
                    // No remaining matches — place cursor at end of
                    // the replacement, not at buffer start.
                    let (line, col) = byte_offset_to_line_col(&new_text, replace_end);
                    tab_data.content.move_to(line, col);
                }
            }
        }

        update_dirty_flag(&mut self.tabs, &self.tab_contents, idx, &path);
        Task::none()
    }

    fn find_toggle_case_sensitivity(&mut self) -> Task<EditorMessage> {
        self.with_active_find_state(|state, content| {
            state.case_sensitive = !state.case_sensitive;
            state.recompute(content);
        })
    }

    fn refresh_file_tree(&mut self) -> Task<EditorMessage> {
        // Suppress during active modal overlays — the file tree should
        // not refresh behind an active overlay.  This covers both the
        // Cmd+R / Ctrl+R keyboard shortcut.
        if self.active_modal.is_some() {
            return Task::none();
        }
        let Some(ref ws_path) = self.selected_workspace_path else {
            return Task::none();
        };

        let mut dirs_to_refresh: Vec<String> = Vec::new();

        // Root directory (empty string) is always included — it's
        // implicitly expanded and not tracked in expanded_dirs.
        dirs_to_refresh.push(String::new());

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
            let dir_gen = self.bump_generation();
            self.dir_generations.insert(dir_path.clone(), dir_gen);
            // NOTE: deliberately NOT adding to `loading_dirs` — this
            // avoids a "Loading…" flicker for every expanded directory
            // on every background refresh. The tree silently updates
            // when results arrive via DirExpanded.

            tasks.push(dir_expanded_task(
                root_path.clone(),
                dir_path.clone(),
                dir_gen,
                true,
            ));
        }

        // Rebuild the gitignore matcher so dimming reflects any .gitignore
        // changes picked up on disk.
        tasks.push(Self::spawn_ignore_matcher(&root_path));

        Task::batch(tasks)
    }

    /// Handle ignore-matcher-ready — store the rebuilt matcher (or `None` for
    /// no dimming) and re-stamp the current tree so dimming follows a late
    /// matcher arrival. Discards a matcher built for a workspace that is no
    /// longer selected (tagged by path).
    fn ignore_matcher_ready(
        &mut self,
        workspace: &str,
        matcher: Option<ignore::IncrementalIgnore>,
    ) -> Task<EditorMessage> {
        if self.selected_workspace_path.as_deref() != Some(workspace) {
            return Task::none();
        }
        self.ignore_matcher = matcher;
        stamp_ignored_nodes(&mut self.file_tree.nodes, &mut self.ignore_matcher);
        Task::none()
    }

    /// Build a gitignore matcher for `workspace_path` and return a task that
    /// delivers it as `IgnoreMatcherReady` tagged with the workspace path (so
    /// a matcher built for a previously-selected workspace is discarded on arrival).
    fn spawn_ignore_matcher(workspace_path: &str) -> Task<EditorMessage> {
        let path = workspace_path.to_string();
        let tag = path.clone();
        Task::perform(
            async move { build_ignore_matcher(Path::new(&path)) },
            move |matcher| EditorMessage::IgnoreMatcherReady {
                workspace: tag,
                matcher,
            },
        )
    }

    /// Handle git-status-updated — per-file statuses forwarded from the Git
    /// state. Stores them and clears the in-flight guard. A forward whose
    /// workspace no longer matches the selected one is dropped (on a workspace
    /// switch the editor clears its cache via its own WorkspaceSelected path).
    fn git_status_updated(
        &mut self,
        workspace: Option<&str>,
        statuses: HashMap<String, GitFileStatus>,
    ) -> Task<EditorMessage> {
        // The forwarded workspace name matches the selected one exactly for a
        // resolved workspace — compare directly and drop forwards from a
        // superseded workspace.
        if workspace != self.selected_workspace_name.as_deref() {
            return Task::none();
        }
        self.git_status_cache = statuses;
        self.git_status_loading = false;
        // Re-latch the detector now that a refresh resolved. A git-status run
        // rewrites `.git/index`, so without this the detector would see an
        // index mtime change and re-trigger forever.
        self.relatch_git_dirty_mtimes();
        Task::none()
    }

    /// Probe `.git/index` and `.git/logs/HEAD` for mtime changes. If the latch
    /// is empty, record the fresh values (no trigger — the repo may be freshly
    /// cloned/initialized and we have no baseline). If a file's mtime changed
    /// since the last latch, staging grew or moved on disk without a watcher
    /// event — request a git-status refresh (routed to GitState by the
    /// Dashboard). Does NOT re-latch; the `GitStatusUpdated` handler re-latches
    /// after the refresh resolves so the git-status own index write is absorbed.
    ///
    /// Skips entirely while a git-status refresh is already in flight (the
    /// in-flight guard is shared with the workspace-selected eager refresh).
    ///
    /// For a `.git` that is a file (worktree/submodule), `index`/`logs/HEAD`
    /// under it don't exist, so the probe reads `(None, None)` and never
    /// triggers — an accepted edge (those repos need the watcher or Cmd+R).
    fn check_git_dirty(&mut self) -> Task<EditorMessage> {
        if self.git_status_loading {
            return Task::none();
        }
        let Some(git_dir) = self.git_dir.as_ref() else {
            self.git_dirty_mtimes = None;
            return Task::none();
        };
        let now = stat_git_dirty_mtimes(git_dir);
        match self.git_dirty_mtimes {
            None => {
                self.git_dirty_mtimes = Some(now);
                Task::none()
            }
            Some(prev) if prev != now => {
                self.git_status_loading = true;
                Task::done(EditorMessage::RequestGitStatusRefresh)
            }
            Some(_) => Task::none(),
        }
    }

    /// Re-stat the git dir and re-latch the mtimes. Called after every resolved
    /// git-status update so the baseline tracks the post-refresh disk state.
    fn relatch_git_dirty_mtimes(&mut self) {
        self.git_dirty_mtimes = self.git_dir.as_ref().map(|g| stat_git_dirty_mtimes(g));
    }

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
                let line_ending = detect_line_ending(&text);

                if let Some(tab) = self.tabs.get_mut(idx) {
                    tab.is_dirty = false;
                    tab.line_ending = line_ending;
                }

                if let Some(tab_data) = self.tab_contents.get_mut(&path) {
                    // Clear find/replace state — match byte ranges are now stale.
                    tab_data.find_replace_state = None;
                    tab_data.content = EditorBuffer::from_file(&text, &path);
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
    fn navigate_find_match(&mut self, direction: FindDirection) -> Task<EditorMessage> {
        self.with_active_find_state(|state, content| {
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
                    let (line, col) = byte_offset_to_line_col(&content.text(), range.start);
                    content.move_to(line, col);
                }
            }
        })
    }

    fn navigate_search_results(
        selected_index: &mut usize,
        results_len: usize,
        direction: TreeNavDirection,
    ) {
        match direction {
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
    fn navigate_tree_vertical(&mut self, direction: TreeNavDirection) -> Task<EditorMessage> {
        if let Some(ModalKind::GlobalSearch(ref mut gs)) = self.active_modal {
            Self::navigate_search_results(&mut gs.selected_index, gs.results.len(), direction);
            return Task::none();
        }
        if let Some(ModalKind::QuickOpen(ref mut qo)) = self.active_modal {
            Self::navigate_search_results(&mut qo.selected_index, qo.results.len(), direction);
            return Task::none();
        }
        // When another modal overlay (GotoLine, NewItem, DeleteConfirm,
        // CloseDialog, etc.) is active, suppress tree navigation.  The search
        // overlay redirects above have already returned, so only non-search
        // overlays reach this guard.
        if self.active_modal.is_some() {
            return Task::none();
        }
        self.file_tree.nav_and_scroll::<EditorMessage>(direction)
    }

    /// Render the editor. The dashboard renders this page only for a resolved
    /// workspace, so this is only ever reached for one.
    #[must_use]
    pub fn view(&self, dashboard_modal_open: bool) -> Element<'_, EditorMessage> {
        let tree_panel = self.build_tree_panel();
        let editor_panel = self.build_editor_panel(dashboard_modal_open);

        let split = row![tree_panel, editor_panel]
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fill);

        let body = column([split.into()])
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fill);

        // The Stack widget type must stay stable: a Column→Stack change between frames destroys
        // widget state (scroll positions, ContextMenu overlays) and leaves stale overlay links.
        let placeholder: Element<'_, EditorMessage> = widgets::empty_stack_placeholder();

        let overlay: Element<'_, EditorMessage> = match &self.active_modal {
            Some(ModalKind::CloseDialog(tab_idx)) => editor_dialog::build_close_dialog(
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
                editor_dialog::build_close_dialog(
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
            Some(ModalKind::GlobalSearch(gs)) => editor_dialog::build_global_search_overlay(gs),
            Some(ModalKind::QuickOpen(qo)) => editor_dialog::build_quick_open_overlay(qo),
            Some(ModalKind::NewItem(target)) => editor_dialog::build_new_item_input(target),
            Some(ModalKind::DeleteConfirm(target)) => {
                editor_dialog::build_delete_confirm_dialog(target)
            }
            // GotoLine and Rename are rendered inline (not as stack overlays).
            Some(ModalKind::GotoLine | ModalKind::Rename(_)) | None => placeholder,
        };

        iced::widget::stack([body.into(), overlay]).into()
    }

    fn build_tree_panel(&self) -> Element<'_, EditorMessage> {
        let elements: Vec<Element<'_, EditorMessage>> = self
            .file_tree
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| self.render_tree_node(n, 0, 0, i == self.file_tree.nodes.len() - 1))
            .collect();
        // Natural width of every rendered row, in render order, for the
        // auto-sizing tree panel. Replicates the exact row composition:
        // guide (depth*2 box-drawing chars at 14px) + lucide icon
        // (TREE_ICON_SIZE for dirs and files) + 4px gap + name label at 14px,
        // with the loading/error suffixes and `[⚠]` error marker. The closure
        // always measures node.name so the panel width stays stable while the
        // rename text-input replaces the rendered label.
        let row_widths = widgets::collect_tree_row_widths(
            &self.file_tree.nodes,
            &self.file_tree.expanded_dirs,
            |node, depth| {
                let guide_chars = depth * 2;
                if node.is_dir {
                    let label: std::borrow::Cow<'_, str> =
                        if self.loading_dirs.contains(&node.full_path) {
                            std::borrow::Cow::Owned(format!("{}  Loading…", node.name))
                        } else if let Some(ref err) = node.error {
                            std::borrow::Cow::Owned(format!("{} [⚠ {err}]", node.name))
                        } else {
                            std::borrow::Cow::Borrowed(&node.name)
                        };
                    widgets::tree_row_natural_width(
                        guide_chars,
                        widgets::TREE_ICON_SIZE,
                        &label,
                        widgets::TREE_FONT_SIZE,
                        None,
                        None,
                    )
                } else {
                    let suffix = node.error.as_ref().map(|_| ("[⚠]", 11.0));
                    widgets::tree_row_natural_width(
                        guide_chars,
                        widgets::TREE_ICON_SIZE,
                        &node.name,
                        widgets::TREE_FONT_SIZE,
                        suffix,
                        None,
                    )
                }
            },
        );
        let panel = widgets::build_tree_panel(
            &self.file_tree,
            elements,
            &row_widths,
            // The tree meets the top of the content area; keep its first row
            // off the window edge.
            theme::PAD_6,
            |viewport| {
                EditorMessage::TreeScrolled(viewport.absolute_offset().y, viewport.bounds().height)
            },
        );

        // A context menu for empty space below the tree nodes: the node-level menus capture
        // their own right-clicks, so only clicks past the last row reach this fallback.
        ContextMenu::new(
            panel,
            vec![
                MenuItem::new(
                    "New File".into(),
                    EditorMessage::NewFileRequested(String::new()),
                ),
                MenuItem::new(
                    "New Directory".into(),
                    EditorMessage::NewDirectoryRequested(String::new()),
                ),
            ],
        )
        .into()
    }

    /// The most "interesting" child git status (Modified > Added); only meaningful for an
    /// expanded directory, which is the only kind with populated children.
    fn dir_git_status(&self, node: &widgets::TreeNode) -> Option<GitFileStatus> {
        let mut best: Option<GitFileStatus> = None;
        for child in &node.children {
            if child.is_dir {
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

    /// Shared tree-node row: guide lines, icon and name wrapped in a `tree_node_button` inside a
    /// `ContextMenu` whose caller items (`extra_context_items`) precede the shared Copy Relative
    /// Path / Copy Absolute Path / Reveal in Finder. `full_path` (workspace-relative) builds the
    /// paths those shared items copy.
    #[expect(clippy::too_many_arguments)]
    fn render_tree_node_row<'a>(
        &'a self,
        guide: String,
        icon: Element<'a, EditorMessage>,
        name: Element<'a, EditorMessage>,
        highlight: bool,
        message: EditorMessage,
        extra_context_items: Vec<MenuItem<EditorMessage>>,
        full_path: &str,
    ) -> Element<'a, EditorMessage> {
        let row = widgets::tree_row(guide, icon, name, None);

        let btn = widgets::tree_node_button(row, highlight, Some(message));

        let rel_path = full_path.to_string();

        let mut menu_items: Vec<MenuItem<EditorMessage>> = extra_context_items;
        menu_items.push(MenuItem::new(
            "Copy Relative Path".into(),
            EditorMessage::CopyRelativePath(rel_path),
        ));

        if let Some(abs_path) = self.abs_path(full_path) {
            menu_items.push(MenuItem::new(
                "Copy Absolute Path".into(),
                EditorMessage::CopyAbsolutePath(abs_path.clone()),
            ));
            menu_items.push(MenuItem::new(
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
        widgets::render_tree_node(
            node.is_dir,
            || self.render_dir_node(node, depth, ancestor_mask, is_last),
            || self.render_file_node(node, depth, ancestor_mask, is_last),
        )
    }

    fn render_dir_node<'a>(
        &'a self,
        node: &'a widgets::TreeNode,
        depth: usize,
        ancestor_mask: u64,
        is_last: bool,
    ) -> Element<'a, EditorMessage> {
        let is_expanded = self.file_tree.expanded_dirs.contains(&node.full_path);
        let is_loading = self.loading_dirs.contains(&node.full_path);
        let is_ignored = node.ignored;
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
            theme::TEXT_FAINT
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
            (node.name.clone(), theme::TEXT_FAINT)
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
                MenuItem::new(
                    "New File".into(),
                    EditorMessage::NewFileRequested(node.full_path.clone()),
                ),
                MenuItem::new(
                    "New Directory".into(),
                    EditorMessage::NewDirectoryRequested(node.full_path.clone()),
                ),
                MenuItem::new(
                    "Rename".into(),
                    EditorMessage::RenameRequested(node.full_path.clone()),
                ),
                MenuItem::new(
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
        let is_ignored = node.ignored;
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
            theme::TEXT_FAINT
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
                        Space::new().width(theme::SPACE_4),
                        text("[⚠]").size(theme::TEXT_11).color(theme::STATUS_ERROR),
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
            icon.size(widgets::TREE_ICON_SIZE).color(icon_color).into();

        self.render_tree_node_row(
            guide,
            icon_element,
            name_text,
            is_selected || is_focused,
            EditorMessage::SelectFile(node.full_path.clone()),
            vec![
                MenuItem::new(
                    "Rename".into(),
                    EditorMessage::RenameRequested(node.full_path.clone()),
                ),
                MenuItem::new(
                    "Delete".into(),
                    EditorMessage::DeleteFileRequested(node.full_path.clone()),
                ),
            ],
            &node.full_path,
        )
    }

    fn build_editor_panel(&self, dashboard_modal_open: bool) -> Element<'_, EditorMessage> {
        if self.tabs.is_empty() {
            // A failed restore stays visible where the legitimately empty state would render.
            if let Some(err) = &self.tabs_error {
                return widgets::empty_state_placeholder(
                    lucide::triangle_alert(),
                    err,
                    theme::STATUS_ERROR,
                );
            }
            return widgets::empty_state_placeholder(
                lucide::file_text(),
                "Select a file to edit",
                theme::TEXT_MUTED,
            );
        }

        let tab_bar = self.build_tab_bar();
        let find_bar = self.build_find_replace_bar();
        let go_to_line = self.build_go_to_line_bar();
        let editor_widget = self.build_editor_widget(dashboard_modal_open);

        let mut col = column![tab_bar].spacing(0).width(Length::Fill);
        if let Some(bar) = find_bar {
            col = col.push(bar);
        } else if let Some(bar) = go_to_line {
            col = col.push(bar);
        }
        col = col.push(editor_widget);

        match &self.tabs_error {
            // The restore failure — and the paused tab persistence that goes with it — stays
            // visible until a successful reload replaces it, even once tabs exist.
            Some(err) => column![widgets::error_banner(err), col]
                .spacing(0)
                .width(Length::Fill)
                .height(Length::Fill)
                .into(),
            None => col.height(Length::Fill).into(),
        }
    }

    fn build_tab_bar(&self) -> Element<'_, EditorMessage> {
        let mut tab_buttons: Vec<Element<'_, EditorMessage>> = Vec::new();

        for (i, tab) in self.tabs.iter().enumerate() {
            let is_active = i == self.active_tab_index;

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

            let name_color = theme::tab_label_color(is_active);
            let name_text = text(&tab.file_name).size(theme::TEXT_12).color(name_color);

            let mut tab_row = row![].spacing(theme::SPACE_2).align_y(Alignment::Center);
            if let Some(dot) = dirty_dot {
                tab_row = tab_row.push(dot);
            }
            tab_row = tab_row.push(name_text).push(widgets::tab_close_button(
                is_active,
                EditorMessage::TabClosed(i),
            ));

            let tab_btn = button(tab_row.padding(theme::PAD_8))
                .on_press(EditorMessage::TabSelected(i))
                .style(theme::tab_button_style(is_active))
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
                    MenuItem::new("Close".into(), EditorMessage::TabClosed(i)),
                    MenuItem::new("Close Others".into(), EditorMessage::CloseOtherTabs(i)),
                    MenuItem::new(
                        "Copy Relative Path".into(),
                        EditorMessage::CopyRelativePath(tab_rel_path),
                    ),
                    MenuItem::new(
                        "Copy Absolute Path".into(),
                        EditorMessage::CopyAbsolutePath(tab_abs_path),
                    ),
                ],
            );

            tab_buttons.push(ctx_menu.into());
        }

        widgets::tab_scrollable(
            tab_buttons,
            Some(self.tab_scroll_id.clone()),
            Some(|viewport: scrollable::Viewport| {
                EditorMessage::TabBarScrolled(viewport.absolute_offset().x, viewport.bounds().width)
            }),
        )
    }

    fn build_find_replace_bar(&self) -> Option<Element<'_, EditorMessage>> {
        let idx = self.active_tab_idx()?;
        let path = &self.tabs[idx].path;
        let state = self.tab_contents.get(path)?.find_replace_state.as_ref()?;

        let search_input = widgets::single_line_editor(
            &state.query.buffer,
            "Find…",
            true,
            Length::Fixed(200.0),
            Some(Id::new(FIND_SEARCH_ID)),
            |action| match action {
                super::editor_widget::EditorAction::Submit => EditorMessage::FindNext,
                other => EditorMessage::FindQueryInput(other),
            },
        );

        let replace_input = widgets::single_line_editor(
            &state.replace.buffer,
            "Replace…",
            true,
            Length::Fixed(160.0),
            Some(Id::new(FIND_REPLACE_ID)),
            |action| match action {
                super::editor_widget::EditorAction::Submit => EditorMessage::FindNext,
                other => EditorMessage::FindReplaceInput(other),
            },
        );

        let total = state.matches.len();
        let query = state.query.text();
        let match_label = if !query.is_empty() && query.len() < 2 {
            "Min 2 chars".to_string()
        } else if !query.is_empty() && total > 0 {
            format!("{}/{}", state.current_match_idx.saturating_add(1), total)
        } else if !query.is_empty() {
            "0/0".to_string()
        } else {
            String::new()
        };

        let prev_btn = bar_text_button(
            "‹",
            EditorMessage::FindPrev,
            theme::TEXT_14,
            theme::TEXT_SECONDARY,
            [theme::PAD_2, theme::PAD_8],
        );

        let next_btn = bar_text_button(
            "›",
            EditorMessage::FindNext,
            theme::TEXT_14,
            theme::TEXT_SECONDARY,
            [theme::PAD_2, theme::PAD_8],
        );

        let replace_btn = bar_text_button(
            "Replace",
            EditorMessage::FindReplace,
            theme::TEXT_11,
            theme::TEXT_SECONDARY,
            [theme::PAD_2, theme::PAD_6],
        );

        let replace_all_btn = bar_text_button(
            "All",
            EditorMessage::FindReplaceAll,
            theme::TEXT_11,
            theme::TEXT_SECONDARY,
            [theme::PAD_2, theme::PAD_6],
        );

        let case_label_color = if state.case_sensitive {
            theme::ACCENT_LIGHT
        } else {
            theme::TEXT_SECONDARY
        };
        let case_btn = bar_text_button(
            "Aa",
            EditorMessage::FindToggleCaseSensitivity,
            theme::TEXT_11,
            case_label_color,
            [theme::PAD_2, theme::PAD_6],
        );

        let bar = row![
            search_input,
            replace_input,
            prev_btn,
            text(match_label)
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY),
            next_btn,
            Space::new().width(Length::Fixed(4.0)),
            case_btn,
            replace_btn,
            replace_all_btn,
        ]
        .spacing(theme::SPACE_4)
        .align_y(Alignment::Center)
        .padding([theme::PAD_4, theme::PAD_8]);

        Some(bar_harness(bar))
    }

    /// The go-to-line bar, which shares the find bar's slot below the tabs and excludes it.
    fn build_go_to_line_bar(&self) -> Option<Element<'_, EditorMessage>> {
        let ModalKind::GotoLine = self.active_modal.as_ref()? else {
            return None;
        };

        let line_input = widgets::single_line_editor(
            &self.goto_line_input.buffer,
            "Line #",
            true,
            Length::Fixed(120.0),
            Some(Id::new(GOTO_LINE_INPUT_ID)),
            |action| match action {
                super::editor_widget::EditorAction::Submit => EditorMessage::GoToLineGo,
                other => EditorMessage::GoToLineInput(other),
            },
        );

        let go_btn = bar_text_button(
            "Go",
            EditorMessage::GoToLineGo,
            theme::TEXT_12,
            theme::TEXT_SECONDARY,
            [theme::PAD_2, theme::PAD_8],
        );

        let bar = row![
            text("Go to line:")
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY),
            Space::new().width(theme::SPACE_4),
            line_input,
            go_btn,
        ]
        .spacing(theme::SPACE_4)
        .align_y(Alignment::Center)
        .padding([theme::PAD_4, theme::PAD_8]);

        Some(bar_harness(bar))
    }

    fn build_editor_widget(&self, dashboard_modal_open: bool) -> Element<'_, EditorMessage> {
        let Some(idx) = self.active_tab_idx() else {
            return widgets::empty_state_placeholder(
                lucide::file(),
                "No file selected",
                theme::TEXT_MUTED,
            );
        };

        let path = &self.tabs[idx].path;
        let Some(tab_data) = self.tab_contents.get(path) else {
            return widgets::empty_state_placeholder(
                lucide::triangle_alert(),
                "Error: tab content missing",
                theme::STATUS_ERROR,
            );
        };

        let content = &tab_data.content;
        let tree_focused = self.file_tree.tree_focused;
        let find_bar_open = tab_data.find_replace_state.is_some();
        // Modal overlays own keyboard input entirely — editor-internal modals as well as
        // dashboard-level ones (diff, git branch, update confirm).
        let modal_overlay_open = self.active_modal.is_some() || dashboard_modal_open;
        // The open find/replace bar's text inputs own focus the same way, so navigation must
        // not move the code cursor while they are being edited.
        let ignore_keyboard = tree_focused || modal_overlay_open || find_bar_open;

        // Match highlights as (line, byte_col_start, byte_col_end) for cosmic_text cursors.
        let (match_tuples, match_current_idx) = tab_data
            .find_replace_state
            .as_ref()
            .map(|state| {
                let text = tab_data.content.text();
                let tuples: Vec<(usize, usize, usize)> = state
                    .matches
                    .iter()
                    .filter_map(|range| {
                        let (line, byte_col_start, line_start) =
                            byte_offset_to_line_byte_col(&text, range.start)?;
                        let byte_col_end = range.end.saturating_sub(line_start);
                        Some((line, byte_col_start, byte_col_end))
                    })
                    .collect();
                (tuples, state.current_match_idx)
            })
            .unwrap_or_default();

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
            bracket_pair,
        )
    }

    fn build_highlighted_editor<'a>(
        content: &'a super::editor_widget::EditorBuffer,
        buffer_key: Option<&'a str>,
        ignore_keyboard: bool,
        matches: Vec<(usize, usize, usize)>,
        match_current_idx: usize,
        bracket_pair: Option<super::editor_widget::BracketPair>,
    ) -> Element<'a, EditorMessage> {
        let editor = super::editor_widget::EditorWidget::new(content)
            .padding(theme::PAD_8)
            .ignore_keyboard(ignore_keyboard)
            .matches(matches, match_current_idx)
            .bracket_pair(bracket_pair)
            .buffer_key(buffer_key);
        let element = iced::Element::new(editor);
        let mapped = element.map(EditorMessage::EditorAction);

        widgets::page_bare(mapped)
    }

    /// Perform file deletion: close tab, delete file, re-read parent directory.
    fn perform_file_delete(&mut self, target: &DeleteConfirmTarget) -> Task<EditorMessage> {
        if let Some(tab_idx) = self.tabs.iter().position(|t| t.path == target.abs_path) {
            self.remove_tab_at(tab_idx);
        }
        // `selected_file` stores relative paths (set by the SelectFile handler).
        if self.selected_file.as_deref() == Some(&target.path) {
            self.selected_file = None;
        }
        self.file_mtimes.remove(&target.abs_path);
        self.deleted_file_toasted.remove(&target.abs_path);

        self.perform_delete_with_refresh(
            target.abs_path.clone(),
            &target.path,
            "file",
            None,
            |abs_path| async move {
                tokio::fs::remove_file(&abs_path)
                    .await
                    .map_err(|e| e.to_string())
            },
        )
    }

    /// Delete a directory and close the affected tabs. The tree/cache prune waits for the success
    /// path ([`EditorMessage::DirDeleted`]), so a failed delete (permission denied) leaves the
    /// still-existing directory intact in the tree.
    fn perform_dir_delete(&mut self, target: &DeleteConfirmTarget) -> Task<EditorMessage> {
        // Collect open tabs inside this directory (close in reverse order).
        let mut affected_indices: Vec<usize> = self
            .tabs
            .iter()
            .enumerate()
            .filter(|(_, t)| is_within_abs(&t.path, &target.abs_path))
            .map(|(i, _)| i)
            .collect();
        affected_indices.sort_unstable_by(|a, b| b.cmp(a));

        for &idx in &affected_indices {
            self.remove_tab_at(idx);
        }

        self.perform_delete_with_refresh(
            target.abs_path.clone(),
            &target.path,
            "directory",
            Some(EditorMessage::DirDeleted {
                dir_path: target.path.clone(),
                workspace_path: self.selected_workspace_path.clone().unwrap_or_default(),
            }),
            |abs_path| async move {
                tokio::fs::remove_dir_all(&abs_path)
                    .await
                    .map_err(|e| e.to_string())
            },
        )
    }

    /// Shared preamble for a file or directory delete: bump a generation, run `delete_op` on the
    /// absolute path, then on success either emit `success_msg` (a directory delete that must
    /// prune tree state) or re-read the parent and emit a [`DirExpanded`]. `error_label` names the
    /// thing in the failure toast.
    fn perform_delete_with_refresh<D, F>(
        &mut self,
        abs_path: String,
        rel_path: &str,
        error_label: &'static str,
        success_msg: Option<EditorMessage>,
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
        // Only the file-delete path re-reads the parent via a DirExpanded
        // message and thus needs a registered generation slot. The
        // directory-delete path emits DirDeleted, which prunes state and
        // re-reads the parent with a fresh generation of its own —
        // registering a slot here would leave an orphaned (never consumed)
        // generation entry.
        let r#gen = if success_msg.is_none() {
            let r#gen = self.bump_generation();
            // Register the generation so DirExpanded handler accepts the result.
            self.dir_generations.insert(parent_dir.clone(), r#gen);
            Some(r#gen)
        } else {
            None
        };

        Task::perform(
            async move {
                if let Err(e) = delete_op(abs_path).await {
                    return EditorMessage::Toast(super::ToastMessage::Error(format!(
                        "Failed to delete {error_label}: {e}"
                    )));
                }
                if let Some(msg) = success_msg {
                    return msg;
                }
                // Re-read parent directory (the file-delete path registered
                // the generation slot above).
                let r#gen = r#gen.expect("file-delete path always registers a generation");
                dir_expanded_msg(ws_path, parent_dir, r#gen, false).await
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
        let r#gen = self.bump_generation();
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
                dir_expanded_msg(ws_root, parent_dir, r#gen, false).await
            },
            |msg| msg,
        )
    }

    /// Fire-and-forget reveal in system file manager.
    fn perform_reveal_in_finder(path: String) -> Task<EditorMessage> {
        Task::perform(
            async move {
                if let Err(e) = super::reveal_in_file_manager(std::path::Path::new(&path)) {
                    tracing::warn!("Failed to reveal {path}: {e}");
                }
            },
            |()| (),
        )
        .discard::<EditorMessage>()
    }
}

/// Map a keyboard event to an [`EditorMessage`], or `None`. The `filter_map` predicate of
/// `subscription()`, kept standalone so timer setup and shortcut precedence read separately.
/// With `dashboard_modal_open` the arrow/Enter tree keys are swallowed (the tree sits behind the
/// modal) while the global shortcuts above them (Cmd+S, Cmd+F, Ctrl+B, F3, …) keep working.
fn map_editor_shortcut(
    event: keyboard::Event,
    dashboard_modal_open: bool,
) -> Option<EditorMessage> {
    use keyboard::Key;
    let (key, modifiers, physical_key) = super::parse_key_press(event)?;
    let km = super::detect_keyboard_mods(modifiers);

    // Helper: match a Character key by its Latin equivalent.
    let latin = |target: char| -> bool { key.to_latin(physical_key) == Some(target) };

    if km.is_shortcut_platform_mod() && latin('b') {
        return Some(EditorMessage::TreeFocusToggled);
    }
    // Cmd+Shift+F / Ctrl+Shift+F → global search. Must come BEFORE the Cmd+F check below, and
    // that check's shift guard keeps Cmd+Shift+F from also triggering FindToggle.
    if km.is_platform_mod && !km.altgr_active && modifiers.shift() && latin('f') {
        return Some(EditorMessage::GlobalSearchToggle);
    }
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
    // Ctrl+W closes the tab on every platform: macOS gives Cmd+W to the window manager.
    if !km.altgr_active && modifiers.control() && latin('w') {
        return Some(EditorMessage::CloseActiveTab);
    }
    // Go-to-line: Ctrl+G on every platform. Cmd+L is the dashboard's Logs
    // shortcut now, and macOS keeps Cmd+G for find-next.
    let go_to_line_mod = if cfg!(target_os = "macos") {
        modifiers.control() && !modifiers.command()
    } else {
        !km.altgr_active && modifiers.control()
    };
    if go_to_line_mod && latin('g') {
        return Some(EditorMessage::GoToLineToggle);
    }
    if km.is_shortcut_platform_mod() && latin('p') {
        return Some(EditorMessage::QuickOpenToggle);
    }
    if km.is_shortcut_platform_mod() && latin('r') {
        return Some(EditorMessage::RefreshFileTree);
    }
    // Cmd+G / F3 → next match, +Shift → previous. Cmd+G is macOS's (Ctrl+G is go-to-line
    // everywhere), so F3 and Shift+F3 carry the other platform.
    #[cfg(target_os = "macos")]
    if modifiers.command() && !modifiers.control() && latin('g') {
        return if modifiers.shift() {
            Some(EditorMessage::FindPrev)
        } else {
            Some(EditorMessage::FindNext)
        };
    }
    if matches!(key, Key::Named(keyboard::key::Named::F3)) {
        return if modifiers.shift() {
            Some(EditorMessage::FindPrev)
        } else {
            Some(EditorMessage::FindNext)
        };
    }
    // Shift+Enter → previous match for the find bar (a no-op while it is closed).
    if modifiers.shift() && matches!(key, Key::Named(keyboard::key::Named::Enter)) {
        return Some(EditorMessage::FindPrev);
    }
    // Arrow/Enter tree navigation. With quick-open active the arrows drive its result list
    // (`update` checks quick-open before tree focus); with a dashboard modal open the tree sits
    // behind it, so these keys are swallowed entirely, guarding against a stale `tree_focused`.
    if dashboard_modal_open {
        return None;
    }
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
}

impl FindReplaceState {
    /// Recompute matches against the current text and jump to the first match (or reset the index).
    fn recompute(&mut self, content: &EditorBuffer) {
        let text = content.text();
        self.matches = compute_text_matches(&text, &self.query.text(), self.case_sensitive);
        auto_jump_to_first_match(&text, content, self);
    }
}

/// Convert a byte offset to (line, byte column within line, line byte start).
#[must_use]
fn byte_offset_to_line_byte_col(text: &str, offset: usize) -> Option<(usize, usize, usize)> {
    if offset > text.len() {
        return None;
    }
    let (line, line_start) = super::text_rendering::byte_line_and_start(text, offset);
    Some((line, offset - line_start, line_start))
}

/// Auto-jump the cursor to the first find match and reset the match index to 0.
fn auto_jump_to_first_match(
    text: &str,
    content: &super::editor_widget::EditorBuffer,
    state: &mut FindReplaceState,
) {
    state.current_match_idx = 0;
    if let Some(range) = state.matches.first() {
        let (line, col) = byte_offset_to_line_col(text, range.start);
        content.move_to(line, col);
    }
}

#[cfg(test)]
#[path = "editor_tests.rs"]
mod tests;
