//! Diff dashboard page — view git diff against HEAD with syntax highlighting.
//!
//! Shows staged + unstaged changes via `git diff HEAD`, untracked files
//! via `git status --porcelain`, with per-file tree-sitter syntax highlighting.
//! Files are parsed in their entirety for correct multi-line token coloring;
//! working-tree diffs read the old version from HEAD and the new version from
//! disk, while commit views read both sides via git (`run_git_show`).
//!
//! The page layout splits into an auto-sizing directory tree sidebar (left)
//! and a scrollable diff panel (right, filling the remaining width). Click a
//! file in the tree to filter the diff to just that file; click again to show
//! all files.
//!
//! Auto-refreshes every 5 seconds when a workspace is selected.
use super::diff_widget::{self, DiffBufferWidget, DiffFileBuffer};
use super::highlight::{FileHighlights, HighlightLanguage, parse_highlights};

use crate::git::commands::{
    CommitInfo, DiscardTarget, MAX_UNTRACKED_SIZE, UntrackedFileRead, git_has_commits,
    git_is_installed, is_git_repo, parse_untracked_from_porcelain, read_untracked_file,
    run_git_commit, run_git_diff, run_git_discard, run_git_show, run_git_status,
};
use crate::git::diff_parse::{
    DiffContent, DiffFileStatus, DiffLineKind, make_untracked_diff_file, parse_git_diff,
};

use iced::widget::Id;
use iced::widget::{Space, button, column, container, row, text};
use iced::{Alignment, Color, Element, Length, Task, keyboard};

use iced_fonts::lucide;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::menus::{ContextMenu, MenuItem};
use super::theme;
use super::widgets::{self, FileTree, TreeNavDirection};

const MAX_DIFF_LINES: usize = 20000;
const MAX_HUNKS: usize = 1000;

/// Returns `true` if `file` matches the optional `selected_file` path.
///
/// When `selected_file` is `None`, every file matches (no filter is active).
/// When `selected_file` is `Some(path)`, only the file whose `.path` equals
/// `path` matches.
///
/// # Behavioural invariant
///
/// This filter must be applied identically in `compute_truncation_index`,
/// `build_diff_content`, and `diff_widget::build_file_buffers` so that
/// truncation boundaries and rendered content agree.  This function is the
/// single point of truth.
pub(super) fn file_matches_selection(file: &DiffFile, selected_file: Option<&str>) -> bool {
    selected_file.is_none_or(|sel| file.path == sel)
}

/// Compute the index of the first [`DiffFile`] to exclude due to truncation limits.
///
/// Iterates over `diff_files`, applying the same `selected_file` filter and
/// binary/too-large skipping as the rendering functions. Returns `Some(idx)`
/// where `idx` is the first file that would exceed either `max_hunks` or
/// `max_lines`, or `None` if all files fit within the limits.
///
/// When `limits` is `None`, always returns `None` (no truncation).
///
/// # Behavioural contract
///
/// - Truncation is **file-boundary only**: no mid-hunk or mid-file cut-off.
/// - Binary and too-large files consume no hunk/line capacity but occupy
///   array positions — the returned index accounts for them.
/// - File filtering delegates to [`file_matches_selection`] — the single
///   source of truth.
pub(super) fn compute_truncation_index(
    diff_files: &[DiffFile],
    selected_file: Option<&str>,
    limits: Option<(usize, usize)>,
) -> Option<usize> {
    let (max_hunks, max_lines) = limits?;
    let mut total_hunks = 0usize;
    let mut total_lines = 0usize;

    for (idx, file) in diff_files.iter().enumerate() {
        if !file_matches_selection(file, selected_file) {
            continue;
        }

        // Binary and too-large files consume no hunk/line capacity.
        if !file.has_parseable_content() {
            continue;
        }

        let file_hunks = file.hunks.len();
        let file_lines: usize = file.hunks.iter().map(|h| h.lines.len()).sum();

        if total_hunks + file_hunks > max_hunks || total_lines + file_lines > max_lines {
            return Some(idx);
        }

        total_hunks += file_hunks;
        total_lines += file_lines;
    }

    None
}

const FILE_HEADER_COLOR: Color = theme::STATUS_WARNING;
const RENAME_COLOR: Color = theme::ACCENT_LIGHT;

#[derive(Debug, Clone)]
pub enum DiffMessage {
    /// The workspace the dashboard resolved for the current selection, as the
    /// filesystem path every git operation of this view runs against. The diff
    /// view never resolves a workspace by name; the dashboard sends
    /// [`DiffMessage::WorkspaceUnresolved`] instead while nothing resolves.
    WorkspaceSelected(String),
    /// Nothing is resolved: drop the workspace this view was showing.
    WorkspaceUnresolved,
    DiffLoaded(u64, Result<Vec<DiffFile>, String>),
    Tick,
    ToggleDir(String),
    SelectFile(String),
    CommitMessageChanged(super::editor_widget::EditorAction),
    CommitClicked,
    CommitResult(Result<CommitInfo, String>),
    /// Emitted on successful manual commit — signals the parent to close the modal.
    CloseModal,
    Toast(super::ToastMessage),
    // ── Tree keyboard navigation ──────────────────────────────────
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
    /// Navigate to a specific commit diff view (`commit_hash`) of the workspace
    /// this view already resolves, keeping that workspace across the navigation.
    NavigateToCommit(String),
    /// Return from historical commit view to working tree diff.
    BackToWorkingTree,
    /// Commit message fetched for a historical commit (commit_hash, message).
    CommitMessageFetched(String, Option<String>),
    /// Clear commit state (ref + message) without triggering any diff load.
    /// Used when the modal is closed to prevent stale accessor returns.
    ClearCommitState,
    /// Discard changes for a file or directory (path, target).
    DiscardPath(String, DiscardTarget),
    /// Result of a discard operation.
    DiscardResult(Result<(), String>),
}

/// A single changed file, enhanced with highlight data and line counts.
#[derive(Debug, Clone)]
pub struct DiffFile {
    dfile: crate::git::diff_parse::DiffFile,
    /// Per-line highlight spans for removed lines (from old version).
    pub old_highlights: Option<FileHighlights>,
    /// Per-line highlight spans for added/context lines (from new version).
    pub new_highlights: Option<FileHighlights>,
    /// Count of added lines across all hunks.
    pub add_count: usize,
    /// Count of removed lines across all hunks.
    pub remove_count: usize,
}

impl std::ops::Deref for DiffFile {
    type Target = crate::git::diff_parse::DiffFile;

    fn deref(&self) -> &Self::Target {
        &self.dfile
    }
}

impl DiffFile {
    /// Construct a `DiffFile` from a parsed diff file, computing
    /// `add_count` and `remove_count` automatically from hunks.
    #[must_use]
    pub fn from_parsed(
        dfile: crate::git::diff_parse::DiffFile,
        old_highlights: Option<FileHighlights>,
        new_highlights: Option<FileHighlights>,
    ) -> Self {
        let (add_count, remove_count) = count_lines(&dfile);
        Self {
            dfile,
            old_highlights,
            new_highlights,
            add_count,
            remove_count,
        }
    }
}

/// What the diff tree counter slot shows for a file row.
enum CountsSlot {
    Binary,
    Stats { added: i64, removed: i64 },
    Unchanged,
}

#[expect(clippy::cast_possible_wrap)] // usize line counts → i64 for diff_stats_row
fn counts_slot(file: Option<&DiffFile>) -> CountsSlot {
    match file {
        Some(f) if f.content == DiffContent::Binary => CountsSlot::Binary,
        Some(f) if f.add_count > 0 || f.remove_count > 0 => CountsSlot::Stats {
            added: f.add_count as i64,
            removed: f.remove_count as i64,
        },
        _ => CountsSlot::Unchanged,
    }
}

/// Wrap content in the standard diff status-pane chrome (padding + fill + style).
fn diff_pane_container<'a>(
    content: impl Into<Element<'a, DiffMessage>>,
) -> Element<'a, DiffMessage> {
    container(content)
        .padding([theme::PAD_8, theme::PAD_12])
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::base_container_style)
        .into()
}

pub struct DiffState {
    error: Option<String>,
    /// The filesystem path of the workspace the dashboard resolved for the
    /// current selection; `None` while nothing is resolved.
    resolved_workspace_path: Option<PathBuf>,
    generation: u64,
    diff_files: Vec<DiffFile>,
    diff_loading: bool,
    /// Whether at least one diff fetch has completed successfully
    /// (prevents "Loading diff…" flicker on auto-poll Ticks).
    diff_has_loaded: bool,
    status_message: Option<String>,
    /// Directory tree nodes (built from diff_files).
    file_tree: FileTree,
    /// Currently selected file (full path), or None to show all.
    selected_file: Option<String>,
    /// Per-file pre-computed cosmic_text buffer data. Built in `update()` when
    /// diff data or file selection changes; consumed by `view()`.
    file_buffers: Vec<DiffFileBuffer>,
    /// Commit message typed by the user.
    commit_message: super::common::SingleLineEditorState,
    /// Whether a commit is in-flight.
    committing: bool,
    /// Current commit being viewed, if any.
    /// `None` means we're viewing the working-tree diff (`git diff HEAD`).
    current_commit_ref: Option<String>,
    /// Commit message of the commit being viewed (fetched during NavigateToCommit).
    current_commit_message: Option<String>,
    /// When true, the next successful [`DiffMessage::DiffLoaded`] recursively
    /// expands all directory nodes in the file tree (nested folders included).
    /// Cleared after expansion. Not set on periodic auto-refresh ticks.
    tree_auto_expand_pending: bool,
}

impl DiffState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            error: None,
            resolved_workspace_path: None,
            generation: 0,
            diff_files: Vec::new(),
            diff_loading: false,
            diff_has_loaded: false,
            status_message: None,
            file_tree: FileTree::new(Id::new("diff_tree_panel")),
            selected_file: None,
            file_buffers: Vec::new(),
            commit_message: super::common::SingleLineEditorState::new(""),
            committing: false,
            current_commit_ref: None,
            current_commit_message: None,
            tree_auto_expand_pending: false,
        }
    }

    /// Whether this state is viewing a historical commit (vs working tree).
    #[must_use]
    pub fn is_viewing_commit(&self) -> bool {
        self.current_commit_ref.is_some()
    }

    /// Reset all diff view state fields that could render stale data during
    /// the async loading window after a context switch.
    ///
    /// Handlers that also need to update `resolved_workspace_path`,
    /// `diff_loading`, `diff_has_loaded`, or `generation` should call this
    /// first, then set those fields.
    fn clear_diff_state(&mut self) {
        self.error = None;
        self.status_message = None;
        self.file_tree.nodes.clear();
        self.file_tree.expanded_dirs.clear();
        self.selected_file = None;
        self.diff_files.clear();
        self.file_buffers.clear();
        self.current_commit_ref = None;
        self.current_commit_message = None;
        self.commit_message.clear();
        self.committing = false;
        self.tree_auto_expand_pending = true;
    }

    /// The commit message of the commit currently being viewed, if any.
    #[must_use]
    pub fn commit_message(&self) -> Option<&str> {
        self.current_commit_message.as_deref()
    }

    /// The short hash (7 chars) of the commit currently being viewed, if any.
    #[must_use]
    pub fn commit_short_hash(&self) -> Option<&str> {
        self.current_commit_ref
            .as_deref()
            .map(|h| h.get(..7).unwrap_or(h))
    }

    pub fn subscription(&self) -> iced::Subscription<DiffMessage> {
        let mut subs: Vec<iced::Subscription<DiffMessage>> = Vec::new();
        if self.resolved_workspace_path.is_some() {
            subs.push(iced::time::every(Duration::from_secs(5)).map(|_| DiffMessage::Tick));
        }
        // Keyboard: Ctrl+B toggles tree focus; arrow/Enter messages are ignored
        // by update handlers unless the tree is currently focused. This closure
        // must stay non-capturing because iced validates that in release builds.
        //
        // NOTE: Escape-dismiss of tree focus is deliberately NOT wired here.
        // `DiffMessage::Escape` was removed as dead code (the Dashboard never
        // forwards Escape to the diff modal, and this subscription maps only
        // Ctrl+B plus the arrow/Enter nav keys). If Escape-dismiss is ever
        // desired, wire it into this subscription first — do not re-add the
        // variant without the key mapping.
        subs.push(keyboard::listen().filter_map(|event| {
            use keyboard::Key;
            let (key, modifiers, physical_key) = super::parse_key_press(event)?;
            let km = super::detect_keyboard_mods(modifiers);
            if !km.altgr_active && km.is_cmd && key.to_latin(physical_key) == Some('b') {
                return Some(DiffMessage::TreeFocusToggled);
            }
            match &key {
                Key::Named(named) => match named {
                    keyboard::key::Named::ArrowUp => Some(DiffMessage::TreeNavUp),
                    keyboard::key::Named::ArrowDown => Some(DiffMessage::TreeNavDown),
                    keyboard::key::Named::ArrowLeft => Some(DiffMessage::TreeNavLeft),
                    keyboard::key::Named::ArrowRight => Some(DiffMessage::TreeNavRight),
                    keyboard::key::Named::Enter => Some(DiffMessage::TreeNavEnter),
                    _ => None,
                },
                _ => None,
            }
        }));
        iced::Subscription::batch(subs)
    }

    /// Start loading a diff. Returns a [`Task`] that will produce a [`DiffMessage::DiffLoaded`]
    /// when complete. Guards against stale results via the generation counter.
    ///
    /// Sets `diff_loading = true` and increments the generation counter.
    /// Callers are responsible for setting `diff_has_loaded` as appropriate before calling this
    /// (it should be `false` on context switches, left unchanged on auto-refresh ticks).
    fn spawn_diff_load(&mut self, commit_ref: Option<String>) -> Task<DiffMessage> {
        self.diff_loading = true;
        self.generation = self.generation.wrapping_add(1);
        let generation_num = self.generation;
        let ws_path = self.resolved_workspace_path.clone();
        Task::perform(load_diff(ws_path, commit_ref), move |r| {
            DiffMessage::DiffLoaded(generation_num, r)
        })
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: DiffMessage) -> Task<DiffMessage> {
        match msg {
            DiffMessage::WorkspaceSelected(path) => {
                self.clear_diff_state();
                self.resolved_workspace_path = Some(PathBuf::from(path));
                self.diff_has_loaded = false;
                self.spawn_diff_load(None)
            }
            DiffMessage::WorkspaceUnresolved => {
                self.clear_diff_state();
                self.resolved_workspace_path = None;
                self.diff_has_loaded = false;
                self.diff_loading = false;
                // Invalidate a load still in flight, so its result cannot land
                // over the workspace that was just cleared.
                self.generation = self.generation.wrapping_add(1);
                Task::none()
            }
            DiffMessage::DiffLoaded(generation_num, result) => {
                if generation_num != self.generation {
                    return Task::none();
                }
                self.diff_loading = false;
                self.diff_has_loaded = true;
                self.error = None;
                self.status_message = None;
                match result {
                    Ok(files) => {
                        if files.is_empty() {
                            self.status_message = if self.current_commit_ref.is_some() {
                                Some("No files changed in this commit.".to_string())
                            } else {
                                Some("Working tree clean.".to_string())
                            };
                        }
                        self.file_tree.nodes = build_tree(&files);
                        // Expand all directories recursively on first load for this
                        // workspace/commit context. Periodic refreshes do not reset this.
                        if self.tree_auto_expand_pending {
                            collect_dir_paths(
                                &self.file_tree.nodes,
                                &mut self.file_tree.expanded_dirs,
                            );
                            self.tree_auto_expand_pending = false;
                        }
                        // Preserve file selection across refreshes if the file still exists.
                        if let Some(ref sel) = self.selected_file {
                            if !files.iter().any(|f| f.path == *sel) {
                                self.selected_file = None;
                            }
                        }
                        self.diff_files = files;
                        self.rebuild_file_buffers();
                        self.file_tree.rebuild_visible();
                    }
                    Err(e) => {
                        self.error = Some(e);
                        self.diff_files = Vec::new();
                        self.file_tree.nodes = Vec::new();
                        self.selected_file = None;
                        self.file_tree.rebuild_visible();
                    }
                }
                Task::none()
            }
            DiffMessage::Tick => {
                if self.current_commit_ref.is_some() {
                    // Commit diffs are immutable — no auto-refresh needed.
                    return Task::none();
                }
                if self.resolved_workspace_path.is_some() && !self.diff_loading && !self.committing
                {
                    self.spawn_diff_load(None)
                } else {
                    Task::none()
                }
            }
            DiffMessage::NavigateToCommit(hash) => {
                self.clear_diff_state();
                // clear_diff_state() resets all viewing state (file tree, buffers,
                // error, etc.) including current_commit_ref. We re-establish
                // current_commit_ref below; the rest stays cleared for the new
                // commit's diff to load into.
                // Set commit ref before spawning task
                // (prevents Tick race: subscription checks .is_some() to skip).
                self.current_commit_ref = Some(hash.clone());
                self.diff_has_loaded = false;

                // Load the diff and fetch the commit message in parallel — both
                // against the workspace the dashboard resolved, which this view
                // keeps across the navigation.
                let msg_path = self.resolved_workspace_path.clone();
                let msg_hash = hash.clone();
                let msg_hash_for_git = hash.clone();
                let msg_task = Task::perform(
                    async move {
                        let path = msg_path?;
                        crate::git::commands::run_git_commit_message(&path, Some(&msg_hash_for_git))
                            .await
                            .ok()
                    },
                    move |msg| DiffMessage::CommitMessageFetched(msg_hash, msg),
                );

                Task::batch([self.spawn_diff_load(Some(hash)), msg_task])
            }
            DiffMessage::BackToWorkingTree => {
                if self.resolved_workspace_path.is_none() {
                    return Task::none();
                }
                // clear_diff_state() sets current_commit_ref to None.
                // Iced's update() is single-threaded, so Tick can't race.
                self.clear_diff_state();
                self.diff_has_loaded = false;
                self.spawn_diff_load(None)
            }
            DiffMessage::CommitMessageFetched(hash, msg) => {
                // Only accept if we're still viewing the same commit.
                if self.current_commit_ref.as_deref() == Some(&hash) {
                    self.current_commit_message = msg;
                }
                Task::none()
            }
            DiffMessage::ClearCommitState => {
                self.current_commit_ref = None;
                self.current_commit_message = None;
                Task::none()
            }
            DiffMessage::ToggleDir(path) => {
                self.file_tree.tree_focused = true;
                if self.file_tree.expanded_dirs.contains(&path) {
                    self.file_tree.expanded_dirs.remove(&path);
                    return self
                        .file_tree
                        .collapse_dir_and_keep_focus::<DiffMessage>(&path);
                }
                self.file_tree.expanded_dirs.insert(path.clone());
                // Expand: rebuild visible tree, keep focus on the directory (don't advance).
                self.file_tree.rebuild_visible();
                self.file_tree.focus_path(&path);
                Task::none()
            }
            DiffMessage::SelectFile(path) => {
                // Clicking a file tree row transfers keyboard focus to the tree
                // so that arrow keys navigate the tree instead of the editor.
                self.file_tree.tree_focused = true;
                // Remember the clicked file's position for Ctrl+B re-focus.
                self.file_tree.focus_path(&path);
                if self.selected_file.as_deref() == Some(&path) {
                    self.selected_file = None;
                } else {
                    self.selected_file = Some(path);
                }
                self.rebuild_file_buffers();
                Task::none()
            }
            DiffMessage::CommitMessageChanged(action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                self.commit_message.apply_action(action);
                self.file_tree.tree_focused = false;
                Task::none()
            }
            DiffMessage::CommitClicked => {
                let trimmed = self.commit_message.text().trim().to_string();
                if trimmed.is_empty() || self.committing {
                    return Task::none();
                }
                let Some(ws_path) = self.resolved_workspace_path.clone() else {
                    tracing::error!("CommitClicked with no workspace resolved");
                    return Task::none();
                };
                self.committing = true;
                Task::perform(
                    async move {
                        run_git_commit(&ws_path, &trimmed)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    DiffMessage::CommitResult,
                )
            }
            DiffMessage::CommitResult(result) => {
                self.committing = false;
                match result {
                    Ok(info) => {
                        self.commit_message.clear();
                        self.error = None;
                        let toast_msg = match (info.lines_added, info.lines_removed) {
                            (0, 0) => {
                                format!("Committed {}", info.short_hash())
                            }
                            (a, 0) => format!("Committed {} (+{a})", info.short_hash()),
                            (0, r) => format!("Committed {} (\u{2212}{r})", info.short_hash()),
                            (a, r) => format!("Committed {} (+{a}/\u{2212}{r})", info.short_hash()),
                        };
                        // Immediately refresh the diff.
                        if self.resolved_workspace_path.is_some() {
                            return Task::batch([
                                Task::done(DiffMessage::CloseModal),
                                Task::done(DiffMessage::Toast(super::ToastMessage::SuccessMsg(
                                    toast_msg,
                                ))),
                                self.spawn_diff_load(None),
                            ]);
                        }
                        Task::batch([
                            Task::done(DiffMessage::CloseModal),
                            Task::done(DiffMessage::Toast(super::ToastMessage::SuccessMsg(
                                toast_msg,
                            ))),
                        ])
                    }
                    Err(e) => {
                        self.error = Some(e);
                        Task::none()
                    }
                }
            }
            DiffMessage::CloseModal | DiffMessage::Toast(_) => Task::none(),

            DiffMessage::DiscardPath(path, target) => {
                // Guard: no discard from historical commit view.
                if self.current_commit_ref.is_some() {
                    return Task::none();
                }
                // Nothing resolved → nothing to discard into.
                let Some(ws_path) = self.resolved_workspace_path.clone() else {
                    return Task::none();
                };
                // Mark as loading so the user sees the diff update.
                self.diff_loading = true;
                Task::perform(
                    async move {
                        run_git_discard(&ws_path, &path, target)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    DiffMessage::DiscardResult,
                )
            }

            DiffMessage::DiscardResult(result) => {
                match result {
                    Ok(()) => {
                        // Refresh the diff immediately.
                        if self.resolved_workspace_path.is_some() {
                            return Task::batch([
                                Task::done(DiffMessage::Toast(super::ToastMessage::SuccessMsg(
                                    "Changes discarded.".to_string(),
                                ))),
                                self.spawn_diff_load(None),
                            ]);
                        }
                        self.diff_loading = false;
                        Task::done(DiffMessage::Toast(super::ToastMessage::SuccessMsg(
                            "Changes discarded.".to_string(),
                        )))
                    }
                    Err(e) => {
                        self.diff_loading = false;
                        Task::done(DiffMessage::Toast(super::ToastMessage::Error(e)))
                    }
                }
            }

            // ── Tree keyboard navigation ─────────────────────────────
            DiffMessage::TreeFocusToggled => {
                self.file_tree.tree_focused = !self.file_tree.tree_focused;
                if self.file_tree.tree_focused && self.file_tree.visible_tree_nodes.is_empty() {
                    self.file_tree.rebuild_visible();
                }
                if !self.file_tree.tree_focused || self.file_tree.visible_tree_nodes.is_empty() {
                    self.file_tree.tree_focused = false;
                }
                Task::none()
            }

            DiffMessage::TreeScrolled(scroll_y, viewport_h) => {
                self.file_tree.scroll_y = scroll_y;
                self.file_tree.viewport_h = Some(viewport_h);
                Task::none()
            }

            DiffMessage::TreeNavUp => self
                .file_tree
                .nav_and_scroll::<DiffMessage>(TreeNavDirection::Up),

            DiffMessage::TreeNavDown => self
                .file_tree
                .nav_and_scroll::<DiffMessage>(TreeNavDirection::Down),

            DiffMessage::TreeNavEnter => {
                let Some((_idx, path, is_dir)) = self.file_tree.focused_tree_node() else {
                    return Task::none();
                };
                if self.file_tree.focused_is_expanded_dir() {
                    // Collapse and keep focus on the collapsed directory.
                    self.file_tree.expanded_dirs.remove(&path);
                    return self
                        .file_tree
                        .collapse_dir_and_keep_focus::<DiffMessage>(&path);
                }
                if is_dir {
                    // Expand directory and move focus to the first child.
                    self.file_tree.expanded_dirs.insert(path.clone());
                    return self
                        .file_tree
                        .expand_dir_and_focus_first_child::<DiffMessage>(&path);
                }
                // Open file.
                Task::done(DiffMessage::SelectFile(path))
            }

            DiffMessage::TreeNavLeft => {
                let Some((_idx, path, _)) = self.file_tree.focused_tree_node() else {
                    return Task::none();
                };

                if self.file_tree.focused_is_expanded_dir() {
                    // Collapse expanded directory and keep focus on it.
                    self.file_tree.expanded_dirs.remove(&path);
                    return self
                        .file_tree
                        .collapse_dir_and_keep_focus::<DiffMessage>(&path);
                }

                // ArrowLeft on collapsed directory or file — navigate to parent.
                self.file_tree.focus_parent::<DiffMessage>()
            }

            DiffMessage::TreeNavRight => {
                let Some((idx, path, is_dir)) = self.file_tree.focused_tree_node() else {
                    return Task::none();
                };

                if !is_dir {
                    // ArrowRight on a file does nothing.
                    return Task::none();
                }

                if !self.file_tree.expanded_dirs.contains(&path) {
                    // Expand directory and move focus to the first child.
                    self.file_tree.expanded_dirs.insert(path.clone());
                    return self
                        .file_tree
                        .expand_dir_and_focus_first_child::<DiffMessage>(&path);
                }

                // Already expanded directory — move focus to first child (if any).
                self.file_tree.focus_next_row::<DiffMessage>(idx)
            }
        }
    }

    pub fn view(&self) -> Element<'_, DiffMessage> {
        let has_changes = !self.diff_files.is_empty();
        let show_commit_bar =
            self.current_commit_ref.is_none() && has_changes && self.error.is_none();

        // Build the header row: commit controls or commit-view banner.
        let header = if let Some(short_hash) = self.commit_short_hash() {
            // Historical commit view — show banner with Back button.
            let back_btn = button("Back to working tree")
                .on_press(DiffMessage::BackToWorkingTree)
                .style(theme::button_secondary);
            container(
                row![
                    text(format!("Viewing commit {short_hash}"))
                        .size(theme::TEXT_12)
                        .color(theme::TEXT_SECONDARY),
                    Space::new().width(Length::Fill),
                    back_btn,
                ]
                .align_y(Alignment::Center),
            )
            .width(Length::Fill)
            .padding([theme::PAD_8, theme::PAD_12])
        } else if show_commit_bar {
            let commit_input = super::widgets::single_line_editor(
                &self.commit_message.buffer,
                "Commit message…",
                true,
                Length::Fill,
                Some(Id::new("diff_commit_message")),
                |action| match action {
                    super::editor_widget::EditorAction::Submit => DiffMessage::CommitClicked,
                    other => DiffMessage::CommitMessageChanged(other),
                },
            );

            let msg_empty = self.commit_message.text().trim().is_empty();
            let commit_disabled = msg_empty || self.committing;

            let commit_btn = button(if self.committing {
                text("Committing…").size(theme::TEXT_12)
            } else {
                text("Commit").size(theme::TEXT_12)
            })
            .on_press_maybe(if commit_disabled {
                None
            } else {
                Some(DiffMessage::CommitClicked)
            })
            .style(theme::button_secondary);

            container(
                row![commit_input, Space::new().width(theme::SPACE_8), commit_btn]
                    .align_y(Alignment::Center),
            )
            .width(Length::Fill)
            .padding([theme::PAD_8, theme::PAD_12])
        } else {
            // Empty spacer to maintain layout consistency.
            container(Space::new().width(Length::Fill).height(0))
        };

        let status: Element<'_, DiffMessage> = if let Some(ref err) = self.error {
            widgets::error_banner(err)
        } else if let Some(ref s) = self.status_message {
            diff_pane_container(text(s).size(theme::TEXT_13).color(theme::TEXT_SECONDARY))
        } else if self.diff_loading && !self.diff_has_loaded {
            diff_pane_container(widgets::loading_text())
        } else if self.resolved_workspace_path.is_none() {
            // Defensive: the modal is gated on a resolution, so this is not
            // reached in production; it keeps an unresolved view from claiming
            // a clean working tree.
            diff_pane_container(
                text("Select a workspace to view its diff.")
                    .size(theme::TEXT_13)
                    .color(theme::TEXT_MUTED),
            )
        } else if self.diff_files.is_empty() {
            container(
                column![
                    lucide::check::<iced::Theme, iced::Renderer>()
                        .size(32)
                        .color(theme::STATUS_SUCCESS),
                    Space::new().height(8),
                    text("Working tree clean.")
                        .size(theme::TEXT_14)
                        .color(theme::TEXT_SECONDARY),
                ]
                .align_x(Alignment::Center)
                .spacing(theme::SPACE_4),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(theme::base_container_style)
            .into()
        } else {
            self.build_split_view()
        };

        column![header, status]
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// Build the two-panel layout: tree sidebar (left) + diff content (right).
    fn build_split_view(&self) -> Element<'_, DiffMessage> {
        let tree_panel = self.build_tree_panel();
        let diff_panel = self.build_diff_content();

        row![tree_panel, diff_panel]
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// Build the directory tree sidebar.
    fn build_tree_panel(&self) -> Element<'_, DiffMessage> {
        let nodes = &self.file_tree.nodes;
        let count = nodes.len();
        let elements: Vec<Element<'_, DiffMessage>> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| self.render_tree_node(n, 0, 0, i == count - 1))
            .collect();
        // Natural width of every rendered row, in render order, for the
        // auto-sizing tree panel. Replicates the exact row composition:
        // guide (depth*2 box-drawing chars at 14px) + lucide icon
        // (TREE_ICON_SIZE for dirs and files) + 4px gap + name at 14px, plus
        // the ± change counts at 10px with a 6px trailing gap on file rows.
        let row_widths = widgets::collect_tree_row_widths(
            &self.file_tree.nodes,
            &self.file_tree.expanded_dirs,
            |node, depth| {
                let guide_chars = depth * 2;
                if node.is_dir {
                    widgets::tree_row_natural_width(
                        guide_chars,
                        widgets::TREE_ICON_SIZE,
                        &node.name,
                        widgets::TREE_FONT_SIZE,
                        None,
                        None,
                    )
                } else {
                    // File rows always render a counts slot (empty when the
                    // file has no changes or is not in the diff list) plus a
                    // 6px trailing gap.
                    let (add_label, rem_label) = match counts_slot(
                        self.diff_files.iter().find(|f| f.path == node.full_path),
                    ) {
                        CountsSlot::Binary => ("binary".to_string(), String::new()),
                        CountsSlot::Stats { added, removed } => (
                            if added > 0 {
                                format!("+{added}")
                            } else {
                                String::new()
                            },
                            if removed > 0 {
                                format!("\u{2212}{removed}")
                            } else {
                                String::new()
                            },
                        ),
                        CountsSlot::Unchanged => (String::new(), String::new()),
                    };
                    widgets::tree_row_natural_width(
                        guide_chars,
                        widgets::TREE_ICON_SIZE,
                        &node.name,
                        widgets::TREE_FONT_SIZE,
                        None,
                        Some((add_label.as_str(), rem_label.as_str())),
                    )
                }
            },
        );
        widgets::build_tree_panel(&self.file_tree, elements, &row_widths, |viewport| {
            DiffMessage::TreeScrolled(viewport.absolute_offset().y, viewport.bounds().height)
        })
    }

    /// Recursively render a tree node and its children.
    fn render_tree_node<'a>(
        &'a self,
        node: &'a widgets::TreeNode,
        depth: usize,
        ancestor_mask: u64,
        is_last: bool,
    ) -> Element<'a, DiffMessage> {
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
    ) -> Element<'a, DiffMessage> {
        let is_expanded = self.file_tree.expanded_dirs.contains(&node.full_path);
        let icon: iced::widget::Text<'static, iced::Theme, iced::Renderer> = if is_expanded {
            lucide::folder_open()
        } else {
            lucide::folder()
        };
        let icon_color = if is_expanded {
            theme::ACCENT_LIGHT
        } else {
            theme::TEXT_MUTED
        };

        let guide = widgets::tree_guide_prefix(ancestor_mask, depth, is_last);
        let icon_element: Element<'_, DiffMessage> =
            icon.size(widgets::TREE_ICON_SIZE).color(icon_color).into();
        let name_element: Element<'_, DiffMessage> = text(&node.name)
            .size(widgets::TREE_FONT_SIZE)
            .color(theme::TEXT_SECONDARY)
            .into();

        let header_row = widgets::tree_row(guide, icon_element, name_element, None);

        let full_path = node.full_path.clone();
        let is_focused = widgets::tree_node_focused(&self.file_tree, &node.full_path);

        let header_btn = widgets::tree_node_button(
            header_row,
            is_focused,
            Some(DiffMessage::ToggleDir(full_path.clone())),
        );

        let header_element: Element<'_, DiffMessage> = discard_menu_button(
            header_btn,
            full_path,
            DiscardTarget::Directory,
            self.current_commit_ref.is_some(),
        );

        let mut col = column![header_element].spacing(0);
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
    ) -> Element<'a, DiffMessage> {
        let file = self.diff_files.iter().find(|f| f.path == node.full_path);
        let is_selected = self.selected_file.as_deref() == Some(&node.full_path);

        let guide = widgets::tree_guide_prefix(ancestor_mask, depth, is_last);

        // File status icon
        let (icon, icon_color) = if let Some(f) = file {
            if f.status == DiffFileStatus::Renamed {
                (lucide::arrow_right(), RENAME_COLOR)
            } else if matches!(f.status, DiffFileStatus::Added | DiffFileStatus::Untracked) {
                (lucide::file_plus(), FILE_HEADER_COLOR)
            } else if f.status == DiffFileStatus::Deleted {
                (lucide::file_minus(), FILE_HEADER_COLOR)
            } else if f.content == DiffContent::Binary {
                (lucide::file(), theme::TEXT_MUTED)
            } else {
                (lucide::file_text(), FILE_HEADER_COLOR)
            }
        } else {
            (lucide::file_text(), FILE_HEADER_COLOR)
        };

        // Line count labels
        let counts: Element<'_, DiffMessage> = match counts_slot(file) {
            CountsSlot::Binary => text("binary")
                .size(theme::TEXT_10)
                .color(theme::TEXT_SECONDARY)
                .into(),
            CountsSlot::Stats { added, removed } => {
                widgets::diff_stats_row(added, removed, 10.0).into()
            }
            CountsSlot::Unchanged => text("").size(theme::TEXT_10).into(),
        };

        let name_color = if is_selected {
            theme::TEXT_PRIMARY
        } else {
            theme::TEXT_SECONDARY
        };
        let name_weight = if is_selected {
            iced::font::Weight::Bold
        } else {
            iced::font::Weight::Normal
        };

        let icon_element: Element<'_, DiffMessage> =
            icon.size(widgets::TREE_ICON_SIZE).color(icon_color).into();
        let name_element: Element<'_, DiffMessage> = text(&node.name)
            .size(widgets::TREE_FONT_SIZE)
            .color(name_color)
            .font(iced::Font {
                weight: name_weight,
                ..theme::FONT_REGULAR
            })
            .into();

        let btn_row = widgets::tree_row(guide, icon_element, name_element, Some(counts));

        let full_path = node.full_path.clone();
        let is_focused = widgets::tree_node_focused(&self.file_tree, &node.full_path);

        let file_btn = widgets::tree_node_button(
            btn_row,
            is_selected || is_focused,
            Some(DiffMessage::SelectFile(full_path.clone())),
        );

        discard_menu_button(
            file_btn,
            full_path,
            DiscardTarget::File,
            self.current_commit_ref.is_some(),
        )
    }

    /// Return the diff content panel: file headers, binary/too-large placeholders,
    /// truncation warnings, and per-file [`DiffBufferWidget`]s interleaved.
    #[expect(clippy::too_many_lines)]
    fn build_diff_content(&self) -> Element<'_, DiffMessage> {
        if self.diff_files.is_empty() {
            return widgets::page_bare(widgets::vscroll(column![]));
        }

        let truncate_at = compute_truncation_index(
            &self.diff_files,
            self.selected_file.as_deref(),
            Some((MAX_HUNKS, MAX_DIFF_LINES)),
        );
        let mut rows: Vec<Element<'_, DiffMessage>> = Vec::new();
        let mut truncated = false;
        let mut buffer_idx = 0usize;

        for (idx, file) in self.diff_files.iter().enumerate() {
            // File selection filter
            if !file_matches_selection(file, self.selected_file.as_deref()) {
                continue;
            }

            // Truncation check — stop before the first file that would exceed caps
            if let Some(limit) = truncate_at {
                if idx >= limit {
                    truncated = true;
                    break;
                }
            }

            // File header
            let (header_label, header_icon) = if file.status == DiffFileStatus::Renamed {
                (
                    format!(
                        "Rename: {} \u{2192} {}",
                        file.old_path.as_deref().unwrap_or("?"),
                        file.path
                    ),
                    lucide::arrow_right::<iced::Theme, iced::Renderer>(),
                )
            } else if file.status == DiffFileStatus::Added {
                (
                    format!("New file: {}", file.path),
                    lucide::file_plus::<iced::Theme, iced::Renderer>(),
                )
            } else if file.status == DiffFileStatus::Deleted {
                (
                    format!("Deleted: {}", file.path),
                    lucide::file_minus::<iced::Theme, iced::Renderer>(),
                )
            } else if file.status == DiffFileStatus::Untracked {
                (
                    format!("Untracked: {}", file.path),
                    lucide::file_plus::<iced::Theme, iced::Renderer>(),
                )
            } else {
                (
                    file.path.clone(),
                    lucide::file_text::<iced::Theme, iced::Renderer>(),
                )
            };

            // The tinted band spans the wrapper's content width, so it sits at
            // the uniform 8px inset from the panel edge instead of touching it.
            rows.push(
                container(
                    row![
                        header_icon.size(theme::TEXT_12).color(FILE_HEADER_COLOR),
                        Space::new().width(theme::SPACE_6),
                        text(header_label)
                            .size(theme::TEXT_12)
                            .color(FILE_HEADER_COLOR),
                    ]
                    .align_y(Alignment::Center),
                )
                .width(Length::Fill)
                .padding([theme::PAD_6, theme::PAD_4])
                .style(move |_t: &iced::Theme| iced::widget::container::Style {
                    background: Some(iced::Background::Color(theme::DIFF_FILE_HEADER_BG)),
                    ..Default::default()
                })
                .into(),
            );

            // Binary / too-large placeholders
            match file.content {
                DiffContent::Binary => {
                    rows.push(
                        container(
                            text(format!("Binary file: {}", file.path))
                                .size(theme::TEXT_13)
                                .color(theme::TEXT_MUTED),
                        )
                        .padding([theme::PAD_2, theme::PAD_4])
                        .into(),
                    );
                    continue;
                }
                DiffContent::TooLarge(sz) => {
                    rows.push(
                        container(
                            text(format!("File too large: {}, {sz} bytes", file.path))
                                .size(theme::TEXT_13)
                                .color(theme::TEXT_MUTED),
                        )
                        .padding([theme::PAD_2, theme::PAD_4])
                        .into(),
                    );
                    continue;
                }
                DiffContent::Parseable => {}
            }

            // Find the matching buffer by index
            if let Some(buf) = self.file_buffers.get(buffer_idx) {
                buffer_idx += 1;

                rows.push(iced::Element::new(DiffBufferWidget::new(buf)));
            }
        }

        if truncated {
            rows.push(
                container(
                    text(format!(
                        "\u{26a0} Diff truncated (max {MAX_DIFF_LINES} lines / {MAX_HUNKS} hunks)",
                    ))
                    .size(theme::TEXT_12)
                    .color(theme::STATUS_WARNING),
                )
                .padding([theme::PAD_8, theme::PAD_4])
                .into(),
            );
        }

        widgets::page_bare(widgets::vscroll(
            column(rows).spacing(0).width(Length::Fill),
        ))
    }

    /// Rebuild per-file cosmic_text buffer data. Called from `update()` when
    /// `diff_files` or `selected_file` changes.
    fn rebuild_file_buffers(&mut self) {
        if self.diff_files.is_empty() {
            self.file_buffers.clear();
            return;
        }
        self.file_buffers = diff_widget::build_file_buffers(
            &self.diff_files,
            self.selected_file.as_deref(),
            Some((MAX_HUNKS, MAX_DIFF_LINES)),
        );
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Load the diff, compute per-file highlights, and return enhanced DiffFiles.
/// `ws_path` is the path the dashboard resolved for the selected workspace.
async fn load_diff(
    ws_path: Option<PathBuf>,
    commit_ref: Option<String>,
) -> Result<Vec<DiffFile>, String> {
    let Some(ws_path) = ws_path else {
        return Err("No workspace resolved.".to_string());
    };

    if !git_is_installed().await {
        return Err("Git is not installed.".to_string());
    }
    if !is_git_repo(&ws_path) {
        return Err("This workspace is not a git repository.".to_string());
    }
    if !git_has_commits(&ws_path).await {
        return Err("No commits yet \u{2014} nothing to diff against.".to_string());
    }

    let diff_output = run_git_diff(&ws_path, commit_ref.as_deref())
        .await
        .map_err(|e| format!("Failed to run git diff: {e}"))?;
    let mut parsed = parse_git_diff(&diff_output);

    // Untracked files — only relevant for working-tree diffs.
    // Historical commits don't have untracked files.
    if commit_ref.is_none() {
        add_untracked_files(&mut parsed, &ws_path).await?;
    }

    // Compute highlights for each file off the UI thread.
    let mut enhanced: Vec<DiffFile> = Vec::with_capacity(parsed.len());
    for dfile in parsed {
        let (old_hl, new_hl) = if dfile.has_parseable_content() {
            compute_highlights(&dfile, &ws_path, commit_ref.as_deref()).await
        } else {
            (None, None)
        };
        enhanced.push(DiffFile::from_parsed(dfile, old_hl, new_hl));
    }

    Ok(enhanced)
}

/// Add untracked/new files from `git status --porcelain` to the parsed diff list.
/// Only called for working-tree diffs (commit_ref is None).
async fn add_untracked_files(
    parsed: &mut Vec<crate::git::diff_parse::DiffFile>,
    ws_path: &Path,
) -> Result<(), String> {
    let status_output = run_git_status(ws_path)
        .await
        .map_err(|e| format!("Failed to run git status: {e}"))?;
    let untracked = parse_untracked_from_porcelain(&status_output);

    for path in &untracked {
        match read_untracked_file(&ws_path.join(path), MAX_UNTRACKED_SIZE).await {
            UntrackedFileRead::Text(text) => parsed.push(make_untracked_diff_file(path, &text)),
            UntrackedFileRead::TooLarge(size) => {
                parsed.push(crate::git::diff_parse::DiffFile::placeholder(
                    path.clone(),
                    DiffContent::TooLarge(size),
                ));
            }
            UntrackedFileRead::Binary => {
                parsed.push(crate::git::diff_parse::DiffFile::placeholder(
                    path.clone(),
                    DiffContent::Binary,
                ));
            }
            // Unreadable / non-file paths produce no placeholder.
            UntrackedFileRead::Skip => {}
        }
    }

    Ok(())
}

/// Count added and removed lines in a diff file.
fn count_lines(dfile: &crate::git::diff_parse::DiffFile) -> (usize, usize) {
    let mut add = 0;
    let mut remove = 0;
    for hunk in &dfile.hunks {
        for line in &hunk.lines {
            match line.kind {
                DiffLineKind::Added => add += 1,
                DiffLineKind::Removed => remove += 1,
                DiffLineKind::Context => {}
            }
        }
    }
    (add, remove)
}

/// Compute highlight spans for both old and new versions of a file.
async fn compute_highlights(
    dfile: &crate::git::diff_parse::DiffFile,
    ws_path: &Path,
    commit_ref: Option<&str>,
) -> (Option<FileHighlights>, Option<FileHighlights>) {
    let lang = HighlightLanguage::from_path(&dfile.path);
    let Some(lang) = lang else {
        return (None, None);
    };

    // Build parsers for this file (created in the async task, not on UI thread).
    // Run old and new highlight computation concurrently to overlap I/O
    // (git subprocess for old version, git/disk read for new version).
    let (old_hl, new_hl) = tokio::join!(
        compute_old_highlights(dfile, ws_path, lang, commit_ref),
        compute_new_highlights(dfile, ws_path, lang, commit_ref),
    );

    (old_hl, new_hl)
}

async fn compute_old_highlights(
    dfile: &crate::git::diff_parse::DiffFile,
    ws_path: &Path,
    lang: HighlightLanguage,
    commit_ref: Option<&str>,
) -> Option<FileHighlights> {
    // For new and untracked files, there is no old version.
    if matches!(
        dfile.status,
        DiffFileStatus::Added | DiffFileStatus::Untracked
    ) {
        return None;
    }

    let old_path = dfile.old_path.as_deref().unwrap_or(&dfile.path);
    // For historical commits, read the parent version (~1).
    // For working-tree diffs, read HEAD.
    let show_ref = commit_ref.map(|hash| format!("{hash}~1"));
    let content = run_git_show(ws_path, old_path, show_ref.as_deref()).await?;

    parse_highlights(&content, lang)
}

async fn compute_new_highlights(
    dfile: &crate::git::diff_parse::DiffFile,
    ws_path: &Path,
    lang: HighlightLanguage,
    commit_ref: Option<&str>,
) -> Option<FileHighlights> {
    // Deleted files have no new version on disk.
    if dfile.status == DiffFileStatus::Deleted {
        return None;
    }

    let content = if let Some(hash) = commit_ref {
        // Historical commit: read file content from git, NOT from disk.
        // The working-tree path may not match the commit version.
        run_git_show(ws_path, &dfile.path, Some(hash)).await?
    } else {
        // Working-tree diff: read from disk (existing behavior).
        let full_path = ws_path.join(&dfile.path);
        tokio::fs::read_to_string(&full_path).await.ok()?
    };

    parse_highlights(&content, lang)
}

/// Wrap a tree-row button in the "Discard changes" context menu; commit views
/// return the button unchanged (discarding only applies to the working tree).
fn discard_menu_button(
    btn: Element<'_, DiffMessage>,
    full_path: String,
    target: DiscardTarget,
    commit_view: bool,
) -> Element<'_, DiffMessage> {
    if commit_view {
        return btn;
    }
    ContextMenu::new(
        btn,
        vec![MenuItem::new(
            "Discard changes".into(),
            DiffMessage::DiscardPath(full_path, target),
        )],
    )
    .into()
}

/// Build a directory tree from the list of diff files.
fn build_tree(files: &[DiffFile]) -> Vec<widgets::TreeNode> {
    let mut roots: HashMap<String, widgets::TreeNode> = HashMap::new();

    for file in files {
        let path = &file.path;
        let components: Vec<&str> = path.split('/').collect();
        if components.is_empty() {
            continue;
        }

        // Ensure root-level directory exists.
        let root_name = components[0].to_string();
        let root_full = root_name.clone();
        let root_node = roots
            .entry(root_full.clone())
            .or_insert_with(|| widgets::TreeNode {
                name: root_name,
                full_path: root_full,
                is_dir: components.len() > 1,
                children: Vec::new(),
                error: None,
                ignored: false,
            });

        if components.len() == 1 {
            // This is a root-level file — convert or keep as file node.
            if root_node.is_dir {
                // A directory with the same name exists; this shouldn't happen
                // in a git repo, but handle gracefully.
                root_node.children.push(widgets::TreeNode {
                    name: components[0].to_string(),
                    full_path: path.clone(),
                    is_dir: false,
                    children: Vec::new(),
                    error: None,
                    ignored: false,
                });
            } else {
                // Already a file node — just ensure path is correct.
                root_node.full_path.clone_from(path);
                root_node.is_dir = false;
            }
        } else {
            // Multi-component path — ensure intermediate directories exist.
            root_node.is_dir = true;
            let mut current = root_node;
            for (i, comp) in components.iter().enumerate().skip(1) {
                let full = components[..=i].join("/");
                let is_last = i == components.len() - 1;
                let child = current.children.iter_mut().find(|c| c.name == *comp);
                match child {
                    Some(existing) => {
                        if is_last {
                            existing.is_dir = false;
                            existing.full_path.clone_from(path);
                        }
                    }
                    None => {
                        current.children.push(widgets::TreeNode {
                            name: (*comp).to_string(),
                            full_path: if is_last { path.clone() } else { full },
                            is_dir: !is_last,
                            children: Vec::new(),
                            error: None,
                            ignored: false,
                        });
                    }
                }
                // Move current to the child we just found/created.
                if let Some(idx) = current.children.iter().position(|c| c.name == *comp) {
                    // Safe: we just checked/added this child.
                    // Need to split the borrow — use index after the loop.
                    current = &mut current.children[idx];
                } else {
                    break;
                }
            }
        }
    }

    let mut sorted: Vec<widgets::TreeNode> = roots.into_values().collect();
    FileTree::sort_nodes(&mut sorted);
    sorted
}

/// Collect `full_path` of every directory node in the tree (recursive).
fn collect_dir_paths(nodes: &[widgets::TreeNode], paths: &mut HashSet<String>) {
    for node in nodes {
        if node.is_dir {
            paths.insert(node.full_path.clone());
            collect_dir_paths(&node.children, paths);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_file(path: &str, add: usize, remove: usize) -> DiffFile {
        DiffFile {
            dfile: crate::git::diff_parse::DiffFile::new(
                path.to_string(),
                Vec::new(),
                crate::git::diff_parse::DiffFileStatus::Modified,
            ),
            old_highlights: None,
            new_highlights: None,
            add_count: add,
            remove_count: remove,
        }
    }

    #[test]
    fn test_build_tree_single_file() {
        let files = vec![make_test_file("src/main.rs", 3, 1)];
        let tree = build_tree(&files);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].name, "src");
        assert!(tree[0].is_dir);
        assert_eq!(tree[0].children.len(), 1);
        assert_eq!(tree[0].children[0].name, "main.rs");
        assert!(!tree[0].children[0].is_dir);
    }

    #[test]
    fn test_build_tree_multiple_files_same_dir() {
        let files = vec![
            make_test_file("src/lib.rs", 5, 2),
            make_test_file("src/main.rs", 3, 1),
        ];
        let tree = build_tree(&files);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].name, "src");
        assert!(tree[0].is_dir);
        assert_eq!(tree[0].children.len(), 2);
        // Dirs sorted before files, alphabetically
        assert_eq!(tree[0].children[0].name, "lib.rs");
        assert_eq!(tree[0].children[1].name, "main.rs");
    }

    #[test]
    fn test_build_tree_root_file() {
        let files = vec![make_test_file("README.md", 0, 0)];
        let tree = build_tree(&files);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].name, "README.md");
        assert!(!tree[0].is_dir);
    }

    #[test]
    fn test_build_tree_nested() {
        let files = vec![make_test_file("src/gui/diff.rs", 10, 5)];
        let tree = build_tree(&files);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].name, "src");
        assert!(tree[0].is_dir);
        assert_eq!(tree[0].children.len(), 1);
        assert_eq!(tree[0].children[0].name, "gui");
        assert!(tree[0].children[0].is_dir);
        assert_eq!(tree[0].children[0].children.len(), 1);
        assert_eq!(tree[0].children[0].children[0].name, "diff.rs");
        assert!(!tree[0].children[0].children[0].is_dir);
    }

    // ── Tree keyboard navigation focus state tests ──────────────────

    fn make_diff_with_tree() -> DiffState {
        let mut state = DiffState::new();
        state.diff_files = vec![
            make_test_file("src/main.rs", 0, 0),
            make_test_file("src/lib.rs", 0, 0),
        ];
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        state
    }

    #[test]
    fn test_diff_rebuild_visible_tree() {
        let mut state = make_diff_with_tree();
        // Initially only root nodes visible (directories collapsed).
        assert!(!state.file_tree.visible_tree_nodes.is_empty());
        // Expand "src" and rebuild.
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        assert_eq!(
            state.file_tree.visible_tree_nodes.len(),
            3,
            "expected 3 nodes: src, src/main.rs, src/lib.rs"
        );
    }

    #[test]
    fn test_diff_tree_focus_toggled_sets_focus() {
        let mut state = make_diff_with_tree();
        assert!(!state.file_tree.tree_focused);
        let _ = state.update(DiffMessage::TreeFocusToggled);
        assert!(state.file_tree.tree_focused);
        let _ = state.update(DiffMessage::TreeFocusToggled);
        assert!(!state.file_tree.tree_focused);
    }

    #[test]
    fn test_diff_tree_focus_toggled_empty_tree_stays_off() {
        let mut state = DiffState::new();
        let _ = state.update(DiffMessage::TreeFocusToggled);
        assert!(!state.file_tree.tree_focused);
    }

    #[test]
    fn test_diff_select_file_sets_tree_focused_when_not_focused() {
        // When tree_focused starts false, clicking a file should set it true.
        let mut state = make_diff_with_tree();
        state.file_tree.tree_focused = false;
        let _ = state.update(DiffMessage::SelectFile("src/main.rs".to_owned()));
        assert!(
            state.file_tree.tree_focused,
            "SelectFile should set tree_focused to true",
        );
    }

    #[test]
    fn test_diff_visible_tree_clamps_focus() {
        let mut state = make_diff_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focus_index = 999;
        state.file_tree.rebuild_visible();
        assert_eq!(
            state.file_tree.tree_focus_index,
            state.file_tree.visible_tree_nodes.len() - 1
        );
    }

    // ── Tree arrow-key navigation tests ─────────────────────────────

    #[test]
    fn test_tree_nav_left_right() {
        struct Case {
            name: &'static str,
            start_idx: usize,
            /// Pre-expand "src" before the message
            pre_expand_src: bool,
            msg: DiffMessage,
            expected_idx: usize,
            /// Additional per-case assertions
            check: Option<fn(&DiffState, name: &str)>,
        }
        let cases: &[Case] = &[
            Case {
                name: "left_on_expanded_dir_collapses",
                start_idx: 0,
                pre_expand_src: true,
                msg: DiffMessage::TreeNavLeft,
                expected_idx: 0,
                check: Some(|s, name| {
                    assert!(!s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                }),
            },
            Case {
                name: "left_on_file_navigates_to_parent",
                start_idx: 1,
                pre_expand_src: true,
                msg: DiffMessage::TreeNavLeft,
                expected_idx: 0,
                check: Some(|s, name| {
                    assert_eq!(s.file_tree.visible_tree_nodes[0].0, "src", "case: {name}");
                }),
            },
            Case {
                name: "left_on_root_item_noop",
                start_idx: 0,
                pre_expand_src: false,
                msg: DiffMessage::TreeNavLeft,
                expected_idx: 0,
                check: None,
            },
            Case {
                name: "right_on_collapsed_dir_expands_and_advances",
                start_idx: 0,
                pre_expand_src: false,
                msg: DiffMessage::TreeNavRight,
                expected_idx: 1,
                check: Some(|s, name| {
                    assert!(s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                    assert_eq!(
                        s.file_tree.visible_tree_nodes[1].0, "src/lib.rs",
                        "case: {name}",
                    );
                }),
            },
            Case {
                name: "right_on_expanded_dir_moves_to_first_child",
                start_idx: 0,
                pre_expand_src: true,
                msg: DiffMessage::TreeNavRight,
                expected_idx: 1,
                check: Some(|s, name| {
                    assert_eq!(
                        s.file_tree.visible_tree_nodes[1].0, "src/lib.rs",
                        "case: {name}",
                    );
                }),
            },
            Case {
                name: "right_on_file_noop",
                start_idx: 1,
                pre_expand_src: true,
                msg: DiffMessage::TreeNavRight,
                expected_idx: 1,
                check: None,
            },
        ];
        for case in cases {
            let mut state = make_diff_with_tree();
            if case.pre_expand_src {
                state.file_tree.expanded_dirs.insert("src".to_string());
                state.file_tree.nodes = build_tree(&state.diff_files);
                state.file_tree.rebuild_visible();
            }
            state.file_tree.tree_focused = true;
            state.file_tree.tree_focus_index = case.start_idx;
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

    // ── TreeNavEnter tests ──────────────────────────────────────────

    #[test]
    fn test_tree_nav_enter() {
        struct Case {
            name: &'static str,
            focused: bool,
            start_idx: usize,
            /// Use DiffState::new() instead of make_diff_with_tree()
            empty_tree: bool,
            /// Pre-expand "src" before the message
            pre_expand_src: bool,
            expected_idx: usize,
            /// Additional per-case assertions
            check: Option<fn(&DiffState, name: &str)>,
        }
        let cases: &[Case] = &[
            Case {
                name: "on_collapsed_dir_expands_and_advances",
                focused: true,
                start_idx: 0,
                empty_tree: false,
                pre_expand_src: false,
                expected_idx: 1,
                check: Some(|s, name| {
                    assert!(s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                    assert_eq!(
                        s.file_tree.visible_tree_nodes[s.file_tree.tree_focus_index].0,
                        "src/lib.rs",
                        "case: {name}",
                    );
                }),
            },
            Case {
                name: "on_expanded_dir_collapses_and_keeps_focus",
                focused: true,
                start_idx: 0,
                empty_tree: false,
                pre_expand_src: true,
                expected_idx: 0,
                check: Some(|s, name| {
                    assert!(!s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                    assert_eq!(s.file_tree.visible_tree_nodes[0].0, "src", "case: {name}");
                }),
            },
            Case {
                name: "on_file_does_not_expand_or_collapse",
                focused: true,
                start_idx: 1,
                empty_tree: false,
                pre_expand_src: true,
                expected_idx: 1,
                check: Some(|s, name| {
                    assert_eq!(s.file_tree.expanded_dirs.len(), 1, "case: {name}");
                    assert!(s.file_tree.expanded_dirs.contains("src"), "case: {name}");
                    assert!(s.file_tree.tree_focused, "case: {name}");
                }),
            },
            Case {
                name: "not_focused_noop",
                focused: false,
                start_idx: 0,
                empty_tree: false,
                pre_expand_src: false,
                expected_idx: 0,
                check: Some(|s, name| {
                    assert!(!s.file_tree.tree_focused, "case: {name}");
                }),
            },
            Case {
                name: "empty_tree_noop",
                focused: true,
                start_idx: 0,
                empty_tree: true,
                pre_expand_src: false,
                expected_idx: 0,
                check: Some(|s, name| {
                    assert!(s.file_tree.visible_tree_nodes.is_empty(), "case: {name}");
                    assert!(s.file_tree.tree_focused, "case: {name}");
                }),
            },
        ];
        for case in cases {
            let mut state = if case.empty_tree {
                DiffState::new()
            } else {
                let mut s = make_diff_with_tree();
                if case.pre_expand_src {
                    s.file_tree.expanded_dirs.insert("src".to_string());
                    s.file_tree.nodes = build_tree(&s.diff_files);
                    s.file_tree.rebuild_visible();
                }
                s
            };
            state.file_tree.tree_focused = case.focused;
            state.file_tree.tree_focus_index = case.start_idx;
            let _ = state.update(DiffMessage::TreeNavEnter);
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
    fn test_diff_toggle_dir_sets_tree_focus_index() {
        let mut state = make_diff_with_tree();
        let _ = state.update(DiffMessage::ToggleDir("src".to_owned()));
        assert!(state.file_tree.tree_focused);
        assert_eq!(state.file_tree.tree_focus_index, 0);
        assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
    }

    #[test]
    fn test_diff_select_file_sets_tree_focus_index() {
        let mut state = make_diff_with_tree();
        // Expand "src" so "src/main.rs" is visible in the flat list.
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        let _ = state.update(DiffMessage::SelectFile("src/main.rs".to_owned()));
        // SelectFile keeps tree_focused and remembers focus index.
        assert!(state.file_tree.tree_focused);
        // tree_focus_index should point to "src/main.rs" for Ctrl+B re-focus.
        assert_eq!(
            state.file_tree.visible_tree_nodes[state.file_tree.tree_focus_index].0,
            "src/main.rs"
        );
    }

    // ── Discard changes tests ──────────────────────────────────────

    /// Mark `state` as showing the workspace the dashboard resolved — the diff
    /// view tracks it by its path alone.
    fn mark_workspace_resolved(state: &mut DiffState) {
        state.resolved_workspace_path = Some("/tmp/test-ws".into());
    }

    #[test]
    fn test_discard_changes() {
        struct Case {
            name: &'static str,
            setup: fn(&mut DiffState),
            msg: DiffMessage,
            check: fn(&DiffState, name: &str),
        }
        let cases: &[Case] = &[
            Case {
                name: "discard_path_noop_in_commit_view",
                // Must not mark loading in commit-view — DiscardPath is a no-op
                // when viewing a historical commit (you can't discard history).
                setup: |s| {
                    s.current_commit_ref = Some("abc1234".to_owned());
                },
                msg: DiffMessage::DiscardPath("src/main.rs".to_owned(), DiscardTarget::File),
                check: |s, name| assert!(!s.diff_loading, "case: {name}"),
            },
            Case {
                name: "discard_path_noop_without_workspace",
                // Without a resolved workspace there's nothing to discard into,
                // so DiscardPath is a no-op.
                setup: |s| {
                    s.resolved_workspace_path = None;
                },
                msg: DiffMessage::DiscardPath("src/main.rs".to_owned(), DiscardTarget::File),
                check: |s, name| assert!(!s.diff_loading, "case: {name}"),
            },
            Case {
                name: "discard_path_sets_loading",
                setup: mark_workspace_resolved,
                msg: DiffMessage::DiscardPath("src/main.rs".to_owned(), DiscardTarget::File),
                check: |s, name| assert!(s.diff_loading, "case: {name}"),
            },
            Case {
                name: "discard_result_success_no_workspace_resets_loading",
                // Without a resolved workspace the success path falls through
                // to the no-refresh branch and resets loading.
                setup: |s| {
                    s.resolved_workspace_path = None;
                    s.diff_loading = true;
                },
                msg: DiffMessage::DiscardResult(Ok(())),
                check: |s, name| assert!(!s.diff_loading, "case: {name}"),
            },
            Case {
                name: "discard_result_success_with_workspace_keeps_loading_for_refresh",
                // When there IS a resolved workspace, a successful discard
                // triggers an immediate diff refresh — diff_loading stays true
                // throughout so the UI shows a loading indicator.
                setup: |s| {
                    mark_workspace_resolved(s);
                    s.diff_loading = true;
                },
                msg: DiffMessage::DiscardResult(Ok(())),
                check: |s, name| assert!(s.diff_loading, "case: {name}"),
            },
            Case {
                name: "discard_result_error_resets_loading",
                setup: |s| {
                    mark_workspace_resolved(s);
                    s.diff_loading = true;
                },
                msg: DiffMessage::DiscardResult(Err("something went wrong".to_owned())),
                check: |s, name| assert!(!s.diff_loading, "case: {name}"),
            },
            Case {
                name: "discard_path_file_target_vs_dir_target",
                setup: mark_workspace_resolved,
                msg: DiffMessage::DiscardPath("src".to_owned(), DiscardTarget::Directory),
                check: |s, name| assert!(s.diff_loading, "case: {name}"),
            },
        ];

        for case in cases {
            let mut state = make_diff_with_tree();
            // All cases build from scratch with make_diff_with_tree, which
            // starts with diff_loading = false.
            assert!(
                !state.diff_loading,
                "case: {} — expected clean state",
                case.name
            );
            (case.setup)(&mut state);
            let _ = state.update(case.msg.clone());
            (case.check)(&state, case.name);
        }
    }

    fn make_diff_loaded(files: Vec<DiffFile>) -> DiffState {
        let mut state = DiffState::new();
        state.tree_auto_expand_pending = true;
        let _ = state.update(DiffMessage::DiffLoaded(state.generation, Ok(files)));
        state
    }

    #[test]
    fn test_tree_auto_expand_nested_on_first_load() {
        let files = vec![
            make_test_file("src/gui/diff.rs", 1, 0),
            make_test_file("src/lib.rs", 0, 1),
        ];
        let state = make_diff_loaded(files);
        assert!(state.file_tree.expanded_dirs.contains("src"));
        assert!(state.file_tree.expanded_dirs.contains("src/gui"));
        assert!(!state.tree_auto_expand_pending);
    }

    #[test]
    fn test_collapsed_tree_preserved_on_refresh() {
        let files = vec![make_test_file("src/main.rs", 1, 0)];
        let mut state = make_diff_loaded(files);
        state.file_tree.expanded_dirs.clear();
        state.tree_auto_expand_pending = false;

        let _ = state.update(DiffMessage::DiffLoaded(
            state.generation,
            Ok(vec![make_test_file("src/main.rs", 2, 0)]),
        ));

        assert!(state.file_tree.expanded_dirs.is_empty());
    }

    #[test]
    fn test_workspace_switch_resets_auto_expand() {
        let mut state = DiffState::new();
        state.tree_auto_expand_pending = false;
        let _ = state.update(DiffMessage::WorkspaceSelected("/tmp/ws".to_owned()));
        assert!(state.tree_auto_expand_pending);
        assert!(state.file_tree.expanded_dirs.is_empty());
    }

    #[test]
    fn test_commit_message_clears_tree_focus() {
        let mut state = make_diff_with_tree();
        state.file_tree.tree_focused = true;
        let _ = state.update(DiffMessage::CommitMessageChanged(
            crate::gui::editor_widget::EditorAction::Paste("fix bug".to_owned()),
        ));
        assert!(!state.file_tree.tree_focused);
    }

    // ── compute_truncation_index tests ─────────────────────────────

    fn make_file_with_hunks(path: &str, num_hunks: usize, lines_per_hunk: usize) -> DiffFile {
        let hunks = (0..num_hunks)
            .map(|_i| crate::git::diff_parse::DiffHunk {
                header: format!("@@ -1,{lines_per_hunk} +1,{lines_per_hunk} @@"),
                lines: (0..lines_per_hunk)
                    .map(|_| crate::git::diff_parse::DiffLine {
                        kind: crate::git::diff_parse::DiffLineKind::Added,
                        old_line_number: None,
                        new_line_number: Some(1),
                        content: String::new(),
                    })
                    .collect(),
            })
            .collect();

        DiffFile::from_parsed(
            crate::git::diff_parse::DiffFile::new(
                path.to_string(),
                hunks,
                crate::git::diff_parse::DiffFileStatus::Modified,
            ),
            None,
            None,
        )
    }

    fn make_binary_file(path: &str) -> DiffFile {
        DiffFile::from_parsed(
            crate::git::diff_parse::DiffFile {
                path: path.to_string(),
                old_path: None,
                hunks: Vec::new(),
                status: crate::git::diff_parse::DiffFileStatus::Modified,
                content: DiffContent::Binary,
            },
            None,
            None,
        )
    }

    fn make_too_large_file(path: &str) -> DiffFile {
        DiffFile::from_parsed(
            crate::git::diff_parse::DiffFile {
                path: path.to_string(),
                old_path: None,
                hunks: Vec::new(),
                status: crate::git::diff_parse::DiffFileStatus::Modified,
                content: DiffContent::TooLarge(5_000_000),
            },
            None,
            None,
        )
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn test_truncation_index() {
        struct Case {
            name: &'static str,
            files: Vec<DiffFile>,
            selected_file: Option<&'static str>,
            limits: Option<(usize, usize)>,
            expected: Option<usize>,
        }

        let cases: &[Case] = &[
            Case {
                name: "no_limits",
                files: vec![make_file_with_hunks("a.rs", 200, 200)],
                selected_file: None,
                limits: None,
                expected: None,
            },
            Case {
                name: "empty_slice",
                files: Vec::new(),
                selected_file: None,
                limits: Some((100, 5000)),
                expected: None,
            },
            Case {
                name: "all_files_fit — 60 hunks, 2400 lines, both under limits",
                files: vec![
                    make_file_with_hunks("a.rs", 30, 40),
                    make_file_with_hunks("b.rs", 30, 40),
                ],
                selected_file: None,
                limits: Some((100, 5000)),
                expected: None,
            },
            Case {
                name: "hunk_cap — a has 60 hunks, adding b makes 110 > 100",
                files: vec![
                    make_file_with_hunks("a.rs", 60, 10),
                    make_file_with_hunks("b.rs", 50, 10),
                ],
                selected_file: None,
                limits: Some((100, 5000)),
                expected: Some(1),
            },
            Case {
                name: "line_cap — a+b=2000 lines, adding c makes 7000 > 5000",
                files: vec![
                    make_file_with_hunks("a.rs", 10, 100),
                    make_file_with_hunks("b.rs", 10, 100),
                    make_file_with_hunks("c.rs", 10, 500),
                ],
                selected_file: None,
                limits: Some((100, 5000)),
                expected: Some(2),
            },
            Case {
                name: "binary_files_skipped — binary at idx=0 consumes no capacity",
                files: vec![
                    make_binary_file("binary.bin"),
                    make_file_with_hunks("a.rs", 60, 10),
                    make_file_with_hunks("b.rs", 50, 10),
                ],
                selected_file: None,
                limits: Some((100, 5000)),
                expected: Some(2),
            },
            Case {
                name: "selected_file_filter — only b.rs passes, 10 hunks 100 lines within cap",
                files: vec![
                    make_file_with_hunks("a.rs", 80, 50),
                    make_file_with_hunks("b.rs", 10, 10),
                    make_file_with_hunks("c.rs", 80, 50),
                ],
                selected_file: Some("b.rs"),
                limits: Some((100, 5000)),
                expected: None,
            },
            Case {
                name: "selected_file_exceeds_cap — b.rs has 6000 lines > 5000",
                files: vec![make_file_with_hunks("b.rs", 60, 100)],
                selected_file: Some("b.rs"),
                limits: Some((100, 5000)),
                expected: Some(0),
            },
            Case {
                name: "too_large_files_skipped — large.bin skipped, a.rs+b.rs within cap",
                files: vec![
                    make_too_large_file("large.bin"),
                    make_file_with_hunks("a.rs", 40, 30),
                    make_file_with_hunks("b.rs", 40, 30),
                ],
                selected_file: None,
                limits: Some((100, 5000)),
                expected: None,
            },
        ];

        for case in cases {
            assert_eq!(
                compute_truncation_index(&case.files, case.selected_file, case.limits),
                case.expected,
                "case: {}",
                case.name,
            );
        }
    }

    // ── Stale-data regression tests ───────────────────────────────────
    //
    // Verify that context-switching message handlers clear all state
    // synchronously (before the async load task runs), preventing stale data
    // from rendering during the loading transition.

    /// Shared helper: verify that all fields managed by [`DiffState::clear_diff_state`]
    /// have been reset to their cleared values.
    fn assert_clear_diff_state(state: &DiffState) {
        assert!(
            state.diff_files.is_empty(),
            "diff_files should be cleared synchronously to avoid stale data"
        );
        assert!(
            state.error.is_none(),
            "error should be cleared to prevent stale error banner"
        );
        assert!(
            state.status_message.is_none(),
            "status_message should be cleared to prevent stale status text"
        );
        assert!(
            state.commit_message.text().is_empty(),
            "commit_message should be cleared to prevent stale typed text"
        );
        assert!(!state.committing, "committing should be reset to false");
        assert!(
            state.file_tree.nodes.is_empty(),
            "file_tree.nodes should be cleared to prevent stale tree entries"
        );
        assert!(
            state.file_tree.expanded_dirs.is_empty(),
            "file_tree.expanded_dirs should be cleared"
        );
        assert!(
            state.selected_file.is_none(),
            "selected_file should be cleared to prevent stale file filter"
        );
        assert!(
            state.file_buffers.is_empty(),
            "file_buffers should be cleared to prevent stale buffer data"
        );
        assert!(
            state.tree_auto_expand_pending,
            "tree_auto_expand_pending should be true to trigger expansion on next DiffLoaded"
        );
        assert!(
            state.current_commit_message.is_none(),
            "current_commit_message should be cleared to prevent stale commit message"
        );
    }

    /// Shared helper: verify that a context switch has reset all fields that
    /// could render stale data during the async loading window.
    fn assert_diff_state_reset(state: &DiffState) {
        assert_clear_diff_state(state);
        assert!(
            state.diff_loading,
            "diff_loading should be set before async load"
        );
        assert!(
            !state.diff_has_loaded,
            "diff_has_loaded should be false to enable loading guard"
        );
    }

    /// Create a `DiffState` populated with old/stale data to verify that each
    /// context-switching handler resets it before the async load runs.
    fn make_state_with_stale_data() -> DiffState {
        let mut state = make_diff_with_tree();
        assert!(!state.diff_files.is_empty());
        assert!(!state.file_tree.nodes.is_empty());
        state.diff_has_loaded = true;
        state.error = Some("stale error".into());
        state.status_message = Some("stale status".into());
        state.commit_message.set_text("stale commit");
        state.committing = true;
        state.selected_file = Some("stale/path.rs".into());
        state.file_buffers.push(DiffFileBuffer {
            text: "stale buffer".into(),
            span_data: Vec::new(),
            line_kinds: Vec::new(),
            line_numbers: Vec::new(),
            gutter_digits: 0,
            content_fingerprint: 0,
        });
        state.file_tree.expanded_dirs.insert("stale/dir".into());
        state.tree_auto_expand_pending = false;
        state.current_commit_ref = Some("stale-hash".into());
        state.current_commit_message = Some("stale message".into());
        state
    }

    #[test]
    fn test_context_switches_clear_stale_diff_files() {
        // Navigating to a commit clears stale working-tree diff state; the
        // workspace itself is kept, so set one first.
        let mut state = make_state_with_stale_data();
        mark_workspace_resolved(&mut state);
        let _task = state.update(DiffMessage::NavigateToCommit("abc123".into()));
        assert_diff_state_reset(&state);

        // Selecting a workspace clears stale diff state.
        let mut state = make_state_with_stale_data();
        let _task = state.update(DiffMessage::WorkspaceSelected("/p/new-ws".into()));
        assert_diff_state_reset(&state);

        // Returning to the working tree clears stale commit-view state; the
        // handler early-returns without a resolved workspace, so set one.
        let mut state = make_state_with_stale_data();
        mark_workspace_resolved(&mut state);
        let _task = state.update(DiffMessage::BackToWorkingTree);
        assert_diff_state_reset(&state);
    }

    #[test]
    fn test_workspace_unresolved_clears_stale_diff_files() {
        let mut state = make_state_with_stale_data();
        mark_workspace_resolved(&mut state);
        let stale_generation = state.generation;
        let _task = state.update(DiffMessage::WorkspaceUnresolved);
        // The clear calls clear_diff_state(), drops the resolved workspace and
        // invalidates any load still in flight.
        assert_clear_diff_state(&state);
        assert!(
            state.resolved_workspace_path.is_none(),
            "the resolved workspace path should be cleared to prevent stale workspace context"
        );
        assert_ne!(
            state.generation, stale_generation,
            "an in-flight load must not apply its result over the cleared workspace"
        );
    }

    // ── add_untracked_files — classifier + placeholder emission ──

    /// Verify the untracked-file classifier mapping: text files become parseable
    /// entries, null-byte and invalid-UTF-8 files become Binary placeholders
    /// (the latter must NOT be dropped), oversized files become TooLarge with
    /// their real size, and directories are skipped entirely.
    #[tokio::test]
    async fn test_add_untracked_files_classification() {
        let (_dir, repo_path) = crate::util::test::init_temp_repo();
        std::fs::write(repo_path.join("new.txt"), b"line1\nline2\n").unwrap();
        std::fs::write(repo_path.join("binary.bin"), b"a\x00b").unwrap();
        // Invalid UTF-8 without a null byte — Binary, not skipped.
        std::fs::write(repo_path.join("invalid.bin"), [0xC3, 0x28]).unwrap();
        // Oversized — gated before read, placeholder carries the real size.
        let big_size = usize::try_from(MAX_UNTRACKED_SIZE).unwrap() + 1;
        let mut big = vec![b'a'; big_size];
        big[big_size - 1] = b'\n';
        std::fs::write(repo_path.join("big.txt"), &big).unwrap();
        // Directory (with contents, so porcelain emits `?? new_dir/`) — the
        // classifier's is_file() gate skips it, no placeholder.
        std::fs::create_dir(repo_path.join("new_dir")).unwrap();
        std::fs::write(repo_path.join("new_dir/inner.txt"), b"x").unwrap();

        let mut parsed = Vec::new();
        add_untracked_files(&mut parsed, &repo_path).await.unwrap();

        let find = |path: &str| parsed.iter().find(|f| f.path == path);
        assert_eq!(find("new.txt").unwrap().content, DiffContent::Parseable);
        assert_eq!(find("binary.bin").unwrap().content, DiffContent::Binary);
        assert_eq!(find("invalid.bin").unwrap().content, DiffContent::Binary);
        assert_eq!(
            find("big.txt").unwrap().content,
            DiffContent::TooLarge(big_size as u64)
        );
        // Porcelain lists non-empty untracked dirs as `?? new_dir/` — the
        // parsed entry is skipped, so no placeholder with that path exists.
        assert!(find("new_dir/").is_none(), "directories must be skipped");
    }
}
