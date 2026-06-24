//! Diff dashboard page — view git diff against HEAD with syntax highlighting.
//!
//! Shows staged + unstaged changes via `git diff HEAD`, untracked files
//! via `git status --porcelain`, with per-file tree-sitter syntax highlighting.
//! Files are parsed in their entirety (old version from HEAD, new version from
//! disk) for correct multi-line token coloring.
//!
//! The page layout splits into a directory tree sidebar (left, ~25%) and a
//! scrollable diff panel (right, ~75%). Click a file in the tree to filter
//! the diff to just that file; click again to show all files.
//!
//! Auto-refreshes every 5 seconds when a workspace is selected.
use super::diff_widget::{self, ADDED_COLOR, DiffBufferWidget, DiffFileBuffer, REMOVED_COLOR};
use super::highlight::{FileHighlights, HighlightLanguage, parse_file_highlights};
use super::text_rendering::MAX_HIGHLIGHT_SIZE;

use crate::diff_parse::{
    CommitInfo, DiffFileStatus, DiffLineKind, git_has_commits, git_is_installed, is_git_repo,
    make_untracked_diff_file, parse_git_diff, run_git_command, run_git_commit, run_git_diff,
    run_git_show, run_git_status,
};

use iced::widget::Id;
use iced::widget::{Space, button, column, container, row, scrollable, text, text_input};
use iced::{Alignment, Color, Element, Length, Task, keyboard};

use iced_fonts::lucide;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Duration;

use super::context_menu::ContextMenu;
use super::theme;
use super::widgets::{self, FileTree};

const MAX_DIFF_LINES: usize = 5000;
const MAX_HUNKS: usize = 100;
const MAX_UNTRACKED_SIZE: u64 = 1024 * 1024;

const FILE_HEADER_COLOR: Color = theme::STATUS_WARNING;
const RENAME_COLOR: Color = theme::ACCENT_LIGHT;

/// Whether to discard changes in a single file or an entire directory tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscardTarget {
    File,
    /// Recursively discard all changes within a directory (and its subdirectories).
    Directory,
}

#[derive(Debug, Clone)]
pub enum DiffMessage {
    /// workspace name and optional filesystem path override (used for personal workspaces
    /// that don't exist in workspaces.db).
    WorkspaceSelected(String, Option<String>),
    DiffLoaded(u64, Result<Vec<DiffFile>, String>),
    Tick,
    /// Escape key — dismiss tree focus or exit.
    Escape,
    ToggleDir(String),
    SelectFile(String),
    CommitMessageChanged(String),
    CommitClicked,
    CommitResult(Result<CommitInfo, String>),
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
    /// Navigate to a specific commit diff view.
    /// (workspace_name, commit_hash)
    NavigateToCommit(String, String),
    /// Return from historical commit view to working tree diff.
    BackToWorkingTree,
    /// Discard changes for a file or directory (path, target).
    DiscardPath(String, DiscardTarget),
    /// Result of a discard operation.
    DiscardResult(Result<(), String>),
}

/// A single changed file, enhanced with highlight data and line counts.
#[derive(Debug, Clone)]
pub struct DiffFile {
    pub dfile: crate::diff_parse::DiffFile,
    /// Per-line highlight spans for removed lines (from old version).
    pub old_highlights: Option<FileHighlights>,
    /// Per-line highlight spans for added/context lines (from new version).
    pub new_highlights: Option<FileHighlights>,
    /// Count of added lines across all hunks.
    pub add_count: usize,
    /// Count of removed lines across all hunks.
    pub remove_count: usize,
}

/// Icon identifier for file headers (avoids widget construction at cache time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedIcon {
    ArrowRight,
    FilePlus,
    FileMinus,
    FileText,
}

impl CachedIcon {
    fn to_text<'a>(self) -> iced::widget::Text<'a, iced::Theme, iced::Renderer> {
        match self {
            CachedIcon::ArrowRight => lucide::arrow_right(),
            CachedIcon::FilePlus => lucide::file_plus(),
            CachedIcon::FileMinus => lucide::file_minus(),
            CachedIcon::FileText => lucide::file_text(),
        }
    }
}

pub struct DiffState {
    error: Option<String>,
    selected_workspace_name: Option<String>,
    /// Filesystem path when the workspace is a personal workspace
    /// (not registered in workspaces.db). `None` for shared workspaces.
    personal_workspace_path: Option<String>,
    generation: u64,
    diff_files: Vec<DiffFile>,
    diff_empty: bool,
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
    /// Generation at which file_buffers were built (compared against self.generation).
    buffers_generation: u64,
    /// Commit message typed by the user.
    commit_message: String,
    /// Whether a commit is in-flight.
    committing: bool,
    /// Current commit being viewed, if any.
    /// `None` means we're viewing the working-tree diff (`git diff HEAD`).
    current_commit_ref: Option<String>,
    /// When true, the next successful [`DiffMessage::DiffLoaded`] recursively
    /// expands all directory nodes in the file tree (nested folders included).
    /// Cleared after expansion. Not set on periodic auto-refresh ticks.
    tree_auto_expand_pending: bool,
}

impl DiffState {
    pub fn new() -> Self {
        Self {
            error: None,
            selected_workspace_name: None,
            personal_workspace_path: None,
            generation: 0,
            diff_files: Vec::new(),
            diff_empty: true,
            diff_loading: false,
            diff_has_loaded: false,
            status_message: None,
            file_tree: FileTree::new(Id::new("diff_tree_panel")),
            selected_file: None,
            file_buffers: Vec::new(),
            buffers_generation: 0,
            commit_message: String::new(),
            committing: false,
            current_commit_ref: None,
            tree_auto_expand_pending: false,
        }
    }

    pub fn subscription(&self) -> iced::Subscription<DiffMessage> {
        let mut subs: Vec<iced::Subscription<DiffMessage>> = Vec::new();
        if self.selected_workspace_name.is_some() || self.personal_workspace_path.is_some() {
            subs.push(iced::time::every(Duration::from_secs(5)).map(|_| DiffMessage::Tick));
        }
        // Keyboard: Ctrl+B toggles tree focus; arrow/Enter messages are ignored
        // by update handlers unless the tree is currently focused. This closure
        // must stay non-capturing because iced validates that in release builds.
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
            let is_cmd = modifiers.command();
            // On non-macOS, AltGr (Ctrl+Alt) is character input — block
            // shortcuts from firing.
            #[cfg(not(target_os = "macos"))]
            let altgr_active = modifiers.alt() && modifiers.control();
            #[cfg(target_os = "macos")]
            let altgr_active = false;
            if !altgr_active && is_cmd && key.to_latin(physical_key) == Some('b') {
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

    pub fn update(&mut self, msg: DiffMessage) -> Task<DiffMessage> {
        match msg {
            DiffMessage::WorkspaceSelected(name, path_override) => {
                // Accept personal workspaces when a path is provided.
                if name.is_empty() && path_override.is_none() {
                    self.selected_workspace_name = None;
                    self.personal_workspace_path = None;
                    self.diff_files = Vec::new();
                    self.diff_empty = true;
                    self.file_tree.nodes = Vec::new();
                    self.file_tree.expanded_dirs.clear();
                    self.selected_file = None;
                    self.error = None;
                    self.status_message = None;
                    self.commit_message.clear();
                    self.committing = false;
                    return Task::none();
                }
                self.file_tree.expanded_dirs.clear();
                self.tree_auto_expand_pending = true;
                self.commit_message.clear();
                self.committing = false;
                self.selected_workspace_name = Some(name.clone());
                self.personal_workspace_path.clone_from(&path_override);
                self.current_commit_ref = None;
                self.generation = self.generation.wrapping_add(1);
                let generation_num = self.generation;
                let workspace_name = name.clone();
                let ws_path = path_override.clone();
                self.diff_loading = true;
                self.selected_file = None;
                Task::perform(load_diff(workspace_name, ws_path, None), move |r| {
                    DiffMessage::DiffLoaded(generation_num, r)
                })
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
                        self.diff_empty = files.is_empty();
                        if self.diff_empty {
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
                            if !files.iter().any(|f| f.dfile.path == *sel) {
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
                        self.diff_empty = true;
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
                if let Some(ref ws_name) = self.selected_workspace_name.clone() {
                    if !self.diff_loading && !self.committing {
                        self.diff_loading = true;
                        self.generation = self.generation.wrapping_add(1);
                        let generation_num = self.generation;
                        let workspace_name = ws_name.clone();
                        let ws_path = self.personal_workspace_path.clone();
                        Task::perform(load_diff(workspace_name, ws_path, None), move |r| {
                            DiffMessage::DiffLoaded(generation_num, r)
                        })
                    } else {
                        Task::none()
                    }
                } else {
                    Task::none()
                }
            }
            DiffMessage::NavigateToCommit(ws_name, hash) => {
                self.selected_workspace_name = Some(ws_name.clone());
                self.personal_workspace_path = None;
                self.error = None;
                self.file_tree.expanded_dirs.clear();
                self.tree_auto_expand_pending = true;
                self.commit_message.clear();
                self.committing = false;
                self.selected_file = None;
                self.file_buffers.clear();
                // Set commit ref and loading BEFORE spawning task
                // (prevents Tick race: subscription checks .is_some() to skip).
                self.current_commit_ref = Some(hash.clone());
                self.diff_loading = true;
                self.diff_has_loaded = false;
                self.generation = self.generation.wrapping_add(1);
                let generation_num = self.generation;
                Task::perform(load_diff(ws_name, None, Some(hash)), move |result| {
                    DiffMessage::DiffLoaded(generation_num, result)
                })
            }
            DiffMessage::BackToWorkingTree => {
                let ws_name = match &self.selected_workspace_name {
                    Some(n) => n.clone(),
                    None => return Task::none(),
                };
                // Set loading BEFORE clearing ref (prevents Tick race).
                self.diff_loading = true;
                self.current_commit_ref = None;
                self.file_tree.expanded_dirs.clear();
                self.tree_auto_expand_pending = true;
                self.generation = self.generation.wrapping_add(1);
                let generation_num = self.generation;
                let ws_path = self.personal_workspace_path.clone();
                Task::perform(load_diff(ws_name, ws_path, None), move |result| {
                    DiffMessage::DiffLoaded(generation_num, result)
                })
            }
            DiffMessage::ToggleDir(path) => {
                self.file_tree.tree_focused = true;
                let path_clone = path.clone();
                if self.file_tree.expanded_dirs.contains(&path_clone) {
                    self.file_tree.expanded_dirs.remove(&path_clone);
                } else {
                    self.file_tree.expanded_dirs.insert(path_clone);
                }
                // Rebuild tree with updated expanded state.
                self.file_tree.nodes = build_tree(&self.diff_files);
                self.file_tree.rebuild_visible();
                // Place focus on the toggled directory.
                self.file_tree.focus_path(&path);
                Task::none()
            }
            DiffMessage::SelectFile(path) => {
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
            DiffMessage::CommitMessageChanged(msg) => {
                self.commit_message = msg;
                self.file_tree.tree_focused = false;
                Task::none()
            }
            DiffMessage::CommitClicked => {
                let trimmed = self.commit_message.trim().to_string();
                if trimmed.is_empty() || self.committing {
                    return Task::none();
                }
                self.committing = true;
                let ws_name = self
                    .selected_workspace_name
                    .as_deref()
                    .unwrap_or_default()
                    .to_string();
                let ws_path_for_commit = self.personal_workspace_path.clone();
                Task::perform(
                    async move {
                        let ws_path_buf =
                            resolve_workspace_path(&ws_name, ws_path_for_commit).await?;
                        run_git_commit(&ws_path_buf, &trimmed).await
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
                                format!("Committed {}", &info.hash.get(..7).unwrap_or(&info.hash))
                            }
                            (a, 0) => format!(
                                "Committed {} (+{a})",
                                &info.hash.get(..7).unwrap_or(&info.hash)
                            ),
                            (0, r) => format!(
                                "Committed {} (-{r})",
                                &info.hash.get(..7).unwrap_or(&info.hash)
                            ),
                            (a, r) => format!(
                                "Committed {} (+{a}/-{r})",
                                &info.hash.get(..7).unwrap_or(&info.hash)
                            ),
                        };
                        // Immediately refresh the diff.
                        if let Some(ref ws_name) = self.selected_workspace_name.clone() {
                            self.diff_loading = true;
                            self.generation = self.generation.wrapping_add(1);
                            let generation_num = self.generation;
                            let workspace_name = ws_name.clone();
                            let ws_path = self.personal_workspace_path.clone();
                            return Task::batch([
                                Task::done(DiffMessage::Toast(super::ToastMessage::SuccessMsg(
                                    toast_msg,
                                ))),
                                Task::perform(load_diff(workspace_name, ws_path, None), move |r| {
                                    DiffMessage::DiffLoaded(generation_num, r)
                                }),
                            ]);
                        }
                        Task::done(DiffMessage::Toast(super::ToastMessage::SuccessMsg(
                            toast_msg,
                        )))
                    }
                    Err(e) => {
                        self.error = Some(e);
                        Task::none()
                    }
                }
            }
            DiffMessage::Toast(_) => Task::none(),

            DiffMessage::DiscardPath(path, target) => {
                // Guard: no discard from historical commit view.
                if self.current_commit_ref.is_some() {
                    return Task::none();
                }
                let Some(ws_name) = self.selected_workspace_name.clone() else {
                    return Task::none();
                };
                let ws_path = self.personal_workspace_path.clone();
                // Mark as loading so the user sees the diff update.
                self.diff_loading = true;
                Task::perform(
                    async move {
                        let ws_path_buf = resolve_workspace_path(&ws_name, ws_path).await?;

                        // Three-step git command sequence to handle ALL file states:
                        //
                        // 1. git checkout HEAD -- <path>
                        //    — restores tracked files from HEAD (handles Modified, Deleted,
                        //      Renamed). For files staged in the index (Added) that don't
                        //      exist in HEAD, checkout removes them from both index and
                        //      working tree. Untracked files fail with "did not match" —
                        //      absorbed below.
                        //
                        // 2. git reset HEAD -- <path>
                        //    — unstages staged new (Added) files so that `git clean` can
                        //      remove them. For files already restored by checkout this is
                        //      a no-op. Errors (e.g. untracked files not in index) are
                        //      absorbed.
                        //
                        // 3. git clean -f[d] -- <path>
                        //    — removes untracked files. -f for files, -fd for directories
                        //      (recurses into subdirectories). Errors (file already tracked,
                        //      already removed by checkout) are absorbed.
                        //
                        // All errors from all three steps are absorbed. After the sequence,
                        // we verify via `git status --porcelain -- <path>`: if empty, success;
                        // otherwise report the remaining changes.
                        //
                        // Safety: `git reset HEAD` also exists with non-zero status when the
                        // path is outside the repository — we rely on the workspace-bound
                        // path validation to prevent this.

                        let _ =
                            run_git_command(&ws_path_buf, &["checkout", "HEAD", "--", &path]).await;

                        let _ =
                            run_git_command(&ws_path_buf, &["reset", "HEAD", "--", &path]).await;

                        let clean_args: &[&str] = match target {
                            DiscardTarget::Directory => &["clean", "-fd", "--", &path],
                            DiscardTarget::File => &["clean", "-f", "--", &path],
                        };
                        let _ = run_git_command(&ws_path_buf, clean_args).await;

                        // Verify: check if any changes remain.
                        match run_git_command(&ws_path_buf, &["status", "--porcelain", "--", &path])
                            .await
                        {
                            Ok(status) if status.trim().is_empty() => Ok(()),
                            Ok(status) => {
                                Err(format!("Changes remain after discard:\n{}", status.trim()))
                            }
                            Err(e) => Err(format!("Discard ran but verification failed: {e}")),
                        }
                    },
                    DiffMessage::DiscardResult,
                )
            }

            DiffMessage::DiscardResult(result) => {
                match result {
                    Ok(()) => {
                        // Refresh the diff immediately.
                        if let Some(ref ws_name) = self.selected_workspace_name.clone() {
                            self.diff_loading = true;
                            self.generation = self.generation.wrapping_add(1);
                            let generation_num = self.generation;
                            let workspace_name = ws_name.clone();
                            let ws_path = self.personal_workspace_path.clone();
                            return Task::batch([
                                Task::done(DiffMessage::Toast(super::ToastMessage::SuccessMsg(
                                    "Changes discarded.".to_string(),
                                ))),
                                Task::perform(load_diff(workspace_name, ws_path, None), move |r| {
                                    DiffMessage::DiffLoaded(generation_num, r)
                                }),
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

            DiffMessage::Escape => {
                if self.file_tree.tree_focused {
                    self.file_tree.tree_focused = false;
                }
                Task::none()
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

            DiffMessage::TreeNavUp => {
                if self.file_tree.tree_focused && self.file_tree.tree_focus_index > 0 {
                    self.file_tree.tree_focus_index -= 1;
                    return widgets::scroll_to_tree_focus(&self.file_tree);
                }
                Task::none()
            }

            DiffMessage::TreeNavDown => {
                if self.file_tree.tree_focused
                    && self.file_tree.tree_focus_index + 1 < self.file_tree.visible_tree_nodes.len()
                {
                    self.file_tree.tree_focus_index += 1;
                    return widgets::scroll_to_tree_focus(&self.file_tree);
                }
                Task::none()
            }

            DiffMessage::TreeNavEnter => {
                if !self.file_tree.tree_focused || self.file_tree.visible_tree_nodes.is_empty() {
                    return Task::none();
                }
                let idx = self
                    .file_tree
                    .tree_focus_index
                    .min(self.file_tree.visible_tree_nodes.len() - 1);
                let (ref path, is_dir) = self.file_tree.visible_tree_nodes[idx];
                if is_dir {
                    // Expand directory and move focus to the first child.
                    if self.file_tree.expanded_dirs.contains(path) {
                        self.file_tree.expanded_dirs.remove(path);
                        self.file_tree.nodes = build_tree(&self.diff_files);
                        self.file_tree.rebuild_visible();
                        return Task::none();
                    }
                    self.file_tree.expanded_dirs.insert(path.clone());
                    self.file_tree.nodes = build_tree(&self.diff_files);
                    self.file_tree.rebuild_visible();
                    // Move focus to first child (right after the directory).
                    if idx + 1 < self.file_tree.visible_tree_nodes.len() {
                        self.file_tree.tree_focus_index = idx + 1;
                        return widgets::scroll_to_tree_focus(&self.file_tree);
                    }
                    Task::none()
                } else {
                    // Open file.
                    Task::done(DiffMessage::SelectFile(path.clone()))
                }
            }

            DiffMessage::TreeNavLeft => {
                if !self.file_tree.tree_focused || self.file_tree.visible_tree_nodes.is_empty() {
                    return Task::none();
                }
                let idx = self
                    .file_tree
                    .tree_focus_index
                    .min(self.file_tree.visible_tree_nodes.len() - 1);
                let path = self.file_tree.visible_tree_nodes[idx].0.clone();
                let is_dir = self.file_tree.visible_tree_nodes[idx].1;

                if is_dir && self.file_tree.expanded_dirs.contains(&path) {
                    // Collapse expanded directory and keep focus on it.
                    self.file_tree.expanded_dirs.remove(&path);
                    self.file_tree.nodes = build_tree(&self.diff_files);
                    self.file_tree.rebuild_visible();
                    if self.file_tree.focus_path(&path).is_some() {
                        return widgets::scroll_to_tree_focus(&self.file_tree);
                    }
                    return Task::none();
                }

                // ArrowLeft on collapsed directory or file — navigate to parent.
                let parent = Path::new(&path)
                    .parent()
                    .map(|p| p.to_string_lossy().to_string());
                match parent {
                    Some(ref p) if !p.is_empty() && self.file_tree.focus_path(p).is_some() => {
                        return widgets::scroll_to_tree_focus(&self.file_tree);
                    }
                    _ => {} // Root-level item has no parent — no-op.
                }
                Task::none()
            }

            DiffMessage::TreeNavRight => {
                if !self.file_tree.tree_focused || self.file_tree.visible_tree_nodes.is_empty() {
                    return Task::none();
                }
                let idx = self
                    .file_tree
                    .tree_focus_index
                    .min(self.file_tree.visible_tree_nodes.len() - 1);
                let path = self.file_tree.visible_tree_nodes[idx].0.clone();
                let is_dir = self.file_tree.visible_tree_nodes[idx].1;

                if !is_dir {
                    // ArrowRight on a file does nothing.
                    return Task::none();
                }

                if !self.file_tree.expanded_dirs.contains(&path) {
                    // Expand directory.
                    self.file_tree.expanded_dirs.insert(path.clone());
                    self.file_tree.nodes = build_tree(&self.diff_files);
                    self.file_tree.rebuild_visible();
                    // Move focus to first child (right after the directory).
                    if let Some(dir_idx) = self.file_tree.focus_path(&path) {
                        if dir_idx + 1 < self.file_tree.visible_tree_nodes.len() {
                            self.file_tree.tree_focus_index = dir_idx + 1;
                            return widgets::scroll_to_tree_focus(&self.file_tree);
                        }
                    }
                    return Task::none();
                }

                // Already expanded directory — move focus to first child (if any).
                if idx + 1 < self.file_tree.visible_tree_nodes.len() {
                    self.file_tree.tree_focus_index = idx + 1;
                    return widgets::scroll_to_tree_focus(&self.file_tree);
                }
                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<'_, DiffMessage> {
        let has_changes = !self.diff_files.is_empty();
        let show_commit_bar =
            self.current_commit_ref.is_none() && has_changes && self.error.is_none();

        // Build the header row: commit controls or commit-view banner.
        let header = if let Some(ref hash) = self.current_commit_ref {
            // Historical commit view — show banner with Back button.
            let short_hash: String = hash.chars().take(7).collect();
            let back_btn = button("Back to working tree")
                .on_press(DiffMessage::BackToWorkingTree)
                .style(theme::button_secondary);
            container(
                row![
                    text(format!("Viewing commit {short_hash}"))
                        .size(12)
                        .color(theme::TEXT_SECONDARY),
                    Space::new().width(Length::Fill),
                    back_btn,
                ]
                .align_y(Alignment::Center),
            )
            .width(Length::Fill)
            .padding([8, 12])
        } else if show_commit_bar {
            let commit_input = text_input("Commit message…", &self.commit_message)
                .on_input(DiffMessage::CommitMessageChanged)
                .on_submit(DiffMessage::CommitClicked)
                .width(Length::Fill)
                .size(12);

            let msg_empty = self.commit_message.trim().is_empty();
            let commit_disabled = msg_empty || self.committing;

            let commit_btn = button(if self.committing {
                text("Committing…").size(12)
            } else {
                text("Commit").size(12)
            })
            .on_press_maybe(if commit_disabled {
                None
            } else {
                Some(DiffMessage::CommitClicked)
            })
            .style(if commit_disabled {
                theme::button_secondary
            } else {
                theme::button_primary
            });

            container(
                row![commit_input, Space::new().width(8), commit_btn].align_y(Alignment::Center),
            )
            .width(Length::Fill)
            .padding([8, 12])
        } else {
            // Empty spacer to maintain layout consistency.
            container(Space::new().width(Length::Fill).height(0))
        };

        let status: Element<'_, DiffMessage> = if let Some(ref err) = self.error {
            widgets::error_banner(err)
        } else if let Some(ref s) = self.status_message {
            container(text(s).size(13).color(theme::TEXT_SECONDARY))
                .padding([8, 12])
                .width(Length::Fill)
                .height(Length::Fill)
                .style(|_t: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::BG_BASE)),
                    ..Default::default()
                })
                .into()
        } else if self.diff_loading && !self.diff_has_loaded {
            container(text("Loading diff…").size(12).color(theme::TEXT_MUTED))
                .padding([8, 12])
                .width(Length::Fill)
                .height(Length::Fill)
                .style(|_t: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::BG_BASE)),
                    ..Default::default()
                })
                .into()
        } else if self.selected_workspace_name.is_none() {
            container(
                text("Select a workspace to view its diff.")
                    .size(13)
                    .color(theme::TEXT_MUTED),
            )
            .padding([8, 12])
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_t: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..Default::default()
            })
            .into()
        } else if self.diff_empty {
            container(
                column![
                    lucide::check::<iced::Theme, iced::Renderer>()
                        .size(32)
                        .color(theme::STATUS_SUCCESS),
                    Space::new().height(8),
                    text("Working tree clean.")
                        .size(14)
                        .color(theme::TEXT_SECONDARY),
                ]
                .align_x(Alignment::Center)
                .spacing(4),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .style(|_t: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..Default::default()
            })
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
        widgets::build_tree_panel(&self.file_tree, elements)
    }

    /// Recursively render a tree node and its children.
    fn render_tree_node<'a>(
        &'a self,
        node: &'a widgets::TreeNode,
        depth: usize,
        ancestor_mask: u64,
        is_last: bool,
    ) -> Element<'a, DiffMessage> {
        if node.is_dir {
            self.render_dir_node(node, depth, ancestor_mask, is_last)
        } else {
            self.render_file_node(node, depth, ancestor_mask, is_last)
        }
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
        let guide_text: Element<'_, DiffMessage> = text(guide)
            .size(widgets::TREE_FONT_SIZE)
            .color(theme::TEXT_MUTED)
            .into();

        let header_row = row![
            guide_text,
            icon.size(widgets::TREE_ICON_SIZE).color(icon_color),
            Space::new().width(4),
            text(&node.name)
                .size(widgets::TREE_FONT_SIZE)
                .color(theme::TEXT_SECONDARY),
            Space::new().width(Length::Fill),
        ]
        .align_y(Alignment::Center)
        .padding([0, 8]);

        let full_path = node.full_path.clone();
        let is_focused = widgets::tree_node_focused(&self.file_tree, &node.full_path);

        let header_btn = widgets::tree_node_button(
            header_row,
            is_focused,
            Some(DiffMessage::ToggleDir(full_path.clone())),
        );

        // Show context menu with "Discard changes" for working-tree diffs only.
        let header_element: Element<'_, DiffMessage> = if self.current_commit_ref.is_some() {
            header_btn
        } else {
            ContextMenu::new(
                header_btn,
                vec![(
                    "Discard changes".into(),
                    DiffMessage::DiscardPath(full_path, DiscardTarget::Directory),
                )],
            )
            .into()
        };

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
        let file = self
            .diff_files
            .iter()
            .find(|f| f.dfile.path == node.full_path);
        let is_selected = self.selected_file.as_deref() == Some(&node.full_path);

        let guide = widgets::tree_guide_prefix(ancestor_mask, depth, is_last);
        let guide_text: Element<'_, DiffMessage> = text(guide)
            .size(widgets::TREE_FONT_SIZE)
            .color(theme::TEXT_MUTED)
            .into();

        // File status icon
        let (icon, icon_color) = if let Some(f) = file {
            if f.dfile.status == DiffFileStatus::Renamed {
                (lucide::arrow_right(), RENAME_COLOR)
            } else if matches!(
                f.dfile.status,
                DiffFileStatus::Added | DiffFileStatus::Untracked
            ) {
                (lucide::file_plus(), FILE_HEADER_COLOR)
            } else if f.dfile.status == DiffFileStatus::Deleted {
                (lucide::file_minus(), FILE_HEADER_COLOR)
            } else if f.dfile.is_binary {
                (lucide::file(), theme::TEXT_MUTED)
            } else {
                (lucide::file_text(), FILE_HEADER_COLOR)
            }
        } else {
            (lucide::file_text(), FILE_HEADER_COLOR)
        };

        // Line count labels
        let counts: Element<'_, DiffMessage> = if let Some(f) = file {
            if f.dfile.is_binary {
                text("binary").size(10).color(theme::TEXT_MUTED).into()
            } else if f.add_count > 0 || f.remove_count > 0 {
                let mut parts: Vec<Element<'_, DiffMessage>> = Vec::new();
                if f.add_count > 0 {
                    parts.push(
                        text(format!("+{}", f.add_count))
                            .size(10)
                            .color(ADDED_COLOR)
                            .into(),
                    );
                }
                if f.add_count > 0 && f.remove_count > 0 {
                    parts.push(text(" ").size(10).into());
                }
                if f.remove_count > 0 {
                    parts.push(
                        text(format!("-{}", f.remove_count))
                            .size(10)
                            .color(REMOVED_COLOR)
                            .into(),
                    );
                }
                row(parts).spacing(0).into()
            } else {
                text("").size(10).into()
            }
        } else {
            text("").size(10).into()
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

        let btn_row = row![
            guide_text,
            icon.size(widgets::TREE_FONT_SIZE).color(icon_color),
            Space::new().width(4),
            text(&node.name)
                .size(widgets::TREE_FONT_SIZE)
                .color(name_color)
                .font(iced::Font {
                    weight: name_weight,
                    ..theme::FONT_REGULAR
                }),
            Space::new().width(Length::Fill),
            counts,
            Space::new().width(6),
        ]
        .align_y(Alignment::Center)
        .padding([0, 8]);

        let full_path = node.full_path.clone();
        let is_focused = widgets::tree_node_focused(&self.file_tree, &node.full_path);

        let file_btn = widgets::tree_node_button(
            btn_row,
            is_selected || is_focused,
            Some(DiffMessage::SelectFile(full_path.clone())),
        );

        if self.current_commit_ref.is_some() {
            file_btn
        } else {
            ContextMenu::new(
                file_btn,
                vec![(
                    "Discard changes".into(),
                    DiffMessage::DiscardPath(full_path, DiscardTarget::File),
                )],
            )
            .into()
        }
    }

    /// Return the diff content panel: file headers, binary/too-large placeholders,
    /// truncation warnings, and per-file [`DiffBufferWidget`]s interleaved.
    fn build_diff_content(&self) -> Element<'_, DiffMessage> {
        if self.diff_files.is_empty() {
            return container(
                scrollable(column![])
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .direction(theme::vertical_scrollbar())
                    .style(theme::scrollbar_style),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_t: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..Default::default()
            })
            .into();
        }

        let mut rows: Vec<Element<'_, DiffMessage>> = Vec::new();
        let mut total_hunks: usize = 0;
        let mut total_lines: usize = 0;
        let mut truncated = false;
        let mut buffer_idx = 0usize;

        for file in &self.diff_files {
            // File selection filter
            if let Some(ref sel) = self.selected_file {
                if file.dfile.path != *sel {
                    continue;
                }
            }
            if truncated {
                break;
            }

            total_hunks += file.dfile.hunks.len();
            if total_hunks > MAX_HUNKS || total_lines > MAX_DIFF_LINES {
                truncated = true;
                break;
            }

            // File header
            let (header_label, header_icon) = if file.dfile.status == DiffFileStatus::Renamed {
                (
                    format!(
                        "Rename: {} \u{2192} {}",
                        file.dfile.old_path.as_deref().unwrap_or("?"),
                        file.dfile.path
                    ),
                    CachedIcon::ArrowRight,
                )
            } else if file.dfile.status == DiffFileStatus::Added {
                (
                    format!("New file: {}", file.dfile.path),
                    CachedIcon::FilePlus,
                )
            } else if file.dfile.status == DiffFileStatus::Deleted {
                (
                    format!("Deleted: {}", file.dfile.path),
                    CachedIcon::FileMinus,
                )
            } else if file.dfile.status == DiffFileStatus::Untracked {
                (
                    format!("Untracked: {}", file.dfile.path),
                    CachedIcon::FilePlus,
                )
            } else {
                (file.dfile.path.clone(), CachedIcon::FileText)
            };

            rows.push(
                container(
                    row![
                        header_icon.to_text().size(12).color(FILE_HEADER_COLOR),
                        Space::new().width(6),
                        text(header_label).size(12).color(FILE_HEADER_COLOR),
                    ]
                    .align_y(Alignment::Center),
                )
                .width(Length::Fill)
                .padding([6, 12])
                .style(move |_t: &iced::Theme| iced::widget::container::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgba(
                        1.0, 0.667, 0.0, 0.06,
                    ))),
                    ..Default::default()
                })
                .into(),
            );

            // Binary / too-large placeholders
            if file.dfile.is_binary {
                rows.push(
                    container(
                        text(format!("Binary file: {}", file.dfile.path))
                            .size(13)
                            .color(theme::TEXT_MUTED),
                    )
                    .padding([2, 12])
                    .into(),
                );
                continue;
            }
            if let Some(sz) = file.dfile.too_large_size {
                rows.push(
                    container(
                        text(format!("File too large: {}, {sz} bytes", file.dfile.path))
                            .size(13)
                            .color(theme::TEXT_MUTED),
                    )
                    .padding([2, 12])
                    .into(),
                );
                continue;
            }

            // Find the matching buffer by index
            if let Some(buf) = self.file_buffers.get(buffer_idx) {
                buffer_idx += 1;

                // Count lines for truncation check
                for hunk in &file.dfile.hunks {
                    total_lines += hunk.lines.len();
                }

                rows.push(iced::Element::new(DiffBufferWidget::new(buf)));
            }
        }

        if truncated {
            rows.push(
                container(
                    text("\u{26a0} Diff truncated (max 5000 lines / 100 hunks)")
                        .size(12)
                        .color(theme::STATUS_WARNING),
                )
                .padding([8, 12])
                .into(),
            );
        }

        container(
            scrollable(column(rows).spacing(0).width(Length::Fill))
                .width(Length::Fill)
                .height(Length::Fill)
                .direction(theme::vertical_scrollbar())
                .style(theme::scrollbar_style),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_t: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(theme::BG_BASE)),
            ..Default::default()
        })
        .into()
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
        self.buffers_generation = self.generation;
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Resolve a workspace's filesystem path, supporting both DB-registered
/// shared workspaces and personal workspace path overrides.
async fn resolve_workspace_path(
    ws_name: &str,
    ws_path_override: Option<String>,
) -> Result<std::path::PathBuf, String> {
    if let Some(p) = ws_path_override {
        Ok(std::path::PathBuf::from(p))
    } else {
        let store = crate::workspace::store();
        let ws = store
            .get_by_name(ws_name)
            .await
            .map_err(|e| format!("Failed to look up workspace: {e}"))?
            .ok_or_else(|| "Workspace not found.".to_string())?;
        Ok(ws.as_path().to_path_buf())
    }
}

/// Load the diff, compute per-file highlights, and return enhanced DiffFiles.
/// `ws_path_override` is used for personal workspaces that don't exist in
/// workspaces.db — when provided, it's used directly as the filesystem path.
async fn load_diff(
    ws_name: String,
    ws_path_override: Option<String>,
    commit_ref: Option<String>,
) -> Result<Vec<DiffFile>, String> {
    let ws_path = resolve_workspace_path(&ws_name, ws_path_override).await?;

    if !git_is_installed().await {
        return Err("Git is not installed.".to_string());
    }
    if !is_git_repo(&ws_path).await {
        return Err("This workspace is not a git repository.".to_string());
    }
    if !git_has_commits(&ws_path)
        .await
        .map_err(|e| format!("Git error: {e}"))?
    {
        return Err("No commits yet \u{2014} nothing to diff against.".to_string());
    }

    let diff_output = run_git_diff(&ws_path, commit_ref.as_deref())
        .await
        .map_err(|e| format!("Failed to run git diff: {e}"))?;
    let mut parsed = parse_git_diff(&diff_output);

    // Untracked files — only relevant for working-tree diffs.
    // Historical commits don't have untracked files.
    if commit_ref.is_none() {
        let status_output = run_git_status(&ws_path)
            .await
            .map_err(|e| format!("Failed to run git status: {e}"))?;
        let untracked: Vec<String> = status_output
            .lines()
            .filter(|l| l.starts_with("?? "))
            .map(|l| l[3..].trim().to_string())
            .collect();

        for path in &untracked {
            let full = ws_path.join(path);
            if !full.is_file() {
                continue;
            }
            let meta = match tokio::fs::metadata(&full).await {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.len() > MAX_UNTRACKED_SIZE {
                parsed.push(crate::diff_parse::DiffFile::placeholder(
                    path.clone(),
                    false,
                    Some(meta.len()),
                ));
                continue;
            }
            let content = match tokio::fs::read(&full).await {
                Ok(c) => c,
                Err(_) => continue,
            };
            if content.contains(&0) {
                parsed.push(crate::diff_parse::DiffFile::placeholder(
                    path.clone(),
                    true,
                    None,
                ));
                continue;
            }
            match String::from_utf8(content) {
                Ok(text) => {
                    parsed.push(make_untracked_diff_file(path, &text));
                }
                Err(_) => {
                    parsed.push(crate::diff_parse::DiffFile::placeholder(
                        path.clone(),
                        true,
                        None,
                    ));
                }
            }
        }
    }

    // Compute highlights for each file off the UI thread.
    let mut enhanced: Vec<DiffFile> = Vec::with_capacity(parsed.len());
    for dfile in parsed {
        let (add_count, remove_count) = count_lines(&dfile);
        let (old_hl, new_hl) = if dfile.is_binary || dfile.too_large_size.is_some() {
            (None, None)
        } else {
            compute_highlights(&dfile, &ws_path, commit_ref.as_deref()).await
        };
        enhanced.push(DiffFile {
            dfile,
            old_highlights: old_hl,
            new_highlights: new_hl,
            add_count,
            remove_count,
        });
    }

    Ok(enhanced)
}

/// Count added and removed lines in a diff file.
fn count_lines(dfile: &crate::diff_parse::DiffFile) -> (usize, usize) {
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
    dfile: &crate::diff_parse::DiffFile,
    ws_path: &Path,
    commit_ref: Option<&str>,
) -> (Option<FileHighlights>, Option<FileHighlights>) {
    let lang = HighlightLanguage::from_path(&dfile.path);
    let Some(lang) = lang else {
        return (None, None);
    };

    // Build parsers for this file (created in the async task, not on UI thread).
    let old_hl = compute_old_highlights(dfile, ws_path, lang, commit_ref).await;
    let new_hl = compute_new_highlights(dfile, ws_path, lang, commit_ref).await;

    (old_hl, new_hl)
}

async fn compute_old_highlights(
    dfile: &crate::diff_parse::DiffFile,
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
    let content = run_git_show(ws_path, old_path, show_ref.as_deref())
        .await
        .ok()
        .flatten()?;

    // Skip highlighting for files over the size limit.
    if content.len() > MAX_HIGHLIGHT_SIZE {
        return None;
    }

    let mut parser = make_parser(lang);
    Some(parse_file_highlights(&mut parser, &content, lang))
}

async fn compute_new_highlights(
    dfile: &crate::diff_parse::DiffFile,
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
        run_git_show(ws_path, &dfile.path, Some(hash))
            .await
            .ok()
            .flatten()?
    } else {
        // Working-tree diff: read from disk (existing behavior).
        let full_path = ws_path.join(&dfile.path);
        tokio::fs::read_to_string(&full_path).await.ok()?
    };

    // Skip highlighting for files over the size limit.
    if content.len() > MAX_HIGHLIGHT_SIZE {
        return None;
    }

    let mut parser = make_parser(lang);
    Some(parse_file_highlights(&mut parser, &content, lang))
}

fn make_parser(lang: HighlightLanguage) -> tree_sitter::Parser {
    let mut parser = tree_sitter::Parser::new();
    let ts_lang = lang.tree_sitter_language();
    let _ = parser.set_language(&ts_lang);
    parser
}

/// Build a directory tree from the list of diff files.
fn build_tree(files: &[DiffFile]) -> Vec<widgets::TreeNode> {
    let mut roots: HashMap<String, widgets::TreeNode> = HashMap::new();

    for file in files {
        let path = &file.dfile.path;
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
            dfile: crate::diff_parse::DiffFile {
                path: path.to_string(),
                old_path: None,
                hunks: Vec::new(),
                status: crate::diff_parse::DiffFileStatus::Modified,
                is_binary: false,
                too_large_size: None,
            },
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
            DiffFile {
                dfile: crate::diff_parse::DiffFile {
                    path: "src/main.rs".to_owned(),
                    old_path: None,
                    hunks: Vec::new(),
                    status: crate::diff_parse::DiffFileStatus::Modified,
                    is_binary: false,
                    too_large_size: None,
                },
                old_highlights: None,
                new_highlights: None,
                add_count: 0,
                remove_count: 0,
            },
            DiffFile {
                dfile: crate::diff_parse::DiffFile {
                    path: "src/lib.rs".to_owned(),
                    old_path: None,
                    hunks: Vec::new(),
                    status: crate::diff_parse::DiffFileStatus::Modified,
                    is_binary: false,
                    too_large_size: None,
                },
                old_highlights: None,
                new_highlights: None,
                add_count: 0,
                remove_count: 0,
            },
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
    fn test_diff_tree_nav_up_clamped() {
        let mut state = make_diff_with_tree();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 0;
        let _ = state.update(DiffMessage::TreeNavUp);
        assert_eq!(state.file_tree.tree_focus_index, 0);
    }

    #[test]
    fn test_diff_tree_nav_down_clamped() {
        let mut state = make_diff_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        let last = state.file_tree.visible_tree_nodes.len() - 1;
        state.file_tree.tree_focus_index = last;
        let _ = state.update(DiffMessage::TreeNavDown);
        assert_eq!(state.file_tree.tree_focus_index, last);
    }

    #[test]
    fn test_diff_tree_nav_up_down_moves_focus() {
        let mut state = make_diff_with_tree();
        // Expand "src" so there are multiple nodes to navigate.
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 1;
        let _ = state.update(DiffMessage::TreeNavUp);
        assert_eq!(state.file_tree.tree_focus_index, 0);
        let _ = state.update(DiffMessage::TreeNavDown);
        assert_eq!(state.file_tree.tree_focus_index, 1);
    }

    #[test]
    fn test_diff_escape_clears_tree_focus() {
        let mut state = make_diff_with_tree();
        state.file_tree.tree_focused = true;
        let _ = state.update(DiffMessage::Escape);
        assert!(!state.file_tree.tree_focused);
    }

    #[test]
    fn test_diff_toggle_dir_sets_tree_focus() {
        let mut state = make_diff_with_tree();
        // Find a directory in the tree
        if let Some((dir_path, _)) = state
            .file_tree
            .visible_tree_nodes
            .iter()
            .find(|(_, is_dir)| *is_dir)
            .cloned()
        {
            let _ = state.update(DiffMessage::ToggleDir(dir_path));
            assert!(state.file_tree.tree_focused);
        }
    }

    #[test]
    fn test_diff_select_file_keeps_tree_focus() {
        let mut state = make_diff_with_tree();
        state.file_tree.tree_focused = true;
        let _ = state.update(DiffMessage::SelectFile("src/main.rs".to_owned()));
        assert!(state.file_tree.tree_focused);
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
    fn test_diff_tree_nav_left_on_expanded_dir_collapses() {
        let mut state = make_diff_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 0; // "src" (expanded)
        assert!(state.file_tree.expanded_dirs.contains("src"));

        let _ = state.update(DiffMessage::TreeNavLeft);
        assert!(!state.file_tree.expanded_dirs.contains("src"));
        assert_eq!(state.file_tree.tree_focus_index, 0);
    }

    #[test]
    fn test_diff_tree_nav_left_on_file_navigates_to_parent() {
        let mut state = make_diff_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 1; // "src/main.rs"

        let _ = state.update(DiffMessage::TreeNavLeft);
        // Should navigate to parent "src"
        assert_eq!(state.file_tree.tree_focus_index, 0);
        assert_eq!(state.file_tree.visible_tree_nodes[0].0, "src");
    }

    #[test]
    fn test_diff_tree_nav_left_on_root_item_noop() {
        let mut state = make_diff_with_tree();
        state.file_tree.tree_focused = true;
        // Focus on "src" (root-level collapsed dir) — no parent to navigate to.
        state.file_tree.tree_focus_index = 0;

        let _ = state.update(DiffMessage::TreeNavLeft);
        assert_eq!(state.file_tree.tree_focus_index, 0);
    }

    #[test]
    fn test_diff_tree_nav_right_on_collapsed_dir_expands_and_advances() {
        let mut state = make_diff_with_tree();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 0; // "src" (collapsed)

        let _ = state.update(DiffMessage::TreeNavRight);
        assert!(state.file_tree.expanded_dirs.contains("src"));
        // After expanding, focus moves to first child (sorted alphabetically).
        assert_eq!(state.file_tree.tree_focus_index, 1);
        assert_eq!(state.file_tree.visible_tree_nodes[1].0, "src/lib.rs");
    }

    #[test]
    fn test_diff_tree_nav_right_on_expanded_dir_moves_to_first_child() {
        let mut state = make_diff_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 0; // "src" (already expanded)

        let _ = state.update(DiffMessage::TreeNavRight);
        assert_eq!(state.file_tree.tree_focus_index, 1);
        assert_eq!(state.file_tree.visible_tree_nodes[1].0, "src/lib.rs");
    }

    #[test]
    fn test_diff_tree_nav_right_on_file_noop() {
        let mut state = make_diff_with_tree();
        state.file_tree.expanded_dirs.insert("src".to_string());
        state.file_tree.nodes = build_tree(&state.diff_files);
        state.file_tree.rebuild_visible();
        state.file_tree.tree_focused = true;
        state.file_tree.tree_focus_index = 1; // "src/main.rs" (file)

        let _ = state.update(DiffMessage::TreeNavRight);
        // ArrowRight on file does nothing
        assert_eq!(state.file_tree.tree_focus_index, 1);
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

    #[test]
    fn test_discard_path_noop_in_commit_view() {
        let mut state = make_diff_with_tree();
        state.current_commit_ref = Some("abc1234".to_owned());
        assert!(!state.diff_loading);

        let _ = state.update(DiffMessage::DiscardPath(
            "src/main.rs".to_owned(),
            DiscardTarget::File,
        ));

        // Must not mark loading in commit-view.
        assert!(!state.diff_loading);
    }

    #[test]
    fn test_discard_path_noop_without_workspace() {
        let mut state = make_diff_with_tree();
        state.selected_workspace_name = None;
        assert!(!state.diff_loading);

        let _ = state.update(DiffMessage::DiscardPath(
            "src/main.rs".to_owned(),
            DiscardTarget::File,
        ));

        assert!(!state.diff_loading);
    }

    #[test]
    fn test_discard_path_sets_loading() {
        let mut state = make_diff_with_tree();
        state.selected_workspace_name = Some("test-ws".to_owned());
        assert!(!state.diff_loading);

        let _ = state.update(DiffMessage::DiscardPath(
            "src/main.rs".to_owned(),
            DiscardTarget::File,
        ));

        // Must mark loading to show progress.
        assert!(state.diff_loading);
    }

    #[test]
    fn test_discard_result_success_no_workspace_resets_loading() {
        let mut state = make_diff_with_tree();
        // Without a selected workspace, the success path falls through to
        // the no-refresh branch and resets loading.
        state.selected_workspace_name = None;
        state.diff_loading = true;

        let _ = state.update(DiffMessage::DiscardResult(Ok(())));

        assert!(!state.diff_loading);
    }

    #[test]
    fn test_discard_result_success_with_workspace_keeps_loading_for_refresh() {
        // When there IS a selected workspace, a successful discard triggers an
        // immediate diff refresh — diff_loading stays true throughout.
        let mut state = make_diff_with_tree();
        state.selected_workspace_name = Some("test-ws".to_owned());
        state.diff_loading = true;

        let _ = state.update(DiffMessage::DiscardResult(Ok(())));

        assert!(state.diff_loading);
    }

    #[test]
    fn test_discard_result_error_resets_loading() {
        let mut state = make_diff_with_tree();
        state.selected_workspace_name = Some("test-ws".to_owned());
        state.diff_loading = true;

        let _ = state.update(DiffMessage::DiscardResult(Err(
            "something went wrong".to_owned()
        )));

        assert!(!state.diff_loading);
    }

    #[test]
    fn test_discard_path_file_target_vs_dir_target() {
        // Verify that DiscardTarget::File and DiscardTarget::Directory are
        // distinct values that both produce a Task (state changes on both).
        let mut state_file = make_diff_with_tree();
        state_file.selected_workspace_name = Some("ws".to_owned());
        let _task_file = state_file.update(DiffMessage::DiscardPath(
            "src/main.rs".to_owned(),
            DiscardTarget::File,
        ));
        assert!(state_file.diff_loading);

        let mut state_dir = make_diff_with_tree();
        state_dir.selected_workspace_name = Some("ws".to_owned());
        let _task_dir = state_dir.update(DiffMessage::DiscardPath(
            "src".to_owned(),
            DiscardTarget::Directory,
        ));
        assert!(state_dir.diff_loading);
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
        let _ = state.update(DiffMessage::WorkspaceSelected(
            "ws".to_owned(),
            Some("/tmp/ws".to_owned()),
        ));
        assert!(state.tree_auto_expand_pending);
        assert!(state.file_tree.expanded_dirs.is_empty());
    }

    #[test]
    fn test_commit_message_clears_tree_focus() {
        let mut state = make_diff_with_tree();
        state.file_tree.tree_focused = true;
        let _ = state.update(DiffMessage::CommitMessageChanged("fix bug".to_owned()));
        assert!(!state.file_tree.tree_focused);
    }
}
