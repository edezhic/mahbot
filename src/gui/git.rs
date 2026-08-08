//! Git state management — branch info, sync, diff stats, branch modal.
//!
//! Extracted from the monolithic `Dashboard::update` (mod.rs) to reduce
//! coupling between git operations and the rest of the dashboard UI.
//!
//! The Dashboard owns a single `git_state: GitState` field and routes
//! [`Message::Git`](super::Message::Git) messages to it. Cross-modal
//! coordination (diff modal ↔ branch modal mutual exclusion) is handled
//! at the Dashboard level.

use super::theme;

use iced::widget::{Column, Space, button, column, container, row, scrollable, text, text_input};
use iced::{Alignment, Element, Length, Task};

use std::path::PathBuf;

/// Git state owned by the Dashboard.
///
/// All git-related fields are encapsulated here. The Dashboard accesses
/// state via query methods (`current_branch()`, `is_modal_open()`, etc.)
/// and drives updates via [`GitMessage`].
#[derive(Debug)]
pub struct GitState {
    // ── Cached git info ──────────────────────────────────────────
    /// Filesystem path for the currently selected workspace.
    workspace_path: Option<PathBuf>,
    /// Cached diff stats (+N / -M) from periodic refresh.
    diff_stats: Option<(i64, i64)>,
    /// Cached current branch name from periodic refresh.
    current_branch: Option<String>,
    /// Cached behind/ahead counts from periodic refresh.
    behind_ahead: Option<(usize, usize)>,

    // ── Branch management modal ──────────────────────────────────
    /// Whether the branch management modal is open.
    show_branch_modal: bool,
    /// Branch search query text.
    branch_search_query: String,
    /// Cached list of local branches.
    local_branches: Vec<String>,
    /// Whether a git sync/switch/create operation is in-flight.
    syncing: bool,
    /// Error message from branch switch/create failure.
    branch_error: Option<String>,
    /// Current value of the "new branch name" text input.
    new_branch_name: String,

    // ── Refresh state ───────────────────────────────────────────
    /// Whether git state was eagerly refreshed recently — skip the next
    /// Tick-based refresh to avoid double-firing after workspace switch.
    refresh_eagerly: bool,
    /// Monotonic generation for git refreshes. Each refresh batch captures
    /// the current value and results are only applied while it is current,
    /// so stale/out-of-order results never overwrite newer state.
    refresh_generation: u64,
}

/// Messages for the git sub-state.
///
/// Analogous to [`DiffMessage`](super::diff::DiffMessage) — the Dashboard
/// wraps these in [`Message::Git`](super::Message::Git) and routes them
/// to [`GitState::update`].
#[derive(Debug, Clone)]
pub enum GitMessage {
    // ── Refresh results ─────────────────────────────────────────
    /// Result of `run_git_diff_stats`. Carries the refresh generation;
    /// results from a superseded generation are discarded. `Ok` on genuine
    /// success (including a clean tree); `Err` on transient failure keeps
    /// the cached last-known-good value.
    DiffStats(u64, Result<(i64, i64), String>),
    /// Result of `run_git_current_branch`. Same generation/staleness
    /// semantics as [`GitMessage::DiffStats`].
    CurrentBranch(u64, Result<String, String>),
    /// Result of `run_git_behind_ahead`. `Ok((0, 0))` is a genuine
    /// clean/no-upstream state (hides the sync button); `Err` keeps the
    /// last-known-good counts. Same generation semantics as
    /// [`GitMessage::DiffStats`].
    BehindAhead(u64, Result<(usize, usize), String>),

    // ── Branch listing ──────────────────────────────────────────
    /// Result of listing local branches.
    ListBranches(Result<Vec<String>, String>),

    // ── Modal control ──────────────────────────────────────────
    /// Open the branch management modal.
    OpenModal,
    /// Close the branch management modal.
    CloseModal,

    // ── Branch search ──────────────────────────────────────────
    /// Branch search query changed.
    BranchQueryChanged(String),

    // ── Sync ────────────────────────────────────────────────────
    /// Trigger a git sync (pull --ff-only + push).
    Sync,
    /// Result of `run_git_sync`.
    SyncResult(Result<String, String>),

    // ── Switch branch ──────────────────────────────────────────
    /// Switch to a branch.
    Switch(String),
    /// Result of switching to a branch.
    SwitchResult(Result<(), String>),

    // ── Create branch ───────────────────────────────────────────
    /// Create a new branch from the value in `new_branch_name`.
    Create,
    /// Result of creating a new branch.
    CreateBranchResult(Result<(), String>),
    /// The new-branch name input changed.
    NewBranchNameChanged(String),

    // ── Cross-state communication ───────────────────────────────
    /// Toast notification for Dashboard interception.
    Toast(super::ToastMessage),
}

impl GitState {
    /// Create a new, empty git state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            workspace_path: None,
            diff_stats: None,
            current_branch: None,
            behind_ahead: None,
            show_branch_modal: false,
            branch_search_query: String::new(),
            local_branches: Vec::new(),
            syncing: false,
            branch_error: None,
            new_branch_name: String::new(),
            refresh_eagerly: false,
            refresh_generation: 0,
        }
    }

    // ── Query methods (for Dashboard view rendering) ─────────────

    /// Cached diff stats (+N / -M), if available.
    #[must_use]
    pub fn diff_stats(&self) -> Option<(i64, i64)> {
        self.diff_stats
    }

    /// Cached current branch name, if available.
    #[must_use]
    pub fn current_branch(&self) -> Option<&str> {
        self.current_branch.as_deref()
    }

    /// Cached behind/ahead counts, if available and non-zero.
    #[must_use]
    pub fn behind_ahead(&self) -> Option<(usize, usize)> {
        self.behind_ahead
    }

    /// Whether a git sync/switch/create operation is in-flight.
    #[must_use]
    pub fn is_syncing(&self) -> bool {
        self.syncing
    }

    /// Whether the branch management modal is open.
    #[must_use]
    pub fn is_modal_open(&self) -> bool {
        self.show_branch_modal
    }

    /// Whether a workspace filesystem path is set (i.e. git operations
    /// can proceed).
    #[must_use]
    pub fn has_filesystem_path(&self) -> bool {
        self.workspace_path.is_some()
    }

    // ── State mutators (called from Dashboard during workspace switch) ──

    /// Clear all cached git info (diff stats, branch, behind/ahead, modal).
    /// Does **not** clear `workspace_path` or `refresh_eagerly` — those are
    /// managed explicitly by [`Self::set_workspace_path`] / [`Self::update_tick`].
    /// Bumps `refresh_generation` so refresh results still in flight for the
    /// previous state are discarded as stale.
    pub fn clear(&mut self) {
        self.diff_stats = None;
        self.current_branch = None;
        self.behind_ahead = None;
        self.local_branches.clear();
        self.branch_search_query.clear();
        self.branch_error = None;
        self.show_branch_modal = false;
        self.new_branch_name.clear();
        self.syncing = false;
        self.refresh_generation += 1;
        // Keep workspace_path — it's set explicitly via set_workspace_path.
        // Keep refresh_eagerly — it's managed by set_workspace_path / tick.
    }

    /// Set the workspace filesystem path and trigger an eager refresh
    /// of git info (diff stats, branch, behind/ahead). Clears all
    /// cached state first.
    ///
    /// After this call the next [`Self::update_tick`] is skipped (via
    /// `refresh_eagerly`) to avoid double-refreshing.
    ///
    /// Returns a batch of [`Task`]s that produce [`GitMessage`] results
    /// when the async operations complete.
    pub fn set_workspace_path(&mut self, path: Option<String>) -> Task<GitMessage> {
        // clear() bumps refresh_generation — results still in flight from the
        // previous workspace are discarded as stale.
        self.clear();
        self.workspace_path = path.map(PathBuf::from);
        // Signal the next tick to skip — the refresh tasks spawned below
        // already cover the initial load.
        self.refresh_eagerly = true;
        match &self.workspace_path {
            Some(p) => Self::refresh_inner(p.clone(), self.refresh_generation),
            None => Task::none(),
        }
    }

    // ── Update / message handling ─────────────────────────────────

    /// Process a [`GitMessage`] and return any resulting tasks.
    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, msg: GitMessage) -> Task<GitMessage> {
        match msg {
            // ── Refresh results ─────────────────────────────────
            GitMessage::DiffStats(generation, result) => {
                // Only apply current-generation successes — a transient git
                // failure (`Err`) keeps the last-known-good value so the
                // footer controls don't flicker.
                if generation == self.refresh_generation
                    && let Ok(stats) = result
                {
                    self.diff_stats = Some(stats);
                }
                Task::none()
            }
            GitMessage::CurrentBranch(generation, result) => {
                if generation == self.refresh_generation
                    && let Ok(branch) = result
                {
                    self.current_branch = Some(branch);
                }
                Task::none()
            }
            GitMessage::BehindAhead(generation, result) => {
                if generation == self.refresh_generation
                    && let Ok(ba) = result
                {
                    // Genuine clean/no-upstream state hides the sync button.
                    self.behind_ahead = (ba.0 > 0 || ba.1 > 0).then_some(ba);
                }
                Task::none()
            }

            // ── Branch listing result ───────────────────────────
            GitMessage::ListBranches(result) => {
                match result {
                    Ok(branches) => self.local_branches = branches,
                    Err(e) => self.branch_error = Some(e),
                }
                Task::none()
            }

            // ── Modal control ──────────────────────────────────
            GitMessage::OpenModal => {
                self.show_branch_modal = true;
                self.branch_search_query.clear();
                self.branch_error = None;
                let ws_path = self.workspace_path.clone();
                Task::perform(
                    async move {
                        match ws_path {
                            Some(path) => {
                                let out = crate::git_commands::run_git_command(
                                    &path,
                                    &["branch", "--format=%(refname:short)"],
                                )
                                .await
                                .map_err(|e| e.to_string())?;
                                Ok(out.lines().map(ToString::to_string).collect())
                            }
                            None => Ok(Vec::new()),
                        }
                    },
                    GitMessage::ListBranches,
                )
            }
            GitMessage::CloseModal => {
                self.show_branch_modal = false;
                Task::none()
            }

            // ── Branch search ──────────────────────────────────
            GitMessage::BranchQueryChanged(query) => {
                self.branch_search_query = query;
                Task::none()
            }

            // ── Sync ────────────────────────────────────────────
            GitMessage::Sync => {
                if self.syncing {
                    return Task::none();
                }
                self.syncing = true;
                let ws_path = self.workspace_path.clone();
                Task::perform(
                    with_ws_path(ws_path, |path: PathBuf| async move {
                        crate::git_commands::run_git_sync(&path)
                            .await
                            .map_err(|e| e.to_string())
                    }),
                    GitMessage::SyncResult,
                )
            }
            GitMessage::SyncResult(result) => {
                self.syncing = false;
                match result {
                    Ok(output) => {
                        let msg = if output.trim().is_empty() {
                            "Already up-to-date".to_string()
                        } else {
                            format!("Sync completed:\n{output}")
                        };
                        Task::done(GitMessage::Toast(super::ToastMessage::SuccessMsg(msg)))
                    }
                    Err(e) => Task::done(GitMessage::Toast(super::ToastMessage::Error(format!(
                        "Sync failed: {e}"
                    )))),
                }
            }

            // ── Switch branch ──────────────────────────────────
            GitMessage::Switch(branch) => {
                if self.syncing {
                    return Task::none();
                }
                self.syncing = true;
                let ws_path = self.workspace_path.clone();
                let branch_clone = branch;
                Task::perform(
                    with_ws_path(ws_path, |path: PathBuf| async move {
                        crate::git_commands::run_git_command(
                            &path,
                            &["switch", branch_clone.as_str()],
                        )
                        .await
                        .map_err(|e| e.to_string())?;
                        Ok(())
                    }),
                    GitMessage::SwitchResult,
                )
            }
            GitMessage::SwitchResult(result) => self.finish_branch_op(result, "Switched branch"),

            // ── Create branch ──────────────────────────────────
            GitMessage::Create => {
                if self.syncing {
                    return Task::none();
                }
                let branch = self.new_branch_name.clone();
                if branch.trim().is_empty() {
                    self.branch_error = Some("Branch name cannot be empty".to_string());
                    return Task::none();
                }
                let ws_path = self.workspace_path.clone();
                let branch_clone = branch.trim().to_string();
                self.syncing = true;
                Task::perform(
                    with_ws_path(ws_path, |path: PathBuf| async move {
                        crate::git_commands::run_git_command(
                            &path,
                            &["switch", "-c", branch_clone.as_str()],
                        )
                        .await
                        .map_err(|e| e.to_string())?;
                        Ok(())
                    }),
                    GitMessage::CreateBranchResult,
                )
            }
            GitMessage::CreateBranchResult(result) => {
                self.finish_branch_op(result, "Created and switched to new branch")
            }
            GitMessage::NewBranchNameChanged(name) => {
                self.new_branch_name = name;
                Task::none()
            }

            // ── Toast passthrough ─────────────────────────────
            GitMessage::Toast(_) => {
                // Dashboard intercepts this variant before it reaches
                // us — this arm is unreachable in practice.
                Task::none()
            }
        }
    }

    /// Handle a branch switch/create result: clear syncing, close the modal
    /// on success (with toast), record the error on failure.
    fn finish_branch_op(
        &mut self,
        result: Result<(), String>,
        success_msg: &str,
    ) -> Task<GitMessage> {
        self.syncing = false;
        match result {
            Ok(()) => {
                self.show_branch_modal = false;
                Task::done(GitMessage::Toast(super::ToastMessage::SuccessMsg(
                    success_msg.to_string(),
                )))
            }
            Err(e) => {
                self.branch_error = Some(e);
                Task::none()
            }
        }
    }

    // ── Tick ──────────────────────────────────────────────────────

    /// Called every second from the Dashboard's [`super::Message::Tick`]
    /// handler. Refreshes git info (diff stats, branch, behind/ahead)
    /// if not gated by an eager refresh (see [`Self::set_workspace_path`]).
    pub fn update_tick(&mut self) -> Task<GitMessage> {
        // Skip if an eager refresh was just triggered (e.g. after
        // workspace switch) to avoid 6 subprocess calls in <1 second.
        if self.refresh_eagerly {
            self.refresh_eagerly = false;
            return Task::none();
        }

        // Use the stored workspace path.
        match &self.workspace_path {
            Some(p) => {
                // Bump the generation so results still in flight from the
                // previous tick are discarded when they arrive out of order.
                self.refresh_generation += 1;
                Self::refresh_inner(p.clone(), self.refresh_generation)
            }
            None => Task::none(),
        }
    }

    // ── View ──────────────────────────────────────────────────────

    /// Render the branch management modal content (search, branch list,
    /// create section). Does **not** wrap in a `modal_overlay` — that is
    /// done by the Dashboard so the close message is consistent with the
    /// rest of the overlay stack.
    pub fn view(&self) -> Element<'_, GitMessage> {
        let search_input = text_input("Search branches…", &self.branch_search_query)
            .on_input(GitMessage::BranchQueryChanged)
            .padding(8)
            .size(14);

        // Filter branches by search query
        let filtered: Vec<&String> = if self.branch_search_query.is_empty() {
            self.local_branches.iter().collect()
        } else {
            let q = self.branch_search_query.to_lowercase();
            self.local_branches
                .iter()
                .filter(|b| b.to_lowercase().contains(&q))
                .collect()
        };

        let branch_items: Vec<Element<'_, GitMessage>> = filtered
            .iter()
            .map(|branch| {
                let b = (*branch).clone();
                button(text(b.clone()).size(14).color(theme::TEXT_PRIMARY))
                    .padding([6, 12])
                    .width(Length::Fill)
                    .style(theme::button_text)
                    .on_press_maybe(if self.syncing {
                        None
                    } else {
                        Some(GitMessage::Switch(b))
                    })
                    .into()
            })
            .collect();

        let list = scrollable(Column::with_children(branch_items).spacing(2))
            .height(Length::Fill)
            .style(theme::scrollbar_style);

        // Error display
        let error_elem: Element<'_, GitMessage> = if let Some(ref err) = self.branch_error {
            text(err).size(12).color(theme::STATUS_ERROR).into()
        } else {
            container(text("")).into()
        };

        // Create new branch input + button
        let create_input = text_input("New branch name…", &self.new_branch_name)
            .on_input(GitMessage::NewBranchNameChanged)
            .on_submit(GitMessage::Create)
            .padding(8)
            .size(14);

        let create_btn = button(text("Create & Switch").size(14).color(theme::TEXT_PRIMARY))
            .padding([6, 12])
            .style(theme::button_primary)
            .on_press_maybe(if self.syncing {
                None
            } else {
                Some(GitMessage::Create)
            });

        column![
            text("Branches").size(18).color(theme::TEXT_PRIMARY),
            Space::new().height(8),
            search_input,
            Space::new().height(8),
            list,
            error_elem,
            Space::new().height(8),
            row![create_input, create_btn]
                .spacing(8)
                .align_y(Alignment::Center),
        ]
        .spacing(0)
        .height(Length::Fill)
        .into()
    }
}

// ── Private helpers ──────────────────────────────────────────────

/// Run `op` against the workspace path, mapping a missing path to the
/// standard error used by all git operations.
async fn with_ws_path<T>(
    ws_path: Option<PathBuf>,
    op: impl AsyncFnOnce(PathBuf) -> Result<T, String>,
) -> Result<T, String> {
    match ws_path {
        Some(path) => op(path).await,
        None => Err("No workspace path".to_string()),
    }
}

impl GitState {
    /// Spawn three parallel async tasks to refresh diff stats, current
    /// branch, and behind/ahead counts. Results carry `generation` and are applied
    /// by [`Self::update`] only while the generation is current — stale or
    /// out-of-order results are dropped. Transient git failures surface as
    /// `Err` and leave the cached last-known-good values untouched.
    fn refresh_inner(path: PathBuf, generation: u64) -> Task<GitMessage> {
        if !crate::git_commands::is_git_repo(&path) {
            return Task::none();
        }

        // Diff stats
        let stats_path = path.clone();
        let stats_task = Task::perform(
            async move {
                crate::git_commands::run_git_diff_stats(&stats_path)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |res| GitMessage::DiffStats(generation, res),
        );

        // Current branch
        let branch_path = path.clone();
        let branch_task = Task::perform(
            async move {
                crate::git_commands::run_git_current_branch(&branch_path)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |res| GitMessage::CurrentBranch(generation, res),
        );

        // Behind/ahead
        let ahead_path = path;
        let ahead_task = Task::perform(
            async move {
                crate::git_commands::run_git_behind_ahead(&ahead_path)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |res| GitMessage::BehindAhead(generation, res),
        );

        Task::batch([stats_task, branch_task, ahead_task])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with_gen(generation: u64) -> GitState {
        let mut s = GitState::new();
        s.refresh_generation = generation;
        s
    }

    // ── Refresh results: last-known-good + generation guard ───────

    #[test]
    fn test_refresh_success_applies_current_generation() {
        let mut s = state_with_gen(7);
        let _ = s.update(GitMessage::DiffStats(7, Ok((3, 1))));
        let _ = s.update(GitMessage::CurrentBranch(7, Ok("main".into())));
        let _ = s.update(GitMessage::BehindAhead(7, Ok((2, 5))));
        assert_eq!(s.diff_stats(), Some((3, 1)));
        assert_eq!(s.current_branch(), Some("main"));
        assert_eq!(s.behind_ahead(), Some((2, 5)));
    }

    #[test]
    fn test_refresh_error_keeps_last_known_good() {
        let mut s = state_with_gen(7);
        let _ = s.update(GitMessage::DiffStats(7, Ok((3, 1))));
        let _ = s.update(GitMessage::CurrentBranch(7, Ok("main".into())));
        let _ = s.update(GitMessage::BehindAhead(7, Ok((2, 5))));
        // Transient failures — cached values must survive (no flicker).
        let _ = s.update(GitMessage::DiffStats(7, Err("spawn failed".into())));
        let _ = s.update(GitMessage::CurrentBranch(7, Err("spawn failed".into())));
        let _ = s.update(GitMessage::BehindAhead(7, Err("spawn failed".into())));
        assert_eq!(s.diff_stats(), Some((3, 1)));
        assert_eq!(s.current_branch(), Some("main"));
        assert_eq!(s.behind_ahead(), Some((2, 5)));
    }

    #[test]
    fn test_genuine_clean_state_hides_sync() {
        let mut s = state_with_gen(7);
        let _ = s.update(GitMessage::BehindAhead(7, Ok((2, 5))));
        // Genuine clean/no-upstream — sync button must hide.
        let _ = s.update(GitMessage::BehindAhead(7, Ok((0, 0))));
        assert_eq!(s.behind_ahead(), None);
    }

    #[test]
    fn test_stale_generation_results_are_dropped() {
        let mut s = state_with_gen(9);
        // Results from a superseded generation must not overwrite newer state.
        let _ = s.update(GitMessage::DiffStats(8, Ok((99, 99))));
        let _ = s.update(GitMessage::CurrentBranch(8, Ok("old-branch".into())));
        let _ = s.update(GitMessage::BehindAhead(8, Ok((99, 99))));
        assert_eq!(s.diff_stats(), None);
        assert_eq!(s.current_branch(), None);
        assert_eq!(s.behind_ahead(), None);
    }

    #[test]
    fn test_clear_invalidates_inflight_results() {
        let mut s = state_with_gen(3);
        let _ = s.update(GitMessage::DiffStats(3, Ok((3, 1))));
        let _ = s.update(GitMessage::CurrentBranch(3, Ok("main".into())));
        s.clear(); // bumps the generation — in-flight results become stale
        let _ = s.update(GitMessage::DiffStats(3, Ok((7, 7))));
        assert_eq!(s.diff_stats(), None);
    }
}
