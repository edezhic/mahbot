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

use crate::git::commands::DiffStats;

use iced::widget::{Column, Space, button, column, container, row, scrollable, text};
use iced::{Alignment, Element, Length, Task};

use fff_search::{Error as FffError, WatchId, WatchOptions};

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Broadcast for workspace file-change events. The fff-search watcher callback
/// sends here (non-blocking) from its dedicated thread; the iced
/// [`super::common::broadcast_stream_producer`] subscription consumes it and
/// emits [`GitMessage::FileChanged`]. Capacity is generous so a burst of events
/// never blocks the sender.
pub(super) static FILE_CHANGE_TX: OnceLock<tokio::sync::broadcast::Sender<()>> = OnceLock::new();

/// Initialise the git file-change broadcast. Called from [`crate::gui::init_git_file_change_tx`]
/// in `main` before the iced application runs — the subscription producer uses
/// `FILE_CHANGE_TX.get()` and a missing source yields an immediately-ending
/// stream, which iced's subscription tracker does not re-spawn.
pub(super) fn init_file_change_tx() {
    let _ = FILE_CHANGE_TX.set(tokio::sync::broadcast::channel(64).0);
}

fn file_change_tx() -> &'static tokio::sync::broadcast::Sender<()> {
    FILE_CHANGE_TX.get_or_init(|| tokio::sync::broadcast::channel(64).0)
}

/// Minimum interval between event-driven local git refreshes (diff stats +
/// branch). Coalesces bursts of file-change events into at most one git
/// subprocess batch per interval.
const LOCAL_REFRESH_MIN_INTERVAL: Duration = Duration::from_millis(300);

/// Interval between periodic remote syncs (behind/ahead + `git fetch`). The
/// standalone remote-sync subscription fires [`GitMessage::PeriodicSyncTick`]
/// at this cadence.
pub(crate) const REMOTE_REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// Non-blocking retry window for registering the file-change watch while the
/// watcher is not ready (initial scan) or the engine is not yet created.
const WATCH_RETRY_WINDOW: Duration = Duration::from_secs(30);

/// Non-blocking retry cadence, within the bounded [`WATCH_RETRY_WINDOW`], for
/// the scoped self-rescheduling watch-retry timer (matches the previous
/// per-tick retry cadence).
const WATCH_RETRY_POLL: Duration = Duration::from_secs(1);

/// Git state owned by the Dashboard.
///
/// All git-related fields are encapsulated here. The Dashboard accesses
/// state via query methods (`current_branch()`, `is_modal_open()`, etc.)
/// and drives updates via [`GitMessage`].
pub struct GitState {
    // ── Cached git info ──────────────────────────────────────────
    /// Filesystem path for the currently selected workspace.
    workspace_path: Option<PathBuf>,
    /// Cached diff stats (+N / -M) from periodic refresh.
    diff_stats: Option<DiffStats>,
    /// Cached current branch name from periodic refresh.
    current_branch: Option<String>,
    /// Cached behind/ahead counts from periodic refresh.
    behind_ahead: Option<(usize, usize)>,

    // ── Branch management modal ──────────────────────────────────
    /// Whether the branch management modal is open.
    show_branch_modal: bool,
    /// Branch search query text.
    branch_search_query: super::common::SingleLineEditorState,
    /// Cached list of local branches.
    local_branches: Vec<String>,
    /// Whether a git sync/switch/create operation is in-flight.
    syncing: bool,
    /// Error message from branch switch/create failure.
    branch_error: Option<String>,
    /// Current value of the "new branch name" text input.
    new_branch_name: super::common::SingleLineEditorState,

    // ── Refresh state ───────────────────────────────────────────
    /// Workspace name for the currently selected workspace, used to resolve the
    /// per-workspace [`SharedFilePicker`](fff_search::SharedFilePicker) watch
    /// by name (the registry is keyed by workspace name, not path). `None` for
    /// Personal/unnamed workspaces, which fall back to the timer.
    workspace_name: Option<String>,
    /// Generation for local refreshes (diff stats + current branch). Each
    /// local refresh captures the current value and only applies while it is
    /// current, so stale/out-of-order results never overwrite newer state.
    local_generation: u64,
    /// Generation for remote refreshes (behind/ahead + fetch). Independent of
    /// [`Self::local_generation`] so a long-running fetch is not invalidated by
    /// a concurrent event-driven local refresh.
    remote_generation: u64,
    /// When the last event-driven local refresh ran — the throttle gate for
    /// coalescing bursts of file-change events.
    last_local_refresh: Option<Instant>,
    /// A file-change event arrived during the throttle window and a local
    /// refresh is still owed; fired by the scoped deferred-refresh timer once
    /// the window elapses.
    local_refresh_pending: bool,
    /// When the last remote sync (behind/ahead + fetch) ran. Only updated by an
    /// actual remote sync — a workspace switch alone does not defer the periodic
    /// timer.
    last_remote_refresh: Option<Instant>,
    /// The active file-change watch subscription id. `Some` only when a watch
    /// was successfully registered on the current workspace's picker.
    watch_id: Option<WatchId>,
    /// Registration of the file-change watch failed (watcher not ready, or the
    /// engine not created yet) — retried non-blocking by a scoped
    /// self-rescheduling timer within [`WATCH_RETRY_WINDOW`].
    watch_pending: bool,
    /// Bounded window for [`Self::watch_pending`] retries; after it elapses the
    /// watch is abandoned in favour of the timer fallback.
    watch_retry_deadline: Option<Instant>,
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
    DiffStats(u64, Result<DiffStats, String>),
    /// Result of `run_git_current_branch`. Same generation/staleness
    /// semantics as [`GitMessage::DiffStats`].
    CurrentBranch(u64, Result<String, String>),
    /// Result of `run_git_behind_ahead`. `Ok((0, 0))` is a genuine
    /// clean/no-upstream state (hides the sync button); `Err` keeps the
    /// last-known-good counts. Same generation semantics as
    /// [`GitMessage::DiffStats`] but against [`GitState::remote_generation`].
    BehindAhead(u64, Result<(usize, usize), String>),

    // ── Event-driven refresh ────────────────────────────────────
    /// A workspace file-change event arrived from the fff-search watcher. The
    /// local git refresh (diff stats + branch) is throttled/coalesced here.
    FileChanged,
    /// A manual commit completed (diff modal). A commit is a ref-only change
    /// (HEAD + `.git/index`) that the file watcher never reports, so the footer
    /// is refreshed promptly.
    RefreshAfterCommit,
    /// A pipeline auto-commit completed (sanitation phase committed a ticket).
    /// Carries the repo path so the footer refresh is applied only when it
    /// matches the currently-viewed workspace.
    PipelineCommit(PathBuf),

    // ── Self-driven timers ──────────────────────────────────────
    /// Periodic 60s tick from the standalone remote-sync subscription. Runs the
    /// behind/ahead + `git fetch` sync and also refreshes the local footer — the
    /// recovery path for ref-only/commit-only changes (bare commit, fetch, push,
    /// `checkout -c`) that produce no file-change event. Runs for every
    /// workspace, watched or not.
    PeriodicSyncTick,
    /// Fired by the scoped self-rescheduling timer armed when a file-change event
    /// was deferred by the 300ms local-refresh throttle. Fires the owed local
    /// refresh, or reschedules itself when a newer event restarted the throttle
    /// window. A stale timer firing with no owed refresh is a no-op.
    DeferredLocalRefresh,
    /// Fired by the scoped self-rescheduling 1s retry timer while a file-change
    /// watch registration is pending (watcher not ready / engine not created).
    /// Retries within the bounded [`WATCH_RETRY_WINDOW`]; abandons the watch in
    /// favour of the periodic timer when the window elapses. No-op when nothing
    /// is pending.
    WatchRetryTick,

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
    BranchQueryChanged(super::editor_widget::EditorAction),

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
    NewBranchNameChanged(super::editor_widget::EditorAction),

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
            branch_search_query: super::common::SingleLineEditorState::new(""),
            local_branches: Vec::new(),
            syncing: false,
            branch_error: None,
            new_branch_name: super::common::SingleLineEditorState::new(""),
            workspace_name: None,
            local_generation: 0,
            remote_generation: 0,
            last_local_refresh: None,
            local_refresh_pending: false,
            last_remote_refresh: None,
            watch_id: None,
            watch_pending: false,
            watch_retry_deadline: None,
        }
    }

    // ── Query methods (for Dashboard view rendering) ─────────────

    /// Cached diff stats (+N / -M), if available.
    #[must_use]
    pub fn diff_stats(&self) -> Option<DiffStats> {
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
    /// Does **not** clear `workspace_path`, `workspace_name`, the watch state,
    /// or the refresh throttle/timer state — those are managed explicitly by
    /// [`Self::set_workspace_path`] and the periodic/deferred timers. Bumps both
    /// generations so refresh results still in flight for the previous state
    /// are discarded as stale.
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
        self.local_generation += 1;
        self.remote_generation += 1;
    }

    /// Set the workspace name and filesystem path, and trigger an eager refresh
    /// of git info (diff stats, branch, behind/ahead for the new workspace).
    /// Clears all cached state first and re-registers the file-change watch for
    /// the new workspace (best-effort; retried by a self-rescheduling timer if
    /// the watcher is not ready yet).
    ///
    /// `name` is `None` for Personal/unnamed workspaces, which have no search
    /// engine subscribe to and therefore fall back to the periodic timer.
    ///
    /// Returns a batch of [`Task`]s that produce [`GitMessage`] results
    /// when the async operations complete.
    pub fn set_workspace_path(
        &mut self,
        name: Option<String>,
        path: Option<String>,
    ) -> Task<GitMessage> {
        // Unwatch the previous workspace's watch before switching the name.
        self.unwatch_current();
        // clear() bumps both generations — results still in flight from the
        // previous workspace are discarded as stale.
        self.clear();
        self.workspace_name = name.filter(|n| !n.is_empty());
        self.workspace_path = path.map(PathBuf::from);
        // Reset the local throttle for the new workspace (nothing refreshed yet).
        self.last_local_refresh = None;
        self.local_refresh_pending = false;
        self.watch_pending = false;
        self.watch_retry_deadline = None;
        // Register the file-change watch (best-effort; retried by a
        // self-rescheduling timer while the watcher is not ready yet).
        let watch_task = self.try_register_watch();
        match &self.workspace_path {
            Some(p) => {
                // Mark the local throttle as just-refreshed so a watcher echo of
                // the eager refresh does not double-fire. The remote timer is NOT
                // reset here — a switch alone must not defer the periodic fetch.
                self.last_local_refresh = Some(Instant::now());
                Task::batch([
                    Self::refresh_local(p.clone(), self.local_generation),
                    Self::refresh_remote(p.clone(), self.remote_generation, false),
                    watch_task,
                ])
            }
            None => watch_task,
        }
    }

    // ── Update / message handling ─────────────────────────────────

    /// Process a [`GitMessage`] and return any resulting tasks.
    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: GitMessage) -> Task<GitMessage> {
        match msg {
            // ── Refresh results ─────────────────────────────────
            GitMessage::DiffStats(generation, result) => {
                // Only apply current-generation successes — a transient git
                // failure (`Err`) keeps the last-known-good value so the
                // footer controls don't flicker.
                if generation == self.local_generation
                    && let Ok(stats) = result
                {
                    self.diff_stats = Some(stats);
                }
                Task::none()
            }
            GitMessage::CurrentBranch(generation, result) => {
                if generation == self.local_generation
                    && let Ok(branch) = result
                {
                    self.current_branch = Some(branch);
                }
                Task::none()
            }
            GitMessage::BehindAhead(generation, result) => {
                if generation == self.remote_generation
                    && let Ok(ba) = result
                {
                    // Genuine clean/no-upstream state hides the sync button.
                    self.behind_ahead = (ba.0 > 0 || ba.1 > 0).then_some(ba);
                }
                Task::none()
            }

            // ── Event-driven local refresh ───────────────────────
            GitMessage::FileChanged => self.on_file_changed(),
            // A manual commit is a ref-only change the watcher never reports,
            // so refresh the footer promptly (same as other git operations).
            GitMessage::RefreshAfterCommit => self.refresh_after_git_op(),
            GitMessage::PipelineCommit(path) => {
                // Only refresh the footer when the committed workspace is the
                // one currently displayed — commits to other workspaces must
                // not trigger a needless git subprocess batch.
                if self.workspace_path.as_deref() == Some(path.as_path()) {
                    self.refresh_after_git_op()
                } else {
                    Task::none()
                }
            }

            // ── Self-driven timers (replacing the dashboard tick) ───
            GitMessage::PeriodicSyncTick => {
                // Periodic remote sync (behind/ahead + fetch). The timer also
                // refreshes the local footer (diff stats + branch) so
                // ref-only/commit-only changes that never reach the file
                // watcher still surface here, for every workspace. Clearing an
                // owed event-driven refresh avoids a redundant deferred-local
                // refresh firing right after the periodic one.
                let now = Instant::now();
                if self
                    .last_remote_refresh
                    .is_none_or(|t| now.duration_since(t) >= REMOTE_REFRESH_INTERVAL)
                {
                    self.last_remote_refresh = Some(now);
                    self.local_generation += 1;
                    self.remote_generation += 1;
                    self.last_local_refresh = Some(now);
                    self.local_refresh_pending = false;
                    if let Some(path) = self.workspace_path.clone() {
                        Task::batch([
                            Self::refresh_local(path.clone(), self.local_generation),
                            Self::refresh_remote(path, self.remote_generation, true),
                        ])
                    } else {
                        Task::none()
                    }
                } else {
                    Task::none()
                }
            }
            GitMessage::DeferredLocalRefresh => {
                // Overlapping timers are safe: the pending flag makes a stale
                // firing a no-op, so only the timer responsible for the owed
                // refresh acts.
                if !self.local_refresh_pending {
                    return Task::none();
                }
                let now = Instant::now();
                match self.fire_local_refresh(now) {
                    Some(task) => task,
                    None if self.workspace_path.is_none() => {
                        // No workspace to refresh — the owed refresh is
                        // unreachable.
                        self.local_refresh_pending = false;
                        Task::none()
                    }
                    None => {
                        // A newer event restarted the throttle window —
                        // reschedule.
                        let remaining = LOCAL_REFRESH_MIN_INTERVAL
                            .saturating_sub(
                                now.duration_since(
                                    self.last_local_refresh
                                        .expect("window not elapsed implies throttle set"),
                                ),
                            )
                            .max(Duration::from_millis(1));
                        Task::perform(
                            // Sleep constructed lazily at first poll (inside
                            // the async block) — creating it eagerly here
                            // would panic off the tokio runtime context.
                            async move { tokio::time::sleep(remaining).await },
                            |()| GitMessage::DeferredLocalRefresh,
                        )
                    }
                }
            }
            GitMessage::WatchRetryTick => {
                // A stale retry firing with nothing pending is a no-op.
                if !self.watch_pending {
                    return Task::none();
                }
                let now = Instant::now();
                // Once the bounded retry window elapses (or is unanchored),
                // abandon the watch in favour of the periodic timer fallback.
                if self.watch_retry_deadline.is_none_or(|d| now >= d) {
                    self.watch_pending = false;
                    self.watch_retry_deadline = None;
                    return Task::none();
                }
                let task = self.try_register_watch();
                if self.watch_pending {
                    task
                } else {
                    Task::none()
                }
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
                                let out = crate::git::commands::run_git_command(
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
            GitMessage::BranchQueryChanged(action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                self.branch_search_query.apply_action(action);
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
                        crate::git::commands::run_git_sync(&path)
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
                        // A sync (pull/push) moved refs and possibly files, so
                        // refresh promptly — the watcher is silent to ref-only
                        // changes.
                        let refresh = self.refresh_after_git_op();
                        let msg = if output.trim().is_empty() {
                            "Already up-to-date".to_string()
                        } else {
                            format!("Sync completed:\n{output}")
                        };
                        Task::batch([
                            Task::done(GitMessage::Toast(super::ToastMessage::SuccessMsg(msg))),
                            refresh,
                        ])
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
                        crate::git::commands::run_git_command(
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
                let branch = self.new_branch_name.text();
                if branch.trim().is_empty() {
                    self.branch_error = Some("Branch name cannot be empty".to_string());
                    return Task::none();
                }
                let ws_path = self.workspace_path.clone();
                let branch_clone = branch.trim().to_string();
                self.syncing = true;
                Task::perform(
                    with_ws_path(ws_path, |path: PathBuf| async move {
                        crate::git::commands::run_git_command(
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
            GitMessage::NewBranchNameChanged(action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                self.new_branch_name.apply_action(action);
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
                // A branch switch/create moved refs and possibly files, so
                // refresh promptly — the watcher is silent to ref-only changes.
                let refresh = self.refresh_after_git_op();
                Task::batch([
                    Task::done(GitMessage::Toast(super::ToastMessage::SuccessMsg(
                        success_msg.to_string(),
                    ))),
                    refresh,
                ])
            }
            Err(e) => {
                self.branch_error = Some(e);
                Task::none()
            }
        }
    }

    // ── View ──────────────────────────────────────────────────────

    /// Render the branch management dialog content (search, branch list,
    /// create section). Does **not** wrap in a modal backdrop — the
    /// Dashboard wraps it in a small centered dialog so the close message
    /// is consistent with the rest of the overlay stack.
    pub fn view(&self) -> Element<'_, GitMessage> {
        let search_input = super::widgets::single_line_editor(
            &self.branch_search_query.buffer,
            "Search branches…",
            false,
            Length::Fill,
            Some(iced::widget::Id::new("git_branch_search")),
            GitMessage::BranchQueryChanged,
        );

        // Filter branches by search query
        let filtered: Vec<&String> = if self.branch_search_query.text().is_empty() {
            self.local_branches.iter().collect()
        } else {
            let q = self.branch_search_query.text().to_lowercase();
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
            .height(Length::Fixed(300.0))
            .style(theme::scrollbar_style);

        // Error display
        let error_elem: Element<'_, GitMessage> = if let Some(ref err) = self.branch_error {
            text(err).size(12).color(theme::STATUS_ERROR).into()
        } else {
            container(text("")).into()
        };

        // Create new branch input + button
        let create_input = super::widgets::single_line_editor(
            &self.new_branch_name.buffer,
            "New branch name…",
            true,
            Length::Fill,
            Some(iced::widget::Id::new("git_new_branch")),
            |action| match action {
                super::editor_widget::EditorAction::Submit => GitMessage::Create,
                other => GitMessage::NewBranchNameChanged(other),
            },
        );

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
        .width(Length::Fill)
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

/// Self-rescheduling task that fires [`GitMessage::WatchRetryTick`] after
/// [`WATCH_RETRY_POLL`] — the non-blocking retry cadence for a pending watch
/// registration.
fn watch_retry_task() -> Task<GitMessage> {
    // Sleep built lazily at first poll — eager construction would panic off
    // the tokio runtime context (see the deferred-refresh timer).
    Task::perform(
        async move { tokio::time::sleep(WATCH_RETRY_POLL).await },
        |()| GitMessage::WatchRetryTick,
    )
}

impl GitState {
    /// Spawn two parallel async tasks to refresh the local working-tree state
    /// (diff stats + current branch). Results carry `generation` and are applied
    /// by [`Self::update`] only while the local generation is current — stale or
    /// out-of-order results are dropped. Transient git failures surface as
    /// `Err` and leave the cached last-known-good values untouched.
    fn refresh_local(path: PathBuf, generation: u64) -> Task<GitMessage> {
        if !crate::git::commands::is_git_repo(&path) {
            return Task::none();
        }

        // Diff stats
        let stats_path = path.clone();
        let stats_task = Task::perform(
            async move {
                crate::git::commands::run_git_diff_stats(&stats_path)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |res| GitMessage::DiffStats(generation, res),
        );

        // Current branch
        let branch_path = path;
        let branch_task = Task::perform(
            async move {
                crate::git::commands::run_git_current_branch(&branch_path)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |res| GitMessage::CurrentBranch(generation, res),
        );

        Task::batch([stats_task, branch_task])
    }

    /// Spawn a remote refresh: best-effort `git fetch` (when `fetch` is set)
    /// followed by behind/ahead. `fetch=false` compares against the locally
    /// available remote ref; `fetch=true` first updates them so behind/ahead
    /// reflects upstream state. A fetch failure (offline/timeout/no remote) is
    /// logged at debug level and behind/ahead is still computed against local
    /// refs.
    fn refresh_remote(path: PathBuf, generation: u64, fetch: bool) -> Task<GitMessage> {
        if !crate::git::commands::is_git_repo(&path) {
            return Task::none();
        }
        Task::perform(
            async move {
                if fetch {
                    if let Err(e) = crate::git::commands::run_git_fetch(&path).await {
                        tracing::debug!(error = %e, path = %path.display(), "git fetch failed");
                    }
                }
                crate::git::commands::run_git_behind_ahead(&path)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |res| GitMessage::BehindAhead(generation, res),
        )
    }

    /// Handle a workspace file-change event: coalesce bursts into at most one
    /// local refresh per [`LOCAL_REFRESH_MIN_INTERVAL`]. If the throttle window
    /// has elapsed, refresh now; otherwise mark the refresh as owed and arm a
    /// scoped deferred-refresh timer to fire it once the window elapses.
    fn on_file_changed(&mut self) -> Task<GitMessage> {
        let now = Instant::now();
        match self.fire_local_refresh(now) {
            Some(task) => task,
            None if self.workspace_path.is_some() => {
                // The throttle window has not elapsed — mark the refresh as owed
                // and arm a scoped deferred-refresh timer to fire it once the
                // window elapses.
                self.local_refresh_pending = true;
                let remaining = LOCAL_REFRESH_MIN_INTERVAL
                    .saturating_sub(
                        now.duration_since(
                            self.last_local_refresh
                                .expect("window not elapsed implies throttle set"),
                        ),
                    )
                    .max(Duration::from_millis(1));
                Task::perform(
                    // Sleep built lazily at first poll (see watch_retry_task).
                    async move { tokio::time::sleep(remaining).await },
                    |()| GitMessage::DeferredLocalRefresh,
                )
            }
            None => Task::none(),
        }
    }

    /// Fire an event-driven local refresh if the local-refresh throttle window
    /// has elapsed: record the throttle, clear any owed refresh, bump the local
    /// generation, and spawn the refresh task. Returns [`None`] when the window
    /// has not elapsed (or there is no workspace path), leaving any owed refresh
    /// pending for the scoped deferred-refresh timer.
    fn fire_local_refresh(&mut self, now: Instant) -> Option<Task<GitMessage>> {
        if self
            .last_local_refresh
            .is_none_or(|t| now.duration_since(t) >= LOCAL_REFRESH_MIN_INTERVAL)
        {
            // Only advance the throttle/generation when there is a workspace to
            // refresh — a path-less refresh would otherwise consume the throttle
            // window for nothing.
            let path = self.workspace_path.clone()?;
            self.last_local_refresh = Some(now);
            self.local_refresh_pending = false;
            self.local_generation += 1;
            Some(Self::refresh_local(path, self.local_generation))
        } else {
            None
        }
    }

    /// Refresh local + remote state promptly after a git operation (sync,
    /// branch switch/create). The watcher is silent to ref-only changes, so
    /// these paths must refresh explicitly. Marks the local throttle as
    /// just-refreshed so the watcher's echo of the operation coalesces.
    fn refresh_after_git_op(&mut self) -> Task<GitMessage> {
        let Some(path) = self.workspace_path.clone() else {
            return Task::none();
        };
        let now = Instant::now();
        self.local_generation += 1;
        self.remote_generation += 1;
        self.last_local_refresh = Some(now);
        self.local_refresh_pending = false;
        Task::batch([
            Self::refresh_local(path.clone(), self.local_generation),
            Self::refresh_remote(path, self.remote_generation, false),
        ])
    }

    /// Unwatch the current file-change subscription, if any, resolving the
    /// picker by workspace name. The engine may be gone (workspace deleted) —
    /// then there is nothing to unwatch (the picker drops its watch registry).
    fn unwatch_current(&mut self) {
        let Some(name) = self.workspace_name.clone() else {
            return;
        };
        let Some(id) = self.watch_id.take() else {
            return;
        };
        if let Some(entry) = crate::search_engine::get_engine_by_name(&name) {
            let _ = entry.picker.unwatch(id);
        }
    }

    /// Register the file-change watch on the current workspace's picker. Reuses
    /// an existing engine — never creates one as a side effect. A `WatcherNotReady`
    /// (initial scan) or missing engine sets [`Self::watch_pending`] and returns a
    /// self-rescheduling retry timer that fires [`GitMessage::WatchRetryTick`] after
    /// [`WATCH_RETRY_POLL`], within a bounded window; any other failure falls back
    /// to the periodic timer.
    fn try_register_watch(&mut self) -> Task<GitMessage> {
        self.watch_pending = false;

        let (Some(name), Some(_path)) = (&self.workspace_name, &self.workspace_path) else {
            // Personal/unnamed or no filesystem path — timer fallback.
            return Task::none();
        };
        let Some(entry) = crate::search_engine::get_engine_by_name(name) else {
            // Engine not created yet; retry once it appears (eager scan/search).
            // Anchor the retry deadline only on the first failure so the window
            // is actually bounded.
            self.watch_pending = true;
            if self.watch_retry_deadline.is_none() {
                self.watch_retry_deadline = Some(Instant::now() + WATCH_RETRY_WINDOW);
            }
            return watch_retry_task();
        };

        let picker = entry.picker.clone();
        // The callback runs on the watcher's dedicated thread and must only do
        // a non-blocking send — never lock the picker (deadlock). An empty
        // pattern watches the whole tree (the watcher already excludes `.git`,
        // gitignored paths, and Access events, so git status/index self-trigger
        // loops cannot occur).
        let tx = file_change_tx().clone();
        match picker.watch("", WatchOptions::default(), move |_id, _events| {
            let _ = tx.send(());
        }) {
            Ok(id) => {
                self.watch_id = Some(id);
                self.watch_retry_deadline = None;
                Task::none()
            }
            Err(FffError::WatcherNotReady) => {
                self.watch_pending = true;
                if self.watch_retry_deadline.is_none() {
                    self.watch_retry_deadline = Some(Instant::now() + WATCH_RETRY_WINDOW);
                }
                watch_retry_task()
            }
            Err(_) => {
                // WatcherDisabled or another error — rely on the timer fallback.
                self.watch_retry_deadline = None;
                Task::none()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_with_gen(generation: u64) -> GitState {
        let mut s = GitState::new();
        s.local_generation = generation;
        s.remote_generation = generation;
        s
    }

    // ── Refresh results: last-known-good + generation guard ───────

    #[test]
    fn test_refresh_success_applies_current_generation() {
        let mut s = state_with_gen(7);
        let _ = s.update(GitMessage::DiffStats(
            7,
            Ok(DiffStats {
                added: 3,
                removed: 1,
                huge_binary_file_count: 0,
            }),
        ));
        let _ = s.update(GitMessage::CurrentBranch(7, Ok("main".into())));
        let _ = s.update(GitMessage::BehindAhead(7, Ok((2, 5))));
        assert_eq!(
            s.diff_stats(),
            Some(DiffStats {
                added: 3,
                removed: 1,
                huge_binary_file_count: 0
            })
        );
        assert_eq!(s.current_branch(), Some("main"));
        assert_eq!(s.behind_ahead(), Some((2, 5)));
    }

    #[test]
    fn test_refresh_error_keeps_last_known_good() {
        let mut s = state_with_gen(7);
        let _ = s.update(GitMessage::DiffStats(
            7,
            Ok(DiffStats {
                added: 3,
                removed: 1,
                huge_binary_file_count: 0,
            }),
        ));
        let _ = s.update(GitMessage::CurrentBranch(7, Ok("main".into())));
        let _ = s.update(GitMessage::BehindAhead(7, Ok((2, 5))));
        // Transient failures — cached values must survive (no flicker).
        let _ = s.update(GitMessage::DiffStats(7, Err("spawn failed".into())));
        let _ = s.update(GitMessage::CurrentBranch(7, Err("spawn failed".into())));
        let _ = s.update(GitMessage::BehindAhead(7, Err("spawn failed".into())));
        assert_eq!(
            s.diff_stats(),
            Some(DiffStats {
                added: 3,
                removed: 1,
                huge_binary_file_count: 0
            })
        );
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
        let _ = s.update(GitMessage::DiffStats(
            8,
            Ok(DiffStats {
                added: 99,
                removed: 99,
                huge_binary_file_count: 0,
            }),
        ));
        let _ = s.update(GitMessage::CurrentBranch(8, Ok("old-branch".into())));
        let _ = s.update(GitMessage::BehindAhead(8, Ok((99, 99))));
        assert_eq!(s.diff_stats(), None);
        assert_eq!(s.current_branch(), None);
        assert_eq!(s.behind_ahead(), None);
    }

    #[test]
    fn test_clear_invalidates_inflight_results() {
        let mut s = state_with_gen(3);
        let _ = s.update(GitMessage::DiffStats(
            3,
            Ok(DiffStats {
                added: 3,
                removed: 1,
                huge_binary_file_count: 0,
            }),
        ));
        let _ = s.update(GitMessage::CurrentBranch(3, Ok("main".into())));
        s.clear(); // bumps both generations — in-flight results become stale
        let _ = s.update(GitMessage::DiffStats(
            3,
            Ok(DiffStats {
                added: 7,
                removed: 7,
                huge_binary_file_count: 0,
            }),
        ));
        assert_eq!(s.diff_stats(), None);
    }

    // ── Event-driven local refresh throttle ──────────────────────

    #[test]
    fn test_file_changed_coalesces_burst_within_window() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        // First event: the throttle window has no prior refresh → refresh now.
        let _ = s.on_file_changed();
        let first_gen = s.local_generation;
        assert!(!s.local_refresh_pending);
        // Second event immediately after: coalesced into a pending refresh.
        let _ = s.on_file_changed();
        assert!(s.local_refresh_pending);
        assert_eq!(
            s.local_generation, first_gen,
            "no new refresh within window"
        );
    }

    #[test]
    fn test_file_changed_fires_when_throttle_elapsed() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        s.last_local_refresh = Some(Instant::now() - Duration::from_secs(2));
        let before = s.local_generation;
        let _ = s.on_file_changed();
        assert!(!s.local_refresh_pending);
        assert_eq!(s.local_generation, before + 1);
    }

    #[test]
    fn test_file_changed_no_path_is_noop() {
        let mut s = GitState::new();
        s.workspace_path = None;
        let before = s.local_generation;
        let _ = s.on_file_changed();
        // No workspace path — the refresh is a no-op: no owed refresh, no
        // generation bump, and the throttle window is not consumed.
        assert!(!s.local_refresh_pending);
        assert_eq!(s.local_generation, before);
    }

    // ── Periodic remote sync and timer fallback ──────────────────

    #[test]
    fn test_periodic_sync_tick_runs_full_refresh_no_watch_fallback() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        // No watch and no prior remote sync → the periodic timer runs a full refresh.
        let before_local = s.local_generation;
        let before_remote = s.remote_generation;
        let _ = s.update(GitMessage::PeriodicSyncTick);
        assert_eq!(s.local_generation, before_local + 1);
        assert_eq!(s.remote_generation, before_remote + 1);
        assert!(s.last_remote_refresh.is_some());
    }

    #[test]
    fn test_periodic_sync_watched_runs_full_refresh() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        // Simulate an active watch: no pending local refresh and no prior
        // remote sync.
        s.watch_id = Some(WatchId(1));
        let before_local = s.local_generation;
        let before_remote = s.remote_generation;
        let _ = s.update(GitMessage::PeriodicSyncTick);
        // The periodic timer is the recovery path for ref-only/commit-only
        // changes (which produce no file-change event), so it refreshes local
        // (diff stats + branch) and remote (behind/ahead + fetch) even for a
        // watched workspace. The generation bumps prove both the local and
        // remote refresh paths were entered — a regression that ran only remote
        // for a watched workspace would leave `local_generation` untouched.
        assert_eq!(s.local_generation, before_local + 1);
        assert_eq!(s.remote_generation, before_remote + 1);
    }

    #[test]
    fn test_deferred_local_refresh_fires_after_window() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        s.local_refresh_pending = true;
        s.last_local_refresh = Some(Instant::now() - Duration::from_secs(2));
        // A recent remote sync means the periodic timer is quiet — only the owed
        // event-driven local refresh acts, leaving the remote generation alone.
        s.last_remote_refresh = Some(Instant::now());
        let before_local = s.local_generation;
        let before_remote = s.remote_generation;
        let _ = s.update(GitMessage::DeferredLocalRefresh);
        // The throttle window has elapsed — the owed event-driven local refresh
        // fires (local generation bumps). The remote generation is untouched.
        assert!(!s.local_refresh_pending);
        assert_eq!(s.local_generation, before_local + 1);
        assert_eq!(s.remote_generation, before_remote);
    }

    #[test]
    fn test_deferred_local_refresh_within_window_reschedules() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        // Simulate an event deferred within the throttle window.
        s.last_local_refresh = Some(Instant::now());
        s.local_refresh_pending = true;
        let before = s.local_generation;
        let _ = s.update(GitMessage::DeferredLocalRefresh);
        // A newer event restarted the throttle window — the owed refresh stays
        // pending and the timer reschedules itself (the returned Task is opaque).
        assert!(s.local_refresh_pending);
        assert_eq!(s.local_generation, before);
    }

    #[test]
    fn test_deferred_local_refresh_without_pending_is_noop() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        // Stale timestamps but nothing owed — a stale timer firing is a no-op.
        s.last_local_refresh = Some(Instant::now() - Duration::from_secs(10));
        let before = s.local_generation;
        let _ = s.update(GitMessage::DeferredLocalRefresh);
        assert!(!s.local_refresh_pending);
        assert_eq!(s.local_generation, before);
    }

    // ── Watch registration retry loop ─────────────────────────────

    #[test]
    fn test_watch_retry_loop_rearms_while_pending() {
        let mut s = GitState::new();
        s.workspace_name = Some("no-such-engine".into());
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        // First registration attempt: no engine under that name → pending,
        // the 30s window is anchored, and a re-arm task is scheduled.
        let _ = s.try_register_watch();
        assert!(s.watch_pending);
        assert!(s.watch_retry_deadline.is_some());
        // A retry tick before the deadline re-attempts; the engine is still
        // missing, so the loop re-arms (the returned task is the next timer).
        let _ = s.update(GitMessage::WatchRetryTick);
        assert!(s.watch_pending);
        assert!(s.watch_retry_deadline.is_some());
        // Deadline elapsed → the loop abandons the watch in favour of the
        // periodic timer fallback and stops re-arming.
        s.watch_retry_deadline = Some(Instant::now() - Duration::from_secs(1));
        let _ = s.update(GitMessage::WatchRetryTick);
        assert!(!s.watch_pending);
        assert!(s.watch_retry_deadline.is_none());
    }

    #[test]
    fn test_refresh_after_commit_refreshes_footer() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        s.local_refresh_pending = true;
        let before_local = s.local_generation;
        let before_remote = s.remote_generation;
        let _ = s.update(GitMessage::RefreshAfterCommit);
        // A commit is a ref-only change — the footer refreshes promptly (local +
        // behind/ahead) and the owed file-change refresh is superseded.
        assert!(s.last_local_refresh.is_some());
        assert!(!s.local_refresh_pending);
        assert_eq!(s.local_generation, before_local + 1);
        assert_eq!(s.remote_generation, before_remote + 1);
    }

    #[test]
    fn test_pipeline_commit_refreshes_footer_when_path_matches() {
        let mut s = GitState::new();
        let ws = PathBuf::from("/some/ws");
        s.workspace_path = Some(ws.clone());
        s.local_refresh_pending = true;
        let before_local = s.local_generation;
        let before_remote = s.remote_generation;
        let _ = s.update(GitMessage::PipelineCommit(ws));
        // A pipeline commit on the displayed workspace is a ref-only change — the
        // footer refreshes promptly (local + behind/ahead) and any owed
        // file-change refresh is superseded.
        assert!(s.last_local_refresh.is_some());
        assert!(!s.local_refresh_pending);
        assert_eq!(s.local_generation, before_local + 1);
        assert_eq!(s.remote_generation, before_remote + 1);
    }

    #[test]
    fn test_pipeline_commit_ignored_for_other_workspace() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/some/ws"));
        let before_local = s.local_generation;
        let before_remote = s.remote_generation;
        let _ = s.update(GitMessage::PipelineCommit(PathBuf::from("/other/ws")));
        assert_eq!(s.local_generation, before_local);
        assert_eq!(s.remote_generation, before_remote);
    }
}
