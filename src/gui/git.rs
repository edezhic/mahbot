//! Git state management — branch info, sync, diff stats, branch modal.
//!
//! Extracted from the monolithic `Dashboard::update` (mod.rs) to reduce
//! coupling between git operations and the rest of the dashboard UI.
//!
//! The Dashboard owns a single `git_state: GitState` field and routes
//! [`Message::Git`](super::Message::Git) messages to it. Cross-modal
//! coordination (diff modal ↔ branch modal mutual exclusion) is handled at the Dashboard level.

use super::dialog;
use super::theme;
use super::widgets;

use crate::git::commands::{
    DestinationProbe, DiffStats, GitFileStatus, GitPushDestination, GitRemoteState, GitSyncReport,
    GitWorktreeSnapshot,
};
use crate::git::failure::{GitFailure, GitStep};

use iced::widget::{Column, Space, button, column, container, row, text};
use iced::{Alignment, Element, Length, Task};

use fff_search::{WatchId, WatchOptions};

use std::collections::HashMap;
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

const LOCAL_REFRESH_MIN_INTERVAL: Duration = Duration::from_millis(300);

pub(crate) const REMOTE_REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// Bounded wait for a lazily-created workspace engine's file watcher to become
/// ready before registering the file-change watch. Registration happens once per
/// workspace selection — there is no retry loop.
const WATCH_READY_WAIT: Duration = Duration::from_secs(30);

pub(super) fn nowhere_message(reason: &str) -> String {
    format!("Nowhere to push: {reason}")
}

/// The logs target of a git sync failure row.
const GIT_SYNC_TARGET: &str = "git-sync";

/// The toast for a finished sync: git's own report, or — when nothing could be
/// pushed — the reason, so a push that did not happen is never reported as one that did.
fn sync_toast(report: GitSyncReport) -> super::ToastMessage {
    let output = report.output.trim();
    match report.not_pushed {
        Some(reason) => super::ToastMessage::Warning(if output.is_empty() {
            nowhere_message(&reason)
        } else {
            format!("Sync completed:\n{output}\n{}", nowhere_message(&reason))
        }),
        None if output.is_empty() => super::ToastMessage::SuccessMsg("Sync completed".to_string()),
        None => super::ToastMessage::SuccessMsg(format!("Sync completed:\n{output}")),
    }
}

/// The toast for a failed sync. Its message already opens with the step that failed, so it is
/// shown as-is rather than behind the generic `Failed: ` frame; a result that belongs to a
/// workspace no longer on screen also names it, so the toast cannot be read as a failure of what
/// is on screen now.
fn failure_toast(
    workspace: &str,
    failure: &GitFailure,
    same_workspace: bool,
) -> super::ToastMessage {
    let message = failure.toast();
    super::ToastMessage::SyncFailure(if same_workspace {
        message
    } else {
        format!("{workspace}: {message}")
    })
}

/// Git state owned by the Dashboard.
///
/// All git-related fields are encapsulated here. The Dashboard accesses
/// state via query methods (`current_branch()`, `is_modal_open()`, etc.)
/// and drives updates via [`GitMessage`].
pub struct GitState {
    workspace_path: Option<PathBuf>,
    diff_stats: Option<DiffStats>,
    file_statuses: HashMap<String, GitFileStatus>,
    current_branch: Option<String>,
    remote: Option<GitRemoteState>,

    show_branch_modal: bool,
    branch_search_query: super::common::SingleLineEditorState,
    local_branches: Vec<String>,
    syncing: bool,
    branch_error: Option<String>,
    new_branch_name: super::common::SingleLineEditorState,

    /// The last sync-path failure on this branch: the pull, the push, or asking the destination
    /// what it carries. The toast is gone after four seconds and the tooltip cannot be read at
    /// length, so the branch modal shows this in full until the next sync attempt replaces it.
    last_sync_failure: Option<GitFailure>,
    /// The identity of the sync press in flight. Its own result is what may release
    /// [`Self::syncing`], so a result for a press a selection change abandoned never releases a
    /// control whose newer press is still running.
    sync_press: Option<u64>,
    /// The next press identity: monotonic and never reset, so no two presses share one.
    sync_press_seq: u64,
    /// The reason already filed in the logs per workspace, keyed by the workspace it belongs to.
    /// The destination probe runs on a timer, so one reason is filed once — again only when it
    /// changes or after a success in between — instead of a row a minute into a store whose
    /// warnings are never pruned, and returning to a workspace does not re-file what it filed.
    filed_failure: HashMap<String, String>,

    /// Workspace name for the currently selected workspace, used to resolve the
    /// per-workspace [`SharedFilePicker`](fff_search::SharedFilePicker) watch
    /// by name (the registry is keyed by workspace name, not path). `Some`
    /// exactly when [`Self::has_filesystem_path`] is.
    workspace_name: Option<String>,
    /// One-shot flag: a [`GitMessage::WorktreeSnapshot`] result was applied for
    /// the current generation (Ok or Err), so [`Self::take_status_forward`]
    /// should hand the editor the (possibly last-known-good) statuses. Set on
    /// every current-generation snapshot result; reset by [`Self::clear`],
    /// [`Self::set_workspace_path`], and [`Self::take_status_forward`].
    statuses_dirty: bool,
    /// Generation for local refreshes (diff stats + current branch). Each
    /// local refresh captures the current value and only applies while it is
    /// current, so stale/out-of-order results never overwrite newer state.
    local_generation: u64,
    remote_generation: u64,
    /// When the last event-driven local refresh ran — the throttle gate for
    /// coalescing bursts of file-change events.
    last_local_refresh: Option<Instant>,
    /// A file-change event arrived during the throttle window and a local
    /// refresh is still owed; fired by the scoped deferred-refresh timer once the window elapses.
    local_refresh_pending: bool,
    /// When the last remote sync (remote state + fetch) ran. Only updated by an
    /// actual remote sync — a workspace switch alone does not defer the periodic timer.
    last_remote_refresh: Option<Instant>,
    /// The active file-change watch subscription id. `Some` only when a watch
    /// was successfully registered on the current workspace's picker.
    watch_id: Option<WatchId>,
}

/// Messages for the git sub-state.
///
/// Analogous to [`DiffMessage`](super::diff::DiffMessage) — the Dashboard
/// wraps these in [`Message::Git`](super::Message::Git) and routes them to [`GitState::update`].
#[derive(Debug, Clone)]
pub enum GitMessage {
    /// Result of `run_git_worktree_snapshot`. Carries the refresh generation;
    /// results from a superseded generation are discarded. `Ok` on genuine
    /// success (including a clean tree); `Err` on transient failure keeps
    /// the cached last-known-good value. Either outcome marks the statuses
    /// for forwarding to the editor so its in-flight flag cannot stick.
    WorktreeSnapshot(u64, Result<GitWorktreeSnapshot, String>),
    CurrentBranch(u64, Result<String, String>),
    /// Result of `run_git_remote_state`. Carries behind (against the branch's
    /// own update source) plus where a push would go; an in-sync state is a
    /// genuine result and is stored, and `Ok(None)` means `HEAD` names no branch
    /// (nothing to sync). `Err` keeps the last-known-good state. Same generation
    /// semantics as [`GitMessage::WorktreeSnapshot`] but against
    /// [`GitState::remote_generation`].
    RemoteState(u64, Result<Option<GitRemoteState>, String>),

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

    /// Periodic 60s tick from the standalone remote-sync subscription. Runs the
    /// remote-state + `git fetch` sync and also refreshes the local footer — the
    /// recovery path for ref-only/commit-only changes (bare commit, fetch, push,
    /// `checkout -c`) that produce no file-change event. Runs for every
    /// workspace, watched or not.
    PeriodicSyncTick,
    /// Fired by the scoped self-rescheduling timer armed when a file-change event
    /// was deferred by the 300ms local-refresh throttle. Fires the owed local
    /// refresh, or reschedules itself when a newer event restarted the throttle
    /// window. A stale timer firing with no owed refresh is a no-op.
    DeferredLocalRefresh,
    /// Result of a deferred file-change watch registration. Carries the
    /// registration generation; results from a superseded generation (a
    /// workspace switch raced the deferred registration) are discarded.
    WatchRegistered(u64, Result<WatchId, String>),

    ListBranches(Result<Vec<String>, String>),

    OpenModal,
    CloseModal,

    BranchQueryChanged(super::editor_widget::EditorAction),

    Sync,
    /// A sync press finished. `id` is the press it belongs to, and `workspace`/`branch` are what
    /// it was pressed for: a result that lands after a switch belongs to what was on screen then,
    /// not now, so it is never attributed to what replaced it.
    SyncResult {
        id: u64,
        workspace: String,
        branch: Option<String>,
        result: Result<GitSyncReport, GitFailure>,
    },
    /// Copy the remembered failure's full text, which the modal also shows selectable.
    CopyFailure,

    Switch(String),
    SwitchResult(Result<(), String>),

    Create,
    CreateBranchResult(Result<(), String>),
    NewBranchNameChanged(super::editor_widget::EditorAction),

    Toast(super::ToastMessage),
}

impl GitState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            workspace_path: None,
            diff_stats: None,
            file_statuses: HashMap::new(),
            current_branch: None,
            remote: None,
            show_branch_modal: false,
            branch_search_query: super::common::SingleLineEditorState::new(""),
            local_branches: Vec::new(),
            syncing: false,
            branch_error: None,
            new_branch_name: super::common::SingleLineEditorState::new(""),
            last_sync_failure: None,
            sync_press: None,
            sync_press_seq: 0,
            filed_failure: HashMap::new(),
            workspace_name: None,
            statuses_dirty: false,
            local_generation: 0,
            remote_generation: 0,
            last_local_refresh: None,
            local_refresh_pending: false,
            last_remote_refresh: None,
            watch_id: None,
        }
    }

    #[must_use]
    pub fn diff_stats(&self) -> Option<DiffStats> {
        self.diff_stats
    }

    #[must_use]
    pub fn current_branch(&self) -> Option<&str> {
        self.current_branch.as_deref()
    }

    #[must_use]
    pub fn remote_state(&self) -> Option<&GitRemoteState> {
        let state = self.remote.as_ref()?;
        (self.current_branch.as_deref() == Some(state.branch.as_str())).then_some(state)
    }

    #[must_use]
    pub fn is_syncing(&self) -> bool {
        self.syncing
    }

    #[must_use]
    pub fn is_modal_open(&self) -> bool {
        self.show_branch_modal
    }

    /// Whether a workspace filesystem path is set, i.e. the dashboard resolved a
    /// workspace for the current selection: git operations can proceed and the
    /// footer git status may render. Set and cleared only by
    /// [`Self::set_workspace_path`], so it tracks the resolution.
    #[must_use]
    pub fn has_filesystem_path(&self) -> bool {
        self.workspace_path.is_some()
    }

    /// Clear all cached git info (diff stats, per-file statuses, branch, remote state, modal, and
    /// the remembered sync failure).
    /// Does **not** clear `workspace_path`, `workspace_name`, the watch state, the refresh
    /// throttle/timer state, or the filed-failure record (which is per workspace and outlives a
    /// selection) — the first four are managed explicitly by [`Self::set_workspace_path`] and the
    /// periodic/deferred timers, the last by `file_failure` and the sync result arms.
    /// Bumps both generations so refresh results still in flight for the previous state
    /// are discarded as stale.
    pub fn clear(&mut self) {
        self.diff_stats = None;
        self.file_statuses.clear();
        self.statuses_dirty = false;
        self.current_branch = None;
        self.remote = None;
        self.local_branches.clear();
        self.branch_search_query.clear();
        self.branch_error = None;
        self.show_branch_modal = false;
        self.new_branch_name.clear();
        self.syncing = false;
        // A failure belongs to the branch that produced it: the new selection starts without
        // one, and a press still in flight for the previous workspace no longer owns the
        // control (its result is still toasted and filed for its own workspace).
        self.last_sync_failure = None;
        self.sync_press = None;
        self.local_generation += 1;
        self.remote_generation += 1;
    }

    /// Set the workspace name and filesystem path, and trigger an eager refresh
    /// of git info (diff stats, per-file statuses, branch, remote state for the new workspace).
    /// Clears all cached state first and re-registers the file-change watch for
    /// the new workspace (one deferred attempt when the engine's watcher is
    /// ready — no retry loop; registration failure is a WARN and the workspace
    /// stays unwatched until the next switch).
    ///
    /// `name` is `Some` only for a workspace the dashboard resolved; `None`
    /// means nothing is resolved (the footer git status is hidden then), so
    /// there is nothing to watch.
    ///
    /// Returns a batch of [`Task`]s that produce [`GitMessage`] results
    /// when the async operations complete.
    pub fn set_workspace_path(
        &mut self,
        name: Option<String>,
        path: Option<String>,
    ) -> Task<GitMessage> {
        self.unwatch_current();
        self.clear();
        self.workspace_name = name;
        self.workspace_path = path.map(PathBuf::from);
        // Reset the local throttle for the new workspace (nothing refreshed yet).
        self.last_local_refresh = None;
        self.local_refresh_pending = false;
        // Register the file-change watch (one deferred attempt — no retry).
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

    /// Resolve the statuses to forward to the editor, if the last
    /// current-generation snapshot applied (Ok or Err) has not yet been
    /// claimed. Returns `Some((workspace_name, statuses))` once, resetting the
    /// one-shot flag; `None` when nothing new is pending.
    ///
    /// `workspace_name` is the workspace the snapshot belongs to (`None` while
    /// nothing is resolved) — passed through so the editor can drop a forward
    /// that no longer matches its selected workspace.
    pub(crate) fn take_status_forward(
        &mut self,
    ) -> Option<(Option<String>, HashMap<String, GitFileStatus>)> {
        if !self.statuses_dirty {
            return None;
        }
        self.statuses_dirty = false;
        Some((self.workspace_name.clone(), self.file_statuses.clone()))
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: GitMessage) -> Task<GitMessage> {
        match msg {
            GitMessage::WorktreeSnapshot(generation, result) => {
                // Only apply current-generation snapshots. On `Ok` store both
                // the stats and the per-file statuses; on `Err` keep the
                // last-known-good values so the footer controls don't flicker.
                // Either outcome flags the statuses for forwarding so the
                // editor re-renders with last-known-good (its in-flight flag
                // must never stick on a transient error).
                //
                // A stale-generation snapshot is dropped without forwarding —
                // safe because the generation only moves when a newer refresh
                // (workspace switch, throttle fire, periodic sync) starts, and
                // that newer refresh produces its own current-generation
                // forward. The editor's in-flight guard therefore resolves
                // with the superseding snapshot, never the dropped one.
                if generation == self.local_generation {
                    if let Ok(snapshot) = result {
                        self.diff_stats = Some(snapshot.stats);
                        self.file_statuses = snapshot.file_statuses;
                    }
                    self.statuses_dirty = true;
                }
                Task::none()
            }
            GitMessage::CurrentBranch(generation, result) => {
                if generation == self.local_generation
                    && let Ok(branch) = result
                {
                    // A failure belongs to the branch that was checked out when it happened.
                    if self
                        .current_branch
                        .as_deref()
                        .is_some_and(|known| known != branch)
                    {
                        self.last_sync_failure = None;
                    }
                    self.current_branch = Some(branch);
                }
                Task::none()
            }
            GitMessage::RemoteState(generation, result) => {
                if generation == self.remote_generation
                    && let Ok(state) = result
                {
                    // The probe describes the branch it asked for: a state that arrived after the
                    // checked-out branch changed is not what the modal or the log may attribute
                    // here. Before the branch has resolved (a switch batches both refreshes) the
                    // current generation is the evidence there is, and it does belong to this
                    // workspace — gating on a branch that is not known yet would drop the first
                    // failure the owner sees after selecting a workspace.
                    let same_branch = state.as_ref().is_some_and(|state| {
                        self.current_branch
                            .as_deref()
                            .is_none_or(|known| known == state.branch)
                    });
                    if let Some(GitPushDestination::Target(target)) =
                        state.as_ref().map(|state| &state.push)
                        && same_branch
                    {
                        match &target.probe {
                            DestinationProbe::Unknown {
                                failure: Some(failure),
                            } => self.remember_probe_failure(failure),
                            // A probe that answered is the success in between that makes a
                            // later failure of the same reason news again; one that never ran
                            // (another fetch held the ref locks) proves nothing either way.
                            DestinationProbe::Carries | DestinationProbe::Absent => {
                                if let Some(workspace) = &self.workspace_name {
                                    self.filed_failure.remove(workspace);
                                }
                            }
                            DestinationProbe::Unknown { failure: None } => {}
                        }
                    }
                    self.remote = state;
                }
                Task::none()
            }

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

            GitMessage::PeriodicSyncTick => {
                // Periodic remote sync (remote state + fetch). The timer also
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
                // firing a no-op, so only the timer responsible for the owed refresh acts.
                if !self.local_refresh_pending {
                    return Task::none();
                }
                let now = Instant::now();
                match self.fire_local_refresh(now) {
                    Some(task) => task,
                    None if self.workspace_path.is_none() => {
                        self.local_refresh_pending = false;
                        Task::none()
                    }
                    None => {
                        // A newer event restarted the throttle window — reschedule.
                        self.arm_deferred_refresh(now)
                    }
                }
            }
            GitMessage::WatchRegistered(generation, result) => {
                // Only apply current-generation registrations — a workspace
                // switch raced the deferred registration and superseded it.
                if generation != self.local_generation {
                    return Task::none();
                }
                match result {
                    Ok(id) => {
                        self.watch_id = Some(id);
                        tracing::debug!(
                            ?id,
                            workspace = %self.workspace_name.as_deref().unwrap_or("<none>"),
                            "file-change watch registered",
                        );
                    }
                    Err(e) => {
                        // Registration failed (watcher not ready within the
                        // bounded wait, or the engine could not be created).
                        // No retry: the workspace stays unwatched (periodic
                        // sync still refreshes the footer) until the next
                        // workspace switch re-attempts.
                        tracing::warn!(
                            error = %e,
                            workspace = %self.workspace_name.as_deref().unwrap_or("<none>"),
                            "file-change watch registration failed — workspace stays unwatched \
                             until the next workspace switch",
                        );
                    }
                }
                Task::none()
            }

            GitMessage::ListBranches(result) => {
                match result {
                    Ok(branches) => self.local_branches = branches,
                    Err(e) => self.branch_error = Some(e),
                }
                Task::none()
            }

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

            GitMessage::BranchQueryChanged(action) => {
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                self.branch_search_query.apply_action(action);
                Task::none()
            }

            GitMessage::Sync => {
                if self.syncing {
                    return Task::none();
                }
                // A press needs a workspace to sync; the branch is only how the result is
                // attributed, and the first tick after a workspace switch may not have resolved
                // it yet.
                let (Some(path), Some(workspace)) =
                    (self.workspace_path.clone(), self.workspace_name.clone())
                else {
                    return Task::none();
                };
                let branch = self.current_branch.clone();
                self.syncing = true;
                self.sync_press_seq += 1;
                let id = self.sync_press_seq;
                self.sync_press = Some(id);
                Task::perform(
                    async move { crate::git::commands::run_git_sync(&path).await },
                    move |result| GitMessage::SyncResult {
                        id,
                        workspace,
                        branch,
                        result,
                    },
                )
            }
            GitMessage::SyncResult {
                id,
                workspace,
                branch,
                result,
            } => {
                // Only the press in flight releases its own control: a result for a press that a
                // selection change already abandoned must not re-enable a control whose newer
                // press is still running, and a press whose branch was switched under it must
                // not leave the control disabled.
                if self.sync_press == Some(id) {
                    self.sync_press = None;
                    self.syncing = false;
                }
                // Whatever the screen shows now, the result belongs to the workspace and branch
                // that produced it: it is toasted, and a failure is filed for that workspace.
                // Only the branch modal is scoped to what is on screen.
                let same_workspace = self.workspace_name.as_deref() == Some(workspace.as_str());
                let on_screen = same_workspace
                    && branch
                        .as_deref()
                        .is_none_or(|pressed| self.current_branch.as_deref() == Some(pressed));
                match result {
                    Ok(report) => {
                        // The attempt succeeded: the remembered failure is over, and a later
                        // recurrence of the same reason is news again.
                        self.filed_failure.remove(&workspace);
                        if !on_screen {
                            return Task::done(GitMessage::Toast(sync_toast(report)));
                        }
                        self.last_sync_failure = None;
                        Task::batch([
                            Task::done(GitMessage::Toast(sync_toast(report))),
                            // A sync (pull/push) moved refs and possibly files, so
                            // refresh promptly — the watcher is silent to ref-only changes.
                            self.refresh_after_git_op(),
                        ])
                    }
                    Err(failure) => {
                        self.file_failure(&workspace, &failure);
                        // A press's own result replaces whatever the modal showed, and is what
                        // the screen may attribute to the branch it was pressed on.
                        if on_screen {
                            self.last_sync_failure = Some(failure.clone());
                        }
                        Task::done(GitMessage::Toast(failure_toast(
                            &workspace,
                            &failure,
                            same_workspace,
                        )))
                    }
                }
            }
            GitMessage::CopyFailure => self
                .last_sync_failure
                .as_ref()
                .map_or_else(Task::none, |failure| {
                    iced::clipboard::write(failure.message())
                }),

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

            GitMessage::Toast(_) => Task::none(),
        }
    }

    /// The remembered sync failure for the branch modal: the claim, git's reason, and a copy
    /// control. The text is selectable and already credential-scrubbed and length-capped, and the
    /// bounded scroll keeps a long report from stretching the dialog.
    fn failure_block(&self) -> Element<'_, GitMessage> {
        let Some(failure) = self.last_sync_failure.as_ref() else {
            return container(text("")).into();
        };
        column![
            row![
                text("Last sync failure")
                    .size(theme::TEXT_12)
                    .color(theme::STATUS_ERROR),
                Space::new().width(Length::Fill),
                button(
                    text("Copy")
                        .size(theme::TEXT_11)
                        .color(theme::TEXT_SECONDARY),
                )
                .padding([theme::PAD_2, theme::PAD_6])
                .style(theme::button_text)
                .on_press(GitMessage::CopyFailure),
            ]
            .align_y(Alignment::Center),
            super::widgets::vscroll_sized(
                super::widgets::selectable_text(failure.message(), theme::STATUS_ERROR)
                    .size(theme::TEXT_12)
                    .width(Length::Fill),
                Length::Fill,
                Length::Fixed(96.0),
            ),
        ]
        .spacing(theme::SPACE_4)
        .into()
    }

    /// Keep the destination probe's failure for the branch modal and file it for the workspace on
    /// screen.
    ///
    /// The probe runs every minute, so its failure must not bury what the owner's own press
    /// reported: it never replaces a press's result, while a press replaces whatever was there.
    fn remember_probe_failure(&mut self, failure: &GitFailure) {
        let presses_own = self
            .last_sync_failure
            .as_ref()
            .is_some_and(|kept| kept.step() != GitStep::Probe);
        if !presses_own {
            self.last_sync_failure = Some(failure.clone());
        }
        let Some(workspace) = self.workspace_name.clone() else {
            return;
        };
        self.file_failure(&workspace, failure);
    }

    /// File the failure in the product's logs, attributed to `workspace` — the one it belongs to,
    /// which is not necessarily the one on screen — and once per distinct cause. The reason is
    /// already scrubbed of credentials and length-capped, so the never-pruned row cannot grow
    /// with a pathological report, and the record is kept per workspace, so a result that arrives
    /// after a switch is still filed without disturbing the workspace on screen.
    fn file_failure(&mut self, workspace: &str, failure: &GitFailure) {
        let key = failure.dedupe_key();
        if self.filed_failure.get(workspace) == Some(&key) {
            return;
        }
        crate::logs::record_warn_row(
            GIT_SYNC_TARGET,
            &failure.message(),
            serde_json::json!({
                "step": failure.step().label(),
                "kind": failure.kind().label(),
                "remote": failure.remote(),
            }),
            workspace,
        );
        self.filed_failure.insert(workspace.to_string(), key);
    }

    fn finish_branch_op(
        &mut self,
        result: Result<(), String>,
        success_msg: &str,
    ) -> Task<GitMessage> {
        self.syncing = false;
        match result {
            Ok(()) => {
                self.show_branch_modal = false;
                // The failure belonged to the branch that was checked out then.
                self.last_sync_failure = None;
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

    /// Render the branch management dialog content (search, branch list,
    /// create section) with the corner close cross. Does **not** wrap in a
    /// modal backdrop — the Dashboard wraps it in a small centered dialog so
    /// the close message is consistent with the rest of the overlay stack.
    pub fn view(&self) -> Element<'_, GitMessage> {
        let search_input = super::widgets::single_line_editor(
            &self.branch_search_query.buffer,
            "Search branches…",
            false,
            Length::Fill,
            Some(iced::widget::Id::new("git_branch_search")),
            GitMessage::BranchQueryChanged,
        );

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
                button(
                    text(b.clone())
                        .size(theme::TEXT_14)
                        .color(theme::TEXT_PRIMARY),
                )
                .padding([theme::PAD_6, theme::PAD_4])
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

        let list = super::widgets::vscroll_sized(
            Column::with_children(branch_items).spacing(theme::SPACE_2),
            Length::Fill,
            Length::Fixed(300.0),
        );

        let error_elem: Element<'_, GitMessage> = if let Some(ref err) = self.branch_error {
            super::widgets::scroll_h_inset(
                text(err).size(theme::TEXT_12).color(theme::STATUS_ERROR),
            )
        } else {
            container(text("")).into()
        };

        // The last sync failure, in full: the toast is gone in four seconds and the sync
        // control's tooltip cannot be read at length or copied, so this is where the reason
        // stays.
        let failure_elem = self.failure_block();

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

        let create_btn = button(
            text("Create & Switch")
                .size(theme::TEXT_14)
                .color(theme::TEXT_PRIMARY),
        )
        .padding([theme::PAD_6, theme::PAD_12])
        .style(theme::button_secondary)
        .on_press_maybe(if self.syncing {
            None
        } else {
            Some(GitMessage::Create)
        });

        column![
            dialog::dialog_title_row(widgets::section_heading("Branches"), GitMessage::CloseModal),
            Space::new().height(8),
            search_input,
            Space::new().height(8),
            list,
            error_elem,
            failure_elem,
            Space::new().height(8),
            row![create_input, create_btn]
                .spacing(theme::SPACE_8)
                .align_y(Alignment::Center),
        ]
        .spacing(0)
        .width(Length::Fill)
        .into()
    }
}

async fn with_ws_path<T>(
    ws_path: Option<PathBuf>,
    op: impl AsyncFnOnce(PathBuf) -> Result<T, String>,
) -> Result<T, String> {
    match ws_path {
        Some(path) => op(path).await,
        None => Err("No workspace path".to_string()),
    }
}

/// One deferred attempt to register the file-change watch on a workspace's
/// picker, blocking off the UI thread until the engine's watcher is ready.
///
/// Lazily initialises the engine (creating it + spawning the background scan)
/// if it does not already exist, then registers an empty-pattern watch over the
/// whole tree. The engine may already exist from an agent search; reusing it is
/// the common path. On `WatcherNotReady` (initial scan still running) or engine
/// creation failure, returns `Err` — there is no retry loop; the caller warns
/// and the workspace stays unwatched until the next switch (the periodic sync
/// still refreshes the footer).
///
/// The watcher callback runs on the watcher's dedicated thread and must only do
/// a non-blocking send — never lock the picker (deadlock).
async fn register_watch_task(name: String, path: String) -> Result<WatchId, String> {
    tokio::task::spawn_blocking(move || {
        let entry = match crate::workspace_index::get_engine_by_name(&name) {
            Some(entry) => entry,
            None => crate::workspace_index::get_or_init_engine(&name, std::path::Path::new(&path))?,
        };
        if !entry.picker.wait_for_watcher(WATCH_READY_WAIT) {
            return Err(format!(
                "watcher not ready within {}s for workspace {name}",
                WATCH_READY_WAIT.as_secs()
            ));
        }
        let picker = entry.picker.clone();
        let tx = file_change_tx().clone();
        picker
            .watch("", WatchOptions::default(), move |_id, _events| {
                let _ = tx.send(());
            })
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("watch registration task join error: {e}"))?
}

impl GitState {
    /// Spawn two parallel async tasks to refresh the local working-tree state
    /// (diff stats + per-file statuses + current branch). Results carry
    /// `generation` and are applied by [`Self::update`] only while the local
    /// generation is current — stale or out-of-order results are dropped.
    /// Transient git failures surface as `Err` and leave the cached
    /// last-known-good values untouched.
    fn refresh_local(path: PathBuf, generation: u64) -> Task<GitMessage> {
        if crate::git::commands::resolve_git_top_level(&path).is_none() {
            return Task::none();
        }

        let stats_path = path.clone();
        let stats_task = Task::perform(
            async move {
                crate::git::commands::run_git_worktree_snapshot(&stats_path)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |res| GitMessage::WorktreeSnapshot(generation, res),
        );

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
    /// followed by the branch's remote state. `fetch=true` first updates the
    /// remote-tracking refs so the state reflects upstream. A fetch failure
    /// (offline/timeout/no remote) is logged at debug level and the state is
    /// still computed. Resolving a push destination that is not the branch's
    /// tracked remote asks that remote directly, so its count never depends on
    /// what happens to be fetched.
    fn refresh_remote(path: PathBuf, generation: u64, fetch: bool) -> Task<GitMessage> {
        if crate::git::commands::resolve_git_top_level(&path).is_none() {
            return Task::none();
        }
        Task::perform(
            async move {
                if fetch {
                    if let Err(e) = crate::git::commands::run_git_fetch(&path).await {
                        tracing::debug!(error = %e, path = %path.display(), "git fetch failed");
                    }
                }
                crate::git::commands::run_git_remote_state(&path)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |res| GitMessage::RemoteState(generation, res),
        )
    }

    /// Handle a workspace file-change event: coalesce bursts into at most one
    /// local refresh per [`LOCAL_REFRESH_MIN_INTERVAL`]. If the throttle window
    /// has elapsed, refresh now; otherwise mark the refresh as owed and arm the
    /// deferred-refresh timer.
    fn on_file_changed(&mut self) -> Task<GitMessage> {
        let now = Instant::now();
        match self.fire_local_refresh(now) {
            Some(task) => task,
            None if self.workspace_path.is_some() => {
                self.local_refresh_pending = true;
                self.arm_deferred_refresh(now)
            }
            None => Task::none(),
        }
    }

    /// Arm the scoped timer that fires the owed local refresh once the remaining
    /// throttle window (floored at 1 ms) elapses. A window that has not elapsed
    /// implies the throttle was recorded; the fallback duration degrades a
    /// violated invariant instead of panicking the GUI update loop.
    fn arm_deferred_refresh(&self, now: Instant) -> Task<GitMessage> {
        debug_assert!(
            self.last_local_refresh.is_some(),
            "window not elapsed implies throttle set"
        );
        let remaining = self
            .last_local_refresh
            .map_or(Duration::from_millis(1), |last| {
                LOCAL_REFRESH_MIN_INTERVAL
                    .saturating_sub(now.duration_since(last))
                    .max(Duration::from_millis(1))
            });
        // Sleep constructed lazily at first poll (inside the async block) —
        // creating it eagerly here would panic off the tokio runtime context.
        Task::perform(async move { tokio::time::sleep(remaining).await }, |()| {
            GitMessage::DeferredLocalRefresh
        })
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

    fn unwatch_current(&mut self) {
        let Some(name) = self.workspace_name.clone() else {
            return;
        };
        let Some(id) = self.watch_id.take() else {
            return;
        };
        if let Some(entry) = crate::workspace_index::get_engine_by_name(&name) {
            let _ = entry.picker.unwatch(id);
        }
    }

    /// Register the file-change watch on the current workspace's picker.
    ///
    /// Requires a resolved (name, path) pair — `None` means no workspace is
    /// selected. Returns a single deferred [`Task`] that, once the engine's
    /// watcher is ready, registers an empty-pattern whole-tree watch. This is
    /// ONE attempt, not a retry loop: if the engine must be created lazily or
    /// the watcher is not ready within [`WATCH_READY_WAIT`], the resulting
    /// [`GitMessage::WatchRegistered`] carries the error and the workspace stays
    /// unwatched until the next switch.
    fn try_register_watch(&mut self) -> Task<GitMessage> {
        let (Some(name), Some(path)) = (
            self.workspace_name.clone(),
            self.workspace_path
                .clone()
                .map(|p| p.to_string_lossy().to_string()),
        ) else {
            return Task::none();
        };
        let generation = self.local_generation;
        Task::perform(register_watch_task(name, path), move |res| {
            GitMessage::WatchRegistered(generation, res)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::commands::{GitPushDestination, GitPushTarget};
    use crate::git::failure::GitFailureKind;
    use Watch::{Failed, Register, Registered};

    fn state_with_gen(generation: u64) -> GitState {
        let mut s = GitState::new();
        s.local_generation = generation;
        s.remote_generation = generation;
        s
    }

    fn snapshot(added: i64, removed: i64) -> GitWorktreeSnapshot {
        GitWorktreeSnapshot {
            stats: DiffStats {
                added,
                removed,
                huge_binary_file_count: 0,
                omitted_file_count: 0,
            },
            file_statuses: HashMap::new(),
            unborn_head: false,
            changes: Vec::new(),
            repo_root: PathBuf::new(),
        }
    }

    fn origin_target(commits: usize) -> GitPushDestination {
        GitPushDestination::Target(GitPushTarget {
            remote: "origin".into(),
            commits,
            probe: DestinationProbe::Carries,
        })
    }

    const FAILED: &str = "spawn failed";

    fn remote(branch: &str, behind: usize) -> GitRemoteState {
        GitRemoteState {
            branch: branch.into(),
            behind,
            push: origin_target(5),
        }
    }

    /// A remote state whose destination probe failed (`Some`) or never ran (`None`).
    fn probe_state(failure: Option<GitFailure>) -> GitRemoteState {
        GitRemoteState {
            branch: "main".into(),
            behind: 0,
            push: GitPushDestination::Target(GitPushTarget {
                remote: "origin".into(),
                commits: 1,
                probe: DestinationProbe::Unknown { failure },
            }),
        }
    }

    fn status_map(statuses: bool) -> HashMap<String, GitFileStatus> {
        if statuses {
            HashMap::from([("src/main.rs".to_string(), GitFileStatus::Modified)])
        } else {
            HashMap::new()
        }
    }

    fn changed_snapshot() -> GitWorktreeSnapshot {
        GitWorktreeSnapshot {
            file_statuses: status_map(true),
            ..snapshot(3, 1)
        }
    }

    #[derive(Clone, Copy)]
    enum Watch {
        Register,
        Registered(u64, u64),
        Failed(u64),
    }

    #[test]
    fn test_refresh_results_apply_or_keep_last_known_good() {
        let remote = remote("main", 2);

        for (name, failures) in [
            ("success_applies_current_generation", false),
            ("error_keeps_last_known_good", true),
        ] {
            let mut s = state_with_gen(7);
            let _ = s.update(GitMessage::WorktreeSnapshot(7, Ok(snapshot(3, 1))));
            let _ = s.update(GitMessage::CurrentBranch(7, Ok("main".into())));
            let _ = s.update(GitMessage::RemoteState(7, Ok(Some(remote.clone()))));
            if failures {
                // Transient failures — cached values must survive (no flicker).
                let _ = s.update(GitMessage::WorktreeSnapshot(7, Err("spawn failed".into())));
                let _ = s.update(GitMessage::CurrentBranch(7, Err("spawn failed".into())));
                let _ = s.update(GitMessage::RemoteState(7, Err("spawn failed".into())));
            }
            let stats = s.diff_stats().map(|d| (d.added, d.removed));
            assert_eq!(stats, Some((3, 1)), "case {name}: stats");
            assert_eq!(s.current_branch(), Some("main"), "case {name}: branch");
            assert_eq!(s.remote_state(), Some(&remote), "case {name}: remote");
        }
    }

    type GatingCase<'a> = (&'a str, Option<&'a str>, Option<&'a str>, Option<&'a str>);
    /// No branch means no sync control, and a remote result shows only for the branch it describes.
    #[test]
    fn test_remote_state_gating() {
        let cases: &[GatingCase] = &[
            ("no branch checked out", None, Some("main"), None),
            ("nothing to sync with", None, None, None),
            ("gated by its branch", Some("main"), Some("next"), None),
            (
                "shown for its branch",
                Some("next"),
                Some("next"),
                Some("next"),
            ),
        ];

        for &(name, checked_out, announced, shown) in cases {
            let mut s = state_with_gen(7);
            if announced.is_none() {
                // A state shown before must be dropped by an empty result.
                let _ = s.update(GitMessage::CurrentBranch(7, Ok("main".into())));
                let _ = s.update(GitMessage::RemoteState(7, Ok(Some(remote("main", 2)))));
            }
            if let Some(branch) = checked_out {
                let _ = s.update(GitMessage::CurrentBranch(7, Ok(branch.into())));
            }
            let result = announced.map(|branch| remote(branch, 2));
            let _ = s.update(GitMessage::RemoteState(7, Ok(result)));
            let branch = s.remote_state().map(|r| r.branch.as_str());
            assert_eq!(branch, shown, "case {name}: the remote state it shows");
        }
    }

    /// A superseded generation never overwrites newer state, stale by age or by `clear()`.
    #[test]
    fn test_generation_guards() {
        let cases: &[(&str, u64, u64, bool)] = &[
            ("a stale generation", 9, 8, false),
            ("a cleared state", 3, 3, true),
        ];

        for &(name, generation, msg_gen, cleared) in cases {
            let mut s = state_with_gen(generation);
            if cleared {
                let _ = s.update(GitMessage::WorktreeSnapshot(generation, Ok(snapshot(3, 1))));
                let _ = s.update(GitMessage::CurrentBranch(generation, Ok("main".into())));
                let _ = s.update(GitMessage::RemoteState(
                    generation,
                    Ok(Some(remote("main", 2))),
                ));
                s.clear();
            }
            let _ = s.update(GitMessage::WorktreeSnapshot(msg_gen, Ok(snapshot(99, 99))));
            let _ = s.update(GitMessage::CurrentBranch(msg_gen, Ok("old".into())));
            let _ = s.update(GitMessage::RemoteState(
                msg_gen,
                Ok(Some(remote("main", 99))),
            ));
            assert_eq!(s.diff_stats(), None, "case {name}: stats");
            assert_eq!(s.current_branch(), None, "case {name}: branch");
            assert_eq!(s.remote_state(), None, "case {name}: remote");
        }
    }

    /// An accepted snapshot forwards its statuses once; a failure forwards too, a stale one never.
    #[test]
    fn test_worktree_snapshot_forwarding() {
        let cases: &[(&str, u64, u64, bool, Option<&str>)] = &[
            ("an accepted snapshot", 7, 7, true, Some("ws")),
            ("a failure", 7, 7, false, None),
            ("a stale generation", 9, 8, true, None),
        ];

        for &(name, generation, msg_gen, statuses, workspace) in cases {
            let mut s = state_with_gen(generation);
            s.workspace_name = workspace.map(Into::into);
            let result = if statuses {
                Ok(changed_snapshot())
            } else {
                Err(FAILED.into())
            };
            let _ = s.update(GitMessage::WorktreeSnapshot(msg_gen, result));
            let accepted = msg_gen == generation;
            let stats = (accepted && statuses).then_some((3, 1));
            assert_eq!(
                s.diff_stats().map(|d| (d.added, d.removed)),
                stats,
                "case {name}: stats"
            );
            let forward = accepted.then(|| (workspace.map(Into::into), status_map(statuses)));
            assert_eq!(s.take_status_forward(), forward, "case {name}: forward");
            assert_eq!(s.take_status_forward(), None, "case {name}: claimed once");
        }
    }

    /// A failure the owner's press produced stays until the next press, and a probe failure —
    /// which runs every minute — never buries it. The probe only fills an empty slot.
    #[test]
    fn test_press_failure_outlives_probe_failure() {
        let mut s = state_with_gen(7);
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        s.workspace_name = Some("ws".into());
        let press = GitFailure::new(
            GitStep::Push,
            Some("origin"),
            GitFailureKind::UpdateRejected,
            "! [rejected] main -> main (fetch first)",
        );
        let probe = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "fatal: Authentication failed",
        );

        let _ = s.update(GitMessage::CurrentBranch(7, Ok("main".into())));
        let _ = s.update(GitMessage::Sync);
        let id = s.sync_press.expect("the press is in flight");
        let _ = s.update(GitMessage::SyncResult {
            id,
            workspace: "ws".into(),
            branch: Some("main".into()),
            result: Err(press.clone()),
        });
        let _ = s.update(GitMessage::RemoteState(
            7,
            Ok(Some(probe_state(Some(probe.clone())))),
        ));
        assert_eq!(
            s.last_sync_failure,
            Some(press),
            "the press's own result is kept"
        );

        // A press that succeeds is the next attempt: the failure is over.
        let _ = s.update(GitMessage::Sync);
        let id = s.sync_press.expect("the press is in flight");
        let _ = s.update(GitMessage::SyncResult {
            id,
            workspace: "ws".into(),
            branch: Some("main".into()),
            result: Ok(GitSyncReport {
                output: "Everything up-to-date".into(),
                not_pushed: None,
            }),
        });
        assert_eq!(s.last_sync_failure, None);

        // With nothing remembered, the probe's own failure is what the modal shows. A successful
        // press arms the footer's refresh, which bumps the generation the probe result must match.
        let generation = s.remote_generation;
        let _ = s.update(GitMessage::RemoteState(
            generation,
            Ok(Some(probe_state(Some(probe.clone())))),
        ));
        assert_eq!(s.last_sync_failure, Some(probe));

        // A branch switch is not the branch that failed: the modal stops showing it.
        let generation = s.local_generation;
        let _ = s.update(GitMessage::CurrentBranch(generation, Ok("next".into())));
        assert_eq!(s.last_sync_failure, None);
    }

    /// The probe runs on a timer, so one reason is filed once — again only after a success in
    /// between — and a probe that never ran proves nothing either way. The same git text from a
    /// different step is a different fact: the log must hold the failed pull too.
    #[test]
    fn test_failure_filed_once_per_workspace() {
        let mut s = state_with_gen(7);
        s.workspace_name = Some("ws".into());
        s.current_branch = Some("main".into());
        let report = "fatal: unable to access 'https://x/': Could not resolve host: x";
        let probe = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::Unreachable,
            report,
        );

        let _ = s.update(GitMessage::CurrentBranch(7, Ok("main".into())));
        let _ = s.update(GitMessage::RemoteState(
            7,
            Ok(Some(probe_state(Some(probe.clone())))),
        ));
        let filed = s.filed_failure.get("ws").cloned();
        assert!(filed.is_some(), "the first failure is filed");

        let _ = s.update(GitMessage::RemoteState(
            7,
            Ok(Some(probe_state(Some(probe)))),
        ));
        assert_eq!(
            s.filed_failure.get("ws"),
            filed.as_ref(),
            "the same reason is not filed twice"
        );

        // The probe failing and the owner's own press failing are two facts, even when git said
        // the same thing: the row for the press must exist too.
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        let _ = s.update(GitMessage::Sync);
        let id = s.sync_press.expect("the press is in flight");
        let pull = GitFailure::new(
            GitStep::Pull,
            Some("origin"),
            GitFailureKind::Unreachable,
            report,
        );
        let _ = s.update(GitMessage::SyncResult {
            id,
            workspace: "ws".into(),
            branch: Some("main".into()),
            result: Err(pull),
        });
        let pull_filed = s.filed_failure.get("ws").cloned();
        assert!(
            pull_filed.is_some() && pull_filed != filed,
            "the press is filed"
        );

        // A probe that never ran (another fetch held the ref locks) proves nothing either way.
        let _ = s.update(GitMessage::RemoteState(7, Ok(Some(probe_state(None)))));
        assert_eq!(
            s.filed_failure.get("ws"),
            pull_filed.as_ref(),
            "a probe that never ran is not a success in between"
        );

        // A probe that answers is the success in between: the same reason is news again.
        let _ = s.update(GitMessage::RemoteState(7, Ok(Some(remote("main", 0)))));
        assert_eq!(
            s.filed_failure.get("ws"),
            None,
            "a success forgets the filed reason"
        );
    }

    /// A result that lands after a workspace switch is still the owner's own: it is filed for the
    /// workspace that produced it, and never shown for the one now on screen. Returning to a
    /// workspace does not file what it already filed.
    #[test]
    fn test_failure_after_switch_is_filed_for_its_own_workspace() {
        let mut s = state_with_gen(7);
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        s.workspace_name = Some("a".into());
        s.current_branch = Some("main".into());
        let _ = s.update(GitMessage::Sync);
        let id = s.sync_press.expect("the press is in flight");

        // The owner selects another workspace while the press is still running.
        let _ = s.set_workspace_path(Some("b".into()), Some("/nonexistent-b".into()));
        let failure = GitFailure::new(
            GitStep::Pull,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "fatal: Authentication failed",
        );
        let _ = s.update(GitMessage::SyncResult {
            id,
            workspace: "a".into(),
            branch: Some("main".into()),
            result: Err(failure.clone()),
        });
        assert_eq!(
            s.last_sync_failure, None,
            "not shown for the workspace now on screen"
        );
        assert_eq!(
            s.filed_failure.get("a"),
            Some(&failure.dedupe_key()),
            "filed for the workspace it belongs to"
        );
        assert_eq!(s.filed_failure.get("b"), None);

        // Back on "a", the periodic probe's failure is filed once; selecting another workspace
        // and returning does not file it again, because the record is per workspace and outlives
        // the selection.
        let _ = s.set_workspace_path(Some("a".into()), Some("/nonexistent".into()));
        let generation = s.remote_generation;
        let _ = s.update(GitMessage::RemoteState(
            generation,
            Ok(Some(probe_state(Some(failure.clone())))),
        ));
        let probe_key = s.filed_failure.get("a").cloned();
        assert!(probe_key.is_some(), "the probe's own failure is filed");

        let _ = s.set_workspace_path(Some("b".into()), Some("/nonexistent-b".into()));
        let _ = s.set_workspace_path(Some("a".into()), Some("/nonexistent".into()));
        let generation = s.remote_generation;
        let _ = s.update(GitMessage::RemoteState(
            generation,
            Ok(Some(probe_state(Some(failure)))),
        ));
        assert_eq!(
            s.filed_failure.get("a"),
            probe_key.as_ref(),
            "returning to a workspace does not re-file it"
        );
    }

    /// A press releases the control only when its own result lands: a branch switched under it
    /// must not leave the control disabled, and a superseded press must not release a newer one.
    #[test]
    fn test_sync_result_releases_its_own_press() {
        let mut s = state_with_gen(7);
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        s.workspace_name = Some("ws".into());
        s.current_branch = Some("main".into());
        let failure = GitFailure::new(
            GitStep::Push,
            Some("origin"),
            GitFailureKind::Local,
            "no branch is checked out",
        );

        let _ = s.update(GitMessage::Sync);
        assert!(s.is_syncing(), "the press is in flight");
        let first = s.sync_press.expect("the press is in flight");
        // The branch changed under the press (an external checkout): the result releases the
        // control but is not what the modal shows for the branch on screen.
        let _ = s.update(GitMessage::CurrentBranch(7, Ok("next".into())));
        let _ = s.update(GitMessage::SyncResult {
            id: first,
            workspace: "ws".into(),
            branch: Some("main".into()),
            result: Err(failure.clone()),
        });
        assert!(!s.is_syncing(), "the press that ended releases the control");
        assert_eq!(
            s.last_sync_failure, None,
            "the previous branch's failure is not shown"
        );

        // A press whose result was superseded does not release the newer press's control.
        let _ = s.update(GitMessage::Sync);
        assert!(s.is_syncing());
        let second = s.sync_press.expect("the press is in flight");
        assert_ne!(first, second, "every press has its own identity");
        let _ = s.update(GitMessage::SyncResult {
            id: first,
            workspace: "ws".into(),
            branch: Some("main".into()),
            result: Err(failure.clone()),
        });
        assert!(
            s.is_syncing(),
            "a superseded press does not release the one in flight"
        );
        let _ = s.update(GitMessage::SyncResult {
            id: second,
            workspace: "ws".into(),
            branch: None,
            result: Err(failure),
        });
        assert!(!s.is_syncing(), "its own result releases the control");
    }

    /// A result that belongs to a workspace no longer on screen names it, so the toast cannot be
    /// read as a failure of what is on screen now. The payload is already a full sentence, so the
    /// generic `Failed: ` frame is not applied to it.
    #[test]
    fn test_failure_toast_names_the_workspace_it_belongs_to() {
        let failure = GitFailure::new(
            GitStep::Push,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "fatal: Authentication failed",
        );
        let crate::gui::ToastMessage::SyncFailure(here) = failure_toast("a", &failure, true) else {
            panic!("a failed sync is an error toast");
        };
        assert_eq!(here, failure.toast());
        assert!(
            !here.starts_with("Failed"),
            "the toast frame supplies that wording: {here}"
        );

        let crate::gui::ToastMessage::SyncFailure(elsewhere) = failure_toast("b", &failure, false)
        else {
            panic!("a failed sync is an error toast");
        };
        assert_eq!(elsewhere, format!("b: {}", failure.toast()));
    }

    /// A sync press needs only a workspace: the branch is how the result is attributed, and the
    /// first tick after a workspace switch may not have resolved it yet.
    #[test]
    fn test_sync_press_runs_before_the_branch_resolves() {
        let mut s = state_with_gen(7);
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        s.workspace_name = Some("ws".into());
        let _ = s.update(GitMessage::Sync);
        assert!(s.is_syncing(), "the press runs without a resolved branch");
        assert!(s.sync_press.is_some());

        // Nothing resolved at all: there is no workspace to sync.
        let mut idle = state_with_gen(7);
        let _ = idle.update(GitMessage::Sync);
        assert!(!idle.is_syncing());
    }

    /// The first probe failure after a workspace switch is kept even though the branch has not
    /// resolved yet: the generation already says the state belongs to this workspace.
    #[test]
    fn test_first_probe_failure_before_branch_resolves_is_kept() {
        let mut s = state_with_gen(7);
        s.workspace_name = Some("ws".into());
        let failure = GitFailure::new(
            GitStep::Probe,
            Some("origin"),
            GitFailureKind::AccessRefused,
            "fatal: Authentication failed",
        );
        let _ = s.update(GitMessage::RemoteState(
            7,
            Ok(Some(probe_state(Some(failure.clone())))),
        ));
        assert_eq!(s.last_sync_failure, Some(failure));
        assert!(s.filed_failure.contains_key("ws"));
    }

    #[test]
    fn test_file_changed_coalesces_burst_within_window() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        // First event: the throttle window has no prior refresh → refresh now.
        let _ = s.on_file_changed();
        let first_gen = s.local_generation;
        assert!(!s.local_refresh_pending);
        let _ = s.on_file_changed();
        assert!(s.local_refresh_pending);
        assert_eq!(
            s.local_generation, first_gen,
            "no new refresh within window"
        );
    }

    #[expect(clippy::unchecked_time_subtraction)] // fixture shifts a monotonic clock backwards by a small fixed offset
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

    #[test]
    fn test_periodic_sync_watched_runs_full_refresh() {
        let mut s = GitState::new();
        s.workspace_path = Some(PathBuf::from("/nonexistent"));
        s.watch_id = Some(WatchId(1));
        let before_local = s.local_generation;
        let before_remote = s.remote_generation;
        let _ = s.update(GitMessage::PeriodicSyncTick);
        // The periodic timer is the recovery path for ref-only/commit-only
        // changes (which produce no file-change event), so it refreshes local
        // (diff stats + branch) and remote (remote state + fetch) even for a
        // watched workspace. The generation bumps prove both the local and
        // remote refresh paths were entered — a regression that ran only remote
        // for a watched workspace would leave `local_generation` untouched.
        assert_eq!(s.local_generation, before_local + 1);
        assert_eq!(s.remote_generation, before_remote + 1);
    }

    #[expect(clippy::unchecked_time_subtraction)] // fixture shifts a monotonic clock backwards by a small fixed offset
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

    /// A watch id is stored only for a successful registration at the current generation.
    #[test]
    fn test_watch_registration() {
        let cases: &[(&str, Option<u64>, Watch, Option<u64>)] = &[
            ("no selection", None, Register, None),
            ("current generation", Some(7), Registered(7, 42), Some(42)),
            ("an error", Some(7), Failed(7), None),
            ("stale generation", Some(9), Registered(8, 99), None),
        ];

        for &(name, generation, msg, stored) in cases {
            let mut s = match generation {
                Some(generation) => state_with_gen(generation),
                None => GitState::new(),
            };
            let _ = match msg {
                Register => s.try_register_watch(),
                Registered(generation, id) => {
                    s.update(GitMessage::WatchRegistered(generation, Ok(WatchId(id))))
                }
                Failed(generation) => {
                    s.update(GitMessage::WatchRegistered(generation, Err(FAILED.into())))
                }
            };
            assert_eq!(s.watch_id, stored.map(WatchId), "case {name}: watch id");
        }
    }

    #[test]
    fn test_commit_refreshes_the_footer() {
        struct Case {
            name: &'static str,
            workspace: &'static str,
            msg: GitMessage,
            refreshed: bool,
        }
        let cases: &[Case] = &[
            Case {
                name: "refresh_after_commit",
                workspace: "/nonexistent",
                msg: GitMessage::RefreshAfterCommit,
                refreshed: true,
            },
            Case {
                name: "pipeline_commit_for_the_displayed_workspace",
                workspace: "/some/ws",
                msg: GitMessage::PipelineCommit(PathBuf::from("/some/ws")),
                refreshed: true,
            },
            Case {
                name: "pipeline_commit_for_another_workspace",
                workspace: "/some/ws",
                msg: GitMessage::PipelineCommit(PathBuf::from("/other/ws")),
                refreshed: false,
            },
        ];

        for case in cases {
            let name = case.name;
            let mut s = GitState::new();
            s.workspace_path = Some(PathBuf::from(case.workspace));
            s.local_refresh_pending = true;
            let before_local = s.local_generation;
            let before_remote = s.remote_generation;
            let _ = s.update(case.msg.clone());
            if case.refreshed {
                assert!(
                    s.last_local_refresh.is_some(),
                    "case {name}: the local refresh was stamped"
                );
                assert!(
                    !s.local_refresh_pending,
                    "case {name}: the owed refresh was claimed"
                );
                assert_eq!(
                    s.local_generation,
                    before_local + 1,
                    "case {name}: the local generation bumped"
                );
                assert_eq!(
                    s.remote_generation,
                    before_remote + 1,
                    "case {name}: the remote generation bumped"
                );
            } else {
                assert_eq!(
                    s.local_generation, before_local,
                    "case {name}: the local generation is kept"
                );
                assert_eq!(
                    s.remote_generation, before_remote,
                    "case {name}: the remote generation is kept"
                );
            }
        }
    }
}
