//! Native Iced dashboard — application entry point, navigation, and shared state.
//!
//! Iced owns the process Tokio runtime (`iced` feature `tokio`). MahBot
//! bootstraps via a startup [`iced::Task`] before the UI becomes interactive.

#![allow(
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::too_many_lines,
    clippy::struct_excessive_bools,
    clippy::match_same_arms,
    clippy::if_not_else,
    clippy::collapsible_if,
    clippy::manual_let_else,
    clippy::manual_div_ceil
)]

pub mod board;
pub mod context_menu;
pub mod diff;
pub mod diff_widget;
pub mod editor;
pub mod editor_widget;
pub mod highlight;
pub mod home;
pub mod logs;
pub mod sessions;
pub mod settings;
pub mod shell;
pub mod text_rendering;
pub mod theme;
pub mod tool_failures;
pub mod users;
pub mod widgets;
pub mod workspaces;

use crate::board::{Ticket, TicketPhase};
use crate::gui::tool_failures::ToolFailuresMessage;
use crate::logs::LogStore;

use iced::keyboard;
use iced::widget::Space;
use iced::widget::{Column, Row, button, column, container, row, scrollable, text, tooltip};
use iced::window;
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use widgets::PickOption;

/// JetBrains Mono as the dashboard default font.
/// Registered at startup via `.default_font()` on the Iced application builder,
/// so all text widgets use JetBrains Mono by default. The font bytes are loaded
/// via `.font()` calls in the application builder.
pub const JETBRAINS_MONO: iced::Font = iced::Font {
    family: iced::font::Family::Name("JetBrains Mono"),
    weight: iced::font::Weight::Normal,
    stretch: iced::font::Stretch::Normal,
    style: iced::font::Style::Normal,
};

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

// ── Global log broadcast for live streaming ──────────────────────

/// Global broadcast sender for live log streaming. Set during `startup()`.
pub static LOG_BROADCAST: OnceLock<broadcast::Sender<String>> = OnceLock::new();

// ── Navigation pages ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Home,
    Sessions,
    Logs,
    Shell,
    Editor,
    Settings,
}

impl Page {
    /// Pages shown in the sidebar (Home, Editor, Shell).
    const fn sidebar_pages() -> &'static [Page] {
        &[Page::Home, Page::Editor, Page::Shell]
    }

    /// Pages shown in the footer nav (Sessions, Logs, Settings).
    const fn footer_pages() -> &'static [Page] {
        &[Page::Sessions, Page::Logs, Page::Settings]
    }

    const fn label(self) -> &'static str {
        match self {
            Page::Home => "Home",
            Page::Sessions => "Sessions",
            Page::Logs => "Logs",
            Page::Shell => "Shell",
            Page::Editor => "Editor",
            Page::Settings => "Settings",
        }
    }
}

// ── Main message type ────────────────────────────────────────────

/// Toast notification kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Success,
    Warning,
    Error,
}

/// A floating toast notification.
#[derive(Debug, Clone)]
pub struct Toast {
    pub id: usize,
    pub message: String,
    pub kind: ToastKind,
    pub created_at: Instant,
}

/// Message emitted by a page to request a toast from the dashboard.
#[derive(Debug, Clone)]
pub enum ToastMessage {
    Saved,
    Created,
    Deleted,
    Error(String),
    Warning(String),
    /// Generic success with custom message.
    SuccessMsg(String),
}

/// Auto-incrementing toast ID counter.
static TOAST_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Toast {
    fn new(message: String, kind: ToastKind) -> Self {
        let id = TOAST_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            id,
            message,
            kind,
            created_at: Instant::now(),
        }
    }

    fn from_toast_msg(msg: &ToastMessage) -> Self {
        match msg {
            ToastMessage::Saved => Toast::new("Saved".to_string(), ToastKind::Success),
            ToastMessage::Created => Toast::new("Created".to_string(), ToastKind::Success),
            ToastMessage::Deleted => Toast::new("Deleted".to_string(), ToastKind::Error),
            ToastMessage::Error(s) => Toast::new(format!("Failed: {s}"), ToastKind::Error),
            ToastMessage::Warning(s) => Toast::new(s.clone(), ToastKind::Warning),
            ToastMessage::SuccessMsg(s) => Toast::new(s.clone(), ToastKind::Success),
        }
    }

    const fn duration(&self) -> Duration {
        match self.kind {
            ToastKind::Success => Duration::from_secs(2),
            ToastKind::Warning | ToastKind::Error => Duration::from_secs(4),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    /// MahBot finished async startup (or failed). On success, [`BOOT_LOG_STORE`] is set.
    Boot(Result<(), String>),
    Navigation(Page),
    Tick,
    /// Shutdown signaled — close the dashboard window so `run()` returns.
    /// Triggered by the shutdown token (self-update restart, SIGTERM/SIGINT).
    Shutdown,
    /// Window close button pressed — persist position and size before exiting.
    CloseRequested(window::Id),
    /// Window geometry event (move/resize) — tracks state for persist-on-close.
    WindowEvent(window::Id, window::Event),
    /// Dismiss a toast by ID.
    DismissToast(usize),
    /// Keyboard shortcut: Cmd+F — focus the primary search input on the current page.
    FocusSearch,
    /// Keyboard shortcut: Escape — dismiss modal/panel/confirmation on the current page.
    EscapePressed,
    /// Update button pressed — trigger self-update.
    UpdateBot,
    /// Self-update result.
    UpdateResult(Result<String, String>),
    /// Toggle the selected workspace's pipeline pause state.
    TogglePause,
    /// Result of a per-workspace pause toggle DB write. Carries (result, workspace_name, intended_state).
    /// On success, workspace state is refreshed from DB; on error an error toast is shown.
    TogglePauseResult(Result<(), String>, String, bool),
    /// Toggle the selected workspace's maintainer toggle.
    ToggleMaintenance,
    /// Result of a per-workspace maintenance toggle DB write.
    ToggleMaintenanceResult(Result<(), String>, String, bool),
    /// Periodic refresh of workspace paused/maintenance state from DB.
    WorkspaceStatesRefreshed(HashMap<String, bool>, HashMap<String, bool>),
    /// No-op — produced by refresh helpers on transient DB errors to avoid
    /// sending empty state maps that would wipe cached toggle state.
    Nop,
    /// Workspace options loaded during boot (options, paths, paused, maintenance, restored selection).
    BootWorkspaces(
        Vec<PickOption>,
        HashMap<String, String>,
        HashMap<String, bool>,
        HashMap<String, bool>,
        Option<String>,
    ),
    Home(home::HomeMessage),
    Logs(logs::LogMessage),
    Board(board::BoardMessage),
    Sessions(sessions::SessionsMessage),
    /// Diff modal overlay (not a page) — wraps [`diff::DiffMessage`].
    /// Named `DiffModal` rather than `Diff` to avoid ambiguity with the
    /// removed `Page::Diff` variant and the existing page-message convention.
    DiffModal(diff::DiffMessage),
    Shell(shell::ShellMessage),
    Editor(editor::EditorMessage),
    Settings(settings::SettingsMessage),

    // ── Diff modal ──────────────────────────────────────────────
    /// Open the diff modal. Optional commit hash — `None` = working tree diff.
    OpenDiffModal(Option<String>),
    /// Close the diff modal.
    CloseDiffModal,

    // ── Git state ────────────────────────────────────────────────
    /// Result of `run_git_diff_stats`. `None` when not a git repo.
    GitDiffStats(Option<(i64, i64)>),
    /// Result of `run_git_current_branch`. `None` when not a git repo.
    GitCurrentBranch(Option<String>),
    /// Result of `run_git_behind_ahead`. `None` when not a git repo / no upstream.
    GitBehindAhead(Option<(usize, usize)>),
    /// Open the branch management modal.
    OpenBranchModal,
    /// Close the branch management modal.
    CloseBranchModal,
    /// Branch search query changed.
    BranchQueryChanged(String),
    /// Result of `run_git_list_branches`.
    GitListBranches(Result<Vec<String>, String>),
    /// Result of `run_git_sync`.
    GitSyncResult(Result<String, String>),
    /// Trigger a git sync (pull --ff-only + push).
    GitSync,
    /// Switch to a branch.
    GitSwitch(String),
    /// Result of `run_git_switch_branch`.
    GitSwitchResult(Result<(), String>),
    /// Create a new branch from the value in `new_branch_name`.
    GitCreate,
    /// Result of `run_git_create_branch`.
    GitCreateBranchResult(Result<(), String>),
    /// The new-branch name input changed.
    NewBranchNameChanged(String),
}

// ── Keyboard modifier helper ─────────────────────────────────────

/// Platform-aware keyboard modifier state computed from a
/// [`keyboard::Modifiers`] value.  Centralises the duplicated
/// `#[cfg]`-gated setup that was repeated across four GUI keyboard
/// subscription handlers.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KeyboardMods {
    /// True if the Command key (macOS) is held (⌘).
    pub is_cmd: bool,
    /// True if the platform modifier is held — Command (⌘) on macOS,
    /// Control (Ctrl) on other platforms.
    pub is_platform_mod: bool,
    /// True if the Control key is held (any platform).
    pub ctrl_held: bool,
    /// On macOS: true if Ctrl is held without Cmd (triggers terminal
    /// control characters / emacs bindings).  Always false on other
    /// platforms.
    pub is_emacs_ctrl: bool,
    /// On non-macOS: true if Ctrl+Alt is held (AltGr character input).
    /// Always false on macOS.
    pub altgr_active: bool,
}

impl KeyboardMods {
    /// Platform modifier for shortcut-like navigation
    /// (arrow-key movement, line start/end, etc.).
    ///
    /// On macOS: Cmd only — Ctrl is reserved for Emacs-style bindings
    /// (Ctrl+F/B/A/E/P/N etc.) and terminal control characters.
    ///
    /// On other platforms: Cmd or Ctrl.
    #[must_use]
    pub fn is_nav_platform_mod(self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.is_cmd
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.is_platform_mod
        }
    }

    /// Platform modifier for text-affecting shortcuts
    /// (clipboard C/X/V, IME guard).
    ///
    /// Stricter than [`is_nav_platform_mod`]: on macOS, Cmd+Ctrl combos
    /// are excluded (Ctrl+C/X/V are terminal control characters); on
    /// other platforms, AltGr (Ctrl+Alt) is excluded because it produces
    /// text characters for international keyboard layouts.
    #[must_use]
    pub fn is_text_platform_mod(self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.is_cmd && !self.ctrl_held
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.is_platform_mod && !self.altgr_active
        }
    }
}

/// Compute [`KeyboardMods`] from an Iced [`keyboard::Modifiers`] value.
///
/// Encapsulates the `#[cfg(target_os = "macos")]` / `#[cfg(not(...))]`
/// blocks that every keyboard subscription handler previously inlined.
pub(crate) fn detect_keyboard_mods(modifiers: keyboard::Modifiers) -> KeyboardMods {
    let is_cmd = modifiers.command();
    let is_platform_mod = modifiers.command() || modifiers.control();
    let ctrl_held = modifiers.control();

    #[cfg(target_os = "macos")]
    let is_emacs_ctrl = modifiers.control() && !modifiers.command();
    #[cfg(not(target_os = "macos"))]
    let is_emacs_ctrl = false;

    #[cfg(not(target_os = "macos"))]
    let altgr_active = modifiers.alt() && modifiers.control();
    #[cfg(target_os = "macos")]
    let altgr_active = false;

    KeyboardMods {
        is_cmd,
        is_platform_mod,
        ctrl_held,
        is_emacs_ctrl,
        altgr_active,
    }
}

// ── Dashboard state ──────────────────────────────────────────────

/// Log store created during boot; read when handling [`Message::Boot`].
pub static BOOT_LOG_STORE: OnceLock<LogStore> = OnceLock::new();

pub struct Dashboard {
    ready: bool,
    boot_error: Option<String>,
    page: Page,
    log_store: Option<LogStore>,

    /// Tracked window geometry for persist-on-close.
    last_size: iced::Size,
    last_position: iced::Point,
    /// Toast notification stack.
    toasts: Vec<Toast>,

    /// Global workspace picker state.
    workspace_options: Vec<PickOption>,
    /// Maps workspace name → filesystem path.
    workspace_paths: HashMap<String, String>,
    /// Maps workspace name → paused state (for sidebar toggle).
    workspace_paused: HashMap<String, bool>,
    /// Maps workspace name → maintenance state (for sidebar toggle).
    workspace_maintenance: HashMap<String, bool>,
    /// Currently selected workspace name from the global picker.
    selected_workspace_name: Option<String>,
    /// Currently selected user name (for impersonation). Persisted in window state.
    selected_user_name: Option<String>,
    /// Whether a self-update is in progress (button disabled while building).
    updating: bool,
    /// Whether self-update is available on this installation (controls button visibility).
    update_available: bool,
    /// Whether the selected workspace's pipeline is paused (no new tickets claimed).
    paused: bool,
    /// Whether the selected workspace's maintainer is enabled.
    maintenance: bool,

    logs_state: logs::LogsState,
    board_state: board::BoardState,
    sessions_state: sessions::SessionsState,
    diff_state: diff::DiffState,
    home_state: home::HomeState,
    shell_state: shell::ShellState,
    editor_state: editor::EditorState,
    settings_state: settings::SettingsState,

    // ── Diff modal ──────────────────────────────────────────────
    show_diff_modal: bool,

    // ── Git state ───────────────────────────────────────────────
    /// Cached filesystem path for the currently selected workspace.
    workspace_filesystem_path: Option<String>,
    /// Cached diff stats (+N / -M) from periodic refresh.
    git_diff_stats: Option<(i64, i64)>,
    /// Cached current branch name from periodic refresh.
    git_current_branch: Option<String>,
    /// Cached behind/ahead counts from periodic refresh.
    git_behind_ahead: Option<(usize, usize)>,
    /// Whether the branch management modal is open.
    show_branch_modal: bool,
    /// Branch search query text.
    branch_search_query: String,
    /// Cached list of local branches.
    local_branches: Vec<String>,
    /// Whether a git sync operation is in-flight.
    git_syncing: bool,
    /// Error message from branch switch/create failure.
    git_branch_error: Option<String>,
    /// Current value of the "new branch name" text input.
    new_branch_name: String,
    /// Whether git state was eagerly refreshed recently — skip the next
    /// Tick-based refresh to avoid double-firing after workspace switch.
    git_refresh_eagerly: bool,
}

impl Dashboard {
    pub fn loading(update_available: bool) -> Self {
        Self {
            ready: false,
            boot_error: None,
            page: Page::Home,
            log_store: None,
            last_size: iced::Size::new(1500.0, 800.0),
            last_position: iced::Point::new(-1.0, -1.0),
            toasts: Vec::new(),
            workspace_options: Vec::new(),
            workspace_paths: HashMap::new(),
            workspace_paused: HashMap::new(),
            workspace_maintenance: HashMap::new(),
            selected_workspace_name: None,
            selected_user_name: None,
            updating: false,
            update_available,
            paused: false,
            maintenance: false,
            logs_state: logs::LogsState::new(),
            board_state: board::BoardState::new(),
            sessions_state: sessions::SessionsState::new(),
            diff_state: diff::DiffState::new(),
            home_state: home::HomeState::new(),
            shell_state: shell::ShellState::new(),
            editor_state: editor::EditorState::new(),
            settings_state: settings::SettingsState::new(),
            show_diff_modal: false,
            workspace_filesystem_path: None,
            git_diff_stats: None,
            git_current_branch: None,
            git_behind_ahead: None,
            show_branch_modal: false,
            branch_search_query: String::new(),
            local_branches: Vec::new(),
            git_syncing: false,
            git_branch_error: None,
            new_branch_name: String::new(),
            git_refresh_eagerly: false,
        }
    }

    fn finish_boot(&mut self, result: Result<(), String>) -> Task<Message> {
        match result {
            Ok(()) => {
                let log_store = BOOT_LOG_STORE
                    .get()
                    .cloned()
                    .expect("BOOT_LOG_STORE set before Boot(Ok)");
                self.ready = true;
                self.boot_error = None;
                let refresh_logs = self.logs_state.refresh(&log_store);
                let refresh_board = self.board_state.refresh();
                self.log_store = Some(log_store);
                let prev = read_window_state();
                self.selected_user_name = prev.selected_user;
                let boot_workspaces = Task::perform(
                    load_workspace_options(prev.selected_workspace),
                    std::convert::identity,
                );
                Task::batch([
                    refresh_logs.map(Message::Logs),
                    refresh_board.map(Message::Board),
                    boot_workspaces,
                ])
            }
            Err(e) => {
                self.boot_error = Some(e);
                Task::none()
            }
        }
    }

    pub const fn theme(&self) -> iced::Theme {
        iced::Theme::Dark
    }

    fn save_and_exit(&self) -> Task<Message> {
        save_window_state(
            self.last_position,
            self.last_size,
            self.selected_workspace_name.as_deref(),
            self.selected_user_name.as_deref(),
        );
        iced::exit()
    }

    /// Window title with page name.
    pub fn title(&self) -> String {
        let page_name = self.page.label();
        format!("MahBot — {page_name}")
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Boot(result) => self.finish_boot(result),
            Message::BootWorkspaces(options, paths, paused_map, maintenance_map, restored_name) => {
                self.workspace_options.clone_from(&options);
                self.workspace_paths = paths;
                self.workspace_paused = paused_map;
                self.workspace_maintenance = maintenance_map;
                // Derive paused & maintenance states from the selected workspace.
                // Reads from dash.workspace_paused / dash.workspace_maintenance
                // while writing dash.paused / dash.maintenance (disjoint fields).
                let update_states = |dash: &mut Self, ws_name: Option<&str>| {
                    dash.paused = ws_name
                        .and_then(|n| dash.workspace_paused.get(n))
                        .copied()
                        .unwrap_or(false);
                    dash.maintenance = ws_name
                        .and_then(|n| dash.workspace_maintenance.get(n))
                        .copied()
                        .unwrap_or(false);
                };
                // Pre-set Home's selected_user from persisted window state
                // so UsersLoaded doesn't auto-select the first user when
                // a previous user was saved.
                if let Some(ref user_name) = self.selected_user_name {
                    self.home_state.selected_user = Some(user_name.clone());
                }
                // Forward workspace options to the Home page.
                let home_opts: Task<Message> =
                    Task::done(home::HomeMessage::WorkspaceOptions(options.clone()))
                        .map(Message::Home);
                let load_users = self.home_state.load_users().map(Message::Home);

                // restored_name is always Some — load_workspace_options sets it.
                // Empty string => "Personal" workspace (no shared workspace).
                let ws_name = match restored_name {
                    Some(ref name) if name.is_empty() => {
                        self.selected_workspace_name = None;
                        update_states(self, None);
                        String::new()
                    }
                    Some(ref name) => {
                        self.selected_workspace_name = Some(name.clone());
                        update_states(self, Some(name));
                        name.clone()
                    }
                    None => {
                        // Unreachable: load_workspace_options always produces Some.
                        // Defensive fallback — treat as Personal workspace.
                        self.selected_workspace_name = None;
                        update_states(self, None);
                        String::new()
                    }
                };
                Task::batch([
                    self.propagate_workspace_selection(&ws_name),
                    home_opts,
                    load_users,
                ])
            }
            Message::Navigation(_) if !self.ready => Task::none(),
            Message::Navigation(page) => {
                self.page = page;
                // Notify sessions state when navigating to/from Sessions page
                // so the auto-refresh timer starts/stops accordingly.
                self.sessions_state.set_page_active(page == Page::Sessions);
                match page {
                    Page::Logs => Task::none(),
                    Page::Home => {
                        let load_users = self.home_state.load_users().map(Message::Home);
                        let ws_opts = Task::done(home::HomeMessage::WorkspaceOptions(
                            self.workspace_options.clone(),
                        ))
                        .map(Message::Home);
                        let snap =
                            iced::widget::operation::snap_to_end::<Message>(home::CHAT_SCROLL_ID);
                        let board_refresh = self.board_state.refresh().map(Message::Board);
                        Task::batch([load_users, ws_opts, snap, board_refresh])
                    }
                    Page::Shell => Task::none(),
                    Page::Sessions => self.sessions_state.refresh().map(Message::Sessions),
                    // Editor receives workspace state via WorkspaceSelected
                    // from the Home page picker, not via refresh().
                    Page::Editor => Task::none(),
                    Page::Settings => {
                        self.settings_state.refresh();
                        let refresh_workspaces =
                            self.settings_state.workspaces_state.refresh().map(|msg| {
                                Message::Settings(settings::SettingsMessage::WorkspaceMsg(msg))
                            });
                        let refresh_users =
                            self.settings_state.users_state.refresh().map(|msg| {
                                Message::Settings(settings::SettingsMessage::UserMsg(msg))
                            });
                        Task::batch([refresh_workspaces, refresh_users])
                    }
                }
            }
            Message::Tick => {
                // Auto-dismiss expired toasts
                let now = Instant::now();
                self.toasts
                    .retain(|t| now.duration_since(t.created_at) < t.duration());
                // Auto-poll visible page at 1-second intervals (with loading guard)
                if !self.ready {
                    return Task::none();
                }

                // Auto-refresh workspace paused/maintenance state every tick.
                // Only runs when a workspace is selected — the toggle result
                // handler already re-reads authoritative state after writes.
                let ws_refresh = if self.has_active_workspace() {
                    refresh_workspace_states_task()
                } else {
                    Task::none()
                };

                let page_task = match self.page {
                    Page::Home if !self.board_state.loading => {
                        self.board_state.loading = true;
                        self.board_state.refresh().map(Message::Board)
                    }
                    Page::Shell => Task::none(),
                    Page::Sessions if !self.sessions_state.loading => {
                        self.sessions_state.loading = true;
                        self.sessions_state.refresh().map(Message::Sessions)
                    }
                    Page::Settings => {
                        // Refresh workspace and user lists when on Settings page
                        let ws_loading = self.settings_state.workspaces_state.loading;
                        let us_loading = self.settings_state.users_state.loading;
                        let ws = if !ws_loading {
                            self.settings_state.workspaces_state.loading = true;
                            self.settings_state.workspaces_state.refresh().map(|msg| {
                                Message::Settings(settings::SettingsMessage::WorkspaceMsg(msg))
                            })
                        } else {
                            Task::none()
                        };
                        let us = if !us_loading {
                            self.settings_state.users_state.loading = true;
                            self.settings_state.users_state.refresh().map(|msg| {
                                Message::Settings(settings::SettingsMessage::UserMsg(msg))
                            })
                        } else {
                            Task::none()
                        };
                        Task::batch([ws, us])
                    }
                    _ => Task::none(),
                };

                // ── Git state refresh (every second) ────────────────
                // Skip if an eager refresh was just triggered (e.g. after
                // workspace switch) to avoid 6 subprocess calls in <1 second.
                let git_tasks = if self.git_refresh_eagerly {
                    self.git_refresh_eagerly = false;
                    Task::none()
                } else {
                    self.refresh_git_state()
                };

                Task::batch([ws_refresh, page_task, git_tasks])
            }
            Message::Home(msg) if self.ready => {
                // Intercept RequestWorkspaceChange: the Home page detected
                // that the selected user's DB workspace differs from the
                // sidebar selection.  Perform a Dashboard-level workspace
                // switch so the sidebar, saved state, and all pages stay
                // consistent.
                if let home::HomeMessage::RequestWorkspaceChange(ref name) = msg {
                    return self.select_workspace(name);
                }
                // Intercept WorkspacePicked: the Home page's own workspace
                // picker changed — update the sidebar and all pages to match.
                if let home::HomeMessage::WorkspacePicked(ref name) = msg {
                    return self.select_workspace(name);
                }
                // Intercept UserSelected: user changed (from picker, Users page
                // icon, or auto-selected at boot) — sync the Dashboard's
                // selected_user_name and persist to window state.
                if let home::HomeMessage::UserSelected(ref user) = msg {
                    self.selected_user_name = Some(user.clone());
                    save_window_state(
                        self.last_position,
                        self.last_size,
                        self.selected_workspace_name.as_deref(),
                        self.selected_user_name.as_deref(),
                    );
                }
                // Intercept Toast: push to dashboard toast stack.
                if let home::HomeMessage::Toast(ref tm) = msg {
                    self.toasts.push(Toast::from_toast_msg(tm));
                }
                self.home_state.update(msg).map(Message::Home)
            }
            Message::Shell(msg) if self.ready => self.shell_state.update(msg).map(Message::Shell),
            Message::Logs(msg) if self.ready => {
                // Intercept Toast messages — both direct LogMessage::Toast
                // and nested ToolFailuresMessage::Toast from the TF tab.
                if let logs::LogMessage::Toast(ref tm) = msg {
                    self.toasts.push(Toast::from_toast_msg(tm));
                }
                if let logs::LogMessage::ToolFailures(ToolFailuresMessage::Toast(ref tm)) = msg {
                    self.toasts.push(Toast::from_toast_msg(tm));
                }
                self.logs_state
                    .update(msg, self.log_store.as_ref().expect("ready"))
                    .map(Message::Logs)
            }
            Message::Board(msg) if self.ready => {
                // Intercept ViewCommitDiff for cross-page navigation
                // before it reaches board_state.update.
                if let board::BoardMessage::ViewCommitDiff {
                    ref commit_hash,
                    ref workspace_name,
                } = msg
                {
                    // Close any board modal
                    let close_board = self
                        .board_state
                        .update(board::BoardMessage::CloseModal)
                        .map(Message::Board);
                    // Open diff modal — the diff_state will receive
                    // NavigateToCommit which loads the commit data.
                    self.show_diff_modal = true;
                    self.show_branch_modal = false;
                    let hash = commit_hash.clone();
                    let ws = workspace_name.clone();
                    return Task::batch([
                        close_board,
                        Task::done(Message::DiffModal(diff::DiffMessage::NavigateToCommit(
                            ws, hash,
                        ))),
                    ]);
                }
                if let board::BoardMessage::LinkClicked(ref url) = msg {
                    open_url(url);
                }
                if let board::BoardMessage::Toast(ref tm) = msg {
                    self.toasts.push(Toast::from_toast_msg(tm));
                }
                self.board_state.update(msg).map(Message::Board)
            }
            Message::Sessions(msg) if self.ready => {
                if let sessions::SessionsMessage::LinkClicked(ref url) = msg {
                    open_url(url);
                }
                if let sessions::SessionsMessage::Toast(ref tm) = msg {
                    self.toasts.push(Toast::from_toast_msg(tm));
                }
                self.sessions_state.update(msg).map(Message::Sessions)
            }
            Message::DiffModal(msg) if self.ready => {
                if let diff::DiffMessage::Toast(ref tm) = msg {
                    self.toasts.push(Toast::from_toast_msg(tm));
                }
                self.diff_state.update(msg).map(Message::DiffModal)
            }
            Message::Editor(msg) if self.ready => {
                if let editor::EditorMessage::Toast(ref tm) = msg {
                    self.toasts.push(Toast::from_toast_msg(tm));
                }
                self.editor_state.update(msg).map(Message::Editor)
            }
            Message::Settings(msg) if self.ready => {
                // Intercept workspace link clicks from the context-view modal
                if let settings::SettingsMessage::WorkspaceMsg(
                    ref wm @ workspaces::WorkspacesMessage::LinkClicked(ref url),
                ) = msg
                {
                    open_url(url);
                    return self
                        .settings_state
                        .update(settings::SettingsMessage::WorkspaceMsg(wm.clone()))
                        .map(Message::Settings);
                }
                // Intercept toast messages from workspace and user state.
                let toast = if let settings::SettingsMessage::WorkspaceMsg(
                    workspaces::WorkspacesMessage::Toast(ref tm),
                ) = msg
                {
                    Some(tm.clone())
                } else if let settings::SettingsMessage::UserMsg(users::UsersMessage::Toast(
                    ref tm,
                )) = msg
                {
                    Some(tm.clone())
                } else {
                    None
                };
                if let Some(tm) = toast {
                    self.toasts.push(Toast::from_toast_msg(&tm));
                }
                // Intercept SwitchUser from user messages.
                if let settings::SettingsMessage::UserMsg(users::UsersMessage::SwitchUser(
                    ref user,
                )) = msg
                {
                    let switch = Task::done(home::HomeMessage::UserSelected(user.clone()))
                        .map(Message::Home);
                    return switch;
                }
                // Intercept DeleteResult: if the deleted user was the active
                // user, fall back to admin.
                if let settings::SettingsMessage::UserMsg(
                    ref msg_inner @ users::UsersMessage::DeleteResult(Ok(()), ref deleted_user),
                ) = msg
                {
                    if self.selected_user_name.as_deref() == Some(deleted_user.as_str()) {
                        self.selected_user_name = Some("admin".to_string());
                        save_window_state(
                            self.last_position,
                            self.last_size,
                            self.selected_workspace_name.as_deref(),
                            self.selected_user_name.as_deref(),
                        );
                        let switch =
                            Task::done(home::HomeMessage::UserSelected("admin".to_string()))
                                .map(Message::Home);
                        let settings_task = self
                            .settings_state
                            .update(settings::SettingsMessage::UserMsg(msg_inner.clone()))
                            .map(Message::Settings);
                        return Task::batch([switch, settings_task]);
                    }
                }
                // Intercept UpdateWorkspace: propagate to Dashboard if it's
                // the active user.
                if let settings::SettingsMessage::UserMsg(
                    ref msg_inner @ users::UsersMessage::UpdateWorkspace(ref sender, ref ws),
                ) = msg
                {
                    if self.selected_user_name.as_deref() == Some(sender.as_str()) {
                        let select_task = self.select_workspace(ws);
                        let settings_task = self
                            .settings_state
                            .update(settings::SettingsMessage::UserMsg(msg_inner.clone()))
                            .map(Message::Settings);
                        return Task::batch([select_task, settings_task]);
                    }
                }
                // Check whether workspace list changed (add/delete) and
                // needs_global_reload is computed right before consumption,
                // after all early-return intercepts above.
                let needs_global_reload = matches!(
                    msg,
                    settings::SettingsMessage::WorkspaceMsg(
                        workspaces::WorkspacesMessage::DeleteResult(Ok(()))
                    ) | settings::SettingsMessage::AddWorkspaceResult(Ok(_))
                );
                let task = self.settings_state.update(msg).map(Message::Settings);
                if needs_global_reload {
                    let reload_task = self.reload_workspace_options();
                    Task::batch([task, reload_task])
                } else {
                    task
                }
            }
            Message::Shutdown => self.save_and_exit(),
            Message::CloseRequested(_id) => self.save_and_exit(),
            Message::WindowEvent(_id, event) => {
                match event {
                    window::Event::Resized(new_size) => self.last_size = new_size,
                    window::Event::Moved(new_pos) => self.last_position = new_pos,
                    _ => {}
                }
                Task::none()
            }
            Message::DismissToast(id) => {
                self.toasts.retain(|t| t.id != id);
                Task::none()
            }
            // ── Diff modal ────────────────────────────────────────
            Message::OpenDiffModal(commit_hash) if self.ready => {
                // Close any board modal and branch modal
                let close_board = self
                    .board_state
                    .update(board::BoardMessage::CloseModal)
                    .map(Message::Board);
                self.show_diff_modal = true;
                self.show_branch_modal = false;
                if let Some(hash) = commit_hash {
                    let ws = self.selected_workspace_name.clone().unwrap_or_default();
                    let hash_clone = hash;
                    Task::batch([
                        close_board,
                        Task::done(Message::DiffModal(diff::DiffMessage::NavigateToCommit(
                            ws, hash_clone,
                        ))),
                    ])
                } else {
                    // Working tree — send BackToWorkingTree to clear any
                    // stale commit ref and load the working-tree diff.
                    Task::batch([
                        close_board,
                        Task::done(Message::DiffModal(diff::DiffMessage::BackToWorkingTree)),
                    ])
                }
            }
            Message::CloseDiffModal => {
                self.show_diff_modal = false;
                Task::done(Message::DiffModal(diff::DiffMessage::ClearCommitState))
            }
            // ── Git state ─────────────────────────────────────────
            Message::GitDiffStats(stats) => {
                self.git_diff_stats = stats;
                Task::none()
            }
            Message::GitCurrentBranch(branch) => {
                self.git_current_branch = branch;
                Task::none()
            }
            Message::GitBehindAhead(ba) => {
                self.git_behind_ahead = ba;
                Task::none()
            }
            Message::OpenBranchModal if self.ready => {
                self.show_branch_modal = true;
                self.show_diff_modal = false;
                self.branch_search_query.clear();
                self.git_branch_error = None;
                let ws_path = self.workspace_filesystem_path.clone();
                Task::perform(
                    async move {
                        match ws_path {
                            Some(path) => {
                                let path = std::path::PathBuf::from(path);
                                crate::diff_parse::run_git_list_branches(&path).await
                            }
                            None => Ok(Vec::new()),
                        }
                    },
                    Message::GitListBranches,
                )
            }
            Message::CloseBranchModal => {
                self.show_branch_modal = false;
                Task::none()
            }
            Message::BranchQueryChanged(query) => {
                self.branch_search_query = query;
                Task::none()
            }
            Message::GitListBranches(result) => {
                match result {
                    Ok(branches) => self.local_branches = branches,
                    Err(e) => self.git_branch_error = Some(e),
                }
                Task::none()
            }
            Message::GitSync if self.ready => {
                self.git_syncing = true;
                let ws_path = self.workspace_filesystem_path.clone();
                Task::perform(
                    async move {
                        match ws_path {
                            Some(path) => {
                                let path = std::path::PathBuf::from(path);
                                crate::diff_parse::run_git_sync(&path).await
                            }
                            None => Err("No workspace path".to_string()),
                        }
                    },
                    Message::GitSyncResult,
                )
            }
            Message::GitSyncResult(result) => {
                self.git_syncing = false;
                match result {
                    Ok(output) => {
                        self.toasts.push(Toast::new(
                            if output.trim().is_empty() {
                                "Already up-to-date".to_string()
                            } else {
                                format!("Sync completed:\n{output}")
                            },
                            ToastKind::Success,
                        ));
                    }
                    Err(e) => {
                        self.toasts
                            .push(Toast::new(format!("Sync failed: {e}"), ToastKind::Error));
                    }
                }
                Task::none()
            }
            Message::GitSwitch(branch) if !self.git_syncing && self.ready => {
                let ws_path = self.workspace_filesystem_path.clone();
                let branch_clone = branch.clone();
                self.git_syncing = true;
                let task = async move {
                    match ws_path {
                        Some(path) => {
                            let path = std::path::PathBuf::from(path);
                            crate::diff_parse::run_git_switch_branch(&path, &branch_clone).await
                        }
                        None => Err("No workspace path".to_string()),
                    }
                };
                Task::perform(task, Message::GitSwitchResult)
            }
            Message::GitSwitchResult(result) => {
                self.git_syncing = false;
                match result {
                    Ok(()) => {
                        self.toasts.push(Toast::new(
                            "Switched branch".to_string(),
                            ToastKind::Success,
                        ));
                        self.show_branch_modal = false;
                    }
                    Err(e) => {
                        self.git_branch_error = Some(e);
                    }
                }
                Task::none()
            }
            Message::GitCreate if !self.git_syncing && self.ready => {
                let branch = self.new_branch_name.clone();
                if branch.trim().is_empty() {
                    self.git_branch_error = Some("Branch name cannot be empty".to_string());
                    return Task::none();
                }
                let ws_path = self.workspace_filesystem_path.clone();
                let branch_clone = branch.trim().to_string();
                self.git_syncing = true;
                let task = async move {
                    match ws_path {
                        Some(path) => {
                            let path = std::path::PathBuf::from(path);
                            crate::diff_parse::run_git_create_branch(&path, &branch_clone).await
                        }
                        None => Err("No workspace path".to_string()),
                    }
                };
                Task::perform(task, Message::GitCreateBranchResult)
            }
            Message::NewBranchNameChanged(name) => {
                self.new_branch_name = name;
                Task::none()
            }
            Message::GitCreateBranchResult(result) => {
                self.git_syncing = false;
                match result {
                    Ok(()) => {
                        self.toasts.push(Toast::new(
                            "Created and switched to new branch".to_string(),
                            ToastKind::Success,
                        ));
                        self.show_branch_modal = false;
                    }
                    Err(e) => {
                        self.git_branch_error = Some(e);
                    }
                }
                Task::none()
            }
            Message::FocusSearch => match self.page {
                Page::Logs => self
                    .logs_state
                    .update(
                        logs::LogMessage::FocusSearch,
                        self.log_store.as_ref().expect("ready"),
                    )
                    .map(Message::Logs),
                _ => Task::none(),
            },
            Message::EscapePressed => {
                // Modal close priority: diff modal first, then branch modal,
                // then page-level escapes.
                if self.show_diff_modal {
                    self.show_diff_modal = false;
                    Task::done(Message::DiffModal(diff::DiffMessage::ClearCommitState))
                } else if self.show_branch_modal {
                    self.show_branch_modal = false;
                    Task::none()
                } else {
                    match self.page {
                        Page::Home => {
                            if self.board_state.is_modal_open() {
                                self.board_state
                                    .update(board::BoardMessage::Escape)
                                    .map(Message::Board)
                            } else {
                                Task::none()
                            }
                        }
                        Page::Shell => Task::none(),
                        Page::Logs => self
                            .logs_state
                            .update(
                                logs::LogMessage::Escape,
                                self.log_store.as_ref().expect("ready"),
                            )
                            .map(Message::Logs),
                        Page::Sessions => self
                            .sessions_state
                            .update(sessions::SessionsMessage::Escape)
                            .map(Message::Sessions),
                        Page::Editor => self
                            .editor_state
                            .update(editor::EditorMessage::Escape)
                            .map(Message::Editor),
                        Page::Settings => {
                            if self.settings_state.is_modal_open() {
                                self.settings_state
                                    .update(settings::SettingsMessage::Escape)
                                    .map(Message::Settings)
                            } else {
                                Task::none()
                            }
                        }
                    }
                }
            }
            Message::UpdateBot if self.ready => {
                self.updating = true;
                // Save window state before update (synchronous).
                save_window_state(
                    self.last_position,
                    self.last_size,
                    self.selected_workspace_name.as_deref(),
                    self.selected_user_name.as_deref(),
                );
                Task::perform(
                    async {
                        crate::self_update::execute_update()
                            .await
                            .map_err(|e| format!("{e:#}"))
                            .map(|()| "ok".to_string())
                    },
                    Message::UpdateResult,
                )
            }
            Message::UpdateResult(result) if self.ready => match result {
                Ok(_msg) => {
                    // Success — execute_update() called exit(0), so we
                    // never actually reach this branch. The window closes
                    // as the only success signal to the user.
                    self.updating = false;
                    Task::none()
                }
                Err(err) => {
                    self.updating = false;
                    self.toasts
                        .push(Toast::from_toast_msg(&ToastMessage::Error(err)));
                    Task::none()
                }
            },
            Message::TogglePause if self.ready => {
                let ws_name = if let Some(n) = self.active_workspace_name() {
                    n
                } else {
                    self.toasts.push(Toast::new(
                        "No workspace selected — select a workspace first".to_string(),
                        ToastKind::Warning,
                    ));
                    return Task::none();
                };
                let new_paused = !self.paused;
                // Persist to DB; refresh state from DB on completion.
                let ws_name_clone = ws_name.clone();
                Task::perform(
                    async move {
                        let store = crate::workspace::store();
                        store
                            .set_paused(&ws_name_clone, new_paused)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |result| Message::TogglePauseResult(result, ws_name, new_paused),
                )
            }
            Message::TogglePauseResult(result, ws_name, intended_state) if self.ready => {
                match result {
                    Ok(()) => {
                        self.toasts.push(Toast::new(
                            if intended_state {
                                format!("Pipeline paused for {ws_name}")
                            } else {
                                format!("Pipeline resumed for {ws_name}")
                            },
                            ToastKind::Success,
                        ));
                        refresh_workspace_states_task()
                    }
                    Err(e) => {
                        self.toasts.push(Toast::new(
                            format!("Failed to toggle pipeline pause: {e}"),
                            ToastKind::Error,
                        ));
                        Task::none()
                    }
                }
            }
            Message::ToggleMaintenance if self.ready => {
                let ws_name = if let Some(n) = self.active_workspace_name() {
                    n
                } else {
                    self.toasts.push(Toast::new(
                        "No workspace selected — select a workspace first".to_string(),
                        ToastKind::Warning,
                    ));
                    return Task::none();
                };
                let new_enabled = !self.maintenance;
                // Persist to DB; refresh state from DB on completion.
                let ws_name_clone = ws_name.clone();
                Task::perform(
                    async move {
                        let store = crate::workspace::store();
                        store
                            .set_maintenance(&ws_name_clone, new_enabled)
                            .await
                            .map_err(|e| e.to_string())
                    },
                    move |result| Message::ToggleMaintenanceResult(result, ws_name, new_enabled),
                )
            }
            Message::ToggleMaintenanceResult(result, ws_name, intended_state) if self.ready => {
                match result {
                    Ok(()) => {
                        self.toasts.push(Toast::new(
                            if intended_state {
                                format!("Maintainer enabled for {ws_name}")
                            } else {
                                format!("Maintainer disabled for {ws_name}")
                            },
                            ToastKind::Success,
                        ));
                        refresh_workspace_states_task()
                    }
                    Err(e) => {
                        self.toasts.push(Toast::new(
                            format!("Failed to toggle maintainer: {e}"),
                            ToastKind::Error,
                        ));
                        Task::none()
                    }
                }
            }
            Message::WorkspaceStatesRefreshed(paused_map, maintenance_map) if self.ready => {
                self.workspace_paused = paused_map;
                self.workspace_maintenance = maintenance_map;
                // Update active state for the currently selected workspace.
                if let Some(ref name) = self.selected_workspace_name {
                    self.paused = self.workspace_paused.get(name).copied().unwrap_or(false);
                    self.maintenance = self
                        .workspace_maintenance
                        .get(name)
                        .copied()
                        .unwrap_or(false);
                } else {
                    self.paused = false;
                    self.maintenance = false;
                }
                Task::none()
            }
            Message::Home(_)
            | Message::Shell(_)
            | Message::Logs(_)
            | Message::Board(_)
            | Message::Sessions(_)
            | Message::DiffModal(_)
            | Message::Editor(_)
            | Message::Settings(_)
            | Message::UpdateBot
            | Message::UpdateResult(_)
            | Message::TogglePause
            | Message::TogglePauseResult(..)
            | Message::ToggleMaintenance
            | Message::ToggleMaintenanceResult(..)
            | Message::WorkspaceStatesRefreshed(..)
            | Message::Nop
            | Message::OpenDiffModal(_)
            | Message::GitSwitch(_)
            | Message::GitCreate
            | Message::GitSync
            | Message::OpenBranchModal => Task::none(),
        }
    }

    /// Persist the workspace selection (sidebar state, window-state.json,
    /// and all page broadcasts). This is the canonical entry point for
    /// workspace switching throughout the dashboard.
    ///
    /// An empty name selects the "Personal" workspace (no shared workspace).
    fn select_workspace(&mut self, name: &str) -> Task<Message> {
        // Clear git state — eagerly refreshed below; Tick skips once.
        self.git_diff_stats = None;
        self.git_current_branch = None;
        self.git_behind_ahead = None;
        self.local_branches.clear();
        self.branch_search_query.clear();
        self.git_branch_error = None;
        self.git_refresh_eagerly = true;

        if name.is_empty() {
            self.selected_workspace_name = None;
            self.paused = false;
            self.maintenance = false;
            save_window_state(
                self.last_position,
                self.last_size,
                None,
                self.selected_user_name.as_deref(),
            );
            self.propagate_workspace_selection("")
        } else {
            self.selected_workspace_name = Some(name.to_string());
            self.paused = self.workspace_paused.get(name).copied().unwrap_or(false);
            self.maintenance = self
                .workspace_maintenance
                .get(name)
                .copied()
                .unwrap_or(false);
            save_window_state(
                self.last_position,
                self.last_size,
                Some(name),
                self.selected_user_name.as_deref(),
            );
            self.propagate_workspace_selection(name)
        }
    }

    /// Propagate the global workspace selection to all affected pages.
    /// Sets workspace state on each page and triggers refreshes via their
    /// existing `WorkspaceSelected` handlers.
    fn propagate_workspace_selection(&mut self, name: &str) -> Task<Message> {
        let ws_path = self.workspace_paths.get(name).cloned();

        // Set board's workspace filter directly, then refresh
        self.board_state.workspace_name = Some(name.to_string());
        let board_refresh = self.board_state.refresh().map(Message::Board);

        // Resolve the personal workspace path when name is empty (Personal)
        // and a user is selected.  Editor, Shell, and Diff need a real
        // filesystem path to work with.
        let personal_path = if name.is_empty() {
            self.selected_user_name.as_ref().map(|u| {
                crate::users::personal_workspace_path(u)
                    .to_string_lossy()
                    .to_string()
            })
        } else {
            None
        };

        // If the path is missing from the map, send an empty selection so
        // downstream pages clear their state (the workspace picker
        // guards against this in normal operation, but guard for db
        // inconsistency).  Personal workspaces get their resolved path.
        let (editor_name, editor_path) =
            resolve_workspace_path(name, ws_path.as_ref(), personal_path.as_ref());
        let editor_task: Task<Message> = Task::done(editor::EditorMessage::WorkspaceSelected(
            editor_name,
            editor_path,
        ))
        .map(Message::Editor);

        let diff_name = name.to_string();
        let diff_path = personal_path.clone();

        // Cache workspace filesystem path on Dashboard for git state + modal use.
        let resolved_path = ws_path.clone().or_else(|| personal_path.clone());
        self.workspace_filesystem_path = resolved_path;

        let diff_task: Task<Message> =
            Task::done(diff::DiffMessage::WorkspaceSelected(diff_name, diff_path))
                .map(Message::DiffModal);

        let (shell_name, shell_path) =
            resolve_workspace_path(name, ws_path.as_ref(), personal_path.as_ref());
        let shell_task: Task<Message> = Task::done(shell::ShellMessage::WorkspaceSelected(
            shell_name, shell_path,
        ))
        .map(Message::Shell);

        // Notify the Home page so it can reload chat history.
        let home_name = name.to_string();
        let home_task: Task<Message> =
            Task::done(home::HomeMessage::WorkspaceChanged(Some(home_name))).map(Message::Home);

        Task::batch([
            board_refresh,
            editor_task,
            diff_task,
            shell_task,
            home_task,
            self.refresh_git_state(),
        ])
    }

    /// Reload workspace options from storage (e.g. after add/delete on the
    /// Workspaces page). Preserves current selection if it still exists;
    /// otherwise falls back to the first available workspace.
    fn reload_workspace_options(&self) -> Task<Message> {
        let prev_selection = self.selected_workspace_name.clone();
        Task::perform(
            load_workspace_options(prev_selection),
            std::convert::identity,
        )
    }

    /// Refresh git state information (diff stats, current branch, behind/ahead).
    /// Called every tick when a workspace with a git repo is selected.
    fn refresh_git_state(&self) -> Task<Message> {
        let ws_path = match &self.workspace_filesystem_path {
            Some(p) => p.clone(),
            None => return Task::none(),
        };

        let ws_path = std::path::PathBuf::from(ws_path);
        if !crate::diff_parse::is_git_repo(&ws_path) {
            return Task::none();
        }

        // Diff stats
        let stats_path = ws_path.clone();
        let stats_task = Task::perform(
            async move {
                match crate::diff_parse::run_git_diff_stats(&stats_path).await {
                    Ok(stats) => Message::GitDiffStats(Some(stats)),
                    Err(_) => Message::GitDiffStats(None),
                }
            },
            std::convert::identity,
        );

        // Current branch
        let branch_path = ws_path.clone();
        let branch_task = Task::perform(
            async move {
                match crate::diff_parse::run_git_current_branch(&branch_path).await {
                    Ok(b) => Message::GitCurrentBranch(Some(b)),
                    Err(_) => Message::GitCurrentBranch(None),
                }
            },
            std::convert::identity,
        );

        // Behind/ahead
        let ahead_path = ws_path;
        let ahead_task = Task::perform(
            async move {
                match crate::diff_parse::run_git_behind_ahead(&ahead_path).await {
                    Ok(ba) if ba.0 > 0 || ba.1 > 0 => Message::GitBehindAhead(Some(ba)),
                    _ => Message::GitBehindAhead(None),
                }
            },
            std::convert::identity,
        );

        Task::batch([stats_task, branch_task, ahead_task])
    }

    /// Return the selected workspace name, or `None` if no shared workspace
    /// is currently selected (empty-string "Personal" is treated as None).
    fn active_workspace_name(&self) -> Option<String> {
        match self.selected_workspace_name.as_deref() {
            Some(n) if !n.is_empty() => Some(n.to_string()),
            _ => None,
        }
    }

    /// Returns `true` when a shared (non-Personal) workspace is selected.
    /// Avoids the allocation of [`active_workspace_name`] for presence-only checks.
    fn has_active_workspace(&self) -> bool {
        self.selected_workspace_name
            .as_deref()
            .is_some_and(|n| !n.is_empty())
    }

    pub fn view(&self) -> Element<'_, Message> {
        if let Some(err) = &self.boot_error {
            return container(
                column![
                    text("MahBot failed to start")
                        .size(20)
                        .color(theme::STATUS_ERROR),
                    text(err).size(14).color(theme::TEXT_SECONDARY),
                ]
                .spacing(12)
                .padding(24),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into();
        }

        if !self.ready {
            return container(
                column![
                    text("MahBot").size(24).color(theme::ACCENT),
                    text("Starting…").size(16).color(theme::TEXT_MUTED),
                ]
                .spacing(16)
                .align_x(Alignment::Center),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x(Length::Fill)
            .center_y(Length::Fill)
            .into();
        }

        let sidebar = self.sidebar_view();
        let footer = self.footer_view();
        let content = match self.page {
            Page::Home => {
                let home_view = self.home_state.view().map(Message::Home);
                let clear_enabled = self.home_state.can_clear_chat();
                let sidebar = ticket_sidebar(&self.board_state, clear_enabled);
                let base = row![
                    container(home_view).width(Length::FillPortion(7)),
                    container(sidebar).width(Length::FillPortion(3))
                ];
                let modal = self.board_state.render_modal_overlay().map(Message::Board);
                iced::widget::stack([base.into(), modal]).into()
            }
            Page::Logs => self.logs_state.view().map(Message::Logs),
            Page::Sessions => self.sessions_state.view().map(Message::Sessions),
            Page::Shell => self.shell_state.view().map(Message::Shell),
            Page::Editor => self.editor_state.view().map(Message::Editor),
            Page::Settings => self
                .settings_state
                .view(self.selected_user_name.as_deref())
                .map(Message::Settings),
        };

        let body = column![
            row![sidebar, content]
                .width(Length::Fill)
                .height(Length::Fill),
            footer,
        ]
        .width(Length::Fill)
        .height(Length::Fill);

        // Keep Stack widget type stable to prevent state loss on toast
        // transitions. A type tag change (Column ↔ Stack) would cause Iced
        // to destroy the entire widget tree, losing scroll positions, cursor
        // states, and all other widget state.
        let overlay: Element<'_, Message> = if self.toasts.is_empty() {
            container(text(""))
                .width(Length::Shrink)
                .height(Length::Shrink)
                .into()
        } else {
            let mut toast_col = Column::new().spacing(6).align_x(Alignment::Center);
            for toast in &self.toasts {
                let (color, _bg) = match toast.kind {
                    ToastKind::Success => (theme::STATUS_SUCCESS, theme::BG_ELEVATED),
                    ToastKind::Warning => (theme::STATUS_WARNING, theme::BG_ELEVATED),
                    ToastKind::Error => (theme::STATUS_ERROR, theme::BG_ELEVATED),
                };
                let pill = container(text(&toast.message).size(12).color(color))
                    .padding([6, 14])
                    .style(move |_theme: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                        border: iced::Border {
                            radius: 20.0.into(),
                            width: 1.0,
                            color: theme::BORDER,
                        },
                        ..container::Style::default()
                    });
                toast_col = toast_col.push(pill);
            }
            container(toast_col)
                .width(Length::Fill)
                .align_x(Alignment::Center)
                .padding(iced::Padding {
                    bottom: 44.0,
                    ..Default::default()
                })
                .align_bottom(Length::Fill)
                .into()
        };

        // ── Diff modal overlay ─────────────────────────────────────
        let diff_overlay: Element<'_, Message> = if self.show_diff_modal {
            render_diff_modal(&self.diff_state)
        } else {
            container(text(""))
                .width(Length::Shrink)
                .height(Length::Shrink)
                .into()
        };

        // ── Branch management modal overlay ─────────────────────────
        let branch_overlay: Element<'_, Message> = if self.show_branch_modal {
            render_branch_modal(
                &self.local_branches,
                &self.branch_search_query,
                self.git_branch_error.as_ref(),
                self.git_syncing,
                &self.new_branch_name,
            )
        } else {
            container(text(""))
                .width(Length::Shrink)
                .height(Length::Shrink)
                .into()
        };

        iced::widget::stack![body, diff_overlay, branch_overlay, overlay].into()
    }
}

/// Render the diff modal (80% width, 100% height, centered).
fn render_diff_modal(diff_state: &diff::DiffState) -> Element<'_, Message> {
    let backdrop = iced::widget::mouse_area(
        container(text(""))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(
                    0.0, 0.0, 0.0, 0.5,
                ))),
                ..container::Style::default()
            }),
    )
    .on_press(Message::CloseDiffModal);

    let viewing_commit = diff_state.is_viewing_commit();

    // Outer header: commit message (large, bold) + short hash (muted) for
    // historical commits, or "Uncommitted changes" for working-tree diff.
    let header: Element<'_, Message> = if viewing_commit {
        let msg = diff_state
            .commit_message()
            .unwrap_or("(no commit message)")
            .to_string();
        let hash = diff_state.commit_short_hash().unwrap_or("????????");
        column![
            text(msg).size(18).color(theme::TEXT_PRIMARY),
            text(hash).size(12).color(theme::TEXT_MUTED),
        ]
        .spacing(2)
        .padding(iced::Padding {
            top: 0.0,
            right: 0.0,
            bottom: 12.0,
            left: 0.0,
        })
        .into()
    } else {
        column![
            text("Uncommitted changes")
                .size(18)
                .color(theme::TEXT_PRIMARY),
            text("Working tree diff \u{2014} press Escape to close")
                .size(11)
                .color(theme::TEXT_MUTED),
        ]
        .spacing(4)
        .padding(iced::Padding {
            top: 0.0,
            right: 0.0,
            bottom: 12.0,
            left: 0.0,
        })
        .into()
    };

    let diff_content: Element<'_, diff::DiffMessage> = diff_state.view();
    let inner = column![header, diff_content.map(Message::DiffModal)].spacing(0);

    // 80% width with 10% margin each side via a row with FillPortion spacers
    let dialog = container(inner)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(16)
        .style(|_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(theme::BG_ELEVATED)),
            border: iced::Border {
                radius: 8.0.into(),
                width: 1.0,
                color: theme::BORDER_STRONG,
            },
            ..container::Style::default()
        });

    let centered = row![
        Space::new().width(Length::FillPortion(1)), // 10% margin
        dialog.width(Length::FillPortion(8)),       // 80% content
        Space::new().width(Length::FillPortion(1)), // 10% margin
    ]
    .width(Length::Fill)
    .height(Length::Fill);

    iced::widget::stack([backdrop.into(), centered.into()]).into()
}

/// Render the branch management modal (80% width, 100% height).
fn render_branch_modal<'a>(
    branches: &'a [String],
    search_query: &'a str,
    error: Option<&'a String>,
    syncing: bool,
    new_branch_name: &'a str,
) -> Element<'a, Message> {
    use iced::widget::text_input;
    let backdrop = iced::widget::mouse_area(
        container(text(""))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(iced::Color::from_rgba(
                    0.0, 0.0, 0.0, 0.5,
                ))),
                ..container::Style::default()
            }),
    )
    .on_press(Message::CloseBranchModal);

    let search_input = text_input("Search branches…", search_query)
        .on_input(Message::BranchQueryChanged)
        .on_submit(Message::Nop)
        .padding(8)
        .size(14);

    // Filter branches by search query
    let filtered: Vec<&String> = if search_query.is_empty() {
        branches.iter().collect()
    } else {
        let q = search_query.to_lowercase();
        branches
            .iter()
            .filter(|b| b.to_lowercase().contains(&q))
            .collect()
    };

    let branch_items: Vec<Element<'a, Message>> = filtered
        .iter()
        .map(|branch| {
            let b = (*branch).clone();
            button(text(b.clone()).size(14).color(theme::TEXT_PRIMARY))
                .padding([6, 12])
                .width(Length::Fill)
                .style(theme::button_text)
                .on_press_maybe(if syncing {
                    None
                } else {
                    Some(Message::GitSwitch(b.clone()))
                })
                .into()
        })
        .collect();

    let list = scrollable(Column::with_children(branch_items).spacing(2))
        .height(Length::Fill)
        .style(theme::scrollbar_style);

    // Error display
    let error_elem: Element<'a, Message> = if let Some(err) = error {
        text(err).size(12).color(theme::STATUS_ERROR).into()
    } else {
        container(text("")).into()
    };

    // Create new branch input + button
    let create_input = text_input("New branch name…", new_branch_name)
        .on_input(Message::NewBranchNameChanged)
        .on_submit(Message::GitCreate)
        .padding(8)
        .size(14);

    let create_btn = button(text("Create & Switch").size(14).color(theme::TEXT_PRIMARY))
        .padding([6, 12])
        .style(theme::button_primary)
        .on_press_maybe(if syncing {
            None
        } else {
            Some(Message::GitCreate)
        });

    let inner = column![
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
    .height(Length::Fill);

    // 80% width with 10% margin each side via a row with FillPortion spacers
    let dialog = container(inner)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(16)
        .style(|_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(theme::BG_ELEVATED)),
            border: iced::Border {
                radius: 8.0.into(),
                width: 1.0,
                color: theme::BORDER_STRONG,
            },
            ..container::Style::default()
        });

    let centered = row![
        Space::new().width(Length::FillPortion(1)), // 10% margin
        dialog.width(Length::FillPortion(8)),       // 80% content
        Space::new().width(Length::FillPortion(1)), // 10% margin
    ]
    .width(Length::Fill)
    .height(Length::Fill);

    iced::widget::stack([backdrop.into(), centered.into()]).into()
}

// ── Ticket sidebar (Home page, right side) ────────────────────────

/// Ticket sidebar shown on the right side of the Home page.
/// Displays all non-archived tickets grouped by status, with a
/// batch-archive button for completed tickets.
fn ticket_sidebar(board_state: &board::BoardState, clear_enabled: bool) -> Element<'_, Message> {
    let (pending, pipeline, completed) = board::BoardState::partition_tickets(&board_state.tickets);

    // Split pipeline into "pinned" (actively working) and "ready"
    // (ReadyForDevelopment only). partition_tickets lumps them together,
    // but the sidebar separates them visually.
    let pinned: Vec<&Ticket> = pipeline
        .iter()
        .filter(|t| t.status != TicketPhase::ReadyForDevelopment)
        .copied()
        .collect();
    let ready: Vec<&Ticket> = pipeline
        .iter()
        .filter(|t| t.status == TicketPhase::ReadyForDevelopment)
        .copied()
        .collect();

    let has_completed = !completed.is_empty();
    let is_empty = pending.is_empty() && pipeline.is_empty() && completed.is_empty();

    // Header row: Clear button (replaces "Tickets" title) + archive button
    let clear_icon = lucide::eraser::<iced::Theme, iced::Renderer>()
        .size(14)
        .color(if clear_enabled {
            theme::TEXT_MUTED
        } else {
            theme::TEXT_FAINT
        });
    let clear_btn = tooltip(
        button(clear_icon)
            .on_press_maybe(if clear_enabled {
                Some(Message::Home(home::HomeMessage::ClearChat))
            } else {
                None
            })
            .padding(4)
            .style(theme::button_text),
        text("Clear chat").size(11),
        tooltip::Position::Bottom,
    )
    .style(theme::tooltip_style);
    let archive_icon = lucide::archive::<iced::Theme, iced::Renderer>()
        .size(12)
        .color(if has_completed {
            theme::TEXT_SECONDARY
        } else {
            theme::TEXT_FAINT
        });
    let archive_btn = tooltip(
        button(archive_icon)
            .on_press_maybe(if has_completed {
                Some(Message::Board(board::BoardMessage::ArchiveAllCompleted))
            } else {
                None
            })
            .padding(4)
            .style(theme::button_text),
        text("Archive done & cancelled").size(11),
        tooltip::Position::Top,
    )
    .style(theme::tooltip_style);
    let header =
        row![clear_btn, Space::new().width(Length::Fill), archive_btn].align_y(Alignment::Center);

    // Body: loading, empty, or ticket groups
    let body: Element<'_, Message> = if !board_state.has_loaded {
        column![
            Space::new().height(8),
            text("Loading…").size(12).color(theme::TEXT_MUTED),
        ]
        .spacing(4)
        .padding([8, 0])
        .into()
    } else if is_empty {
        column![
            Space::new().height(8),
            text("No tickets").size(12).color(theme::TEXT_MUTED),
        ]
        .spacing(4)
        .padding([8, 0])
        .into()
    } else {
        let mut groups = Column::new().spacing(8);
        if !pinned.is_empty() {
            groups = groups.push(group_section("In Progress", &pinned, board_state));
        }
        if !ready.is_empty() {
            groups = groups.push(group_section("Ready", &ready, board_state));
        }
        if !pending.is_empty() {
            groups = groups.push(group_section("Pending", &pending, board_state));
        }
        if !completed.is_empty() {
            groups = groups.push(group_section("Completed", &completed, board_state));
        }
        scrollable(groups)
            .height(Length::Fill)
            .direction(theme::vertical_scrollbar())
            .style(theme::scrollbar_style)
            .into()
    };

    let content = column![header, Space::new().height(8), body].spacing(0);

    container(content)
        .padding([8, 12])
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::surface_container_style)
        .into()
}

/// Render a group of tickets with a header label.
fn group_section<'a>(
    label: &'static str,
    tickets: &[&'a Ticket],
    board_state: &'a board::BoardState,
) -> Element<'a, Message> {
    let header = text(label).size(11).color(theme::TEXT_SECONDARY);

    let mut cards = Column::new().spacing(2);
    for ticket in tickets {
        cards = cards.push(board_state.render_ticket_card(ticket).map(Message::Board));
    }

    column![header, Space::new().height(4), cards]
        .spacing(0)
        .into()
}

impl Dashboard {
    pub fn subscription(&self) -> iced::Subscription<Message> {
        if !self.ready {
            return iced::Subscription::none();
        }
        iced::Subscription::batch([
            iced::time::every(Duration::from_secs(1)).map(|_| Message::Tick),
            window::close_requests().map(Message::CloseRequested),
            window::resize_events()
                .map(|(id, size)| Message::WindowEvent(id, window::Event::Resized(size))),
            window::events().filter_map(|(id, event)| {
                matches!(
                    &event,
                    window::Event::Moved(_) | window::Event::Opened { .. }
                )
                .then_some(Message::WindowEvent(id, event))
            }),
            keyboard::listen().filter_map(|event| {
                use keyboard::{Event, Key};
                let pressed = matches!(event, Event::KeyPressed { .. });
                if !pressed {
                    return None;
                }
                let Event::KeyPressed {
                    key,
                    modifiers,
                    physical_key,
                    ..
                } = event
                else {
                    return None;
                };
                let km = detect_keyboard_mods(modifiers);

                let latin = key.to_latin(physical_key);
                // Cmd+F (macOS) / Ctrl+F (other) → focus search.
                if !km.altgr_active && km.is_cmd && !modifiers.shift() && latin == Some('f') {
                    return Some(Message::FocusSearch);
                }
                if let Key::Named(iced::keyboard::key::Named::Escape) = key {
                    Some(Message::EscapePressed)
                } else if km.is_cmd && !km.altgr_active {
                    // Cmd+number → navigate to page.
                    if let Some(digit) = latin.and_then(|c| c.to_digit(10)) {
                        let idx = digit as usize;
                        if idx >= 1 {
                            let pages = Page::sidebar_pages();
                            if let Some(page) = pages.get(idx - 1).copied() {
                                return Some(Message::Navigation(page));
                            }
                        }
                    }
                    None
                } else {
                    None
                }
            }),
            self.shell_state.subscription().map(Message::Shell),
            self.logs_state.subscription().map(Message::Logs),
            self.board_state.subscription().map(Message::Board),
            self.sessions_state.subscription().map(Message::Sessions),
            self.editor_state.subscription().map(Message::Editor),
            self.home_state.subscription().map(Message::Home),
            iced::Subscription::run(shutdown_subscription),
            // Diff modal subscription (keyboard shortcuts, auto-refresh).
            // Only active when the modal is open to avoid intercepting
            // global keyboard shortcuts unnecessarily.
            if self.show_diff_modal {
                self.diff_state.subscription().map(Message::DiffModal)
            } else {
                iced::Subscription::none()
            },
        ])
    }
}

/// Subscription that emits [`Message::Shutdown`] when the global shutdown
/// token fires (self-update restart, SIGTERM/SIGINT).
fn shutdown_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    iced::stream::channel(1, |mut output: mpsc::Sender<Message>| async move {
        crate::shutdown::shutdown_token().cancelled().await;
        let _ = output.try_send(Message::Shutdown);
    })
}

// ── Navigation sidebar ──────────────────────────────────────────

impl Dashboard {
    fn sidebar_view(&self) -> Element<'_, Message> {
        // Sidebar navigation: Home, Editor, Shell (icon-only, 28px)
        let mut nav_col = Column::new().spacing(4);
        for page in Page::sidebar_pages() {
            let is_active = self.page == *page;
            // Editor, Shell require any workspace (shared or personal with a user selected).
            let has_any_workspace =
                self.selected_workspace_name.is_some() || self.selected_user_name.is_some();
            let requires_workspace = matches!(*page, Page::Editor | Page::Shell);
            let disabled = requires_workspace && !has_any_workspace;

            let color = if is_active {
                theme::ACCENT
            } else if disabled {
                theme::TEXT_FAINT
            } else {
                theme::TEXT_MUTED
            };
            let icon: iced::Element<'_, Message> = match page {
                Page::Home => lucide::layout_dashboard::<iced::Theme, iced::Renderer>()
                    .size(28)
                    .color(color)
                    .into(),
                Page::Shell => lucide::terminal::<iced::Theme, iced::Renderer>()
                    .size(28)
                    .color(color)
                    .into(),
                Page::Editor => lucide::pencil_line::<iced::Theme, iced::Renderer>()
                    .size(28)
                    .color(color)
                    .into(),
                _ => text("").into(),
            };
            let btn = button(
                container(icon)
                    .width(Length::Fill)
                    .center_x(Length::Fill)
                    .padding([4, 0]),
            )
            .width(Length::Fill)
            .padding(0)
            .style(theme::button_text)
            .on_press_maybe(if disabled {
                None
            } else {
                Some(Message::Navigation(*page))
            });
            nav_col = nav_col.push(btn);
        }

        // Spacer to push buttons to the bottom of the sidebar
        nav_col = nav_col.push(Space::new().height(Length::Fill));

        // Determine whether a shared workspace is selected (Personal mode has no
        // workspace-level pipeline or maintainer toggles).
        let has_ws = self.has_active_workspace();

        // Per-workspace Maintainer toggle.
        // Positioned immediately above the pause button.
        let maint_icon = column![
            text("Maint").size(8).color(theme::TEXT_MUTED),
            text(if self.maintenance { "ON" } else { "OFF" })
                .size(9)
                .color(if self.maintenance {
                    theme::ACCENT
                } else {
                    theme::TEXT_MUTED
                }),
        ]
        .spacing(0)
        .align_x(Alignment::Center);
        let maint_btn = tooltip(
            button(
                container(maint_icon)
                    .width(Length::Fill)
                    .center_x(Length::Fill)
                    .padding([4, 0]),
            )
            .width(Length::Fill)
            .padding(0)
            .style(theme::button_text)
            .on_press_maybe(if has_ws {
                Some(Message::ToggleMaintenance)
            } else {
                None
            }),
            text(if !has_ws {
                "Select a workspace to toggle maintainer"
            } else if self.maintenance {
                "Maintainer ON"
            } else {
                "Maintainer OFF"
            })
            .size(11),
            tooltip::Position::Top,
        )
        .style(theme::tooltip_style);
        nav_col = nav_col.push(maint_btn);

        // Per-workspace pipeline pause/unpause toggle.
        // Disabled when no workspace is selected (Personal mode).
        let pause_icon = if !has_ws {
            lucide::pause::<iced::Theme, iced::Renderer>()
                .size(28)
                .color(theme::TEXT_FAINT)
        } else if self.paused {
            lucide::play::<iced::Theme, iced::Renderer>()
                .size(28)
                .color(theme::ACCENT)
        } else {
            lucide::pause::<iced::Theme, iced::Renderer>()
                .size(28)
                .color(theme::TEXT_MUTED)
        };
        let pause_btn = tooltip(
            button(
                container(pause_icon)
                    .width(Length::Fill)
                    .center_x(Length::Fill)
                    .padding([4, 0]),
            )
            .width(Length::Fill)
            .padding(0)
            .style(theme::button_text)
            .on_press_maybe(if has_ws {
                Some(Message::TogglePause)
            } else {
                None
            }),
            text(if !has_ws {
                "Select a workspace to pause"
            } else if self.paused {
                "Resume pipeline"
            } else {
                "Pause pipeline"
            })
            .size(11),
            tooltip::Position::Top,
        )
        .style(theme::tooltip_style);
        nav_col = nav_col.push(pause_btn);

        let inner = nav_col.spacing(2);

        container(inner)
            .width(Length::Fixed(130.0))
            .height(Length::Fill)
            .style(theme::surface_container_style)
            .padding(12)
            .into()
    }

    /// 24px footer bar — nav items (left) and active agents (right).
    fn footer_view(&self) -> Element<'_, Message> {
        // Left: footer navigation (Sessions, Logs, Settings) + git blocks
        // Icon-only, 16px. Active page in ACCENT, inactive in TEXT_MUTED.
        let mut left_icons = Vec::with_capacity(10);

        // Update button — leftmost, disabled while updating.
        // Only shown when self-update is available on this installation.
        if self.update_available {
            let update_color = if self.updating {
                theme::TEXT_FAINT
            } else {
                theme::ACCENT
            };
            let update_icon = lucide::refresh_cw::<iced::Theme, iced::Renderer>()
                .size(16)
                .color(update_color);
            let update_btn = button(update_icon)
                .style(theme::button_text)
                .padding(2)
                .on_press_maybe(if self.updating {
                    None
                } else {
                    Some(Message::UpdateBot)
                });
            left_icons.push(update_btn.into());
        }

        for page in Page::footer_pages() {
            let is_active = self.page == *page;
            let color = if is_active {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            };
            let icon: iced::Element<'_, Message> = match page {
                Page::Sessions => lucide::scroll_text::<iced::Theme, iced::Renderer>()
                    .size(16)
                    .color(color)
                    .into(),
                Page::Logs => lucide::activity::<iced::Theme, iced::Renderer>()
                    .size(16)
                    .color(color)
                    .into(),
                Page::Settings => lucide::settings::<iced::Theme, iced::Renderer>()
                    .size(16)
                    .color(color)
                    .into(),
                _ => text("").into(),
            };
            let btn = button(icon)
                .style(theme::button_text)
                .padding(2)
                .on_press(Message::Navigation(*page));
            left_icons.push(btn.into());
        }

        // Git blocks — branch, sync, diff — after Settings,
        // visually grouped with a small gap from nav buttons.
        let has_fs = self.workspace_filesystem_path.is_some();

        if has_fs {
            left_icons.push(Space::new().width(6).into());

            // a) Branch name — clickable -> branch modal
            // Only shown when a branch is known.
            if let Some(b) = &self.git_current_branch {
                let truncated = if b.len() > 20 {
                    // Safe truncation at char boundary
                    let mut end = 19;
                    while !b.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!("{}…", &b[..end])
                } else {
                    b.clone()
                };
                let branch_btn = button(text(truncated).size(11).color(theme::ACCENT))
                    .style(theme::button_text)
                    .padding(2)
                    .on_press(Message::OpenBranchModal);
                left_icons.push(branch_btn.into());
            }

            // b) Sync — ↻ icon + behind/ahead counts, clickable -> git sync
            // ↑ for ahead (push up), ↓ for behind (pull down)
            if let Some((behind, ahead)) = self.git_behind_ahead {
                if behind > 0 || ahead > 0 {
                    let sync_text = if ahead > 0 && behind > 0 {
                        format!("\u{2191}{ahead} \u{2193}{behind}")
                    } else if ahead > 0 {
                        format!("\u{2191}{ahead}")
                    } else {
                        format!("\u{2193}{behind}")
                    };
                    let sync_btn = button(
                        row![
                            text("\u{21bb}").size(14).color(if self.git_syncing {
                                theme::TEXT_MUTED
                            } else {
                                theme::ACCENT
                            }),
                            text(sync_text).size(11).color(theme::TEXT_MUTED),
                        ]
                        .spacing(4)
                        .align_y(Alignment::Center),
                    )
                    .padding(2)
                    .style(theme::button_text)
                    .on_press_maybe(if self.git_syncing {
                        None
                    } else {
                        Some(Message::GitSync)
                    });
                    left_icons.push(sync_btn.into());
                }
            }

            // c) Diff stats — ticket card format (+X/−Y), clickable -> diff modal
            // Only rendered when there are non-zero changes.
            if let Some((added, removed)) = self.git_diff_stats {
                if added > 0 || removed > 0 {
                    let stats_row = widgets::diff_stats_row::<Message>(added, removed);
                    let diff_btn = button(stats_row)
                        .style(theme::button_text)
                        .padding(2)
                        .on_press(Message::OpenDiffModal(None));
                    left_icons.push(diff_btn.into());
                }
            }
        }

        let left = Row::with_children(left_icons)
            .spacing(4)
            .align_y(Alignment::Center);

        // Right: active agent icons (horizontal, one per role, "×N" for multiples)
        let right: iced::Element<'_, Message> = {
            let handles = crate::registry::AGENT_REGISTRY.list();
            let mut role_counts: std::collections::BTreeMap<&str, usize> =
                std::collections::BTreeMap::new();
            for h in &handles {
                *role_counts.entry(h.role.as_str()).or_insert(0) += 1;
            }
            if role_counts.is_empty() {
                text("").into()
            } else {
                let mut icons: Vec<iced::Element<'_, Message>> = Vec::new();
                for (role_str, count) in &role_counts {
                    let role: crate::Role = role_str.parse().unwrap_or(crate::Role::Engineer);
                    let (color, _bg) = theme::role_badge_color_for(&role);
                    let icon = theme::role_icon(&role).size(16).color(color);
                    if *count > 1 {
                        let label = text(format!("×{count}")).size(10).color(color);
                        icons.push(
                            container(row![icon, label].spacing(2).align_y(Alignment::Center))
                                .padding(iced::Padding {
                                    left: 2.0,
                                    right: 2.0,
                                    top: 0.0,
                                    bottom: 0.0,
                                })
                                .into(),
                        );
                    } else {
                        icons.push(
                            container(icon)
                                .padding(iced::Padding {
                                    left: 2.0,
                                    right: 2.0,
                                    top: 0.0,
                                    bottom: 0.0,
                                })
                                .into(),
                        );
                    }
                }
                let c = Row::with_children(icons)
                    .spacing(8)
                    .align_y(Alignment::Center);
                c.into()
            }
        };

        let footer_row = row![left, Space::new().width(Length::Fill), right]
            .align_y(Alignment::Center)
            .padding(iced::Padding {
                top: 0.0,
                right: 12.0,
                bottom: 4.0,
                left: 12.0,
            });

        container(footer_row)
            .height(Length::Fixed(24.0))
            .width(Length::Fill)
            .style(theme::surface_container_style)
            .into()
    }
}

/// Persisted window geometry.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct WindowState {
    pub width: f32,
    pub height: f32,
    pub x: i32,
    pub y: i32,
    #[serde(default)]
    pub selected_workspace: Option<String>,
    #[serde(default)]
    pub selected_user: Option<String>,
}

impl WindowState {
    /// Position to use when restoring the window.
    #[allow(clippy::cast_precision_loss)]
    pub const fn position(&self) -> iced::window::Position {
        iced::window::Position::Specific(iced::Point::new(self.x as f32, self.y as f32))
    }
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            width: 1500.0,
            height: 800.0,
            x: -1,
            y: -1,
            selected_workspace: None,
            selected_user: None,
        }
    }
}

/// Read persisted window state from `~/.mahbot/window-state.json`.
/// Returns defaults if the file is missing or unreadable.
pub fn read_window_state() -> WindowState {
    let dir = std::env::var("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".mahbot"))
        .ok();
    let path = dir.map(|d| d.join("window-state.json"));
    path.as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

/// Save current window geometry and last-used workspace to `~/.mahbot/window-state.json`.
#[allow(clippy::cast_possible_truncation)]
fn save_window_state(
    pos: iced::Point,
    size: iced::Size,
    selected_workspace: Option<&str>,
    selected_user: Option<&str>,
) {
    let mut state = serde_json::json!({
        "width": size.width,
        "height": size.height,
        "x": pos.x as i32,
        "y": pos.y as i32,
    });
    if let Some(ws) = selected_workspace {
        state["selected_workspace"] = serde_json::Value::String(ws.to_string());
    }
    if let Some(user) = selected_user {
        state["selected_user"] = serde_json::Value::String(user.to_string());
    }
    if let Ok(dir) = std::env::var("HOME") {
        let path = std::path::PathBuf::from(dir)
            .join(".mahbot")
            .join("window-state.json");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, state.to_string());
    }
}

/// Lightweight async task that re-reads all workspace paused and maintenance
/// states from the DB, returning a [`Message::WorkspaceStatesRefreshed`].
///
/// This is a targeted refresh of only the boolean toggle state — unlike
/// [`load_workspace_options`] it does not rebuild the workspace picker list,
/// trigger page re-propagation, or load users.
fn refresh_workspace_states_task() -> Task<Message> {
    Task::perform(
        async {
            let store = crate::workspace::store();
            match store.list_states().await {
                Ok(states) => {
                    let mut paused = HashMap::with_capacity(states.len());
                    let mut maintenance = HashMap::with_capacity(states.len());
                    for (name, is_paused, is_maint) in states {
                        paused.insert(name.clone(), is_paused);
                        maintenance.insert(name, is_maint);
                    }
                    Message::WorkspaceStatesRefreshed(paused, maintenance)
                }
                Err(e) => {
                    tracing::warn!("Failed to refresh workspace states: {e}");
                    // Keep existing cached state — don't loop back into the
                    // periodic tick handler.  The next real 1-second Tick will
                    // re-attempt the refresh.
                    Message::Nop
                }
            }
        },
        std::convert::identity,
    )
}

/// Load workspace `PickOption` list and path map from the workspace store,
/// resolving `prev_selection` against the loaded list. Falls back to the
/// first available workspace when `prev_selection` is absent or stale.
/// Returns a `BootWorkspaces` message ready for use with `Task::perform`.
async fn load_workspace_options(prev_selection: Option<String>) -> Message {
    let store = crate::workspace::store();
    let mut options = Vec::new();
    let mut paths = HashMap::new();
    let mut paused_map = HashMap::new();
    let mut maintenance_map = HashMap::new();
    let mut restored_name = None;

    // "Personal" option — represents a user's personal workspace (selected_workspace=NULL).
    options.push(PickOption {
        value: String::new(),
        label: "Personal".to_string(),
    });

    if let Ok(ws_list) = store.list().await {
        for ws in &ws_list {
            let display = ws.display_name();
            paths.insert(ws.name.clone(), ws.path.clone());
            paused_map.insert(ws.name.clone(), ws.paused);
            maintenance_map.insert(ws.name.clone(), ws.maintenance);
            options.push(PickOption {
                value: ws.name.clone(),
                label: display,
            });
        }
    }

    if let Some(ref name) = prev_selection {
        // Empty string means "Personal" — always valid.
        if name.is_empty() || paths.contains_key(name.as_str()) {
            restored_name = Some(name.clone());
        }
    }
    if restored_name.is_none() && !options.is_empty() {
        restored_name = Some(options[0].value.clone());
    }

    Message::BootWorkspaces(options, paths, paused_map, maintenance_map, restored_name)
}

/// Open a URL in the system browser (fire-and-forget).
fn open_url(url: &str) {
    let _ = if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else if cfg!(target_os = "linux") {
        std::process::Command::new("xdg-open").arg(url).spawn()
    } else if cfg!(target_os = "windows") {
        std::process::Command::new("cmd")
            .args(["/c", "start", url])
            .spawn()
    } else {
        return;
    };
}

/// Resolve a workspace name+path pair from the workspace map and optional
/// personal workspace path.  If `ws_path` is `Some`, that takes priority;
/// otherwise `personal_path` is used as a fallback ("Personal workspace — use
/// resolved user path").  When `name` is empty and no path is available,
/// returns `None` for the path ("Personal workspace without a selected user —
/// no path to send").  Logs a warning for non-empty names where neither path
/// source is available (possible DB inconsistency).
fn resolve_workspace_path(
    name: &str,
    ws_path: Option<&String>,
    personal_path: Option<&String>,
) -> (String, Option<String>) {
    if let Some(p) = ws_path {
        (name.to_string(), Some(p.clone()))
    } else if let Some(p) = personal_path {
        (name.to_string(), Some(p.clone()))
    } else if name.is_empty() {
        (String::new(), None)
    } else {
        tracing::warn!(
            workspace = name,
            "Workspace path not found in map — sending empty selection"
        );
        (String::new(), None)
    }
}
