//! Native Iced dashboard — application entry point, navigation, and shared state.
//!
//! Iced owns the process Tokio runtime (`iced` feature `tokio`). MahBot
//! bootstraps via a startup [`iced::Task`] before the UI becomes interactive.

#![allow(
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::struct_excessive_bools,
    clippy::if_not_else,
    clippy::collapsible_if
)]

pub mod board;
pub mod common;
pub mod context_menu;
pub mod diff;
pub mod diff_widget;
pub mod editor;
pub mod editor_widget;
pub mod git;
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
use iced::widget::rule;
use iced::widget::{Column, Row, button, column, container, row, scrollable, text, tooltip};
use iced::window;
use iced::{Alignment, Color, Element, Length, Task};

use self::context_menu::ContextMenu;

use iced_fonts::lucide;

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
    /// WAL checkpoint complete — safe to exit now.
    CheckpointAndExit,
    /// Periodic refresh of workspace paused/maintenance state from DB.
    WorkspaceStatesRefreshed(HashMap<String, bool>, HashMap<String, bool>),
    /// No-op — produced by refresh helpers on transient DB errors to avoid
    /// sending empty state maps that would wipe cached toggle state.
    Nop,
    /// Workspace paths and state loaded during boot (paths, paused, maintenance, restored selection).
    BootWorkspaces(
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
    /// Git sub-state message.
    Git(git::GitMessage),
    Shell(shell::ShellMessage),
    Editor(editor::EditorMessage),
    Settings(settings::SettingsMessage),

    // ── Diff modal ──────────────────────────────────────────────
    /// Open the diff modal. Optional commit hash — `None` = working tree diff.
    OpenDiffModal(Option<String>),
    /// Close the diff modal.
    CloseDiffModal,
}

// ── Message introspection helpers ────────────────────────────────
//
// These methods let [`Dashboard::update`] intercept Toast and
// LinkClicked messages before dispatching to page handlers,
// consolidating what would otherwise be per-page boilerplate.

impl Message {
    /// Returns a reference to the inner [`ToastMessage`] if this message wraps one.
    pub(crate) fn as_toast(&self) -> Option<&ToastMessage> {
        match self {
            Message::Home(home::HomeMessage::Toast(tm))
            | Message::Logs(
                logs::LogMessage::Toast(tm)
                | logs::LogMessage::ToolFailures(ToolFailuresMessage::Toast(tm)),
            )
            | Message::Board(board::BoardMessage::Toast(tm))
            | Message::DiffModal(diff::DiffMessage::Toast(tm))
            | Message::Git(git::GitMessage::Toast(tm))
            | Message::Editor(editor::EditorMessage::Toast(tm))
            | Message::Settings(
                settings::SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::Toast(tm))
                | settings::SettingsMessage::UserMsg(users::UsersMessage::Toast(tm)),
            ) => Some(tm),
            _ => None,
        }
    }

    /// Returns the URL string if this message wraps a `LinkClicked`.
    ///
    /// # Design note
    ///
    /// [`HomeMessage::LinkClicked`] is deliberately **not** included here
    /// because `home.rs` handles its own inline context links internally
    /// (see `HomeState::update`).  Do not add it without understanding
    /// the Home page's self-handling logic.
    pub(crate) fn as_link_url(&self) -> Option<&str> {
        match self {
            Message::Board(board::BoardMessage::LinkClicked(url))
            | Message::Sessions(sessions::SessionsMessage::LinkClicked(url))
            | Message::Settings(settings::SettingsMessage::WorkspaceMsg(
                workspaces::WorkspacesMessage::LinkClicked(url),
            )) => Some(url.as_str()),
            _ => None,
        }
    }
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

    /// Platform modifier for general keyboard shortcuts
    /// (everything except navigation and text operations).
    ///
    /// On macOS: Cmd is pressed (with or without Ctrl), but not Ctrl alone
    /// (which triggers terminal control characters / Emacs bindings).
    ///
    /// On other platforms: Cmd or Ctrl is pressed, but not AltGr
    /// (Ctrl+Alt, which produces international text characters).
    #[must_use]
    pub fn is_shortcut_platform_mod(self) -> bool {
        self.is_platform_mod && !self.is_emacs_ctrl && !self.altgr_active
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
    /// All git-related state (branch info, sync, branch modal).
    git_state: git::GitState,
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
            workspace_paths: HashMap::new(),
            workspace_paused: HashMap::new(),
            workspace_maintenance: HashMap::new(),
            selected_workspace_name: None,
            selected_user_name: None,
            updating: false,
            update_available,
            logs_state: logs::LogsState::new(),
            board_state: board::BoardState::new(),
            sessions_state: sessions::SessionsState::new(),
            diff_state: diff::DiffState::new(),
            home_state: home::HomeState::new(),
            shell_state: shell::ShellState::new(),
            editor_state: editor::EditorState::new(),
            settings_state: settings::SettingsState::new(),
            show_diff_modal: false,
            git_state: git::GitState::new(),
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

    /// Look up a boolean flag for the currently selected workspace.
    fn workspace_flag(&self, map: &HashMap<String, bool>) -> bool {
        self.selected_workspace_name
            .as_ref()
            .and_then(|name| map.get(name))
            .copied()
            .unwrap_or(false)
    }

    /// Whether the selected workspace's pipeline is paused (no new tickets claimed).
    fn paused(&self) -> bool {
        self.workspace_flag(&self.workspace_paused)
    }

    /// Whether the selected workspace's maintainer is enabled.
    fn maintenance(&self) -> bool {
        self.workspace_flag(&self.workspace_maintenance)
    }

    pub const fn theme(&self) -> iced::Theme {
        iced::Theme::Dark
    }

    /// Persist the current window position, size, selected workspace, and
    /// selected user to `~/.mahbot/window-state.json`.
    fn persist_window_state(&self) {
        save_window_state(
            self.last_position,
            self.last_size,
            self.selected_workspace_name.as_deref(),
            self.selected_user_name.as_deref(),
        );
    }

    fn save_and_exit(&self) -> Task<Message> {
        self.persist_window_state();
        Task::perform(crate::checkpoint::checkpoint_all_databases(), |()| {
            Message::CheckpointAndExit
        })
    }

    /// Window title with page name.
    pub fn title(&self) -> String {
        let page_name = self.page.label();
        format!("MahBot — {page_name}")
    }

    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, message: Message) -> Task<Message> {
        // ── Centralized Toast and LinkClicked interception ──────────
        // Intercepted at the Dashboard level (before page dispatch)
        // so page handlers only need a Task::none() arm for match exhaustiveness.
        // HomeMessage::LinkClicked is handled by Home itself — see
        // as_link_url() for details.
        if self.ready {
            if let Some(tm) = message.as_toast() {
                self.toasts.push(Toast::from_toast_msg(tm));
                return Task::none();
            }

            if let Some(url) = message.as_link_url() {
                open_url(url);
                return Task::none();
            }
        }

        match message {
            Message::Boot(result) => self.finish_boot(result),
            Message::BootWorkspaces(paths, paused_map, maintenance_map, restored_name) => {
                self.workspace_paths = paths;
                self.workspace_paused = paused_map;
                self.workspace_maintenance = maintenance_map;
                // Pre-set Home's selected_user from persisted window state
                // so UsersLoaded doesn't auto-select the first user when
                // a previous user was saved.
                if let Some(ref user_name) = self.selected_user_name {
                    self.home_state.selected_user = Some(user_name.clone());
                }
                let load_users = self.home_state.load_users().map(Message::Home);

                // restored_name is always Some — load_workspace_options sets it.
                // Empty string => "Personal" workspace (no shared workspace).
                let ws_name = match restored_name {
                    Some(ref name) if name.is_empty() => {
                        self.selected_workspace_name = None;
                        String::new()
                    }
                    Some(ref name) => {
                        self.selected_workspace_name = Some(name.clone());
                        name.clone()
                    }
                    None => {
                        // Unreachable: load_workspace_options always produces Some.
                        // Defensive fallback — treat as Personal workspace.
                        self.selected_workspace_name = None;
                        String::new()
                    }
                };
                Task::batch([self.propagate_workspace_selection(&ws_name), load_users])
            }
            Message::Navigation(_) if !self.ready => Task::none(),
            Message::Navigation(page) => {
                self.page = page;
                // Notify sessions state when navigating to/from Sessions page
                // so the auto-refresh timer starts/stops accordingly.
                self.sessions_state.set_page_active(page == Page::Sessions);
                match page {
                    // Logs and Shell maintain their own internal state; Editor
                    // receives workspace state via WorkspaceSelected from the
                    // Home page picker — none need a refresh on navigation.
                    Page::Logs | Page::Shell | Page::Editor => Task::none(),
                    Page::Home => {
                        let load_users = self.home_state.load_users().map(Message::Home);
                        let snap =
                            iced::widget::operation::snap_to_end::<Message>(home::CHAT_SCROLL_ID);
                        let board_refresh = self.board_state.refresh().map(Message::Board);
                        Task::batch([load_users, snap, board_refresh])
                    }
                    Page::Sessions => self.sessions_state.refresh().map(Message::Sessions),
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
                    Page::Home if !self.board_state.load_state.loading() => {
                        self.board_state.load_state.start_loading();
                        self.board_state.refresh().map(Message::Board)
                    }
                    Page::Sessions if !self.sessions_state.load_state.loading() => {
                        self.sessions_state.load_state.start_loading();
                        self.sessions_state.refresh().map(Message::Sessions)
                    }
                    Page::Settings => {
                        // Refresh workspace and user lists when on Settings page
                        let ws_loading = self.settings_state.workspaces_state.load_state.loading();
                        let us_loading = self.settings_state.users_state.load_state.loading();
                        let ws = if !ws_loading {
                            self.settings_state
                                .workspaces_state
                                .load_state
                                .start_loading();
                            self.settings_state.workspaces_state.refresh().map(|msg| {
                                Message::Settings(settings::SettingsMessage::WorkspaceMsg(msg))
                            })
                        } else {
                            Task::none()
                        };
                        let us = if !us_loading {
                            self.settings_state.users_state.load_state.start_loading();
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
                // GitState handles eager-refresh gating internally.
                let git_tasks = self.git_state.update_tick().map(Message::Git);

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
                    self.persist_window_state();
                }
                self.home_state.update(msg).map(Message::Home)
            }
            Message::Shell(msg) if self.ready => self.shell_state.update(msg).map(Message::Shell),
            Message::Logs(msg) if self.ready => self
                .logs_state
                .update(msg, self.log_store.as_ref().expect("ready"))
                .map(Message::Logs),
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
                    // Close branch modal synchronously if open.
                    // CloseModal always returns Task::none() so discarding is safe.
                    let _ = self.git_state.update(git::GitMessage::CloseModal);
                    let hash = commit_hash.clone();
                    let ws = workspace_name.clone();
                    return Task::batch([
                        close_board,
                        Task::done(Message::DiffModal(diff::DiffMessage::NavigateToCommit(
                            ws, hash,
                        ))),
                    ]);
                }
                self.board_state.update(msg).map(Message::Board)
            }
            Message::Sessions(msg) if self.ready => {
                self.sessions_state.update(msg).map(Message::Sessions)
            }
            // Intercept CloseModal from successful manual commit — auto-close
            // the diff modal while keeping the diff state in working-tree view.
            // ClearCommitState is intentionally not emitted; the commit handler
            // already cleared commit state and kicked off a diff refresh.
            Message::DiffModal(diff::DiffMessage::CloseModal) if self.ready => {
                self.show_diff_modal = false;
                Task::none()
            }
            Message::DiffModal(msg) if self.ready => {
                self.diff_state.update(msg).map(Message::DiffModal)
            }
            Message::Editor(msg) if self.ready => {
                self.editor_state.update(msg).map(Message::Editor)
            }
            Message::Settings(msg) if self.ready => {
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
                        self.persist_window_state();
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
            Message::Shutdown | Message::CloseRequested(_) => self.save_and_exit(),
            Message::CheckpointAndExit => iced::exit(),
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
                // Close branch modal synchronously if open.
                // CloseModal always returns Task::none() so discarding is safe.
                let _ = self.git_state.update(git::GitMessage::CloseModal);
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
            // ── Git state (routed to self.git_state) ─────────────────
            Message::Git(msg) if self.ready => {
                // Cross-modal close: if opening the branch modal,
                // close the diff modal from Dashboard side.
                if matches!(msg, git::GitMessage::OpenModal) {
                    self.show_diff_modal = false;
                }
                self.git_state.update(msg).map(Message::Git)
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
                } else if self.git_state.is_modal_open() {
                    self.git_state
                        .update(git::GitMessage::CloseModal)
                        .map(Message::Git)
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
                self.persist_window_state();
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
            Message::UpdateResult(result) if self.ready => {
                // execute_update() calls exit(0) on success, so we never
                // actually reach this branch for the Ok case. The window
                // closes as the only success signal to the user.
                if let Err(err) = result {
                    self.updating = false;
                    self.toasts
                        .push(Toast::from_toast_msg(&ToastMessage::Error(err)));
                }
                Task::none()
            }
            Message::TogglePause if self.ready => {
                let Some(ws_name) = self.active_workspace_name() else {
                    self.toasts.push(Toast::new(
                        "No workspace selected — select a workspace first".to_string(),
                        ToastKind::Warning,
                    ));
                    return Task::none();
                };
                let new_paused = !self.paused();
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
            Message::TogglePauseResult(result, ws_name, intended_state) if self.ready => self
                .handle_toggle_result(
                    result,
                    &ws_name,
                    intended_state,
                    "Pipeline paused",
                    "Pipeline resumed",
                    "Failed to toggle pipeline pause",
                ),
            Message::ToggleMaintenance if self.ready => {
                let Some(ws_name) = self.active_workspace_name() else {
                    self.toasts.push(Toast::new(
                        "No workspace selected — select a workspace first".to_string(),
                        ToastKind::Warning,
                    ));
                    return Task::none();
                };
                let new_enabled = !self.maintenance();
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
            Message::ToggleMaintenanceResult(result, ws_name, intended_state) if self.ready => self
                .handle_toggle_result(
                    result,
                    &ws_name,
                    intended_state,
                    "Maintainer enabled",
                    "Maintainer disabled",
                    "Failed to toggle maintainer",
                ),
            Message::WorkspaceStatesRefreshed(paused_map, maintenance_map) if self.ready => {
                self.workspace_paused = paused_map;
                self.workspace_maintenance = maintenance_map;
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
            | Message::Git(_) => Task::none(),
        }
    }

    /// Shared handler for toggle-pause / toggle-maintenance results.
    fn handle_toggle_result(
        &mut self,
        result: Result<(), String>,
        ws_name: &str,
        intended_state: bool,
        on_label: &str,
        off_label: &str,
        err_prefix: &str,
    ) -> Task<Message> {
        match result {
            Ok(()) => {
                let label = if intended_state { on_label } else { off_label };
                self.toasts.push(Toast::new(
                    format!("{label} for {ws_name}"),
                    ToastKind::Success,
                ));
                refresh_workspace_states_task()
            }
            Err(e) => {
                self.toasts
                    .push(Toast::new(format!("{err_prefix}: {e}"), ToastKind::Error));
                Task::none()
            }
        }
    }

    /// Persist the workspace selection (sidebar state, window-state.json,
    /// and all page broadcasts). This is the canonical entry point for
    /// workspace switching throughout the dashboard.
    ///
    /// An empty name selects the "Personal" workspace (no shared workspace).
    fn select_workspace(&mut self, name: &str) -> Task<Message> {
        // Git state is cleared and eagerly refreshed below via
        // propagate_workspace_selection → set_workspace_path.
        self.selected_workspace_name = if name.is_empty() {
            None
        } else {
            Some(name.to_string())
        };
        self.persist_window_state();
        self.propagate_workspace_selection(name)
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
            resolve_dashboard_workspace_path(name, ws_path.as_deref(), personal_path.as_deref());
        let editor_task: Task<Message> = Task::done(editor::EditorMessage::WorkspaceSelected(
            editor_name,
            editor_path,
        ))
        .map(Message::Editor);

        let diff_name = name.to_string();
        let diff_path = personal_path.clone();

        // Propagate workspace path to git state, triggering eager refresh.
        // GitState owns the single source of truth for this path.
        let resolved_path = ws_path.clone().or_else(|| personal_path.clone());
        let git_task: Task<Message> = self
            .git_state
            .set_workspace_path(resolved_path)
            .map(Message::Git);

        let diff_task: Task<Message> =
            Task::done(diff::DiffMessage::WorkspaceSelected(diff_name, diff_path))
                .map(Message::DiffModal);

        let (shell_name, shell_path) =
            resolve_dashboard_workspace_path(name, ws_path.as_deref(), personal_path.as_deref());
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
            git_task,
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

    #[allow(clippy::too_many_lines)]
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
                let sidebar = ticket_sidebar(&self.board_state);
                // Wrap chat area in a right-click context menu with "Clear chat" option.
                let home_view: Element<'_, Message> = ContextMenu::new(
                    home_view,
                    vec![(
                        "Clear chat".into(),
                        Message::Home(home::HomeMessage::ClearChat),
                    )],
                )
                .into();
                // Wrap sidebar in a right-click context menu with "Archive done & cancelled" option.
                let sidebar: Element<'_, Message> = ContextMenu::new(
                    sidebar,
                    vec![(
                        "Archive done & cancelled".into(),
                        Message::Board(board::BoardMessage::ArchiveAllCompleted),
                    )],
                )
                .into();
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
        let branch_overlay: Element<'_, Message> = if self.git_state.is_modal_open() {
            let inner = self.git_state.view().map(Message::Git);
            modal_overlay(inner, Message::Git(git::GitMessage::CloseModal))
        } else {
            container(text(""))
                .width(Length::Shrink)
                .height(Length::Shrink)
                .into()
        };

        iced::widget::stack![body, diff_overlay, branch_overlay, overlay].into()
    }
}

/// Wrap dialog content in a modal overlay with a semi-transparent backdrop
/// and centered 80%-width dialog container.
///
/// Creates a backdrop that dismisses the modal on click, wraps `inner` in the
/// standard dialog container style with 16px padding, and centers it at 80%
/// width using a `FillPortion(1/8/1)` row layout.
fn modal_overlay<'a>(
    inner: impl Into<Element<'a, Message>>,
    on_close: Message,
) -> Element<'a, Message> {
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
    .on_press(on_close);

    let dialog = container(inner)
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(16)
        .style(theme::dialog_container_style);

    let centered = row![
        Space::new().width(Length::FillPortion(1)), // 10% margin
        dialog.width(Length::FillPortion(8)),       // 80% content
        Space::new().width(Length::FillPortion(1)), // 10% margin
    ]
    .width(Length::Fill)
    .height(Length::Fill);

    iced::widget::stack([backdrop.into(), centered.into()]).into()
}

/// Render the diff modal (80% width, 100% height, centered).
fn render_diff_modal(diff_state: &diff::DiffState) -> Element<'_, Message> {
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

    modal_overlay(inner, Message::CloseDiffModal)
}

// ── Ticket sidebar (Home page, right side) ────────────────────────

/// Ticket sidebar shown on the right side of the Home page.
/// Displays all non-archived tickets grouped by status. A right-click
/// context menu on this panel offers "Archive done & cancelled".
fn ticket_sidebar(board_state: &board::BoardState) -> Element<'_, Message> {
    let (pending, pipeline, completed) = board::BoardState::partition_tickets(&board_state.tickets);

    // Split pipeline into "pinned" (actively working) and "ready"
    // (ReadyForDevelopment only). partition_tickets lumps them together,
    // but the sidebar separates them visually.
    let pinned: Vec<&Ticket> = pipeline
        .iter()
        .filter(|t| t.phase != TicketPhase::ReadyForDevelopment)
        .copied()
        .collect();
    let ready: Vec<&Ticket> = pipeline
        .iter()
        .filter(|t| t.phase == TicketPhase::ReadyForDevelopment)
        .copied()
        .collect();

    let is_empty = pending.is_empty() && pipeline.is_empty() && completed.is_empty();

    // Body: loading, empty, or ticket groups
    let body: Element<'_, Message> = if !board_state.load_state.has_loaded() {
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

    let content = column![Space::new().height(8), body].spacing(0);

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

/// Map a [`Page`] variant to its corresponding Lucide icon element.
///
/// Exhaustive match — adding a new `Page` variant produces a compile error
/// until its icon is assigned here.
fn page_icon(page: Page, size: u32, color: Color) -> Element<'static, Message> {
    let text = match page {
        Page::Home => lucide::layout_dashboard::<iced::Theme, iced::Renderer>(),
        Page::Editor => lucide::pencil_line::<iced::Theme, iced::Renderer>(),
        Page::Shell => lucide::terminal::<iced::Theme, iced::Renderer>(),
        Page::Sessions => lucide::scroll_text::<iced::Theme, iced::Renderer>(),
        Page::Logs => lucide::activity::<iced::Theme, iced::Renderer>(),
        Page::Settings => lucide::settings::<iced::Theme, iced::Renderer>(),
    };
    text.size(size).color(color).into()
}

impl Dashboard {
    fn sidebar_view(&self) -> Element<'_, Message> {
        container(
            column![
                self.render_sidebar_nav(),
                Space::new().height(Length::Fill),
                self.render_maintainer_toggle(),
                self.render_pause_toggle(),
            ]
            .spacing(2),
        )
        .width(Length::Fixed(56.0))
        .height(Length::Fill)
        .style(theme::surface_container_style)
        .padding(12)
        .into()
    }

    /// Sidebar navigation icons: Home, Editor, Shell (28px).
    ///
    /// Nav buttons use position::Position::Right to avoid clipping off the left
    /// edge of the 56px-wide sidebar container — Position::Top would overflow
    /// the narrow column. The adjacent toggle buttons below use Position::Top
    /// because they have more vertical room before reaching the sidebar top edge.
    fn render_sidebar_nav(&self) -> Element<'_, Message> {
        let mut col = Column::new().spacing(2);
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
            let icon: iced::Element<'_, Message> = page_icon(*page, 28, color);
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
            let tooltip_text = if disabled {
                format!("Select a workspace to access {}", page.label())
            } else {
                page.label().to_string()
            };
            let nav_btn = tooltip(btn, text(tooltip_text).size(11), tooltip::Position::Right)
                .style(theme::tooltip_style);
            col = col.push(nav_btn);
        }
        col.into()
    }

    /// Per-workspace Maintainer toggle button.
    /// Disabled when no workspace is selected (Personal mode).
    fn render_maintainer_toggle(&self) -> Element<'_, Message> {
        let has_ws = self.has_active_workspace();
        let maint_icon = column![
            text("Maint").size(8).color(theme::TEXT_MUTED),
            text(if self.maintenance() { "ON" } else { "OFF" })
                .size(9)
                .color(if self.maintenance() {
                    theme::ACCENT
                } else {
                    theme::TEXT_MUTED
                }),
        ]
        .spacing(0)
        .align_x(Alignment::Center);
        tooltip(
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
            } else if self.maintenance() {
                "Maintainer ON"
            } else {
                "Maintainer OFF"
            })
            .size(11),
            tooltip::Position::Top,
        )
        .style(theme::tooltip_style)
        .into()
    }

    /// Per-workspace pipeline pause/unpause toggle button.
    /// Disabled when no workspace is selected (Personal mode).
    fn render_pause_toggle(&self) -> Element<'_, Message> {
        let has_ws = self.has_active_workspace();
        let pause_icon = if !has_ws {
            lucide::pause::<iced::Theme, iced::Renderer>()
                .size(28)
                .color(theme::TEXT_FAINT)
        } else if self.paused() {
            lucide::play::<iced::Theme, iced::Renderer>()
                .size(28)
                .color(theme::ACCENT)
        } else {
            lucide::pause::<iced::Theme, iced::Renderer>()
                .size(28)
                .color(theme::TEXT_MUTED)
        };
        tooltip(
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
            } else if self.paused() {
                "Resume pipeline"
            } else {
                "Pause pipeline"
            })
            .size(11),
            tooltip::Position::Top,
        )
        .style(theme::tooltip_style)
        .into()
    }

    /// Render the self-update button in the footer bar.
    /// Returns `None` when self-update is not available on this installation.
    fn render_update_button(&self) -> Option<Element<'_, Message>> {
        if !self.update_available {
            return None;
        }
        let update_color = if self.updating {
            theme::TEXT_FAINT
        } else {
            theme::ACCENT
        };
        let update_icon = lucide::refresh_cw::<iced::Theme, iced::Renderer>()
            .size(24)
            .color(update_color);
        let update_btn = button(update_icon)
            .style(theme::button_text)
            .padding(3)
            .on_press_maybe(if self.updating {
                None
            } else {
                Some(Message::UpdateBot)
            });
        let update_tooltip = if self.updating {
            "Updating…"
        } else {
            "Update MahBot"
        };
        Some(
            tooltip(
                update_btn,
                text(update_tooltip).size(11),
                tooltip::Position::Top,
            )
            .style(theme::tooltip_style)
            .into(),
        )
    }

    /// Render the footer navigation icons (Sessions, Logs, Settings).
    fn render_nav_icons(&self) -> Element<'_, Message> {
        let mut icons: Vec<Element<'_, Message>> = Vec::with_capacity(3);
        for page in Page::footer_pages() {
            let is_active = self.page == *page;
            let color = if is_active {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            };
            let icon: Element<'_, Message> = page_icon(*page, 24, color);
            let btn = button(icon)
                .style(theme::button_text)
                .padding(3)
                .on_press(Message::Navigation(*page));
            icons.push(
                tooltip(btn, text(page.label()).size(11), tooltip::Position::Top)
                    .style(theme::tooltip_style)
                    .into(),
            );
        }
        Row::with_children(icons)
            .spacing(6)
            .align_y(Alignment::Center)
            .into()
    }

    /// Vertical divider between nav icons and git blocks.
    fn render_git_divider() -> Element<'static, Message> {
        rule::vertical(1)
            .style(|_: &iced::Theme| rule::Style {
                color: theme::TEXT_MUTED,
                radius: 0.0.into(),
                fill_mode: rule::FillMode::Padded(8),
                snap: true,
            })
            .into()
    }

    /// Render the current git branch button (clickable -> branch modal).
    /// Returns `None` when no branch is known.
    fn render_git_branch(&self) -> Option<Element<'_, Message>> {
        let b = self.git_state.current_branch()?;
        let truncated = if b.len() > 20 {
            let mut end = 19;
            while !b.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…", &b[..end])
        } else {
            b.to_string()
        };
        let branch_content = row![
            lucide::git_branch::<iced::Theme, iced::Renderer>()
                .size(24)
                .color(theme::ACCENT),
            text(truncated).size(16).color(theme::ACCENT),
        ]
        .spacing(6)
        .align_y(Alignment::Center);
        Some(
            button(branch_content)
                .style(theme::button_text)
                .padding(3)
                .on_press(Message::Git(git::GitMessage::OpenModal))
                .into(),
        )
    }

    /// Render the git sync indicator (refresh icon + behind/ahead counts, clickable).
    /// Uses lucide arrow_up/arrow_down at 16px (same as number text) for
    /// consistent vertical alignment with the 24px refresh icon.
    /// Returns `None` when there are no behind/ahead counts or both are zero.
    fn render_git_sync(&self) -> Option<Element<'_, Message>> {
        let (behind, ahead) = self.git_state.behind_ahead()?;
        if behind == 0 && ahead == 0 {
            return None;
        }
        // Build arrow+number text using lucide icons (not Unicode arrows)
        // so all elements share the same vertical baseline.
        let sync_text_label: Element<'_, Message> = {
            let mut parts: Vec<Element<'_, Message>> = Vec::new();
            if ahead > 0 {
                parts.push(
                    lucide::arrow_up::<iced::Theme, iced::Renderer>()
                        .size(16)
                        .color(theme::TEXT_MUTED)
                        .into(),
                );
                parts.push(
                    text(format!("{ahead}"))
                        .size(16)
                        .color(theme::TEXT_MUTED)
                        .into(),
                );
            }
            if behind > 0 {
                if ahead > 0 {
                    parts.push(Space::new().width(8).into());
                }
                parts.push(
                    lucide::arrow_down::<iced::Theme, iced::Renderer>()
                        .size(16)
                        .color(theme::TEXT_MUTED)
                        .into(),
                );
                parts.push(
                    text(format!("{behind}"))
                        .size(16)
                        .color(theme::TEXT_MUTED)
                        .into(),
                );
            }
            Row::with_children(parts)
                .spacing(2)
                .align_y(Alignment::Center)
                .into()
        };
        let sync_icon_color = if self.git_state.is_syncing() {
            theme::TEXT_MUTED
        } else {
            theme::ACCENT
        };
        let sync_content = row![
            lucide::refresh_cw::<iced::Theme, iced::Renderer>()
                .size(24)
                .color(sync_icon_color),
            sync_text_label,
        ]
        .spacing(6)
        .align_y(Alignment::Center);
        Some(
            button(sync_content)
                .style(theme::button_text)
                .padding(3)
                .on_press_maybe(if self.git_state.is_syncing() {
                    None
                } else {
                    Some(Message::Git(git::GitMessage::Sync))
                })
                .into(),
        )
    }

    /// Render the git diff stats button (+X/−Y, clickable -> diff modal).
    /// Returns `None` when there are no non-zero changes.
    fn render_git_diff_stats(&self) -> Option<Element<'_, Message>> {
        let (added, removed) = self.git_state.diff_stats()?;
        if added == 0 && removed == 0 {
            return None;
        }
        let stats_row = widgets::diff_stats_row::<Message>(added, removed, 15.0);
        Some(
            button(stats_row)
                .style(theme::button_text)
                .padding(3)
                .on_press(Message::OpenDiffModal(None))
                .into(),
        )
    }

    /// Render the git block: divider, branch, sync, and diff stats.
    /// Returns `None` when the workspace has no filesystem path.
    fn render_git_block(&self) -> Option<Element<'_, Message>> {
        if !self.git_state.has_filesystem_path() {
            return None;
        }
        let mut elements: Vec<Element<'_, Message>> = Vec::with_capacity(4);
        elements.push(Self::render_git_divider());
        if let Some(el) = self.render_git_branch() {
            elements.push(el);
        }
        if let Some(el) = self.render_git_sync() {
            elements.push(el);
        }
        if let Some(el) = self.render_git_diff_stats() {
            elements.push(el);
        }
        Some(
            Row::with_children(elements)
                .spacing(6)
                .align_y(Alignment::Center)
                .into(),
        )
    }

    /// Render the active agent icons in the right side of the footer.
    fn render_active_agents() -> Element<'static, Message> {
        let handles = crate::registry::AGENT_REGISTRY.list();
        let mut role_counts: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        for h in &handles {
            *role_counts.entry(h.role.as_str()).or_insert(0) += 1;
        }
        if role_counts.is_empty() {
            return text("").into();
        }
        let mut icons: Vec<Element<'_, Message>> = Vec::new();
        for (role_str, count) in &role_counts {
            let role: crate::Role = role_str.parse().unwrap_or(crate::Role::Engineer);
            let (color, _bg) = theme::role_badge_color_for(&role);
            let icon = theme::role_icon(&role).size(24).color(color);
            if *count > 1 {
                let label = text(format!("×{count}")).size(15).color(color);
                icons.push(
                    container(row![icon, label].spacing(3).align_y(Alignment::Center))
                        .padding(iced::Padding {
                            left: 3.0,
                            right: 3.0,
                            top: 0.0,
                            bottom: 0.0,
                        })
                        .into(),
                );
            } else {
                icons.push(
                    container(icon)
                        .padding(iced::Padding {
                            left: 3.0,
                            right: 3.0,
                            top: 0.0,
                            bottom: 0.0,
                        })
                        .into(),
                );
            }
        }
        Row::with_children(icons)
            .spacing(12)
            .align_y(Alignment::Center)
            .into()
    }

    /// 42px footer bar — nav items (left) and active agents (right).
    fn footer_view(&self) -> Element<'_, Message> {
        let mut left_elements: Vec<Element<'_, Message>> = Vec::with_capacity(3);

        if let Some(el) = self.render_update_button() {
            left_elements.push(el);
        }

        left_elements.push(self.render_nav_icons());

        if let Some(el) = self.render_git_block() {
            left_elements.push(el);
        }

        let left = Row::with_children(left_elements)
            .spacing(6)
            .align_y(Alignment::Center);

        let right = Self::render_active_agents();

        let footer_row = row![left, Space::new().width(Length::Fill), right]
            .align_y(Alignment::Center)
            .padding(iced::Padding {
                top: 3.0,
                right: 18.0,
                bottom: 3.0,
                left: 18.0,
            });

        container(footer_row)
            .align_y(Alignment::Center)
            .height(Length::Fixed(42.0))
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

/// Load workspace path and state maps from the workspace store, resolving
/// `prev_selection` against the loaded list. Falls back to an empty-string
/// "Personal" default when `prev_selection` is absent or stale.
/// Returns a `BootWorkspaces` message ready for use with `Task::perform`.
async fn load_workspace_options(prev_selection: Option<String>) -> Message {
    let store = crate::workspace::store();
    let mut paths = HashMap::new();
    let mut paused_map = HashMap::new();
    let mut maintenance_map = HashMap::new();
    let mut restored_name = None;

    if let Ok(ws_list) = store.list().await {
        for ws in &ws_list {
            paths.insert(ws.name.clone(), ws.path.clone());
            paused_map.insert(ws.name.clone(), ws.paused);
            maintenance_map.insert(ws.name.clone(), ws.maintenance);
        }
    }

    if let Some(ref name) = prev_selection {
        // Empty string means "Personal" — always valid.
        if name.is_empty() || paths.contains_key(name.as_str()) {
            restored_name = Some(name.clone());
        }
    }
    if restored_name.is_none() {
        restored_name = Some(String::new());
    }

    Message::BootWorkspaces(paths, paused_map, maintenance_map, restored_name)
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

/// Resolve a workspace name+path pair from the dashboard's in-memory workspace
/// map and optional personal workspace path.  This is a synchronous lookup —
/// it does **not** query the database.  For a DB-backed resolution (async),
/// see `gui::diff::resolve_workspace_path`.
///
/// If `ws_path` is `Some`, that takes priority; otherwise `personal_path` is
/// used as a fallback ("Personal workspace — use resolved user path").  When
/// `name` is empty and no path is available, returns `None` for the path
/// ("Personal workspace without a selected user — no path to send").  Logs a
/// warning for non-empty names where neither path source is available (possible
/// DB inconsistency).
fn resolve_dashboard_workspace_path(
    name: &str,
    ws_path: Option<&str>,
    personal_path: Option<&str>,
) -> (String, Option<String>) {
    if let Some(p) = ws_path {
        (name.to_string(), Some(p.to_string()))
    } else if let Some(p) = personal_path {
        (name.to_string(), Some(p.to_string()))
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
