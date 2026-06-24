//! Native Iced dashboard — application entry point, navigation, and shared state.
//!
//! Iced owns the process Tokio runtime (`iced` feature `tokio`). MahBot
//! bootstraps via a startup [`iced::Task`] before the UI becomes interactive.

#![allow(
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::too_many_lines,
    clippy::struct_excessive_bools,
    clippy::trivially_copy_pass_by_ref,
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
    Diff,
    Shell,
    Editor,
    Settings,
}

impl Page {
    /// Pages shown in the sidebar (Home, Editor, Diff, Shell).
    const fn sidebar_pages() -> &'static [Page] {
        &[Page::Home, Page::Editor, Page::Diff, Page::Shell]
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
            Page::Diff => "Diff",
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
    /// On error, local state is reverted and an error toast is shown.
    TogglePauseResult(Result<(), String>, String, bool),
    /// Workspace options loaded during boot (options, paths, paused, restored selection).
    BootWorkspaces(
        Vec<PickOption>,
        HashMap<String, String>,
        HashMap<String, bool>,
        Option<String>,
    ),
    Home(home::HomeMessage),
    Logs(logs::LogMessage),
    Board(board::BoardMessage),
    Sessions(sessions::SessionsMessage),
    Diff(diff::DiffMessage),
    Shell(shell::ShellMessage),
    Editor(editor::EditorMessage),
    Settings(settings::SettingsMessage),
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

    logs_state: logs::LogsState,
    board_state: board::BoardState,
    sessions_state: sessions::SessionsState,
    diff_state: diff::DiffState,
    home_state: home::HomeState,
    shell_state: shell::ShellState,
    editor_state: editor::EditorState,
    settings_state: settings::SettingsState,
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
            selected_workspace_name: None,
            selected_user_name: None,
            updating: false,
            update_available,
            paused: false,
            logs_state: logs::LogsState::new(),
            board_state: board::BoardState::new(),
            sessions_state: sessions::SessionsState::new(),
            diff_state: diff::DiffState::new(),
            home_state: home::HomeState::new(),
            shell_state: shell::ShellState::new(),
            editor_state: editor::EditorState::new(),
            settings_state: settings::SettingsState::new(),
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
            Message::BootWorkspaces(options, paths, paused_map, restored_name) => {
                self.workspace_options.clone_from(&options);
                self.workspace_paths = paths;
                // Clone the paused map before moving it into self so the closure
                // below can reference it without borrowing from self.
                let paused_map_for_closure = paused_map.clone();
                self.workspace_paused = paused_map;
                // Derive paused state from the selected workspace.
                let update_paused = |dash: &mut Self, ws_name: Option<&str>| {
                    dash.paused = ws_name
                        .and_then(|n| paused_map_for_closure.get(n))
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
                if let Some(ref name) = restored_name {
                    if name.is_empty() {
                        // "Personal" workspace — no shared workspace selected.
                        self.selected_workspace_name = None;
                        update_paused(self, None);
                        return Task::batch([
                            self.propagate_workspace_selection(""),
                            home_opts,
                            load_users,
                        ]);
                    }
                    self.selected_workspace_name = Some(name.clone());
                    update_paused(self, Some(name));
                    return Task::batch([
                        self.propagate_workspace_selection(name),
                        home_opts,
                        load_users,
                    ]);
                }
                // Auto-select first workspace when nothing was restored
                // (belt-and-suspenders: load_workspace_options already does this,
                // but guard against any future call site that doesn't).
                if let Some(first) = options.first() {
                    if first.value.is_empty() {
                        self.selected_workspace_name = None;
                        update_paused(self, None);
                        return Task::batch([
                            self.propagate_workspace_selection(""),
                            home_opts,
                            load_users,
                        ]);
                    }
                    self.selected_workspace_name = Some(first.value.clone());
                    update_paused(self, Some(&first.value));
                    return Task::batch([
                        self.propagate_workspace_selection(&first.value),
                        home_opts,
                        load_users,
                    ]);
                }
                self.selected_workspace_name = None;
                update_paused(self, None);
                Task::batch([home_opts, load_users])
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
                    // Editor and Diff receive workspace state via WorkspaceSelected
                    // from the Home page picker, not via refresh().
                    Page::Diff => Task::none(),
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
                match self.page {
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
                    Page::Diff => Task::none(),
                    _ => Task::none(),
                }
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
                    self.page = Page::Diff;
                    return Task::done(diff::DiffMessage::NavigateToCommit(
                        workspace_name.clone(),
                        commit_hash.clone(),
                    ))
                    .map(Message::Diff);
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
            Message::Diff(msg) if self.ready => {
                if let diff::DiffMessage::Toast(ref tm) = msg {
                    self.toasts.push(Toast::from_toast_msg(tm));
                }
                self.diff_state.update(msg).map(Message::Diff)
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
            Message::EscapePressed => match self.page {
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
                Page::Diff => self
                    .diff_state
                    .update(diff::DiffMessage::Escape)
                    .map(Message::Diff),
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
            },
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
                let ws_name = match self.selected_workspace_name.clone() {
                    Some(n) if !n.is_empty() => n,
                    _ => {
                        self.toasts.push(Toast::new(
                            "No workspace selected — select a workspace first".to_string(),
                            ToastKind::Warning,
                        ));
                        return Task::none();
                    }
                };
                let new_paused = !self.paused;
                // Optimistic local update for responsive UI.
                self.paused = new_paused;
                self.workspace_paused.insert(ws_name.clone(), new_paused);
                let msg = if new_paused {
                    format!("Pipeline paused for {ws_name}")
                } else {
                    format!("Pipeline resumed for {ws_name}")
                };
                self.toasts.push(Toast::new(msg, ToastKind::Success));
                // Persist to DB — revert on failure.
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
                if let Err(e) = result {
                    // Revert local state on DB write failure to prevent drift.
                    let actual = !intended_state;
                    self.paused = actual;
                    self.workspace_paused.insert(ws_name, actual);
                    self.toasts.push(Toast::new(
                        format!("Failed to toggle pipeline pause: {e}"),
                        ToastKind::Error,
                    ));
                }
                Task::none()
            }
            Message::Home(_)
            | Message::Shell(_)
            | Message::Logs(_)
            | Message::Board(_)
            | Message::Sessions(_)
            | Message::Diff(_)
            | Message::Editor(_)
            | Message::Settings(_)
            | Message::UpdateBot
            | Message::UpdateResult(_)
            | Message::TogglePause
            | Message::TogglePauseResult(..) => Task::none(),
        }
    }

    /// Persist the workspace selection (sidebar state, window-state.json,
    /// and all page broadcasts). This is the canonical entry point for
    /// workspace switching throughout the dashboard.
    ///
    /// An empty name selects the "Personal" workspace (no shared workspace).
    fn select_workspace(&mut self, name: &str) -> Task<Message> {
        if name.is_empty() {
            self.selected_workspace_name = None;
            self.paused = false;
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
        let diff_task: Task<Message> =
            Task::done(diff::DiffMessage::WorkspaceSelected(diff_name, diff_path))
                .map(Message::Diff);

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

        Task::batch([board_refresh, editor_task, diff_task, shell_task, home_task])
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
            Page::Diff => self.diff_state.view().map(Message::Diff),
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

        iced::widget::stack![body, overlay].into()
    }
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
    );
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
    );
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
            .direction(scrollable::Direction::Vertical(theme::thin_scrollbar()))
            .style(theme::scrollbar_style)
            .into()
    };

    let content = column![header, Space::new().height(8), body].spacing(0);

    container(content)
        .padding([8, 12])
        .width(Length::Fill)
        .height(Length::Fill)
        .style(|_theme: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(theme::BG_SURFACE)),
            ..container::Style::default()
        })
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
                let is_cmd = modifiers.command();
                // On non-macOS, AltGr (Ctrl+Alt) is character input — block
                // shortcuts from firing.
                #[cfg(not(target_os = "macos"))]
                let altgr_active = modifiers.alt() && modifiers.control();
                #[cfg(target_os = "macos")]
                let altgr_active = false;

                let latin = key.to_latin(physical_key);
                // Cmd+F (macOS) / Ctrl+F (other) → focus search.
                if !altgr_active && is_cmd && !modifiers.shift() && latin == Some('f') {
                    return Some(Message::FocusSearch);
                }
                if let Key::Named(iced::keyboard::key::Named::Escape) = key {
                    Some(Message::EscapePressed)
                } else if is_cmd && !altgr_active {
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
            self.diff_state.subscription().map(Message::Diff),
            self.editor_state.subscription().map(Message::Editor),
            self.home_state.subscription().map(Message::Home),
            iced::Subscription::run(shutdown_subscription),
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
        // Sidebar navigation: Home, Editor, Diff, Shell (icon-only, 28px)
        let mut nav_col = Column::new().spacing(4);
        for page in Page::sidebar_pages() {
            let is_active = self.page == *page;
            // Editor, Diff, Shell require any workspace (shared or personal with a user selected).
            let has_any_workspace =
                self.selected_workspace_name.is_some() || self.selected_user_name.is_some();
            let requires_workspace = matches!(*page, Page::Editor | Page::Shell | Page::Diff);
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
                Page::Diff => lucide::git_pull_request::<iced::Theme, iced::Renderer>()
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

        // Spacer to push pause button to the bottom of the sidebar
        nav_col = nav_col.push(Space::new().height(Length::Fill));

        // Per-workspace pipeline pause/unpause toggle.
        // Disabled when no workspace is selected (Personal mode).
        let has_ws = self.selected_workspace_name.is_some()
            && self.selected_workspace_name.as_deref() != Some("");
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
        );
        nav_col = nav_col.push(pause_btn);

        let inner = nav_col.spacing(2);

        container(inner)
            .width(Length::Fixed(56.0))
            .height(Length::Fill)
            .style(move |_theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_SURFACE)),
                ..container::Style::default()
            })
            .padding(12)
            .into()
    }

    /// 24px footer bar — nav items (left) and active agents (right).
    fn footer_view(&self) -> Element<'_, Message> {
        // Left: footer navigation (Sessions, Logs, Settings)
        // Icon-only, 16px. Active page in ACCENT, inactive in TEXT_MUTED.
        let mut left_icons = Vec::with_capacity(6);
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
        // Update button — inline with navigation, disabled while updating.
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
            .style(move |_theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_SURFACE)),
                ..container::Style::default()
            })
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

/// Load workspace `PickOption` list and path map from the workspace store,
/// resolving `prev_selection` against the loaded list. Falls back to the
/// first available workspace when `prev_selection` is absent or stale.
/// Returns a `BootWorkspaces` message ready for use with `Task::perform`.
async fn load_workspace_options(prev_selection: Option<String>) -> Message {
    let store = crate::workspace::store();
    let mut options = Vec::new();
    let mut paths = HashMap::new();
    let mut paused_map = HashMap::new();
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

    Message::BootWorkspaces(options, paths, paused_map, restored_name)
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
/// returns two empty strings ("Personal workspace without a selected user —
/// no path to send").  Logs a warning for non-empty names where neither path
/// source is available (possible DB inconsistency).
fn resolve_workspace_path(
    name: &str,
    ws_path: Option<&String>,
    personal_path: Option<&String>,
) -> (String, String) {
    if let Some(p) = ws_path {
        (name.to_string(), p.clone())
    } else if let Some(p) = personal_path {
        (name.to_string(), p.clone())
    } else if name.is_empty() {
        (String::new(), String::new())
    } else {
        tracing::warn!(
            workspace = name,
            "Workspace path not found in map — sending empty selection"
        );
        (String::new(), String::new())
    }
}
