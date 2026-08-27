//! Native Iced dashboard — application entry point, navigation, and shared state.
//!
//! Iced owns the process Tokio runtime (`iced` feature `tokio`). MahBot
//! bootstraps via a startup [`iced::Task`] before the UI becomes interactive.

#![expect(
    clippy::struct_excessive_bools,
    clippy::if_not_else,
    clippy::collapsible_if
)]

pub(crate) mod board;
pub(crate) mod common;
pub(crate) mod diff;
pub(crate) mod diff_widget;
pub(crate) mod editor;
pub(crate) mod editor_widget;
pub(crate) mod git;
pub(crate) mod highlight;
pub(crate) mod home;
pub(crate) mod logs;
pub(crate) mod markdown_breaks;
pub(crate) mod media_markers;
pub(crate) mod menus;
pub(crate) mod running;
pub(crate) mod session_preview;
pub(crate) mod sessions;
pub(crate) mod settings;
pub(crate) mod shell;
pub(crate) mod text_rendering;
pub(crate) mod theme;
pub(crate) mod tool_failures;
pub(crate) mod users;
pub(crate) mod widgets;
pub(crate) mod workspaces;

use crate::logs::LogStore;
use crate::pipeline::board::Ticket;

use iced::keyboard;
use iced::widget::{
    Column, Row, Space, button, column, container, row, rule, scrollable, text, text_input, tooltip,
};
use iced::window;
use iced::{Alignment, Color, Element, Length, Task};

use crate::Role;
use crate::audio::voice::VoiceStatus;

use self::menus::ContextMenu;

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

/// Initialise the git file-change broadcast before the iced application runs
/// (matching the [`LOG_BROADCAST`] convention) so the file-change subscription
/// always has a source. Called from `main`.
pub fn init_git_file_change_tx() {
    git::init_file_change_tx();
}

/// Initialise the git-commit broadcast before the iced application runs
/// (matching the [`LOG_BROADCAST`] convention) so the pipeline-commit
/// subscription always has a source. Called from `main`.
pub fn init_git_commit_tx() {
    crate::git::commands::init_git_commit_tx();
}

/// Initialise the CDC ticket change broadcast before the iced application runs
/// (same convention as [`init_git_file_change_tx`] / [`LOG_BROADCAST`]) so the
/// board change subscription always has a source. An uninitialized `OnceLock`
/// would end the subscription stream on its first poll and freeze the board at
/// the initial snapshot — Iced does not re-spawn an ended subscription. Called
/// from `main` before `iced::application(...).run()`.
pub fn init_board_change_tx() {
    let _ = crate::db::cdc::ticket_sender();
}

// ── Navigation pages ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Home,
    Sessions,
    Logs,
    Shell,
    Editor,
    Settings,
    /// Live view of running agents and in-flight non-agent LLM work.
    RunningAgents,
}

impl Page {
    /// Pages shown in the sidebar (Home, Editor, Shell). Running Agents is
    /// NOT in the sidebar — it is reachable only via the footer activity
    /// indicators (role icons / zap counter). Because the Cmd+number
    /// shortcuts index this same list, Cmd+4 no longer navigates to it
    /// (user-approved).
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
            Page::RunningAgents => "Running Agents",
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

impl Toast {
    fn new(message: String, kind: ToastKind) -> Self {
        Self {
            message,
            kind,
            created_at: Instant::now(),
        }
    }

    fn from_toast_msg(msg: &ToastMessage) -> Self {
        match msg {
            ToastMessage::Saved => Toast::new("Saved".to_string(), ToastKind::Success),
            ToastMessage::Created => Toast::new("Created".to_string(), ToastKind::Success),
            ToastMessage::Deleted => Toast::new("Deleted".to_string(), ToastKind::Success),
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

/// Distinguishes between pause-toggle and maintenance-toggle toggles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToggleKind {
    /// Pause/unpause the pipeline.
    Pause,
    /// Enable/disable the maintainer.
    Maintenance,
}

impl ToggleKind {
    fn label_on(self) -> &'static str {
        match self {
            Self::Pause => "Pipeline paused",
            Self::Maintenance => "Maintainer enabled",
        }
    }

    fn label_off(self) -> &'static str {
        match self {
            Self::Pause => "Pipeline resumed",
            Self::Maintenance => "Maintainer disabled",
        }
    }

    fn label_err(self) -> &'static str {
        match self {
            Self::Pause => "Failed to toggle pipeline pause",
            Self::Maintenance => "Failed to toggle maintainer",
        }
    }

    /// Persist the new toggle state to the workspace store.
    async fn persist_to_store(self, name: String, state: bool) -> Result<(), String> {
        let store = crate::workspace::store();
        match self {
            Self::Pause => store
                .set_paused(&name, state)
                .await
                .map_err(|e| e.to_string()),
            Self::Maintenance => store
                .set_maintenance_enabled(&name, state)
                .await
                .map_err(|e| e.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
#[expect(private_interfaces)]
#[expect(clippy::large_enum_variant)]
pub enum Message {
    /// MahBot finished async startup (or failed). On success, [`BOOT_LOG_STORE`] is set.
    Boot(Result<(), String>),
    Navigation(Page),
    Tick,
    /// Shutdown signaled — close the dashboard window so `run()` returns.
    /// Triggered by the shutdown token (self-update restart, SIGTERM/SIGINT).
    Shutdown,
    /// The graceful-drain window began (first signal / window-close /
    /// self-update). Draining disables input; iced::exit is deferred until
    /// the drain completes (the token fires).
    DrainStarted,
    /// Window close button pressed — persist position and size before exiting.
    CloseRequested(window::Id),
    /// Window geometry event (move/resize) — tracks state for persist-on-close.
    WindowEvent(window::Id, window::Event),
    /// Keyboard shortcut: Cmd+F — focus the primary search input on the current page.
    FocusSearch,
    /// Keyboard shortcut: Escape — dismiss modal/panel/confirmation on the current page.
    EscapePressed,
    /// Update button pressed — open the self-update confirmation modal.
    UpdateBot,
    /// Self-update confirmed — start the background build/install.
    ConfirmUpdate,
    /// Self-update confirmation dismissed — nothing happens.
    CancelUpdate,
    /// Self-update result.
    UpdateResult(Result<String, String>),
    /// Toggle the selected workspace's pipeline pause or maintainer state.
    Toggle(ToggleKind),
    /// Result of a per-workspace toggle DB write. Carries (kind, result, workspace_name, intended_state).
    /// On success, workspace state is refreshed from DB; on error an error toast is shown.
    ToggleResult(ToggleKind, Result<(), String>, String, bool),
    /// Periodic refresh of workspace paused/maintenance state from DB.
    /// Carries (name, paused, maintenance_enabled) tuples to merge into
    /// the existing `workspaces` map without overwriting paths.
    WorkspaceStatesRefreshed(Vec<(String, bool, bool)>),
    /// No-op — produced by refresh helpers on transient DB errors to avoid
    /// sending empty state maps that would wipe cached toggle state.
    Nop,
    /// Workspace info and restored selection loaded during boot.
    BootWorkspaces(HashMap<String, WorkspaceInfo>, Option<String>),
    Home(home::HomeMessage),
    Logs(logs::LogMessage),
    Board(board::BoardMessage),
    Sessions(sessions::SessionsMessage),
    /// Running Agents page messages (manual research-run cancel).
    RunningAgents(running::RunningMessage),
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
    /// Set the selected user's active role + pool for the role switcher
    /// indicator. Fetched together (single task, single pool read) to
    /// avoid a double `user_roles` query.
    RoleAndPoolLoaded {
        role: Option<Role>,
        pool: Vec<Role>,
    },
    /// A role switch failed — refresh the role cache so the composer's
    /// optimistic role icon reverts to the persisted active role.
    RoleSwitchFailed(String),
    /// TTS model download progress event.
    TtsDownloadEvent(crate::audio::tts::TtsDownloadEvent),
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
            | Message::Board(board::BoardMessage::Toast(tm))
            | Message::DiffModal(diff::DiffMessage::Toast(tm))
            | Message::Git(git::GitMessage::Toast(tm))
            | Message::Editor(editor::EditorMessage::Toast(tm))
            | Message::Sessions(sessions::SessionsMessage::Toast(tm))
            | Message::Settings(
                settings::SettingsMessage::Toast(tm)
                | settings::SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::Toast(tm))
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
    pub(crate) fn is_nav_platform_mod(self) -> bool {
        if cfg!(target_os = "macos") {
            self.is_cmd
        } else {
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
    pub(crate) fn is_text_platform_mod(self) -> bool {
        if cfg!(target_os = "macos") {
            self.is_cmd && !self.ctrl_held
        } else {
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
    pub(crate) fn is_shortcut_platform_mod(self) -> bool {
        self.is_platform_mod && !self.is_emacs_ctrl && !self.altgr_active
    }
}

/// Compute [`KeyboardMods`] from an Iced [`keyboard::Modifiers`] value.
///
/// Encapsulates the `cfg!()`-based platform branching that every keyboard
/// subscription handler previously inlined.
pub(crate) fn detect_keyboard_mods(modifiers: keyboard::Modifiers) -> KeyboardMods {
    let is_cmd = modifiers.command();
    let is_platform_mod = modifiers.command() || modifiers.control();
    let ctrl_held = modifiers.control();

    let is_emacs_ctrl = cfg!(target_os = "macos") && modifiers.control() && !modifiers.command();
    let altgr_active = !cfg!(target_os = "macos") && modifiers.alt() && modifiers.control();

    KeyboardMods {
        is_cmd,
        is_platform_mod,
        ctrl_held,
        is_emacs_ctrl,
        altgr_active,
    }
}

/// Extract the key, modifiers, and physical key from a `KeyPressed` event.
#[must_use]
pub(crate) fn parse_key_press(
    event: keyboard::Event,
) -> Option<(keyboard::Key, keyboard::Modifiers, keyboard::key::Physical)> {
    let keyboard::Event::KeyPressed {
        key,
        modifiers,
        physical_key,
        ..
    } = event
    else {
        return None;
    };
    Some((key, modifiers, physical_key))
}

// ── Dashboard state ──────────────────────────────────────────────

/// Log store created during boot; read when handling [`Message::Boot`].
pub static BOOT_LOG_STORE: OnceLock<LogStore> = OnceLock::new();

/// Per-workspace metadata held in memory for fast sidebar lookup.
/// Populated during boot from the DB and updated periodically via
/// [`Message::WorkspaceStatesRefreshed`] (which only refreshes booleans).
/// The struct stays `pub(crate)` because the Running Agents page takes the
/// map for registered-workspace checks (the "(external)" marker).
#[derive(Debug, Clone)]
pub(crate) struct WorkspaceInfo {
    path: String,
    paused: bool,
    maintenance_enabled: bool,
}

#[cfg(test)]
impl WorkspaceInfo {
    /// Test-only constructor — the Running Agents page tests build a
    /// registered-workspace map without sidebar metadata.
    pub(crate) fn test_new(path: String, paused: bool, maintenance_enabled: bool) -> Self {
        Self {
            path,
            paused,
            maintenance_enabled,
        }
    }
}

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

    /// Per-workspace info (path, paused, maintenance_enabled), keyed by name.
    workspaces: HashMap<String, WorkspaceInfo>,
    /// Currently selected workspace name from the global picker.
    selected_workspace_name: Option<String>,
    /// Currently selected user name (for impersonation). Persisted in window state.
    selected_user_name: Option<String>,
    /// Cached selected role for the current user — the composer role dropdown
    /// uses it to visually indicate which role is active.
    selected_user_role: Option<Role>,
    /// Cached role pool for the current user — the switchable roles shown in
    /// the composer role dropdown.
    selected_user_roles: Vec<Role>,
    /// True when a genuine window close was requested while the update was in
    /// its finalizing window (daemon shut down; checkpoint + spawn + exit
    /// pending). Only a user close (`CloseRequested`) sets this — the update's
    /// own step-10 `shutdown()` also fires `Message::Shutdown` via the
    /// subscription, which must not be mistaken for an exit request. If the
    /// update fails, this flag makes the GUI run its own checkpoint + exit.
    exit_requested_during_update: bool,
    /// Graceful-drain window active: the window stays open with input
    /// disabled until the drain completes.
    draining: bool,
    /// Whether the self-update confirmation modal is open. Set by
    /// [`Message::UpdateBot`]; the build/install itself only starts on
    /// [`Message::ConfirmUpdate`], so dismissing leaves the system untouched.
    show_update_confirm: bool,
    /// Research run awaiting manual-cancel confirmation — the run's durable
    /// job id. `Some` while the confirm dialog is shown on the Running Agents
    /// page; the cancel runs only on [`running::RunningMessage::CancelConfirmed`].
    /// The id is never displayed as text — the button message carries it.
    pending_research_cancel: Option<String>,
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

    // ── TTS download progress ───────────────────────────────────
    /// Current TTS download progress: (file_name, progress 0.0–1.0).
    /// `None` when no download is active.
    tts_download_progress: Option<(String, f32)>,
}

impl Dashboard {
    #[must_use]
    pub fn loading() -> Self {
        Self {
            ready: false,
            boot_error: None,
            page: Page::Home,
            log_store: None,
            last_size: iced::Size::new(1500.0, 800.0),
            last_position: iced::Point::new(-1.0, -1.0),
            toasts: Vec::new(),
            workspaces: HashMap::new(),
            selected_workspace_name: None,
            selected_user_name: None,
            selected_user_role: None,
            selected_user_roles: Vec::new(),
            exit_requested_during_update: false,
            draining: false,
            show_update_confirm: false,
            pending_research_cancel: None,
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
            tts_download_progress: None,
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
                self.board_state.current_user_name = self.selected_user_name.clone();
                let boot_workspaces = Task::perform(
                    load_workspace_options(prev.selected_workspace),
                    std::convert::identity,
                );

                // Start on Settings while no LLM provider is configured. The
                // decision must be made here, after the config is fully
                // loaded (bootstrap_mahbot finished), not in the initial
                // loading state where CONFIG still holds defaults. A trimmed
                // non-empty key counts as set — `provider_key()` collapses
                // empty/whitespace to None, so those re-arm this startup. An
                // active custom endpoint counts as configured too
                // — a keyless custom endpoint user must not be parked on
                // Settings.
                let settings_start = if !crate::config::provider_configured() {
                    self.navigate_to(Page::Settings)
                } else {
                    Task::none()
                };
                Task::batch([
                    refresh_logs.map(Message::Logs),
                    refresh_board.map(Message::Board),
                    boot_workspaces,
                    settings_start,
                ])
            }
            Err(e) => {
                self.boot_error = Some(e);
                Task::none()
            }
        }
    }

    /// Returns the [`WorkspaceInfo`] for the currently selected workspace, if any.
    fn selected_workspace_info(&self) -> Option<&WorkspaceInfo> {
        self.selected_workspace_name
            .as_ref()
            .and_then(|name| self.workspaces.get(name))
    }

    /// Whether the selected workspace's pipeline is paused (a strict freeze:
    /// all in-flight work stops and no pipeline stage advances until resume).
    fn paused(&self) -> bool {
        self.selected_workspace_info().is_some_and(|w| w.paused)
    }

    /// Whether the selected workspace's maintainer is enabled.
    fn maintenance_enabled(&self) -> bool {
        self.selected_workspace_info()
            .is_some_and(|w| w.maintenance_enabled)
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
        if crate::self_update::update_in_progress() && crate::self_update::update_is_finalizing() {
            // The update path has shut down the daemon and owns the exit:
            // checkpoint (execute_update step 11), lock release, spawn, exit(0).
            // Exiting here would drop the iced runtime and abort that sequence
            // mid-checkpoint, leaving the daemon down without a replacement —
            // so wait instead. A close requested meanwhile is recorded in
            // Message::CloseRequested; if the update fails, UpdateResult
            // re-enters this path (in-progress cleared, flag cleared)
            // and honors the close.
            return Task::none();
        }
        // The checkpoint is deliberately NOT run here: it relocated to
        // shutdown_after_dashboard, which runs after the iced runtime drops —
        // genuinely single-writer (today's in-iced checkpoint ran while
        // background writers were still live).
        iced::exit()
    }

    /// Window title with page name.
    pub fn title(&self) -> String {
        let page_name = self.page.label();
        format!("MahBot — {page_name}")
    }

    /// Apply workspace configuration loaded during boot.
    fn apply_boot_workspaces(
        &mut self,
        workspaces: HashMap<String, WorkspaceInfo>,
        restored_name: &str,
    ) -> Task<Message> {
        self.workspaces = workspaces;
        // Pre-set Home's selected_user from persisted window state
        // so UsersLoaded doesn't auto-select the first user when
        // a previous user was saved.
        if let Some(ref user_name) = self.selected_user_name {
            self.home_state.selected_user = Some(user_name.clone());
            crate::audio::voice::set_active_user_name(user_name);
        }
        // Load the selected user's role and role pool for the role switcher.
        let role_cache = self.refresh_selected_user_role_cache();

        let load_users = self.home_state.load_users().map(Message::Home);

        // Empty string => "Personal" workspace (no shared workspace).
        let ws_name = if restored_name.is_empty() {
            self.selected_workspace_name = None;
            String::new()
        } else {
            self.selected_workspace_name = Some(restored_name.to_owned());
            restored_name.to_owned()
        };
        Task::batch([
            role_cache,
            self.propagate_workspace_selection(&ws_name),
            load_users,
        ])
    }

    /// Navigate to a page, refreshing page-specific state as needed.
    fn navigate_to(&mut self, page: Page) -> Task<Message> {
        self.page = page;
        // Notify sessions state when navigating to/from Sessions page
        // so the auto-refresh timer starts/stops accordingly.
        self.sessions_state.set_page_active(page == Page::Sessions);
        match page {
            // Logs and Shell maintain their own internal state; Editor
            // receives workspace state via WorkspaceSelected from the
            // Dashboard — none need a refresh on navigation. Running Agents
            // reads the live registries at render time.
            Page::Logs | Page::Shell | Page::Editor | Page::RunningAgents => Task::none(),
            Page::Home => {
                let load_users = self.home_state.load_users().map(Message::Home);
                let snap = iced::widget::operation::snap_to_end::<Message>(home::CHAT_SCROLL_ID);
                let board_refresh = self.board_state.refresh().map(Message::Board);
                Task::batch([load_users, snap, board_refresh])
            }
            Page::Sessions => sessions::SessionsState::refresh().map(Message::Sessions),
            Page::Settings => {
                self.settings_state.refresh();
                self.refresh_settings_lists(false)
            }
        }
    }

    /// Refresh Settings workspace/user lists, gating each list independently
    /// on its in-flight load. `gate_loading` is set by the per-second tick,
    /// since refresh() does not self-gate; navigation passes `false` to keep
    /// its original unconditional refresh (no load_state marking). Does not
    /// absorb the config snapshot (`settings_state.refresh()`).
    fn refresh_settings_lists(&mut self, gate_loading: bool) -> Task<Message> {
        let ws = if gate_loading && self.settings_state.workspaces_state.load_state.loading() {
            Task::none()
        } else {
            if gate_loading {
                self.settings_state
                    .workspaces_state
                    .load_state
                    .start_loading();
            }
            self.settings_state
                .workspaces_state
                .refresh()
                .map(|msg| Message::Settings(settings::SettingsMessage::WorkspaceMsg(msg)))
        };
        let us = if gate_loading && self.settings_state.users_state.load_state.loading() {
            Task::none()
        } else {
            if gate_loading {
                self.settings_state.users_state.load_state.start_loading();
            }
            self.settings_state
                .users_state
                .refresh()
                .map(|msg| Message::Settings(settings::SettingsMessage::UserMsg(msg)))
        };
        Task::batch([ws, us])
    }

    /// Toggle the selected workspace's pause or maintainer state.
    fn toggle_workspace_state(&mut self, kind: ToggleKind) -> Task<Message> {
        let current_state = match kind {
            ToggleKind::Pause => self.paused(),
            ToggleKind::Maintenance => self.maintenance_enabled(),
        };
        let Some(ws_name) = self.active_workspace_name() else {
            self.toasts.push(Toast::new(
                "No workspace selected — select a workspace first".to_string(),
                ToastKind::Warning,
            ));
            return Task::none();
        };
        let new_state = !current_state;
        let ws_name_clone = ws_name.clone();
        Task::perform(
            kind.persist_to_store(ws_name_clone, new_state),
            move |result| Message::ToggleResult(kind, result, ws_name, new_state),
        )
    }

    /// Handle the result of a workspace toggle write.
    fn finish_toggle(
        &mut self,
        kind: ToggleKind,
        result: Result<(), String>,
        ws_name: &str,
        intended_state: bool,
    ) -> Task<Message> {
        match result {
            Ok(()) => {
                let label = if intended_state {
                    kind.label_on()
                } else {
                    kind.label_off()
                };
                self.toasts.push(Toast::new(
                    format!("{label} for {ws_name}"),
                    ToastKind::Success,
                ));
                refresh_workspace_states_task()
            }
            Err(e) => {
                self.toasts.push(Toast::new(
                    format!("{}: {e}", kind.label_err()),
                    ToastKind::Error,
                ));
                Task::none()
            }
        }
    }

    /// Process a periodic tick: expire old toasts, refresh workspace state,
    /// poll the visible page, and update git state.
    fn process_tick(&mut self) -> Task<Message> {
        // Auto-dismiss expired toasts
        let now = Instant::now();
        self.toasts
            .retain(|t| now.duration_since(t.created_at) < t.duration());

        // Auto-refresh workspace paused/maintenance state every tick.
        // Only runs when a workspace is selected — the toggle result
        // handler already re-reads authoritative state after writes.
        let ws_refresh = if self.has_active_workspace() {
            refresh_workspace_states_task()
        } else {
            Task::none()
        };

        let page_task = match self.page {
            Page::Home => {
                // The board ticket list is driven by CDC change-stream deltas
                // (see `board_change_subscription`), not a full re-poll. The tick
                // only refreshes the open-ticket detail as a fallback/health
                // check (comments are not part of the board delta).
                let detail_refresh = self
                    .board_state
                    .refresh_selected_ticket()
                    .map(Message::Board);
                Task::batch([detail_refresh])
            }
            Page::Sessions if !self.sessions_state.load_state.loading() => {
                self.sessions_state.load_state.start_loading();
                sessions::SessionsState::refresh().map(Message::Sessions)
            }
            Page::Settings => self.refresh_settings_lists(true),
            // Running Agents reads the live registries at render time — the
            // 1-second tick re-render is its only refresh mechanism.
            _ => Task::none(),
        };

        // ── Git state refresh (periodic remote sync + throttle) ───
        // The every-second local refresh is event-driven (via the file-change
        // subscription); the tick only runs the throttle/periodic remote work.
        let git_tasks = self.git_state.update_tick().map(Message::Git);

        Task::batch([ws_refresh, page_task, git_tasks])
    }

    /// Handle Escape key: dismiss modals in priority order (self-update
    /// confirmation → diff modal → git branch modal → page-level escape
    /// dispatch).
    fn process_escape(&mut self) -> Task<Message> {
        // Modal close priority: self-update confirmation first, then diff
        // modal, then branch modal, then page-level escapes.
        if self.show_update_confirm {
            self.show_update_confirm = false;
            Task::none()
        } else if self.show_diff_modal {
            self.show_diff_modal = false;
            Task::done(Message::DiffModal(diff::DiffMessage::ClearCommitState))
        } else if self.git_state.is_modal_open() {
            self.git_state
                .update(git::GitMessage::CloseModal)
                .map(Message::Git)
        } else {
            match self.page {
                Page::Home => self
                    .board_state
                    .update(board::BoardMessage::Escape)
                    .map(Message::Board),
                // Shell and Running Agents have no escape handling.
                // Running Agents: Escape dismisses the pending research-cancel
                // confirmation (Keep) — never confirms it.
                Page::RunningAgents => {
                    self.pending_research_cancel = None;
                    Task::none()
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
                Page::Settings => self
                    .settings_state
                    .update(settings::SettingsMessage::Escape)
                    .map(Message::Settings),
            }
        }
    }

    /// Load the selected user's active role + role pool into the cached
    /// role-switcher state (composer role dropdown pool guard and 'current'
    /// marker). Called at boot, on user switch, and after Settings edits
    /// that touch the selected user's role or pool.
    fn refresh_selected_user_role_cache(&self) -> Task<Message> {
        let Some(ref user) = self.selected_user_name else {
            return Task::none();
        };
        let user = user.clone();
        // Single task, single pool read: resolve the role from the
        // already-fetched pool instead of re-querying `user_roles`.
        Task::perform(
            async move {
                let pool = crate::users::role_pool(&user).await;
                let role = crate::users::resolve_active_role_from_pool(&user, &pool).await;
                Message::RoleAndPoolLoaded { role, pool }
            },
            std::convert::identity,
        )
    }

    /// Process a Settings message, intercepting user-related messages for
    /// cross-page side effects (user switching, workspace switching,
    /// deletion recovery) and optionally reloading workspace options when
    /// workspaces are added or deleted.
    fn process_settings_message(&mut self, msg: settings::SettingsMessage) -> Task<Message> {
        // Capture cross-page side effects for intercepted UserMsg
        // variants. These batch alongside the delegation call below.
        let mut intercept_task: Option<Task<Message>> = None;

        if let settings::SettingsMessage::UserMsg(ref inner) = msg {
            match inner {
                users::UsersMessage::SwitchUser(user) => {
                    // SwitchUser is a documented no-op in UsersState,
                    // so unconditional delegation below is safe.
                    intercept_task = Some(
                        Task::done(home::HomeMessage::UserSelected(user.clone()))
                            .map(Message::Home),
                    );
                }
                users::UsersMessage::DeleteResult(Ok(()), deleted_user) => {
                    if self.selected_user_name.as_deref() == Some(deleted_user.as_str()) {
                        self.selected_user_name = Some("admin".to_string());
                        self.persist_window_state();
                        intercept_task = Some(
                            Task::done(home::HomeMessage::UserSelected("admin".to_string()))
                                .map(Message::Home),
                        );
                    }
                }
                users::UsersMessage::UpdateWorkspace(sender, ws)
                    if self.selected_user_name.as_deref() == Some(sender.as_str()) =>
                {
                    intercept_task = Some(self.select_workspace(ws));
                }
                // A pool edit or active-role change in Settings leaves the
                // Dashboard's cached role/pool stale (composer role dropdown
                // guard and 'current' marker) — refresh them for the selected
                // user. Workspace changes are handled by the UpdateWorkspace
                // arm above and need no role/pool refresh.
                users::UsersMessage::PoolEditResult(Ok(()))
                | users::UsersMessage::RoleUpdateResult(Ok(())) => {
                    intercept_task = Some(self.refresh_selected_user_role_cache());
                }
                _ => {}
            }
        }

        let needs_global_reload = matches!(
            msg,
            settings::SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::DeleteResult(
                Ok(())
            )) | settings::SettingsMessage::AddWorkspaceResult(Ok(_))
        );

        let settings_task = self.settings_state.update(msg).map(Message::Settings);

        // Stack-allocated batch: only Some tasks are included via
        // flatten. Avoids Vec heap allocation for the common
        // no-intercept no-reload path.
        let tasks = [
            intercept_task,
            Some(settings_task),
            needs_global_reload.then(|| self.reload_workspace_options()),
        ];

        Task::batch(tasks.into_iter().flatten())
    }

    /// Process a Running Agents page message — the manual research-run
    /// cancel flow: button → pending-confirmation → async cancel → toast.
    fn process_running_message(&mut self, msg: running::RunningMessage) -> Task<Message> {
        match msg {
            running::RunningMessage::CancelRequest(run_key) => {
                // Pending-confirmation only: the dialog renders over the
                // Running Agents page; the run key is never displayed.
                self.pending_research_cancel = Some(run_key);
                Task::none()
            }
            running::RunningMessage::CancelConfirmed => {
                let Some(run_key) = self.pending_research_cancel.take() else {
                    return Task::none();
                };
                // The cancel is async: fire the run's signal, stop the run's
                // agents, remove the durable rows, release the folder and
                // archive. Silent to the Manager/caller — only the GUI gets a
                // toast.
                Task::perform(
                    async move { crate::research_cancel::cancel_research_run(&run_key).await },
                    |result| {
                        Message::RunningAgents(running::RunningMessage::CancelFinished(result))
                    },
                )
            }
            running::RunningMessage::CancelDismissed => {
                self.pending_research_cancel = None;
                Task::none()
            }
            running::RunningMessage::CancelFinished(result) => {
                let toast = match result {
                    Ok(()) => ToastMessage::SuccessMsg(
                        "Research run cancelled — agents stopped, run removed permanently."
                            .to_string(),
                    ),
                    Err(e) => ToastMessage::Error(format!("research cancel failed: {e}")),
                };
                self.toasts.push(Toast::from_toast_msg(&toast));
                Task::none()
            }
        }
    }

    #[expect(clippy::too_many_lines)]
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
            // ── Pre-ready handlers (execute regardless of ready state) ──
            Message::Boot(result) => self.finish_boot(result),
            Message::BootWorkspaces(workspaces, restored_name) => {
                self.apply_boot_workspaces(workspaces, restored_name.as_deref().unwrap_or(""))
            }
            Message::CloseRequested(_) => {
                if crate::self_update::update_in_progress()
                    && crate::self_update::update_is_finalizing()
                {
                    // The update owns the exit (its own drain + checkpoint +
                    // swap + spawn): record the close request and wait — a
                    // force-cancel here would abort the update's full drain.
                    self.persist_window_state();
                    self.exit_requested_during_update = true;
                    Task::none()
                } else if crate::shutdown::is_draining() {
                    // Window-close during drain = second signal: force-cancel
                    // (fires the token → Message::Shutdown → exit path).
                    crate::shutdown::force_cancel();
                    Task::none()
                } else {
                    // First window-close begins the GRACEFUL drain: the
                    // window stays open with input disabled; the drain-watch
                    // fires the token when no in-flight work remains (or
                    // force-cancels at the cap). save_and_exit is deferred
                    // until Message::Shutdown.
                    crate::shutdown::drain_begin();
                    Task::none()
                }
            }
            Message::Shutdown => self.save_and_exit(),
            Message::DrainStarted => {
                // Restart toast: only when the update build/install has
                // finished and finalize is shutting the service down — a
                // plain window close or signal during a mid-build update
                // (in-flight but not yet finalizing) must not show a false
                // "restarting" toast. `!self.draining` is defensive
                // single-emission protection.
                if !self.draining
                    && crate::self_update::update_in_progress()
                    && crate::self_update::update_is_finalizing()
                {
                    self.toasts.push(Toast::new(
                        "Update complete — restarting…".to_string(),
                        ToastKind::Warning,
                    ));
                }
                self.draining = true;
                Task::none()
            }
            Message::WindowEvent(_, event) => match event {
                window::Event::Resized(new_size) => {
                    self.last_size = new_size;
                    Task::none()
                }
                window::Event::Moved(new_pos) => {
                    self.last_position = new_pos;
                    Task::none()
                }
                _ => Task::none(),
            },
            Message::CloseDiffModal => {
                self.show_diff_modal = false;
                Task::done(Message::DiffModal(diff::DiffMessage::ClearCommitState))
            }
            // Navigation has explicit before/after-ready handling
            Message::Navigation(_) if !self.ready => Task::none(),
            Message::Navigation(page) => self.navigate_to(page),

            // ── Everything else before boot → silent no-op ──
            _ if !self.ready => Task::none(),

            // ── Post-ready handlers (no per-arm guards needed) ──
            Message::Tick => self.process_tick(),
            Message::Home(msg) => {
                // Intercept RequestWorkspaceChange: the Home page detected
                // that the selected user's DB workspace differs from the
                // sidebar selection.  Perform a Dashboard-level workspace
                // switch so the sidebar, saved state, and all pages stay
                // consistent.
                if let home::HomeMessage::RequestWorkspaceChange(ref name) = msg {
                    return self.select_workspace(name);
                }
                // Intercept UserSelected: user changed (from picker, Users page
                // icon, or auto-selected at boot) — sync the Dashboard's
                // selected_user_name and persist to window state.
                if let home::HomeMessage::UserSelected(ref user) = msg {
                    self.selected_user_name = Some(user.clone());
                    self.board_state.current_user_name = Some(user.clone());
                    self.selected_user_role = None; // reset until loaded
                    self.selected_user_roles = Vec::new();
                    crate::audio::voice::set_active_user_name(user);
                    self.persist_window_state();
                    let role_cache = self.refresh_selected_user_role_cache();
                    return Task::batch([
                        role_cache,
                        self.home_state.update(msg.clone()).map(Message::Home),
                    ]);
                }
                // Intercept SwitchRole: user selected a new role from the
                // composer role dropdown — persist it and show a toast. The
                // dropdown is built from the cached pool, so the pool check
                // is synchronous against that same list.
                if let home::HomeMessage::SwitchRole(ref user_name, ref role) = msg {
                    if !self.selected_user_roles.contains(role) {
                        return Task::done(Message::Home(home::HomeMessage::Toast(
                            ToastMessage::Error(format!(
                                "Role '{}' is not in {}'s allowed roles.",
                                role.as_str(),
                                user_name
                            )),
                        )));
                    }
                    self.selected_user_role = Some(*role);
                    let user = user_name.clone();
                    let role = *role;
                    return Task::batch([
                        // Close the composer role dropdown (Home never sees
                        // SwitchRole — it is intercepted here).
                        Task::done(Message::Home(home::HomeMessage::RoleMenuClosed)),
                        Task::perform(
                            async move {
                                crate::users::switch_active_role(&user, role)
                                    .await
                                    .map_err(|e| e.to_string())?;
                                Ok(role)
                            },
                            |res| {
                                match res {
                                    Ok(role) => Message::Home(home::HomeMessage::Toast(
                                        ToastMessage::SuccessMsg(format!(
                                            "Switched to {} role",
                                            crate::agent::role::role_info(&role).display_label
                                        )),
                                    )),
                                    // On failure, refresh the role cache so the
                                    // optimistic composer icon reverts to the
                                    // persisted active role.
                                    Err(e) => Message::RoleSwitchFailed(e),
                                }
                            },
                        ),
                    ]);
                }
                self.home_state.update(msg).map(Message::Home)
            }
            Message::Shell(msg) => self.shell_state.update(msg).map(Message::Shell),
            Message::Logs(msg) => self
                .logs_state
                .update(msg, self.log_store.as_ref().expect("ready"))
                .map(Message::Logs),
            Message::Board(msg) => {
                // Intercept ViewCommitDiff for cross-page navigation
                // before it reaches board_state.update.
                if let board::BoardMessage::ViewCommitDiff {
                    ref commit_hash, ..
                } = msg
                {
                    return self.open_diff_modal(Some(commit_hash.clone()));
                }
                self.board_state.update(msg).map(Message::Board)
            }
            Message::Sessions(msg) => self.sessions_state.update(msg).map(Message::Sessions),
            Message::RunningAgents(msg) => self.process_running_message(msg),
            // Intercept CloseModal from successful manual commit — auto-close
            // the diff modal while keeping the diff state in working-tree view.
            // ClearCommitState is intentionally not emitted; the commit handler
            // already cleared commit state and kicked off a diff refresh.
            Message::DiffModal(diff::DiffMessage::CloseModal) => {
                self.show_diff_modal = false;
                Task::none()
            }
            Message::DiffModal(msg) => {
                // A manual commit is a ref-only change (HEAD + `.git/index`)
                // that the file watcher never reports, so refresh the git footer
                // promptly — otherwise diff_stats/behind_ahead stay stale until
                // the periodic timer.
                let commit_succeeded = matches!(&msg, diff::DiffMessage::CommitResult(Ok(_)));
                let diff_task = self.diff_state.update(msg).map(Message::DiffModal);
                if commit_succeeded {
                    Task::batch([
                        diff_task,
                        self.git_state
                            .update(git::GitMessage::RefreshAfterCommit)
                            .map(Message::Git),
                    ])
                } else {
                    diff_task
                }
            }
            Message::Editor(msg) => self.editor_state.update(msg).map(Message::Editor),
            Message::Settings(msg) => self.process_settings_message(msg),
            // ── Diff modal ────────────────────────────────────────
            Message::OpenDiffModal(commit_hash) => self.open_diff_modal(commit_hash),
            // ── Git state (routed to self.git_state) ─────────────────
            Message::Git(msg) => {
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
            Message::EscapePressed => self.process_escape(),
            Message::UpdateBot => {
                // Only opens the confirmation modal — the build/install
                // starts on `ConfirmUpdate`. Availability is read from the
                // shared cache, so a Telegram-initiated update disables the
                // button here too.
                let availability = crate::self_update::update_availability();
                if availability.available && !availability.in_progress {
                    self.show_update_confirm = true;
                }
                Task::none()
            }
            Message::ConfirmUpdate => {
                self.show_update_confirm = false;
                let availability = crate::self_update::update_availability();
                if !availability.available || availability.in_progress {
                    return Task::none();
                }
                // Save window state before update (synchronous).
                self.persist_window_state();
                // Transient confirmation that the build/install has kicked off —
                // condensed from the mode-specific Telegram update notifications
                // ("building from source…" / "installing from crates.io…").
                self.toasts.push(Toast::new(
                    "Update started — building/installing…".to_string(),
                    ToastKind::Success,
                ));
                Task::perform(
                    async {
                        match crate::self_update::execute_update().await {
                            Ok(()) => Ok("ok".to_string()),
                            Err(e) => {
                                // Report to the first admin (Telegram) as well as
                                // the GUI toast, preserving the prior behavior
                                // where the update path notified on failure.
                                // `execute_update` no longer notifies; each caller
                                // is the single failure reporter.
                                let msg = format!("❌ Update failed:\n{e:#}");
                                let target =
                                    crate::self_update::resolve_admin_telegram_target().await;
                                crate::self_update::notify_admin(&msg, target.as_deref()).await;
                                Err(msg)
                            }
                        }
                    },
                    Message::UpdateResult,
                )
            }
            Message::CancelUpdate => {
                self.show_update_confirm = false;
                Task::none()
            }
            Message::UpdateResult(result) => {
                // execute_update() calls exit(0) on success, so we never
                // actually reach this branch for the Ok case. The update
                // toasts (start + restart) are the success signals; the
                // window closing is the final confirmation. On failure
                // `execute_update` already cleared the shared in-progress
                // flag, so the exit guards below no longer wait.
                if let Err(err) = result {
                    self.toasts
                        .push(Toast::from_toast_msg(&ToastMessage::Error(err)));
                    if self.exit_requested_during_update {
                        // A genuine window close was requested while the update
                        // owned the exit; it failed, so run the normal exit
                        // checkpoint.
                        self.exit_requested_during_update = false;
                        return self.save_and_exit();
                    }
                }
                Task::none()
            }
            Message::Toggle(kind) => self.toggle_workspace_state(kind),
            Message::ToggleResult(kind, result, ws_name, intended_state) => {
                self.finish_toggle(kind, result, &ws_name, intended_state)
            }
            Message::WorkspaceStatesRefreshed(states) => {
                for (name, paused, maintenance_enabled) in states {
                    if let Some(info) = self.workspaces.get_mut(&name) {
                        info.paused = paused;
                        info.maintenance_enabled = maintenance_enabled;
                    } else {
                        // New workspace appeared since boot — create a placeholder
                        // entry; the next BootWorkspaces will fill the path.
                        self.workspaces.insert(
                            name,
                            WorkspaceInfo {
                                path: String::new(),
                                paused,
                                maintenance_enabled,
                            },
                        );
                    }
                }
                Task::none()
            }
            Message::Nop => Task::none(),
            Message::RoleAndPoolLoaded { role, pool } => {
                self.selected_user_role = role;
                self.selected_user_roles = pool;
                Task::none()
            }
            Message::RoleSwitchFailed(e) => {
                // Revert the optimistic composer icon and surface the error.
                Task::batch([
                    Task::done(Message::Home(home::HomeMessage::Toast(
                        ToastMessage::Error(e),
                    ))),
                    self.refresh_selected_user_role_cache(),
                ])
            }
            Message::TtsDownloadEvent(event) => self.handle_tts_download_event(event),
        }
    }

    /// Handle a TTS download progress event.
    #[expect(clippy::cast_precision_loss)]
    fn handle_tts_download_event(
        &mut self,
        event: crate::audio::tts::TtsDownloadEvent,
    ) -> Task<Message> {
        match event {
            crate::audio::tts::TtsDownloadEvent::FileStarted { name, .. } => {
                self.tts_download_progress = Some((name, 0.0));
                Task::none()
            }
            crate::audio::tts::TtsDownloadEvent::FileProgress {
                name,
                bytes_downloaded,
                total_bytes,
            } => {
                let progress = if total_bytes > 0 {
                    (bytes_downloaded as f32 / total_bytes as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                self.tts_download_progress = Some((name, progress));
                Task::none()
            }
            crate::audio::tts::TtsDownloadEvent::FileCompleted { name } => {
                // Mark the file as fully done; next FileStarted will show the next file.
                self.tts_download_progress = Some((name, 1.0));
                Task::none()
            }
            crate::audio::tts::TtsDownloadEvent::Complete
            | crate::audio::tts::TtsDownloadEvent::Failed { .. } => {
                self.tts_download_progress = None;
                Task::none()
            }
        }
    }

    /// Open the diff modal, closing any board or branch modal first.
    ///
    /// When `commit_hash` is `Some`, navigates to that commit; when `None`,
    /// navigates to the working tree (clearing any stale commit state).
    fn open_diff_modal(&mut self, commit_hash: Option<String>) -> Task<Message> {
        // Close any board modal
        let close_board = self
            .board_state
            .update(board::BoardMessage::CloseModal)
            .map(Message::Board);
        self.show_diff_modal = true;
        // Close branch modal synchronously if open.
        // CloseModal always returns Task::none() so discarding is safe.
        let _ = self.git_state.update(git::GitMessage::CloseModal);
        // `selected_workspace_name` is `None` in Personal workspace mode (no shared
        // workspace selected). An empty string is the established convention and is
        // safe here: `ws` is only consumed in the `Some(hash)` branch below — the
        // `None` branch (working-tree diff, triggered by the git stats button) does
        // not use it.
        let ws = self.selected_workspace_name.clone().unwrap_or_default();
        let diff_task = match commit_hash {
            Some(hash) => Task::done(Message::DiffModal(diff::DiffMessage::NavigateToCommit(
                ws, hash,
            ))),
            None => Task::done(Message::DiffModal(diff::DiffMessage::BackToWorkingTree)),
        };
        Task::batch([close_board, diff_task])
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
        let ws_path = self.workspaces.get(name).map(|w| w.path.clone());

        // Set board's workspace filter directly, then refresh.
        // Clear any active search when switching workspaces so stale results
        // from the previous workspace don't persist.
        self.board_state.workspace_name = Some(name.to_string());
        self.board_state.search_query.clear();
        self.board_state.search_results.clear();
        self.board_state.search_generation += 1;
        // Bump the board generation so a stale in-flight snapshot from the
        // previous workspace is dropped on arrival.
        self.board_state.board_generation += 1;
        // On switch the next snapshot REPLACES the board (the previous
        // workspace's tickets must not linger), and the removal-tracking set is
        // cleared so old-workspace removals are not applied to the new snapshot.
        self.board_state.replace_on_refresh = true;
        self.board_state.delta_removed_ids.clear();
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
        let (page_name, page_path) =
            resolve_dashboard_workspace_path(name, ws_path.as_deref(), personal_path.as_deref());
        let editor_task: Task<Message> = Task::done(editor::EditorMessage::WorkspaceSelected(
            page_name.clone(),
            page_path.clone(),
        ))
        .map(Message::Editor);

        let diff_name = name.to_string();
        let diff_path = personal_path.clone();

        // Propagate workspace path to git state, triggering eager refresh.
        // GitState owns the single source of truth for this path and the
        // workspace name (empty = Personal/unnamed → timer fallback).
        let resolved_path = ws_path.or_else(|| personal_path.clone());
        let ws_name = (!name.is_empty()).then(|| name.to_string());
        let git_task: Task<Message> = self
            .git_state
            .set_workspace_path(ws_name, resolved_path)
            .map(Message::Git);

        let diff_task: Task<Message> =
            Task::done(diff::DiffMessage::WorkspaceSelected(diff_name, diff_path))
                .map(Message::DiffModal);

        let shell_task: Task<Message> =
            Task::done(shell::ShellMessage::WorkspaceSelected(page_name, page_path))
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

    #[expect(clippy::too_many_lines)]
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
                let home_view = self
                    .home_state
                    .view(
                        self.selected_user_role,
                        &self.selected_user_roles,
                        self.draining,
                    )
                    .map(Message::Home);
                let sidebar = ticket_sidebar(&self.board_state);
                // Wrap chat area in a right-click context menu with
                // "Reset session". Per-bubble ContextMenus (Copy message,
                // see home.rs) capture bubble right-clicks; this outer
                // menu is the fallback for empty-space right-clicks.
                let home_view: Element<'_, Message> = ContextMenu::new(
                    home_view,
                    vec![menus::MenuItem::new(
                        "Reset session".into(),
                        Message::Home(home::HomeMessage::ClearChat),
                    )],
                )
                .into();
                // Wrap sidebar in a right-click context menu with "Archive done & cancelled" option.
                let sidebar: Element<'_, Message> = ContextMenu::new(
                    sidebar,
                    vec![menus::MenuItem::new(
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
            Page::RunningAgents => {
                running::view(&self.workspaces, self.pending_research_cancel.as_deref())
            }
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
            widgets::empty_stack_placeholder()
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
            widgets::empty_stack_placeholder()
        };

        // ── Branch management modal overlay ─────────────────────────
        let branch_overlay: Element<'_, Message> = if self.git_state.is_modal_open() {
            let inner = self.git_state.view().map(Message::Git);
            modal_overlay(inner, Message::Git(git::GitMessage::CloseModal))
        } else {
            widgets::empty_stack_placeholder()
        };

        // ── Self-update confirmation modal overlay ─────────────────
        let update_overlay: Element<'_, Message> = self.render_update_confirm();

        iced::widget::stack![body, diff_overlay, branch_overlay, update_overlay, overlay].into()
    }
}

/// Wrap dialog content in a modal overlay with a semi-transparent backdrop
/// and centered 80%-width dialog container.
///
/// Creates a backdrop that dismisses the modal on click, wraps `inner` in the
/// standard dialog container style with 16px padding, and centers it at 80%
/// width using a `FillPortion(1/8/1)` row layout, then delegates to the shared
/// backdrop helper for the overlay layering.
fn modal_overlay<'a>(
    inner: impl Into<Element<'a, Message>>,
    on_close: Message,
) -> Element<'a, Message> {
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

    widgets::modal_backdrop(centered, on_close, 0.5)
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
/// Displays all non-archived tickets grouped by phase. A right-click
/// context menu on this panel offers "Archive done & cancelled".
fn ticket_sidebar(board_state: &board::BoardState) -> Element<'_, Message> {
    let search_active = !board_state.search_query.is_empty();

    // ── Search input row ───────────────────────────────────────────
    let search_input = text_input("Search tickets…", &board_state.search_query)
        .on_input(|q| Message::Board(board::BoardMessage::SearchInputChanged(q)))
        .style(widgets::text_input_style)
        .size(13)
        .padding([4, 8]);
    let clear_btn = widgets::icon_tooltip_button(
        text("×").size(14),
        "clear search",
        Some(Message::Board(board::BoardMessage::SearchCleared)),
        4,
        theme::button_text,
        tooltip::Position::Top,
    );
    let search_row = row![search_input, clear_btn]
        .spacing(4)
        .align_y(Alignment::Center);

    // ── Body: search results or normal ticket groups ────────────────
    let body: Element<'_, Message> = if search_active {
        render_search_results(board_state)
    } else {
        render_normal_ticket_list(board_state)
    };

    let content = column![
        Space::new().height(8),
        search_row,
        Space::new().height(8),
        body,
    ]
    .spacing(0);

    container(content)
        .padding([8, 12])
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::surface_container_style)
        .into()
}

/// Hint line for empty/loading ticket-list states.
fn section_hint(label: &str) -> Element<'_, Message> {
    column![
        Space::new().height(8),
        text(label).size(12).color(theme::TEXT_MUTED),
    ]
    .spacing(4)
    .padding([8, 0])
    .into()
}

/// Render the normal ticket list partitioned into groups (In Progress,
/// Ready, Pending, Completed).
fn render_normal_ticket_list(board_state: &board::BoardState) -> Element<'_, Message> {
    let [in_progress, ready, pending, completed] =
        board::BoardState::board_sections(&board_state.tickets);

    let is_empty =
        in_progress.is_empty() && ready.is_empty() && pending.is_empty() && completed.is_empty();

    if !board_state.load_state.has_loaded() {
        section_hint("Loading…")
    } else if is_empty {
        section_hint("No tickets")
    } else {
        let mut groups = Column::new().spacing(8);
        if !in_progress.is_empty() {
            groups = groups.push(group_section("In Progress", &in_progress, board_state));
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
    }
}

/// Render FTS search results as ticket cards in a scrollable list.
fn render_search_results(board_state: &board::BoardState) -> Element<'_, Message> {
    if board_state.search_results.is_empty() {
        section_hint("No matching tickets")
    } else {
        let mut cards = Column::new().spacing(2);
        for ticket in &board_state.search_results {
            cards = cards.push(board_state.render_ticket_card(ticket).map(Message::Board));
        }
        scrollable(cards)
            .height(Length::Fill)
            .direction(theme::vertical_scrollbar())
            .style(theme::scrollbar_style)
            .into()
    }
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
        // Window events are subscribed from the start, before boot completes.
        // Pre-boot Resized/Moved events update last_size/last_position, which
        // are persisted to window-state.json on close — without them, closing
        // a never-resized window would overwrite the restored geometry with
        // the hardcoded defaults (iced does not replay missed window events).
        // Close-request handling stays behind the readiness gate so the
        // shutdown/checkpoint path never runs before stores are initialized.
        let window_events = iced::Subscription::batch([
            window::resize_events()
                .map(|(id, size)| Message::WindowEvent(id, window::Event::Resized(size))),
            window::events().filter_map(|(id, event)| {
                matches!(&event, window::Event::Moved(_)).then_some(Message::WindowEvent(id, event))
            }),
        ]);
        if !self.ready {
            return window_events;
        }
        iced::Subscription::batch([
            window_events,
            iced::time::every(Duration::from_secs(1)).map(|_| Message::Tick),
            window::close_requests().map(Message::CloseRequested),
            keyboard::listen().filter_map(|event| {
                use keyboard::Key;
                let (key, modifiers, physical_key) = parse_key_press(event)?;
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
            // Git file-change subscription: drives the event-driven local git
            // footer refresh when workspace files change.
            iced::Subscription::run(git_file_changes_subscription),
            // Git pipeline-commit subscription: refreshes the footer promptly
            // after a pipeline auto-commit (ref-only change the watcher misses).
            iced::Subscription::run(git_commit_subscription),
            // Board change-stream subscription: drives the board ticket list from
            // CDC deltas instead of a 1s full re-poll.
            iced::Subscription::run(board_change_subscription),
            // TTS download progress subscription (always active while ready).
            iced::Subscription::run(tts_download_subscription).map(Message::TtsDownloadEvent),
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
/// token fires (self-update restart, SIGTERM/SIGINT), and
/// [`Message::DrainStarted`] when the graceful-drain window begins (first
/// signal / window-close / self-update) — the token does NOT fire on the
/// first signal, so the GUI needs a separate drain signal to disable input
/// while the drain runs.
fn shutdown_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    iced::stream::channel(1, |mut output: mpsc::Sender<Message>| async move {
        let token = crate::shutdown::shutdown_token();
        // Wait for the drain flag OR the token (a drain always precedes the
        // token in the two-signal protocol, but the token may fire without a
        // drain on force-cancel paths).
        loop {
            if crate::shutdown::is_draining() {
                let _ = output.try_send(Message::DrainStarted);
                break;
            }
            if token.is_cancelled() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        token.cancelled().await;
        let _ = output.try_send(Message::Shutdown);
    })
}

/// Subscription that emits [`crate::audio::tts::TtsDownloadEvent`]s from the global
/// TTS download broadcast channel, forwarded as [`Message::TtsDownloadEvent`].
///
/// Unlike the hand-rolled channel loop this replaces, an uninitialized
/// `DOWNLOAD_EVENTS` source yields an empty stream instead of panicking
/// (unreachable in practice — `tts::init_global` precedes GUI startup).
fn tts_download_subscription()
-> impl futures_util::Stream<Item = crate::audio::tts::TtsDownloadEvent> {
    use iced::futures::channel::mpsc;
    common::broadcast_stream_producer(
        1,
        &crate::audio::tts::DOWNLOAD_EVENTS,
        |output: &mut mpsc::Sender<crate::audio::tts::TtsDownloadEvent>,
         event: Option<crate::audio::tts::TtsDownloadEvent>| {
            Box::pin(async move {
                // Best-effort delivery: drop events if the GUI channel is full;
                // lagged progress slots are dropped (next event is current state).
                if let Some(event) = event {
                    let _ = output.try_send(event);
                }
            })
        },
    )
}

/// Subscription that emits [`Message::Git`] with [`GitMessage::FileChanged`]
/// when workspace files change. Filesystem events are delivered by the
/// fff-search watcher on a dedicated thread (which only does a non-blocking
/// broadcast send) and consumed here to drive the event-driven git local
/// refresh. The broadcast is initialized in `main` before the iced application
/// runs (matching the [`LOG_BROADCAST`] convention); an uninitialized source
/// would yield an immediately-ending stream, which iced's subscription tracker
/// does not re-spawn.
fn git_file_changes_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    common::broadcast_stream_producer(
        1,
        &git::FILE_CHANGE_TX,
        |output: &mut mpsc::Sender<Message>, event: Option<()>| {
            Box::pin(async move {
                if event.is_some() {
                    let _ = output.try_send(Message::Git(git::GitMessage::FileChanged));
                }
            })
        },
    )
}

/// Subscription that emits [`Message::Git`] with
/// [`GitMessage::PipelineCommit`] when a pipeline auto-commit completes. A
/// commit is a ref-only change the file watcher never reports, so the footer
/// is refreshed promptly via [`GitState`]`s path-matched handler.
fn git_commit_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    common::broadcast_stream_producer(
        1,
        &crate::git::commands::GIT_COMMIT_TX,
        |output: &mut mpsc::Sender<Message>, event: Option<std::path::PathBuf>| {
            Box::pin(async move {
                if let Some(path) = event {
                    let _ = output.try_send(Message::Git(git::GitMessage::PipelineCommit(path)));
                }
            })
        },
    )
}

/// Subscription that emits [`BoardMessage::TicketChanged`] deltas from the
/// CDC change stream, and [`BoardMessage::BoardRefreshNeeded`] on a lagged
/// channel (a delta was lost — force a full snapshot refresh). The change
/// source is the shared [`crate::db::cdc`] tickets sender, initialised before
/// the iced app runs via [`crate::gui::init_board_change_tx`]; an uninitialized
/// source would yield an immediately-ending stream (Iced does not re-spawn it),
/// so the warm-up in `main` is required for the board to receive any delta.
fn board_change_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    common::broadcast_stream_producer(
        128,
        crate::db::cdc::ticket_sender_lock(),
        |output: &mut mpsc::Sender<Message>, event: Option<crate::db::cdc::ChangeEvent>| {
            Box::pin(async move {
                // Awaited send (backpressure): a full/lagged channel triggers a
                // refresh rather than silently dropping a delta.
                let msg = Message::Board(match event {
                    Some(ev) => board::BoardMessage::TicketChanged(Box::new(ev)),
                    None => board::BoardMessage::BoardRefreshNeeded,
                });
                let _ = futures_util::SinkExt::send(output, msg).await;
            })
        },
    )
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
        Page::RunningAgents => lucide::radar::<iced::Theme, iced::Renderer>(),
    };
    text.size(size).color(color).into()
}

/// Shared sidebar toggle wrapper — wraps an icon inside a centered,
/// full-width button with a tooltip at the given `position`.
///
/// Used by [`Dashboard::render_maintainer_toggle`],
/// [`Dashboard::render_pause_toggle`] and [`Dashboard::render_sidebar_nav`].
fn render_sidebar_toggle<'a>(
    icon: Element<'a, Message>,
    tooltip_text: impl text::IntoFragment<'a>,
    action: Option<Message>,
    position: tooltip::Position,
) -> Element<'a, Message> {
    tooltip(
        button(
            container(icon)
                .width(Length::Fill)
                .center_x(Length::Fill)
                .padding([4, 0]),
        )
        .width(Length::Fill)
        .padding(0)
        .style(theme::button_text)
        .on_press_maybe(action),
        text(tooltip_text).size(11),
        position,
    )
    .style(theme::tooltip_style)
    .into()
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

    /// Sidebar navigation icons: Home, Editor, Shell (28px). Running Agents
    /// is not in the sidebar — it is reachable only via the footer activity
    /// indicators.
    ///
    /// Uses Position::Right — iced snaps Top tooltips into the viewport,
    /// overlapping the topmost sidebar button.
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
            let tooltip_text = if disabled {
                format!("Select a workspace to access {}", page.label())
            } else {
                page.label().to_string()
            };
            let nav_btn = render_sidebar_toggle(
                page_icon(*page, 28, color),
                tooltip_text,
                if disabled {
                    None
                } else {
                    Some(Message::Navigation(*page))
                },
                tooltip::Position::Right,
            );
            col = col.push(nav_btn);
        }
        col.into()
    }

    /// Per-workspace Maintainer toggle button.
    /// Disabled when no workspace is selected (Personal mode).
    fn render_maintainer_toggle(&self) -> Element<'_, Message> {
        let has_ws = self.has_active_workspace();
        let maint_icon = widgets::maint_badge(self.maintenance_enabled());
        let tooltip_text = if !has_ws {
            "Select a workspace to toggle maintainer"
        } else if self.maintenance_enabled() {
            "stop maintenance"
        } else {
            "start maintenance"
        };
        render_sidebar_toggle(
            maint_icon.into(),
            tooltip_text,
            if has_ws {
                Some(Message::Toggle(ToggleKind::Maintenance))
            } else {
                None
            },
            tooltip::Position::Top,
        )
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
        let tooltip_text = if !has_ws {
            "Select a workspace to pause"
        } else if self.paused() {
            "Resume pipeline"
        } else {
            "Pause pipeline"
        };
        render_sidebar_toggle(
            pause_icon.into(),
            tooltip_text,
            if has_ws {
                Some(Message::Toggle(ToggleKind::Pause))
            } else {
                None
            },
            tooltip::Position::Top,
        )
    }

    /// Render the self-update confirmation modal (opened by
    /// [`Message::UpdateBot`], confirmed by [`Message::ConfirmUpdate`]).
    /// Returns a type-stable placeholder when closed.
    ///
    /// The text truthfully reflects the real sequence: the build/install
    /// runs in the background while the system keeps working — nothing is
    /// paused during the build — then in-flight work drains, databases are
    /// checkpointed, and the app restarts and resumes work automatically.
    /// The window closing is the success signal.
    fn render_update_confirm(&self) -> Element<'_, Message> {
        if !self.show_update_confirm {
            // Keep the Stack widget type stable across open/close transitions
            // (see `empty_stack_placeholder`): the open state is a Stack.
            return iced::widget::stack([widgets::empty_stack_placeholder()]).into();
        }
        let dialog = container(
            column![
                text("Update MahBot?")
                    .size(16)
                    .color(theme::TEXT_PRIMARY)
                    .font(theme::FONT_BOLD),
                Space::new().height(12),
                text(
                    "The new version is built and installed in the background \
                     while the system keeps working — the build can take \
                     10–60 minutes and nothing is paused during it.\n\n\
                     When the build finishes, current in-flight work is \
                     drained (up to ~10 min), databases are checkpointed, and \
                     the app restarts, automatically resuming work afterwards. \
                     The window closing signals the restart.",
                )
                .size(13)
                .color(theme::TEXT_SECONDARY),
                Space::new().height(16),
                row![
                    Space::new().width(Length::Fill),
                    button(text("Cancel").size(13))
                        .style(theme::button_secondary)
                        .on_press(Message::CancelUpdate),
                    Space::new().width(8),
                    button(text("Update").size(13))
                        .style(theme::button_primary)
                        .on_press(Message::ConfirmUpdate),
                ]
                .align_y(Alignment::Center),
            ]
            .width(Length::Fill),
        )
        .width(Length::Fixed(480.0))
        .padding(24)
        .style(theme::dialog_container_style);

        widgets::modal_backdrop(dialog, Message::CancelUpdate, 0.5)
    }

    /// Render the self-update button in the footer bar.
    /// Returns `None` when self-update is not available on this installation.
    ///
    /// Visibility is driven entirely by the shared availability cache — no
    /// Windows gate is needed here. Registry mode on Windows never discovers
    /// an update (the crates.io check returns `Ok(None)`), and
    /// [`execute_registry_update`] bails at its entry point, so the cached
    /// `available` stays false.
    fn render_update_button() -> Option<Element<'static, Message>> {
        let availability = crate::self_update::update_availability();
        // Show the button while an update is in flight even if `available` was
        // transiently clobbered, so "Updating…" is never hidden mid-update.
        if !availability.available && !availability.in_progress {
            return None;
        }
        let (update_color, tooltip_text, clickable) = if availability.in_progress {
            (theme::TEXT_FAINT, "Updating…".to_string(), false)
        } else {
            let tooltip = match crate::self_update::update_latest() {
                Some(v) => format!("Update MahBot to v{v}"),
                None => "Update MahBot".to_string(),
            };
            (theme::ACCENT, tooltip, true)
        };
        let update_icon = lucide::refresh_cw::<iced::Theme, iced::Renderer>()
            .size(24)
            .color(update_color);
        Some(widgets::icon_tooltip_button(
            update_icon,
            tooltip_text,
            if clickable {
                Some(Message::UpdateBot)
            } else {
                None
            },
            3,
            theme::button_text,
            tooltip::Position::Top,
        ))
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
            icons.push(widgets::icon_tooltip_button(
                icon,
                page.label(),
                Some(Message::Navigation(*page)),
                3,
                theme::button_text,
                tooltip::Position::Top,
            ));
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
            format!("{}…", crate::util::truncate_bytes(b, 19))
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
        Some(widgets::icon_tooltip_button(
            branch_content,
            "active branch",
            Some(Message::Git(git::GitMessage::OpenModal)),
            3,
            theme::button_text,
            tooltip::Position::Top,
        ))
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
        Some(widgets::icon_tooltip_button(
            sync_content,
            "sync commits pull and push",
            if self.git_state.is_syncing() {
                None
            } else {
                Some(Message::Git(git::GitMessage::Sync))
            },
            3,
            theme::button_text,
            tooltip::Position::Top,
        ))
    }

    /// Render the git diff stats button (+X/−Y, clickable -> diff modal).
    /// Returns `None` when there are no non-zero changes (including no
    /// oversized/binary untracked files to report).
    fn render_git_diff_stats(&self) -> Option<Element<'_, Message>> {
        let stats = self.git_state.diff_stats()?;
        Some(widgets::icon_tooltip_button(
            widgets::git_footer_stats::<Message>(
                stats.added,
                stats.removed,
                stats.huge_binary_file_count,
                15.0,
            )?,
            "uncommitted changes",
            Some(Message::OpenDiffModal(None)),
            3,
            theme::button_text,
            tooltip::Position::Top,
        ))
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
    /// Returns `None` when no agents are running.
    ///
    /// Each role-icon indicator is a clickable button navigating to the
    /// Running Agents page; its tooltip carries the role name.
    fn render_active_agents() -> Option<Element<'static, Message>> {
        let handles = crate::agent::registry::AGENT_REGISTRY.list();
        let mut role_counts: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        for h in &handles {
            *role_counts.entry(h.role.as_str()).or_insert(0) += 1;
        }
        if role_counts.is_empty() {
            return None;
        }
        let mut icons: Vec<Element<'_, Message>> = Vec::new();
        for (role_str, count) in &role_counts {
            let role: crate::Role = role_str.parse().unwrap_or(crate::Role::Engineer);
            let (color, _bg) = theme::role_badge_color_for(&role);
            let icon = theme::role_icon(&role).size(24).color(color);
            let content: Element<'_, Message> = if *count > 1 {
                container(
                    row![icon, text(format!("×{count}")).size(15).color(color)]
                        .spacing(3)
                        .align_y(Alignment::Center),
                )
                .padding([0, 3])
                .into()
            } else {
                container(icon).padding([0, 3]).into()
            };
            icons.push(widgets::icon_tooltip_button(
                content,
                role.display_label(),
                Some(Message::Navigation(Page::RunningAgents)),
                0,
                theme::button_text,
                tooltip::Position::Top,
            ));
        }
        Some(
            Row::with_children(icons)
                .spacing(12)
                .align_y(Alignment::Center)
                .into(),
        )
    }

    /// Render the in-flight non-agent LLM call counter next to the agent
    /// icons. Distinct marker (zap glyph in accent color); a tooltip lists
    /// the in-flight call kinds as human-readable labels. The whole indicator
    /// is a clickable button navigating to the Running Agents page.
    /// Returns `None` when none are in flight.
    fn render_non_agent_calls() -> Option<Element<'static, Message>> {
        let handles = crate::agent::registry::NON_AGENT_CALLS.list();
        if handles.is_empty() {
            return None;
        }
        let color = theme::ACCENT;
        let content = container(
            row![
                lucide::zap::<iced::Theme, iced::Renderer>()
                    .size(24)
                    .color(color),
                text(format!("×{}", handles.len())).size(15).color(color),
            ]
            .spacing(3)
            .align_y(Alignment::Center),
        )
        .padding([0, 3]);
        // Tooltip lists the in-flight kinds as static human-readable labels —
        // raw snake_case kind names never surface. Map to labels FIRST, then
        // dedup: two unknown kinds share the generic fallback label, so
        // deduping raw kinds first could repeat it.
        let mut labels: Vec<&str> = handles
            .iter()
            .map(|h| crate::agent::registry::call_kind_label(h.kind))
            .collect();
        labels.sort_unstable();
        labels.dedup();
        let tooltip_text = labels.join(", ");
        Some(widgets::icon_tooltip_button(
            content,
            tooltip_text,
            Some(Message::Navigation(Page::RunningAgents)),
            0,
            theme::button_text,
            tooltip::Position::Top,
        ))
    }

    /// Render the TTS download progress indicator in the centre of the footer bar.
    /// Shows the current file name and percentage (e.g. "duration_predictor.onnx 83%").
    fn render_tts_download_progress(&self) -> Element<'_, Message> {
        let Some((file_name, progress)) = &self.tts_download_progress else {
            return Space::new().width(0).into();
        };
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let pct = (progress * 100.0).round() as u32;
        let label = format!("TTS: {file_name} {pct}%");
        container(text(label).size(12).color(theme::TEXT_MUTED))
            .padding([0, 12])
            .into()
    }

    /// Render a compact voice status indicator in the footer bar.
    ///
    /// Shows a compact indicator for every [`VoiceStatus`] variant except
    /// [`VoiceStatus::Disabled`] (which is hidden — voice is off).
    /// [`VoiceStatus::Error`] displays the actual error string rather than
    /// a hardcoded message.
    fn render_voice_status() -> Element<'static, Message> {
        let label: String = match crate::audio::voice::get_status() {
            // Hidden when voice is disabled.
            VoiceStatus::Disabled => return Space::new().width(0).into(),
            VoiceStatus::LoadingModels => "🔊 Loading…".into(),
            VoiceStatus::ModelError => "🔊 ⚠ Model error".into(),
            VoiceStatus::Listening => "🔊 Listening".into(),
            VoiceStatus::Recording | VoiceStatus::RecordingManual => "🔊 Recording…".into(),
            VoiceStatus::Transcribing => "🔊 Transcribing…".into(),
            VoiceStatus::MicPermissionDenied => "🔊 No mic access".into(),
            VoiceStatus::MicDisconnected => "🔊 Mic disconnected".into(),
            VoiceStatus::Enrolling { sample, total, .. } => {
                format!("🔊 Enrolling {sample}/{total}")
            }
            VoiceStatus::ListeningDuringEnrollment { sample, total } => {
                format!("🔊 Listen… {sample}/{total}")
            }
            VoiceStatus::WaitingForSilenceDuringEnrollment { sample, total } => {
                format!("🔊 Wait… {sample}/{total}")
            }
            VoiceStatus::EnrollingNegatives {
                accumulated_secs,
                target_secs,
                ..
            } => {
                format!("🔊 Collecting negatives {accumulated_secs}s/{target_secs}s")
            }
            VoiceStatus::Enrolled => "🔊 ✅ Enrolled".into(),
            VoiceStatus::Error(msg) => format!("🔊 Error: {msg}"),
        };
        container(text(label).size(12).color(theme::TEXT_MUTED))
            .padding([0, 12])
            .into()
    }

    /// 42px footer bar — nav items (left) and active agents (right).
    fn footer_view(&self) -> Element<'_, Message> {
        let mut left_elements: Vec<Element<'_, Message>> = Vec::with_capacity(3);

        if let Some(el) = Self::render_update_button() {
            left_elements.push(el);
        }

        left_elements.push(self.render_nav_icons());

        if let Some(el) = self.render_git_block() {
            left_elements.push(el);
        }

        let left = Row::with_children(left_elements)
            .spacing(6)
            .align_y(Alignment::Center);

        // TTS download progress and voice status indicator (center of footer bar)
        let center = row![
            self.render_tts_download_progress(),
            Self::render_voice_status(),
        ]
        .spacing(8)
        .align_y(Alignment::Center);

        let right = Row::with_children(
            [Self::render_active_agents(), Self::render_non_agent_calls()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        )
        .spacing(12)
        .align_y(Alignment::Center);

        let footer_row = row![left, center, Space::new().width(Length::Fill), right]
            .align_y(Alignment::Center)
            .padding([3, 18]);

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
    #[expect(clippy::cast_precision_loss)]
    #[must_use]
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
#[must_use]
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
#[expect(clippy::cast_possible_truncation)]
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
                Ok(states) => Message::WorkspaceStatesRefreshed(states),
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
    let mut workspaces = HashMap::new();
    let mut restored_name = None;

    if let Ok(ws_list) = store.list().await {
        for ws in &ws_list {
            workspaces.insert(
                ws.name.clone(),
                WorkspaceInfo {
                    path: ws.path.clone(),
                    paused: ws.paused,
                    maintenance_enabled: ws.maintenance_enabled,
                },
            );
        }
    }

    if let Some(ref name) = prev_selection {
        // Empty string means "Personal" — always valid.
        if name.is_empty() || workspaces.contains_key(name.as_str()) {
            restored_name = Some(name.clone());
        }
    }
    if restored_name.is_none() {
        restored_name = Some(String::new());
    }

    Message::BootWorkspaces(workspaces, restored_name)
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
