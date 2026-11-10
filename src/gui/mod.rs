//! Native Iced dashboard — application entry point, navigation, and shared state.
//!
//! Iced owns the process Tokio runtime (`iced` feature `tokio`). MahBot
//! bootstraps via a startup [`iced::Task`] before the UI becomes interactive.

#![expect(
    clippy::struct_excessive_bools,
    clippy::if_not_else,
    clippy::collapsible_if
)]

/// The dashboard's window on the platforms whose window is the app's own (see its docs).
pub mod app_window;
pub(crate) mod attachment;
pub(crate) mod automations;
pub(crate) mod board;
pub(crate) mod common;
pub(crate) mod custom_roles;
pub(crate) mod custom_tools;
pub(crate) mod dialog;
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
pub(crate) mod pasteboard;
pub(crate) mod pulse;
pub(crate) mod running;
/// The screen a fresh window opens on, which bounds its size (see its docs).
pub mod screen;
pub(crate) mod selectable;
pub(crate) mod session_view;
pub(crate) mod sessions;
pub(crate) mod settings;
pub(crate) mod shell;
pub(crate) mod shortcut_keys;
pub(crate) mod text_rendering;
pub(crate) mod theme;
pub(crate) mod users;
pub(crate) mod widgets;
pub(crate) mod workspaces;

pub(crate) use self::pulse::activity_dot;

use crate::git::commands::GitPushDestination;
use crate::logs::LogStore;
use crate::pipeline::board::Ticket;

use iced::keyboard;
use iced::widget::{
    Column, Row, Space, button, column, container, image, pick_list, row, rule, scrollable, text,
    tooltip,
};
// The window-dragging area: the rail drags the window where the platform's title bar is hidden.
#[cfg(any(target_os = "macos", target_os = "windows"))]
use iced::widget::mouse_area;
use iced::window;
use iced::{Alignment, Element, Length, Task};

use crate::Role;
use crate::Workspace;
use crate::audio::voice::VoiceStatus;

use self::menus::ContextMenu;

use iced_fonts::lucide;

pub use theme::JETBRAINS_MONO;

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::broadcast;

use futures_util::StreamExt;

/// Coalescing window for runtime-change re-renders: bounds per-message and
/// per-tool-call bursts while keeping the Running Agents cards responsive.
const RUNTIME_REFRESH_COALESCE: Duration = Duration::from_millis(250);

const SIDEBAR_RAIL_WIDTH: f32 = 56.0;

/// The rail's icon size, in px; the running-agent indicators are smaller.
const RAIL_ICON_SIZE: f32 = 28.0;

/// The window control's vertical padding in its own chrome (start-up splash, a
/// failed boot): room for the system's rounded corner — 12px on macOS, 8px on Windows.
#[cfg(any(target_os = "macos", target_os = "windows"))]
const WINDOW_CONTROL_PADDING: f32 = if cfg!(target_os = "macos") {
    theme::PAD_12
} else {
    theme::PAD_8
};

#[cfg(any(target_os = "macos", target_os = "windows"))]
const RAIL_WINDOW_CONTROL_GAP: f32 = WINDOW_CONTROL_PADDING;

/// The bottom footer bar's fixed height, in px: the toast stack clears it (see
/// [`Dashboard::view`]) unless [`Dashboard::footer_hidden`].
const FOOTER_HEIGHT: f32 = 42.0;

/// The portions the Home page's chat and ticket column split the content area
/// by; the column's share also sizes a fresh window's opening width.
const CHAT_COLUMN_PORTION: u16 = 7;
const TICKET_COLUMN_PORTION: u16 = 3;

/// How long an exit requested before boot finished waits for the boot's result.
/// The boot is the only store writer until then, and a store left before its
/// first catalog run is refused on the next start (`db::has_product_schema`),
/// so waiting is the normal case; this bound covers only a start that never
/// finishes, which must stay closable.
const BOOT_EXIT_WAIT: Duration = Duration::from_secs(15);

/// Global broadcast sender for live log streaming. Set during `startup()`.
pub static LOG_BROADCAST: OnceLock<broadcast::Sender<String>> = OnceLock::new();

/// Warm the git file-change broadcast (see [`init_board_change_tx`]). Called from `main`.
pub fn init_git_file_change_tx() {
    git::init_file_change_tx();
}

/// Warm the git-commit broadcast (see [`init_board_change_tx`]). Called from `main`.
pub fn init_git_commit_tx() {
    crate::git::commands::init_git_commit_tx();
}

/// Warm the CDC ticket-change broadcast before iced runs. Every `init_*_tx` here
/// must be warmed from `main`: an uninitialized `OnceLock` ends its subscription
/// stream on the first poll, and iced never re-spawns it, so the board freezes at
/// the initial snapshot.
pub fn init_board_change_tx() {
    let _ = crate::db::cdc::ticket_sender();
}

/// Warm the CDC workspaces-table sender (see [`init_board_change_tx`]). Called from `main`.
pub fn init_workspace_tx() {
    crate::db::cdc::init_workspace_tx();
}

/// Warm the CDC users-table sender (see [`init_board_change_tx`]). Called from `main`.
pub fn init_users_tx() {
    crate::db::cdc::init_users_tx();
}

/// Warm the CDC user_channels-table sender (see [`init_board_change_tx`]). Called from `main`.
pub fn init_user_channels_tx() {
    crate::db::cdc::init_user_channels_tx();
}

/// Warm the runtime-change broadcast (see [`init_board_change_tx`]). Called from `main`.
pub fn init_runtime_event_tx() {
    crate::runtime_events::init_runtime_event_tx();
}

/// Warm the voice-recording broadcast (see [`init_board_change_tx`]): the composer records on
/// every platform, and an uninitialized sender would end its subscription on the first poll.
/// Called from `main`.
pub fn init_recording_tx() {
    crate::audio::voice::init_recording_tx();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Home,
    /// The session list's viewer. It has no rail entry:
    /// [`DashboardShortcut::Sessions`] is its only way in.
    Sessions,
    /// The log store's viewer: [`DashboardShortcut::Logs`] is its only way in.
    Logs,
    Shell,
    Editor,
    Settings,
    Automations,
    /// Live view of running agents and in-flight non-agent LLM work, reached only from the
    /// activity indicators (the footer's, or the rail's while [`Dashboard::footer_hidden`]).
    RunningAgents,
}

impl Page {
    /// The sidebar top nav, minus the pages [`Dashboard::page_available`] excludes.
    const fn sidebar_pages() -> &'static [Page] {
        &[Page::Home, Page::Editor, Page::Shell, Page::Automations]
    }

    /// Pages in the sidebar bottom nav — Sessions and Logs are absent by design.
    const fn sidebar_bottom_pages() -> &'static [Page] {
        &[Page::Settings]
    }

    const fn label(self) -> &'static str {
        match self {
            Page::Home => "Home",
            Page::Sessions => "Sessions",
            Page::Logs => "Logs",
            Page::Shell => "Shell",
            Page::Editor => "Editor",
            Page::Settings => "Settings",
            Page::Automations => "Automations",
            Page::RunningAgents => "Running Agents",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToastKind {
    Success,
    Warning,
    Error,
}

/// Stable toast id source — monotonically increasing, never reused, so a
/// toast's expiry timer cannot dismiss a later toast that reused its slot.
static NEXT_TOAST_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub(crate) struct Toast {
    pub id: u64,
    pub message: String,
    pub kind: ToastKind,
}

#[derive(Debug, Clone)]
pub enum ToastMessage {
    Saved,
    Deleted,
    Error(String),
    Warning(String),
    SuccessMsg(String),
}

impl Toast {
    fn new(message: String, kind: ToastKind) -> Self {
        Self {
            id: NEXT_TOAST_ID.fetch_add(1, Ordering::Relaxed),
            message,
            kind,
        }
    }

    fn from_toast_msg(msg: &ToastMessage) -> Self {
        match msg {
            ToastMessage::Saved => Toast::new("Saved".to_string(), ToastKind::Success),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToggleKind {
    Pause,
    /// The maintainer, not the pipeline.
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
pub enum Message {
    /// Async startup finished (or failed), carrying the log store the logs page is built from.
    Boot(Result<LogStore, String>),
    Navigation(Page),
    /// A toast's expiry timer fired: dismiss by stable id; an unknown id is a no-op.
    ToastExpired(u64),
    /// Coalesced: the agent registries, live transcript content or voice status changed.
    RuntimeChanged,
    /// A workspaces row changed (or the CDC stream lagged): one full map reload.
    WorkspacesCdcChanged,
    /// A users or user_channels row changed: re-list the Settings users and read the admin's
    /// active workspace back (the `workspace` tool's `add` writes that column).
    UsersCdcChanged,
    /// The admin's stored active workspace read back after a users change, so another
    /// surface's switch applies without a restart. A superseded read (`generation`) cannot.
    ActiveWorkspaceSynced {
        generation: u64,
        name: Option<String>,
    },
    /// Leave the iced runtime for `save_and_exit`, which `main` continues after `run()`: the
    /// shutdown token fires it (self-update restart, SIGTERM/SIGINT, the end of a drain).
    Shutdown,
    /// Input is disabled and `iced::exit` waits for the token.
    DrainStarted,
    /// [`Dashboard::close_window`] hides on macOS/Windows, exits elsewhere.
    CloseRequested(window::Id),
    /// A platform quit ask (⌘Q, the menu's Quit, the Dock's) runs `request_exit`, the
    /// second-quit escape included. Log-out/restart asks are not delivered here (`shutdown`).
    QuitRequested,
    /// Geometry, tracked for persist-on-exit, and file drops, which attach to the composer
    /// while Home is showing.
    WindowEvent(window::Id, window::Event),
    /// The dashboard window was created: macOS hides the title strip first
    /// ([`crate::gui::app_window`]), Windows takes the taskbar/Alt-Tab icons.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    WindowOpened(window::Id),
    /// The window's own platform handle — the `HWND` — answering [`Message::WindowOpened`].
    #[cfg(target_os = "windows")]
    WindowHandle(u64),
    /// A window action the rail asks for ([`WindowAction`]).
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    WindowAction(WindowAction),
    /// A gesture from outside the window ([`crate::window_requests`]): the window
    /// comes up, or the product quits. A show during a drain does nothing.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    WindowRequested(crate::window_requests::WindowRequest),
    /// Keyboard shortcut: Cmd+F — focus the primary search input on the current page.
    FocusSearch,
    /// Keyboard shortcut: Cmd+L — open the Logs page, which has no rail item.
    OpenLogs,
    /// Keyboard shortcut: Cmd+K — open the Sessions page, which has no rail item.
    OpenSessions,
    /// Keyboard shortcut: Escape — dismiss modal/panel/confirmation on the current page.
    EscapePressed,
    UpdateBot,
    ConfirmUpdate,
    CancelUpdate,
    UpdateResult(Result<String, String>),
    /// Toggle the selected workspace's pipeline pause or maintainer state.
    Toggle(ToggleKind),
    /// Footer workspace picker: switch the admin's workspace (managed workspaces only).
    WorkspacePick(String),
    /// A per-workspace toggle DB write's result: (kind, result, name, intended
    /// state). Success reloads the workspace map; an error shows a toast.
    ToggleResult(ToggleKind, Result<(), String>, String, bool),
    /// The completion message for a fire-and-forget write whose failure is logged at the
    /// write site.
    Nop,
    /// Workspace info and restored selection loaded during boot. The read result is carried as-is
    /// so a failed read shows in the picker and Settings list rather than as "no workspaces";
    /// `remember` says whether the name must be written back as the admin's selection
    /// ([`Dashboard::apply_boot_workspaces`]); `generation` is the read's reload generation, so a
    /// reload that answered meanwhile wins.
    BootWorkspaces {
        workspaces: Result<HashMap<String, Workspace>, String>,
        restored_name: String,
        remember: bool,
        generation: u64,
    },
    /// The wait for a hanging boot expired ([`BOOT_EXIT_WAIT`]): run the waiting exit.
    BootExitWaitExpired,
    /// Full workspace map reloaded from the store: the handler replaces the map and re-pushes the
    /// workspace-scoped surfaces, but never re-resolves the live selection, because the map can
    /// lag the stored choice — that is the DB read-back's call. A slow read cannot overwrite a
    /// newer one.
    WorkspacesReloaded {
        workspaces: HashMap<String, Workspace>,
        generation: u64,
    },
    /// The post-ready map reload failed: the current map is preserved and the failure renders
    /// where the picker / Settings list would be — unless a newer reload has answered.
    WorkspacesReloadFailed {
        error: String,
        generation: u64,
    },
    Home(home::HomeMessage),
    Logs(logs::LogMessage),
    Board(board::BoardMessage),
    Sessions(sessions::SessionsMessage),
    Automations(automations::AutomationsMessage),
    RunningAgents(running::RunningMessage),
    /// Diff modal overlay, not a page — named so around the removed `Page::Diff` variant.
    DiffModal(diff::DiffMessage),
    /// The read-only foreign commit view's own messages. Routing is by origin, not by what is
    /// on screen, so a task in flight from the picked view never lands in another project.
    ForeignDiff(diff::DiffMessage),
    Git(git::GitMessage),
    Shell(shell::ShellMessage),
    Editor(editor::EditorMessage),
    Settings(settings::SettingsMessage),

    /// Open the diff modal. Optional commit hash — `None` = working tree diff.
    OpenDiffModal(Option<String>),
    /// A sidebar ticket's project, looked up in the store because the dashboard's map had no
    /// entry for it yet; `path` is `None` when the project is gone.
    TicketCommitProjectResolved {
        workspace: String,
        commit_hash: String,
        path: Option<String>,
    },
    CloseDiffModal,
}

/// Let [`Dashboard::update`] intercept Toast and LinkClicked messages before
/// dispatching to page handlers, instead of repeating that in every page.
impl Message {
    pub(crate) fn as_toast(&self) -> Option<&ToastMessage> {
        match self {
            Message::Home(home::HomeMessage::Toast(tm))
            | Message::Board(board::BoardMessage::Toast(tm))
            | Message::DiffModal(diff::DiffMessage::Toast(tm))
            | Message::ForeignDiff(diff::DiffMessage::Toast(tm))
            | Message::Git(git::GitMessage::Toast(tm))
            | Message::Editor(editor::EditorMessage::Toast(tm))
            | Message::Sessions(sessions::SessionsMessage::Toast(tm))
            | Message::Automations(automations::AutomationsMessage::Toast(tm))
            | Message::Settings(
                settings::SettingsMessage::WorkspaceMsg(workspaces::WorkspacesMessage::Toast(tm))
                | settings::SettingsMessage::UserMsg(users::UsersMessage::Toast(tm)),
            ) => Some(tm),
            _ => None,
        }
    }

    /// The URL if this message wraps a `LinkClicked`. [`HomeMessage::LinkClicked`] is
    /// deliberately absent: `home.rs` handles its own inline context links.
    fn as_link_url(&self) -> Option<&str> {
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

/// Platform-aware keyboard modifier state from a [`keyboard::Modifiers`] value.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct KeyboardMods {
    pub is_cmd: bool,
    /// The platform modifier: ⌘ on macOS, Ctrl elsewhere.
    pub is_platform_mod: bool,
    pub ctrl_held: bool,
    /// macOS: Ctrl without Cmd — terminal control characters / emacs bindings.
    pub is_emacs_ctrl: bool,
    /// Non-macOS: Ctrl+Alt, i.e. AltGr character input.
    pub altgr_active: bool,
}

impl KeyboardMods {
    /// Navigation modifier (arrow-key movement, line start/end). macOS: Cmd only,
    /// because Ctrl is reserved for emacs bindings and control characters.
    #[must_use]
    pub(crate) fn is_nav_platform_mod(self) -> bool {
        if cfg!(target_os = "macos") {
            self.is_cmd
        } else {
            self.is_platform_mod
        }
    }

    /// Text modifier (clipboard C/X/V, IME guard), stricter than
    /// [`Self::is_nav_platform_mod`]: no macOS Cmd+Ctrl, no AltGr.
    #[must_use]
    pub(crate) fn is_text_platform_mod(self) -> bool {
        if cfg!(target_os = "macos") {
            self.is_cmd && !self.ctrl_held
        } else {
            self.is_platform_mod && !self.altgr_active
        }
    }

    /// General-shortcut modifier: the platform modifier, but never emacs Ctrl or AltGr.
    #[must_use]
    pub(crate) fn is_shortcut_platform_mod(self) -> bool {
        self.is_platform_mod && !self.is_emacs_ctrl && !self.altgr_active
    }
}

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

/// One of the dashboard's own shortcuts — the keys that still reach the dashboard
/// while a page holds the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DashboardShortcut {
    /// Close the open overlay, or clear the page's own transient state.
    Escape,
    Search,
    /// The shortcut that replaces the Logs page's rail item.
    Logs,
    /// The shortcut that replaces the Sessions page's rail item.
    Sessions,
}

/// The dashboard shortcut a key press stands for — Escape, Cmd+F, Cmd+L or Cmd+K (Ctrl
/// elsewhere), resolved through the physical key so a non-Latin layout still produces them. A
/// focused input leaves exactly these uncaptured: [`keyboard::listen`] only delivers events no
/// widget captured, and the composer holds the keyboard from Home's first frame.
pub(crate) fn dashboard_shortcut(
    key: &keyboard::Key,
    modifiers: keyboard::Modifiers,
    physical_key: keyboard::key::Physical,
) -> Option<DashboardShortcut> {
    if matches!(key, keyboard::Key::Named(keyboard::key::Named::Escape)) {
        return Some(DashboardShortcut::Escape);
    }
    let km = detect_keyboard_mods(modifiers);
    if km.altgr_active || !km.is_cmd {
        return None;
    }
    match key.to_latin(physical_key)? {
        'f' if !modifiers.shift() => Some(DashboardShortcut::Search),
        'k' if !modifiers.shift() => Some(DashboardShortcut::Sessions),
        'l' if !modifiers.shift() => Some(DashboardShortcut::Logs),
        _ => None,
    }
}

/// Whether a focused input must leave this key press uncaptured, so the shortcut
/// it stands for still reaches [`keyboard::listen`].
pub(crate) fn is_dashboard_shortcut(
    key: &keyboard::Key,
    modifiers: keyboard::Modifiers,
    physical_key: keyboard::key::Physical,
) -> bool {
    dashboard_shortcut(key, modifiers, physical_key).is_some()
}

/// Where the window's one start-up width decision stands. A fresh window opens without the
/// ticket column's width ([`WindowState::opening_size`]), because a fresh board normally has
/// nothing to show; the board's first read takes the fresh width back when the column is drawn
/// after all. Nothing resizes the window later: a column appearing changes the layout only.
#[derive(Clone, Copy)]
enum StartupWidth {
    /// Nothing to decide: saved geometry, or the decision is already made.
    Settled,
    /// Owed, waiting for the window's own report of its size — the platform rounds the
    /// requested size to its pixel grid.
    AwaitingReport,
    /// Owed, with the opening size in [`Dashboard::last_size`]: a different width is the
    /// user's own drag, which wins over the decision.
    Owed,
}

pub struct Dashboard {
    ready: bool,
    boot_error: Option<String>,
    /// An exit requested while the boot task ran: the exit waits for the boot's
    /// result (bounded by [`BOOT_EXIT_WAIT`]) rather than interrupt store bring-up.
    exit_after_boot: bool,
    page: Page,
    log_store: Option<LogStore>,

    /// Tracked window geometry: what the exit path persists.
    last_size: iced::Size,
    last_position: iced::Point,
    /// Where the one start-up width decision stands (see [`StartupWidth`]).
    startup_width: StartupWidth,
    /// The fresh window's size with its ticket column ([`fresh_size`]): the width the start-up
    /// decision widens to, at the height the window has by then.
    startup_size: iced::Size,
    toasts: Vec<Toast>,

    /// Per-workspace records keyed by name, rebuilt whole from the store on
    /// [`Message::WorkspacesReloaded`] (no incremental boolean patching).
    workspaces: HashMap<String, Workspace>,
    /// Last workspace-map read failure, rendered where the footer picker and the
    /// Settings list would be; `None` once a read succeeds.
    workspaces_error: Option<String>,
    /// Generation of the latest map reload, so a slow read cannot apply its stale
    /// outcome over a newer one.
    workspaces_reload_gen: u64,
    /// The read-back guard: bumped wherever this app makes the selection newer than anything a
    /// read in flight can return (a pick, a deletion's relocation, a boot pick written back) and
    /// on every read-back it dispatches; each read applies only while its generation matches. A
    /// read dispatched *after* a pick can still see that pick's pre-commit row and revert the
    /// display until its CDC event re-reads it.
    active_workspace_gen: u64,
    /// The `(name, path)` last pushed to the workspace-scoped surfaces, so a change of
    /// *resolvability* alone still re-pushes.
    pushed_workspace: Option<(String, String)>,
    /// Selected workspace name: `personal:{user}`, a managed workspace, or nothing.
    selected_workspace_name: Option<String>,
    /// A genuine exit request arrived while the update was finalizing (daemon shut down;
    /// checkpoint + spawn + exit pending). Only a real request sets it — the update's shutdown
    /// step also fires `Message::Shutdown`; if the update fails, the GUI exits on its own.
    exit_requested_during_update: bool,
    /// Graceful-drain window active: the window stays open, input disabled.
    draining: bool,
    /// The update starts only on [`Message::ConfirmUpdate`], so dismissing leaves the system
    /// untouched.
    show_update_confirm: bool,
    /// The work item awaiting manual-stop confirmation, while the Running Agents dialog shows;
    /// the stop runs on [`running::RunningMessage::StopConfirmed`], whose message carries the key.
    pending_stop: Option<running::StopTarget>,
    /// Expanded cards, keyed by (agent_id, generation) so a recycled agent_id never inherits a
    /// stale expansion; pruned on each [`Message::RuntimeChanged`].
    running_expanded: HashSet<(String, u64)>,
    logs_state: logs::LogsState,
    board_state: board::BoardState,
    sessions_state: sessions::SessionsState,
    automations_state: automations::AutomationsState,
    diff_state: diff::DiffState,
    home_state: home::HomeState,
    shell_state: shell::ShellState,
    editor_state: editor::EditorState,
    settings_state: settings::SettingsState,

    show_diff_modal: bool,
    /// The read-only view of another project's commit, from a sidebar ticket of another
    /// workspace: the project named in its header while open.
    foreign_workspace: Option<String>,
    /// That view's own state, held for the dashboard's lifetime so its load generation keeps
    /// increasing across opens and a load left in flight can never be accepted by the next one.
    foreign_diff_state: diff::DiffState,

    git_state: git::GitState,
}

impl Dashboard {
    /// The dashboard before its boot task finishes. `window` is the geometry it opens with
    /// (`WindowState::opening_size`, which `work_area` caps); a fresh one leaves the start-up width
    /// owed.
    #[must_use]
    pub fn loading(window: &WindowState, work_area: Option<iced::Size>) -> Self {
        Self {
            ready: false,
            boot_error: None,
            exit_after_boot: false,
            page: Page::Home,
            log_store: None,
            last_size: window.opening_size(work_area),
            last_position: iced::Point::new(-1.0, -1.0),
            startup_width: if window.restored {
                StartupWidth::Settled
            } else {
                StartupWidth::AwaitingReport
            },
            startup_size: fresh_size(work_area),
            toasts: Vec::new(),
            workspaces: HashMap::new(),
            workspaces_error: None,
            workspaces_reload_gen: 0,
            active_workspace_gen: 0,
            pushed_workspace: None,
            selected_workspace_name: None,
            exit_requested_during_update: false,
            draining: false,
            show_update_confirm: false,
            pending_stop: None,
            running_expanded: HashSet::new(),
            logs_state: logs::LogsState::new(),
            board_state: board::BoardState::new(),
            sessions_state: sessions::SessionsState::new(),
            automations_state: automations::AutomationsState::new(),
            diff_state: diff::DiffState::new(),
            home_state: home::HomeState::new(),
            shell_state: shell::ShellState::new(),
            editor_state: editor::EditorState::new(),
            settings_state: settings::SettingsState::new(),
            show_diff_modal: false,
            foreign_workspace: None,
            foreign_diff_state: diff::DiffState::new(),
            git_state: git::GitState::new(),
        }
    }

    fn finish_boot(&mut self, result: Result<LogStore, String>) -> Task<Message> {
        let tasks = match result {
            Ok(log_store) => {
                self.ready = true;
                self.boot_error = None;
                let refresh_logs = self.logs_state.refresh(&log_store);
                let refresh_board = self.board_state.refresh();
                self.log_store = Some(log_store);
                let boot_workspaces = Task::perform(
                    load_workspace_options(self.workspaces_reload_gen),
                    std::convert::identity,
                );

                // The app's conversation opens with the screen, once, with the
                // persisted composer draft; never by a workspace selection.
                let open_chat = self.home_state.open().map(Message::Home);

                // The Home chat runs the Phase-1 scripted onboarding while no LLM provider is
                // configured (and the Phase-2 Assistant kickoff once one is), so no boot redirect
                // to Settings. A trimmed non-empty key counts as set — `provider_key()` collapses
                // empty/whitespace to None — and so does an active custom endpoint.
                Task::batch([
                    refresh_logs.map(Message::Logs),
                    refresh_board.map(Message::Board),
                    boot_workspaces,
                    open_chat,
                ])
            }
            Err(e) => {
                self.boot_error = Some(e);
                Task::none()
            }
        };
        // A failed boot has nothing to keep running hidden, and a launch the updater restarted
        // hidden has no way back without a tray icon: either way the window comes up. A login
        // start never does.
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        if self::app_window::hidden() && (!self.ready || !crate::tray::available()) {
            if self::app_window::login_start() {
                tracing::warn!(
                    "the login start stays quiet with no way to reach it; a hand launch brings the window up"
                );
                return tasks;
            }
            return tasks.chain(self::app_window::show());
        }
        tasks
    }

    fn selected_workspace_info(&self) -> Option<&Workspace> {
        self.selected_workspace_name
            .as_ref()
            .and_then(|name| self.workspaces.get(name))
    }

    /// The resolved selection: a managed workspace in the map with a non-empty path.
    /// A personal name, a missing name and an empty path resolve to nothing.
    fn resolved_workspace(&self) -> Option<(&str, &str)> {
        let name = self.selected_workspace_name.as_deref()?;
        if crate::users::is_personal_workspace(name) {
            return None;
        }
        let path = self.workspaces.get(name)?.path.as_str();
        (!path.is_empty()).then_some((name, path))
    }

    /// Whether the bottom footer bar is hidden: no managed workspace in the map and the personal
    /// folder selected. Both halves are load-bearing — the selection stays unset until a read
    /// settles one, and after the last project is deleted it keeps naming that project while the
    /// map is empty. A failed read must not look like an empty map, so it leaves the bar and its
    /// failure report where they are. Hidden, the bar's content moves into the rail: voice status
    /// under the nav icons and the running-agent indicators above the bottom nav
    /// ([`Self::render_sidebar_mid`]).
    fn footer_hidden(&self) -> bool {
        self.workspaces_error.is_none()
            && self
                .workspaces
                .keys()
                .all(|name| crate::users::is_personal_workspace(name))
            && self
                .selected_workspace_name
                .as_deref()
                .is_some_and(crate::users::is_personal_workspace)
    }

    /// Whether `page` is offered in the navigation: the editor and the shell need a
    /// resolved workspace ([`Self::resolved_workspace`]).
    fn page_available(&self, page: Page) -> bool {
        !matches!(page, Page::Editor | Page::Shell) || self.resolved_workspace().is_some()
    }

    /// Whether the selected workspace's pipeline is paused (a strict freeze: no
    /// in-flight work and no stage advance until resume).
    fn paused(&self) -> bool {
        self.selected_workspace_info().is_some_and(|w| w.paused)
    }

    fn maintenance_enabled(&self) -> bool {
        self.selected_workspace_info()
            .is_some_and(|w| w.maintenance_enabled)
    }

    pub const fn theme(&self) -> iced::Theme {
        iced::Theme::Dark
    }

    fn persist_window_state(&self) {
        save_window_state(self.last_position, self.last_size);
    }

    /// The one start-up width decision, taken on the board's first read ([`StartupWidth`]): a board
    /// with nothing to show keeps the narrow opening, one that draws the column takes the fresh
    /// width back, its work-area cap included.
    fn take_startup_width(&mut self) -> Option<iced::Size> {
        let owed = !matches!(self.startup_width, StartupWidth::Settled);
        self.startup_width = StartupWidth::Settled;
        if !owed || self.board_state.has_nothing_to_show() {
            return None;
        }
        Some(iced::Size::new(
            self.startup_size.width,
            self.last_size.height,
        ))
    }

    fn settle_startup_width(&mut self) -> Task<Message> {
        let Some(size) = self.take_startup_width() else {
            return Task::none();
        };
        iced::window::latest().and_then(move |id| iced::window::resize(id, size))
    }

    /// The window's close — the platform's own request (Alt+F4, the system menu, a shortcut) and
    /// the rail's/splash's/failed boot's control: a hide where a hidden window stays reachable,
    /// the exit path otherwise. A hide is wrong while nothing keeps working (before boot, a failed
    /// boot), while the product is already leaving ([`crate::shutdown::aborting`], a
    /// self-update's drain) and with no tray icon; a platform with no hidden window
    /// ([`self::app_window`]) never hides.
    fn close_window(&mut self) -> Task<Message> {
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        if self.ready
            && crate::tray::available()
            && !crate::shutdown::aborting()
            && !(crate::self_update::update_in_progress()
                && crate::self_update::update_is_finalizing())
        {
            return self::app_window::hide();
        }
        self.request_exit(None)
    }

    /// Act on a request for the product to exit: name the route that asked, then start the wait it
    /// calls for. A close that cannot hide ([`Self::close_window`]), a platform quit and the tray
    /// menu's Quit are handled identically — parity is the point — as are the states with no
    /// drain (a self-update finalizing, a boot unfinished or failed). `route` is the ask's wording,
    /// or `None` for a close; a request that finds the exit pending leaves the record to its cause.
    fn request_exit(&mut self, route: Option<&'static str>) -> Task<Message> {
        if crate::self_update::update_in_progress() && crate::self_update::update_is_finalizing() {
            // The update owns the exit (its drain + checkpoint + swap + spawn): record the
            // request and wait, since a force-cancel would abort its drain. A failed update
            // re-enters for this request (`exit_requested_during_update`).
            if !self.exit_requested_during_update {
                record_exit_route(route);
            }
            self.persist_window_state();
            self.exit_requested_during_update = true;
            Task::none()
        } else if !self.ready {
            // Before boot finished there is no drain to complete — the drain-watch and the
            // shutdown subscription exist only after a successful boot. Mid-boot the exit waits
            // for the boot's result, since a store left before its first catalog run is refused
            // on the next start (`db::has_product_schema`); [`BOOT_EXIT_WAIT`] bounds a
            // hanging start.
            if self.boot_error.is_some() {
                // The start failed, so nothing is writing the stores any more.
                record_exit_route(route);
                self.save_and_exit()
            } else if self.exit_after_boot {
                // Already pending: this request is not what ends the process.
                Task::none()
            } else {
                record_exit_route(route);
                self.exit_after_boot = true;
                Task::perform(async { tokio::time::sleep(BOOT_EXIT_WAIT).await }, |()| {
                    Message::BootExitWaitExpired
                })
            }
        } else if crate::shutdown::is_draining() {
            // A request during a drain is the second request: force-cancel (fires the token →
            // Message::Shutdown → exit path). The rule is the drain flag's, not the gesture's: a
            // first ⌘Q after a close's drain forces it too. Once the token has fired, the exit is
            // under way and this request must not take the record over.
            if !crate::shutdown::shutdown_token().is_cancelled() {
                record_exit_route(route);
            }
            crate::shutdown::force_cancel();
            Task::none()
        } else {
            // Begin the GRACEFUL drain: the window stays open, input disabled, and the drain-watch
            // fires the token when no in-flight work remains (or force-cancels at the cap);
            // save_and_exit waits for Message::Shutdown.
            record_exit_route(route);
            crate::shutdown::drain_begin();
            Task::none()
        }
    }

    /// Leave the iced runtime for the exit path: flush the draft, record the window geometry,
    /// fire the shutdown token, then exit the runtime. The ordinary exit for everything the
    /// interface handles — a quit, a close that could not hide, SIGINT/SIGTERM, the clean end of
    /// a drain — and the entry to the teardown in `main`; [`crate::shutdown`] names the rest.
    fn save_and_exit(&self) -> Task<Message> {
        crate::channels::chat_draft::flush_global();
        self.persist_window_state();
        if crate::self_update::update_in_progress() && crate::self_update::update_is_finalizing() {
            // The update owns the exit (checkpoint, temp-root cleanup, lock release, spawn,
            // exit(0)): exiting here would abandon it mid-checkpoint and leave the daemon down
            // without a replacement, so wait. A failed update re-enters through UpdateResult.
            return Task::none();
        }
        // Fire the token before the runtime drops: `shutdown_after_dashboard` relies on it being
        // cancelled by then (see its note in main.rs), and a pre-boot request reaches here with
        // no drain to have fired it. Idempotent.
        crate::shutdown::force_cancel();
        // No checkpoint here: it lives in shutdown_after_dashboard, after the iced
        // runtime drops, so it is genuinely single-writer.
        iced::exit()
    }

    pub fn title(&self) -> String {
        let page_name = self.page.label();
        format!("MahBot — {page_name}")
    }

    /// Apply the boot's workspace configuration; the boot completes either way, a failed read
    /// leaving an empty map beside the failure.
    ///
    /// `read_generation` is the read's reload generation: a reload that landed meanwhile read the
    /// table later, so its map (or failure) wins and this outcome is dropped. The restored
    /// selection is applied either way; the page surfaces stay empty until it resolves.
    fn apply_boot_workspaces(
        &mut self,
        workspaces: Result<HashMap<String, Workspace>, String>,
        restored_name: &str,
        remember: bool,
        read_generation: u64,
    ) -> Task<Message> {
        if read_generation == self.workspaces_reload_gen {
            self.apply_boot_workspace_map(workspaces);
        }

        // The boot read always yields a name — a project or `personal:{user}`. A pick from a
        // failed read is written by nobody (it is no evidence that nothing is selected), so an
        // in-flight read-back may still correct it.
        if !remember {
            return self.apply_workspace_selection(restored_name);
        }
        // Written back unconditionally, unlike a pick: the name came from the same DB read as
        // the stored selection, so the loaded map has no say. `switch_workspace` would let a
        // stale map skip the write, and the next read-back would drop the desktop into the
        // personal folder.
        Task::batch([
            self.apply_workspace_selection_superseding_read_back(restored_name),
            Self::persist_active_workspace(restored_name.to_owned(), |_| Message::Nop),
        ])
    }

    /// Navigate to a page, refreshing page state; a page that does not resolve
    /// ([`Self::page_available`]) is refused rather than opened over no workspace.
    fn navigate_to(&mut self, page: Page) -> Task<Message> {
        if !self.page_available(page) {
            return Task::none();
        }
        self.page = page;
        // Sent AFTER `self.page` is set, so the handler sees the new state: the single source of
        // truth for the editor's gating — refresh on entry to Editor, close its modals elsewhere.
        let visibility = Task::done(Message::Editor(
            editor::EditorMessage::PageVisibilityChanged(self.page == Page::Editor),
        ));
        let refresh = match page {
            // These four need no refresh: Logs and Shell keep their own state, Editor receives
            // WorkspaceSelected, Running Agents reads the live registries at render time.
            Page::Logs | Page::Shell | Page::Editor | Page::RunningAgents => Task::none(),
            Page::Home => {
                let snap = iced::widget::operation::snap_to_end::<Message>(home::CHAT_SCROLL_ID);
                // The composer takes the keyboard once this page is on screen, so a
                // paste lands in the input without a click first.
                self.home_state.arm_composer_focus();
                let board_refresh = self.board_state.refresh().map(Message::Board);
                Task::batch([snap, board_refresh])
            }
            Page::Sessions => self.sessions_state.refresh().map(Message::Sessions),
            // The page renders nothing but its polled list, so entering it starts a
            // read rather than rendering stale entries.
            Page::Automations => self.automations_state.refresh().map(Message::Automations),
            Page::Settings => {
                // The workspace list comes from the managed-workspace map; only the
                // users, custom-tools and custom-roles lists need a read here.
                self.sync_settings_workspaces_from_map();
                Task::batch([
                    self.settings_state.refresh().map(Message::Settings),
                    self.refresh_settings_users(),
                    self.refresh_settings_custom_tools(),
                    self.refresh_settings_custom_roles(),
                ])
            }
        };
        Task::batch([visibility, refresh])
    }

    /// Enter `page` from the shortcut that replaces its rail item. A dashboard
    /// overlay covers the rail, so nothing navigates out from under one.
    fn open_from_shortcut(&mut self, page: Page) -> Task<Message> {
        if self.overlay_modal_open() {
            return Task::none();
        }
        self.navigate_to(page)
    }

    fn refresh_settings_users(&self) -> Task<Message> {
        self.settings_state
            .users_state
            .refresh()
            .map(|msg| Message::Settings(settings::SettingsMessage::UserMsg(msg)))
    }

    fn refresh_settings_custom_tools(&mut self) -> Task<Message> {
        self.settings_state
            .custom_tools_state
            .refresh()
            .map(|msg| Message::Settings(settings::SettingsMessage::CustomToolsMsg(msg)))
    }

    fn refresh_settings_custom_roles(&self) -> Task<Message> {
        self.settings_state
            .custom_roles_state
            .refresh()
            .map(|msg| Message::Settings(settings::SettingsMessage::CustomRolesMsg(msg)))
    }

    /// Sync the Settings workspace list from the managed-workspace map, sorted by
    /// name to match the store's `ORDER BY name`. No DB read: the map is the source.
    fn sync_settings_workspaces_from_map(&mut self) {
        let mut list: Vec<Workspace> = self.workspaces.values().cloned().collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        self.settings_state.workspaces_state.workspaces = list;
    }

    /// Toggle the resolved project's pause or maintainer state; the name written is the
    /// resolved one, never a stored name that no longer exists.
    fn toggle_workspace_state(&mut self, kind: ToggleKind) -> Task<Message> {
        let Some(ws_name) = self.resolved_workspace().map(|(name, _)| name.to_string()) else {
            return Task::none();
        };
        let current_state = match kind {
            ToggleKind::Pause => self.paused(),
            ToggleKind::Maintenance => self.maintenance_enabled(),
        };
        let new_state = !current_state;
        let ws_name_clone = ws_name.clone();
        Task::perform(
            kind.persist_to_store(ws_name_clone, new_state),
            move |result| Message::ToggleResult(kind, result, ws_name, new_state),
        )
    }

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
                let toast = self.push_toast(format!("{label} for {ws_name}"), ToastKind::Success);
                // Immediate post-toggle refresh; CDC confirms durably later.
                let refresh = self.reload_workspace_map();
                Task::batch([toast, refresh])
            }
            Err(e) => self.push_toast(format!("{}: {e}", kind.label_err()), ToastKind::Error),
        }
    }

    /// Push a toast whose expiry timer emits [`Message::ToastExpired`] with its id.
    fn push_toast(&mut self, message: String, kind: ToastKind) -> Task<Message> {
        self.spawn_toast(Toast::new(message, kind))
    }

    fn push_toast_msg(&mut self, msg: &ToastMessage) -> Task<Message> {
        self.spawn_toast(Toast::from_toast_msg(msg))
    }

    fn spawn_toast(&mut self, toast: Toast) -> Task<Message> {
        let id = toast.id;
        let duration = toast.duration();
        self.toasts.push(toast);
        Task::perform(
            async move { tokio::time::sleep(duration).await },
            move |()| Message::ToastExpired(id),
        )
    }

    /// Escape: dismiss modals in priority order (self-update confirmation → foreign
    /// commit view → diff modal → git branch modal → page-level escape).
    fn process_escape(&mut self) -> Task<Message> {
        if self.show_update_confirm {
            self.show_update_confirm = false;
            Task::none()
        } else if self.foreign_workspace.is_some() {
            self.foreign_workspace = None;
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
                // Shell and Automations have no escape handling; on Running Agents,
                // Escape dismisses the pending stop confirmation (Keep), never confirms.
                Page::RunningAgents => {
                    self.pending_stop = None;
                    Task::none()
                }
                Page::Shell | Page::Automations => Task::none(),
                Page::Logs => self
                    .logs_state
                    .update(
                        logs::LogMessage::Escape,
                        self.log_store.as_ref().expect("ready"),
                    )
                    .map(Message::Logs),
                Page::Sessions => self
                    .sessions_state
                    .update(sessions::SessionsMessage::Escape, &self.workspaces)
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

    /// Process a Settings message, reloading the workspace map when workspaces are
    /// added or deleted.
    fn process_settings_message(&mut self, msg: settings::SettingsMessage) -> Task<Message> {
        // A Settings-side workspace mutation activates at most one project: an add
        // activates the new workspace (as the `workspace` tool's own `add` does), a
        // delete moves the admin's selection in the store and the desktop follows —
        // reported, since changed tabs, git state and toggles are otherwise the only
        // sign of it.
        let mut activation = None;
        match &msg {
            settings::SettingsMessage::AddWorkspaceResult(Ok(ws)) => {
                // Seeded so the picker renders it at once and the activation's map
                // guard sees it; the reload below confirms it.
                self.workspaces.insert(ws.name.clone(), ws.clone());
                activation = Some(self.activate_workspace(&ws.name));
            }
            settings::SettingsMessage::WorkspaceMsg(
                workspaces::WorkspacesMessage::DeleteResult(Ok((removed, active_now))),
            ) => {
                // The store already resolved the admin's selection; aligning the desktop at
                // once keeps it out of the no-project state while the map reload and the users
                // read-back race each other.
                let active_now = active_now.clone().unwrap_or_else(|| {
                    crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME)
                });
                if self.selected_workspace_name.as_deref() != Some(active_now.as_str()) {
                    let notice = if crate::users::is_personal_workspace(&active_now) {
                        format!("Project {removed} removed — the personal folder is active")
                    } else {
                        format!("Project {removed} removed — {active_now} is active")
                    };
                    activation = Some(Task::batch([
                        self.apply_workspace_selection_superseding_read_back(&active_now),
                        self.push_toast(notice, ToastKind::Success),
                    ]));
                }
            }
            _ => {}
        }
        // Any successful Settings-side workspace mutation (add, delete, reanalyze,
        // diagnostics/rules edits) refreshes the managed-workspace map from one full DB read —
        // the map is the source of truth, and syncing the Settings list from it gives the page
        // immediate feedback. The workspaces sub-state never reads the store.
        let needs_global_reload = matches!(
            msg,
            settings::SettingsMessage::WorkspaceMsg(
                workspaces::WorkspacesMessage::DeleteResult(Ok(_))
                    | workspaces::WorkspacesMessage::ReanalyzeResult(Ok(()))
                    | workspaces::WorkspacesMessage::DiagnosticsSaved(_, Ok(()))
                    | workspaces::WorkspacesMessage::RediscoverDiagnosticsResult(_, Ok(()))
                    | workspaces::WorkspacesMessage::RulesSaved(_, Ok(()))
            ) | settings::SettingsMessage::AddWorkspaceResult(Ok(_))
        );

        let settings_task = self.settings_state.update(msg).map(Message::Settings);
        let tail = if needs_global_reload {
            Task::batch([settings_task, self.reload_workspace_map()])
        } else {
            settings_task
        };
        Task::batch(std::iter::once(tail).chain(activation))
    }

    /// The Running Agents manual-stop flow: button → confirmation → async stop → toast.
    fn process_running_message(&mut self, msg: running::RunningMessage) -> Task<Message> {
        match msg {
            running::RunningMessage::StopRequest(target) => {
                // Confirmation only; the target's key is never displayed.
                self.pending_stop = Some(target);
                Task::none()
            }
            running::RunningMessage::StopConfirmed => {
                let Some(target) = self.pending_stop.take() else {
                    return Task::none();
                };
                match target {
                    // Fire the run's signal and stop its agents, silently: only the GUI gets a
                    // toast. The cancelled-run handoff (or a no-orchestrator sweep) owns cleanup
                    // and folder release. Infallible.
                    running::StopTarget::Research(run_key) => Task::perform(
                        async move { crate::research_cancel::cancel_research_run(&run_key).await },
                        |()| {
                            Message::RunningAgents(running::RunningMessage::StopFinished(Ok(
                                "Research run cancelled — agents stopped; the run's temporary \
                                 traces are being removed."
                                    .to_string(),
                            )))
                        },
                    ),
                    // Stopping a launch: cancel its agent, mark the branch deliberately stopped,
                    // drop its pending messages. A failure is a toast, never swallowed.
                    running::StopTarget::Launch(agent_id) => Task::perform(
                        async move {
                            match crate::launch::stop(&agent_id).await {
                                Ok(()) => {
                                    Ok("Launch stopped — its branch is marked stopped and its \
                                     pending messages were dropped."
                                        .to_string())
                                }
                                Err(e) => Err(format!("could not stop the launch: {e}")),
                            }
                        },
                        |result| {
                            Message::RunningAgents(running::RunningMessage::StopFinished(result))
                        },
                    ),
                }
            }
            running::RunningMessage::StopDismissed => {
                self.pending_stop = None;
                Task::none()
            }
            running::RunningMessage::StopFinished(result) => match result {
                Ok(text) => self.push_toast_msg(&ToastMessage::SuccessMsg(text)),
                Err(text) => self.push_toast_msg(&ToastMessage::Error(text)),
            },
            running::RunningMessage::ToggleAgentExpanded {
                agent_id,
                generation,
            } => {
                if self
                    .running_expanded
                    .contains(&(agent_id.clone(), generation))
                {
                    self.running_expanded.remove(&(agent_id, generation));
                } else {
                    self.running_expanded.insert((agent_id, generation));
                }
                Task::none()
            }
        }
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, message: Message) -> Task<Message> {
        // Toast and LinkClicked are intercepted before page dispatch, so page handlers only need
        // a Task::none() arm for exhaustiveness. A Home LinkClicked is handled by Home itself.
        if self.ready {
            if let Some(tm) = message.as_toast() {
                return self.push_toast_msg(tm);
            }

            if let Some(url) = message.as_link_url() {
                open_url(url);
                return Task::none();
            }
        }

        match message {
            Message::Boot(result) => {
                // An exit requested while the boot was writing the stores takes effect now: the
                // boot has stopped, so it cannot be left mid-bring-up (`request_exit`).
                if std::mem::take(&mut self.exit_after_boot) {
                    return self.save_and_exit();
                }
                self.finish_boot(result)
            }
            Message::BootExitWaitExpired => {
                // The boot hung past [`BOOT_EXIT_WAIT`]: the waiting exit ends the
                // process anyway.
                self.save_and_exit()
            }
            Message::BootWorkspaces {
                workspaces,
                restored_name,
                remember,
                generation,
            } => self.apply_boot_workspaces(workspaces, &restored_name, remember, generation),
            Message::CloseRequested(_) => self.close_window(),
            // A quit from outside the window — the tray menu's item — ends the process in
            // every state, as a platform quit does.
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            Message::WindowRequested(crate::window_requests::WindowRequest::Quit) => {
                self.request_exit(Some(crate::shutdown::TRAY_QUIT_TRIGGER))
            }
            // A window a restart left hidden comes up before boot finishes too. A product on its
            // way out has none to bring up — and a second launch asks this first (`db::ipc`).
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            Message::WindowRequested(crate::window_requests::WindowRequest::Show) => {
                if crate::shutdown::aborting() {
                    Task::none()
                } else {
                    self::app_window::show()
                }
            }
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            Message::WindowAction(action) => match action {
                // The window follows the pointer for the rest of the gesture.
                WindowAction::Drag => iced::window::latest().and_then(iced::window::drag),
                // Where it cannot hide — the splash, a failed boot — the control quits
                // instead (`close_window`).
                WindowAction::Close => self.close_window(),
            },
            Message::QuitRequested => self.request_exit(Some(crate::shutdown::MACOS_QUIT_TRIGGER)),
            Message::Shutdown => self.save_and_exit(),
            Message::DrainStarted => {
                // A restart toast only once the update is finalizing and shutting the
                // service down: a plain quit, close or signal during an update still in
                // flight must not claim a restart. `!self.draining` guards single emission.
                if !self.draining
                    && crate::self_update::update_in_progress()
                    && crate::self_update::update_is_finalizing()
                {
                    let toast = self.push_toast(
                        "Update complete — restarting…".to_string(),
                        ToastKind::Warning,
                    );
                    self.draining = true;
                    return toast;
                }
                self.draining = true;
                Task::none()
            }
            Message::WindowEvent(_, event) => match event {
                window::Event::Resized(new_size) => {
                    match self.startup_width {
                        // The first report is the size the platform gave the window, not
                        // a resize.
                        StartupWidth::AwaitingReport => {
                            self.startup_width = StartupWidth::Owed;
                        }
                        // A different width is the user's own drag, even before the board's
                        // first read; a report of the width the window already has is the
                        // platform's again. Only the width decides.
                        #[expect(
                            clippy::float_cmp,
                            reason = "the platform reports a width exactly; a different width is not an approximate one"
                        )]
                        StartupWidth::Owed if new_size.width != self.last_size.width => {
                            self.startup_width = StartupWidth::Settled;
                        }
                        StartupWidth::Settled | StartupWidth::Owed => {}
                    }
                    self.last_size = new_size;
                    Task::none()
                }
                window::Event::Moved(new_pos) => {
                    self.last_position = new_pos;
                    Task::none()
                }
                // A dropped file attaches to the next chat message: the whole Home page
                // is the target, every other page (and an overlay above Home) ignores
                // drops. Several files arrive as several events; a folder is not an attachment.
                window::Event::FileDropped(path) => {
                    if self.page != Page::Home || self.overlay_modal_open() {
                        return Task::none();
                    }
                    let (attachments, refused) = attachment::intake([path]);
                    if attachments.is_empty() && refused.is_empty() {
                        return Task::none();
                    }
                    self.home_state
                        .update(home::HomeMessage::Attach {
                            attachments,
                            refused,
                        })
                        .map(Message::Home)
                }
                _ => Task::none(),
            },
            #[cfg(target_os = "macos")]
            Message::WindowOpened(_) => self::app_window::opened(),
            #[cfg(target_os = "windows")]
            Message::WindowOpened(id) => window::raw_id::<Message>(id).map(Message::WindowHandle),
            #[cfg(target_os = "windows")]
            Message::WindowHandle(raw) => {
                crate::app_icon::apply_window_icons(raw);
                Task::none()
            }
            Message::CloseDiffModal => {
                // Drops whichever view's commit ref, so a later open cannot inherit a
                // stale one. `ClearCommitState` never returns work.
                let was_foreign = self.foreign_workspace.take().is_some();
                self.show_diff_modal = false;
                let state = if was_foreign {
                    &mut self.foreign_diff_state
                } else {
                    &mut self.diff_state
                };
                let _ = state.update(diff::DiffMessage::ClearCommitState);
                Task::none()
            }
            // Navigation has explicit before/after-ready handling
            Message::Navigation(_) if !self.ready => Task::none(),
            Message::Navigation(page) => self.navigate_to(page),

            // Everything else before boot → silent no-op
            _ if !self.ready => Task::none(),

            Message::WorkspacePick(name) => self.select_workspace(&name),
            Message::ToastExpired(id) => {
                // An already-expired or unknown id is a safe no-op.
                self.toasts.retain(|t| t.id != id);
                Task::none()
            }
            Message::RuntimeChanged => {
                running::prune_expanded(
                    &mut self.running_expanded,
                    &crate::agent::registry::AGENT_REGISTRY.list(),
                );
                // The tick's Sessions work is only observable while the page is
                // visible, and navigating onto it refreshes, so this cannot go stale.
                if self.page == Page::Sessions {
                    self.sessions_state
                        .update(sessions::SessionsMessage::RuntimeChanged, &self.workspaces)
                        .map(Message::Sessions)
                } else {
                    Task::none()
                }
            }
            Message::Home(msg) => self.home_state.update(msg).map(Message::Home),
            Message::Shell(msg) => self.shell_state.update(msg).map(Message::Shell),
            Message::Logs(msg) => self
                .logs_state
                .update(msg, self.log_store.as_ref().expect("ready"))
                .map(Message::Logs),
            Message::Board(msg) => {
                // A ticket's commit diff is intercepted for cross-page navigation before
                // it reaches board_state.update.
                if let board::BoardMessage::ViewCommitDiff {
                    ref commit_hash,
                    ref workspace_name,
                } = msg
                {
                    return self.open_ticket_commit_diff(workspace_name, commit_hash.clone());
                }
                // An applied read outcome is when the one start-up width decision is
                // taken (see [`StartupWidth`]); a delta, or an outcome dropped as stale,
                // carries no board state to decide it.
                let read = match &msg {
                    board::BoardMessage::Refreshed { generation, .. }
                    | board::BoardMessage::RefreshError { generation, .. } => Some(*generation),
                    _ => None,
                };
                let board = self.board_state.update(msg).map(Message::Board);
                if read.is_some_and(|generation| self.board_state.applied_read(generation)) {
                    let width = self.settle_startup_width();
                    return Task::batch([board, width]);
                }
                board
            }
            Message::Sessions(msg) => self
                .sessions_state
                .update(msg, &self.workspaces)
                .map(Message::Sessions),
            Message::Automations(msg) => {
                self.automations_state.update(msg).map(Message::Automations)
            }
            Message::RunningAgents(msg) => self.process_running_message(msg),
            // Auto-close the diff modal on a successful manual commit, keeping the diff
            // state in working-tree view: the commit handler already cleared commit state
            // and refreshed the diff, so no ClearCommitState. A foreign view has its own
            // channel and closes through its own controls.
            Message::DiffModal(diff::DiffMessage::CloseModal) => {
                self.show_diff_modal = false;
                Task::none()
            }
            // "Back to working tree" from the foreign view lands on the picked
            // workspace's working tree — the other project's is not a place to land.
            Message::ForeignDiff(diff::DiffMessage::BackToWorkingTree) => {
                self.open_diff_modal(None)
            }
            Message::ForeignDiff(msg) => {
                if self.foreign_workspace.is_none() {
                    return Task::none();
                }
                self.foreign_diff_state
                    .update(msg)
                    .map(Message::ForeignDiff)
            }
            Message::DiffModal(msg) => {
                // A manual commit is a ref-only change (HEAD + `.git/index`) the file
                // watcher never reports, so refresh the git footer promptly — otherwise
                // diff stats and remote state stay stale until the periodic timer.
                let commit_succeeded = matches!(&msg, diff::DiffMessage::CommitResult(Ok(_)));
                // Every `Message::DiffModal` belongs to the picked workspace's view and
                // the foreign view speaks `Message::ForeignDiff`, so neither state can
                // receive the other's messages.
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
            Message::Editor(msg) => {
                // The editor's mtime detector routes through GitState's throttled
                // file-change path (300ms coalescing) rather than its own handler, so the
                // resulting WorktreeSnapshot reaches the editor as a normal
                // GitStatusUpdated.
                if matches!(msg, editor::EditorMessage::RequestGitStatusRefresh) {
                    self.git_state
                        .update(git::GitMessage::FileChanged)
                        .map(Message::Git)
                } else {
                    self.editor_state.update(msg).map(Message::Editor)
                }
            }
            Message::Settings(msg) => self.process_settings_message(msg),
            Message::OpenDiffModal(commit_hash) => self.open_diff_modal(commit_hash),
            Message::TicketCommitProjectResolved {
                workspace,
                commit_hash,
                path,
            } => match path {
                Some(path) => self.show_foreign_commit(&workspace, path, commit_hash),
                None => self.push_toast(
                    format!("Project '{workspace}' is unavailable."),
                    ToastKind::Error,
                ),
            },
            Message::Git(msg) => {
                // One overlay at a time: opening the branch modal closes both diff views
                // from the Dashboard side.
                if matches!(msg, git::GitMessage::OpenModal) {
                    self.show_diff_modal = false;
                    self.foreign_workspace = None;
                }
                let snapshot = matches!(msg, git::GitMessage::WorktreeSnapshot(..));
                let task = self.git_state.update(msg).map(Message::Git);
                // A current-generation worktree snapshot (Ok or Err) forwards the latest
                // (possibly last-known-good) per-file statuses to the editor, so its tree
                // colours track GitState and it runs no `git status` itself.
                if snapshot {
                    if let Some((workspace, statuses)) = self.git_state.take_status_forward() {
                        return Task::batch([
                            task,
                            Task::done(Message::Editor(editor::EditorMessage::GitStatusUpdated {
                                workspace,
                                statuses,
                            })),
                        ]);
                    }
                }
                task
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
            Message::OpenLogs => self.open_from_shortcut(Page::Logs),
            Message::OpenSessions => {
                // Re-entering the page it opens would re-run the read `navigate_to`
                // starts on entry.
                if self.page == Page::Sessions {
                    return Task::none();
                }
                self.open_from_shortcut(Page::Sessions)
            }
            Message::EscapePressed => self.process_escape(),
            Message::UpdateBot => {
                // Opens the confirmation modal only; availability comes from the shared
                // cache, so a Telegram-initiated update disables the button here too.
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
                // Claim the update before spawning it, exactly as the `/update` command
                // does: an update that fires by itself can start between the check above
                // and the task below, and a second attempt must stand down here rather
                // than collide inside `execute_update` and report it as a failure.
                if !crate::self_update::claim_update_start() {
                    return Task::none();
                }
                self.persist_window_state();
                // The update reports itself to the admin in its own words too.
                let toast = self.push_toast("Update started…".to_string(), ToastKind::Success);
                Task::batch([
                    toast,
                    Task::perform(
                        async {
                            match crate::self_update::execute_update(None).await {
                                Ok(()) => Ok("ok".to_string()),
                                Err(e) => {
                                    // Reported exactly as the unattended trigger reports
                                    // it, while the window shows its own toast.
                                    crate::self_update::report_update_failure(&e).await;
                                    Err(format!("❌ Update failed:\n{e:#}"))
                                }
                            }
                        },
                        Message::UpdateResult,
                    ),
                ])
            }
            Message::CancelUpdate => {
                self.show_update_confirm = false;
                Task::none()
            }
            Message::UpdateResult(result) => {
                // `execute_update` calls exit(0) on success, so the Ok case never gets
                // here: the toasts and the window closing are its confirmation. On failure
                // it has already cleared the shared in-progress flag.
                if let Err(err) = result {
                    let toast = self.push_toast_msg(&ToastMessage::Error(err));
                    if self.exit_requested_during_update {
                        // A close or quit arrived while the update owned the exit; it
                        // failed, so run the normal exit checkpoint.
                        self.exit_requested_during_update = false;
                        return Task::batch([toast, self.save_and_exit()]);
                    }
                    return toast;
                }
                Task::none()
            }
            Message::Toggle(kind) => self.toggle_workspace_state(kind),
            Message::ToggleResult(kind, result, ws_name, intended_state) => {
                self.finish_toggle(kind, result, &ws_name, intended_state)
            }
            // One full map reload; the handler re-pushes the workspace-scoped surfaces
            // from the fresh map.
            Message::WorkspacesCdcChanged => self.reload_workspace_map(),
            Message::WorkspacesReloaded {
                workspaces,
                generation,
            } => {
                // Stale-result guard: a slow reload must not overwrite a newer one.
                if generation != self.workspaces_reload_gen {
                    return Task::none();
                }
                self.apply_workspace_reload(Ok(workspaces));

                // Nothing is picked from this map: it can lag the stored choice (a workspace added
                // and selected in one go is missing from a read that started before its insert),
                // and resolving against it would leave the desktop on the personal space while the
                // DB keeps the project active. The DB owns the choice and its read-back applies it
                // (`active_workspace_gen`); this map only re-pushes the workspace-scoped surfaces,
                // whose resolvability it may have changed. Only a desktop that applied no selection
                // yet takes the personal default, which the boot selection then replaces.
                if self.selected_workspace_name.is_some() {
                    return self.sync_workspace_surfaces();
                }
                let personal_name =
                    crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME);
                self.apply_workspace_selection(&personal_name)
            }
            // The event does not say which column changed, so the read-back is the authority
            // on the managed selection; the full re-list is its own recovery.
            Message::UsersCdcChanged => {
                self.active_workspace_gen += 1;
                let generation = self.active_workspace_gen;
                Task::batch([
                    self.refresh_settings_users(),
                    Task::perform(read_active_workspace(), move |name| {
                        Message::ActiveWorkspaceSynced { generation, name }
                    }),
                ])
            }
            Message::ActiveWorkspaceSynced {
                generation,
                name: Some(name),
            } => {
                // Stale reads are dropped (see `active_workspace_gen`); an unchanged
                // value is a no-op rather than a re-run of the propagate's pushes.
                if generation != self.active_workspace_gen
                    || self.selected_workspace_name.as_deref() == Some(name.as_str())
                {
                    return Task::none();
                }
                // "No project selected" is not a steady state: start-up and a deletion both
                // write the fallback back, so a personal answer while the map carries a
                // project is the residue of a failed start-up read — the live pick stands.
                if crate::users::is_personal_workspace(&name)
                    && self
                        .workspaces
                        .keys()
                        .any(|ws_name| !crate::users::is_personal_workspace(ws_name))
                {
                    return Task::none();
                }
                // Applied as stored, not resolved against the loaded map as a reload is:
                // the map can be a reload behind while the DB is authoritative about the
                // choice. In memory only — writing it back would answer our own CDC event.
                self.apply_workspace_selection(&name)
            }
            // A failed read-back changes nothing: the live selection stands.
            Message::ActiveWorkspaceSynced { name: None, .. } => Task::none(),
            Message::WorkspacesReloadFailed { error, generation } => {
                // Stale-result guard, mirroring the success arm: without it, a reload that
                // fails after a newer one succeeded would set the failure over the map
                // that read produced.
                if generation != self.workspaces_reload_gen {
                    return Task::none();
                }
                // Keep the current map (the selection may still resolve) and render the
                // failure where the picker / Settings list would be.
                self.apply_workspace_reload(Err(error));
                Task::none()
            }
            Message::Nop => Task::none(),
        }
    }

    /// Open the picked workspace's diff modal, closing any board or branch modal first and
    /// leaving any foreign commit view behind. `Some(commit_hash)` navigates to that commit,
    /// `None` to the working tree (clearing stale commit state).
    ///
    /// The view is workspace-scoped, so it never opens while nothing is resolved; losing the
    /// resolution has already closed it ([`Self::sync_workspace_surfaces`]).
    fn open_diff_modal(&mut self, commit_hash: Option<String>) -> Task<Message> {
        self.foreign_workspace = None;
        if self.resolved_workspace().is_none() {
            return Task::none();
        }
        let close_board = self
            .board_state
            .update(board::BoardMessage::CloseModal)
            .map(Message::Board);
        self.show_diff_modal = true;
        // CloseModal always returns Task::none(), so discarding the branch modal's is safe.
        let _ = self.git_state.update(git::GitMessage::CloseModal);
        let diff_task = match commit_hash {
            Some(hash) => Task::done(Message::DiffModal(diff::DiffMessage::NavigateToCommit(
                hash,
            ))),
            None => Task::done(Message::DiffModal(diff::DiffMessage::BackToWorkingTree)),
        };
        Task::batch([close_board, diff_task])
    }

    /// Open a sidebar ticket's commit diff against its own project: the picked workspace's
    /// ordinary view, or a read-only view of another project's commit (named in the view, no
    /// file-changing control, the picked view untouched). An unresolvable project is reported.
    fn open_ticket_commit_diff(&mut self, workspace: &str, commit_hash: String) -> Task<Message> {
        if self
            .resolved_workspace()
            .is_some_and(|(name, _)| name == workspace)
        {
            return self.open_diff_modal(Some(commit_hash));
        }
        // A click can beat the boot read of the map, so a miss falls back to the store
        // rather than calling an existing project unavailable.
        let Some(path) = self
            .workspaces
            .get(workspace)
            .map(|ws| ws.path.clone())
            .filter(|path| !path.is_empty())
        else {
            let (workspace, commit_hash) = (workspace.to_string(), commit_hash);
            let lookup_name = workspace.clone();
            return Task::perform(
                async move {
                    crate::workspace::get_by_name(&lookup_name)
                        .await
                        .ok()
                        .flatten()
                        .map(|ws| ws.path)
                        .filter(|path| !path.is_empty())
                },
                move |path| Message::TicketCommitProjectResolved {
                    workspace,
                    commit_hash,
                    path,
                },
            );
        };
        self.show_foreign_commit(workspace, path, commit_hash)
    }

    /// Show `commit_hash` of `workspace` (at `path`) as the read-only foreign
    /// commit view: the picked workspace's own view is left untouched.
    fn show_foreign_commit(
        &mut self,
        workspace: &str,
        path: String,
        commit_hash: String,
    ) -> Task<Message> {
        self.foreign_workspace = Some(workspace.to_string());
        self.show_diff_modal = false;
        let view = self
            .foreign_diff_state
            .update(diff::DiffMessage::CommitInWorkspace {
                path,
                hash: commit_hash,
            });
        // Close any board modal, as `open_diff_modal` does: one overlay at a time.
        let close_board = self
            .board_state
            .update(board::BoardMessage::CloseModal)
            .map(Message::Board);
        Task::batch([close_board, view.map(Message::ForeignDiff)])
    }

    /// Apply a workspace selection in memory and broadcast it to all pages, without
    /// writing the DB. Only for paths nothing supersedes: the read-back itself
    /// (`Message::ActiveWorkspaceSynced`), the pre-boot personal default and a boot pick
    /// from a failed read. Everything else uses
    /// [`Self::apply_workspace_selection_superseding_read_back`].
    fn apply_workspace_selection(&mut self, name: &str) -> Task<Message> {
        // Git state is cleared and eagerly refreshed by propagate → set_workspace_path.
        self.selected_workspace_name = Some(name.to_string());
        self.propagate_workspace_selection(name)
    }

    /// Make `name` active, superseding any read-back in flight: a pick, a deletion whose
    /// transaction has committed and a boot pick that is written back are all newer than
    /// anything a read dispatched earlier can return (see `active_workspace_gen`).
    fn apply_workspace_selection_superseding_read_back(&mut self, name: &str) -> Task<Message> {
        let propagate = self.apply_workspace_selection(name);
        self.active_workspace_gen += 1;
        propagate
    }

    /// The one workspace-switch path, shared by the footer picker and the Settings add
    /// form: [`Self::apply_workspace_selection_superseding_read_back`] plus the write,
    /// skipped when the loaded map dropped the name since the render
    /// ([`Self::selection_persistable`]).
    fn switch_workspace(
        &mut self,
        name: &str,
        feedback: fn(Result<String, String>) -> Message,
    ) -> Task<Message> {
        let activate = self.apply_workspace_selection_superseding_read_back(name);
        let write = self
            .selection_persistable(name)
            .then(|| Self::persist_active_workspace(name.to_string(), feedback));
        Task::batch(std::iter::once(activate).chain(write))
    }

    /// Whether a pick may be written to the DB: a managed name only while the loaded map
    /// carries it — the map can drop one between render and click, and the DB must not
    /// gain a dangling value (the in-memory selection still applies; the read-back
    /// decides). A personal space is always writable: the write stores NULL, the
    /// product's "no managed workspace selected", from which it computes the personal
    /// workspace on the fly.
    fn selection_persistable(&self, name: &str) -> bool {
        crate::users::is_personal_workspace(name) || self.workspaces.contains_key(name)
    }

    /// Persist `name` as the admin's active workspace. A failure is always logged here;
    /// `feedback` decides what the caller hears on top of that — the footer picker drops
    /// it (`Message::Nop`), the Settings side turns it into a toast.
    fn persist_active_workspace(
        name: String,
        feedback: fn(Result<String, String>) -> Message,
    ) -> Task<Message> {
        Task::perform(
            async move {
                users::set_selected_workspace_field(
                    crate::users::ADMIN_USER_NAME.to_string(),
                    name.clone(),
                )
                .await
                .inspect_err(|e| {
                    tracing::warn!(error = %e, "Failed to persist workspace selection");
                })
                .map(|()| name)
            },
            feedback,
        )
    }

    /// The footer picker's switch: [`Self::switch_workspace`], logging a failed write.
    fn select_workspace(&mut self, name: &str) -> Task<Message> {
        self.switch_workspace(name, |_| Message::Nop)
    }

    /// Make a just-added workspace active from the Settings add form, reporting a failed write.
    fn activate_workspace(&mut self, name: &str) -> Task<Message> {
        self.switch_workspace(name, Self::activation_feedback)
    }

    /// The activation write's feedback, through the Settings section's toast message
    /// ([`Message::as_toast`]) so a failure carries the same `Failed: …` wording as the rest.
    fn activation_feedback(outcome: Result<String, String>) -> Message {
        Message::Settings(settings::SettingsMessage::WorkspaceMsg(
            workspaces::WorkspacesMessage::Toast(Self::activation_toast(outcome)),
        ))
    }

    fn activation_toast(outcome: Result<String, String>) -> ToastMessage {
        match outcome {
            Ok(name) => ToastMessage::SuccessMsg(format!("Active workspace: {name}")),
            Err(e) => ToastMessage::Error(e),
        }
    }

    /// Propagate the selection to Home, then push the resolved workspace to the
    /// workspace-scoped surfaces ([`Self::sync_workspace_surfaces`]).
    ///
    /// The ticket sidebar is deliberately untouched: it lists every workspace's
    /// tickets, so a switch (onto the personal folder included) leaves it and its
    /// search exactly as they are.
    fn propagate_workspace_selection(&mut self, name: &str) -> Task<Message> {
        // Home hears the selection as stored, with no substitution, so the composer's
        // send carries the same choice as the picker and the DB. Its conversation is the
        // admin's own and is not touched.
        let home_task: Task<Message> =
            Task::done(home::HomeMessage::WorkspaceChanged(name.to_string())).map(Message::Home);

        let surfaces = self.sync_workspace_surfaces();
        Task::batch([home_task, surfaces])
    }

    /// Push the resolved workspace to the four workspace-scoped surfaces — editor, shell, footer
    /// git status and the change/diff view — and leave a page that stops resolving
    /// ([`Self::leave_unavailable_page`]). Nothing is substituted: an unresolved selection is
    /// never shown the personal workspace.
    ///
    /// The editor and the shell own per-workspace tabs, so they are only pushed a real
    /// resolution and never torn down; the git status and diff view hold derived state, so they
    /// are cleared, dropping the watcher, the refresh loop and the diff tick. An unchanged
    /// resolution is a no-op — the map-reload path calls this on every CDC event.
    fn sync_workspace_surfaces(&mut self) -> Task<Message> {
        let resolved = self
            .resolved_workspace()
            .map(|(name, path)| (name.to_string(), path.to_string()));
        // The diff view is an overlay of the current page: unresolved, it is closed rather
        // than left open and empty, owning Escape and the diff shortcuts.
        if resolved.is_none() {
            self.show_diff_modal = false;
        }
        let leave = self.leave_unavailable_page();
        if resolved == self.pushed_workspace {
            return leave;
        }
        self.pushed_workspace.clone_from(&resolved);
        let surfaces = if let Some((name, path)) = resolved {
            Task::batch([
                Task::done(editor::EditorMessage::WorkspaceSelected(
                    name.clone(),
                    path.clone(),
                ))
                .map(Message::Editor),
                Task::done(shell::ShellMessage::WorkspaceSelected(
                    name.clone(),
                    path.clone(),
                ))
                .map(Message::Shell),
                self.git_state
                    .set_workspace_path(Some(name.clone()), Some(path.clone()))
                    .map(Message::Git),
                Task::done(diff::DiffMessage::WorkspaceSelected(path)).map(Message::DiffModal),
            ])
        } else {
            Task::batch([
                self.git_state
                    .set_workspace_path(None, None)
                    .map(Message::Git),
                Task::done(diff::DiffMessage::WorkspaceUnresolved).map(Message::DiffModal),
            ])
        };
        Task::batch([leave, surfaces])
    }

    /// Return to Home when the current page is one the dashboard no longer offers:
    /// [`Self::navigate_to`] refuses such a page, so leaving it open would show a page
    /// nothing can navigate to.
    fn leave_unavailable_page(&mut self) -> Task<Message> {
        if self.page_available(self.page) {
            return Task::none();
        }
        self.navigate_to(Page::Home)
    }

    /// Reload the full workspace map from storage: the applied map re-pushes the
    /// workspace-scoped surfaces and fills in the pre-boot personal default, but never
    /// picks the active project from itself ([`Message::WorkspacesReloaded`]).
    ///
    /// The read carries its generation because it retries for a second on a transient
    /// failure: a delayed failure could otherwise re-set the error over a newer read
    /// that already succeeded.
    fn reload_workspace_map(&mut self) -> Task<Message> {
        self.workspaces_reload_gen = self.workspaces_reload_gen.wrapping_add(1);
        let generation = self.workspaces_reload_gen;
        Task::perform(load_workspace_map(generation), std::convert::identity)
    }

    /// Apply the boot map read. There is no previous map to keep, so a failure leaves an
    /// empty map beside the failure, and the boot path completes either way. The Settings
    /// list is re-derived here, so it can never diverge from the map.
    fn apply_boot_workspace_map(&mut self, result: Result<HashMap<String, Workspace>, String>) {
        match result {
            Ok(map) => {
                self.workspaces = map;
                self.workspaces_error = None;
            }
            Err(e) => {
                self.workspaces = HashMap::new();
                self.workspaces_error = Some(e);
            }
        }
        self.sync_settings_workspaces_from_map();
    }

    /// Apply a post-ready map reload: a failed read keeps the map the live selection may
    /// still resolve against, recording the failure for the picker and the Settings list.
    fn apply_workspace_reload(&mut self, result: Result<HashMap<String, Workspace>, String>) {
        match result {
            Ok(map) => {
                self.workspaces = map;
                self.workspaces_error = None;
            }
            Err(e) => self.workspaces_error = Some(e),
        }
        self.sync_settings_workspaces_from_map();
    }

    /// Whether a dashboard-level overlay modal is open over the current page. Gates the
    /// editor's keyboard ownership and the rail-replacing shortcuts: the overlay covers
    /// the rail, so those keys do nothing from under one. The board ticket modal is not
    /// included (it renders inside the Home page content, never the global overlay
    /// stack), and neither is `pending_stop` (rendered inside the Running Agents body).
    fn overlay_modal_open(&self) -> bool {
        self.foreign_workspace.is_some()
            || self.show_diff_modal
            || self.show_update_confirm
            || self.git_state.is_modal_open()
    }

    /// The Home page: the chat, and the ticket column beside it while the board has a
    /// ticket to show. The chat is the row's first child either way — a root that flipped
    /// shape with the column would be rebuilt, taking the chat's scroll position and the
    /// composer's caret and draft with it — and a lone fill portion still spans the row.
    fn home_page(&self) -> Element<'_, Message> {
        let home_view = self.home_state.view(self.draining).map(Message::Home);
        // Reset session from a right-click on empty space; per-bubble ContextMenus
        // (Copy message, see home.rs) capture the bubble ones.
        let home_view: Element<'_, Message> = ContextMenu::new(
            home_view,
            vec![menus::MenuItem::new(
                "Reset session".into(),
                Message::Home(home::HomeMessage::ClearChat),
            )],
        )
        .into();
        let mut columns =
            row![container(home_view).width(Length::FillPortion(CHAT_COLUMN_PORTION))];
        if !self.board_state.has_nothing_to_show() {
            let sidebar = ticket_sidebar(&self.board_state);
            // The board's two bulk actions. Pausing is offered only when a queued
            // ticket is listed.
            let mut menu_items = vec![menus::MenuItem::new(
                "Archive done & cancelled".into(),
                Message::Board(board::BoardMessage::ArchiveAllCompleted),
            )];
            if self.board_state.has_queued_tickets() {
                menu_items.push(menus::MenuItem::new(
                    "Pause all queued tickets".into(),
                    Message::Board(board::BoardMessage::PauseAllQueued),
                ));
            }
            let sidebar: Element<'_, Message> = ContextMenu::new(sidebar, menu_items).into();
            columns =
                columns.push(container(sidebar).width(Length::FillPortion(TICKET_COLUMN_PORTION)));
        }
        columns.into()
    }

    #[expect(clippy::too_many_lines)]
    pub fn view(&self) -> Element<'_, Message> {
        if let Some(err) = &self.boot_error {
            return Self::window_chrome(
                container(
                    column![
                        text("MahBot failed to start")
                            .size(20)
                            .color(theme::STATUS_ERROR),
                        text(err).size(theme::TEXT_14).color(theme::TEXT_SECONDARY),
                    ]
                    .spacing(theme::SPACE_12)
                    .padding(theme::PAGE_PADDING),
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into(),
            );
        }

        if !self.ready {
            return Self::window_chrome(
                container(
                    column![
                        image(crate::app_icon::widget_image()).width(96).height(96),
                        text("MahBot")
                            .size(theme::TEXT_24)
                            .color(theme::TEXT_PRIMARY),
                        text("Starting…")
                            .size(theme::TEXT_16)
                            .color(theme::TEXT_MUTED),
                    ]
                    .spacing(theme::SPACE_16)
                    .align_x(Alignment::Center),
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into(),
            );
        }

        let footer_hidden = self.footer_hidden();
        let sidebar = self.sidebar_view(footer_hidden);
        let content = match self.page {
            Page::Home => {
                let modal = self.board_state.render_modal_overlay().map(Message::Board);
                iced::widget::stack([self.home_page(), modal]).into()
            }
            Page::Logs => self.logs_state.view().map(Message::Logs),
            Page::Sessions => self
                .sessions_state
                .view(&self.workspaces)
                .map(Message::Sessions),
            Page::Automations => self.automations_state.view().map(Message::Automations),
            Page::Shell => self.shell_state.view().map(Message::Shell),
            Page::Editor => self
                .editor_state
                .view(self.overlay_modal_open())
                .map(Message::Editor),
            Page::Settings => self
                .settings_state
                .view(self.workspaces_error.as_deref())
                .map(Message::Settings),
            Page::RunningAgents => running::view(
                &self.workspaces,
                self.pending_stop.as_ref(),
                &self.running_expanded,
            ),
        };

        let mut body = column![
            row![sidebar, content]
                .width(Length::Fill)
                .height(Length::Fill)
        ]
        .width(Length::Fill)
        .height(Length::Fill);
        if !footer_hidden {
            body = body.push(self.footer_view());
        }

        // Keep the Stack widget type stable across toast transitions: a type tag change
        // (Column ↔ Stack) makes Iced destroy the whole widget tree, losing scroll
        // positions, cursor state and everything else.
        let overlay: Element<'_, Message> = if self.toasts.is_empty() {
            widgets::empty_stack_placeholder()
        } else {
            let mut toast_col = Column::new()
                .spacing(theme::SPACE_6)
                .align_x(Alignment::Center);
            for toast in &self.toasts {
                let (color, _bg) = match toast.kind {
                    ToastKind::Success => (theme::STATUS_SUCCESS, theme::BG_ELEVATED),
                    ToastKind::Warning => (theme::STATUS_WARNING, theme::BG_ELEVATED),
                    ToastKind::Error => (theme::STATUS_ERROR, theme::BG_ELEVATED),
                };
                let pill = container(text(&toast.message).size(theme::TEXT_12).color(color))
                    .padding([theme::PAD_6, theme::PAD_14])
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
                    // Clears the footer bar; with the bar hidden the stack keeps only
                    // the window's own inset.
                    bottom: if footer_hidden {
                        theme::PAD_16
                    } else {
                        FOOTER_HEIGHT + theme::SPACE_2
                    },
                    ..Default::default()
                })
                .align_bottom(Length::Fill)
                .into()
        };

        // The picked workspace's view is the default; a sidebar ticket from another
        // project opens the read-only foreign view instead, mutually exclusive with it.
        // `sync_workspace_surfaces` closes the picked one when its resolution is lost and
        // never touches the foreign one.
        let diff_overlay: Element<'_, Message> = if let Some(foreign) = &self.foreign_workspace {
            render_diff_modal(
                &self.foreign_diff_state,
                Some(foreign),
                Message::ForeignDiff,
            )
        } else if self.show_diff_modal {
            render_diff_modal(&self.diff_state, None, Message::DiffModal)
        } else {
            widgets::empty_stack_placeholder()
        };

        // The branch management dialog (small, centered).
        let branch_overlay: Element<'_, Message> = if self.git_state.is_modal_open() {
            let dialog = dialog::dialog_shell(self.git_state.view().map(Message::Git), 400.0, 24.0);
            widgets::modal_backdrop(dialog, Message::Git(git::GitMessage::CloseModal), 0.5)
        } else {
            widgets::empty_stack_placeholder()
        };

        let update_overlay: Element<'_, Message> = self.render_update_confirm();

        iced::widget::stack![body, diff_overlay, branch_overlay, update_overlay, overlay].into()
    }
}

/// Wrap dialog content in a modal overlay with a semi-transparent backdrop: 16px padding,
/// the standard dialog container style, centered at 80% width by a `FillPortion(1/8/1)`
/// row. The dialog rectangle claims its clicks ([`widgets::dialog_click_guard`]) so
/// interior clicks do not dismiss it, while the surrounding margin does.
fn modal_overlay<'a>(
    inner: impl Into<Element<'a, Message>>,
    on_close: Message,
) -> Element<'a, Message> {
    let dialog = container(inner)
        .height(Length::Fill)
        .padding(theme::PAD_16)
        .style(theme::dialog_container_style);

    let centered = row![
        Space::new().width(Length::FillPortion(1)), // 10% margin
        widgets::dialog_click_guard(dialog.width(Length::FillPortion(8))), // 80% content
        Space::new().width(Length::FillPortion(1)), // 10% margin
    ]
    .width(Length::Fill)
    .height(Length::Fill);

    iced::widget::stack([
        widgets::modal_backdrop_layer(on_close, 0.5),
        centered.into(),
    ])
    .into()
}

/// The diff modal (80% width, full height, centered).
///
/// `foreign_workspace` is `Some(name)` while it shows another project's commit: that view
/// is read-only and its Back button leaves it for the picked workspace, so the commit
/// header names the project — it must never be taken for the picked workspace's history.
fn render_diff_modal<'a>(
    diff_state: &'a diff::DiffState,
    foreign_workspace: Option<&'a str>,
    wrap: fn(diff::DiffMessage) -> Message,
) -> Element<'a, Message> {
    let viewing_commit = diff_state.is_viewing_commit();

    // Header: the commit message with its short hash for a historical commit, or
    // "Uncommitted changes" for the working tree.
    let header: Element<'_, Message> = if viewing_commit {
        let msg = diff_state
            .commit_message()
            .unwrap_or("(no commit message)")
            .to_string();
        let hash = diff_state.commit_short_hash().unwrap_or("???????");
        let subject = match foreign_workspace {
            Some(workspace) => format!("{hash} in workspace {workspace}"),
            None => hash.to_string(),
        };
        column![
            widgets::section_heading(msg),
            text(subject)
                .size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY),
        ]
        .spacing(theme::SPACE_2)
        .padding(iced::Padding {
            bottom: theme::PAD_12,
            ..iced::Padding::ZERO
        })
        .into()
    } else {
        column![
            widgets::section_heading("Uncommitted changes"),
            text("Working tree diff \u{2014} press Escape to close")
                .size(theme::TEXT_11)
                .color(theme::TEXT_SECONDARY),
        ]
        .spacing(theme::SPACE_4)
        .padding(iced::Padding {
            bottom: theme::PAD_12,
            ..iced::Padding::ZERO
        })
        .into()
    };

    let diff_content: Element<'_, diff::DiffMessage> = diff_state.view();
    let inner = column![header, diff_content.map(wrap)].spacing(0);

    modal_overlay(inner, Message::CloseDiffModal)
}

/// Ticket sidebar on the right of the Home page: every workspace's non-archived
/// tickets, grouped by phase — absent entirely while there is none to show
/// ([`board::BoardState::has_nothing_to_show`]), so the chat takes the whole width.
fn ticket_sidebar(board_state: &board::BoardState) -> Element<'_, Message> {
    let search_active = !board_state.search_query.text().is_empty();

    let search_input = widgets::single_line_editor(
        &board_state.search_query.buffer,
        "Search tickets…",
        false,
        Length::Fill,
        Some(iced::widget::Id::new("board_ticket_search")),
        |action| Message::Board(board::BoardMessage::SearchInputChanged(action)),
    );
    let clear_btn = widgets::icon_tooltip_button(
        text("×").size(theme::TEXT_14),
        "clear search",
        Some(Message::Board(board::BoardMessage::SearchCleared)),
        theme::PAD_4,
        theme::button_text,
        tooltip::Position::Top,
    );
    let mut search_row_children: Vec<Element<'_, Message>> = Vec::with_capacity(2);
    search_row_children.push(search_input);
    if search_active {
        search_row_children.push(clear_btn);
    }
    let search_row = Row::with_children(search_row_children)
        .spacing(theme::SPACE_4)
        .padding([0.0, theme::PAD_8])
        .align_y(Alignment::Center);

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
        .padding([theme::PAD_8, theme::PAD_4])
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::surface_container_style)
        .into()
}

fn section_hint(label: &str) -> Element<'_, Message> {
    column![
        Space::new().height(8),
        text(label)
            .size(theme::TEXT_12)
            .color(theme::TEXT_SECONDARY),
    ]
    .spacing(theme::SPACE_4)
    .padding(theme::PAD_8)
    .into()
}

/// The normal ticket list, partitioned into In Progress, Queued, Pending and Completed.
fn render_normal_ticket_list(board_state: &board::BoardState) -> Element<'_, Message> {
    // A failed list read must show in the body itself — otherwise a first failure
    // renders "Loading…" forever and a later one leaves a stale list looking current.
    if let Some(err) = board_state.load_state.error() {
        return widgets::error_banner(err);
    }

    let [in_progress, queued, pending, completed] =
        board::BoardState::board_sections(&board_state.tickets);

    let is_empty =
        in_progress.is_empty() && queued.is_empty() && pending.is_empty() && completed.is_empty();

    if !board_state.load_state.has_loaded() {
        section_hint("Loading…")
    } else if is_empty {
        section_hint("No tickets")
    } else {
        let mut groups = Column::new().spacing(theme::SPACE_8);
        if !in_progress.is_empty() {
            groups = groups.push(group_section("In Progress", &in_progress, board_state));
        }
        if !queued.is_empty() {
            groups = groups.push(group_section("Queued", &queued, board_state));
        }
        if !pending.is_empty() {
            groups = groups.push(group_section("Pending", &pending, board_state));
        }
        if !completed.is_empty() {
            groups = groups.push(group_section("Completed", &completed, board_state));
        }
        widgets::vscroll(groups)
    }
}

fn render_search_results(board_state: &board::BoardState) -> Element<'_, Message> {
    // A failed search must not render as "No matching tickets".
    if let Some(err) = board_state.search_error.as_deref() {
        return widgets::error_banner(err);
    }
    if board_state.search_results.is_empty() {
        section_hint("No matching tickets")
    } else {
        let mut cards = Column::new().spacing(theme::SPACE_2);
        for ticket in &board_state.search_results {
            cards = cards.push(board_state.render_ticket_card(ticket).map(Message::Board));
        }
        widgets::vscroll(cards)
    }
}

fn group_section<'a>(
    label: &'static str,
    tickets: &[&'a Ticket],
    board_state: &'a board::BoardState,
) -> Element<'a, Message> {
    let header = text(label)
        .size(theme::TEXT_11)
        .color(theme::TEXT_SECONDARY);

    let mut cards = Column::new().spacing(theme::SPACE_2);
    for ticket in tickets {
        cards = cards.push(board_state.render_ticket_card(ticket).map(Message::Board));
    }

    column![header, Space::new().height(4), cards]
        .spacing(0)
        .into()
}

impl Dashboard {
    pub fn subscription(&self) -> iced::Subscription<Message> {
        // Window events are subscribed from the start, before boot completes — a close request
        // included, which must reach the dashboard in every state (see `close_window`).
        // Resized carries the start-up width decision ([`StartupWidth`]), and Resized and Moved
        // keep last_size and last_position at the window's real geometry for the exit path to
        // persist (iced does not replay missed events). The platform's quit requests are
        // subscribed from the start for the same reason: it can ask before boot has finished.
        let always_on = iced::Subscription::batch([
            window::resize_events()
                .map(|(id, size)| Message::WindowEvent(id, window::Event::Resized(size))),
            window::events().filter_map(|(id, event)| match &event {
                window::Event::Moved(_) | window::Event::FileDropped(_) => {
                    Some(Message::WindowEvent(id, event))
                }
                // The window is created after this subscription registers, so its first
                // `Opened` arrives here: macOS hides the title strip, Windows sets the icons.
                #[cfg(any(target_os = "macos", target_os = "windows"))]
                window::Event::Opened { .. } => Some(Message::WindowOpened(id)),
                _ => None,
            }),
            window::close_requests().map(Message::CloseRequested),
            iced::Subscription::run(quit_requests_subscription),
            // The ways into the window from outside it, subscribed from the start like the quit
            // asks: the tray menu's quit ends the process as a platform quit does, and the way
            // back brings up even a booting window. A platform without them has no gestures.
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            iced::Subscription::run(window_requests_subscription),
        ]);
        if !self.ready {
            return always_on;
        }
        iced::Subscription::batch([
            always_on,
            keyboard::listen().filter_map(|event| {
                let (key, modifiers, physical_key) = parse_key_press(event)?;
                match dashboard_shortcut(&key, modifiers, physical_key)? {
                    DashboardShortcut::Escape => Some(Message::EscapePressed),
                    DashboardShortcut::Search => Some(Message::FocusSearch),
                    DashboardShortcut::Logs => Some(Message::OpenLogs),
                    DashboardShortcut::Sessions => Some(Message::OpenSessions),
                }
            }),
            self.shell_state.subscription().map(Message::Shell),
            self.logs_state.subscription().map(Message::Logs),
            // The editor's page is workspace-scoped: while nothing is resolved it gets no
            // subscription at all — no tick and no tree ping may act on the workspace it
            // still holds in memory (its tabs are deliberately kept).
            if self.page_available(Page::Editor) {
                self.editor_state
                    .subscription(self.overlay_modal_open())
                    .map(Message::Editor)
            } else {
                iced::Subscription::none()
            },
            self.home_state.subscription().map(Message::Home),
            // The Automations page and the Settings custom-tools area re-read their source
            // while shown; the tick is page-gated so a page nobody is looking at reads nothing.
            match self.page {
                Page::Automations => iced::time::every(common::POLLED_LIST_REFRESH_INTERVAL)
                    .map(|_| Message::Automations(automations::AutomationsMessage::Tick)),
                Page::Settings => {
                    iced::time::every(common::POLLED_LIST_REFRESH_INTERVAL).map(|_| {
                        Message::Settings(settings::SettingsMessage::CustomToolsMsg(
                            custom_tools::CustomToolsMessage::Tick,
                        ))
                    })
                }
                _ => iced::Subscription::none(),
            },
            iced::Subscription::run(shutdown_subscription),
            // Local git footer refresh, event-driven, when workspace files change.
            iced::Subscription::run(git_file_changes_subscription),
            // Pipeline auto-commits are ref-only changes the watcher misses.
            iced::Subscription::run(git_commit_subscription),
            // The standalone 60s remote-sync tick, which also recovers ref-only and
            // commit-only changes.
            iced::time::every(git::REMOTE_REFRESH_INTERVAL)
                .map(|_| Message::Git(git::GitMessage::PeriodicSyncTick)),
            // Board tickets from CDC deltas instead of a 1s full re-poll.
            iced::Subscription::run(board_change_subscription),
            // Workspaces change-stream subscription: a full map reload whenever a workspaces row
            // changes (or the stream lags / the baseline fires).
            iced::Subscription::run(workspaces_cdc_subscription),
            // Re-list the Settings users on a users/user_channels row change (or a lag —
            // a full re-list is its own recovery).
            iced::Subscription::run(users_cdc_subscription),
            iced::Subscription::run(user_channels_cdc_subscription),
            // Coalesces agent/non-agent registry, live transcript and voice-status
            // broadcasts into one re-render, driving Running Agents and the footer.
            iced::Subscription::run(runtime_change_subscription),
            // The diff state's own shortcuts and auto-refresh, only while its view is
            // open: the foreign view owns it while shown, so the two never both subscribe.
            if self.foreign_workspace.is_some() {
                self.foreign_diff_state
                    .subscription()
                    .map(Message::ForeignDiff)
            } else if self.show_diff_modal {
                self.diff_state.subscription().map(Message::DiffModal)
            } else {
                iced::Subscription::none()
            },
        ])
    }
}

/// Emits [`Message::Shutdown`] when the shutdown token fires (self-update restart,
/// SIGTERM/SIGINT, the end of a drain) and [`Message::DrainStarted`] when the drain window
/// begins — the token does not fire on the first signal, so input needs a separate signal to
/// be disabled while the drain runs.
fn shutdown_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    iced::stream::channel(1, |mut output: mpsc::Sender<Message>| async move {
        let token = crate::shutdown::shutdown_token();
        // A drain always precedes the token in the two-signal protocol, but the token can
        // fire without a drain on the force-cancel paths.
        tokio::select! {
            () = crate::shutdown::drain_wait() => {
                let _ = output.try_send(Message::DrainStarted);
            }
            () = token.cancelled() => {}
        }
        token.cancelled().await;
        let _ = output.try_send(Message::Shutdown);
    })
}

/// Record `route` as the exit route where the ask has wording of its own — a platform quit
/// passes its label, a close passes `None`, which leaves the record alone. Nothing is shown
/// for it (see `shutdown::EXIT_TRIGGER`), so it is set for accuracy, not reporting.
fn record_exit_route(route: Option<&'static str>) {
    if let Some(route) = route {
        crate::shutdown::record_exit_trigger(route);
    }
}

/// Turns an ordinary platform quit — one of the asks [`Message::QuitRequested`] names — into
/// that message, from the channel the quit source fills
/// ([`crate::shutdown::install_stop_request_sources`], before iced runs).
fn quit_requests_subscription() -> impl futures_util::Stream<Item = Message> {
    use futures_util::SinkExt;
    use iced::futures::channel::mpsc;
    iced::stream::channel(1, |mut output: mpsc::Sender<Message>| async move {
        let Some(mut requests) = crate::shutdown::take_quit_requests() else {
            return;
        };
        while requests.recv().await.is_some() {
            // Backpressure, not a dropped request: the next quit gesture during a drain is
            // what forces the exit, so it must not be lost to a full channel.
            if output.send(Message::QuitRequested).await.is_err() {
                return;
            }
        }
    })
}

/// Turns the ways into the window from outside it into their messages: a tray or Dock click
/// brings it up, the tray menu's item is an ordinary exit request. Source:
/// [`crate::window_requests`], filled before iced runs. The platform's handlers only enqueue,
/// so its menu tracking is never held up by the interface.
#[cfg(any(target_os = "macos", target_os = "windows"))]
fn window_requests_subscription() -> impl futures_util::Stream<Item = Message> {
    use futures_util::SinkExt;
    use iced::futures::channel::mpsc;
    iced::stream::channel(1, |mut output: mpsc::Sender<Message>| async move {
        let Some(mut requests) = crate::window_requests::take() else {
            return;
        };
        while let Some(request) = requests.recv().await {
            // Backpressure, not a dropped gesture: the next quit during a drain is what
            // forces the exit, so it must not be lost to a full channel.
            if output
                .send(Message::WindowRequested(request))
                .await
                .is_err()
            {
                return;
            }
        }
    })
}

/// Emits [`Message::RuntimeChanged`] (coalesced) when a source the GUI renders directly
/// changes: the agent registries, live transcript, voice status, or the update availability
/// behind the update entry. Source warmed in `main` (see [`init_board_change_tx`]). A burst of
/// per-message and per-tool-call events collapses into one trailing message per
/// [`RUNTIME_REFRESH_COALESCE`] window — only "something changed" matters for a re-render.
fn runtime_change_subscription() -> impl futures_util::Stream<Item = Message> {
    common::coalesced_broadcast_producer(
        128,
        &crate::runtime_events::RUNTIME_EVENTS,
        RUNTIME_REFRESH_COALESCE,
        || Message::RuntimeChanged,
    )
}

/// Emits [`Message::Git`] with [`GitMessage::FileChanged`] when workspace files change, from
/// the watcher thread's non-blocking broadcast ([`init_board_change_tx`]).
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

/// Emits [`Message::Git`] with [`GitMessage::PipelineCommit`]: a ref-only change the watcher
/// never reports, going through [`GitState`]'s path-matched handler.
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

/// Emits [`BoardMessage::TicketChanged`] CDC deltas, or [`BoardMessage::BoardRefreshNeeded`]
/// after a lag that lost one. Source: [`crate::db::cdc`]'s tickets sender, warmed by
/// [`crate::gui::init_board_change_tx`].
fn board_change_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    common::broadcast_stream_producer(
        128,
        crate::db::cdc::ticket_sender_lock(),
        |output: &mut mpsc::Sender<Message>, event: Option<crate::db::cdc::ChangeEvent>| {
            Box::pin(async move {
                // Awaited send (backpressure): a full or lagged channel triggers a refresh
                // rather than silently dropping a delta.
                let msg = Message::Board(match event {
                    Some(ev) => board::BoardMessage::TicketChanged(Box::new(ev)),
                    None => board::BoardMessage::BoardRefreshNeeded,
                });
                let _ = futures_util::SinkExt::send(output, msg).await;
            })
        },
    )
}

/// Emits [`Message::WorkspacesCdcChanged`] for a full map reload, from
/// [`crate::db::cdc`]'s workspaces sender ([`init_board_change_tx`]).
///
/// It opens with one synthetic BASELINE event, so a full read happens at stream start. The
/// subscription lives only in the post-boot `ready` branch, so the baseline lands after `Boot`
/// set `ready` and is never swallowed by the pre-boot no-op. A delta and a lag both mean a
/// full reload, which is its own lag recovery.
fn workspaces_cdc_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    futures_util::stream::once(async { Message::WorkspacesCdcChanged }).chain(
        common::broadcast_stream_producer(
            128,
            crate::db::cdc::workspace_sender_lock(),
            |output: &mut mpsc::Sender<Message>, _event: Option<crate::db::cdc::ChangeEvent>| {
                Box::pin(async move {
                    // Awaited send (backpressure): a dropped reload trigger would leave
                    // the sidebar stale until the next event.
                    let _ =
                        futures_util::SinkExt::send(output, Message::WorkspacesCdcChanged).await;
                })
            },
        ),
    )
}

/// Emits [`Message::UsersCdcChanged`] when a users row changes or the stream lags: the re-list
/// and the DB read-back that follow are each their own recovery.
///
/// It opens with one synthetic BASELINE event, like [`workspaces_cdc_subscription`]: a switch
/// written between the boot selection read and this stream's first poll would otherwise be
/// missed for the rest of the run. Source: [`crate::db::cdc`]'s users sender, warmed in `main`.
fn users_cdc_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    futures_util::stream::once(async { Message::UsersCdcChanged }).chain(
        common::broadcast_stream_producer(
            128,
            crate::db::cdc::users_sender_lock(),
            |output: &mut mpsc::Sender<Message>, _event: Option<crate::db::cdc::ChangeEvent>| {
                Box::pin(async move {
                    // Awaited send: a dropped trigger would leave the list stale
                    // until the next event (same rationale as the board stream).
                    let _ = futures_util::SinkExt::send(output, Message::UsersCdcChanged).await;
                })
            },
        ),
    )
}

/// Emits [`Message::UsersCdcChanged`] when a user_channels row changes or the stream lags (a
/// bind/unbind updates it). Source: [`crate::db::cdc`]'s user_channels sender, warmed in `main`.
fn user_channels_cdc_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    common::broadcast_stream_producer(
        128,
        crate::db::cdc::user_channels_sender_lock(),
        |output: &mut mpsc::Sender<Message>, _event: Option<crate::db::cdc::ChangeEvent>| {
            Box::pin(async move {
                // Awaited send: a dropped trigger would leave the list stale until the
                // next event (same rationale as the board stream).
                let _ = futures_util::SinkExt::send(output, Message::UsersCdcChanged).await;
            })
        },
    )
}

/// A running indicator — a role's agent count or the in-flight call counter — ready for either
/// place it is drawn, with the wording its tooltip carries.
struct RunningIndicator {
    content: Element<'static, Message>,
    label: String,
}

impl RunningIndicator {
    /// In the footer's right-hand indicator row, tooltip above.
    fn footer(self) -> Element<'static, Message> {
        widgets::icon_tooltip_button(
            container(self.content).padding([0.0, theme::PAD_3]),
            self.label,
            Some(Message::Navigation(Page::RunningAgents)),
            0,
            theme::button_text,
            tooltip::Position::Top,
        )
    }

    /// A full-bleed rail entry, tooltip to the side like every other rail item.
    fn rail(self) -> Element<'static, Message> {
        Dashboard::sidebar_rail_item(
            self.content,
            self.label,
            Some(Message::Navigation(Page::RunningAgents)),
            RailEntry::Item,
        )
    }
}

/// A [`Page`]'s Lucide icon, unstyled so each call site sizes and colours it. An exhaustive
/// match, so a new `Page` variant fails to compile until it has an icon.
fn page_icon(page: Page) -> iced::widget::Text<'static, iced::Theme, iced::Renderer> {
    match page {
        Page::Home => lucide::layout_dashboard::<iced::Theme, iced::Renderer>(),
        Page::Editor => lucide::pencil_line::<iced::Theme, iced::Renderer>(),
        Page::Shell => lucide::terminal::<iced::Theme, iced::Renderer>(),
        Page::Sessions => lucide::scroll_text::<iced::Theme, iced::Renderer>(),
        Page::Automations => lucide::workflow::<iced::Theme, iced::Renderer>(),
        Page::Logs => lucide::activity::<iced::Theme, iced::Renderer>(),
        Page::Settings => lucide::settings::<iced::Theme, iced::Renderer>(),
        Page::RunningAgents => lucide::radar::<iced::Theme, iced::Renderer>(),
    }
}

/// Which vertical padding a rail entry takes. Every entry spans the rail's whole width, its
/// highlight and hit area reaching both sides and, at the rail's ends, the layout's edges; the
/// window control is the one entry that keeps [`theme::PAD_2`] beside its glyph.
#[derive(Clone, Copy)]
enum RailEntry {
    Item,
    AtBottom,
    /// The window's own control: half-height, so it stays a narrow strip at the rail's top.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    WindowControl,
}

impl RailEntry {
    /// [`theme::PAD_14`] on the rail's outer side at its end instead of [`theme::PAD_10`];
    /// [`theme::PAD_6`] for a window control.
    fn content_padding(self) -> iced::Padding {
        let (top, bottom) = match self {
            Self::AtBottom => (theme::PAD_10, theme::PAD_14),
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            Self::WindowControl => (theme::PAD_6, theme::PAD_6),
            Self::Item => (theme::PAD_10, theme::PAD_10),
        };
        iced::Padding {
            top,
            bottom,
            ..iced::Padding::default()
        }
    }
}

/// Which end of the rail a nav column sits at: only the bottom one reaches the layout's edge,
/// so only there does the last entry take [`RailEntry::AtBottom`].
#[derive(Clone, Copy)]
enum NavColumn {
    Top,
    Bottom,
}

/// A window action the app performs for itself, the platform's title bar being hidden on macOS
/// and Windows ([`Dashboard::sidebar_view`]).
#[cfg(any(target_os = "macos", target_os = "windows"))]
#[derive(Debug, Clone, Copy)]
pub(crate) enum WindowAction {
    /// The window follows the pointer: a press no rail button claimed starts this.
    Drag,
    /// The platform's own close, which on these platforms hides the window
    /// ([`Dashboard::close_window`]).
    Close,
}

impl Dashboard {
    /// The rail: the window's own chrome where the title bar is hidden, the top nav icons, the
    /// voice status while the footer is hidden, the stretch in between
    /// ([`Self::render_sidebar_mid`]) and the bottom nav icons. Its empty space drags the window.
    fn sidebar_view(&self, footer_hidden: bool) -> Element<'_, Message> {
        let mut col = Column::new().spacing(theme::SPACE_2);
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        {
            col = col.push(Self::rail_window_control());
        }
        col = col.push(self.render_sidebar_nav());
        if footer_hidden {
            col = col.push(Self::render_sidebar_voice());
        }
        let rail = container(
            col.push(Self::render_sidebar_mid(footer_hidden))
                .push(self.render_sidebar_bottom_nav()),
        )
        .width(Length::Fixed(SIDEBAR_RAIL_WIDTH))
        .height(Length::Fill)
        .style(theme::surface_container_style);
        // The rail drags the window: the mouse area passes a press to the entry under it
        // first, so buttons keep their clicks, and holds no state, so a drag whose release
        // the platform keeps leaves nothing stuck.
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        return mouse_area(rail)
            .on_press(Message::WindowAction(WindowAction::Drag))
            .into();
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        rail.into()
    }

    /// The rail's stretch between the nav icons and the bottom nav. With the footer hidden it
    /// carries the running-agent indicators, bottom-aligned above the bottom nav, filling the
    /// free height and scrolling in its own area so the nav entries keep their size however
    /// many roles run. Otherwise it is the plain spacer keeping the two nav groups apart.
    fn render_sidebar_mid(footer_hidden: bool) -> Element<'static, Message> {
        if footer_hidden && let Some(agents) = Self::render_sidebar_agents() {
            return container(agents)
                .width(Length::Fill)
                .align_bottom(Length::Fill)
                .into();
        }
        Space::new().height(Length::Fill).into()
    }

    /// A rail entry: content centred in the entry's full width, tooltip to the side — iced
    /// snaps Top tooltips into the viewport, overlapping the topmost rail entry. `on_press:
    /// None` makes a static entry (the voice status, which only reports).
    fn sidebar_rail_item(
        content: Element<'static, Message>,
        label: impl Into<Cow<'static, str>>,
        on_press: Option<Message>,
        entry: RailEntry,
    ) -> Element<'static, Message> {
        tooltip(
            button(
                // The container's padding gives the button a large hit area and keeps the
                // icon's place.
                container(content)
                    .width(Length::Fill)
                    .center_x(Length::Fill)
                    .padding(entry.content_padding()),
            )
            .width(Length::Fill)
            .padding(0)
            .style(theme::button_text)
            .on_press_maybe(on_press),
            text(label.into()).size(theme::TEXT_11),
            tooltip::Position::Right,
        )
        .style(theme::tooltip_style)
        .into()
    }

    /// The nav-icon column shared by the top and bottom navs: a full-width icon button per page,
    /// accent-coloured when active, absent when [`Self::page_available`] refuses it.
    fn sidebar_nav_column(&self, pages: &[Page], column: NavColumn) -> Column<'_, Message> {
        // Only the column at the rail's bottom end reaches the layout's edge, so only there
        // is an entry marked.
        let edge = match column {
            NavColumn::Bottom => pages.iter().rposition(|page| self.page_available(*page)),
            NavColumn::Top => None,
        };
        let mut col = Column::new().spacing(theme::SPACE_2);
        for (index, page) in pages.iter().enumerate() {
            if !self.page_available(*page) {
                continue;
            }
            let entry = if Some(index) == edge {
                RailEntry::AtBottom
            } else {
                RailEntry::Item
            };
            let color = if self.page == *page {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            };
            col = col.push(Self::sidebar_rail_item(
                page_icon(*page).size(RAIL_ICON_SIZE).color(color).into(),
                page.label(),
                Some(Message::Navigation(*page)),
                entry,
            ));
        }
        col
    }

    /// A view with no rail — the start-up splash, a failed boot — with the window's own chrome
    /// above it: its control at the top, the whole surface draggable.
    fn window_chrome(view: Element<'_, Message>) -> Element<'_, Message> {
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        return view;
        #[cfg(any(target_os = "macos", target_os = "windows"))]
        mouse_area(column![Self::chrome_window_control(), view])
            .on_press(Message::WindowAction(WindowAction::Drag))
            .into()
    }

    /// The window's one control, built like the rail's entries. `glyph` is what the press does
    /// where the control is drawn: the eye-off that hides the window on the rail
    /// ([`Self::close_window`]), the cross that quits from the surfaces drawn before the app
    /// runs. Minimizing is the platform's own (the taskbar's menu on Windows), and the tray
    /// brings a minimized window back like a hidden one.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn window_control(
        glyph: iced::widget::Text<'static, iced::Theme, iced::Renderer>,
    ) -> Element<'static, Message> {
        Self::sidebar_rail_item(
            glyph
                .size(theme::TEXT_14)
                .color(theme::TEXT_SECONDARY)
                .into(),
            "close window",
            Some(Message::WindowAction(WindowAction::Close)),
            RailEntry::WindowControl,
        )
    }

    /// The rail's placement, with the hide glyph: padding at its sides and below and none above,
    /// so the control meets the window's top edge and the system's rounded corner cuts its top
    /// border. That edge is the platform's: the rows below it are the resize band, where a press
    /// resizes the window instead of reaching the control.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn rail_window_control() -> Element<'static, Message> {
        container(Self::window_control(lucide::eye_off()))
            .width(Length::Fixed(SIDEBAR_RAIL_WIDTH))
            .padding(iced::Padding {
                top: 0.0,
                bottom: RAIL_WINDOW_CONTROL_GAP,
                left: theme::PAD_2,
                right: theme::PAD_2,
            })
            .into()
    }

    /// The rail-less placement — the start-up splash, a failed boot — with the cross: the app
    /// does not run there, so the press quits. Padding on every side, room for the rounded corner.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn chrome_window_control() -> Element<'static, Message> {
        container(Self::window_control(lucide::x()))
            .width(Length::Fixed(SIDEBAR_RAIL_WIDTH))
            .padding([WINDOW_CONTROL_PADDING, theme::PAD_2])
            .into()
    }

    /// The top nav icons ([`Page::sidebar_pages`]). Running Agents is not here — only the
    /// running-agent indicators reach it, in the footer bar or, with the footer hidden, in the
    /// rail ([`Self::render_sidebar_agents`]).
    fn render_sidebar_nav(&self) -> Element<'_, Message> {
        self.sidebar_nav_column(Page::sidebar_pages(), NavColumn::Top)
            .into()
    }

    /// The bottom nav icons ([`Page::sidebar_bottom_pages`]), with the self-update entry
    /// directly above them — its only home.
    fn render_sidebar_bottom_nav(&self) -> Element<'_, Message> {
        let mut col = Column::new().spacing(theme::SPACE_2);
        if let Some(entry) = Self::render_sidebar_update() {
            col = col.push(entry);
        }
        col = col.push(self.sidebar_nav_column(Page::sidebar_bottom_pages(), NavColumn::Bottom));
        col.into()
    }

    /// The rail's voice status, standing in for the footer's label while the footer is hidden: the
    /// status glyph ([`Self::voice_status_icon`]) with the bar's wording as its tooltip. The bar's
    /// resting label is left out — an idle listener is the normal condition — and an empty slot
    /// keeps its place, so no status change rebuilds the column below it.
    fn render_sidebar_voice() -> Element<'static, Message> {
        let status = crate::audio::voice::get_status();
        if matches!(status, VoiceStatus::Listening) {
            return Space::new().height(0).into();
        }
        let Some(label) = Self::voice_status_label(&status) else {
            return Space::new().height(0).into();
        };
        Self::sidebar_rail_item(
            Self::voice_status_icon(&status)
                .size(RAIL_ICON_SIZE)
                .color(theme::TEXT_MUTED)
                .into(),
            label,
            None,
            RailEntry::Item,
        )
    }

    /// The rail's running-agent indicators — one per role in the footer's order, then the
    /// in-flight non-agent call counter — as a column that scrolls once the rail cannot show
    /// them all. `None` when nothing runs, exactly as in the footer.
    fn render_sidebar_agents() -> Option<Element<'static, Message>> {
        let mut items: Vec<Element<'static, Message>> = Self::running_roles()
            .into_iter()
            .map(|(role, count)| Self::running_role_indicator(role, count, theme::TEXT_11).rail())
            .collect();
        if let Some(calls) = Self::non_agent_calls(theme::TEXT_11) {
            items.push(calls.rail());
        }
        if items.is_empty() {
            return None;
        }
        // A scrollbar would be painted beside these entries; a half-visible row
        // already shows there is more.
        Some(
            scrollable(Column::with_children(items).spacing(theme::SPACE_2))
                .direction(theme::hidden_vertical_scrollbar())
                .height(Length::Shrink)
                .width(Length::Fill)
                .into(),
        )
    }

    fn render_maintainer_toggle(&self) -> Element<'_, Message> {
        let enabled = self.maintenance_enabled();
        let icon: Element<'_, Message> = theme::role_icon(&Role::Maintainer)
            .size(theme::TEXT_24)
            .color(if enabled {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            })
            .into();
        widgets::icon_tooltip_button(
            icon,
            if enabled {
                "stop maintenance"
            } else {
                "start maintenance"
            },
            Some(Message::Toggle(ToggleKind::Maintenance)),
            theme::PAD_3,
            theme::button_text,
            tooltip::Position::Top,
        )
    }

    fn render_pause_toggle(&self) -> Element<'_, Message> {
        let paused = self.paused();
        let icon: Element<'_, Message> = if paused {
            lucide::play::<iced::Theme, iced::Renderer>()
                .size(24)
                .color(theme::ACCENT)
                .into()
        } else {
            lucide::pause::<iced::Theme, iced::Renderer>()
                .size(24)
                .color(theme::TEXT_MUTED)
                .into()
        };
        let tooltip_text = if paused {
            "Resume pipeline"
        } else {
            "Pause pipeline"
        };
        widgets::icon_tooltip_button(
            icon,
            tooltip_text,
            Some(Message::Toggle(ToggleKind::Pause)),
            theme::PAD_3,
            theme::button_text,
            tooltip::Position::Top,
        )
    }

    /// The failure the footer picker must show in place of its dropdown: a failed map read
    /// that left nothing to pick. A read that failed but kept a usable map keeps the working
    /// picker — the Settings list shows that failure.
    fn picker_failure<'a>(&'a self, options: &[widgets::PickOption]) -> Option<&'a str> {
        if options.is_empty() {
            self.workspaces_error.as_deref()
        } else {
            None
        }
    }

    /// Footer workspace picker: managed workspaces only, persisted by
    /// [`Self::select_workspace`]. It is offered even for a single workspace, since this is the
    /// only place the admin chooses one. A failed read with nothing to pick leaves a static,
    /// non-selectable "Workspaces unavailable" pill in the same footprint. `None` when no
    /// managed workspace exists.
    fn render_workspace_picker(&self) -> Option<Element<'_, Message>> {
        // The error text lives only in the tooltip — never as an option of the dropdown.
        let options = managed_workspace_options(&self.workspaces);
        if let Some(err) = self.picker_failure(&options) {
            return Some(widgets::tooltip_hint(
                container(
                    text("Workspaces unavailable")
                        .size(theme::TEXT_14)
                        .color(theme::STATUS_ERROR),
                )
                .width(Length::Fixed(140.0))
                .padding([theme::PAD_4, theme::PAD_8])
                .style(theme::pill_style(theme::STATUS_ERROR.scale_alpha(0.08))),
                err,
            ));
        }
        if options.is_empty() {
            // An inert placeholder would suggest a choice that isn't there.
            return None;
        }
        // A dangling persisted value renders no highlight; a personal selection renders the
        // placeholder.
        let selected = self
            .selected_workspace_name
            .as_deref()
            .and_then(|name| options.iter().find(|o| o.value == name).cloned());
        Some(widgets::tooltip_hint(
            pick_list(options, selected, |opt| Message::WorkspacePick(opt.value))
                .placeholder("Select workspace")
                .style(theme::pick_list_style)
                .menu_style(theme::pick_list_menu_style)
                .padding([theme::PAD_4, theme::PAD_8])
                .width(Length::Fixed(140.0)),
            "Active workspace",
        ))
    }

    /// The self-update confirmation modal (opened by [`Message::UpdateBot`], confirmed by
    /// [`Message::ConfirmUpdate`]), or a type-stable placeholder when closed.
    ///
    /// Its text is the real sequence: the update runs in the background while the system keeps
    /// working, then in-flight work drains, databases are checkpointed, and the app restarts
    /// and resumes. The window closing is the success signal.
    fn render_update_confirm(&self) -> Element<'_, Message> {
        if !self.show_update_confirm {
            // The Stack type stays stable across open/close (see `empty_stack_placeholder`).
            return iced::widget::stack([widgets::empty_stack_placeholder()]).into();
        }
        let mode_desc = match crate::self_update::update_mode() {
            crate::self_update::UpdateMode::Downloaded => {
                "The ready-made file for this system is downloaded in the background \
                 while the system keeps working, and put at the standard location: it \
                 replaces the file this copy runs from when that is already where it \
                 sits, and for a copy running from anywhere else it is placed there \
                 while the file it came from is taken away. Nothing is paused during \
                 it.\n\n\
                 When it is in place, current in-flight work is drained (up to \
                 ~10 min), databases are checkpointed, and the app restarts from \
                 the standard location, automatically resuming work afterwards. \
                 The window closing signals the restart."
            }
            crate::self_update::UpdateMode::SourceTree => {
                "The new version is built from this working tree in the \
                 background while the system keeps working — the build can take \
                 10–60 minutes and nothing is paused during it.\n\n\
                 When the build finishes, current in-flight work is drained \
                 (up to ~10 min), databases are checkpointed, and the app \
                 restarts, automatically resuming work afterwards. The window \
                 closing signals the restart."
            }
        };
        widgets::modal_backdrop(
            dialog::confirm_dialog(
                dialog::dialog_title("Update MahBot?"),
                dialog::dialog_body([mode_desc]),
                [
                    dialog::DialogAction::secondary("Cancel", Message::CancelUpdate),
                    dialog::DialogAction::secondary("Update", Message::ConfirmUpdate),
                ],
            ),
            Message::CancelUpdate,
            0.5,
        )
    }

    /// The rail's self-update entry, directly above the bottom nav (where Settings lives): the
    /// refresh glyph at the rail's own glyph size, its tooltip and its press (none while an
    /// update is in flight), or `None` when no update is offered.
    ///
    /// Visibility comes entirely from the shared availability cache, with no platform gate: a
    /// downloaded copy puts this system's ready-made file at the standard location everywhere,
    /// and replacing a file in use is the platform's own rename-aside.
    fn render_sidebar_update() -> Option<Element<'static, Message>> {
        let availability = crate::self_update::update_availability();
        // Shown while an update is in flight even if `available` was transiently clobbered,
        // so "Updating…" is never hidden mid-update.
        if !availability.available && !availability.in_progress {
            return None;
        }
        let (color, tooltip, on_press) = if availability.in_progress {
            (theme::TEXT_FAINT, "Updating…".to_string(), None)
        } else {
            let tooltip = match crate::self_update::update_latest() {
                Some(v) => format!("Update MahBot to v{v}"),
                None => "Update MahBot".to_string(),
            };
            (theme::ACCENT, tooltip, Some(Message::UpdateBot))
        };
        let glyph = lucide::refresh_cw::<iced::Theme, iced::Renderer>()
            .size(RAIL_ICON_SIZE)
            .color(color)
            .into();
        Some(Self::sidebar_rail_item(
            glyph,
            tooltip,
            on_press,
            RailEntry::Item,
        ))
    }

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

    fn render_git_branch(&self) -> Option<Element<'_, Message>> {
        let b = self.git_state.current_branch()?;
        let truncated = crate::util::truncate(b, 20);
        let branch_content = row![
            lucide::git_branch::<iced::Theme, iced::Renderer>()
                .size(theme::TEXT_24)
                .color(theme::ACCENT),
            text(truncated).size(theme::TEXT_16).color(theme::ACCENT),
        ]
        .spacing(theme::SPACE_6)
        .align_y(Alignment::Center);
        Some(widgets::icon_tooltip_button(
            branch_content,
            "active branch",
            Some(Message::Git(git::GitMessage::OpenModal)),
            theme::PAD_3,
            theme::button_text,
            tooltip::Position::Top,
        ))
    }

    fn sync_metric(
        icon: iced::widget::text::Text<'_, iced::Theme, iced::Renderer>,
        count: usize,
    ) -> Element<'_, Message> {
        Row::with_children(vec![
            icon.size(theme::TEXT_16)
                .color(theme::TEXT_SECONDARY)
                .into(),
            text(count.to_string())
                .size(theme::TEXT_16)
                .color(theme::TEXT_SECONDARY)
                .into(),
        ])
        .spacing(theme::SPACE_2)
        .align_y(Alignment::Center)
        .into()
    }

    /// The git sync control: the commits a push would carry (destination in the tooltip) and the
    /// commits a pull would take from the branch's update source.
    ///
    /// Shown whenever there is anything to push or pull — a pushable branch is never hidden
    /// because its upstream is fetch-only or absent, and a destination that could not be asked
    /// reports the commits it carries — and whenever no remote can take the branch, so it can
    /// say why; a branch with somewhere to push and nothing to move hides it. The pull metric
    /// appears only when pressing would pull, since a publish skips the update step. Icons are
    /// lucide at 16px, the number text's size, so all elements share a baseline.
    fn render_git_sync(&self) -> Option<Element<'_, Message>> {
        let state = self.git_state.remote_state()?;
        if !state.has_sync_work() {
            return None;
        }
        let publish = matches!(
            &state.push,
            GitPushDestination::Target(target) if target.publish == Some(true)
        );
        let mut parts: Vec<Element<'_, Message>> = Vec::new();
        let tooltip_text = match &state.push {
            GitPushDestination::Target(target) => {
                let icon = if publish {
                    lucide::git_branch_plus::<iced::Theme, iced::Renderer>()
                } else {
                    lucide::arrow_up::<iced::Theme, iced::Renderer>()
                };
                if target.commits > 0 {
                    parts.push(Self::sync_metric(icon, target.commits));
                }
                if publish {
                    format!(
                        "Publish a new branch to {} ({} commits): it has no such branch yet, \
                         so nothing is pulled",
                        target.remote, target.commits
                    )
                } else if target.publish.is_none() {
                    format!(
                        "Push {} commits to {}: it could not be reached, so whether it already \
                         carries this branch is unknown",
                        target.commits, target.remote
                    )
                } else if target.commits == 0 {
                    format!(
                        "Pull {} from its upstream; nothing to push to {}",
                        state.behind, target.remote
                    )
                } else if state.behind == 0 {
                    format!("Push {} to {}", target.commits, target.remote)
                } else {
                    format!(
                        "Push {} to {} and pull {} from its upstream",
                        target.commits, target.remote, state.behind
                    )
                }
            }
            GitPushDestination::Nowhere { reason } => {
                parts.push(
                    lucide::triangle_alert::<iced::Theme, iced::Renderer>()
                        .size(theme::TEXT_16)
                        .color(theme::STATUS_WARNING)
                        .into(),
                );
                parts.push(
                    text("nowhere to push")
                        .size(theme::TEXT_16)
                        .color(theme::TEXT_SECONDARY)
                        .into(),
                );
                let why = git::nowhere_message(reason);
                if state.behind > 0 {
                    format!("{why}. Pull {} from its upstream", state.behind)
                } else {
                    why
                }
            }
        };
        // The spacer separates the push count from the pull count; with nothing to push the
        // row starts at the pull metric, so it must not lead.
        if state.behind > 0 && !publish {
            if !parts.is_empty() {
                parts.push(Space::new().width(theme::SPACE_8).into());
            }
            parts.push(Self::sync_metric(
                lucide::arrow_down::<iced::Theme, iced::Renderer>(),
                state.behind,
            ));
        }
        Some(widgets::icon_tooltip_button(
            Row::with_children(parts)
                .spacing(theme::SPACE_2)
                .align_y(Alignment::Center),
            tooltip_text,
            if self.git_state.is_syncing() {
                None
            } else {
                Some(Message::Git(git::GitMessage::Sync))
            },
            theme::PAD_3,
            theme::button_text,
            tooltip::Position::Top,
        ))
    }

    /// The git diff stats button (+X/−Y, clickable -> diff modal), `None` when there are no
    /// changes to report (nor any oversized/binary untracked file or cap omission).
    fn render_git_diff_stats(&self) -> Option<Element<'_, Message>> {
        let stats = self.git_state.diff_stats()?;
        let tooltip_text: Cow<'static, str> = stats.omitted_note().map_or_else(
            || Cow::Borrowed("uncommitted changes"),
            |note| Cow::Owned(format!("uncommitted changes \u{2014} {note}")),
        );
        Some(widgets::icon_tooltip_button(
            widgets::git_footer_stats::<Message>(
                stats.added,
                stats.removed,
                stats.huge_binary_file_count,
                stats.omitted_file_count,
                15.0,
            )?,
            tooltip_text,
            Some(Message::OpenDiffModal(None)),
            theme::PAD_3,
            theme::button_text,
            tooltip::Position::Top,
        ))
    }

    /// The git block: divider, branch, sync, diff stats. `None` while nothing is resolved —
    /// the path [`Self::sync_workspace_surfaces`] owns *is* the resolution.
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
                .spacing(theme::SPACE_6)
                .align_y(Alignment::Center)
                .into(),
        )
    }

    /// Roles with agents running and how many of each, in role order — the one grouping every
    /// running-agent indicator is built from.
    fn running_roles() -> Vec<(Role, usize)> {
        let handles = crate::agent::registry::AGENT_REGISTRY.list();
        let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
        for h in &handles {
            *counts.entry(h.role.as_str()).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .map(|(role, count)| (role.parse().unwrap_or(Role::Engineer), count))
            .collect()
    }

    /// One role's indicator: its icon in its badge colour, plus `×N` once more than one agent
    /// of the role runs. `count_size` sizes that count — the rail has less width than the footer.
    fn running_role_indicator(role: Role, count: usize, count_size: f32) -> RunningIndicator {
        let (color, _bg) = theme::role_badge_color_for(&role);
        let icon = theme::role_icon(&role).size(theme::TEXT_24).color(color);
        let content: Element<'static, Message> = if count > 1 {
            row![
                icon,
                text(format!("×{count}")).size(count_size).color(color)
            ]
            .spacing(theme::SPACE_3)
            .align_y(Alignment::Center)
            .into()
        } else {
            icon.into()
        };
        RunningIndicator {
            content,
            label: role.display_label().into(),
        }
    }

    /// The in-flight non-agent LLM call counter, `count_size` as in
    /// [`Self::running_role_indicator`]; `None` when none are in flight.
    fn non_agent_calls(count_size: f32) -> Option<RunningIndicator> {
        let handles = crate::agent::registry::NON_AGENT_CALLS.list();
        if handles.is_empty() {
            return None;
        }
        let color = theme::ACCENT;
        let content = row![
            lucide::zap::<iced::Theme, iced::Renderer>()
                .size(theme::TEXT_24)
                .color(color),
            text(format!("×{}", handles.len()))
                .size(count_size)
                .color(color),
        ]
        .spacing(theme::SPACE_3)
        .align_y(Alignment::Center)
        .into();
        // The tooltip lists the in-flight kinds as human labels — raw snake_case never
        // surfaces. Map to labels FIRST, then dedup: two unknown kinds share the generic
        // fallback, so deduping raw kinds first could repeat it.
        let mut labels: Vec<&str> = handles
            .iter()
            .map(|h| crate::agent::registry::call_kind_label(h.kind))
            .collect();
        labels.sort_unstable();
        labels.dedup();
        Some(RunningIndicator {
            content,
            label: labels.join(", "),
        })
    }

    fn render_active_agents() -> Option<Element<'static, Message>> {
        let icons: Vec<Element<'static, Message>> = Self::running_roles()
            .into_iter()
            .map(|(role, count)| Self::running_role_indicator(role, count, theme::TEXT_14).footer())
            .collect();
        if icons.is_empty() {
            return None;
        }
        Some(
            Row::with_children(icons)
                .spacing(theme::SPACE_12)
                .align_y(Alignment::Center)
                .into(),
        )
    }

    fn render_non_agent_calls() -> Option<Element<'static, Message>> {
        Some(Self::non_agent_calls(theme::TEXT_14)?.footer())
    }

    /// The wording for a [`VoiceStatus`], shared by the footer's label and the rail's tooltip,
    /// without the glyph [`Self::voice_status_icon`] draws. `None` for a plain voice-off, which
    /// both surfaces hide; [`VoiceStatus::ReenrollRequired`] keeps its label even though the toggle
    /// is off, because it names the action the owner has to take.
    fn voice_status_label(status: &VoiceStatus) -> Option<String> {
        Some(match status {
            VoiceStatus::Disabled => return None,
            VoiceStatus::Listening => "Listening".into(),
            VoiceStatus::LoadingModels => "Loading…".into(),
            VoiceStatus::ModelError => "⚠ Model error".into(),
            VoiceStatus::RecordingWakeWord | VoiceStatus::RecordingButton => "Recording…".into(),
            VoiceStatus::Transcribing => "Transcribing…".into(),
            VoiceStatus::MicPermissionDenied => "No mic access".into(),
            VoiceStatus::MicDisconnected => "Mic disconnected".into(),
            VoiceStatus::Enrolling { sample, total, .. } => format!("Enrolling {sample}/{total}"),
            VoiceStatus::ListeningDuringEnrollment { sample, total } => {
                format!("Listen… {sample}/{total}")
            }
            VoiceStatus::WaitingForSilenceDuringEnrollment { sample, total } => {
                format!("Wait… {sample}/{total}")
            }
            VoiceStatus::EnrollingNegatives {
                accumulated_secs,
                target_secs,
                ..
            } => format!("Collecting negatives {accumulated_secs}s/{target_secs}s"),
            VoiceStatus::Enrolled => "✅ Enrolled".into(),
            VoiceStatus::ReenrollRequired => "Enrol again".into(),
            VoiceStatus::Error(msg) => format!("Error: {msg}"),
        })
    }

    /// The footer's compact voice status: the glyph in front of its wording. A
    /// [`VoiceStatus::Error`] shows the actual error string.
    fn render_voice_status() -> Element<'static, Message> {
        let status = crate::audio::voice::get_status();
        let Some(label) = Self::voice_status_label(&status) else {
            return Space::new().width(0).into();
        };
        footer_status_label(label, Some(Self::voice_status_icon(&status)))
    }

    /// The glyph in front of a [`VoiceStatus`]: `ear`, or `pen-line` while transcribing.
    fn voice_status_icon(
        status: &VoiceStatus,
    ) -> iced::widget::text::Text<'static, iced::Theme, iced::Renderer> {
        if matches!(status, VoiceStatus::Transcribing) {
            lucide::pen_line::<iced::Theme, iced::Renderer>()
        } else {
            lucide::ear::<iced::Theme, iced::Renderer>()
        }
    }

    /// The 42px footer bar: picker/toggle chrome and git on the left, active agents on the
    /// right.
    fn footer_view(&self) -> Element<'_, Message> {
        let mut left_elements: Vec<Element<'_, Message>> = Vec::with_capacity(4);

        if let Some(el) = self.render_workspace_picker() {
            left_elements.push(el);
        }
        if self.resolved_workspace().is_some() {
            left_elements.push(self.render_maintainer_toggle());
            left_elements.push(self.render_pause_toggle());
        }
        if let Some(el) = self.render_git_block() {
            left_elements.push(el);
        }

        let left = Row::with_children(left_elements)
            .spacing(theme::SPACE_6)
            .align_y(Alignment::Center);

        // What the microphone is doing: the pipeline's state.
        let center = Self::render_voice_status();

        let right = Row::with_children(
            [Self::render_active_agents(), Self::render_non_agent_calls()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        )
        .spacing(theme::SPACE_12)
        .align_y(Alignment::Center);

        let footer_row = row![left, center, Space::new().width(Length::Fill), right]
            .align_y(Alignment::Center)
            .padding([theme::PAD_3, theme::PAD_18]);

        container(footer_row)
            .align_y(Alignment::Center)
            .height(Length::Fixed(FOOTER_HEIGHT))
            .width(Length::Fill)
            .style(theme::surface_container_style)
            .into()
    }
}

fn footer_status_label(
    label: String,
    icon: Option<iced::widget::text::Text<'static, iced::Theme, iced::Renderer>>,
) -> Element<'static, Message> {
    let label = text(label)
        .size(theme::TEXT_12)
        .color(theme::TEXT_SECONDARY);
    let content: Element<'static, Message> = match icon {
        Some(icon) => Row::with_children(vec![
            icon.size(theme::TEXT_12)
                .color(theme::TEXT_SECONDARY)
                .into(),
            label.into(),
        ])
        .spacing(theme::SPACE_3)
        .align_y(Alignment::Center)
        .into(),
        None => label.into(),
    };
    container(content).padding([0.0, theme::PAD_12]).into()
}

#[derive(serde::Deserialize)]
pub struct WindowState {
    pub width: f32,
    pub height: f32,
    pub x: i32,
    pub y: i32,
    /// Whether a geometry was actually read from the state file rather than falling back to
    /// [`WindowState::default`]: a saved geometry opens as it was saved, a fresh one as
    /// [`WindowState::opening_size`] derives it.
    #[serde(skip)]
    restored: bool,
}

impl WindowState {
    #[expect(clippy::cast_precision_loss)]
    #[must_use]
    pub const fn position(&self) -> iced::window::Position {
        iced::window::Position::Specific(iced::Point::new(self.x as f32, self.y as f32))
    }

    /// The size the window opens at: a saved geometry exactly as saved, a fresh one from
    /// [`fresh_size`] without the ticket column's share, so the chat's line length is the one it
    /// has beside the column (the board's first read then settles the one width decision,
    /// [`StartupWidth`]).
    #[must_use]
    pub fn opening_size(&self, work_area: Option<iced::Size>) -> iced::Size {
        if self.restored {
            return iced::Size::new(self.width, self.height);
        }
        let full = fresh_size(work_area);
        iced::Size::new(
            (full.width - ticket_column_share(full.width)).max(MIN_WINDOW_WIDTH),
            full.height,
        )
    }
}

/// The window's default size in px, sized for Home with its ticket column beside the chat.
const DEFAULT_WINDOW_WIDTH: f32 = 1200.0;
const DEFAULT_WINDOW_HEIGHT: f32 = 760.0;

/// The share of the screen's work area a fresh window may take on either side; the minimum below
/// wins over it.
const WORK_AREA_SHARE: f32 = 0.8;

/// The window's minimum size, shared with the platform settings so neither the width
/// [`WindowState::opening_size`] derives nor [`fresh_size`]'s cap goes below a usable window.
pub const MIN_WINDOW_WIDTH: f32 = 800.0;
pub const MIN_WINDOW_HEIGHT: f32 = 500.0;

/// The fresh window's size with its ticket column: the default, capped to [`WORK_AREA_SHARE`] of
/// the screen's work area on each side — the minimum wins over the cap — and left uncapped where
/// the platform reports no work area.
#[must_use]
fn fresh_size(work_area: Option<iced::Size>) -> iced::Size {
    let (width, height) = work_area.map_or((DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT), |area| {
        (
            DEFAULT_WINDOW_WIDTH.min(area.width * WORK_AREA_SHARE),
            DEFAULT_WINDOW_HEIGHT.min(area.height * WORK_AREA_SHARE),
        )
    });
    iced::Size::new(width.max(MIN_WINDOW_WIDTH), height.max(MIN_WINDOW_HEIGHT))
}

/// The width the ticket column takes of a window `width` px wide: its portion of the content
/// area beside the nav rail, which is what the chat gains back when the column is absent.
#[must_use]
fn ticket_column_share(width: f32) -> f32 {
    (width - SIDEBAR_RAIL_WIDTH) * f32::from(TICKET_COLUMN_PORTION)
        / f32::from(CHAT_COLUMN_PORTION + TICKET_COLUMN_PORTION)
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            width: DEFAULT_WINDOW_WIDTH,
            height: DEFAULT_WINDOW_HEIGHT,
            x: -1,
            y: -1,
            restored: false,
        }
    }
}

const WINDOW_STATE_FILE: &str = "window-state.json";

#[must_use]
pub fn read_window_state() -> WindowState {
    let Ok(dir) = crate::config::default_config_dir() else {
        return WindowState::default();
    };
    std::fs::read_to_string(dir.join(WINDOW_STATE_FILE))
        .ok()
        .and_then(|json| serde_json::from_str::<WindowState>(&json).ok())
        .map(|state| WindowState {
            restored: true,
            ..state
        })
        .unwrap_or_default()
}

#[expect(clippy::cast_possible_truncation)]
fn save_window_state(pos: iced::Point, size: iced::Size) {
    let state = serde_json::json!({
        "width": size.width,
        "height": size.height,
        "x": pos.x as i32,
        "y": pos.y as i32,
    });
    if let Ok(dir) = crate::config::default_config_dir() {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join(WINDOW_STATE_FILE), state.to_string());
    }
}

/// Reload the full workspace map from the store (CDC workspaces event, stream lag, toggle
/// completion, a settings add/delete). A transient read failure — the one scoped retry in this
/// path — is retried once after a short delay; if that fails too,
/// [`Message::WorkspacesReloadFailed`] keeps the current map and renders the failure.
async fn load_workspace_map(generation: u64) -> Message {
    // A transient read failure: retry once after a short delay, then report it while
    // keeping the current map.
    let mut attempt = 0;
    let workspaces = loop {
        match read_workspace_map().await {
            Ok(map) => break map,
            Err(e) => {
                tracing::warn!(error = %e, attempt = attempt + 1, "Failed to load workspaces");
                if attempt == 0 {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_secs(1)).await;
                } else {
                    return Message::WorkspacesReloadFailed {
                        error: e.to_string(),
                        generation,
                    };
                }
            }
        }
    };
    Message::WorkspacesReloaded {
        workspaces,
        generation,
    }
}

async fn read_workspace_map() -> anyhow::Result<HashMap<String, Workspace>> {
    let list = crate::workspace::store().list().await?;
    Ok(list.into_iter().map(|ws| (ws.name.clone(), ws)).collect())
}

/// Resolve the boot selection against a freshly loaded map: the stored name when it is an
/// existing project, otherwise [`crate::users::fallback_workspace_name`]'s pick (the one a
/// deletion makes too), otherwise the personal name, the answer only when the map has no project.
///
/// `prev_selection` is [`crate::users::resolve_selected_workspace_name`]'s answer, so `None` is
/// a failed read: it resolves like any unusable value (the boot must complete) and only decides
/// whether the pick is stored.
#[must_use]
fn resolve_workspace_selection(
    map: &HashMap<String, Workspace>,
    prev_selection: Option<&str>,
    personal_name: &str,
) -> String {
    prev_selection
        .filter(|name| map.contains_key(*name) && !crate::users::is_personal_workspace(name))
        .map(str::to_string)
        .or_else(|| {
            crate::users::fallback_workspace_name(map.keys().map(String::as_str))
                .map(str::to_string)
        })
        .unwrap_or_else(|| personal_name.to_string())
}

/// Load the full map during boot and resolve the active project from the DB
/// (`users.selected_workspace`, the single source of truth) via
/// [`crate::users::resolve_selected_workspace_name`]: an empty or dangling selection lands on a
/// project whenever one exists, the personal name only when none does.
///
/// `remember` is set when a successful stored-selection read differs from the resolved pick
/// ([`Message::BootWorkspaces`]). The read travels to [`Dashboard::apply_boot_workspaces`]
/// as-is: a failed boot read completes with an empty map, but the picker / Settings list render
/// the failure. `generation` is captured at dispatch, so a reload that answered supersedes it.
async fn load_workspace_options(generation: u64) -> Message {
    let user = crate::users::ADMIN_USER_NAME;
    // `None` — the read failed.
    let stored = crate::users::resolve_selected_workspace_name(user).await;
    let personal_name = crate::users::personal_workspace_name(user);
    let (workspaces, restored, remember) = match read_workspace_map().await {
        Ok(map) => {
            let restored = resolve_workspace_selection(&map, stored.as_deref(), &personal_name);
            let remember = stored.as_deref().is_some_and(|stored| stored != restored);
            (Ok(map), restored, remember)
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to load workspaces during boot");
            // A failed read leaves no project to land on.
            (Err(e.to_string()), personal_name, false)
        }
    };
    Message::BootWorkspaces {
        workspaces,
        restored_name: restored,
        remember,
        generation,
    }
}

/// Read the admin's stored active workspace back from the DB (NULL/personal yields the
/// canonical `personal:{user}`). `None` — the read failed; the live selection is kept.
async fn read_active_workspace() -> Option<String> {
    crate::users::resolve_selected_workspace_name(crate::users::ADMIN_USER_NAME).await
}

/// The footer picker's options from the managed-workspace map, sorted by name; a personal
/// workspace is never an option.
#[must_use]
fn managed_workspace_options(map: &HashMap<String, Workspace>) -> Vec<widgets::PickOption> {
    let mut values: Vec<&Workspace> = map
        .values()
        .filter(|ws| !crate::users::is_personal_workspace(&ws.name))
        .collect();
    values.sort_by(|a, b| a.name.cmp(&b.name));
    values
        .into_iter()
        .map(|ws| widgets::PickOption {
            value: ws.name.clone(),
            label: ws.display_name(),
        })
        .collect()
}

/// Reveal a path in the platform's file manager: Finder on macOS, Explorer on Windows, the
/// containing folder on Linux (where `xdg-open` has no select verb). A directory works too —
/// macOS and Windows select it in its parent, Linux opens that parent. The launchers report
/// success and failure unreliably, so their outcome is never trusted.
fn reveal_in_file_manager(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("-R")
            .arg(path)
            .spawn()
            .map_err(|e| format!("Could not open Finder: {e}"))?;
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        let mut cmd = std::process::Command::new("explorer");
        cmd.arg("/select,").arg(path);
        cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
        cmd.spawn()
            .map_err(|e| format!("Could not open Explorer: {e}"))?;
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let parent = path
            .parent()
            .ok_or_else(|| format!("No containing folder: {}", path.display()))?;
        std::process::Command::new("xdg-open")
            .arg(parent)
            .spawn()
            .map_err(|e| format!("Could not open the file manager: {e}"))?;
    }

    Ok(())
}

/// Open an address with the platform's own handler (fire-and-forget). The address is handed
/// over as data — never to something that reads it as a command line.
fn open_url(url: &str) {
    #[cfg(target_os = "windows")]
    shell_open(url);

    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();

    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();

    // No other target has a handler to reach; this keeps `url` used there.
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let _ = url;
}

/// Hand an address to the Windows shell as a *data* argument.
///
/// `ShellExecuteExW` resolves it against the registered protocol and file associations and
/// takes it whole in `lpFile`, with `lpParameters` and `lpVerb` NULL (forcing `open` would
/// change what some addresses do), so `&`, `^`, `%`, `"`, `?`, `|` or a space in it is data,
/// never a command or an argument — `SEE_MASK_DOENVSUBST` is left out so `%NAME%` is not
/// expanded either. The launch runs on a detached throw-away thread, since the call is
/// synchronous and `open_url` runs on the iced event-loop thread, which must not wait on the
/// shell. An empty or whitespace-only address is not an address: nothing may be launched.
#[cfg(target_os = "windows")]
fn shell_open(address: &str) {
    if address.trim().is_empty() {
        return;
    }
    let address = address.to_owned();
    // The NUL-terminated UTF-16 spelling the Win32 API takes; an interior NUL would end the
    // string the shell sees.
    let mut wide: Vec<u16> = address.encode_utf16().collect();
    wide.push(0);
    let spawned = std::thread::Builder::new()
        .name("shell-open".to_owned())
        .spawn(move || {
            if let Some(code) = shell_execute_address(&wide) {
                tracing::warn!(%address, code, "ShellExecuteExW did not open the address");
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "Failed to spawn the thread that opens an address");
    }
}

/// One `ShellExecuteExW` call for the NUL-terminated UTF-16 `address`, on a thread whose COM
/// apartment was just initialised: the single-threaded one the API documents for itself, which
/// some shell extensions require. Returns the `GetLastError` code when the shell did not open
/// the address.
///
/// `SEE_MASK_NOASYNC` is the one `fMask` flag needed: the API requires it of a caller without a
/// message loop that does not outlive the call. No `SEE_MASK_*` suppression flag is set —
/// whatever the shell shows for an address it cannot open is its own, and the code is logged.
#[cfg(target_os = "windows")]
fn shell_execute_address(address: &[u16]) -> Option<u32> {
    use windows_sys::Win32::Foundation::{FALSE, GetLastError};
    use windows_sys::Win32::System::Com::{
        COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE, CoInitializeEx, CoUninitialize,
    };
    use windows_sys::Win32::UI::Shell::{SEE_MASK_NOASYNC, SHELLEXECUTEINFOW, ShellExecuteExW};
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    // `CoInitializeEx` takes the flags as an unsigned word, while `windows-sys` types the
    // constants as the signed `COINIT` alias.
    let coinit = u32::try_from(COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)
        .expect("COINIT flags are small positives");
    // SAFETY: `pv_reserved` is required to be null, and the apartment is the one
    // the API's own documentation names.
    let apartment = unsafe { CoInitializeEx(std::ptr::null(), coinit) };
    // SAFETY: all-zero is this struct's "nothing optional set" form. `cbSize`,
    // `fMask`, `lpVerb`, `lpFile`, `lpParameters` and `nShow` are written below;
    // every remaining member is an output, or an optional input whose NULL/zero
    // means "not set".
    let mut info: SHELLEXECUTEINFOW = unsafe { std::mem::zeroed() };
    info.cbSize = u32::try_from(std::mem::size_of::<SHELLEXECUTEINFOW>())
        .expect("shell execute info fits in u32");
    info.fMask = SEE_MASK_NOASYNC;
    info.lpVerb = std::ptr::null();
    info.lpFile = address.as_ptr();
    info.lpParameters = std::ptr::null();
    info.nShow = SW_SHOWNORMAL;
    // SAFETY: `info` is fully initialised and its `lpFile` borrows `address`,
    // which outlives the call.
    let failed = unsafe { ShellExecuteExW(&raw mut info) } == FALSE;
    // SAFETY: `GetLastError` takes no arguments, and it is read before any other
    // call can overwrite the error the failed launch set.
    let code = failed.then(|| unsafe { GetLastError() });
    if apartment >= 0 {
        // SAFETY: pairs with the successful `CoInitializeEx` above, on the same thread.
        unsafe { CoUninitialize() };
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::board::TicketPhase;
    use crate::util::test::TicketFixture;

    fn ws_at(name: &str, path: &str) -> Workspace {
        Workspace {
            name: name.to_string(),
            path: path.to_string(),
            ..Default::default()
        }
    }

    /// The map a workspace read reports for a comma-separated `names` list.
    fn ws_map(names: &str) -> HashMap<String, Workspace> {
        let mut map = HashMap::new();
        for name in names.split(',').filter(|name| !name.is_empty()) {
            map.insert(name.to_string(), ws_at(name, &format!("/p/{name}")));
        }
        map
    }

    /// A ready dashboard, so the post-ready handlers run instead of the `!ready` guard.
    fn ready_dashboard() -> Dashboard {
        let mut dash = Dashboard::loading(&WindowState::default(), None);
        dash.ready = true;
        dash
    }

    fn map_keys(dash: &Dashboard) -> Vec<&str> {
        let mut keys: Vec<&str> = dash.workspaces.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    fn resolved_pair(dash: &Dashboard) -> Option<(String, String)> {
        dash.resolved_workspace()
            .map(|(name, path)| (name.to_string(), path.to_string()))
    }

    fn assert_workspace_state(
        dash: &Dashboard,
        selection: Option<&str>,
        workspaces: &str,
        name: &str,
    ) {
        let selected = dash.selected_workspace_name.as_deref();
        assert_eq!(selected, selection, "case {name}");
        let keys = map_keys(dash);
        assert_eq!(keys.join(","), workspaces, "case {name}");
        let resolved = selection
            .filter(|selected| keys.contains(selected))
            .map(|selected| (selected.to_string(), format!("/p/{selected}")));
        assert_eq!(resolved_pair(dash), resolved, "case {name}");
        assert_eq!(dash.pushed_workspace, resolved, "case {name}");
        let options = managed_workspace_options(&dash.workspaces);
        let error = dash.workspaces_error.as_deref();
        let picker = options.is_empty().then_some(error).flatten();
        assert_eq!(dash.picker_failure(&options), picker, "case {name}");
    }

    fn empty_board() -> board::BoardMessage {
        board::BoardMessage::Refreshed {
            tickets: Vec::new(),
            generation: 0,
        }
    }

    fn board_with_a_ticket() -> board::BoardMessage {
        board::BoardMessage::Refreshed {
            tickets: vec![TicketFixture::new("T-1", TicketPhase::Backlog).build()],
            generation: 0,
        }
    }

    fn failed_board() -> board::BoardMessage {
        board::BoardMessage::RefreshError {
            error: "db down".to_string(),
            generation: 0,
        }
    }

    fn resize(width: f32, height: f32) -> Message {
        Message::WindowEvent(
            iced::window::Id::unique(),
            window::Event::Resized(iced::Size::new(width, height)),
        )
    }

    fn reloaded(names: &str, generation: u64) -> Message {
        Message::WorkspacesReloaded {
            workspaces: ws_map(names),
            generation,
        }
    }

    fn reload_failed(error: &str, generation: u64) -> Message {
        Message::WorkspacesReloadFailed {
            error: error.to_string(),
            generation,
        }
    }

    fn workspace_deleted(removed: &str, moved_to: Option<&str>) -> Message {
        let done = (removed.to_string(), moved_to.map(str::to_string));
        let result = workspaces::WorkspacesMessage::DeleteResult(Ok(done));
        Message::Settings(settings::SettingsMessage::WorkspaceMsg(result))
    }

    fn assert_surfaces(dash: &mut Dashboard, shown: bool, name: &str) {
        if shown {
            assert!(dash.resolved_workspace().is_some(), "case {name}");
            assert!(dash.render_git_block().is_some(), "case {name}");
            assert!(dash.page_available(Page::Editor), "case {name}");
            assert!(dash.page_available(Page::Shell), "case {name}");
            let _ = dash.open_diff_modal(None);
            assert!(dash.show_diff_modal, "case {name}");
            return;
        }
        assert_eq!(dash.resolved_workspace(), None, "case {name}");
        assert!(!dash.page_available(Page::Editor), "case {name}");
        assert!(!dash.page_available(Page::Shell), "case {name}");
        assert!(dash.page_available(Page::Home), "case {name}");
        assert_eq!(dash.page, Page::Home, "case {name}: left for Home");
        let _ = dash.navigate_to(Page::Editor);
        assert_eq!(dash.page, Page::Home, "case {name}: and stays");
        assert!(dash.render_git_block().is_none(), "case {name}");
        assert_eq!(dash.pushed_workspace, None, "case {name}");
        let _ = dash.open_diff_modal(None);
        assert!(!dash.show_diff_modal, "case {name}: no change view");
    }

    /// The start-up width: settled once, and only by a board read that applied.
    #[test]
    fn start_up_width_cases() {
        enum Drive {
            /// The outcome reaches the board state alone, as a dispatched read's does.
            Direct(board::BoardMessage),
            /// A board read delivered through the dashboard.
            Outcome(board::BoardMessage),
            /// A resize delivered through the dashboard.
            Resize(f32, f32),
            /// A newer read is dispatched, so the outcome that follows is stale.
            Refresh,
        }
        use Drive::*;
        let default = Some(iced::Size::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT));
        let saved = WindowState {
            width: 1000.0,
            height: 700.0,
            x: 10,
            y: 10,
            restored: true,
        };
        for (name, window, steps, expected) in [
            (
                "an empty board fits the narrow opening",
                None,
                vec![Direct(empty_board())],
                ("awaiting", None),
            ),
            (
                "a drawn column takes the width back",
                None,
                vec![Direct(board_with_a_ticket())],
                ("awaiting", default),
            ),
            (
                "a failed read keeps the width",
                None,
                vec![Direct(failed_board())],
                ("awaiting", default),
            ),
            (
                "a read that applied settles it",
                None,
                vec![Outcome(empty_board())],
                ("settled", None),
            ),
            (
                "a stale outcome decides nothing",
                None,
                vec![Refresh, Outcome(empty_board())],
                ("awaiting", default),
            ),
            (
                "the rounded width stays owed",
                None,
                vec![Resize(857.0, 760.0), Resize(857.0, 760.0)],
                ("owed", default),
            ),
            (
                "a height-only drag is not a width",
                None,
                vec![Resize(857.0, 760.0), Resize(857.0, 700.0)],
                ("owed", Some(iced::Size::new(DEFAULT_WINDOW_WIDTH, 700.0))),
            ),
            (
                "a width the user drags is the user's",
                None,
                vec![Resize(857.0, 760.0), Resize(900.0, 760.0)],
                ("settled", None),
            ),
            (
                "a saved geometry is its own answer",
                Some(saved),
                vec![Outcome(board_with_a_ticket())],
                ("settled", None),
            ),
        ] {
            let window = window.unwrap_or_default();
            let mut dash = Dashboard::loading(&window, None);
            dash.ready = true;
            assert_eq!(dash.last_size, window.opening_size(None), "case {name}");
            for step in steps {
                match step {
                    Direct(read) => drop(dash.board_state.update(read)),
                    Outcome(read) => drop(dash.update(Message::Board(read))),
                    Resize(width, height) => drop(dash.update(resize(width, height))),
                    Refresh => drop(dash.board_state.refresh()),
                }
            }
            let state = match dash.startup_width {
                StartupWidth::Settled => "settled",
                StartupWidth::AwaitingReport => "awaiting",
                StartupWidth::Owed => "owed",
            };
            let taken = dash.take_startup_width();
            assert_eq!((state, taken), expected, "case {name}");
            assert_eq!(dash.take_startup_width(), None, "case {name}: once");
        }
    }

    /// The start-up width's target is the fresh size, so the work area's cap reaches the widen too.
    #[test]
    fn the_start_up_width_follows_the_work_area() {
        let area = Some(iced::Size::new(1400.0, 900.0));
        let mut dash = Dashboard::loading(&WindowState::default(), area);
        dash.ready = true;
        // The narrow opening: 1120 - (1120 - 56) * 3/10.
        let opening = dash.last_size;
        assert!(
            (opening.width - 800.8).abs() < 0.01 && (opening.height - 720.0).abs() < 0.01,
            "{opening:?}"
        );
        drop(dash.board_state.update(board_with_a_ticket()));
        assert_eq!(
            dash.take_startup_width(),
            Some(iced::Size::new(1120.0, 720.0)),
            "the capped widen"
        );
    }

    /// The conversation's first load hangs off the app opening, never off a workspace selection.
    #[tokio::test]
    async fn the_app_opening_loads_the_conversation() {
        let (log_store, _logs_dir) = crate::open_test_store!(LogStore, "logs");

        let mut dash = ready_dashboard();
        let generation = dash.home_state.history_generation();
        let _ = dash.update(Message::Boot(Ok(log_store)));
        assert_eq!(
            dash.home_state.history_generation(),
            generation.wrapping_add(1),
            "opening the app must issue the conversation's load"
        );
    }

    enum ReloadStep {
        Booting,
        /// The map a read already left in the state.
        Map(&'static str),
        Select(&'static str),
        /// A newer reload is in flight: the generation the next outcome is measured against.
        Bump(u64),
        /// A reload outcome, through the dashboard.
        Read(Result<&'static str, &'static str>, u64),
        /// The boot read's outcome, applied directly.
        Boot(Result<&'static str, &'static str>, &'static str, bool, u64),
    }

    /// The reload state machine: a read replaces the map and the Settings list, never the live
    /// selection.
    #[test]
    fn workspace_reload_cases() {
        use ReloadStep::*;
        let read = |names: &'static str, dispatched: u64| Read(Ok(names), dispatched);
        let failed = |error: &'static str, dispatched: u64| Read(Err(error), dispatched);
        let boot_failed = |error: &'static str, restored, remember, dispatched| {
            Boot(Err(error), restored, remember, dispatched)
        };
        for (name, steps, (selection, error, workspaces, settings)) in [
            (
                "a known selection resolves",
                vec![Map("ws1,ws2"), Select("ws2")],
                (Some("ws2"), None, "ws1,ws2", ""),
            ),
            (
                "a reload replaces the map and leaves an unknown name standing",
                vec![Map("ws1,ws2"), Select("ws2"), read("ws1,ws3", 0)],
                (Some("ws2"), None, "ws1,ws3", "ws1,ws3"),
            ),
            (
                "a reload whose map holds the selection leaves it",
                vec![Map("ws3"), Select("ws3"), read("ws3", 0)],
                (Some("ws3"), None, "ws3", "ws3"),
            ),
            (
                "a boot-restored selection stands",
                vec![Select("mahbot"), read("mahbot", 0)],
                (Some("mahbot"), None, "mahbot", "mahbot"),
            ),
            (
                "a reload syncs the Settings list",
                vec![read("beta,alpha", 0)],
                (Some("personal:admin"), None, "alpha,beta", "alpha,beta"),
            ),
            (
                "a superseded failure and map change nothing",
                vec![Bump(2), failed("db down", 1), read("ws2", 1)],
                (None, None, "ws1", ""),
            ),
            (
                "a failed reload keeps the map and says why",
                vec![failed("db down", 0)],
                (None, Some("db down"), "ws1", "ws1"),
            ),
            (
                "a reload after a failure clears it",
                vec![failed("db down", 0), read("ws2", 0)],
                (Some("personal:admin"), None, "ws2", "ws2"),
            ),
            (
                "a failed boot read completes",
                vec![
                    Booting,
                    boot_failed("boot read failed", "personal:admin", false, 0),
                ],
                (Some("personal:admin"), Some("boot read failed"), "", ""),
            ),
            (
                "a late boot read is dropped",
                vec![
                    Booting,
                    Map("ws1"),
                    Bump(1),
                    boot_failed("boot read failed", "personal:admin", false, 0),
                ],
                (Some("personal:admin"), None, "ws1", ""),
            ),
        ] {
            let mut dash = ready_dashboard();
            dash.workspaces = ws_map("ws1");
            for step in steps {
                match step {
                    Booting => dash = Dashboard::loading(&WindowState::default(), None),
                    Map(names) => dash.workspaces = ws_map(names),
                    Select(selected) => drop(dash.apply_workspace_selection(selected)),
                    Bump(generation) => dash.workspaces_reload_gen = generation,
                    Read(Ok(names), dispatched) => drop(dash.update(reloaded(names, dispatched))),
                    Read(Err(error), dispatched) => {
                        drop(dash.update(reload_failed(error, dispatched)));
                    }
                    Boot(Ok(names), restored, remember, dispatched) => {
                        let map = ws_map(names);
                        drop(dash.apply_boot_workspaces(Ok(map), restored, remember, dispatched));
                    }
                    Boot(Err(error), restored, remember, dispatched) => {
                        let failure = Err(error.to_string());
                        drop(dash.apply_boot_workspaces(failure, restored, remember, dispatched));
                    }
                }
            }
            assert_workspace_state(&dash, selection, workspaces, name);
            assert_eq!(dash.workspaces_error.as_deref(), error, "case {name}");
            let listed = dash.settings_state.workspaces_state.workspaces.iter();
            let synced: Vec<&str> = listed.map(|ws| ws.name.as_str()).collect();
            assert_eq!(synced.join(","), settings, "case {name}: settings");
        }
    }

    #[derive(Clone, Copy)]
    enum SelectionStep {
        /// Another map than the family's two projects.
        Map(&'static str),
        Select(&'static str),
        /// A pick in the footer, which writes the choice back.
        Switch(&'static str),
        /// A pick that applies at once, superseding any read-back.
        Activate(&'static str),
        /// A read-back, at the generation it was dispatched at.
        Sync(u64, Option<&'static str>),
        /// A newer read is dispatched.
        Bump(u64),
        /// The boot read's map, its restored name and whether that pick is written back.
        Boot(&'static str, &'static str, bool, u64),
        /// A reload whose map arrives after the selection.
        Reload(&'static str, u64),
    }

    type SelectionExpected<'a> = (Option<&'a str>, u64, &'a str);
    type SelectionCase<'a> = (&'a str, &'a [SelectionStep], SelectionExpected<'a>);
    use SelectionStep::*;

    const SELECTION_CASES: &[SelectionCase<'static>] = &[
        (
            "an external switch is adopted",
            &[Select("ws1"), Sync(0, Some("ws2"))],
            (Some("ws2"), 0, "ws1,ws2"),
        ),
        (
            "a pick supersedes a read dispatched before it",
            &[Select("ws1"), Switch("ws1"), Sync(0, Some("ws2"))],
            (Some("ws1"), 1, "ws1,ws2"),
        ),
        (
            "a failed read changes nothing and a stale one cannot land last",
            &[
                Select("ws1"),
                Sync(0, None),
                Sync(0, Some("ws1")),
                Bump(1),
                Sync(0, Some("ws2")),
            ],
            (Some("ws1"), 1, "ws1,ws2"),
        ),
        (
            "a stored name the map lacks is adopted, and a reload does not settle it",
            &[Select("ws1"), Sync(0, Some("late")), Reload("ws1", 0)],
            (Some("late"), 0, "ws1"),
        ),
        (
            "a personal answer does not displace a project",
            &[
                Select("ws1"),
                Sync(0, Some("late")),
                Sync(0, Some("personal:admin")),
            ],
            (Some("late"), 0, "ws1,ws2"),
        ),
        (
            "with no project left the personal answer stands",
            &[
                Select("ws1"),
                Sync(0, Some("late")),
                Map(""),
                Sync(0, Some("personal:admin")),
            ],
            (Some("personal:admin"), 0, ""),
        ),
        (
            "activate applies at once and supersedes a read-back",
            &[Select("ws1"), Activate("ws2")],
            (Some("ws2"), 1, "ws1,ws2"),
        ),
        (
            "activate skips a name the map lacks",
            &[Select("ws1"), Activate("late")],
            (Some("late"), 1, "ws1,ws2"),
        ),
        (
            "a boot pick that is not written back supersedes nothing",
            &[Boot("ws1", "ws1", false, 0)],
            (Some("ws1"), 0, "ws1"),
        ),
        (
            "a written-back boot pick supersedes a read-back in flight",
            &[Boot("ws1", "ws1", true, 0)],
            (Some("ws1"), 1, "ws1"),
        ),
        (
            "boot without the workspace map pushes nothing",
            &[Boot("", "ws1", false, 0)],
            (Some("ws1"), 0, ""),
        ),
        (
            "the reload that resolves the unchanged selection re-pushes it",
            &[Boot("", "ws1", false, 0), Reload("ws1", 0)],
            (Some("ws1"), 0, "ws1"),
        ),
    ];

    /// Only the selection's own entry points change it, and a superseded read-back never reverts
    /// a pick.
    #[test]
    fn workspace_selection_cases() {
        use SelectionStep::*;
        for &(name, steps, (selection, generation, workspaces)) in SELECTION_CASES {
            let mut dash = ready_dashboard();
            dash.workspaces = ws_map("ws1,ws2");
            for &step in steps {
                match step {
                    Map(names) => dash.workspaces = ws_map(names),
                    Select(selected) => drop(dash.apply_workspace_selection(selected)),
                    Switch(name) => drop(dash.select_workspace(name)),
                    Activate(name) => drop(dash.activate_workspace(name)),
                    Sync(dispatched, name) => drop(dash.update(Message::ActiveWorkspaceSynced {
                        generation: dispatched,
                        name: name.map(str::to_string),
                    })),
                    Bump(dispatched) => dash.active_workspace_gen = dispatched,
                    Boot(names, restored, remember, dispatched) => {
                        let map = ws_map(names);
                        drop(dash.apply_boot_workspaces(Ok(map), restored, remember, dispatched));
                    }
                    Reload(names, dispatched) => drop(dash.update(reloaded(names, dispatched))),
                }
            }
            assert_workspace_state(&dash, selection, workspaces, name);
            assert_eq!(dash.active_workspace_gen, generation, "case {name}");
            let resolved = dash.resolved_workspace().is_some();
            let personal = selection.is_some_and(|s| s.starts_with("personal:"));
            let written = selection.is_some_and(|s| dash.selection_persistable(s));
            assert_eq!(written, resolved || personal, "case {name}: writable");
        }
    }

    /// A selection resolves only to a managed workspace in the loaded map with a usable path.
    #[test]
    fn workspace_resolution_cases() {
        enum Probe {
            /// What the footer resolves one selection to.
            Dash(Option<&'static str>),
            /// What the fallback resolves a stored name to while `personal:alice` exists.
            Fallback(Option<&'static str>),
            Options,
        }
        use Probe::*;
        let map = &[("ws1", "/p/ws1"), ("pathless", "")][..];
        let projects = &[
            ("ws1", "/p/ws1"),
            ("ws2", "/p/ws2"),
            ("personal:alice", "/p/personal:alice"),
        ][..];
        let mixed = &[
            ("personal:admin", "/p/personal:admin"),
            ("ws1", "/p/ws1"),
            ("ws2", "/p/ws2"),
        ][..];
        for (name, workspaces, probe, expected) in [
            (
                "a personal selection resolves nothing",
                map,
                Dash(Some("personal:admin")),
                "",
            ),
            (
                "a name missing from the map resolves nothing",
                map,
                Dash(Some("gone")),
                "",
            ),
            (
                "an entry without a path resolves nothing",
                map,
                Dash(Some("pathless")),
                "",
            ),
            ("an empty name resolves nothing", map, Dash(Some("")), ""),
            ("no selection resolves nothing", map, Dash(None), ""),
            (
                "a managed workspace with a path resolves",
                map,
                Dash(Some("ws1")),
                "ws1 /p/ws1",
            ),
            (
                "a stored project stands",
                projects,
                Fallback(Some("ws2")),
                "ws2",
            ),
            (
                "a dangling name lands on the first project",
                projects,
                Fallback(Some("gone")),
                "ws1",
            ),
            (
                "no stored name lands on the first project",
                projects,
                Fallback(None),
                "ws1",
            ),
            (
                "with no project at all it is personal",
                &[],
                Fallback(Some("gone")),
                "personal:alice",
            ),
            (
                "a personal row is never offered",
                mixed,
                Options,
                "ws1, ws2",
            ),
        ] {
            let mut map: HashMap<String, Workspace> = HashMap::new();
            for (name, path) in workspaces {
                map.insert((*name).to_string(), ws_at(name, path));
            }
            let observed = match probe {
                Dash(selection) => {
                    let mut dash = ready_dashboard();
                    dash.workspaces = map;
                    dash.selected_workspace_name = selection.map(str::to_string);
                    let pair = resolved_pair(&dash);
                    pair.map_or_else(String::new, |(name, path)| format!("{name} {path}"))
                }
                Fallback(selection) => {
                    let selected = resolve_workspace_selection(&map, selection, "personal:alice");
                    selected.clone()
                }
                Options => {
                    let options = managed_workspace_options(&map);
                    let offered: Vec<&str> = options.iter().map(|o| o.value.as_str()).collect();
                    offered.join(", ")
                }
            };
            assert_eq!(observed, expected, "case {name}");
        }
    }

    enum SurfaceStep {
        /// Another map than the family's one project.
        Map(&'static str),
        Select(&'static str),
        Navigate(Page),
        /// A deleted project and where the store moved the selection to.
        Delete(&'static str, Option<&'static str>),
        Toggle,
        OpenDiff,
    }

    enum SurfaceSee {
        Shown,
        Hidden,
        /// A deletion: where the desktop moved and what it said.
        Moved(Option<&'static str>, Option<&'static str>),
    }

    /// The workspace-scoped surfaces follow the resolution, and a deleted project says where the
    /// desktop moved.
    #[test]
    fn workspace_scoped_surface_cases() {
        use SurfaceSee as See;
        use SurfaceStep::*;
        for (name, steps, see) in [
            (
                "a resolved selection shows the surfaces",
                vec![Select("ws1")],
                See::Shown,
            ),
            (
                "a personal selection hides them",
                vec![
                    Select("ws1"),
                    Navigate(Page::Editor),
                    Select("personal:admin"),
                ],
                See::Hidden,
            ),
            (
                "a name missing from the map hides them, the toggle included",
                vec![Select("ws1"), Select("gone"), Toggle],
                See::Hidden,
            ),
            (
                "a resolved selection brings them back",
                vec![Select("personal:admin"), Select("ws1")],
                See::Shown,
            ),
            (
                "a delete that left the desktop where it is says nothing",
                vec![
                    Map("ws1,ws2,ws3"),
                    Select("ws1"),
                    Delete("ws2", Some("ws1")),
                ],
                See::Moved(Some("ws1"), None),
            ),
            (
                "a delete that moved it says where",
                vec![
                    Map("ws1,ws2,ws3"),
                    Select("ws1"),
                    Delete("ws2", Some("ws3")),
                ],
                See::Moved(Some("ws3"), Some("Project ws2 removed — ws3 is active")),
            ),
            (
                "the active project's own deletion applies at once",
                vec![
                    Map("ws1,ws2,ws3"),
                    Select("ws1"),
                    Delete("ws2", Some("ws3")),
                    Delete("ws3", Some("ws1")),
                ],
                See::Moved(Some("ws1"), Some("Project ws3 removed — ws1 is active")),
            ),
            (
                "the last project leaves the desktop personal",
                vec![Map("ws1,ws2,ws3"), Select("ws1"), Delete("ws1", None)],
                See::Moved(
                    Some("personal:admin"),
                    Some("Project ws1 removed — the personal folder is active"),
                ),
            ),
            (
                "losing the resolution closes the change view",
                vec![Select("ws1"), OpenDiff, Select("personal:admin")],
                See::Hidden,
            ),
        ] {
            let mut dash = ready_dashboard();
            dash.workspaces = ws_map("ws1");
            for step in steps {
                match step {
                    Map(names) => dash.workspaces = ws_map(names),
                    Select(selected) => drop(dash.apply_workspace_selection(selected)),
                    Navigate(page) => drop(dash.navigate_to(page)),
                    Toggle => drop(dash.update(Message::Toggle(ToggleKind::Pause))),
                    OpenDiff => drop(dash.open_diff_modal(None)),
                    Delete(removed, moved_to) => {
                        drop(dash.update(workspace_deleted(removed, moved_to)));
                    }
                }
            }
            let said = match see {
                See::Moved(selection, message) => {
                    assert_eq!(dash.selected_workspace_name.as_deref(), selection, "{name}");
                    let moved = selection.filter(|name| dash.workspaces.contains_key(*name));
                    let resolved = dash.resolved_workspace().map(|(name, _)| name);
                    assert_eq!(resolved, moved, "case {name}: resolved");
                    message.map(|m| (m, ToastKind::Success))
                }
                see => {
                    assert_surfaces(&mut dash, matches!(see, See::Shown), name);
                    None
                }
            };
            let last = dash.toasts.last().map(|t| (t.message.as_str(), t.kind));
            assert_eq!(last, said, "case {name}");
        }
    }

    /// A foreign project's commit opens read-only without re-pointing the picked workspace's own
    /// change view.
    #[test]
    fn foreign_ticket_commit_diff_keeps_the_picked_view() {
        let mut dash = ready_dashboard();
        dash.workspaces = ws_map("ws1,ws2");
        let _ = dash.apply_workspace_selection("ws1");

        let _ = dash.open_ticket_commit_diff("ws2", "abc1234".to_string());
        assert_eq!(dash.foreign_workspace.as_deref(), Some("ws2"));
        assert!(
            dash.foreign_diff_state.is_viewing_commit(),
            "the foreign view shows ws2's commit"
        );
        assert!(
            !dash.show_diff_modal,
            "the picked workspace's view must not be the one showing"
        );

        let _ = dash.update(Message::DiffModal(diff::DiffMessage::WorkspaceSelected(
            "/p/ws3".to_string(),
        )));
        assert_eq!(dash.foreign_workspace.as_deref(), Some("ws2"));
        assert!(
            dash.foreign_diff_state.is_viewing_commit(),
            "the foreign view must not be re-pointed at the picked workspace"
        );

        let _ = dash.update(Message::ForeignDiff(diff::DiffMessage::BackToWorkingTree));
        assert!(dash.foreign_workspace.is_none());
        assert!(dash.show_diff_modal);

        let _ = dash.open_ticket_commit_diff("ws1", "abc1234".to_string());
        assert!(dash.foreign_workspace.is_none());
        assert!(dash.show_diff_modal);

        let _ = dash.open_ticket_commit_diff("gone", "abc1234".to_string());
        assert!(dash.foreign_workspace.is_none());
        assert!(
            dash.show_diff_modal,
            "the view that was open stands untouched"
        );
        let _ = dash.update(Message::TicketCommitProjectResolved {
            workspace: "gone".to_string(),
            commit_hash: "abc1234".to_string(),
            path: None,
        });
        assert!(matches!(
            dash.toasts.as_slice(),
            [Toast {
                message,
                kind: ToastKind::Error,
                ..
            }] if message == "Project 'gone' is unavailable."
        ));
    }

    /// A workspace switch never touches the ticket sidebar's load failure or in-progress search.
    #[test]
    fn workspace_switch_leaves_the_ticket_sidebar_untouched() {
        let mut dash = ready_dashboard();
        dash.workspaces = ws_map("ws1");
        dash.board_state.load_state.fail("db down".to_string());
        let _ = dash.update(Message::Board(board::BoardMessage::SearchInputChanged(
            editor_widget::EditorAction::Paste("foo".into()),
        )));
        dash.board_state.search_results =
            vec![TicketFixture::new("T-1", TicketPhase::Backlog).build()];

        let _ = dash.propagate_workspace_selection("ws1");

        assert_eq!(dash.board_state.load_state.error(), Some("db down"));
        assert_eq!(dash.board_state.search_query.text(), "foo");
        assert_eq!(dash.board_state.search_results.len(), 1);
    }

    /// The DB-sourced boot resolution: a stored project present in the map is restored and
    /// nothing is written; a stored value naming no existing project lands on the fallback — the
    /// first in the footer switcher's order — and that pick is remembered.
    #[tokio::test]
    #[serial_test::serial(gui_admin_workspace)] // writes the shared seeded admin row
    async fn boot_workspace_options_resolve_from_db() {
        crate::util::test::init_test_stores().await;
        crate::workspace::store()
            .conn
            .execute(
                "INSERT OR IGNORE INTO workspaces (name, path, created_at, updated_at) \
                 VALUES ('boot_ws_gui', '/p/boot_ws_gui', '2026-01-01T00:00:00+00:00', \
                         '2026-01-01T00:00:00+00:00')",
                crate::db::params![],
            )
            .await
            .expect("seed workspace");

        let store = crate::users::store();
        // The guard restores the shared seeded admin row even when an assertion
        // below panics.
        crate::users::test_util::with_admin_workspace_restored(async {
            // A managed stored workspace present in the map → restored as it is.
            store
                .set_selected_workspace(crate::users::ADMIN_USER_NAME, Some("boot_ws_gui"))
                .await
                .expect("update admin selected_workspace");
            let Message::BootWorkspaces {
                workspaces: Ok(map),
                restored_name,
                remember,
                ..
            } = load_workspace_options(0).await
            else {
                panic!("expected BootWorkspaces with a loaded map");
            };
            assert!(map.contains_key("boot_ws_gui"));
            assert_eq!(restored_name, "boot_ws_gui");
            assert!(!remember, "a stored project is restored, not picked");

            // A stored value no project carries → the fallback project, and the
            // pick is remembered so the next read cannot undo it.
            store
                .set_selected_workspace(crate::users::ADMIN_USER_NAME, Some("gone_ws_gui"))
                .await
                .expect("update admin selected_workspace");
            let Message::BootWorkspaces {
                workspaces: Ok(map),
                restored_name,
                remember,
                ..
            } = load_workspace_options(0).await
            else {
                panic!("expected BootWorkspaces");
            };
            assert_ne!(restored_name, "personal:admin");
            assert!(map.contains_key(&restored_name));
            assert_eq!(
                Some(restored_name.as_str()),
                crate::users::fallback_workspace_name(map.keys().map(String::as_str)),
                "the pick is the first project in the footer switcher's order"
            );
            assert!(remember);
        })
        .await;
    }
}
