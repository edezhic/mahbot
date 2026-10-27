//! Native Iced dashboard — application entry point, navigation, and shared state.
//!
//! Iced owns the process Tokio runtime (`iced` feature `tokio`). MahBot
//! bootstraps via a startup [`iced::Task`] before the UI becomes interactive.

#![expect(
    clippy::struct_excessive_bools,
    clippy::if_not_else,
    clippy::collapsible_if
)]

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
pub(crate) mod running;
pub(crate) mod selectable;
pub(crate) mod session_view;
pub(crate) mod sessions;
pub(crate) mod settings;
pub(crate) mod shell;
pub(crate) mod shortcut_keys;
pub(crate) mod spinner;
pub(crate) mod text_rendering;
pub(crate) mod theme;
pub(crate) mod users;
pub(crate) mod widgets;
pub(crate) mod workspaces;

pub(crate) use self::spinner::spinner;

use crate::git::commands::GitPushDestination;
use crate::logs::LogStore;
use crate::pipeline::board::Ticket;

use iced::keyboard;
use iced::widget::{
    Column, Row, Space, button, column, container, image, pick_list, row, rule, scrollable, text,
    tooltip,
};
use iced::window;
use iced::{Alignment, Element, Length, Task};

use crate::Role;
use crate::Workspace;
#[cfg(target_os = "macos")]
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

/// Coalescing window for runtime-change re-renders. Bounds re-render frequency
/// for per-message / per-tool-call broadcasts while keeping the Running Agents
/// cards and footer responsive.
const RUNTIME_REFRESH_COALESCE: Duration = Duration::from_millis(250);

/// The left nav rail's fixed width, in px: the content area starts to its right.
const SIDEBAR_RAIL_WIDTH: f32 = 56.0;

/// The bottom footer bar's fixed height, in px. The toast stack clears it (see
/// [`Dashboard::view`]), and there is no bar to clear while
/// [`Dashboard::footer_hidden`].
const FOOTER_HEIGHT: f32 = 42.0;

/// The portions the Home page's chat and its ticket column split the content
/// area by. The column's share also sizes a fresh window's opening width (see
/// [`ticket_column_share`]).
const CHAT_COLUMN_PORTION: u16 = 7;
const TICKET_COLUMN_PORTION: u16 = 3;

/// How long an exit requested before boot finished waits for the boot task to
/// deliver its result before the exit path runs anyway. The boot is the only
/// writer to the stores until then, and a store left before its first catalog
/// run is refused on the next start (`db::has_product_schema`) — so waiting is
/// the normal case (a boot takes well under a second), and this limit covers
/// only a start that never finishes, which must stay closable.
const BOOT_EXIT_WAIT: Duration = Duration::from_secs(15);

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

/// Warm the CDC workspaces-table sender before the iced application runs (same
/// convention as [`init_board_change_tx`]) so the workspaces CDC subscription
/// always has a source. Called from `main`.
pub fn init_workspace_tx() {
    crate::db::cdc::init_workspace_tx();
}

/// Warm the CDC users-table sender before the iced application runs (same
/// convention as [`init_board_change_tx`]). Called from `main`.
pub fn init_users_tx() {
    crate::db::cdc::init_users_tx();
}

/// Warm the CDC user_channels-table sender before the iced application runs
/// (same convention as [`init_board_change_tx`]). Called from `main`.
pub fn init_user_channels_tx() {
    crate::db::cdc::init_user_channels_tx();
}

/// Initialise the runtime-change broadcast before the iced application runs
/// (same convention as [`init_board_change_tx`]) so the dashboard's
/// runtime-change subscription always has a source. Called from `main`.
pub fn init_runtime_event_tx() {
    crate::runtime_events::init_runtime_event_tx();
}

/// Warm the voice-recording broadcast before the iced application runs (same
/// convention as [`init_board_change_tx`]) so the Home composer's recording
/// subscription always has a source. Called from `main`; nothing to warm on a
/// platform with no voice stack.
pub fn init_recording_tx() {
    #[cfg(target_os = "macos")]
    crate::audio::voice::init_recording_tx();
}

// ── Navigation pages ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Home,
    /// The session list's viewer. It has no rail entry:
    /// [`DashboardShortcut::Sessions`] is its only way in.
    Sessions,
    /// The log store's viewer. It has no rail entry: [`DashboardShortcut::Logs`]
    /// is its only way in.
    Logs,
    Shell,
    Editor,
    Settings,
    Automations,
    /// Live view of running agents and in-flight non-agent LLM work. Its way in
    /// is the activity indicators — the footer bar's, or the rail column's while
    /// [`Dashboard::footer_hidden`] — so, like them, it has no entry while
    /// nothing runs.
    RunningAgents,
}

impl Page {
    /// Pages in the sidebar top nav. The top nav renders this list minus the
    /// pages [`Dashboard::page_available`] excludes.
    const fn sidebar_pages() -> &'static [Page] {
        &[Page::Home, Page::Editor, Page::Shell, Page::Automations]
    }

    /// Pages in the sidebar bottom nav — Sessions and Logs are absent by design
    /// (see [`Page::Sessions`] and [`Page::Logs`]).
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

// ── Main message type ────────────────────────────────────────────

/// Toast notification kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToastKind {
    Success,
    Warning,
    Error,
}

/// Stable toast id source — monotonically increasing, never reused, so a
/// toast's expiry timer cannot dismiss a later toast that reused its slot.
static NEXT_TOAST_ID: AtomicU64 = AtomicU64::new(0);

/// A floating toast notification.
#[derive(Debug, Clone)]
pub(crate) struct Toast {
    /// Stable id used to dismiss the toast when its expiry timer fires.
    pub id: u64,
    pub message: String,
    pub kind: ToastKind,
}

/// Message emitted by a page to request a toast from the dashboard.
#[derive(Debug, Clone)]
pub enum ToastMessage {
    Saved,
    Deleted,
    Error(String),
    Warning(String),
    /// Generic success with custom message.
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
pub enum Message {
    /// MahBot finished async startup (or failed), carrying the log store it
    /// opened — the logs page is built from it.
    Boot(Result<LogStore, String>),
    Navigation(Page),
    /// A toast's expiry timer fired — dismiss the toast by its stable id.
    /// Dismissing an already-expired/unknown id is a safe no-op.
    ToastExpired(u64),
    /// Coalesced notification that the agent/non-agent registries, live
    /// transcript content, or voice status changed — triggers a re-render.
    RuntimeChanged,
    /// A workspaces-table row changed (or the CDC stream lagged/baseline
    /// fired) — triggers one full map reload via `load_workspace_map`.
    WorkspacesCdcChanged,
    /// A users-table or user_channels-table row changed (or the CDC stream
    /// lagged) — triggers a full users re-list on the Settings page and a
    /// read-back of the admin's active workspace (the `workspace` tool's `add`
    /// writes that same column).
    UsersCdcChanged,
    /// The admin's DB-stored active workspace read back after a users-table
    /// change — the desktop and the DB are one and the same choice, so a
    /// switch made by another surface applies here without a restart. A personal
    /// answer applies only while no project exists to show (see the handler: an
    /// empty selection is not a steady state). `generation` is the
    /// `active_workspace_gen` guard value captured at dispatch, so a read a
    /// newer pick or read superseded cannot apply.
    ActiveWorkspaceSynced {
        generation: u64,
        name: Option<String>,
    },
    /// Shutdown signaled — leave the iced runtime for the product's exit path
    /// (`save_and_exit`), which `main` continues after `run()` returns.
    /// Triggered by the shutdown token: a self-update restart, SIGTERM/SIGINT, and the
    /// end of a drain a window close or a macOS quit began.
    Shutdown,
    /// The graceful-drain window began (first signal / window-close / quit /
    /// self-update). Draining disables input; iced::exit is deferred until
    /// the drain completes (the token fires).
    DrainStarted,
    /// The dashboard window's close button: an exit request, handled by
    /// `request_exit` — the graceful drain once boot has finished, the exit path directly
    /// before it.
    CloseRequested(window::Id),
    /// One of the platform's own ordinary quit asks — macOS's ⌘Q, the application menu's Quit
    /// item or the Dock's Quit — delivered by `quit_requests_subscription`, subscribed from the
    /// first frame. Runs the same exit handling as a window close, the second-quit escape
    /// included; the system's asks during a log-out, restart or shut-down are not delivered here
    /// (see `shutdown`).
    QuitRequested,
    /// A window event delivered to the dashboard: geometry (move/resize), which
    /// is tracked for persist-on-close, and file drops, which attach to the
    /// chat composer while the Home page is showing.
    WindowEvent(window::Id, window::Event),
    /// The dashboard window was created, carrying the id of the new window.
    /// Windows only: the icons the taskbar and Alt-Tab show have to be attached to
    /// the window's own handle (see [`crate::app_icon`]).
    #[cfg(target_os = "windows")]
    WindowOpened(window::Id),
    /// The window's own platform handle, the answer to [`Message::WindowOpened`].
    /// Windows only — the value is the window's `HWND`.
    #[cfg(target_os = "windows")]
    WindowHandle(u64),
    /// Keyboard shortcut: Cmd+F — focus the primary search input on the current page.
    FocusSearch,
    /// Keyboard shortcut: Cmd+L — open the Logs page, which has no rail item.
    OpenLogs,
    /// Keyboard shortcut: Cmd+K — open the Sessions page, which has no rail
    /// item.
    OpenSessions,
    /// Keyboard shortcut: Escape — dismiss modal/panel/confirmation on the current page.
    EscapePressed,
    /// Update button pressed — open the self-update confirmation modal.
    UpdateBot,
    /// Self-update confirmed — start the background update.
    ConfirmUpdate,
    /// Self-update confirmation dismissed — nothing happens.
    CancelUpdate,
    /// Self-update result.
    UpdateResult(Result<String, String>),
    /// Toggle the selected workspace's pipeline pause or maintainer state.
    Toggle(ToggleKind),
    /// Footer workspace picker: switch the admin's workspace (managed
    /// workspaces only).
    WorkspacePick(String),
    /// Result of a per-workspace toggle DB write. Carries (kind, result, workspace_name, intended_state).
    /// On success, the workspace map is reloaded from the store; on error an error toast is shown.
    ToggleResult(ToggleKind, Result<(), String>, String, bool),
    /// No-op — the completion message for a fire-and-forget write whose
    /// failure is already logged at the write site.
    Nop,
    /// Workspace info and restored selection loaded during boot. The read
    /// result is carried as-is so a failed boot read surfaces in the picker
    /// and Settings list instead of silently rendering as "no workspaces".
    /// The selection is the resolved name. `remember` says whether that name has
    /// to be written back as the admin's stored selection — it differs from a
    /// successful stored read (see [`Dashboard::apply_boot_workspaces`]). The
    /// generation is the reload generation the read was issued under, so a reload
    /// that answered while it was in flight wins.
    BootWorkspaces {
        workspaces: Result<HashMap<String, Workspace>, String>,
        restored_name: String,
        remember: bool,
        generation: u64,
    },
    /// The wait for a still-running boot expired ([`BOOT_EXIT_WAIT`], a start
    /// that hangs): run the exit path for the exit request that was waiting.
    BootExitWaitExpired,
    /// Full workspace map reloaded from the store (CDC workspaces event,
    /// stream lag/baseline, toggle completion, or a settings add/delete). The
    /// handler replaces the map and re-pushes the workspace-scoped surfaces; it
    /// never re-resolves the live selection against it, because the map can lag
    /// the stored choice — that is the DB read-back's call (see
    /// `Message::ActiveWorkspaceSynced`). The generation is the reload it
    /// answers, so a slow read cannot overwrite a newer one.
    WorkspacesReloaded {
        workspaces: HashMap<String, Workspace>,
        generation: u64,
    },
    /// The post-ready workspace map reload failed (all retries exhausted). The
    /// current map is preserved and the failure is rendered where the picker /
    /// Settings list would be — unless a newer reload has already answered.
    WorkspacesReloadFailed {
        error: String,
        generation: u64,
    },
    Home(home::HomeMessage),
    Logs(logs::LogMessage),
    Board(board::BoardMessage),
    Sessions(sessions::SessionsMessage),
    /// Automations page messages (refresh and card-menu deletion).
    Automations(automations::AutomationsMessage),
    /// Running Agents page messages (manual stop of a research run or a
    /// custom-role launch).
    RunningAgents(running::RunningMessage),
    /// Diff modal overlay (not a page) — wraps [`diff::DiffMessage`].
    /// Named `DiffModal` rather than `Diff` to avoid ambiguity with the
    /// removed `Page::Diff` variant and the existing page-message convention.
    DiffModal(diff::DiffMessage),
    /// The read-only foreign commit view's own [`diff::DiffMessage`]s — a
    /// sidebar ticket of a project that is not the picked one. Routing is by
    /// origin, not by what is on screen, so a task still in flight from the
    /// picked view can never land in another project's view.
    ForeignDiff(diff::DiffMessage),
    /// Git sub-state message.
    Git(git::GitMessage),
    Shell(shell::ShellMessage),
    Editor(editor::EditorMessage),
    Settings(settings::SettingsMessage),

    // ── Diff modal ──────────────────────────────────────────────
    /// Open the diff modal. Optional commit hash — `None` = working tree diff.
    OpenDiffModal(Option<String>),
    /// A sidebar ticket's project, looked up in the store because the
    /// dashboard's workspace map had no entry for it yet. `path` is `None`
    /// when the project is gone.
    TicketCommitProjectResolved {
        workspace: String,
        commit_hash: String,
        path: Option<String>,
    },
    /// Close the diff modal.
    CloseDiffModal,
    /// TTS model download progress event.
    #[cfg(target_os = "macos")]
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

    /// Returns the URL string if this message wraps a `LinkClicked`.
    ///
    /// # Design note
    ///
    /// [`HomeMessage::LinkClicked`] is deliberately **not** included here
    /// because `home.rs` handles its own inline context links internally
    /// (see `HomeState::update`).  Do not add it without understanding
    /// the Home page's self-handling logic.
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

/// One of the dashboard's own shortcuts — the keys that still reach the
/// dashboard while a page holds the keyboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DashboardShortcut {
    /// Close the open overlay, or clear the page's own transient state.
    Escape,
    /// Focus the page's search.
    Search,
    /// Open the Logs page — the shortcut that replaces its rail item.
    Logs,
    /// Open the Sessions page — the shortcut that replaces its rail item.
    Sessions,
}

/// The dashboard shortcut a key press stands for, or `None` — Escape, Cmd+F,
/// Cmd+L or Cmd+K (Ctrl+F / Ctrl+L / Ctrl+K elsewhere, resolved through the
/// physical key so a non-Latin layout still produces them).
///
/// A focused input leaves exactly these uncaptured, because
/// [`keyboard::listen`] only delivers events no widget captured — and the chat
/// composer holds the keyboard from the moment Home is drawn, so capturing
/// them would take the dashboard's own shortcuts away from the whole page.
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

// ── Dashboard state ──────────────────────────────────────────────

/// Where the window's one start-up width decision stands. A fresh window opens
/// without the ticket column's width ([`WindowState::opening_size`]), because a
/// fresh board normally has nothing to show; the board's first read takes the
/// default width back when the column is drawn after all. Nothing resizes the
/// window after that decision — a column appearing or disappearing later
/// changes the layout, never the window.
#[derive(Clone, Copy)]
enum StartupWidth {
    /// Nothing to decide: a saved geometry opened at the size it was closed
    /// with, or the decision is already made.
    Settled,
    /// Owed, waiting for the window's own report of the size it opened at. The
    /// requested size is not that size — the platform rounds it to its pixel
    /// grid and keeps it on screen.
    AwaitingReport,
    /// Owed, with the opening size in [`Dashboard::last_size`]: a width the
    /// window was not at is the user's own drag, which wins over the decision.
    Owed,
}

pub struct Dashboard {
    ready: bool,
    boot_error: Option<String>,
    /// An exit requested while the boot task was still running: the exit waits
    /// for the boot's result (bounded by [`BOOT_EXIT_WAIT`]) instead of
    /// interrupting the store bring-up. See `request_exit`.
    exit_after_boot: bool,
    page: Page,
    log_store: Option<LogStore>,

    /// Tracked window geometry for persist-on-close.
    last_size: iced::Size,
    last_position: iced::Point,
    /// Where the one start-up width decision stands (see [`StartupWidth`]).
    startup_width: StartupWidth,
    /// Toast notification stack.
    toasts: Vec<Toast>,

    /// Full per-workspace records (path, paused, maintenance, metadata),
    /// keyed by name. Populated during boot and updated via CDC-driven
    /// [`Message::WorkspacesReloaded`], which rebuilds the whole map from the
    /// store (no incremental boolean patching).
    workspaces: HashMap<String, Workspace>,
    /// Last workspace-map read failure (boot or a post-ready reload), rendered
    /// where the footer picker and the Settings workspace list would be.
    /// `None` once a successful read replaces it.
    workspaces_error: Option<String>,
    /// Generation of the latest workspace-map reload, so a slow read cannot
    /// apply its (now stale) outcome over a newer one.
    workspaces_reload_gen: u64,
    /// The read-back guard: bumped wherever this app makes the selection newer
    /// than anything a read in flight can return — a pick, a deletion's relocation
    /// (its transaction has committed) and a boot pick that is written back — and
    /// on every read-back it dispatches. Each read carries the generation it was
    /// dispatched with and applies only while it still matches, so a read
    /// dispatched before a local pick — or before a newer read — can never apply. A
    /// read dispatched *after* a pick can still observe that pick's pre-commit row
    /// (the commit lands after the write is issued) and revert the display until
    /// the pick's own CDC event re-reads it.
    active_workspace_gen: u64,
    /// The `(name, path)` pair last pushed to the workspace-scoped surfaces;
    /// `None` while none has been pushed. A workspace-map reload compares
    /// against it, so a change of *resolvability* alone — the selected name
    /// staying the same — still re-pushes (see [`Self::sync_workspace_surfaces`]).
    pushed_workspace: Option<(String, String)>,
    /// Currently selected workspace name. `Some("personal:admin")` = the
    /// admin's "Personal" workspace; `Some("ws")` = a managed workspace;
    /// `None` = nothing selected.
    selected_workspace_name: Option<String>,
    /// True when a genuine exit request — the window's close or a platform quit — arrived
    /// while the update was in its finalizing window (daemon shut down; checkpoint + spawn +
    /// exit pending). Only a real request sets this: the update's own shutdown step also fires
    /// `Message::Shutdown` via the subscription, which must not be mistaken for one. If the
    /// update fails, this flag makes the GUI run its own checkpoint + exit.
    exit_requested_during_update: bool,
    /// Graceful-drain window active: the window stays open with input
    /// disabled until the drain completes.
    draining: bool,
    /// Whether the self-update confirmation modal is open. Set by
    /// [`Message::UpdateBot`]; the update itself only starts on
    /// [`Message::ConfirmUpdate`], so dismissing leaves the system untouched.
    show_update_confirm: bool,
    /// Running work item awaiting manual-stop confirmation: a research run or a
    /// custom-role launch. `Some` while the confirm dialog is shown on the
    /// Running Agents page; the stop runs only on
    /// [`running::RunningMessage::StopConfirmed`]. The carried key is never
    /// displayed as text — the button message carries it.
    pending_stop: Option<running::StopTarget>,
    /// Running Agents page expanded-card state, keyed by (agent_id, generation)
    /// so a recycled agent_id never inherits a stale expansion. Pruned against
    /// the live agent list on each [`Message::RuntimeChanged`] registry event.
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

    // ── Diff modal ──────────────────────────────────────────────
    /// Whether the picked workspace's diff view is open.
    show_diff_modal: bool,
    /// The read-only view of another project's commit, opened from a sidebar
    /// ticket card whose workspace is not the picked one: the project named in
    /// the view's header while it is open, `None` while it is closed. Mutually
    /// exclusive with [`Self::show_diff_modal`], and independent of the picked
    /// workspace's own diff state — viewing a foreign commit never re-points
    /// the working-tree view.
    foreign_workspace: Option<String>,
    /// That view's own state, held for the dashboard's lifetime so its load
    /// generation keeps increasing across opens: a load left in flight when the
    /// view is closed can then never be accepted by the next one. A state
    /// rebuilt per open would restart at generation 1 and could paint one
    /// project's commit under another project's header.
    foreign_diff_state: diff::DiffState,

    // ── Git state ───────────────────────────────────────────────
    /// All git-related state (branch info, sync, branch modal).
    git_state: git::GitState,

    // ── TTS download progress ───────────────────────────────────
    /// Current TTS download progress: (file_name, progress 0.0–1.0).
    /// `None` when no download is active.
    #[cfg(target_os = "macos")]
    tts_download_progress: Option<(String, f32)>,
}

impl Dashboard {
    /// The dashboard before its boot task finishes. `window` is the geometry the
    /// window opens with (`WindowState::opening_size`); a fresh one leaves the
    /// start-up width owed (see `StartupWidth`).
    #[must_use]
    pub fn loading(window: &WindowState) -> Self {
        Self {
            ready: false,
            boot_error: None,
            exit_after_boot: false,
            page: Page::Home,
            log_store: None,
            last_size: window.opening_size(),
            last_position: iced::Point::new(-1.0, -1.0),
            startup_width: if window.restored {
                StartupWidth::Settled
            } else {
                StartupWidth::AwaitingReport
            },
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
            #[cfg(target_os = "macos")]
            tts_download_progress: None,
        }
    }

    fn finish_boot(&mut self, result: Result<LogStore, String>) -> Task<Message> {
        match result {
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

                // The app's own conversation opens with the screen — once,
                // because the app started, together with the persisted composer
                // draft. It is never loaded by a workspace selection.
                let open_chat = self.home_state.open().map(Message::Home);

                // The Home chat now runs the Phase-1 scripted onboarding
                // scenario while no LLM provider is configured (and fires the
                // Phase-2 Assistant kickoff once a provider is set), so no boot
                // redirect to Settings is needed here. The decision is made
                // after the config is fully loaded (bootstrap_mahbot
                // finished), not in the initial loading state where CONFIG
                // still holds defaults. A trimmed non-empty key counts as set
                // — `provider_key()` collapses empty/whitespace to None, so
                // those re-arm the script; an active custom endpoint counts as
                // configured too.
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
        }
    }

    /// Returns the [`Workspace`] for the currently selected workspace, if any.
    fn selected_workspace_info(&self) -> Option<&Workspace> {
        self.selected_workspace_name
            .as_ref()
            .and_then(|name| self.workspaces.get(name))
    }

    /// The workspace resolved for the current selection: a managed workspace
    /// present in the loaded map with a non-empty path. A personal name, a name
    /// missing from the map and an empty path resolve to nothing.
    fn resolved_workspace(&self) -> Option<(&str, &str)> {
        let name = self.selected_workspace_name.as_deref()?;
        if crate::users::is_personal_workspace(name) {
            return None;
        }
        let path = self.workspaces.get(name)?.path.as_str();
        (!path.is_empty()).then_some((name, path))
    }

    /// Whether the bottom footer bar is hidden: the workspace map holds no
    /// managed workspace, and the personal folder is the active selection. Both
    /// halves are load-bearing, because each one covers a transient the other
    /// cannot: the selection stays unset until a read settles one, which carries
    /// the bar through the boot window, and after the last project is deleted it
    /// keeps naming that project while the map is already empty. The empty map
    /// must in turn not be a failed read's placeholder, so a failed read leaves
    /// the bar — and how it reports the failure — exactly where it is.
    ///
    /// With the bar hidden, what it carried moves into the rail: the voice
    /// status under the nav icons and the running-agent indicators above the
    /// bottom nav ([`Self::render_sidebar_mid`]).
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

    /// Whether `page` is offered in the navigation. Only the editor and the
    /// shell are conditional, on [`Self::resolved_workspace`].
    fn page_available(&self, page: Page) -> bool {
        !matches!(page, Page::Editor | Page::Shell) || self.resolved_workspace().is_some()
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

    /// Persist the current window position and size.
    fn persist_window_state(&self) {
        save_window_state(self.last_position, self.last_size);
    }

    /// Take the one start-up width decision, on the board's first read (see
    /// [`StartupWidth`]): the width to resize to, once. A board with nothing to
    /// show takes none — the narrow opening already fits it — while one that
    /// draws the column takes the default width back.
    fn take_startup_width(&mut self) -> Option<iced::Size> {
        let owed = !matches!(self.startup_width, StartupWidth::Settled);
        self.startup_width = StartupWidth::Settled;
        if !owed || self.board_state.has_nothing_to_show() {
            return None;
        }
        Some(iced::Size::new(DEFAULT_WINDOW_WIDTH, self.last_size.height))
    }

    fn settle_startup_width(&mut self) -> Task<Message> {
        let Some(size) = self.take_startup_width() else {
            return Task::none();
        };
        iced::window::latest().and_then(move |id| iced::window::resize(id, size))
    }

    /// Act on a request for the product to exit: name the route that asked, then start the wait
    /// the request calls for.
    ///
    /// The two requests that reach it — the window's own close and an ordinary platform quit —
    /// are handled identically, because full parity is the point of the interception; so are the
    /// states with no drain to complete (a self-update finalizing, a boot that has not finished,
    /// a boot that failed). `route` is the platform's wording for the ask
    /// ([`crate::shutdown::MACOS_QUIT_TRIGGER`], beside the route that produces it), or `None`
    /// for the window's own close, whose wording the exit line already falls back to.
    /// It is recorded where this request is what drives the exit (or, in the self-update window,
    /// may yet drive it once the update is seen to fail); a request that finds the exit already
    /// pending leaves the record to the one that caused it.
    fn request_exit(&mut self, route: Option<&'static str>) -> Task<Message> {
        if crate::self_update::update_in_progress() && crate::self_update::update_is_finalizing() {
            // The update owns the exit (its own drain + checkpoint + swap + spawn): record the
            // request and wait — a force-cancel here would abort the update's full drain. The
            // route is recorded because a failed update re-enters the exit path for this request
            // (see `exit_requested_during_update`); a request that finds the update window
            // already claimed changes nothing, and must not take the record over from the
            // request that claimed it.
            if !self.exit_requested_during_update {
                record_exit_route(route);
            }
            self.persist_window_state();
            self.exit_requested_during_update = true;
            Task::none()
        } else if !self.ready {
            // An exit request before boot finished has no drain to complete: the drain-watch,
            // and the shutdown subscription that turns the token into an exit, both exist only
            // after a successful boot — so the exit path runs directly. Not while the boot is
            // still writing the stores, though: a store left before its first catalog run is
            // refused on the next start (`db::has_product_schema`), so a request that arrives
            // mid-boot waits for the boot's result instead, and [`BOOT_EXIT_WAIT`] bounds that
            // wait so a start which never finishes stays closable.
            if self.boot_error.is_some() {
                // The start failed, so the boot stopped with it: nothing is writing the stores
                // any more.
                record_exit_route(route);
                self.save_and_exit()
            } else if self.exit_after_boot {
                // Nothing changes: this exit is already pending, so the request is not what ends
                // the process and must not take the exit record over from the one that is.
                Task::none()
            } else {
                record_exit_route(route);
                self.exit_after_boot = true;
                Task::perform(async { tokio::time::sleep(BOOT_EXIT_WAIT).await }, |()| {
                    Message::BootExitWaitExpired
                })
            }
        } else if crate::shutdown::is_draining() {
            // An exit request during a drain = second request: force-cancel (fires the token →
            // Message::Shutdown → exit path). The rule is the drain flag's, not the gesture's:
            // a first ⌘Q after a window close's drain forces that drain too. A drain whose token
            // has already fired is an exit under way, though — this request changes nothing about
            // it and must not take the record over.
            if !crate::shutdown::shutdown_token().is_cancelled() {
                record_exit_route(route);
            }
            crate::shutdown::force_cancel();
            Task::none()
        } else {
            // Begin the GRACEFUL drain: the window stays open with input disabled; the
            // drain-watch fires the token when no in-flight work remains (or force-cancels at
            // the cap). save_and_exit is deferred until Message::Shutdown.
            record_exit_route(route);
            crate::shutdown::drain_begin();
            Task::none()
        }
    }

    /// Leave the iced runtime for the product's exit path: flush the draft, record the window
    /// geometry, fire the shutdown token, then exit the runtime.
    ///
    /// The ordinary exit for everything the interface handles — a window close, a platform
    /// quit, SIGINT/SIGTERM, the clean end of a drain — and the entry to the teardown in
    /// `main` that the token's cancellation is checked against. It is not the only way this
    /// process ends: [`crate::shutdown`]'s module docs name the exits that never reach it.
    fn save_and_exit(&self) -> Task<Message> {
        crate::channels::chat_draft::flush_global();
        self.persist_window_state();
        if crate::self_update::update_in_progress() && crate::self_update::update_is_finalizing() {
            // The update path has shut down the daemon and owns the exit: checkpoint,
            // temp-root cleanup, lock release, spawn, exit(0). Exiting here would drop the
            // iced runtime and abort that sequence mid-checkpoint, leaving the daemon down
            // without a replacement — so wait instead. A request that arrives meanwhile is
            // recorded by `request_exit`; if the update fails, UpdateResult re-enters this path
            // (in-progress cleared, flag cleared) and honors it.
            return Task::none();
        }
        // Fire the token before the runtime drops — `shutdown_after_dashboard` relies on it
        // being cancelled by then (see its note in main.rs), and a request that arrives before
        // boot finished reaches here with no drain to have fired it. Idempotent: every other
        // arrival has it cancelled already.
        crate::shutdown::force_cancel();
        // The checkpoint is deliberately NOT run here: it relocated to
        // shutdown_after_dashboard, which runs after the iced runtime drops — genuinely
        // single-writer (today's in-iced checkpoint ran while background writers were still
        // live).
        iced::exit()
    }

    /// Window title with page name.
    pub fn title(&self) -> String {
        let page_name = self.page.label();
        format!("MahBot — {page_name}")
    }

    /// Apply workspace configuration loaded during boot. The boot path must
    /// complete either way: a failed read leaves an empty map beside the failure
    /// that stands in for it.
    ///
    /// `read_generation` is the reload generation the read was issued under. A
    /// reload that landed while it was in flight read the same table later, so
    /// its map (or its failure) wins and this read's outcome is dropped — the
    /// rest of the boot path still completes. The restored selection is applied
    /// and propagated either way; the workspace-scoped surfaces stay empty until
    /// it resolves (see [`Self::sync_workspace_surfaces`]).
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

        // The boot read always yields a name — a project or the admin's
        // `personal:{user}` fallback, which is the answer only when no project
        // exists. A pick produced by a failed read is written by nobody — a failed
        // read is no evidence that nothing is selected — so an in-flight read-back
        // may still correct it.
        if !remember {
            return self.apply_workspace_selection(restored_name);
        }
        // Written back unconditionally, unlike a pick: the name came from the same
        // DB read as the stored selection, so the loaded map — which may not even
        // have arrived — has no say in it. Reusing `switch_workspace` here would let
        // a stale map skip the write, and the next read-back would drop the desktop
        // back into the personal folder.
        Task::batch([
            self.apply_workspace_selection_superseding_read_back(restored_name),
            Self::persist_active_workspace(restored_name.to_owned(), |_| Message::Nop),
        ])
    }

    /// Navigate to a page, refreshing page-specific state as needed. A page that
    /// does not resolve ([`Self::page_available`]) is refused rather than opened
    /// over no workspace.
    fn navigate_to(&mut self, page: Page) -> Task<Message> {
        if !self.page_available(page) {
            return Task::none();
        }
        self.page = page;
        // Every navigation notifies the editor of its visibility AFTER
        // `self.page` is set, so the handler sees the new state. This is the
        // single source of truth for the editor's page-visible gating: on the
        // Editor page it triggers refresh-on-entry, on any other page it closes
        // the editor's modals (modals do not survive a page switch).
        let visibility = Task::done(Message::Editor(
            editor::EditorMessage::PageVisibilityChanged(self.page == Page::Editor),
        ));
        let refresh = match page {
            // Logs and Shell maintain their own internal state; Editor
            // receives workspace state via WorkspaceSelected from the
            // Dashboard — none need a refresh on navigation. Running Agents
            // reads the live registries at render time.
            Page::Logs | Page::Shell | Page::Editor | Page::RunningAgents => Task::none(),
            Page::Home => {
                let snap = iced::widget::operation::snap_to_end::<Message>(home::CHAT_SCROLL_ID);
                // The composer takes the keyboard once this page is on screen,
                // so a paste lands in the input without a click first.
                self.home_state.arm_composer_focus();
                let board_refresh = self.board_state.refresh().map(Message::Board);
                Task::batch([snap, board_refresh])
            }
            Page::Sessions => self.sessions_state.refresh().map(Message::Sessions),
            // The Automations page renders nothing but its polled list, so entering
            // it starts a read rather than a render of stale entries.
            Page::Automations => self.automations_state.refresh().map(Message::Automations),
            Page::Settings => {
                self.settings_state.refresh();
                // Workspace list comes from the managed-workspace map; only the users
                // list, the custom-tools area and the custom-roles list need a
                // DB / folder read on navigation.
                self.sync_settings_workspaces_from_map();
                Task::batch([
                    self.refresh_settings_users(),
                    self.refresh_settings_custom_tools(),
                    self.refresh_settings_custom_roles(),
                ])
            }
        };
        Task::batch([visibility, refresh])
    }

    /// Enter `page` from the shortcut that replaces its rail item. A dashboard
    /// overlay covers the rail, so nothing navigates out from under one — Escape
    /// is what closes overlays.
    fn open_from_shortcut(&mut self, page: Page) -> Task<Message> {
        if self.overlay_modal_open() {
            return Task::none();
        }
        self.navigate_to(page)
    }

    /// Refresh the Settings users list. The workspace list is not read here —
    /// it is synced from the dashboard's managed-workspace map. Does not absorb
    /// the config snapshot (`settings_state.refresh()`).
    fn refresh_settings_users(&self) -> Task<Message> {
        self.settings_state
            .users_state
            .refresh()
            .map(|msg| Message::Settings(settings::SettingsMessage::UserMsg(msg)))
    }

    /// Refresh the Settings custom-tools area (its tool-folder read). Does not
    /// absorb the config snapshot (`settings_state.refresh()`).
    fn refresh_settings_custom_tools(&mut self) -> Task<Message> {
        self.settings_state
            .custom_tools_state
            .refresh()
            .map(|msg| Message::Settings(settings::SettingsMessage::CustomToolsMsg(msg)))
    }

    /// Refresh the Settings custom-roles list (its DB read). Does not absorb the
    /// config snapshot (`settings_state.refresh()`).
    fn refresh_settings_custom_roles(&self) -> Task<Message> {
        self.settings_state
            .custom_roles_state
            .refresh()
            .map(|msg| Message::Settings(settings::SettingsMessage::CustomRolesMsg(msg)))
    }

    /// Sync the Settings workspaces section list from the managed-workspace map
    /// (sorted by name, matching the store's `ORDER BY name`). No DB read — the
    /// map is the single source of truth.
    fn sync_settings_workspaces_from_map(&mut self) {
        let mut list: Vec<Workspace> = self.workspaces.values().cloned().collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        self.settings_state.workspaces_state.workspaces = list;
    }

    /// Toggle the active project's pause or maintainer state. Only a resolved
    /// project offers these (see [`Self::resolved_workspace`]), so the name written
    /// is the resolved one — never a stored value that names no existing project.
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
                let toast = self.push_toast(format!("{label} for {ws_name}"), ToastKind::Success);
                // Immediate post-toggle refresh; the CDC stream provides the
                // durable confirmation later.
                let refresh = self.reload_workspace_map();
                Task::batch([toast, refresh])
            }
            Err(e) => self.push_toast(format!("{}: {e}", kind.label_err()), ToastKind::Error),
        }
    }

    /// Push a toast with a kind-specific expiry timer. The timer is a
    /// fire-and-forget task that emits [`Message::ToastExpired`] with the
    /// toast's stable id when it fires.
    fn push_toast(&mut self, message: String, kind: ToastKind) -> Task<Message> {
        self.spawn_toast(Toast::new(message, kind))
    }

    /// Push a toast from a [`ToastMessage`] (see [`Self::push_toast`]).
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

    /// Handle Escape key: dismiss modals in priority order (self-update
    /// confirmation → foreign commit view → diff modal → git branch modal →
    /// page-level escape dispatch).
    fn process_escape(&mut self) -> Task<Message> {
        // Modal close priority: self-update confirmation first, then the
        // foreign commit view, then the picked diff modal, then the branch
        // modal, then page-level escapes.
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
                // Shell and Automations have no escape handling.
                // Running Agents: Escape dismisses the pending stop
                // confirmation (Keep) — never confirms it.
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

    /// Process a Settings message, optionally reloading the workspace map when
    /// workspaces are added or deleted.
    fn process_settings_message(&mut self, msg: settings::SettingsMessage) -> Task<Message> {
        // A Settings-side workspace mutation activates at most one project: an add
        // activates the new workspace (the `workspace` tool's own `add` does the
        // same), a delete moves the admin's selection in the store and the desktop
        // follows it. Whatever moved the desktop is reported — changed tabs, git
        // state and toggles are otherwise the only sign of it.
        let mut activation = None;
        match &msg {
            settings::SettingsMessage::AddWorkspaceResult(Ok(ws)) => {
                // Seeded into the map so the picker renders it at once and the
                // activation's own map-guard sees the workspace it just created;
                // the reload this message triggers below is what confirms it.
                self.workspaces.insert(ws.name.clone(), ws.clone());
                activation = Some(self.activate_workspace(&ws.name));
            }
            settings::SettingsMessage::WorkspaceMsg(
                workspaces::WorkspacesMessage::DeleteResult(Ok((removed, active_now))),
            ) => {
                // The store has already resolved the admin's selection; aligning the
                // desktop with that answer at once keeps it from passing through the
                // no-project state while the map reload and the users read-back race
                // each other.
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
        // Any successful Settings-side workspace mutation (add, delete,
        // reanalyze, diagnostics/rules edits) refreshes the managed
        // workspace map from one full DB read — the map is the single source
        // of truth, and syncing the Settings workspace list from it (see
        // [`Self::sync_settings_workspaces_from_map`]) gives the page its immediate
        // post-op feedback. The workspaces sub-state itself never reads the
        // store.
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

    /// Process a Running Agents page message — the manual-stop flow for
    /// research runs and custom-role launches: button → pending-confirmation →
    /// async stop → toast.
    fn process_running_message(&mut self, msg: running::RunningMessage) -> Task<Message> {
        match msg {
            running::RunningMessage::StopRequest(target) => {
                // Pending-confirmation only: the dialog renders over the
                // Running Agents page; the target's key is never displayed.
                self.pending_stop = Some(target);
                Task::none()
            }
            running::RunningMessage::StopConfirmed => {
                let Some(target) = self.pending_stop.take() else {
                    return Task::none();
                };
                match target {
                    // The cancel is async: fire the run's signal and stop the
                    // run's agents. Silent to the Manager/caller — only the GUI
                    // gets a toast. The cancelled-run handoff (or a
                    // no-orchestrator sweep) owns the cleanup dispatch and
                    // folder release. Infallible.
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
                    // Stopping a launch: cancel its agent, mark the branch as
                    // deliberately stopped and drop its pending messages. A
                    // failure is reported as an error toast, not swallowed.
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
        // ── Centralized Toast and LinkClicked interception ──────────
        // Intercepted at the Dashboard level (before page dispatch)
        // so page handlers only need a Task::none() arm for match exhaustiveness.
        // HomeMessage::LinkClicked is handled by Home itself — see
        // as_link_url() for details.
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
            // ── Pre-ready handlers (execute regardless of ready state) ──
            Message::Boot(result) => {
                // An exit request that arrived while the boot was still writing the
                // stores takes effect now — the boot has stopped, so it cannot
                // be left mid-bring-up (see `request_exit`).
                if std::mem::take(&mut self.exit_after_boot) {
                    return self.save_and_exit();
                }
                self.finish_boot(result)
            }
            Message::BootExitWaitExpired => {
                // The boot did not deliver a result within [`BOOT_EXIT_WAIT`] (a
                // start that hangs): the exit request that was waiting ends the
                // process anyway.
                self.save_and_exit()
            }
            Message::BootWorkspaces {
                workspaces,
                restored_name,
                remember,
                generation,
            } => self.apply_boot_workspaces(workspaces, &restored_name, remember, generation),
            Message::CloseRequested(_) => self.request_exit(None),
            Message::QuitRequested => self.request_exit(Some(crate::shutdown::MACOS_QUIT_TRIGGER)),
            Message::Shutdown => self.save_and_exit(),
            Message::DrainStarted => {
                // Restart toast: only when the update has finished and finalize
                // is shutting the service down — a plain window close, quit or
                // signal during an update still in flight (not yet finalizing)
                // must not show a false "restarting" toast. `!self.draining` is
                // defensive single-emission protection.
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
                        // The window's first report of its size is the one the
                        // platform gave it, not a resize.
                        StartupWidth::AwaitingReport => {
                            self.startup_width = StartupWidth::Owed;
                        }
                        // A width the window was not at is the user's own drag —
                        // before the board's first read included — while a
                        // report of the width it already has is the platform's
                        // again. Only the width decides: the decision is about
                        // the width.
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
                // A dropped file attaches to the next chat message: the whole
                // Home page is the target, while every other page (and a
                // dashboard overlay drawn above Home) ignores drops. Several
                // files arrive as several events, and a folder is not an
                // attachment.
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
            #[cfg(target_os = "windows")]
            Message::WindowOpened(id) => window::raw_id::<Message>(id).map(Message::WindowHandle),
            #[cfg(target_os = "windows")]
            Message::WindowHandle(raw) => {
                crate::app_icon::apply_window_icons(raw);
                Task::none()
            }
            Message::CloseDiffModal => {
                // The backdrop closes whichever view is showing, dropping that
                // view's commit ref with it so a later open cannot inherit a
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

            // ── Everything else before boot → silent no-op ──
            _ if !self.ready => Task::none(),

            // ── Post-ready handlers (no per-arm guards needed) ──
            Message::WorkspacePick(name) => self.select_workspace(&name),
            Message::ToastExpired(id) => {
                // Dismissing an already-expired/unknown id is a safe no-op.
                self.toasts.retain(|t| t.id != id);
                Task::none()
            }
            Message::RuntimeChanged => {
                running::prune_expanded(
                    &mut self.running_expanded,
                    &crate::agent::registry::AGENT_REGISTRY.list(),
                );
                // The tick's Sessions work (list-cache refresh + live
                // transcript merge) is only observable while the page is
                // visible; navigating onto it refreshes, so gating here
                // cannot go stale.
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
                // Intercept a ticket's commit diff for cross-page navigation
                // before it reaches board_state.update.
                if let board::BoardMessage::ViewCommitDiff {
                    ref commit_hash,
                    ref workspace_name,
                } = msg
                {
                    return self.open_ticket_commit_diff(workspace_name, commit_hash.clone());
                }
                // An applied read outcome is when the one start-up width decision
                // is taken (see [`StartupWidth`]); a delta carries no board state
                // to decide it, and an outcome the board dropped as stale carries
                // none either — a newer read is still on its way.
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
            // Intercept CloseModal from successful manual commit — auto-close
            // the diff modal while keeping the diff state in working-tree view.
            // ClearCommitState is intentionally not emitted; the commit handler
            // already cleared commit state and kicked off a diff refresh. A
            // foreign view is untouched: it has a channel of its own and closes
            // through its own controls.
            Message::DiffModal(diff::DiffMessage::CloseModal) => {
                self.show_diff_modal = false;
                Task::none()
            }
            // "Back to working tree" in the foreign view leaves that project
            // for the picked workspace's working tree — the other project's
            // working tree is not a place to land.
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
                // A manual commit is a ref-only change (HEAD + `.git/index`)
                // that the file watcher never reports, so refresh the git footer
                // promptly — otherwise diff stats and remote state stay stale
                // until the periodic timer.
                let commit_succeeded = matches!(&msg, diff::DiffMessage::CommitResult(Ok(_)));
                // Every `Message::DiffModal` belongs to the picked workspace's
                // view — its subscription, its rendered controls, the
                // dashboard's pushes to it. The foreign view speaks
                // `Message::ForeignDiff`, so neither state can receive the
                // other's messages.
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
                // The editor's commit/staging mtime detector requests a
                // git-status refresh. Route it through GitState's standard
                // throttle/coalescing file-change path instead of the editor's
                // own handler, so it reuses the 300ms refresh coalescing (and
                // the resulting WorktreeSnapshot flows back to the editor via
                // the usual GitStatusUpdated forward).
                if matches!(msg, editor::EditorMessage::RequestGitStatusRefresh) {
                    self.git_state
                        .update(git::GitMessage::FileChanged)
                        .map(Message::Git)
                } else {
                    self.editor_state.update(msg).map(Message::Editor)
                }
            }
            Message::Settings(msg) => self.process_settings_message(msg),
            // ── Diff modal ────────────────────────────────────────
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
            // ── Git state (routed to self.git_state) ─────────────────
            Message::Git(msg) => {
                // Cross-modal close: if opening the branch modal, close both
                // diff views from the Dashboard side — one overlay at a time.
                if matches!(msg, git::GitMessage::OpenModal) {
                    self.show_diff_modal = false;
                    self.foreign_workspace = None;
                }
                let snapshot = matches!(msg, git::GitMessage::WorktreeSnapshot(..));
                let task = self.git_state.update(msg).map(Message::Git);
                // A current-generation worktree snapshot (Ok or Err) hands the
                // editor the latest (possibly last-known-good) per-file statuses
                // so its file-tree colors track GitState — the editor no longer
                // runs its own `git status` subprocess.
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
                // A repeat press on the page it opens changes nothing:
                // re-entering it would re-run the read `navigate_to` starts on
                // entry.
                if self.page == Page::Sessions {
                    return Task::none();
                }
                self.open_from_shortcut(Page::Sessions)
            }
            Message::EscapePressed => self.process_escape(),
            Message::UpdateBot => {
                // Only opens the confirmation modal — the update
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
                // Claim the update before spawning it, exactly as the `/update`
                // command does: an update that fires by itself can start between
                // the check above and the task below, and a second attempt must
                // stand down here instead of colliding inside `execute_update`
                // and reporting the collision as a failure.
                if !crate::self_update::claim_update_start() {
                    return Task::none();
                }
                // Save window state before update (synchronous).
                self.persist_window_state();
                // Transient confirmation that the update has kicked off; the
                // update itself tells the admin the same thing in its own words.
                let toast = self.push_toast("Update started…".to_string(), ToastKind::Success);
                Task::batch([
                    toast,
                    Task::perform(
                        async {
                            match crate::self_update::execute_update(None).await {
                                Ok(()) => Ok("ok".to_string()),
                                Err(e) => {
                                    // Reported to the admin exactly as the
                                    // unattended trigger reports it, while the
                                    // window shows its own toast.
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
                // execute_update() calls exit(0) on success, so we never
                // actually reach this branch for the Ok case. The update
                // toasts (start + restart) are the success signals; the
                // window closing is the final confirmation. On failure
                // `execute_update` already cleared the shared in-progress
                // flag, so the exit guards below no longer wait.
                if let Err(err) = result {
                    let toast = self.push_toast_msg(&ToastMessage::Error(err));
                    if self.exit_requested_during_update {
                        // A window close or a platform quit arrived while the
                        // update owned the exit; it failed, so run the normal
                        // exit checkpoint.
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
            // A workspaces-table row changed (or the stream lagged/baseline
            // fired) — one full map reload; the handler below re-pushes the
            // workspace-scoped surfaces from the fresh map.
            Message::WorkspacesCdcChanged => self.reload_workspace_map(),
            Message::WorkspacesReloaded {
                workspaces,
                generation,
            } => {
                // Stale-result guard: a slow reload must not overwrite a newer
                // one that has already answered.
                if generation != self.workspaces_reload_gen {
                    return Task::none();
                }
                self.apply_workspace_reload(Ok(workspaces));

                // Nothing is picked from this map: it can lag the stored choice (a
                // workspace added and selected in one go is missing from a read that
                // started before its insert), and resolving against it would leave
                // the desktop on the personal space while the DB keeps the project
                // active. The DB owns the choice and its read-back applies it (see
                // `active_workspace_gen`), so the applied map only re-pushes the
                // workspace-scoped surfaces — whose resolvability this map may have
                // changed (the boot race leaves the restored selection resolved
                // against the map that was dropped as stale).
                //
                // The one case left to settle is a desktop that has applied no
                // selection at all yet — the reload landed before the boot read —
                // which takes the personal default the boot selection then replaces.
                if self.selected_workspace_name.is_some() {
                    return self.sync_workspace_surfaces();
                }
                let personal_name =
                    crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME);
                self.apply_workspace_selection(&personal_name)
            }
            // A users-table / user_channels-table row changed (or the stream
            // lagged) — the event does not say which column changed, so the
            // read-back is the authority on the managed selection. Full re-list
            // is its own recovery.
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
                // Stale reads are dropped (see `active_workspace_gen`); an
                // unchanged value is a no-op rather than a re-run of the
                // propagate's pushes.
                if generation != self.active_workspace_gen
                    || self.selected_workspace_name.as_deref() == Some(name.as_str())
                {
                    return Task::none();
                }
                // "No project selected" is not a steady state: start-up and a
                // deletion both write the fallback back, so a personal answer while
                // the loaded map carries a project is the residue of a failed
                // start-up read (which wrote nothing, on purpose). Applying it would
                // drop the desktop outside every project — the live pick stands.
                if crate::users::is_personal_workspace(&name)
                    && self
                        .workspaces
                        .keys()
                        .any(|ws_name| !crate::users::is_personal_workspace(ws_name))
                {
                    return Task::none();
                }
                // Applied as stored, not resolved against the loaded map the way a
                // reload is: the map can be a reload behind while the DB is
                // authoritative about the choice. In memory only — writing the value
                // back would answer our own CDC event.
                self.apply_workspace_selection(&name)
            }
            // A failed read-back changes nothing: the live selection stands.
            Message::ActiveWorkspaceSynced { name: None, .. } => Task::none(),
            Message::WorkspacesReloadFailed { error, generation } => {
                // Stale-result guard, mirroring the success arm: without it a
                // reload that fails after a newer one succeeded would re-set the
                // failure over the map that read produced.
                if generation != self.workspaces_reload_gen {
                    return Task::none();
                }
                // Keep the current map (the selection may still resolve) and
                // render the failure where the picker / Settings list would be.
                self.apply_workspace_reload(Err(error));
                Task::none()
            }
            Message::Nop => Task::none(),
            #[cfg(target_os = "macos")]
            Message::TtsDownloadEvent(event) => self.handle_tts_download_event(event),
        }
    }

    /// Handle a TTS download progress event.
    #[cfg(target_os = "macos")]
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

    /// Open the picked workspace's diff modal, closing any board or branch
    /// modal first, and leaving any foreign commit view behind.
    ///
    /// When `commit_hash` is `Some`, navigates to that commit; when `None`,
    /// navigates to the working tree (clearing any stale commit state).
    ///
    /// The change/diff view is workspace-scoped, so it never opens while
    /// nothing is resolved; losing the resolution has already closed it (see
    /// [`Self::sync_workspace_surfaces`]).
    fn open_diff_modal(&mut self, commit_hash: Option<String>) -> Task<Message> {
        self.foreign_workspace = None;
        if self.resolved_workspace().is_none() {
            return Task::none();
        }
        // Close any board modal
        let close_board = self
            .board_state
            .update(board::BoardMessage::CloseModal)
            .map(Message::Board);
        self.show_diff_modal = true;
        // Close branch modal synchronously if open.
        // CloseModal always returns Task::none() so discarding is safe.
        let _ = self.git_state.update(git::GitMessage::CloseModal);
        let diff_task = match commit_hash {
            Some(hash) => Task::done(Message::DiffModal(diff::DiffMessage::NavigateToCommit(
                hash,
            ))),
            None => Task::done(Message::DiffModal(diff::DiffMessage::BackToWorkingTree)),
        };
        Task::batch([close_board, diff_task])
    }

    /// Open a sidebar ticket's commit diff against the ticket's own project.
    ///
    /// A ticket of the picked workspace opens the ordinary diff view. A ticket
    /// of another project is shown read-only — that project's own commit
    /// history, named in the view, with no file-changing control and without
    /// re-pointing the picked workspace's view. A ticket whose project cannot
    /// be resolved reports that instead of showing another project's history.
    fn open_ticket_commit_diff(&mut self, workspace: &str, commit_hash: String) -> Task<Message> {
        if self
            .resolved_workspace()
            .is_some_and(|(name, _)| name == workspace)
        {
            return self.open_diff_modal(Some(commit_hash));
        }
        // The workspace map is read at boot and kept live by CDC, but a click
        // on the sidebar can beat that first read — the store is the authority,
        // so a miss falls back to it rather than reporting a project that
        // exists as unavailable.
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

    /// Apply a workspace selection in memory and broadcast it to all pages,
    /// without writing the DB. Used where nothing supersedes a read-back in flight:
    /// the read-back itself (`Message::ActiveWorkspaceSynced`), the pre-boot
    /// personal default, and a boot pick produced by a failed read. Every other
    /// path goes through [`Self::apply_workspace_selection_superseding_read_back`].
    fn apply_workspace_selection(&mut self, name: &str) -> Task<Message> {
        // Git state is cleared and eagerly refreshed below via
        // propagate_workspace_selection → set_workspace_path.
        self.selected_workspace_name = Some(name.to_string());
        self.propagate_workspace_selection(name)
    }

    /// Make `name` the active project, superseding any read-back already in flight:
    /// what sets the selection here — a pick, a deletion whose transaction has
    /// committed, a boot pick that is written back — is newer than anything a read
    /// dispatched earlier can return (see `active_workspace_gen`).
    fn apply_workspace_selection_superseding_read_back(&mut self, name: &str) -> Task<Message> {
        let propagate = self.apply_workspace_selection(name);
        self.active_workspace_gen += 1;
        propagate
    }

    /// The one workspace-switch path, shared by the footer picker and the
    /// Settings add form: [`Self::apply_workspace_selection_superseding_read_back`]
    /// plus the write, which is skipped when the loaded map has dropped the name
    /// since the render (see [`Self::selection_persistable`]).
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

    /// Whether a pick may be written to the DB: a managed name only when the
    /// loaded map carries it — the map can drop one between render and click, and
    /// the DB must not gain a dangling value (the in-memory selection still
    /// applies; the read-back decides if it stands). A personal space is always
    /// writable: the map holds managed workspaces only and so never carries
    /// one, and the write stores NULL — the product's own "no managed workspace
    /// selected", from which it computes the personal workspace on the fly.
    fn selection_persistable(&self, name: &str) -> bool {
        crate::users::is_personal_workspace(name) || self.workspaces.contains_key(name)
    }

    /// Persist `name` as the admin's active workspace. A failure is always logged
    /// here; `feedback` decides what the caller hears on top of that — the footer
    /// picker drops it (`Message::Nop`), the Settings side turns it into a toast.
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

    /// The footer picker's switch: [`Self::switch_workspace`] with logging-only
    /// feedback — a failed write here is logged, not toasted.
    fn select_workspace(&mut self, name: &str) -> Task<Message> {
        self.switch_workspace(name, |_| Message::Nop)
    }

    /// Make a just-added workspace the active one from the Settings add form —
    /// the same switch path as the picker, with the Settings section's own
    /// feedback so a failed write is reported rather than only logged.
    fn activate_workspace(&mut self, name: &str) -> Task<Message> {
        self.switch_workspace(name, Self::activation_feedback)
    }

    /// The activation write's feedback, through the Settings section's own toast
    /// message (intercepted by [`Message::as_toast`]) rather than a message of
    /// its own — so a failure carries the same `Failed: …` wording as the
    /// section's other error toasts.
    fn activation_feedback(outcome: Result<String, String>) -> Message {
        Message::Settings(settings::SettingsMessage::WorkspaceMsg(
            workspaces::WorkspacesMessage::Toast(Self::activation_toast(outcome)),
        ))
    }

    /// What the Settings list reports for the add form's activation write: the
    /// write's success, or its failure.
    fn activation_toast(outcome: Result<String, String>) -> ToastMessage {
        match outcome {
            Ok(name) => ToastMessage::SuccessMsg(format!("Active workspace: {name}")),
            Err(e) => ToastMessage::Error(e),
        }
    }

    /// Propagate the global workspace selection to the Home page, then push the
    /// resolved workspace to the workspace-scoped surfaces (see
    /// [`Self::sync_workspace_surfaces`]).
    ///
    /// The ticket sidebar is deliberately not one of the surfaces this touches:
    /// it lists every workspace's tickets — never the selected project's — so a
    /// switch, including one onto the personal folder, must leave it (its search
    /// included) exactly as it is.
    fn propagate_workspace_selection(&mut self, name: &str) -> Task<Message> {
        // Notify the Home page of the selection as stored, with no
        // substitution, so the composer's send carries the same choice the
        // picker and the DB carry. The conversation Home shows is the admin's
        // own and is not touched by it.
        let home_task: Task<Message> =
            Task::done(home::HomeMessage::WorkspaceChanged(name.to_string())).map(Message::Home);

        let surfaces = self.sync_workspace_surfaces();
        Task::batch([home_task, surfaces])
    }

    /// Push the resolved workspace to the four workspace-scoped surfaces — the
    /// editor, the shell, the footer git status and the change/diff view — and
    /// leave a page that stops resolving (see [`Self::leave_unavailable_page`]).
    /// No substitution of any kind happens here: an unresolved selection is
    /// never shown the personal workspace.
    ///
    /// The editor and the shell own per-workspace tab state, so they are only
    /// ever pushed a real resolution; while nothing is resolved the Dashboard
    /// hides their pages and drops the editor's subscription instead of tearing
    /// their tabs down. The footer git status and the change/diff view hold
    /// derived state only, so they are cleared — that is what drops the
    /// watcher, the refresh loop and the diff tick of a workspace that is no
    /// longer selected.
    ///
    /// An unchanged resolution is a no-op: the workspace-map reload path calls
    /// this on every CDC event and the push must not run then.
    fn sync_workspace_surfaces(&mut self) -> Task<Message> {
        let resolved = self
            .resolved_workspace()
            .map(|(name, path)| (name.to_string(), path.to_string()));
        // The change/diff view is an overlay of the current page: while nothing
        // is resolved it is closed rather than left open and empty (it would
        // keep owning Escape and the diff keyboard shortcuts).
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

    /// Return to Home when the current page is one the dashboard no longer
    /// offers (a workspace-scoped page whose selection stopped resolving).
    /// [`Self::navigate_to`] refuses such a page, so leaving it open would mean
    /// showing a page nothing can navigate to.
    fn leave_unavailable_page(&mut self) -> Task<Message> {
        if self.page_available(self.page) {
            return Task::none();
        }
        self.navigate_to(Page::Home)
    }

    /// Reload the full workspace map from storage (e.g. after add/delete on
    /// the Workspaces page or a post-toggle refresh). The applied map re-pushes
    /// the workspace-scoped surfaces and fills in the pre-boot personal default;
    /// it never picks the active project from itself (see
    /// [`Message::WorkspacesReloaded`]).
    ///
    /// The reload carries the generation it was issued under: this read retries
    /// for a second on a transient failure, so without it a delayed failure
    /// could re-set the error over a newer read that has already succeeded.
    fn reload_workspace_map(&mut self) -> Task<Message> {
        self.workspaces_reload_gen = self.workspaces_reload_gen.wrapping_add(1);
        let generation = self.workspaces_reload_gen;
        Task::perform(load_workspace_map(generation), std::convert::identity)
    }

    /// Apply the boot workspace-map read. There is no previous map to keep, so a
    /// failure leaves an empty map beside the failure that stands in for it and
    /// the boot path completes either way. The Settings list is re-derived here,
    /// so it can never diverge from the map.
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

    /// Apply a post-ready workspace-map reload: a failed read keeps the map the
    /// live selection may still resolve against, while the failure is recorded
    /// for the picker and the Settings list.
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

    /// Whether a dashboard-level overlay modal is open over the current page.
    /// Gates the editor page's keyboard ownership and the rail-replacing
    /// shortcuts — the overlay covers the rail, so the keys that replace rail
    /// entries do nothing from under one. The board ticket modal is not
    /// included: it renders only inside the Home page content (never the global
    /// overlay stack), so it cannot sit above the editor.
    ///
    /// `pending_stop` is likewise excluded: it is rendered inside
    /// the Running Agents page body, not as an overlay over other pages.
    fn overlay_modal_open(&self) -> bool {
        self.foreign_workspace.is_some()
            || self.show_diff_modal
            || self.show_update_confirm
            || self.git_state.is_modal_open()
    }

    /// The Home page: the chat, and the ticket column beside it while the board
    /// has a ticket to show. The chat is the row's first child either way — a
    /// root that flipped shape with the column would be rebuilt on the flip,
    /// taking the chat's scroll position and the composer's caret and draft with
    /// it — and a lone fill portion still spans the row.
    fn home_page(&self) -> Element<'_, Message> {
        let home_view = self.home_state.view(self.draining).map(Message::Home);
        // Wrap chat area in a right-click context menu with "Reset session".
        // Per-bubble ContextMenus (Copy message, see home.rs) capture bubble
        // right-clicks; this outer menu is the fallback for empty-space ones.
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
            // Sidebar right-click menu: the two bulk actions over the board —
            // archiving done and cancelled tickets, pausing queued ones.
            // Pausing is offered only when a queued ticket is listed.
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
            return container(
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
            .into();
        }

        if !self.ready {
            return container(
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
            .into();
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

        // Keep Stack widget type stable to prevent state loss on toast
        // transitions. A type tag change (Column ↔ Stack) would cause Iced
        // to destroy the entire widget tree, losing scroll positions, cursor
        // states, and all other widget state.
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
                    // Clears the footer bar; with the bar hidden the stack keeps
                    // only the window's own inset instead of a vacated strip.
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

        // ── Diff modal overlay ─────────────────────────────────────
        // The picked workspace's view is the default; a sidebar ticket card
        // from another project opens the read-only foreign view instead, and
        // the two are mutually exclusive. `sync_workspace_surfaces` closes the
        // picked one when its resolution is lost and never touches the foreign
        // one.
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

        // ── Branch management modal (small centered dialog) ──────────
        let branch_overlay: Element<'_, Message> = if self.git_state.is_modal_open() {
            let dialog = dialog::dialog_shell(self.git_state.view().map(Message::Git), 400.0, 24.0);
            widgets::modal_backdrop(dialog, Message::Git(git::GitMessage::CloseModal), 0.5)
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
/// width using a `FillPortion(1/8/1)` row layout. The dialog rectangle is
/// click-claimed (via [`widgets::dialog_click_guard`]) so interior clicks do
/// not fall through to the backdrop, while the surrounding margin still does.
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

/// Render the diff modal (80% width, 100% height, centered).
///
/// `foreign_workspace` is `Some(name)` while the modal shows another project's
/// commit (opened from a sidebar ticket card). That view is read-only and its
/// Back button leaves it for the picked workspace, so the commit header names
/// the project: it must never be taken for the picked workspace's history.
fn render_diff_modal<'a>(
    diff_state: &'a diff::DiffState,
    foreign_workspace: Option<&'a str>,
    wrap: fn(diff::DiffMessage) -> Message,
) -> Element<'a, Message> {
    let viewing_commit = diff_state.is_viewing_commit();

    // Outer header: commit message (large, bold) + short hash (muted) for
    // historical commits, or "Uncommitted changes" for working-tree diff.
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

// ── Ticket sidebar (Home page, right side) ────────────────────────

/// Ticket sidebar shown on the right side of the Home page.
/// Displays every workspace's non-archived tickets, grouped by phase — absent
/// entirely while there is no such ticket to show
/// ([`board::BoardState::has_nothing_to_show`]), so the chat takes the whole
/// width. A right-click context menu on this panel offers the two bulk actions
/// over that list — archiving its done and cancelled tickets, pausing its queued
/// ones.
fn ticket_sidebar(board_state: &board::BoardState) -> Element<'_, Message> {
    let search_active = !board_state.search_query.text().is_empty();

    // ── Search input row ───────────────────────────────────────────
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
        .padding([theme::PAD_8, theme::PAD_4])
        .width(Length::Fill)
        .height(Length::Fill)
        .style(theme::surface_container_style)
        .into()
}

/// Hint line for empty/loading ticket-list states.
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

/// Render the normal ticket list partitioned into groups (In Progress,
/// Queued, Pending, Completed).
fn render_normal_ticket_list(board_state: &board::BoardState) -> Element<'_, Message> {
    // A failed list read must show the failure in the body itself — otherwise
    // a first failure renders "Loading…" forever and a later failure leaves a
    // stale list that looks current.
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

/// Render FTS search results as ticket cards in a scrollable list.
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

/// Render a group of tickets with a header label.
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
        // Window events are subscribed from the start, before boot completes —
        // a close request included: it must reach the dashboard in every state
        // (see `request_exit`). Resized also carries the start-up width decision
        // (see `StartupWidth`), and both it and Moved keep last_size and
        // last_position at the geometry the window really has, which a close
        // persists (iced does not replay missed window events).
        //
        // The platform's quit requests are subscribed from the start for the
        // same reason: the platform can ask before boot has finished.
        let always_on = iced::Subscription::batch([
            window::resize_events()
                .map(|(id, size)| Message::WindowEvent(id, window::Event::Resized(size))),
            window::events().filter_map(|(id, event)| match &event {
                window::Event::Moved(_) | window::Event::FileDropped(_) => {
                    Some(Message::WindowEvent(id, event))
                }
                // The window is created after this subscription is registered, so
                // its first `Opened` event arrives here. Windows needs the id it
                // carries: it is the window's handle, and the taskbar/Alt-Tab icon
                // has to be attached to it (see `crate::app_icon`).
                #[cfg(target_os = "windows")]
                window::Event::Opened { .. } => Some(Message::WindowOpened(id)),
                _ => None,
            }),
            window::close_requests().map(Message::CloseRequested),
            iced::Subscription::run(quit_requests_subscription),
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
            // The editor holds a workspace-scoped page: while nothing is
            // resolved it is given no subscription at all — no tick and no tree
            // ping may act on the workspace it still holds in memory (its tabs
            // are deliberately kept).
            if self.page_available(Page::Editor) {
                self.editor_state
                    .subscription(self.overlay_modal_open())
                    .map(Message::Editor)
            } else {
                iced::Subscription::none()
            },
            self.home_state.subscription().map(Message::Home),
            // The Automations page and the Settings page's custom-tools area re-read
            // their source while they are shown; the tick is page-gated so a
            // page nobody is looking at reads nothing.
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
            // Git file-change subscription: drives the event-driven local git
            // footer refresh when workspace files change.
            iced::Subscription::run(git_file_changes_subscription),
            // Git pipeline-commit subscription: refreshes the footer promptly
            // after a pipeline auto-commit (ref-only change the watcher misses).
            iced::Subscription::run(git_commit_subscription),
            // Standalone 60s remote-sync subscription (replaces the tick duty;
            // also the recovery path for ref-only/commit-only changes).
            iced::time::every(git::REMOTE_REFRESH_INTERVAL)
                .map(|_| Message::Git(git::GitMessage::PeriodicSyncTick)),
            // Board change-stream subscription: drives the board ticket list from
            // CDC deltas instead of a 1s full re-poll.
            iced::Subscription::run(board_change_subscription),
            // Workspaces change-stream subscription: a full map reload whenever
            // a workspaces row changes (or the stream lags / the baseline fires).
            iced::Subscription::run(workspaces_cdc_subscription),
            // Users / user_channels change-stream subscriptions: re-list the
            // Settings users page when a users/user_channels row changes (or
            // the stream lags — full re-list is its own recovery).
            iced::Subscription::run(users_cdc_subscription),
            iced::Subscription::run(user_channels_cdc_subscription),
            // Runtime-change subscription: coalesces agent/non-agent registry,
            // live transcript, and voice-status broadcasts into a single
            // re-render, driving the Running Agents page and footer.
            iced::Subscription::run(runtime_change_subscription),
            // TTS download progress subscription (always active while ready).
            #[cfg(target_os = "macos")]
            iced::Subscription::run(tts_download_subscription).map(Message::TtsDownloadEvent),
            // Diff modal subscription (keyboard shortcuts, auto-refresh).
            // Only active when the modal is open to avoid intercepting
            // global keyboard shortcuts unnecessarily. The foreign commit view
            // owns it while it is the one shown, so the two never both
            // subscribe.
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

/// Subscription that emits [`Message::Shutdown`] when the global shutdown
/// token fires (self-update restart, SIGTERM/SIGINT, the end of a drain), and
/// [`Message::DrainStarted`] when the graceful-drain window begins (first
/// signal / window-close / quit / self-update) — the token does NOT fire on the
/// first signal, so the GUI needs a separate drain signal to disable input
/// while the drain runs.
fn shutdown_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    iced::stream::channel(1, |mut output: mpsc::Sender<Message>| async move {
        let token = crate::shutdown::shutdown_token();
        // Wait for the drain flag OR the token (a drain always precedes the
        // token in the two-signal protocol, but the token may fire without a
        // drain on force-cancel paths).
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

/// Record `route` as the exit route, where the ask has wording of its own — a platform quit passes
/// its label, a window close passes `None`, and this is the one home of the rule that `None` leaves
/// the record alone. Nothing is shown for it (the exit path's line is dropped with the iced
/// runtime — see `shutdown::EXIT_TRIGGER`), so it is set for accuracy, not for reporting.
fn record_exit_route(route: Option<&'static str>) {
    if let Some(route) = route {
        crate::shutdown::record_exit_trigger(route);
    }
}

/// Subscription that turns an ordinary quit the platform's own application machinery delivered
/// — one of the asks [`Message::QuitRequested`] names — into that message.
///
/// The source is the process-lifetime channel the quit source fills
/// ([`crate::shutdown::install_stop_request_sources`], called before the iced application runs).
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

/// Subscription that emits [`crate::audio::tts::TtsDownloadEvent`]s from the global
/// TTS download broadcast channel, forwarded as [`Message::TtsDownloadEvent`].
///
/// Unlike the hand-rolled channel loop this replaces, an uninitialized
/// `DOWNLOAD_EVENTS` source yields an empty stream instead of panicking
/// (unreachable in practice — `tts::init_global` precedes GUI startup).
#[cfg(target_os = "macos")]
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

/// Subscription that emits [`Message::RuntimeChanged`] (coalesced) when any
/// runtime source the GUI renders directly changes: agent/non-agent registries,
/// live transcript content, or voice status. The [`RuntimeEvent`] broadcast is
/// initialized in `main` before the iced application runs (matching the
/// [`LOG_BROADCAST`] convention); an uninitialized source would yield an
/// immediately-ending stream, which iced's subscription tracker does not
/// re-spawn. A burst of per-message / per-tool-call events collapses into a
/// single trailing [`Message::RuntimeChanged`] per [`RUNTIME_REFRESH_COALESCE`]
/// window — only "something changed" matters for a re-render.
fn runtime_change_subscription() -> impl futures_util::Stream<Item = Message> {
    common::coalesced_broadcast_producer(
        128,
        &crate::runtime_events::RUNTIME_EVENTS,
        RUNTIME_REFRESH_COALESCE,
        || Message::RuntimeChanged,
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

/// Subscription that emits [`Message::WorkspacesCdcChanged`] for a full
/// workspace-map reload. The source is the shared [`crate::db::cdc`] workspaces
/// sender, warmed in `main` before the iced app runs (same immediately-ending-
/// stream rationale as the board sender — Iced does not re-spawn an ended
/// subscription).
///
/// The stream opens with one synthetic BASELINE event, so a full read happens
/// at stream start regardless of upstream activity. Because this subscription
/// only exists in the post-boot (`ready`) branch of [`Dashboard::subscription`],
/// the baseline is delivered after the `Boot` arm has set `ready` — it can
/// never be swallowed by the pre-boot no-op catch-all. Every received delta AND
/// a lagged channel map to the same message (a full reload is its own Lagged
/// recovery).
fn workspaces_cdc_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    futures_util::stream::once(async { Message::WorkspacesCdcChanged }).chain(
        common::broadcast_stream_producer(
            128,
            crate::db::cdc::workspace_sender_lock(),
            |output: &mut mpsc::Sender<Message>, _event: Option<crate::db::cdc::ChangeEvent>| {
                Box::pin(async move {
                    // Awaited send (backpressure): a dropped reload trigger
                    // would leave the sidebar stale until the next event.
                    let _ =
                        futures_util::SinkExt::send(output, Message::WorkspacesCdcChanged).await;
                })
            },
        ),
    )
}

/// Subscription that emits [`Message::UsersCdcChanged`] when a users-table row
/// changes or the stream lags. Both map to the same message: the full users
/// re-list and the DB read-back of the active workspace (the same single choice
/// the footer picker writes) that follow are each their own recovery.
///
/// The stream opens with one synthetic BASELINE event, like
/// [`workspaces_cdc_subscription`]: a switch written between the boot selection
/// read and this subscription's first poll would otherwise be missed for the
/// rest of the run, leaving the dashboard showing a project the account no
/// longer has active. The subscription only exists in the post-boot (`ready`)
/// branch of [`Dashboard::subscription`], so the baseline can never be
/// swallowed by the pre-boot no-op catch-all. Source is the shared
/// [`crate::db::cdc`] users sender, warmed in `main` before the iced app runs.
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

/// Subscription that emits [`Message::UsersCdcChanged`] when a user_channels
/// row changes or the stream lags — the Settings users page re-lists on either
/// (a channel bind/unbind updates the user_channels table). Source is the shared
/// [`crate::db::cdc`] user_channels sender, warmed in `main` before the iced app
/// runs.
fn user_channels_cdc_subscription() -> impl futures_util::Stream<Item = Message> {
    use iced::futures::channel::mpsc;
    common::broadcast_stream_producer(
        128,
        crate::db::cdc::user_channels_sender_lock(),
        |output: &mut mpsc::Sender<Message>, _event: Option<crate::db::cdc::ChangeEvent>| {
            Box::pin(async move {
                // Awaited send: a dropped trigger would leave the list stale
                // until the next event (same rationale as the board stream).
                let _ = futures_util::SinkExt::send(output, Message::UsersCdcChanged).await;
            })
        },
    )
}

/// A running indicator — a role's agent count, or the in-flight non-agent call
/// counter — ready for either place it is drawn: its content, and the wording its
/// tooltip carries.
struct RunningIndicator {
    content: Element<'static, Message>,
    label: String,
}

impl RunningIndicator {
    /// The footer's presentation: in the right-hand indicator row, tooltip above.
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

    /// The rail's presentation: a full-width rail entry, tooltip to the side
    /// like every other entry.
    fn rail(self) -> Element<'static, Message> {
        Dashboard::sidebar_rail_item(
            self.content,
            self.label,
            Some(Message::Navigation(Page::RunningAgents)),
        )
    }
}

// ── Navigation sidebar ──────────────────────────────────────────

/// Map a [`Page`] variant to its corresponding Lucide icon element, unstyled so
/// each call site sizes and colors it.
///
/// Exhaustive match — adding a new `Page` variant produces a compile error
/// until its icon is assigned here.
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

impl Dashboard {
    /// The nav rail: the top nav icons, the voice status while the footer is
    /// hidden, the stretch in between ([`Self::render_sidebar_mid`]) and the
    /// bottom nav icons.
    fn sidebar_view(&self, footer_hidden: bool) -> Element<'_, Message> {
        let mut col = Column::new()
            .spacing(theme::SPACE_2)
            .push(self.render_sidebar_nav());
        if footer_hidden {
            col = col.push(Self::render_sidebar_voice());
        }
        container(
            col.push(Self::render_sidebar_mid(footer_hidden))
                .push(self.render_sidebar_bottom_nav()),
        )
        .width(Length::Fixed(SIDEBAR_RAIL_WIDTH))
        .height(Length::Fill)
        .style(theme::surface_container_style)
        .padding(theme::PAD_4)
        .into()
    }

    /// The rail's stretch between the nav icons and the bottom nav. With the
    /// footer hidden it carries the running-agent indicators, bottom-aligned so
    /// the column sits directly above the bottom nav; it fills the free height
    /// and scrolls inside its own area, so the nav entries keep their size
    /// however many roles run. Otherwise it is the plain spacer that keeps the
    /// two nav groups apart.
    fn render_sidebar_mid(footer_hidden: bool) -> Element<'static, Message> {
        if footer_hidden && let Some(agents) = Self::render_sidebar_agents() {
            return container(agents)
                .width(Length::Fill)
                .align_bottom(Length::Fill)
                .into();
        }
        Space::new().height(Length::Fill).into()
    }

    /// A rail entry: its content centered in the rail's full width, tooltip to
    /// the side — iced snaps Top tooltips into the viewport, overlapping the
    /// topmost rail entry. `on_press: None` makes a static entry (the voice
    /// status, which only reports).
    fn sidebar_rail_item(
        content: Element<'static, Message>,
        label: impl Into<Cow<'static, str>>,
        on_press: Option<Message>,
    ) -> Element<'static, Message> {
        tooltip(
            button(
                // The container's generous vertical padding gives the button a
                // large hit area within the fixed 56px rail.
                container(content)
                    .width(Length::Fill)
                    .center_x(Length::Fill)
                    .padding([theme::PAD_10, 0.0]),
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

    /// Shared nav-icon column for the sidebar top and bottom navs. Each page is
    /// a 48px-tall full-width icon button (28px icon) with the active page
    /// accent-colored; a page that is not available ([`Self::page_available`]) is
    /// absent from the column entirely.
    fn sidebar_nav_column(&self, pages: &[Page]) -> Column<'_, Message> {
        let mut col = Column::new().spacing(theme::SPACE_2);
        for page in pages {
            if !self.page_available(*page) {
                continue;
            }
            let color = if self.page == *page {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            };
            col = col.push(Self::sidebar_rail_item(
                page_icon(*page).size(28).color(color).into(),
                page.label(),
                Some(Message::Navigation(*page)),
            ));
        }
        col
    }

    /// Sidebar top nav icons (see [`Page::sidebar_pages`]). Running Agents is not
    /// in the sidebar — it is reachable only via the running-agent indicators,
    /// which live in the footer bar or, with the footer hidden, in the rail
    /// ([`Self::render_sidebar_agents`]).
    fn render_sidebar_nav(&self) -> Element<'_, Message> {
        self.sidebar_nav_column(Page::sidebar_pages()).into()
    }

    /// Sidebar bottom nav icons (see [`Page::sidebar_bottom_pages`]).
    fn render_sidebar_bottom_nav(&self) -> Element<'_, Message> {
        self.sidebar_nav_column(Page::sidebar_bottom_pages()).into()
    }

    /// The rail's voice status, standing in for the footer bar's label while the
    /// footer is hidden: the status's glyph (see [`Self::voice_status_icon`]) with
    /// the bar's own wording as its tooltip. The bar's resting label is the one
    /// state the rail leaves out — an idle listener is the normal condition, not
    /// something to report from the navigation — and an empty slot then keeps its
    /// place, so no status change rebuilds the column below it.
    #[cfg(target_os = "macos")]
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
                .size(28)
                .color(theme::TEXT_MUTED)
                .into(),
            label,
            None,
        )
    }

    /// No audio subsystem, so no voice status to report.
    #[cfg(not(target_os = "macos"))]
    fn render_sidebar_voice() -> Element<'static, Message> {
        Space::new().height(0).into()
    }

    /// The rail's running-agent indicators — one per role with agents running,
    /// in the footer's order, then the in-flight non-agent call counter — as a
    /// column that scrolls once the rail cannot show them all. `None` when
    /// nothing is running: neither indicator renders, exactly as in the footer.
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
        // A scrollbar would be painted over the right edge of these 48px
        // entries; a half-visible row already shows there is more.
        Some(
            scrollable(Column::with_children(items).spacing(theme::SPACE_2))
                .direction(theme::hidden_vertical_scrollbar())
                .height(Length::Shrink)
                .width(Length::Fill)
                .into(),
        )
    }

    /// Footer per-workspace Maintainer toggle icon button.
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

    /// Footer per-workspace pipeline pause/unpause toggle icon button.
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

    /// The failure the footer picker must show in place of its dropdown: a
    /// failed workspace-map read that left nothing to pick. A read that failed
    /// but preserved a usable map keeps the working picker — the Settings
    /// workspace list shows that failure.
    fn picker_failure<'a>(&'a self, options: &[widgets::PickOption]) -> Option<&'a str> {
        if options.is_empty() {
            self.workspaces_error.as_deref()
        } else {
            None
        }
    }

    /// Footer workspace picker: managed workspaces only, persists via
    /// [`Self::select_workspace`]. A project also becomes active by being added
    /// and when the active one is deleted, but this is the only place the admin
    /// chooses one, so it is offered even when a single managed workspace
    /// exists. When a failed workspace-map read leaves nothing to pick, a static,
    /// non-selectable "Workspaces unavailable" pill replaces it in the same
    /// footprint. Returns `None` for the zero-managed-workspaces state.
    fn render_workspace_picker(&self) -> Option<Element<'_, Message>> {
        // Nothing left to pick: the failed read renders a static, non-selectable
        // failure pill in the picker's footprint. The error text lives only in
        // the tooltip — it must never become an option of the dropdown.
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
            // No managed workspaces exist — an inert placeholder dropdown would
            // suggest a choice that isn't there; hide the picker instead.
            return None;
        }
        // Guard the displayed selection: a dangling persisted value (the
        // workspace vanished from the map) renders no highlight; a personal
        // selection renders the placeholder.
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

    /// Render the self-update confirmation modal (opened by
    /// [`Message::UpdateBot`], confirmed by [`Message::ConfirmUpdate`]).
    /// Returns a type-stable placeholder when closed.
    ///
    /// The text truthfully reflects the real sequence for this copy: the update
    /// runs in the background while the system keeps working — nothing is paused
    /// during it — then in-flight work drains, databases are checkpointed, and the
    /// app restarts and resumes work automatically. The window closing is the
    /// success signal.
    fn render_update_confirm(&self) -> Element<'_, Message> {
        if !self.show_update_confirm {
            // Keep the Stack widget type stable across open/close transitions
            // (see `empty_stack_placeholder`): the open state is a Stack.
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

    /// Render the self-update button in the footer bar.
    /// Returns `None` when self-update is not available on this installation.
    ///
    /// Visibility is driven entirely by the shared availability cache — no
    /// platform gate is needed: a downloaded copy finds and puts this system's
    /// ready-made file at the standard location on every platform, and replacing a
    /// file that is in use there is the platform's own rename-aside, so a running
    /// copy is never in the way.
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
            .size(theme::TEXT_24)
            .color(update_color);
        Some(widgets::icon_tooltip_button(
            update_icon,
            tooltip_text,
            if clickable {
                Some(Message::UpdateBot)
            } else {
                None
            },
            theme::PAD_3,
            theme::button_text,
            tooltip::Position::Top,
        ))
    }

    /// Vertical divider between the footer activity chrome and git blocks.
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

    /// One icon+count cell of the footer's sync control.
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

    /// Render the git sync control: the commits a push would carry (with the
    /// destination the tooltip names) and the commits a pull would take from the
    /// branch's own update source.
    ///
    /// Visible whenever there is anything to push or pull — a branch that can be
    /// pushed is never hidden because its tracked upstream is fetch-only or
    /// absent, and a destination that could not be asked is reported with the
    /// commits the branch carries rather than as nothing to do — and whenever no
    /// remote can take the branch at all, so it can say why. A branch with
    /// somewhere to push and nothing to push or pull hides it, as before. The pull
    /// metric is shown only when pressing would actually pull: a publish skips the
    /// update step. Icons are lucide (not Unicode arrows) at 16px, the number
    /// text's size, so all elements share the same vertical baseline.
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
        // A spacer separates a push count from the pull count; with nothing to
        // push the row starts at the pull metric, so it must not lead.
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

    /// Render the git diff stats button (+X/−Y, clickable -> diff modal).
    /// Returns `None` when there are no non-zero changes (including no
    /// oversized/binary untracked files or cap omissions to report).
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

    /// Render the git block: divider, branch, sync, and diff stats.
    ///
    /// `None` while nothing is resolved: the path that
    /// [`Self::sync_workspace_surfaces`] owns *is* the resolution.
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

    /// Roles with agents running and how many of each, in role order — the one
    /// grouping every running-agent indicator is built from.
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

    /// One role's indicator: the role icon in its badge color, plus a `×N` count
    /// once more than one agent of the role runs. `count_size` sizes that count —
    /// the rail has less width for it than the footer.
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
    /// [`Self::running_role_indicator`]. `None` when none are in flight.
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
        Some(RunningIndicator {
            content,
            label: labels.join(", "),
        })
    }

    /// The active agent icons in the right side of the footer. `None` when no
    /// agents are running.
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

    /// The in-flight non-agent call counter next to the agent icons in the
    /// footer. `None` when none are in flight.
    fn render_non_agent_calls() -> Option<Element<'static, Message>> {
        Some(Self::non_agent_calls(theme::TEXT_14)?.footer())
    }

    /// Render the TTS download progress indicator in the centre of the footer bar.
    /// Shows the current file name and percentage (e.g. "duration_predictor.onnx 83%").
    #[cfg(target_os = "macos")]
    fn render_tts_download_progress(&self) -> Element<'_, Message> {
        let Some((file_name, progress)) = &self.tts_download_progress else {
            return Space::new().width(0).into();
        };
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let pct = (progress * 100.0).round() as u32;
        let label = format!("TTS: {file_name} {pct}%");
        footer_status_label(label, None)
    }

    /// The wording for a [`VoiceStatus`], shared by the footer bar's label and
    /// the rail's tooltip, without the glyph [`Self::voice_status_icon`] draws in
    /// front of it. `None` for voice off, which both surfaces hide.
    #[cfg(target_os = "macos")]
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
            VoiceStatus::Error(msg) => format!("Error: {msg}"),
        })
    }

    /// Render a compact voice status indicator in the footer bar: the status's
    /// glyph in front of its wording (see [`Self::voice_status_icon`]).
    ///
    /// Shows a compact indicator for every [`VoiceStatus`] the shared label
    /// covers (see [`Self::voice_status_label`]); [`VoiceStatus::Error`]
    /// displays the actual error string rather than a hardcoded message.
    #[cfg(target_os = "macos")]
    fn render_voice_status() -> Element<'static, Message> {
        let status = crate::audio::voice::get_status();
        let Some(label) = Self::voice_status_label(&status) else {
            return Space::new().width(0).into();
        };
        footer_status_label(label, Some(Self::voice_status_icon(&status)))
    }

    /// The glyph drawn in front of a [`VoiceStatus`]: `ear` for the microphone
    /// side, `pen-line` while a recording is being transcribed.
    #[cfg(target_os = "macos")]
    fn voice_status_icon(
        status: &VoiceStatus,
    ) -> iced::widget::text::Text<'static, iced::Theme, iced::Renderer> {
        if matches!(status, VoiceStatus::Transcribing) {
            lucide::pen_line::<iced::Theme, iced::Renderer>()
        } else {
            lucide::ear::<iced::Theme, iced::Renderer>()
        }
    }

    /// 42px footer bar — update/picker/toggle chrome (left) and active
    /// agents (right).
    fn footer_view(&self) -> Element<'_, Message> {
        let mut left_elements: Vec<Element<'_, Message>> = Vec::with_capacity(5);

        if let Some(el) = Self::render_update_button() {
            left_elements.push(el);
        }
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

        // TTS download progress and voice status indicator (center of footer bar)
        #[cfg(target_os = "macos")]
        let center = row![
            self.render_tts_download_progress(),
            Self::render_voice_status(),
        ]
        .spacing(theme::SPACE_8)
        .align_y(Alignment::Center);
        // Off macOS there is no audio subsystem, so the centre slot collapses to
        // nothing — the footer row below is shared and stays untouched.
        #[cfg(not(target_os = "macos"))]
        let center = Space::new().width(0);

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

/// Muted 12px status label with side padding, shown in the centre of the footer
/// bar, with an optional glyph in front of it.
#[cfg(target_os = "macos")]
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

/// Persisted window geometry: the last size and position the window reported.
#[derive(serde::Deserialize)]
pub struct WindowState {
    pub width: f32,
    pub height: f32,
    pub x: i32,
    pub y: i32,
    /// Whether a geometry was actually read from the state file, rather than
    /// falling back to [`WindowState::default`]: a saved geometry opens as it
    /// was saved, a fresh one as [`WindowState::opening_size`] derives it.
    #[serde(skip)]
    restored: bool,
}

impl WindowState {
    /// Position to use when restoring the window.
    #[expect(clippy::cast_precision_loss)]
    #[must_use]
    pub const fn position(&self) -> iced::window::Position {
        iced::window::Position::Specific(iced::Point::new(self.x as f32, self.y as f32))
    }

    /// The size the window opens at: a saved geometry as it was saved, a fresh
    /// one without the ticket column's share, so the chat's line length is the
    /// one it has beside the column in the default window (the board's first
    /// read then settles the one width decision, `StartupWidth`).
    #[must_use]
    pub fn opening_size(&self) -> iced::Size {
        let full = iced::Size::new(self.width, self.height);
        if self.restored {
            return full;
        }
        iced::Size::new(
            (full.width - ticket_column_share(full.width)).max(MIN_WINDOW_WIDTH),
            full.height,
        )
    }
}

/// The window's default size in px, sized for the Home page with its ticket
/// column beside the chat.
const DEFAULT_WINDOW_WIDTH: f32 = 1500.0;
const DEFAULT_WINDOW_HEIGHT: f32 = 800.0;

/// The window's minimum size, shared with the platform settings so the one
/// width the app derives ([`WindowState::opening_size`]) stays a usable window.
pub const MIN_WINDOW_WIDTH: f32 = 800.0;
pub const MIN_WINDOW_HEIGHT: f32 = 500.0;

/// The width the ticket column takes of a window `width` px wide: its portion
/// of the content area beside the nav rail, which is exactly what the chat
/// gains back when the column is absent.
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

/// The window geometry file in the storage root.
const WINDOW_STATE_FILE: &str = "window-state.json";

/// Read persisted window state, falling back to defaults when it is missing or unreadable.
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

/// Save current window geometry.
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

/// Reload the full workspace map from the store after boot (CDC workspaces
/// event, stream lag, toggle completion, or a settings add/delete).
///
/// On a transient read failure — the one scoped retry in this reload path — it
/// retries once after a short delay; if the retry also fails, it returns
/// [`Message::WorkspacesReloadFailed`] so the current (non-empty) map is
/// preserved and the failure is rendered instead of silently vanishing.
async fn load_workspace_map(generation: u64) -> Message {
    // Transient read failure: retry once after a short delay, then report the
    // failure while keeping the current map.
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

/// Read the full workspace map from the store (one row per name).
async fn read_workspace_map() -> anyhow::Result<HashMap<String, Workspace>> {
    let list = crate::workspace::store().list().await?;
    Ok(list.into_iter().map(|ws| (ws.name.clone(), ws)).collect())
}

/// Resolve the boot selection against a freshly loaded map: the stored name when
/// it is an existing project, otherwise
/// [`crate::users::fallback_workspace_name`]'s pick — the one a deletion makes
/// too — otherwise the personal name, which is the answer only when the map
/// carries no project at all.
///
/// `prev_selection` is [`crate::users::resolve_selected_workspace_name`]'s
/// answer, so `None` is a failed read: it resolves like any other unusable value
/// (the boot path must complete) and only decides whether the pick is stored.
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

/// Load the full workspace map during boot and resolve the active project from
/// the DB (`users.selected_workspace` for the admin — the single source of truth)
/// via the admin-aware [`crate::users::resolve_selected_workspace_name`]
/// primitive, together with the projects that actually exist. An empty or
/// dangling stored selection lands on a project whenever one exists; the personal
/// name only when none does.
///
/// `remember` is set when a successful stored-selection read differs from the
/// resolved pick (see [`Message::BootWorkspaces`]).
///
/// The read result travels to [`Dashboard::apply_boot_workspaces`] as-is: a
/// failed boot read still completes with an empty map (nothing to land on), but
/// the failure is rendered by the workspace picker / Settings list instead of
/// being indistinguishable from an installation with no workspaces.
///
/// `generation` is captured when the read is dispatched and travels back with
/// it, so a reload that answered meanwhile supersedes this read.
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

/// Read the admin's stored active workspace back from the DB (the admin-aware
/// resolution: a NULL/personal value yields the canonical `personal:{user}`
/// name). `None` — the read failed; the live selection is kept.
async fn read_active_workspace() -> Option<String> {
    crate::users::resolve_selected_workspace_name(crate::users::ADMIN_USER_NAME).await
}

/// Build the footer workspace picker options from the managed-workspace map:
/// values sorted by name, labeled by each workspace's display name. A personal
/// workspace is never an option — a lingering `personal:{user}` row is filtered
/// out.
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

/// Reveal a path in the platform's file manager: Finder on macOS, Explorer on
/// Windows, and the containing folder on Linux (where `xdg-open` has no select
/// verb). A directory is accepted too — macOS and Windows select it in its
/// parent, Linux just opens that parent. The launchers report success and
/// failure unreliably, so their outcome is never trusted: a caller that
/// requires a real file checks for it beforehand.
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

/// Open an address with the platform's own handler (fire-and-forget).
///
/// The address is handed over as data — never to something that reads it as a
/// command line.
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
/// `ShellExecuteExW` resolves the address against the registered protocol and
/// file associations, and takes it whole in `lpFile`, with `lpParameters` NULL
/// and `lpVerb` NULL (the system's default verb for the item — forcing `open`
/// would change what some addresses do). The address is therefore one opaque
/// value: `&`, `^`, `%`, `"`, `?`, `|` or a space in it is data, never a command,
/// a second command or an argument of one. `SEE_MASK_DOENVSUBST` is left out of
/// `fMask` too, so the shell does not expand a `%NAME%`-shaped address either.
///
/// The launch runs on a throw-away thread because `ShellExecuteExW` is a
/// synchronous shell call and `open_url` is called on the iced event-loop thread,
/// which must not wait on the shell. The thread is detached, so a process that
/// exits right after the click takes the launch with it.
///
/// An empty or whitespace-only address is not an address: it never reaches the
/// shell, because there is nothing to open and nothing may be launched for one.
#[cfg(target_os = "windows")]
fn shell_open(address: &str) {
    if address.trim().is_empty() {
        return;
    }
    let address = address.to_owned();
    // The NUL-terminated UTF-16 spelling the Win32 API takes; an interior NUL in
    // the address would end the string the shell sees there.
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

/// One `ShellExecuteExW` call for the NUL-terminated UTF-16 `address`, on a
/// thread whose COM apartment has just been initialised for it. Returns the
/// `GetLastError` code when the shell did not open the address.
///
/// The apartment is the single-threaded one the API documents for itself (some
/// of the shell extensions it may activate require it). `SEE_MASK_NOASYNC` is
/// the one `fMask` flag the call needs: the API requires it of a caller without
/// a message loop that does not outlive the call. No `SEE_MASK_*` suppression
/// flag is set — whatever the shell shows for an address it cannot open is its
/// own, and the `GetLastError` code is logged alongside it.
#[cfg(target_os = "windows")]
fn shell_execute_address(address: &[u16]) -> Option<u32> {
    use windows_sys::Win32::Foundation::{FALSE, GetLastError};
    use windows_sys::Win32::System::Com::{
        COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE, CoInitializeEx, CoUninitialize,
    };
    use windows_sys::Win32::UI::Shell::{SEE_MASK_NOASYNC, SHELLEXECUTEINFOW, ShellExecuteExW};
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    // `CoInitializeEx` takes the flags as an unsigned word, while `windows-sys`
    // types the constants as the signed `COINIT` alias.
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
        // SAFETY: pairs with the successful `CoInitializeEx` above, on the same
        // thread.
        unsafe { CoUninitialize() };
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::board::TicketPhase;
    use crate::util::test::TicketFixture;

    /// Build a minimal [`Workspace`] with a name + derived path for map fixtures.
    fn ws(name: &str) -> Workspace {
        Workspace {
            name: name.to_string(),
            path: format!("/p/{name}"),
            ..Default::default()
        }
    }

    /// A ready dashboard so post-ready handlers (e.g. `WorkspacesReloaded`)
    /// run instead of the `!ready` guard.
    fn ready_dashboard() -> Dashboard {
        let mut dash = Dashboard::loading(&WindowState::default());
        dash.ready = true;
        dash
    }

    /// The board's read outcome for a board with no ticket, as the refresh the
    /// dashboard dispatched would report it.
    fn empty_board() -> board::BoardMessage {
        board::BoardMessage::Refreshed {
            tickets: Vec::new(),
            generation: 0,
        }
    }

    /// The board's read outcome for a board holding one ticket.
    fn board_with_a_ticket() -> board::BoardMessage {
        board::BoardMessage::Refreshed {
            tickets: vec![TicketFixture::new("T-1", TicketPhase::Backlog).build()],
            generation: 0,
        }
    }

    fn resize(width: f32, height: f32) -> Message {
        Message::WindowEvent(
            iced::window::Id::unique(),
            window::Event::Resized(iced::Size::new(width, height)),
        )
    }

    /// A fresh window opens at the width the chat needs: the area the chat gets
    /// without the ticket column is the area it has beside the column in the
    /// default window, and it is never a window below the minimum.
    #[test]
    fn a_fresh_window_opens_without_the_ticket_column() {
        let fresh = WindowState::default().opening_size();
        let beside_column =
            DEFAULT_WINDOW_WIDTH - SIDEBAR_RAIL_WIDTH - ticket_column_share(DEFAULT_WINDOW_WIDTH);
        let without_column = fresh.width - SIDEBAR_RAIL_WIDTH;
        assert!(
            (without_column - beside_column).abs() < 0.01,
            "{without_column} vs {beside_column}"
        );
        assert!(fresh.width >= MIN_WINDOW_WIDTH && fresh.height >= MIN_WINDOW_HEIGHT);
    }

    /// The width the first board read resolves the window to: a board with
    /// nothing to show leaves the narrow opening as it is, one that draws the
    /// column takes the default width back, a failed read keeps both the column
    /// and the width, and the decision is taken once.
    #[test]
    fn the_start_up_width_follows_the_board() {
        let default_window = Some(iced::Size::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT));

        let mut dash = ready_dashboard();
        let _ = dash.board_state.update(empty_board());
        assert_eq!(dash.take_startup_width(), None, "the narrow opening fits");

        let mut dash = ready_dashboard();
        let _ = dash.board_state.update(board_with_a_ticket());
        assert_eq!(dash.take_startup_width(), default_window);
        assert_eq!(
            dash.take_startup_width(),
            None,
            "the decision is taken once"
        );

        let mut dash = ready_dashboard();
        let _ = dash.board_state.update(board::BoardMessage::RefreshError {
            error: "db down".to_string(),
            generation: 0,
        });
        assert_eq!(
            dash.take_startup_width(),
            default_window,
            "a failed read keeps the column, and the width with it"
        );
    }

    /// The dashboard takes the decision on a board read that applied; an outcome
    /// the board dropped as stale decides nothing — a newer read is still on its
    /// way.
    #[test]
    fn only_an_applied_board_read_settles_the_start_up_width() {
        let mut dash = ready_dashboard();
        let _ = dash.update(Message::Board(empty_board()));
        assert!(matches!(dash.startup_width, StartupWidth::Settled));

        let mut dash = ready_dashboard();
        let _ = dash.board_state.refresh(); // a newer read is in flight
        let _ = dash.update(Message::Board(empty_board()));
        assert!(matches!(dash.startup_width, StartupWidth::AwaitingReport));
    }

    /// The decision survives the platform's own reports of the size it gave the
    /// window — the requested width rounded to its pixel grid, then the same
    /// size again — and yields to a width the user drags before the first read.
    /// A height-only drag decides nothing: the decision is about the width.
    #[test]
    fn the_start_up_width_survives_the_creation_reports() {
        let mut dash = Dashboard::loading(&WindowState::default());
        let _ = dash.update(resize(1067.0, 800.0));
        let _ = dash.update(resize(1067.0, 800.0));
        assert!(
            matches!(dash.startup_width, StartupWidth::Owed),
            "still owed"
        );

        let mut dash = Dashboard::loading(&WindowState::default());
        let _ = dash.update(resize(1067.0, 800.0));
        let _ = dash.update(resize(1067.0, 700.0));
        assert!(
            matches!(dash.startup_width, StartupWidth::Owed),
            "a height-only drag is not a width"
        );

        let mut dash = Dashboard::loading(&WindowState::default());
        let _ = dash.update(resize(1067.0, 800.0));
        let _ = dash.update(resize(900.0, 800.0));
        assert!(
            matches!(dash.startup_width, StartupWidth::Settled),
            "the drag is the user's"
        );
    }

    /// The shape of a widget tree: every node's tag with its children's. iced
    /// reconciles positionally, so a node keeps its state across a redraw only
    /// while the subtrees under it keep this shape.
    fn tree_shape(tree: &iced::advanced::widget::tree::Tree) -> Shape {
        Shape {
            tag: tree.tag,
            children: tree.children.iter().map(tree_shape).collect(),
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Shape {
        tag: iced::advanced::widget::tree::Tag,
        children: Vec<Shape>,
    }

    /// The Home page keeps its shape when the ticket column comes and goes: the
    /// column is pushed after the chat, never around it, so the chat's subtree is
    /// identical either way and iced reconciles it in place — its scroll
    /// position and the composer's caret and draft survive the flip.
    #[test]
    fn the_ticket_column_does_not_reshape_the_home_page() {
        // The composer draws the voice status, which a started instance has.
        #[cfg(target_os = "macos")]
        let _ = crate::audio::voice::init_global();

        let mut dash = ready_dashboard();
        let _ = dash.update(Message::Board(empty_board()));
        let without = dash.home_page();
        assert_eq!(without.as_widget().children().len(), 1, "the chat alone");
        let chat = tree_shape(&without.as_widget().children()[0]);
        drop(without);

        let _ = dash.update(Message::Board(board_with_a_ticket()));
        let with = dash.home_page();
        assert_eq!(
            with.as_widget().children().len(),
            2,
            "the chat and the column"
        );
        assert_eq!(tree_shape(&with.as_widget().children()[0]), chat);
    }

    /// A saved geometry is its own answer: nothing is owed, so no board read
    /// resizes the window.
    #[test]
    fn a_saved_geometry_opens_as_saved() {
        let saved = WindowState {
            width: 1000.0,
            height: 700.0,
            x: 10,
            y: 10,
            restored: true,
        };
        let mut dash = Dashboard::loading(&saved);
        assert_eq!(dash.last_size, iced::Size::new(1000.0, 700.0));
        let _ = dash.update(Message::Board(board_with_a_ticket()));
        assert_eq!(dash.take_startup_width(), None);
    }

    /// The conversation's first load is dispatched by the app opening — it hangs
    /// off the screen opening, never off a workspace selection (the other routes
    /// are a session reset and a stream-lag re-sync).
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

    /// A `WorkspacesReloaded` for generation 0 — the generation a dashboard that
    /// has not reloaded yet carries (the reload guard drops anything older).
    fn reloaded(map: HashMap<String, Workspace>) -> Message {
        Message::WorkspacesReloaded {
            workspaces: map,
            generation: 0,
        }
    }

    /// [`reloaded`]'s failure counterpart.
    fn reload_failed(error: &str) -> Message {
        Message::WorkspacesReloadFailed {
            error: error.to_string(),
            generation: 0,
        }
    }

    #[test]
    fn workspaces_reloaded_replaces_map_and_defers_an_unknown_selection() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([
            ("ws1".to_string(), ws("ws1")),
            ("ws2".to_string(), ws("ws2")),
        ]);
        dash.selected_workspace_name = Some("ws2".to_string());
        let _ = dash.apply_workspace_selection("ws2");
        assert_eq!(dash.resolved_workspace(), Some(("ws2", "/p/ws2")));

        // A reload whose map lacks ws2 replaces the map, but leaves the live
        // selection standing — never substituted with the admin's Personal
        // workspace — and the now-unresolved surfaces are dropped with it: the
        // DB read-back, not this map, decides whether the selection survives.
        let new_map = HashMap::from([
            ("ws1".to_string(), ws("ws1")),
            ("ws3".to_string(), ws("ws3")),
        ]);
        let _ = dash.update(reloaded(new_map));
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws2"));
        assert_eq!(dash.resolved_workspace(), None);
        assert_eq!(dash.pushed_workspace, None);
        assert!(dash.workspaces.contains_key("ws1"));
        assert!(dash.workspaces.contains_key("ws3"));
        assert!(!dash.workspaces.contains_key("ws2"));

        // A reload whose map still contains the selection → unchanged. The map is
        // never consulted for the selection, so it survives whatever the
        // trigger-time snapshot was — the boot-race regression where a reload
        // triggered while the selection was still None must not reset it.
        dash.selected_workspace_name = Some("ws3".to_string());
        let new_map2 = HashMap::from([("ws3".to_string(), ws("ws3"))]);
        let _ = dash.update(reloaded(new_map2));
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws3"));
    }

    /// A switch made elsewhere (the `workspace` tool's `add`) reaches a running
    /// dashboard: the read-back applies the stored choice in memory, an
    /// unchanged value is a no-op, and a read superseded by a local pick or by a
    /// newer read is dropped — the two surfaces are one choice, and the running
    /// app follows it. A personal answer stands only when the map carries no
    /// project to show.
    #[test]
    fn active_workspace_sync_follows_an_external_switch_and_drops_a_superseded_read() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([
            ("ws1".to_string(), ws("ws1")),
            ("ws2".to_string(), ws("ws2")),
        ]);
        dash.selected_workspace_name = Some("ws1".to_string());

        // A switch made elsewhere — the `workspace` tool's `add` — is adopted
        // as soon as it is read back.
        let _ = dash.update(Message::ActiveWorkspaceSynced {
            generation: 0,
            name: Some("ws2".to_string()),
        });
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws2"));

        // A pick made here issues a write of its own, so a read dispatched
        // before it is stale by construction and must not revert the pick.
        let _ = dash.select_workspace("ws1");
        assert_eq!(dash.active_workspace_gen, 1);
        let _ = dash.update(Message::ActiveWorkspaceSynced {
            generation: 0,
            name: Some("ws2".to_string()),
        });
        assert_eq!(
            dash.selected_workspace_name.as_deref(),
            Some("ws1"),
            "a superseded read must not revert the live selection"
        );

        // A failed read changes nothing, and an unchanged value is a no-op
        // rather than a re-run of the propagate's pushes.
        let _ = dash.update(Message::ActiveWorkspaceSynced {
            generation: 1,
            name: None,
        });
        let _ = dash.update(Message::ActiveWorkspaceSynced {
            generation: 1,
            name: Some("ws1".to_string()),
        });
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws1"));

        // Two reads in flight: the one dispatched first cannot land last and
        // revert the newer answer, because the newer dispatch bumped the guard.
        let stale = dash.active_workspace_gen;
        dash.active_workspace_gen += 1; // a newer read is dispatched
        let _ = dash.update(Message::ActiveWorkspaceSynced {
            generation: stale,
            name: Some("ws2".to_string()),
        });
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws1"));

        // A stored managed name the loaded map does not hold is still adopted:
        // the map can simply be a reload behind (a workspace added and
        // selected at once) and the DB is authoritative about the choice, so the
        // read-back must not pin Personal here.
        assert!(!dash.workspaces.contains_key("late"));
        let _ = dash.update(Message::ActiveWorkspaceSynced {
            generation: dash.active_workspace_gen,
            name: Some("late".to_string()),
        });
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("late"));

        // And a reload whose map read predates that workspace must not settle the
        // desktop on Personal either — the two surfaces stay one choice until the
        // DB says otherwise.
        let _ = dash.update(reloaded(HashMap::from([("ws1".to_string(), ws("ws1"))])));
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("late"));

        // A personal answer while the map carries a project is the residue of a
        // failed start-up read, which writes nothing on purpose: adopting it would
        // drop the desktop outside every project, so the live pick stands.
        let _ = dash.update(Message::ActiveWorkspaceSynced {
            generation: dash.active_workspace_gen,
            name: Some("personal:admin".to_string()),
        });
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("late"));

        // With no project left to show, the personal answer is the right one.
        dash.workspaces.clear();
        let _ = dash.update(Message::ActiveWorkspaceSynced {
            generation: dash.active_workspace_gen,
            name: Some("personal:admin".to_string()),
        });
        assert_eq!(
            dash.selected_workspace_name.as_deref(),
            Some("personal:admin")
        );
    }

    /// The add form's activation: a pick applies in memory at once — superseding
    /// any in-flight read-back — while a name the loaded map does not carry still
    /// applies but is not persisted.
    #[test]
    fn activate_workspace_applies_at_once_and_skips_a_name_the_map_lacks() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([
            ("ws1".to_string(), ws("ws1")),
            ("ws2".to_string(), ws("ws2")),
        ]);
        dash.selected_workspace_name = Some("ws1".to_string());

        let generation = dash.active_workspace_gen;
        let _ = dash.activate_workspace("ws2");
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws2"));
        assert_eq!(dash.active_workspace_gen, generation + 1);

        // Not in the map: applied in memory (the DB read-back decides), nothing
        // is written.
        let _ = dash.activate_workspace("late");
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("late"));
        assert_eq!(dash.active_workspace_gen, generation + 2);

        // What may be written: a managed name only when the map carries it, a
        // personal space always (the map never carries one, and it is stored as
        // NULL — the product's own "no managed workspace selected").
        assert!(dash.selection_persistable("personal:admin"));
        assert!(dash.selection_persistable("ws2"));
        assert!(!dash.selection_persistable("late"));
    }

    #[test]
    fn workspaces_reloaded_keeps_the_boot_restored_selection() {
        // Models the boot race: a CDC baseline reload was triggered while the
        // selection was still None, but lands after `finish_boot` applied the
        // DB-restored selection ("mahbot"). The map is never consulted for the
        // selection, so it stands instead of being wiped to Personal.
        let mut dash = ready_dashboard();
        dash.selected_workspace_name = Some("mahbot".to_string());
        let map = HashMap::from([("mahbot".to_string(), ws("mahbot"))]);
        let _ = dash.update(reloaded(map));
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("mahbot"));
    }

    /// A boot pick that is written back supersedes any read-back already in
    /// flight, as a pick does. A pick produced by a failed read is written by
    /// nobody, so it supersedes none — an in-flight read may still correct it.
    #[test]
    fn a_boot_pick_supersedes_a_read_back_in_flight() {
        let mut dash = ready_dashboard();
        let map = HashMap::from([("ws1".to_string(), ws("ws1"))]);
        let before = dash.active_workspace_gen;

        let _ = dash.apply_boot_workspaces(Ok(map.clone()), "ws1", false, 0);
        assert_eq!(
            dash.active_workspace_gen, before,
            "nothing was written, so a read-back may still correct the pick"
        );
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws1"));

        let _ = dash.apply_boot_workspaces(Ok(map), "ws1", true, 0);
        assert_eq!(
            dash.active_workspace_gen,
            before + 1,
            "the write must supersede a read-back in flight"
        );
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws1"));
    }

    /// The boot race: the map read during boot is dropped as stale while the
    /// restored selection is applied, so the selection is resolved against an
    /// empty map. The workspace-scoped surfaces must stay empty — never the
    /// admin's Personal workspace — and must come back by
    /// themselves once the map arrives, with the selected NAME unchanged
    /// throughout.
    #[test]
    fn boot_without_the_workspace_map_pushes_nothing_then_recovers() {
        let mut dash = ready_dashboard();

        let _ = dash.apply_boot_workspaces(Ok(HashMap::new()), "ws1", false, 0);

        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws1"));
        assert_eq!(dash.resolved_workspace(), None);
        assert_eq!(
            dash.pushed_workspace, None,
            "an unresolvable selection must not be substituted with a workspace"
        );

        let _ = dash.update(reloaded(HashMap::from([("ws1".to_string(), ws("ws1"))])));

        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws1"));
        assert_eq!(dash.resolved_workspace(), Some(("ws1", "/p/ws1")));
        assert_eq!(
            dash.pushed_workspace
                .as_ref()
                .map(|(name, path)| (name.as_str(), path.as_str())),
            Some(("ws1", "/p/ws1")),
            "the reload that made the unchanged selection resolvable must re-push it"
        );
    }

    /// Only a managed workspace present in the loaded map with a usable path
    /// resolves; everything else — a personal selection however it arose, a
    /// name missing from the map, an entry without a path, no selection —
    /// resolves to nothing.
    #[test]
    fn only_a_managed_workspace_with_a_path_resolves() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([
            ("ws1".to_string(), ws("ws1")),
            (
                "pathless".to_string(),
                Workspace {
                    name: "pathless".to_string(),
                    ..Default::default()
                },
            ),
        ]);

        dash.selected_workspace_name = Some("personal:admin".to_string());
        assert_eq!(dash.resolved_workspace(), None);
        dash.selected_workspace_name = Some("gone".to_string());
        assert_eq!(dash.resolved_workspace(), None);
        dash.selected_workspace_name = Some("pathless".to_string());
        assert_eq!(dash.resolved_workspace(), None);
        dash.selected_workspace_name = Some(String::new());
        assert_eq!(dash.resolved_workspace(), None);
        dash.selected_workspace_name = None;
        assert_eq!(dash.resolved_workspace(), None);

        dash.selected_workspace_name = Some("ws1".to_string());
        assert_eq!(dash.resolved_workspace(), Some(("ws1", "/p/ws1")));
    }

    /// The editor page, the shell page, the footer git status and the diff view
    /// follow the resolution: gone while nothing resolves, back by themselves
    /// once it does — and a workspace-scoped page that stops resolving leaves
    /// the desktop on Home. Every step goes through the real selection entry
    /// point (the single push point for all of them), so what the dashboard
    /// derives from that push is asserted here.
    #[test]
    fn unresolved_selection_hides_the_workspace_scoped_surfaces() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([("ws1".to_string(), ws("ws1"))]);

        let _ = dash.apply_workspace_selection("ws1");
        assert!(dash.resolved_workspace().is_some());
        assert!(dash.render_git_block().is_some());
        let _ = dash.navigate_to(Page::Editor);

        // A personal selection never resolves.
        let _ = dash.apply_workspace_selection("personal:admin");
        assert_eq!(dash.resolved_workspace(), None);
        assert!(!dash.page_available(Page::Editor));
        assert!(!dash.page_available(Page::Shell));
        assert_eq!(
            dash.page,
            Page::Home,
            "a page nothing can navigate to is left for Home"
        );
        let _ = dash.navigate_to(Page::Editor);
        assert_eq!(dash.page, Page::Home, "and cannot be navigated back to");
        assert!(
            dash.page_available(Page::Home),
            "other pages stay available"
        );
        assert!(dash.render_git_block().is_none());
        let _ = dash.open_diff_modal(None);
        assert!(!dash.show_diff_modal, "the diff view must not open");

        // Neither does a selected name missing from the loaded map — the toggles
        // included: a stored selection naming a project that no longer exists
        // counts as no project.
        let _ = dash.apply_workspace_selection("gone");
        assert_eq!(dash.resolved_workspace(), None);
        assert!(!dash.page_available(Page::Editor));
        assert!(dash.render_git_block().is_none());

        // Nothing is offered for it either: the toggle has no write to make and
        // reports nothing.
        let toasts = dash.toasts.len();
        let _ = dash.update(Message::Toggle(ToggleKind::Pause));
        assert_eq!(dash.toasts.len(), toasts);

        let _ = dash.apply_workspace_selection("ws1");
        assert!(dash.resolved_workspace().is_some());
        assert!(dash.page_available(Page::Editor));
        assert!(dash.page_available(Page::Shell));
        assert!(dash.render_git_block().is_some());
        let _ = dash.open_diff_modal(None);
        assert!(dash.show_diff_modal);
    }

    /// Deleting a project reports where the desktop moved — or that it left every
    /// project — whenever it moved at all, whichever project was deleted; changed
    /// tabs, git state and toggles are otherwise the only sign of it.
    #[test]
    fn deleting_a_project_reports_where_the_desktop_moved() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([
            ("ws1".to_string(), ws("ws1")),
            ("ws2".to_string(), ws("ws2")),
            ("ws3".to_string(), ws("ws3")),
        ]);
        let _ = dash.apply_workspace_selection("ws1");
        let delete_result = |removed: &str, moved_to: Option<&str>| {
            Message::Settings(settings::SettingsMessage::WorkspaceMsg(
                workspaces::WorkspacesMessage::DeleteResult(Ok((
                    removed.to_string(),
                    moved_to.map(str::to_string),
                ))),
            ))
        };

        // The store left the selection where the desktop already is: silent.
        let _ = dash.update(delete_result("ws2", Some("ws1")));
        assert!(dash.toasts.is_empty(), "no move, nothing to report");
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws1"));

        // The store moved it elsewhere: the desktop follows at once and says where,
        // whichever project was deleted.
        let _ = dash.update(delete_result("ws2", Some("ws3")));
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws3"));
        assert_eq!(dash.resolved_workspace(), Some(("ws3", "/p/ws3")));
        assert!(matches!(
            dash.toasts.as_slice(),
            [Toast { message, kind: ToastKind::Success, .. }]
                if message == "Project ws2 removed — ws3 is active"
        ));

        // The same for the active project's own deletion, applied at once so the
        // desktop never passes through the no-project state while the map reload
        // and the read-back catch up.
        let _ = dash.update(delete_result("ws3", Some("ws1")));
        assert_eq!(dash.selected_workspace_name.as_deref(), Some("ws1"));
        assert!(matches!(
            dash.toasts.as_slice(),
            [.., Toast { message, .. }] if message == "Project ws3 removed — ws1 is active"
        ));

        // The last project leaves the desktop outside any workspace.
        let _ = dash.update(delete_result("ws1", None));
        assert_eq!(
            dash.selected_workspace_name.as_deref(),
            Some("personal:admin")
        );
        assert_eq!(dash.resolved_workspace(), None);
        assert!(matches!(
            dash.toasts.as_slice(),
            [.., Toast { message, .. }] if message == "Project ws1 removed — the personal folder is active"
        ));
    }

    /// An open diff view is closed when the resolution is lost — from then on
    /// nothing of that workspace may be displayed.
    #[test]
    fn losing_the_resolution_closes_the_diff_view() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([("ws1".to_string(), ws("ws1"))]);
        let _ = dash.apply_workspace_selection("ws1");
        let _ = dash.open_diff_modal(None);
        assert!(dash.show_diff_modal);

        let _ = dash.apply_workspace_selection("personal:admin");

        assert!(!dash.show_diff_modal);
        assert_eq!(dash.resolved_workspace(), None);
        assert_eq!(dash.pushed_workspace, None);
    }

    /// A sidebar ticket of another project opens its commit read-only and named,
    /// without re-pointing the picked workspace's own change view; the ticket's
    /// own project uses the ordinary view, and a project that no longer resolves
    /// is reported instead of falling back to another project's history.
    #[test]
    fn foreign_ticket_commit_diff_keeps_the_picked_view() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([
            ("ws1".to_string(), ws("ws1")),
            ("ws2".to_string(), ws("ws2")),
        ]);
        let _ = dash.apply_workspace_selection("ws1");

        let _ = dash.open_ticket_commit_diff("ws2", "abc1234".to_string());
        assert_eq!(dash.foreign_workspace.as_deref(), Some("ws2"));
        // The dashboard's own foreign state is the one that loaded: a state
        // rebuilt per open would be a different one, and its generation would
        // restart at 1 — a load left in flight from a previous view could then
        // land in this one.
        assert!(
            dash.foreign_diff_state.is_viewing_commit(),
            "the foreign view shows ws2's commit"
        );
        assert!(
            !dash.show_diff_modal,
            "the picked workspace's view must not be the one showing"
        );

        // A picker-driven workspace push is the picked view's own message: the
        // foreign view keeps the other project's commit on screen, and the
        // picked state — not the foreign one — is the one re-pointed.
        let _ = dash.update(Message::DiffModal(diff::DiffMessage::WorkspaceSelected(
            "/p/ws3".to_string(),
        )));
        assert_eq!(dash.foreign_workspace.as_deref(), Some("ws2"));
        assert!(
            dash.foreign_diff_state.is_viewing_commit(),
            "the foreign view must not be re-pointed at the picked workspace"
        );

        // "Back to working tree" in a foreign view returns to the picked
        // workspace, never to the other project's working tree.
        let _ = dash.update(Message::ForeignDiff(diff::DiffMessage::BackToWorkingTree));
        assert!(dash.foreign_workspace.is_none());
        assert!(dash.show_diff_modal);

        // A ticket of the picked workspace takes the ordinary view.
        let _ = dash.open_ticket_commit_diff("ws1", "abc1234".to_string());
        assert!(dash.foreign_workspace.is_none());
        assert!(dash.show_diff_modal);

        // A project the map does not know is looked up in the store, and
        // reported when it is gone — never substituted with another project's
        // history. The ordinary view that was open stands untouched.
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

    /// A workspace switch must not touch the ticket sidebar: it lists every
    /// workspace's tickets, so its load failure and an in-progress search — the
    /// query and the results it produced — must survive a switch untouched.
    #[test]
    fn workspace_switch_leaves_the_ticket_sidebar_untouched() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([("ws1".to_string(), ws("ws1"))]);
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

    /// A reload that lands after a newer one has answered is dropped: its
    /// failure must not re-set the error over the map the newer read produced
    /// (the reload retries for a second, so this window is real).
    #[test]
    fn stale_workspace_reload_outcomes_are_dropped() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([("ws1".to_string(), ws("ws1"))]);
        dash.workspaces_reload_gen = 2;

        let _ = dash.update(Message::WorkspacesReloadFailed {
            error: "db down".to_string(),
            generation: 1,
        });
        assert!(
            dash.workspaces_error.is_none(),
            "a stale failure must not re-set the error"
        );

        let _ = dash.update(Message::WorkspacesReloaded {
            workspaces: HashMap::from([("ws2".to_string(), ws("ws2"))]),
            generation: 1,
        });
        assert!(
            dash.workspaces.contains_key("ws1"),
            "a stale map must not replace the newer one"
        );
    }

    /// The boot resolution: an existing stored project is kept, while a stored
    /// value naming no existing project — including a failed read — lands on the
    /// project the desktop falls back to. The personal name is the answer only
    /// when the map carries no project at all.
    #[test]
    fn resolve_workspace_selection_keeps_a_project_and_never_lands_on_personal_while_one_exists() {
        let map = HashMap::from([
            ("ws1".to_string(), ws("ws1")),
            ("ws2".to_string(), ws("ws2")),
            ("personal:alice".to_string(), ws("personal:alice")),
        ]);
        let personal = "personal:alice";
        assert_eq!(
            resolve_workspace_selection(&map, Some("ws2"), personal),
            "ws2"
        );
        // A dangling selection and a failed read both land on the first project
        // in the footer switcher's order; a personal row is no candidate.
        assert_eq!(
            resolve_workspace_selection(&map, Some("gone"), personal),
            "ws1"
        );
        assert_eq!(resolve_workspace_selection(&map, None, personal), "ws1");
        assert_eq!(
            resolve_workspace_selection(&HashMap::new(), Some("gone"), personal),
            personal,
            "with no project at all the desktop is outside any workspace"
        );
    }

    /// A personal workspace is never a picker option — a lingering
    /// `personal:{user}` row in the map is filtered out.
    #[test]
    fn managed_workspace_options_never_offer_a_personal_workspace() {
        let map = HashMap::from([
            ("personal:admin".to_string(), ws("personal:admin")),
            ("ws2".to_string(), ws("ws2")),
            ("ws1".to_string(), ws("ws1")),
        ]);

        let options = managed_workspace_options(&map);

        assert_eq!(
            options.iter().map(|o| o.value.as_str()).collect::<Vec<_>>(),
            vec!["ws1", "ws2"],
            "managed workspaces only, ordered by name"
        );
    }

    #[test]
    fn workspaces_reloaded_syncs_settings_lists() {
        let mut dash = ready_dashboard();
        let map = HashMap::from([
            ("beta".to_string(), ws("beta")),
            ("alpha".to_string(), ws("alpha")),
        ]);
        let _ = dash.update(reloaded(map));

        // Settings workspace list = map values sorted by name.
        let ws_names: Vec<&str> = dash
            .settings_state
            .workspaces_state
            .workspaces
            .iter()
            .map(|w| w.name.as_str())
            .collect();
        assert_eq!(ws_names, vec!["alpha", "beta"]);
    }

    /// A failed workspace-map read is a persistent, visible state: the current
    /// map is preserved, the failure stays on the Dashboard for the picker and
    /// the Settings list, and a later successful reload clears it.
    #[test]
    fn workspaces_reload_failure_sets_and_clears_error() {
        let mut dash = ready_dashboard();
        dash.workspaces = HashMap::from([("ws1".to_string(), ws("ws1"))]);

        let _ = dash.update(reload_failed("db down"));

        assert_eq!(dash.workspaces_error.as_deref(), Some("db down"));
        assert!(
            dash.workspaces.contains_key("ws1"),
            "a failed reload must preserve the current map"
        );
        assert!(
            dash.picker_failure(&managed_workspace_options(&dash.workspaces))
                .is_none(),
            "a preserved map keeps the picker working — the pill is for a read that left nothing to pick"
        );

        let new_map = HashMap::from([("ws2".to_string(), ws("ws2"))]);
        let _ = dash.update(reloaded(new_map));

        assert!(dash.workspaces_error.is_none());
    }

    /// A failed boot read still completes boot: the map stays empty and the
    /// failure is carried so the picker / Settings list can render it.
    #[test]
    fn boot_workspace_read_failure_completes_with_error_state() {
        let mut dash = Dashboard::loading(&WindowState::default());

        let _ = dash.apply_boot_workspaces(
            Err("boot read failed".to_string()),
            "personal:admin",
            false,
            0,
        );

        assert_eq!(dash.workspaces_error.as_deref(), Some("boot read failed"));
        assert!(dash.workspaces.is_empty());
        assert_eq!(
            dash.picker_failure(&managed_workspace_options(&dash.workspaces)),
            Some("boot read failed"),
            "a read that left nothing to pick must show the failure in the picker"
        );
    }

    /// A reload that answered while the boot read was in flight read the same
    /// table later: its map (and its cleared failure) stands, and the boot read's
    /// outcome is dropped.
    #[test]
    fn late_boot_read_does_not_replace_a_newer_reload() {
        let mut dash = Dashboard::loading(&WindowState::default());
        let read_generation = dash.workspaces_reload_gen;
        dash.workspaces = HashMap::from([("ws1".to_string(), ws("ws1"))]);
        dash.workspaces_reload_gen = read_generation + 1; // a reload answered

        let _ = dash.apply_boot_workspaces(
            Err("boot read failed".to_string()),
            "personal:admin",
            false,
            read_generation,
        );

        assert!(
            dash.workspaces_error.is_none(),
            "the superseded boot read must not set the failure over the newer read"
        );
        assert!(
            dash.workspaces.contains_key("ws1"),
            "the superseded boot read must not clear the newer map"
        );
    }

    /// The DB-sourced boot resolution: the seeded admin with a managed stored
    /// workspace present in the map restores it and writes nothing; a stored
    /// value naming no existing project lands on the project the desktop falls
    /// back to — the first in the footer switcher's order — and that pick is
    /// remembered.
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
