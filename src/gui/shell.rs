//! Shell dashboard page — tabbed embedded terminal widget (full page).
//!
//! Multiple persistent PTY terminals can be open per workspace, each as a tab.
//! Tabs survive page navigation (switching away and back preserves all tabs).
//! Tab state is per-workspace — switching workspaces swaps the entire tab set.
//! A right-click context menu on the terminal area offers Clear (Ctrl+L)
//! and Select All (viewport-only visual selection).

#![allow(clippy::too_many_lines)]

use std::collections::HashMap;
use std::path::PathBuf;

use iced::widget::{Space, button, column, container, row, scrollable, text};
use iced::{Alignment, Element, Length, Size, Subscription, Task};
use iced_fonts::lucide;
use iced_term::{BackendCommand, TerminalView};

use super::context_menu::ContextMenu;
use super::theme;

// ── Constants ─────────────────────────────────────────────────────────

/// Large coordinate for "select all" — alacritty clamps to grid bounds.
const LARGE_COORD: f32 = 999_999.0;

// ── Theme ─────────────────────────────────────────────────────────────

/// Zed One Dark ANSI color palette for the embedded terminal.
/// Matches Zed's default dark terminal theme.
fn zed_one_dark_palette() -> iced_term::ColorPalette {
    iced_term::ColorPalette {
        foreground: "#abb2bf".into(),
        background: "#100f0f".into(),
        black: "#282c34".into(),
        red: "#e06c75".into(),
        green: "#98c379".into(),
        yellow: "#e5c07b".into(),
        blue: "#61afef".into(),
        magenta: "#c678dd".into(),
        cyan: "#56b6c2".into(),
        white: "#abb2bf".into(),
        bright_black: "#5c6370".into(),
        bright_red: "#e06c75".into(),
        bright_green: "#98c379".into(),
        bright_yellow: "#e5c07b".into(),
        bright_blue: "#61afef".into(),
        bright_magenta: "#c678dd".into(),
        bright_cyan: "#56b6c2".into(),
        bright_white: "#ffffff".into(),
        bright_foreground: None,
        // Dimmed variants (~65% brightness of normal colors)
        dim_foreground: "#6f747c".into(),
        dim_black: "#1a1d23".into(),
        dim_red: "#92464c".into(),
        dim_green: "#637f4f".into(),
        dim_yellow: "#957d50".into(),
        dim_blue: "#3f729b".into(),
        dim_magenta: "#814e90".into(),
        dim_cyan: "#38767e".into(),
        dim_white: "#6f747c".into(),
    }
}

// ── Types ─────────────────────────────────────────────────────────────

/// A single shell tab with an independent terminal session.
struct ShellTab {
    label: String,
    terminal: iced_term::Terminal,
}

/// Per-workspace tab collection.
struct WorkspaceShellState {
    tabs: Vec<ShellTab>,
    active_idx: usize,
    /// Label counter for generating unique sequential names ("Shell 1", …).
    label_counter: u64,
    /// Cached workspace filesystem path, so new tabs (via '+' or reopen)
    /// start in the correct working directory.
    workspace_path: Option<String>,
    /// If set, all terminal spawns have failed for this workspace; the
    /// message explains why.
    spawn_error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ShellMessage {
    /// Terminal event forwarded from an embedded terminal widget.
    TerminalEvent(iced_term::Event),
    /// Workspace selected via the Home page picker (name, optional filesystem path).
    WorkspaceSelected(String, Option<String>),
    /// Select a tab by its index in the tab list.
    TabSelected(usize),
    /// Close a tab by its index.
    TabClosed(usize),
    /// Create a new shell tab.
    NewTab,
    /// Clear the active terminal (sends Ctrl+L / form feed 0x0C).
    ClearTerminal,
    /// Select all visible content in the active terminal.
    SelectAll,
}

pub struct ShellState {
    /// Currently selected workspace name (set by global sidebar picker via Dashboard).
    selected_workspace_name: Option<String>,
    /// Tabs grouped by workspace name.
    workspace_states: HashMap<String, WorkspaceShellState>,
    /// Monotonically increasing counter for unique terminal IDs.
    next_term_id: u64,
    /// The most recent layout pixel size captured from `BackendCommand::Resize`
    /// events.  Replayed onto newly created terminals so they immediately get
    /// the correct character grid instead of the tiny ~8×2 default.
    last_seen_layout_size: Option<Size>,
}

impl ShellState {
    pub fn new() -> Self {
        Self {
            selected_workspace_name: None,
            workspace_states: HashMap::new(),
            next_term_id: 0,
            last_seen_layout_size: None,
        }
    }

    /// Ensure a workspace state exists; if not, create one with a single default tab.
    ///
    /// Note: `working_dir` is only used during initial creation (via
    /// `entry().or_insert_with()`).  Subsequent calls with different paths
    /// are silently ignored — in practice workspace paths don't change after
    /// selection, so this asymmetry is harmless.
    fn ensure_workspace_state(&mut self, ws_name: &str, working_dir: Option<String>) {
        let layout_size = self.last_seen_layout_size;
        self.workspace_states
            .entry(ws_name.to_string())
            .or_insert_with(|| {
                let mut ws = Self::new_workspace_state(&mut self.next_term_id, working_dir);
                if let Some(tab) = ws.tabs.first_mut() {
                    Self::replay_layout_size(&mut tab.terminal, layout_size);
                }
                ws
            });
    }

    /// Build a fresh workspace state with one default tab.
    fn new_workspace_state(next_id: &mut u64, working_dir: Option<String>) -> WorkspaceShellState {
        match Self::spawn_one_terminal(next_id, "Shell 1", working_dir.clone()) {
            Ok(tab) => WorkspaceShellState {
                tabs: vec![tab],
                active_idx: 0,
                label_counter: 1,
                workspace_path: working_dir,
                spawn_error: None,
            },
            Err(msg) => WorkspaceShellState {
                tabs: Vec::new(),
                active_idx: 0,
                label_counter: 0,
                workspace_path: working_dir,
                spawn_error: Some(msg),
            },
        }
    }

    /// Spawn a single terminal and return a [`ShellTab`], or an error string
    /// explaining why it could not be created.
    fn spawn_one_terminal(
        next_id: &mut u64,
        label: &str,
        working_dir: Option<String>,
    ) -> Result<ShellTab, String> {
        let shell = match std::env::var("SHELL") {
            Ok(s) if !s.is_empty() => s,
            Ok(_) => return Err("$SHELL is empty".into()),
            Err(_) => return Err("$SHELL not set".into()),
        };

        let id = *next_id;
        *next_id += 1;

        let settings = iced_term::settings::Settings {
            theme: iced_term::settings::ThemeSettings::new(Box::new(zed_one_dark_palette())),
            backend: iced_term::settings::BackendSettings {
                program: shell,
                working_directory: working_dir.map(PathBuf::from),
                ..Default::default()
            },
            ..Default::default()
        };

        match iced_term::Terminal::new(id, settings) {
            Ok(terminal) => Ok(ShellTab {
                label: label.to_string(),
                terminal,
            }),
            Err(e) => Err(format!("Failed to create terminal: {e}")),
        }
    }

    /// If a known layout pixel size has been captured from a previous
    /// terminal, replay it onto `terminal` so it immediately gets the
    /// correct character grid instead of the tiny ~8×2 default.
    fn replay_layout_size(terminal: &mut iced_term::Terminal, layout_size: Option<Size>) {
        if let Some(size) = layout_size {
            let _ = terminal.handle(iced_term::Command::ProxyToBackend(BackendCommand::Resize(
                Some(size),
                None,
            )));
        }
    }

    /// Helper: run a closure with a mutable reference to the active tab of the
    /// current workspace.  Returns `None` (and does nothing) when there is no
    /// workspace selected or no tabs exist.
    fn with_active_tab_mut<T>(&mut self, f: impl FnOnce(&mut ShellTab) -> T) -> Option<T> {
        let ws_name = self.selected_workspace_name.as_ref()?;
        let ws_state = self.workspace_states.get_mut(ws_name)?;
        let tab = ws_state.tabs.get_mut(ws_state.active_idx)?;
        Some(f(tab))
    }

    // ── View ─────────────────────────────────────────────────────────

    pub fn view(&self) -> Element<'_, ShellMessage> {
        let Some(ref ws_name) = self.selected_workspace_name else {
            return Self::placeholder_view("Select a workspace to open a terminal.");
        };

        let Some(ws_state) = self.workspace_states.get(ws_name) else {
            return Self::placeholder_view("Select a workspace to open a terminal.");
        };

        // Tab bar.
        let tab_bar = Self::build_tab_bar(ws_state);

        // Terminal area — shows the active tab's terminal.
        let terminal_area = if let Some(tab) = ws_state.tabs.get(ws_state.active_idx) {
            let term_view = TerminalView::show(&tab.terminal).map(ShellMessage::TerminalEvent);
            let term_container = container(term_view)
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(8)
                .style(|_t: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::BG_BASE)),
                    ..Default::default()
                });

            // Wrap in a right-click context menu.
            let ctx_menu: Element<'_, ShellMessage> = ContextMenu::new(
                term_container,
                vec![
                    ("Clear".into(), ShellMessage::ClearTerminal),
                    ("Select All".into(), ShellMessage::SelectAll),
                ],
            )
            .into();
            ctx_menu
        } else {
            // No active tab (either all spawns failed or active_idx is stale).
            let msg = ws_state
                .spawn_error
                .as_deref()
                .unwrap_or("Terminal unavailable");
            Self::centered_title_subtitle(
                "Terminal Error",
                theme::STATUS_ERROR,
                msg,
                14,
                theme::TEXT_SECONDARY,
                16,
            )
        };

        column![tab_bar, terminal_area]
            .spacing(0)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    /// Build a placeholder message when no workspace is selected.
    fn placeholder_view(msg: &str) -> Element<'_, ShellMessage> {
        Self::centered_title_subtitle("Terminal", theme::ACCENT, msg, 13, theme::TEXT_MUTED, 4)
    }

    /// Build a centered title + subtitle message display.
    ///
    /// The layout wraps both texts in a centered container:
    ///   <title>          — 24px bold
    ///   8px spacer
    ///   <subtitle>       — customizable size
    ///
    /// The `spacing` parameter controls the column gap between the two texts.
    /// The 8px spacer is a fixed element between title and subtitle — the
    /// total visual gap from title baseline to subtitle baseline is
    /// the 8px spacer plus the column spacing above/below.
    fn centered_title_subtitle<'a>(
        title: &'a str,
        title_color: iced::Color,
        subtitle: &'a str,
        subtitle_size: u32,
        subtitle_color: iced::Color,
        spacing: u32,
    ) -> Element<'a, ShellMessage> {
        container(
            column![
                text(title)
                    .size(24)
                    .color(title_color)
                    .font(theme::FONT_BOLD),
                Space::new().height(8),
                text(subtitle).size(subtitle_size).color(subtitle_color),
            ]
            .align_x(Alignment::Center)
            .spacing(spacing),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .padding(8)
        .center_x(Length::Fill)
        .center_y(Length::Fill)
        .style(theme::base_container_style)
        .into()
    }

    /// Build the tab bar with tab buttons, close buttons, and a "+" button.
    fn build_tab_bar(ws_state: &WorkspaceShellState) -> Element<'_, ShellMessage> {
        let mut tab_buttons: Vec<Element<'_, ShellMessage>> = Vec::new();

        for (i, tab) in ws_state.tabs.iter().enumerate() {
            let is_active = i == ws_state.active_idx;

            let name_color = if is_active {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            };
            let name_text = text(&tab.label).size(12).color(name_color);

            let close_btn = button(lucide::x::<iced::Theme, iced::Renderer>().size(12).color(
                if is_active {
                    theme::TEXT_SECONDARY
                } else {
                    theme::TEXT_FAINT
                },
            ))
            .on_press(ShellMessage::TabClosed(i))
            .style(theme::button_transparent)
            .padding(0);

            let tab_row = row![name_text, close_btn]
                .spacing(2)
                .align_y(Alignment::Center)
                .padding([8, 8]);

            let tab_btn = button(tab_row)
                .on_press(ShellMessage::TabSelected(i))
                .style(move |_t: &iced::Theme, status| {
                    let bg = if is_active {
                        theme::BG_ELEVATED
                    } else if status == button::Status::Hovered {
                        theme::HOVER
                    } else {
                        theme::BG_SURFACE
                    };
                    button::Style {
                        background: Some(iced::Background::Color(bg)),
                        border: iced::Border {
                            radius: 0.0.into(),
                            width: 0.0,
                            color: iced::Color::TRANSPARENT,
                        },
                        ..Default::default()
                    }
                })
                .padding(0);

            tab_buttons.push(tab_btn.into());
        }

        let new_tab_btn = button(
            lucide::plus::<iced::Theme, iced::Renderer>()
                .size(14)
                .color(theme::TEXT_SECONDARY),
        )
        .on_press(ShellMessage::NewTab)
        .style(theme::button_transparent)
        .padding([8, 8]);

        tab_buttons.push(new_tab_btn.into());

        let scrollable_content = row(tab_buttons).spacing(0).width(Length::Fill);

        container(
            scrollable(scrollable_content)
                .direction(theme::horizontal_scrollbar())
                .style(theme::scrollbar_style)
                .width(Length::Fill)
                .height(Length::Shrink),
        )
        .style(|_t: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(theme::BG_SURFACE)),
            border: iced::Border {
                radius: 0.0.into(),
                width: 0.0,
                color: iced::Color::TRANSPARENT,
            },
            ..Default::default()
        })
        .width(Length::Fill)
        .height(Length::Shrink)
        .into()
    }

    // ── Update ───────────────────────────────────────────────────────

    pub fn update(&mut self, msg: ShellMessage) -> Task<ShellMessage> {
        match msg {
            ShellMessage::WorkspaceSelected(name, path) => {
                if name.is_empty() && path.is_none() {
                    self.selected_workspace_name = None;
                    return Task::none();
                }
                self.selected_workspace_name = Some(name.clone());
                self.ensure_workspace_state(&name, path);
                Task::none()
            }
            ShellMessage::TerminalEvent(iced_term::Event::BackendCall(id, cmd)) => {
                // Capture the most recent layout resize so we can replay it
                // onto newly created terminals (see NewTab / TabClosed /
                // ensure_workspace_state).
                if let BackendCommand::Resize(Some(layout_size), _) = &cmd {
                    self.last_seen_layout_size = Some(*layout_size);
                }

                // Route event by terminal ID across ALL workspaces.
                // We subscribe to all terminals globally to prevent PTY stalls,
                // so the handler must match globally too — matching only the
                // current workspace would silently drop output from background
                // workspace terminals, leaving stale grid buffers on switch-back.
                for ws_state in self.workspace_states.values_mut() {
                    if let Some(tab) = ws_state.tabs.iter_mut().find(|t| t.terminal.id == id) {
                        let _ = tab.terminal.handle(iced_term::Command::ProxyToBackend(cmd));
                        break;
                    }
                }
                Task::none()
            }
            ShellMessage::TabSelected(idx) => {
                let Some(ref ws_name) = self.selected_workspace_name else {
                    return Task::none();
                };
                if let Some(ws_state) = self.workspace_states.get_mut(ws_name) {
                    if idx < ws_state.tabs.len() {
                        ws_state.active_idx = idx;
                    }
                }
                Task::none()
            }
            ShellMessage::TabClosed(idx) => {
                let Some(ref ws_name) = self.selected_workspace_name else {
                    return Task::none();
                };
                if let Some(ws_state) = self.workspace_states.get_mut(ws_name) {
                    if idx >= ws_state.tabs.len() {
                        return Task::none();
                    }
                    ws_state.tabs.remove(idx);

                    if ws_state.tabs.is_empty() {
                        // Last tab closed — reopen a fresh default tab.
                        let label = "Shell 1";
                        let wd = ws_state.workspace_path.clone();
                        match ShellState::spawn_one_terminal(&mut self.next_term_id, label, wd) {
                            Ok(mut tab) => {
                                ws_state.label_counter = 1;
                                Self::replay_layout_size(
                                    &mut tab.terminal,
                                    self.last_seen_layout_size,
                                );
                                ws_state.tabs.push(tab);
                                ws_state.active_idx = 0;
                                ws_state.spawn_error = None;
                            }
                            Err(msg) => {
                                ws_state.spawn_error = Some(msg);
                            }
                        }
                    } else if ws_state.active_idx >= ws_state.tabs.len() {
                        // Active index is stale — clamp to last tab.
                        ws_state.active_idx = ws_state.tabs.len() - 1;
                    } else if idx < ws_state.active_idx {
                        // Closed a tab before the active one — shift down.
                        ws_state.active_idx = ws_state.active_idx.saturating_sub(1);
                    }
                    // else: active index stays valid as-is:
                    //   - idx == active_idx: next tab shifted into this position
                    //   - idx > active_idx: no effect on active index
                }
                Task::none()
            }
            ShellMessage::NewTab => {
                let Some(ref ws_name) = self.selected_workspace_name else {
                    return Task::none();
                };
                if let Some(ws_state) = self.workspace_states.get_mut(ws_name) {
                    let next_counter = ws_state.label_counter + 1;
                    let label = format!("Shell {next_counter}");
                    let wd = ws_state.workspace_path.clone();
                    match ShellState::spawn_one_terminal(&mut self.next_term_id, &label, wd) {
                        Ok(mut tab) => {
                            ws_state.label_counter = next_counter;
                            let new_idx = ws_state.tabs.len();
                            Self::replay_layout_size(&mut tab.terminal, self.last_seen_layout_size);
                            ws_state.tabs.push(tab);
                            ws_state.active_idx = new_idx;
                            ws_state.spawn_error = None;
                        }
                        Err(msg) => {
                            // If other tabs exist, the error is invisible in
                            // the UI — log it so it's at least discoverable.
                            if ws_state.tabs.is_empty() {
                                ws_state.spawn_error = Some(msg);
                            } else {
                                tracing::warn!("NewTab spawn failed: {msg}");
                            }
                        }
                    }
                }
                Task::none()
            }
            ShellMessage::ClearTerminal => {
                let _ = self.with_active_tab_mut(|tab| {
                    let _ = tab.terminal.handle(iced_term::Command::ProxyToBackend(
                        iced_term::BackendCommand::Write(vec![0x0C]),
                    ));
                });
                Task::none()
            }
            ShellMessage::SelectAll => {
                let _ = self.with_active_tab_mut(|tab| {
                    let _ = tab.terminal.handle(iced_term::Command::ProxyToBackend(
                        iced_term::BackendCommand::SelectStart(
                            iced_term::SelectionType::Lines,
                            (0.0, 0.0),
                        ),
                    ));
                    let _ = tab.terminal.handle(iced_term::Command::ProxyToBackend(
                        iced_term::BackendCommand::SelectUpdate((LARGE_COORD, LARGE_COORD)),
                    ));
                });
                Task::none()
            }
        }
    }

    // ── Subscription ─────────────────────────────────────────────────

    pub fn subscription(&self) -> Subscription<ShellMessage> {
        // Subscribe to ALL terminals across ALL workspaces to prevent PTY
        // stalls — iced_term's backend uses a bounded mpsc channel (cap 100)
        // with blocking_send; undrained channels cause the PTY reader thread
        // to block, freezing the shell process.  This applies to both
        // background tabs within a workspace and terminals in workspaces the
        // user has navigated away from.  Events are routed to the correct
        // terminal by ID in the update handler.
        let all_tab_subs: Vec<Subscription<ShellMessage>> = self
            .workspace_states
            .values()
            .flat_map(|ws| &ws.tabs)
            .map(|tab| tab.terminal.subscription().map(ShellMessage::TerminalEvent))
            .collect();
        if all_tab_subs.is_empty() {
            Subscription::none()
        } else {
            Subscription::batch(all_tab_subs)
        }
    }
}
