//! Shell dashboard page — embedded terminal widget (full page).
//!
//! One persistent PTY terminal is spawned per workspace when the global sidebar
//! picker selects a workspace. Switching workspaces just swaps which terminal
//! widget is displayed — no clearing or reconfiguration. When no workspace is
//! selected, a placeholder message is shown.

use std::collections::HashMap;
use std::path::PathBuf;

use iced::widget::{Space, column, container, text};
use iced::{Alignment, Element, Length, Subscription, Task};
use iced_term::TerminalView;

use super::theme;

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

#[derive(Debug, Clone)]
pub enum ShellMessage {
    /// Terminal event forwarded from an embedded terminal widget.
    TerminalEvent(iced_term::Event),
    /// Workspace selected via the Home page picker (name, filesystem path).
    WorkspaceSelected(String, String),
}

pub struct ShellState {
    /// Currently selected workspace name (set by global sidebar picker via Dashboard).
    selected_workspace_name: Option<String>,
    /// One persistent terminal per workspace, keyed by workspace name.
    terms: HashMap<String, iced_term::Terminal>,
    /// Per-workspace terminal spawn errors.
    term_errors: HashMap<String, String>,
    /// Monotonically increasing counter for unique terminal IDs.
    next_term_id: u64,
}

impl ShellState {
    pub fn new() -> Self {
        Self {
            selected_workspace_name: None,
            terms: HashMap::new(),
            term_errors: HashMap::new(),
            next_term_id: 0,
        }
    }

    /// Spawn a single terminal for the given workspace, storing it (or its error)
    /// in the per-workspace maps. Called when the global picker selects a workspace.
    fn spawn_one_terminal(&mut self, ws_name: &str, working_dir: Option<String>) {
        let shell = match std::env::var("SHELL") {
            Ok(s) if !s.is_empty() => s,
            Ok(_) => {
                self.term_errors
                    .insert(ws_name.to_string(), "$SHELL is empty".into());
                return;
            }
            Err(_) => {
                self.term_errors
                    .insert(ws_name.to_string(), "$SHELL not set".into());
                return;
            }
        };

        let id = self.next_term_id;
        self.next_term_id += 1;

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
            Ok(t) => {
                self.terms.insert(ws_name.to_string(), t);
            }
            Err(e) => {
                self.term_errors.insert(
                    ws_name.to_string(),
                    format!("Failed to create terminal: {e}"),
                );
            }
        }
    }

    pub fn view(&self) -> Element<'_, ShellMessage> {
        let body: Element<'_, ShellMessage> =
            if let Some(ref ws_name) = self.selected_workspace_name {
                if let Some(term) = self.terms.get(ws_name) {
                    let term_view = TerminalView::show(term).map(ShellMessage::TerminalEvent);
                    container(term_view)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .padding(8)
                        .style(|_t: &iced::Theme| container::Style {
                            background: Some(iced::Background::Color(theme::BG_BASE)),
                            ..Default::default()
                        })
                        .into()
                } else {
                    // Terminal failed to spawn for this workspace.
                    let msg = self
                        .term_errors
                        .get(ws_name.as_str())
                        .map_or("Terminal unavailable", String::as_str);
                    container(
                        column![
                            text("Terminal Error")
                                .size(24)
                                .color(theme::STATUS_ERROR)
                                .font(theme::FONT_BOLD),
                            Space::new().height(8),
                            text(msg).size(14).color(theme::TEXT_SECONDARY),
                        ]
                        .align_x(Alignment::Center)
                        .spacing(16),
                    )
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .padding(8)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .style(|_t: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(theme::BG_BASE)),
                        ..Default::default()
                    })
                    .into()
                }
            } else {
                // Placeholder when no workspace is selected.
                container(
                    column![
                        text("Terminal")
                            .size(24)
                            .color(theme::ACCENT)
                            .font(theme::FONT_BOLD),
                        Space::new().height(8),
                        text("Select a workspace to open a terminal.")
                            .size(13)
                            .color(theme::TEXT_MUTED),
                    ]
                    .align_x(Alignment::Center)
                    .spacing(4),
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(8)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .style(|_t: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::BG_BASE)),
                    ..Default::default()
                })
                .into()
            };

        body
    }

    pub fn update(&mut self, msg: ShellMessage) -> Task<ShellMessage> {
        match msg {
            ShellMessage::WorkspaceSelected(name, path) => {
                if name.is_empty() && path.is_empty() {
                    self.selected_workspace_name = None;
                    return Task::none();
                }
                self.selected_workspace_name = Some(name.clone());
                // Spawn a terminal for this workspace if one doesn't exist yet.
                if !self.terms.contains_key(&name) {
                    let ws_path = if path.is_empty() { None } else { Some(path) };
                    self.spawn_one_terminal(&name, ws_path);
                }
                Task::none()
            }
            ShellMessage::TerminalEvent(iced_term::Event::BackendCall(id, cmd)) => {
                // Route event to the correct terminal by ID.
                if let Some(term) = self.terms.values_mut().find(|t| t.id == id) {
                    let _ = term.handle(iced_term::Command::ProxyToBackend(cmd));
                }
                Task::none()
            }
        }
    }

    pub fn subscription(&self) -> Subscription<ShellMessage> {
        if self.terms.is_empty() {
            Subscription::none()
        } else {
            Subscription::batch(
                self.terms
                    .values()
                    .map(|term| term.subscription().map(ShellMessage::TerminalEvent)),
            )
        }
    }
}
