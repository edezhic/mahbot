//! Tool Failures dashboard page — browse flattened tool call errors from stats.db.
//!
//! Two-line row layout with role badges and HH:MM:SS timestamps, matching the
//! Logs page style. Filter bar is shared with the Logs page via [`super::logs`].
//! No live streaming — data refreshes on filter changes or tab switch.

use crate::stats::{ToolErrorEntry, ToolErrorQuery};

use iced::widget::{Column, Space, button, column, container, row, scrollable, text};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use super::theme;
use super::widgets;
use super::widgets::selectable_text;

#[derive(Debug, Clone)]
pub enum ToolFailuresMessage {
    /// Data refreshed from the store. Carries entries and total count.
    Refreshed(Vec<ToolErrorEntry>, usize),
    /// Refresh query failed.
    RefreshError(String),
    /// Role filter changed.
    RoleFilterInput(String),
    /// Workspace filter changed.
    WorkspaceInput(String),
    /// Search text filter changed (debounced).
    SearchInput(String),
    /// Debounced refresh triggered after 300ms of inactivity.
    DebouncedRefresh(u64),
    /// Go to previous page.
    PrevPage,
    /// Go to next page.
    NextPage,
    /// Dismiss modals/panels (Escape key).
    Escape,
    /// Request toast notification.
    Toast(super::ToastMessage),
    /// Cmd+F keyboard shortcut — focus the search input.
    FocusSearch,
}

pub struct ToolFailuresState {
    entries: Vec<ToolErrorEntry>,
    total: usize,
    load_state: super::common::AsyncLoadState,
    /// Current page (0-indexed).
    page: usize,
    /// Rows per page.
    page_size: usize,

    // Filters
    /// Role name filter (empty = all roles).
    pub(crate) role_filter: String,
    /// Workspace name filter (empty = all workspaces).
    pub(crate) workspace_filter: String,
    /// Search text filter (empty = no search).
    pub(crate) search_filter: String,

    /// Visual highlight for search input (Cmd+F).
    focus_search: bool,

    /// Debounce counter for the search text input. Each keystroke increments
    /// this; only the most recent generation's sleep-task triggers a DB refresh.
    debounce_generation: u64,
    /// True when a debounced refresh is pending (prevents double-firing).
    debounce_pending: bool,
}

impl ToolFailuresState {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            total: 0,
            load_state: super::common::AsyncLoadState::new(),
            page: 0,
            page_size: 50,
            role_filter: String::new(),
            workspace_filter: String::new(),
            search_filter: String::new(),
            focus_search: false,
            debounce_generation: 0,
            debounce_pending: false,
        }
    }

    const fn total_pages(&self) -> usize {
        if self.total == 0 {
            0
        } else {
            (self.total + self.page_size - 1) / self.page_size
        }
    }

    fn build_query(&self) -> ToolErrorQuery {
        ToolErrorQuery {
            role_filter: if self.role_filter.is_empty() {
                None
            } else {
                Some(self.role_filter.clone())
            },
            workspace_filter: if self.workspace_filter.is_empty() {
                None
            } else {
                Some(self.workspace_filter.clone())
            },
            search: if self.search_filter.is_empty() {
                None
            } else {
                Some(self.search_filter.clone())
            },
        }
    }

    /// Request a refresh from the stats store.
    ///
    /// Delegates to [`AsyncLoadState::start_loading`].
    pub fn refresh(&mut self) -> Task<ToolFailuresMessage> {
        self.load_state.start_loading();
        let query = self.build_query();
        let page = self.page;
        let page_size = self.page_size;
        Task::perform(
            async move {
                let store = crate::stats::store();
                store
                    .query_tool_errors(&query, page_size, page * page_size)
                    .await
                    .map_err(|e| e.to_string())
            },
            |res| match res {
                Ok((entries, total)) => ToolFailuresMessage::Refreshed(entries, total),
                Err(e) => ToolFailuresMessage::RefreshError(e),
            },
        )
    }

    pub fn update(&mut self, message: ToolFailuresMessage) -> Task<ToolFailuresMessage> {
        match message {
            ToolFailuresMessage::Refreshed(entries, total) => {
                self.entries = entries;
                self.total = total;
                self.load_state.finish_loading();
                Task::none()
            }
            ToolFailuresMessage::RefreshError(e) => {
                self.load_state.fail(e);
                // ToolFailures shows "empty state" instead of "Loading…" after
                // the first attempt, even if it failed, so mark has_loaded=true.
                self.load_state.set_has_loaded();
                Task::none()
            }
            ToolFailuresMessage::RoleFilterInput(v) => {
                self.role_filter = v;
                self.page = 0;
                self.refresh()
            }
            ToolFailuresMessage::WorkspaceInput(v) => {
                self.workspace_filter = v;
                self.page = 0;
                self.refresh()
            }
            ToolFailuresMessage::SearchInput(v) => {
                self.search_filter = v;
                self.page = 0;
                self.debounce_generation = self.debounce_generation.wrapping_add(1);
                self.debounce_pending = true;
                let generation = self.debounce_generation;
                Task::perform(
                    widgets::debounce_sleep(300, generation),
                    ToolFailuresMessage::DebouncedRefresh,
                )
            }
            ToolFailuresMessage::DebouncedRefresh(generation) => {
                if widgets::debounce_should_process(
                    generation,
                    self.debounce_generation,
                    self.debounce_pending,
                ) {
                    self.debounce_pending = false;
                    return self.refresh();
                }
                Task::none()
            }
            ToolFailuresMessage::PrevPage => {
                if self.page > 0 {
                    self.page -= 1;
                    return self.refresh();
                }
                Task::none()
            }
            ToolFailuresMessage::NextPage => {
                if self.page + 1 < self.total_pages() {
                    self.page += 1;
                    return self.refresh();
                }
                Task::none()
            }
            ToolFailuresMessage::Escape => {
                self.focus_search = false;
                Task::none()
            }
            ToolFailuresMessage::Toast(_) => Task::none(),
            ToolFailuresMessage::FocusSearch => {
                self.focus_search = true;
                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<'_, ToolFailuresMessage> {
        let mut content = Column::new();

        // Error display
        if let Some(err) = self.load_state.error() {
            content = content.push(widgets::error_banner(err));
            content = content.push(Space::new().height(8));
        }

        // Entries or empty state
        if self.load_state.loading() && !self.load_state.has_loaded() {
            content = content.push(text("Loading...").size(14).color(theme::TEXT_MUTED));
        } else if self.entries.is_empty() && self.load_state.has_loaded() {
            content = content.push(widgets::empty_state_placeholder(
                lucide::bug::<iced::Theme, iced::Renderer>(),
                "No tool failures",
            ));
        } else if !self.entries.is_empty() {
            let entries_view = {
                scrollable(
                    Column::with_children(
                        self.entries
                            .iter()
                            .map(Self::render_error_row)
                            .collect::<Vec<_>>(),
                    )
                    .spacing(2),
                )
                .height(Length::Fill)
                .direction(theme::vertical_scrollbar())
                .style(theme::scrollbar_style)
            };

            content = content.push(entries_view);
        }

        // Pagination bar
        let total_pages = self.total_pages();
        if total_pages > 0 {
            let pagination = row![
                button(text("← Prev").size(12))
                    .style(theme::button_text)
                    .on_press_maybe(if self.page > 0 {
                        Some(ToolFailuresMessage::PrevPage)
                    } else {
                        None
                    }),
                Space::new().width(8),
                text(format!("Page {} of {}", self.page + 1, total_pages))
                    .size(12)
                    .color(theme::TEXT_MUTED),
                Space::new().width(8),
                button(text("Next →").size(12))
                    .style(theme::button_text)
                    .on_press_maybe(if self.page + 1 < total_pages {
                        Some(ToolFailuresMessage::NextPage)
                    } else {
                        None
                    }),
            ]
            .align_y(Alignment::Center);

            content = content.push(Space::new().height(8));
            content = content.push(pagination);
        }

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..container::Style::default()
            })
            .into()
    }

    /// Render a single error row with two-line layout:
    ///   Line 1: HH:MM:SS timestamp | tool name badge | role badge | workspace
    ///   Line 2: error message (selectable monospace text)
    fn render_error_row(entry: &ToolErrorEntry) -> iced::Element<'_, ToolFailuresMessage> {
        let (fg, bg) = theme::role_badge_color(&entry.role);

        let timestamp = if entry.recorded_at.len() > 19 {
            &entry.recorded_at[11..19] // Extract HH:MM:SS from ISO 8601
        } else {
            &entry.recorded_at
        };

        let metadata_row = row![
            // Timestamp
            text(timestamp).size(10).color(theme::TEXT_MUTED),
            Space::new().width(8),
            // Tool name badge
            container(text(&entry.tool_name).size(10).color(theme::TEXT_SECONDARY))
                .padding([1, 6])
                .style(|_theme: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::HOVER)),
                    border: iced::Border {
                        radius: 3.0.into(),
                        ..iced::Border::default()
                    },
                    ..container::Style::default()
                }),
            Space::new().width(4),
            // Role badge
            container(text(&entry.role).size(10).color(fg))
                .padding([1, 6])
                .style(move |_theme: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(bg)),
                    border: iced::Border {
                        radius: 3.0.into(),
                        ..iced::Border::default()
                    },
                    ..container::Style::default()
                }),
            Space::new().width(Length::Fill),
            // Workspace (if present)
            if !entry.workspace.is_empty() {
                text(&entry.workspace).size(10).color(theme::TEXT_MUTED)
            } else {
                text("")
            },
        ]
        .align_y(Alignment::Center)
        .spacing(2);

        let row_content = column![
            metadata_row,
            Space::new().height(2),
            selectable_text(&entry.error, theme::TEXT_PRIMARY)
                .size(13)
                .font(super::JETBRAINS_MONO)
                .width(Length::Fill),
        ]
        .spacing(1);

        container(row_content)
            .padding(6)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_SURFACE)),
                border: iced::Border {
                    radius: 4.0.into(),
                    width: 1.0,
                    color: theme::BORDER,
                },
                ..container::Style::default()
            })
            .into()
    }
}
