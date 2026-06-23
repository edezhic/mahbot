//! Tool Failures dashboard page — browse flattened tool call errors from stats.db.
//!
//! Filter bar (role picklist + search text input + refresh button), two-line row
//! layout with role badges and HH:MM:SS timestamps, matching the Logs page style.
//! No live streaming — manual refresh via the refresh button or filter changes.

use crate::stats::{ToolErrorEntry, ToolErrorQuery};

use iced::widget::{
    Column, Space, button, column, container, pick_list, row, scrollable, text, text_input, tooltip,
};
use iced::{Alignment, Element, Length, Task};
use std::time::Duration;

use iced_fonts::lucide;

use super::theme;
use super::widgets;
use super::widgets::selectable_text;

#[derive(Debug, Clone)]
pub enum ToolFailuresMessage {
    /// Data refreshed from the store.
    Refreshed(Vec<ToolErrorEntry>, usize),
    /// Refresh query failed.
    RefreshError(String),
    /// User clicked the manual refresh button.
    RefreshRequested,
    /// Role filter changed.
    RoleFilterInput(String),
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
    error: Option<String>,
    pub(crate) loading: bool,
    /// Whether at least one refresh has completed.
    has_loaded: bool,
    /// Current page (0-indexed).
    page: usize,
    /// Rows per page.
    page_size: usize,

    // Filters
    /// Role name filter (empty = all roles).
    role_filter: String,
    /// Search text filter (empty = no search).
    search_filter: String,
    /// Dropdown options for the role picklist.
    role_options: Vec<super::widgets::PickOption>,

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
            error: None,
            loading: false,
            has_loaded: false,
            page: 0,
            page_size: 50,
            role_filter: String::new(),
            search_filter: String::new(),
            role_options: <crate::Role as strum::IntoEnumIterator>::iter()
                .map(|r| {
                    let name = r.to_string();
                    super::widgets::PickOption {
                        value: name.clone(),
                        label: name,
                    }
                })
                .collect(),
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
            search: if self.search_filter.is_empty() {
                None
            } else {
                Some(self.search_filter.clone())
            },
        }
    }

    /// Request a refresh from the stats store.
    ///
    /// Sets `self.loading = true` and clears `self.error`.
    pub fn refresh(&mut self) -> Task<ToolFailuresMessage> {
        self.loading = true;
        self.error = None;
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
                self.error = None;
                self.loading = false;
                self.has_loaded = true;
                Task::none()
            }
            ToolFailuresMessage::RefreshError(e) => {
                self.error = Some(e);
                self.loading = false;
                self.has_loaded = true;
                Task::none()
            }
            ToolFailuresMessage::RefreshRequested => {
                self.page = 0;
                self.refresh()
            }
            ToolFailuresMessage::RoleFilterInput(v) => {
                self.role_filter = v;
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
        if let Some(ref err) = self.error {
            content = content.push(widgets::error_banner(err));
            content = content.push(Space::new().height(8));
        }

        // Filter bar
        let search_input: Element<'_, ToolFailuresMessage> = if self.focus_search {
            container(
                text_input("search errors…", &self.search_filter)
                    .on_input(ToolFailuresMessage::SearchInput)
                    .style(super::widgets::text_input_style)
                    .size(13)
                    .padding(4)
                    .width(Length::Fixed(180.0)),
            )
            .padding(2)
            .style(|_theme: &iced::Theme| container::Style {
                border: iced::Border {
                    radius: 4.0.into(),
                    width: 1.0,
                    color: theme::ACCENT,
                },
                ..container::Style::default()
            })
            .into()
        } else {
            text_input("search errors…", &self.search_filter)
                .on_input(ToolFailuresMessage::SearchInput)
                .style(super::widgets::text_input_style)
                .size(13)
                .padding(4)
                .width(Length::Fixed(180.0))
                .into()
        };

        let role_pick_list = {
            let role_selected = self
                .role_options
                .iter()
                .find(|o| o.value == self.role_filter)
                .cloned();
            pick_list(self.role_options.as_slice(), role_selected, |opt| {
                ToolFailuresMessage::RoleFilterInput(opt.value)
            })
            .placeholder("Role")
            .style(super::widgets::pick_list_style)
            .padding([4, 8])
            .width(Length::Fixed(100.0))
        };

        let search_group = row![
            lucide::search::<iced::Theme, iced::Renderer>()
                .size(12)
                .color(theme::TEXT_MUTED),
            Space::new().width(4),
            search_input,
        ]
        .align_y(Alignment::Center);

        let refresh_button = tooltip(
            button(
                lucide::refresh_cw::<iced::Theme, iced::Renderer>()
                    .size(14)
                    .color(theme::TEXT_MUTED),
            )
            .style(theme::button_text)
            .on_press(ToolFailuresMessage::RefreshRequested),
            "Refresh",
            tooltip::Position::Top,
        )
        .delay(Duration::from_millis(400));

        let filter_row = row![
            role_pick_list,
            Space::new().width(Length::Fill),
            search_group,
            Space::new().width(8),
            refresh_button,
        ]
        .align_y(Alignment::Center)
        .width(Length::Fill);

        content = content.push(filter_row);
        content = content.push(Space::new().height(8));

        // Entries or empty state
        if self.loading && !self.has_loaded {
            content = content.push(text("Loading...").size(14).color(theme::TEXT_MUTED));
        } else if self.entries.is_empty() && self.has_loaded {
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
                .direction(scrollable::Direction::Vertical(theme::thin_scrollbar()))
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
            content = content.push(
                container(pagination)
                    .width(Length::Fill)
                    .align_x(Alignment::Center),
            );
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
    ///   Line 1: HH:MM:SS timestamp | tool name badge | role badge
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
            .padding([5, 10])
            .style(|_theme: &iced::Theme| container::Style {
                border: iced::Border {
                    width: 1.0,
                    color: theme::BORDER,
                    ..Default::default()
                },
                ..container::Style::default()
            })
            .into()
    }
}
