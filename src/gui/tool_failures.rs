//! Tool Failures dashboard page — browse flattened tool call errors from stats.db.
//!
//! Two-line row layout with role badges and HH:MM:SS timestamps, matching the
//! Logs page style. Filter bar is shared with the Logs page via [`super::logs`].
//! No live streaming — data refreshes on filter changes or tab switch.

use crate::stats::{ToolErrorEntry, ToolErrorQuery};

use iced::widget::{Column, Space, column, container, row, scrollable, text};
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
}

pub struct ToolFailuresState {
    entries: Vec<ToolErrorEntry>,
    load_state: super::common::AsyncLoadState,

    // Pagination
    pagination: super::common::PaginationState,

    // Filters
    /// Role name filter (empty = all roles).
    role_filter: String,
    /// Workspace name filter (empty = all workspaces).
    workspace_filter: String,
    /// Search text filter (empty = no search).
    search_filter: String,

    /// Debounce state for the search text input.
    debounce: super::common::DebounceState,
}

impl ToolFailuresState {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            pagination: super::common::PaginationState::new(50),
            role_filter: String::new(),
            workspace_filter: String::new(),
            search_filter: String::new(),
            debounce: super::common::DebounceState::new(),
        }
    }

    fn build_query(&self) -> ToolErrorQuery {
        ToolErrorQuery {
            role_filter: super::common::none_if_empty(&self.role_filter),
            workspace_filter: super::common::none_if_empty(&self.workspace_filter),
            search: super::common::none_if_empty(&self.search_filter),
        }
    }

    /// Request a refresh from the stats store.
    ///
    /// Delegates to `AsyncLoadState::start_loading`.
    pub fn refresh(&mut self) -> Task<ToolFailuresMessage> {
        self.load_state.start_loading();
        let query = self.build_query();
        let page = self.pagination.page;
        let page_size = self.pagination.page_size;
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
                self.pagination.total = total;
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
                self.pagination.reset();
                self.refresh()
            }
            ToolFailuresMessage::WorkspaceInput(v) => {
                self.workspace_filter = v;
                self.pagination.reset();
                self.refresh()
            }
            ToolFailuresMessage::SearchInput(v) => {
                self.search_filter = v;
                self.pagination.reset();
                self.debounce
                    .trigger(300)
                    .map(ToolFailuresMessage::DebouncedRefresh)
            }
            ToolFailuresMessage::DebouncedRefresh(generation) => {
                if self.debounce.should_process(generation) {
                    return self.refresh();
                }
                Task::none()
            }
            ToolFailuresMessage::PrevPage => {
                if self.pagination.prev_page() {
                    return self.refresh();
                }
                Task::none()
            }
            ToolFailuresMessage::NextPage => {
                if self.pagination.next_page() {
                    return self.refresh();
                }
                Task::none()
            }
            ToolFailuresMessage::Escape | ToolFailuresMessage::Toast(_) => Task::none(),
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
        content = content.push(widgets::pagination_bar(
            self.pagination.page,
            self.pagination.total_pages(),
            ToolFailuresMessage::PrevPage,
            ToolFailuresMessage::NextPage,
        ));

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
    /// Build the metadata badge row for a tool error entry.
    fn render_metadata_row(entry: &ToolErrorEntry) -> iced::widget::Row<'_, ToolFailuresMessage> {
        let (fg, bg) = theme::role_badge_color(&entry.role);

        let timestamp = if entry.recorded_at.len() > 19 {
            &entry.recorded_at[11..19] // Extract HH:MM:SS from ISO 8601
        } else {
            &entry.recorded_at
        };

        let duration_label = format!("{}ms", entry.duration_ms);

        let badge_style = |_t: &iced::Theme| container::Style {
            background: Some(iced::Background::Color(theme::HOVER)),
            border: iced::Border {
                radius: 3.0.into(),
                ..iced::Border::default()
            },
            ..container::Style::default()
        };

        row![
            // Timestamp
            text(timestamp).size(10).color(theme::TEXT_MUTED),
            Space::new().width(8),
            // Tool name badge
            container(text(&entry.tool_name).size(10).color(theme::TEXT_SECONDARY))
                .padding([1, 6])
                .style(badge_style),
            Space::new().width(4),
            // Duration badge
            container(text(duration_label).size(10).color(theme::TEXT_MUTED))
                .padding([1, 6])
                .style(badge_style),
            Space::new().width(4),
            // Role badge
            container(text(&entry.role).size(10).color(fg))
                .padding([1, 6])
                .style(move |_t: &iced::Theme| container::Style {
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
        .spacing(2)
    }

    /// Compute an optional arguments preview string, truncated to 200 chars.
    fn compute_args_preview(entry: &ToolErrorEntry) -> Option<String> {
        if entry.arguments.is_empty() || entry.arguments == "{}" {
            return None;
        }
        if entry.arguments.len() > 200 {
            let mut s = entry.arguments[..entry.arguments.floor_char_boundary(200)].to_string();
            s.push('…');
            Some(s)
        } else {
            Some(entry.arguments.clone())
        }
    }

    fn render_error_row(entry: &ToolErrorEntry) -> iced::Element<'_, ToolFailuresMessage> {
        let metadata_row = Self::render_metadata_row(entry);
        let args_preview = Self::compute_args_preview(entry);

        let row_content = column![
            metadata_row,
            Space::new().height(2),
            if let Some(ref preview) = args_preview {
                iced::Element::new(
                    selectable_text(preview.clone(), theme::TEXT_MUTED)
                        .size(11)
                        .font(super::JETBRAINS_MONO)
                        .width(Length::Fill),
                )
            } else {
                iced::Element::new(iced::widget::Space::new().height(0))
            },
            if !entry.error_message.is_empty() {
                let mut parts = column![].spacing(0);
                if args_preview.is_some() {
                    parts = parts.push(Space::new().height(2));
                }
                parts = parts.push(
                    selectable_text(&entry.error_message, theme::TEXT_PRIMARY)
                        .size(13)
                        .font(super::JETBRAINS_MONO)
                        .width(Length::Fill),
                );
                iced::Element::new(parts)
            } else {
                iced::Element::new(iced::widget::Space::new().height(0))
            },
        ]
        .spacing(0);

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
