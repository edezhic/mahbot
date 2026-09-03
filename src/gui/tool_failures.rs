//! Tool Failures dashboard page — browse flattened tool call errors from the logs store.
//!
//! Two-line row layout with role badges and HH:MM:SS timestamps, matching the
//! Logs page style. No live streaming — data refreshes on search changes or
//! tab switch. Pagination and search controls live in the Logs page's shared
//! bottom bar (see [`super::logs`]), so this page only renders the entries.

use crate::stats::{ToolErrorEntry, ToolErrorQuery};

use iced::widget::{Column, Space, column, container, row, text};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

use super::common::PaginatedTabState;
use super::theme;
use super::widgets;
use super::widgets::selectable_text;

#[derive(Debug, Clone)]
pub enum ToolFailuresMessage {
    /// Data refreshed from the store. Carries the refresh generation (stale
    /// responses are dropped), the entries, and the total count.
    Refreshed(u64, Vec<ToolErrorEntry>, usize),
    /// Refresh query failed. Carries the generation so stale errors are dropped.
    RefreshError(u64, String),
}

pub struct ToolFailuresState {
    state: PaginatedTabState<ToolErrorEntry>,
}

impl ToolFailuresState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: PaginatedTabState::new(50),
        }
    }

    fn build_query(search: &str) -> ToolErrorQuery {
        ToolErrorQuery {
            search: crate::util::none_if_empty(search),
        }
    }

    /// Request a refresh from the logs store.
    ///
    /// Delegates to `AsyncLoadState::start_loading`.
    pub fn refresh(&mut self) -> Task<ToolFailuresMessage> {
        let generation = self.state.begin_refresh();
        let query = Self::build_query(&self.state.search);
        let page = self.state.pagination.page;
        let page_size = self.state.pagination.page_size;
        Task::perform(
            async move {
                // Fail-open: no logs store → empty result, not a panic.
                let Some(store) = crate::logs::LOG_STORE.get() else {
                    return Ok((generation, Vec::new(), 0));
                };
                store
                    .query_tool_errors(&query, page_size, page * page_size)
                    .await
                    .map(|(entries, total)| (generation, entries, total))
                    .map_err(|e| (generation, e.to_string()))
            },
            |res| match res {
                Ok((generation, entries, total)) => {
                    ToolFailuresMessage::Refreshed(generation, entries, total)
                }
                Err((generation, e)) => ToolFailuresMessage::RefreshError(generation, e),
            },
        )
    }

    pub fn update(&mut self, message: ToolFailuresMessage) -> Task<ToolFailuresMessage> {
        match message {
            ToolFailuresMessage::Refreshed(generation, entries, total) => {
                if self.state.handle_refreshed(generation, entries, total) {
                    self.refresh()
                } else {
                    Task::none()
                }
            }
            ToolFailuresMessage::RefreshError(generation, e) => {
                // Fail-open tab: mark has_loaded even on error so the empty
                // state ("No tool failures") renders instead of "Loading…".
                self.state.handle_refresh_error(generation, e, true);
                Task::none()
            }
        }
    }

    /// Reset pagination to the first page.
    pub fn reset_pagination(&mut self) {
        self.state.pagination.reset();
    }

    /// Go to the previous page and refresh.
    pub fn prev_page(&mut self) -> Task<ToolFailuresMessage> {
        if self.state.pagination.prev_page() {
            self.refresh()
        } else {
            Task::none()
        }
    }

    /// Go to the next page and refresh.
    pub fn next_page(&mut self) -> Task<ToolFailuresMessage> {
        if self.state.pagination.next_page() {
            self.refresh()
        } else {
            Task::none()
        }
    }

    /// Current page index (rendered by the Logs page's shared bottom bar).
    pub(crate) fn page(&self) -> usize {
        self.state.pagination.page
    }

    /// Total number of pages given the current total (rendered by the Logs
    /// page's shared bottom bar).
    pub(crate) fn total_pages(&self) -> usize {
        self.state.pagination.total_pages()
    }

    /// This tab's search query (bound to the Logs page's shared search input).
    pub(crate) fn search(&self) -> &str {
        &self.state.search
    }

    /// Replace this tab's search query (from the Logs page's shared search input).
    pub(crate) fn set_search(&mut self, search: String) {
        self.state.search = search;
    }

    pub fn view(&self) -> Element<'_, ToolFailuresMessage> {
        let state = &self.state;
        let mut content = Column::new();

        // Error display — inset to align with the vscroll-wrapped entries below.
        content = widgets::push_error_banner_inset(content, state.load_state.error());

        // Entries or empty state
        if state.load_state.loading() && !state.load_state.has_loaded() {
            content = content.push(widgets::scroll_h_inset(widgets::loading_text()));
        } else if state.entries.is_empty() && state.load_state.has_loaded() {
            content = content.push(widgets::empty_state_placeholder(
                lucide::bug::<iced::Theme, iced::Renderer>(),
                "No tool failures",
                theme::TEXT_MUTED,
            ));
        } else if !state.entries.is_empty() {
            let entries_view = {
                widgets::vscroll(
                    Column::with_children(
                        state
                            .entries
                            .iter()
                            .map(Self::render_error_row)
                            .collect::<Vec<_>>(),
                    )
                    .spacing(theme::SPACE_2),
                )
            };

            content = content.push(entries_view);
        }

        widgets::page(content)
    }

    /// Render a single error row with two-line layout:
    ///   Line 1: HH:MM:SS timestamp | tool name badge | role badge | workspace
    ///   Line 2: error message (selectable monospace text)
    /// Build the metadata badge row for a tool error entry.
    fn render_metadata_row(entry: &ToolErrorEntry) -> iced::widget::Row<'_, ToolFailuresMessage> {
        let role_colors = theme::role_badge_color(&entry.role);

        let timestamp = theme::format_hhmmss(&entry.recorded_at);

        let duration_label = format!("{}ms", entry.duration_ms);

        row![
            text(timestamp)
                .size(theme::TEXT_10)
                .color(theme::TEXT_SECONDARY),
            Space::new().width(theme::SPACE_8),
            widgets::badge_pill(
                entry.tool_name.clone(),
                (theme::TEXT_SECONDARY, theme::HOVER),
                widgets::PILL_COMPACT,
            ),
            Space::new().width(theme::SPACE_4),
            widgets::badge_pill(
                duration_label,
                (theme::TEXT_SECONDARY, theme::HOVER),
                widgets::PILL_COMPACT,
            ),
            Space::new().width(theme::SPACE_4),
            widgets::badge_pill(entry.role.clone(), role_colors, widgets::PILL_COMPACT),
            Space::new().width(Length::Fill),
            if !entry.workspace.is_empty() {
                text(&entry.workspace)
                    .size(theme::TEXT_10)
                    .color(theme::TEXT_SECONDARY)
            } else {
                text("")
            },
        ]
        .align_y(Alignment::Center)
        .spacing(theme::SPACE_2)
    }

    /// Compute an optional arguments preview string, truncated to 200 chars.
    fn compute_args_preview(entry: &ToolErrorEntry) -> Option<String> {
        if entry.arguments.is_empty() || entry.arguments == "{}" {
            return None;
        }
        if entry.arguments.len() > 200 {
            Some(format!(
                "{}…",
                crate::util::truncate_bytes(&entry.arguments, 200)
            ))
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
                        .size(theme::TEXT_11)
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
                        .size(theme::TEXT_13)
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
            .padding(theme::PAD_6)
            .style(theme::surface_card_style)
            .into()
    }
}
