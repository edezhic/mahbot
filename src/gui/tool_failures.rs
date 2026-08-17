//! Tool Failures dashboard page — browse flattened tool call errors from the logs store.
//!
//! Two-line row layout with role badges and HH:MM:SS timestamps, matching the
//! Logs page style. No live streaming — data refreshes on search changes or
//! tab switch. Pagination and search controls live in the Logs page's shared
//! bottom bar (see [`super::logs`]), so this page only renders the entries.

use crate::stats::{ToolErrorEntry, ToolErrorQuery};

use iced::widget::{Column, Space, column, container, row, scrollable, text};
use iced::{Alignment, Element, Length, Task};

use iced_fonts::lucide;

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
    entries: Vec<ToolErrorEntry>,
    load_state: super::common::AsyncLoadState,

    // Pagination
    pagination: super::common::PaginationState,

    /// This tab's own search query (preserved across tab switches).
    search: String,

    /// Monotonic guard: refresh responses carry the generation they were
    /// issued under; stale responses (issued before a newer refresh) are
    /// dropped so an older query can never overwrite newer data.
    refresh_generation: u64,
}

impl ToolFailuresState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            pagination: super::common::PaginationState::new(50),
            search: String::new(),
            refresh_generation: 0,
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
        self.load_state.start_loading();
        self.refresh_generation = self.refresh_generation.wrapping_add(1);
        let generation = self.refresh_generation;
        let query = Self::build_query(&self.search);
        let page = self.pagination.page;
        let page_size = self.pagination.page_size;
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
                if generation != self.refresh_generation {
                    return Task::none();
                }
                // Page clamp against the fresh `total` from the response:
                // if the total shrank between refreshes a previously valid
                // page may now be past the end — clamp and re-query so the
                // "Page X of Y" indicator stays valid.
                if self.pagination.clamp_page(total) {
                    // Adopt the fresh total immediately so the shared bottom
                    // bar's "Page X of Y" indicator stays consistent with the
                    // clamped page during the re-query window; the old
                    // entries stay on screen.
                    self.pagination.total = total;
                    return self.refresh();
                }
                self.entries = entries;
                self.pagination.total = total;
                self.load_state.finish_loading();
                Task::none()
            }
            ToolFailuresMessage::RefreshError(generation, e) => {
                if generation != self.refresh_generation {
                    return Task::none();
                }
                self.load_state.fail(e);
                // ToolFailures shows "empty state" instead of "Loading…" after
                // the first attempt, even if it failed, so mark has_loaded=true.
                self.load_state.set_has_loaded();
                Task::none()
            }
        }
    }

    /// Reset pagination to the first page.
    pub fn reset_pagination(&mut self) {
        self.pagination.reset();
    }

    /// Go to the previous page and refresh.
    pub fn prev_page(&mut self) -> Task<ToolFailuresMessage> {
        if self.pagination.prev_page() {
            self.refresh()
        } else {
            Task::none()
        }
    }

    /// Go to the next page and refresh.
    pub fn next_page(&mut self) -> Task<ToolFailuresMessage> {
        if self.pagination.next_page() {
            self.refresh()
        } else {
            Task::none()
        }
    }

    /// Current page index (rendered by the Logs page's shared bottom bar).
    pub(crate) fn page(&self) -> usize {
        self.pagination.page
    }

    /// Total number of pages given the current total (rendered by the Logs
    /// page's shared bottom bar).
    pub(crate) fn total_pages(&self) -> usize {
        self.pagination.total_pages()
    }

    /// This tab's search query (bound to the Logs page's shared search input).
    pub(crate) fn search(&self) -> &str {
        &self.search
    }

    /// Replace this tab's search query (from the Logs page's shared search input).
    pub(crate) fn set_search(&mut self, search: String) {
        self.search = search;
    }

    pub fn view(&self) -> Element<'_, ToolFailuresMessage> {
        let mut content = Column::new();

        // Error display
        content = widgets::push_error_banner(content, self.load_state.error());

        // Entries or empty state
        if self.load_state.loading() && !self.load_state.has_loaded() {
            content = content.push(widgets::loading_text());
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

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(theme::base_container_style)
            .into()
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
            text(timestamp).size(10).color(theme::TEXT_MUTED),
            Space::new().width(8),
            widgets::badge_pill(
                entry.tool_name.clone(),
                (theme::HOVER, theme::TEXT_SECONDARY),
                10,
                [1, 6],
            ),
            Space::new().width(4),
            widgets::badge_pill(
                duration_label,
                (theme::HOVER, theme::TEXT_MUTED),
                10,
                [1, 6]
            ),
            Space::new().width(4),
            widgets::role_badge(entry.role.clone(), role_colors, 10, [1, 6], false),
            Space::new().width(Length::Fill),
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
            .style(theme::surface_card_style)
            .into()
    }
}
