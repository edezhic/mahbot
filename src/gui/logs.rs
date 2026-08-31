//! Logs dashboard page — live log viewing with per-tab pagination/search,
//! plus a Tool Failures tab for browsing tool error entries.
//!
//! Each tab (All Logs / Issues / Tool Failures) keeps its own entries,
//! pagination state and search query, so switching tabs never reuses another
//! tab's page index or search. Pause is a single global control. The bottom
//! bar holds pagination + pause + search; the top bar holds only the tabs.

use crate::logs::{LogEntry, LogQuery, LogStore};

use iced::advanced::text::Span;
use iced::widget::{Column, Space, button, column, container, row, text, tooltip};
use iced::{Alignment, Element, Length, Subscription, Task, window};
use iced_anim::Animated;
use iced_anim::transition::Easing;
use std::time::{Duration, Instant};

use iced_fonts::lucide;

use super::common::PaginatedTabState;
use super::menus::{ContextMenu, MenuItem};
use super::theme;
use super::widgets;

/// Tabs within the Logs page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogsTab {
    AllLogs,
    Issues,
    ToolFailures,
}

fn log_stream_producer() -> impl futures_util::Stream<Item = LogMessage> {
    super::common::broadcast_stream_producer(1, &super::LOG_BROADCAST, |output, item| {
        Box::pin(async move {
            match item {
                Some(json) => {
                    if let Ok(entry) = serde_json::from_str::<LogEntry>(&json) {
                        let _ = output.try_send(LogMessage::LiveEntry(entry));
                    }
                }
                None => {
                    let _ = output.try_send(LogMessage::StreamLagged);
                }
            }
        })
    })
}

#[derive(Debug, Clone)]
pub enum LogMessage {
    // Data — tagged with the originating tab and refresh generation so stale
    // async responses (issued before a newer refresh of the same tab) are
    // dropped instead of overwriting newer data.
    Refreshed(LogsTab, u64, Vec<LogEntry>, usize),
    RefreshError(LogsTab, u64, String),

    // Live stream
    LiveEntry(LogEntry),
    StreamLagged,

    // Search
    SearchInput(super::editor_widget::EditorAction),

    // Tab switching
    TabSelected(LogsTab),

    // Debounced refresh after text input (~300ms). Carries the tab whose
    // query changed so the refresh still targets that tab if the user
    // switches tabs within the debounce window.
    DebouncedRefresh(u64, LogsTab),

    // Pagination
    PrevPage,
    NextPage,

    // Pause/Resume
    TogglePause,

    /// Per-frame tick for the fade-in animation.
    AnimTick(Instant),

    /// Dismiss modals/panels (Escape key).
    Escape,

    /// Cmd+F keyboard shortcut — highlight the search input.
    FocusSearch,

    /// Copy a log entry's full formatted content to the clipboard (right-click
    /// context-menu action). Carries the entry so the formatting happens in
    /// `update()`, not in the per-frame `view()`.
    CopyEntry(LogEntry),

    /// Bridged Tool Failures sub-messages.
    ToolFailures(super::tool_failures::ToolFailuresMessage),
}

/// Per-tab state shared by the All Logs and Issues tabs. Each tab keeps its
/// own entries, load state, pagination and search query so switching tabs
/// never reuses another tab's page index or query.
type LogsTabData = PaginatedTabState<LogEntry>;

pub struct LogsState {
    /// All Logs tab data (live-streamed).
    all_logs: LogsTabData,
    /// Issues tab data (ERROR/WARN only).
    issues: LogsTabData,
    /// Tool Failures tab data (delegated sub-state).
    tool_failures_state: super::tool_failures::ToolFailuresState,

    // Tab state
    active_tab: LogsTab,

    // Stream control (global across tabs)
    paused: bool,

    /// Visual highlight for search input (Cmd+F).
    focus_search: bool,

    /// Timestamp of the most recently received live entry (for fade-in animation).
    newest_entry_timestamp: Option<String>,
    /// Fade progress: 0.0 = just appeared, 1.0 = fully settled.
    fade_anim: Animated<f32>,
    /// Debounce state for search-text input filtering.
    debounce: super::common::DebounceState,
    /// Buffer for the shared search input (bound to the active tab's query).
    search_buffer: super::common::SingleLineEditorState,
}

impl LogsState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            all_logs: PaginatedTabState::new(50),
            issues: PaginatedTabState::new(50),
            tool_failures_state: super::tool_failures::ToolFailuresState::new(),
            active_tab: LogsTab::AllLogs,
            paused: false,
            focus_search: false,
            newest_entry_timestamp: None,
            fade_anim: Animated::transition(
                0.0f32,
                Easing::EASE_OUT.with_duration(Duration::from_millis(theme::ANIM_LOG_FADE_MS)),
            ),
            debounce: super::common::DebounceState::new(),
            search_buffer: super::common::SingleLineEditorState::new(""),
        }
    }

    /// Immutable access to one log tab's data. Returns `None` for the Tool
    /// Failures tab (handled via `tool_failures_state`).
    fn tab_data(&self, tab: LogsTab) -> Option<&LogsTabData> {
        match tab {
            LogsTab::AllLogs => Some(&self.all_logs),
            LogsTab::Issues => Some(&self.issues),
            LogsTab::ToolFailures => None,
        }
    }

    /// Mutable access to one log tab's data. Returns `None` for the Tool
    /// Failures tab (handled via `tool_failures_state`).
    fn tab_data_mut(&mut self, tab: LogsTab) -> Option<&mut LogsTabData> {
        match tab {
            LogsTab::AllLogs => Some(&mut self.all_logs),
            LogsTab::Issues => Some(&mut self.issues),
            LogsTab::ToolFailures => None,
        }
    }

    /// The active tab's search query (bound to the shared search input).
    fn active_search(&self) -> &str {
        match self.active_tab {
            LogsTab::AllLogs => &self.all_logs.search,
            LogsTab::Issues => &self.issues.search,
            LogsTab::ToolFailures => self.tool_failures_state.search(),
        }
    }

    /// Reset pagination for whichever tab is currently active.
    fn reset_pagination_for_active_tab(&mut self) {
        match self.tab_data_mut(self.active_tab) {
            Some(data) => data.pagination.reset(),
            None => self.tool_failures_state.reset_pagination(),
        }
    }

    /// Refresh whichever tab is currently active.
    fn refresh_active_tab(&mut self, log_store: &LogStore) -> Task<LogMessage> {
        self.refresh_tab(log_store, self.active_tab)
    }

    /// Public entry point: refresh the currently active tab (used at boot).
    pub fn refresh(&mut self, log_store: &LogStore) -> Task<LogMessage> {
        self.refresh_active_tab(log_store)
    }

    /// Issue a refresh for one specific tab, tagging the async response with
    /// its origin tab and a generation counter so stale responses (issued
    /// before a newer refresh of the same tab) are dropped by the handler.
    /// Total over all three tabs: the Tool Failures tab refreshes through
    /// its own state/store path instead.
    fn refresh_tab(&mut self, log_store: &LogStore, tab: LogsTab) -> Task<LogMessage> {
        // Tool Failures has its own state and store path.
        let Some(data) = self.tab_data_mut(tab) else {
            return self
                .tool_failures_state
                .refresh()
                .map(LogMessage::ToolFailures);
        };
        let generation = data.begin_refresh();
        let query = LogQuery {
            // Issues shows ERROR/WARN entries only; All Logs is unfiltered.
            level: match tab {
                LogsTab::Issues => Some("ERROR,WARN".to_string()),
                LogsTab::AllLogs | LogsTab::ToolFailures => None,
            },
            target: None,
            search: crate::util::none_if_empty(&data.search),
            since: None,
            limit: Some(data.pagination.page_size),
            offset: Some(data.pagination.offset()),
        };
        let store = log_store.clone();
        Task::perform(
            async move {
                store
                    .query(&query)
                    .await
                    .map(|(entries, total)| (generation, entries, total))
                    .map_err(|e| (generation, e.to_string()))
            },
            move |res| match res {
                Ok((generation, entries, total)) => {
                    LogMessage::Refreshed(tab, generation, entries, total)
                }
                Err((generation, e)) => LogMessage::RefreshError(tab, generation, e),
            },
        )
    }

    pub fn subscription(&self) -> Subscription<LogMessage> {
        // Live log stream only while the All Logs tab is active and unpaused.
        // Frame ticks are subscribed only while the fade animation is running —
        // an always-on frames() subscription closes a self-sustaining redraw
        // loop (redraw → AnimTick → redraw).
        let mut subs: Vec<Subscription<LogMessage>> = Vec::new();
        if !self.paused && self.active_tab == LogsTab::AllLogs {
            subs.push(iced::Subscription::run(log_stream_producer));
        }
        if self.fade_anim.is_animating() {
            subs.push(window::frames().map(LogMessage::AnimTick));
        }
        Subscription::batch(subs)
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: LogMessage, log_store: &LogStore) -> Task<LogMessage> {
        match msg {
            LogMessage::Refreshed(tab, generation, entries, total) => {
                // A ToolFailures-tagged refresh cannot originate from
                // `refresh_tab`; drop it defensively rather than panic.
                let Some(data) = self.tab_data_mut(tab) else {
                    return Task::none();
                };
                if data.handle_refreshed(generation, entries, total) {
                    return self.refresh_tab(log_store, tab);
                }
                Task::none()
            }
            LogMessage::RefreshError(tab, generation, e) => {
                let Some(data) = self.tab_data_mut(tab) else {
                    return Task::none();
                };
                data.handle_refresh_error(generation, e, false);
                Task::none()
            }
            LogMessage::LiveEntry(entry) => {
                // Live entries only arrive while the All Logs tab is active and
                // unpaused (the subscription is gated in `subscription()`).
                let data = &mut self.all_logs;
                // Only prepend live entries when on page 0 (the live view).
                // Other pages are static snapshots from the database.
                if data.pagination.page != 0 {
                    return Task::none();
                }

                // Client-side search filter matching this tab's query (the
                // level filter comes from the DB query for the Issues tab,
                // which never receives live entries).
                let passes = data.search.is_empty()
                    || entry
                        .message
                        .to_lowercase()
                        .contains(&data.search.to_lowercase())
                    || entry
                        .target
                        .to_lowercase()
                        .contains(&data.search.to_lowercase());

                if passes {
                    data.entries.insert(0, entry);
                    data.pagination.total += 1;
                    // Auto-evict: keep exactly page_size entries visible.
                    data.entries.truncate(data.pagination.page_size);
                    // Mark this entry as newest so the view can fade it in.
                    self.newest_entry_timestamp = data.entries.first().map(|e| e.timestamp.clone());
                    // Reset the fade animation so it goes 0→1 for the new entry.
                    self.fade_anim = Animated::transition(
                        0.0f32,
                        Easing::EASE_OUT
                            .with_duration(Duration::from_millis(theme::ANIM_LOG_FADE_MS)),
                    );
                    self.fade_anim.set_target(1.0f32);
                }
                Task::none()
            }
            LogMessage::AnimTick(instant) => {
                self.fade_anim.tick(instant);
                Task::none()
            }
            LogMessage::StreamLagged => {
                // Only delivered while the All Logs stream is active.
                self.refresh_active_tab(log_store)
            }
            LogMessage::SearchInput(action) => {
                // Single-line Tab moves focus rather than editing.
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                // The search input is bound to the active tab's own query.
                let changes_text = action.changes_text();
                self.search_buffer.apply_action(action);
                let text = self.search_buffer.text();
                match self.tab_data_mut(self.active_tab) {
                    Some(data) => data.search = text,
                    None => self.tool_failures_state.set_search(text),
                }
                // Only text changes re-trigger the debounced refresh; cursor
                // movement emits actions too but shouldn't restart the timer.
                if !changes_text {
                    return Task::none();
                }
                self.reset_pagination_for_active_tab();
                // Tag the debounced refresh with the tab whose query changed
                // so it still targets that tab if the user switches away
                // within the debounce window.
                let tab = self.active_tab;
                self.debounce
                    .trigger(300)
                    .map(move |g| LogMessage::DebouncedRefresh(g, tab))
            }
            LogMessage::DebouncedRefresh(generation, tab) => {
                if !self.debounce.should_process(generation) {
                    return Task::none();
                }
                self.refresh_tab(log_store, tab)
            }
            LogMessage::PrevPage => match self.tab_data_mut(self.active_tab) {
                Some(data) => {
                    if data.pagination.prev_page() {
                        return self.refresh_active_tab(log_store);
                    }
                    Task::none()
                }
                None => self
                    .tool_failures_state
                    .prev_page()
                    .map(LogMessage::ToolFailures),
            },
            LogMessage::NextPage => match self.tab_data_mut(self.active_tab) {
                Some(data) => {
                    if data.pagination.next_page() {
                        return self.refresh_active_tab(log_store);
                    }
                    Task::none()
                }
                None => self
                    .tool_failures_state
                    .next_page()
                    .map(LogMessage::ToolFailures),
            },
            LogMessage::TogglePause => {
                self.paused = !self.paused;
                if !self.paused {
                    // Resume refreshes whatever tab is active — the pause
                    // button works from every tab.
                    return self.refresh_active_tab(log_store);
                }
                Task::none()
            }
            LogMessage::Escape => {
                self.focus_search = false;
                Task::none()
            }
            LogMessage::FocusSearch => {
                self.focus_search = true;
                Task::none()
            }
            LogMessage::CopyEntry(entry) => {
                // Silent copy — no toast, matching the app's clipboard
                // convention for context-menu actions (Home bubbles, editor).
                iced::clipboard::write(format_log_entry(&entry))
            }
            LogMessage::TabSelected(tab) => {
                self.active_tab = tab;
                // Sync the shared search buffer to the newly-active tab's query.
                let active_search = self.active_search().to_string();
                self.search_buffer.set_text(&active_search);
                self.refresh_active_tab(log_store)
            }
            LogMessage::ToolFailures(msg) => self
                .tool_failures_state
                .update(msg)
                .map(LogMessage::ToolFailures),
        }
    }

    /// Build a tab button element. Returns a highlighted container when the
    /// tab is active, or a plain container when inactive.
    fn tab_button(label: &str, tab: LogsTab, active_tab: LogsTab) -> Element<'_, LogMessage> {
        let is_active = tab == active_tab;
        let color = if is_active {
            theme::ACCENT
        } else {
            theme::TEXT_MUTED
        };
        let b = button(container(text(label.to_string()).size(13).color(color)).padding([6, 14]))
            .style(theme::button_text)
            .on_press(LogMessage::TabSelected(tab));
        if is_active {
            container(b).style(theme::container_bar).into()
        } else {
            container(b).into()
        }
    }

    pub fn view(&self) -> Element<'_, LogMessage> {
        // ── Tab bar (top bar: tabs only) ──────────────────────────
        let all_logs_btn = Self::tab_button("All Logs", LogsTab::AllLogs, self.active_tab);
        let issues_btn = Self::tab_button("Issues", LogsTab::Issues, self.active_tab);
        let tf_btn = Self::tab_button("Tool Failures", LogsTab::ToolFailures, self.active_tab);

        let tab_bar = container(
            row![all_logs_btn, issues_btn, tf_btn]
                .spacing(2)
                .align_y(Alignment::Center),
        )
        .width(Length::Fill)
        .style(theme::surface_container_style);

        // ── Tab content ────────────────────────────────────────────
        let body: Element<'_, LogMessage> = match self.active_tab {
            LogsTab::ToolFailures => self
                .tool_failures_state
                .view()
                .map(LogMessage::ToolFailures),
            LogsTab::AllLogs | LogsTab::Issues => self.logs_view(),
        };

        // ── Log-writer persistence warning banner ──────────────────
        // Surfaces log-store insert failures (the writer task cannot use
        // tracing to report them without recursing into itself).
        let write_error_banner = Self::write_error_banner();

        // ── Bottom bar: pagination + pause + search ───────────────
        let bottom_bar = self.bottom_bar();

        let content = column![tab_bar, write_error_banner, body, bottom_bar]
            .width(Length::Fill)
            .height(Length::Fill);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(theme::base_container_style)
            .into()
    }

    /// Render a warning banner when the log-writer task has observed DB insert
    /// failures (visible on the Logs page — the sanctioned observability
    /// surface for log-persistence outages). Returns an empty element when
    /// there is nothing to report.
    fn write_error_banner() -> Element<'static, LogMessage> {
        let info = crate::logs::log_write_error_info();
        if info.count == 0 {
            return Space::new().height(0).into();
        }

        let last = info
            .last_timestamp
            .as_deref()
            .map(|ts| format!(" (last: {ts})"))
            .unwrap_or_default();
        let detail = info
            .last_message
            .as_deref()
            .unwrap_or("unknown log insert failure");
        let stopped = if info.panic_state.writer_stopped {
            " Log persistence is STOPPED until the daemon restarts."
        } else {
            ""
        };

        container(
            row![
                lucide::triangle_alert::<iced::Theme, iced::Renderer>()
                    .size(14)
                    .color(theme::STATUS_WARNING),
                Space::new().width(6),
                text(format!(
                    "Log store write failures: {} — durable log persistence is degraded{last}. \
                     Latest error: {detail}{stopped}",
                    info.count
                ))
                .size(12)
                .color(theme::TEXT_MUTED),
            ]
            .align_y(Alignment::Center),
        )
        .width(Length::Fill)
        .padding([4, 8])
        .style(|_| {
            theme::container_style(iced::Color::TRANSPARENT, 4.0, 1.0, theme::STATUS_WARNING)
        })
        .into()
    }

    /// Render the bottom bar: pause button, pagination controls, and the
    /// search input. Pagination and search are bound to the active tab; pause
    /// is a single global control that works from every tab.
    #[expect(clippy::too_many_lines)]
    fn bottom_bar(&self) -> Element<'_, LogMessage> {
        let (page, total_pages) = match self.tab_data(self.active_tab) {
            Some(d) => (d.pagination.page, d.pagination.total_pages()),
            None => (
                self.tool_failures_state.page(),
                self.tool_failures_state.total_pages(),
            ),
        };

        // Pagination cluster — hidden entirely when there are no pages so the
        // "Page X of Y" indicator never shows an invalid page.
        let pagination_cluster: Element<'_, LogMessage> = if total_pages == 0 {
            Space::new().width(0).into()
        } else {
            let prev_button = button(text("← Prev").size(12))
                .style(theme::button_text)
                .on_press_maybe(if page > 0 {
                    Some(LogMessage::PrevPage)
                } else {
                    None
                });
            let next_button = button(text("Next →").size(12))
                .style(theme::button_text)
                .on_press_maybe(if page + 1 < total_pages {
                    Some(LogMessage::NextPage)
                } else {
                    None
                });
            row![
                prev_button,
                Space::new().width(8),
                text(format!("Page {} of {}", page + 1, total_pages))
                    .size(12)
                    .color(theme::TEXT_MUTED),
                Space::new().width(8),
                next_button,
            ]
            .align_y(Alignment::Center)
            .into()
        };

        let search_input: Element<'_, LogMessage> = if self.focus_search {
            container(super::widgets::single_line_editor(
                &self.search_buffer.buffer,
                "search",
                false,
                Length::Fill,
                Some(iced::widget::Id::new("logs_search")),
                LogMessage::SearchInput,
            ))
            .padding(2)
            .style(|_| theme::container_style(iced::Color::TRANSPARENT, 4.0, 1.0, theme::ACCENT))
            .into()
        } else {
            super::widgets::single_line_editor(
                &self.search_buffer.buffer,
                "search",
                false,
                Length::Fill,
                Some(iced::widget::Id::new("logs_search")),
                LogMessage::SearchInput,
            )
        };

        let search_group = row![
            lucide::search::<iced::Theme, iced::Renderer>()
                .size(12)
                .color(theme::TEXT_MUTED),
            Space::new().width(4),
            search_input,
        ]
        .align_y(Alignment::Center);

        let pause_button = {
            let pause_btn: iced::Element<'_, LogMessage> = if self.paused {
                lucide::play::<iced::Theme, iced::Renderer>()
                    .size(13)
                    .color(theme::TEXT_MUTED)
                    .into()
            } else {
                lucide::pause::<iced::Theme, iced::Renderer>()
                    .size(13)
                    .color(theme::TEXT_MUTED)
                    .into()
            };
            tooltip(
                button(pause_btn)
                    .style(theme::button_text)
                    .on_press(LogMessage::TogglePause),
                if self.paused { "Resume" } else { "Pause" },
                tooltip::Position::Top,
            )
            .style(theme::tooltip_style)
            .delay(Duration::from_millis(400))
        };

        let bottom_row = row![
            pause_button,
            Space::new().width(12),
            pagination_cluster,
            Space::new().width(Length::Fill),
            search_group,
        ]
        .align_y(Alignment::Center)
        .width(Length::Fill);

        container(bottom_row)
            .width(Length::Fill)
            .padding([8, 24])
            .style(theme::surface_container_style)
            .into()
    }

    /// Render the Logs/Issues tab content (entries list). Pagination lives in
    /// the shared bottom bar.
    fn logs_view(&self) -> Element<'_, LogMessage> {
        // Only the log tabs reach this view (the caller routes Tool Failures
        // separately); fall back to the Tool Failures view rather than panic
        // if a future caller misroutes.
        let Some(data) = self.tab_data(self.active_tab) else {
            return self
                .tool_failures_state
                .view()
                .map(LogMessage::ToolFailures);
        };
        let mut content = Column::new();

        // Error display — inset to align with the vscroll-wrapped entries below.
        content = widgets::push_error_banner_inset(content, data.load_state.error());

        // Log entries
        if data.load_state.loading() && !data.load_state.has_loaded() {
            content = content.push(widgets::scroll_h_inset(widgets::loading_text()));
        } else if data.entries.is_empty() {
            content = content.push(widgets::empty_state_placeholder(
                lucide::activity::<iced::Theme, iced::Renderer>(),
                "No log entries",
            ));
        } else {
            let entries_view = {
                let fade_progress = *self.fade_anim.value();
                let newest_ts = self.newest_entry_timestamp.clone();
                let scroll = widgets::vscroll(
                    Column::with_children(
                        data.entries
                            .iter()
                            .map(|entry| {
                                let is_newest = newest_ts.as_deref() == Some(&entry.timestamp);
                                let rendered = if is_newest && fade_progress < 1.0 {
                                    // Fade-in: render with animated background opacity
                                    LogsState::render_log_entry(entry, fade_progress)
                                } else {
                                    LogsState::render_log_entry(entry, 1.0)
                                };
                                // Per-row context menu: right-click copies the
                                // full entry in formatted form. Wrapping only
                                // the row — not the column — keeps the
                                // inter-row spacing a fall-through (same
                                // convention as the per-bubble menus on the
                                // Home page). Left-click drag selection on the
                                // row is unaffected: ContextMenu forwards all
                                // non-right-click events to the underlay.
                                let row: Element<'_, LogMessage> = ContextMenu::new(
                                    rendered,
                                    vec![MenuItem::new(
                                        "Copy".into(),
                                        LogMessage::CopyEntry(entry.clone()),
                                    )],
                                )
                                .into();
                                row
                            })
                            .collect::<Vec<_>>(),
                    )
                    .spacing(2),
                );

                // Stick to bottom when not paused (latest entries at top, but we
                // want to scroll to latest entries which are at position 0).
                // For new live entries, we insert at position 0, so no scrolling needed.

                scroll
            };

            content = content.push(entries_view);
        }

        let base = container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(theme::base_container_style);

        base.into()
    }

    fn render_log_entry(entry: &LogEntry, fade_progress: f32) -> Element<'_, LogMessage> {
        let (level_color, level_bg) = theme::log_level_color(&entry.level);

        // One selectable Rich widget per entry so a whole log line can be
        // copied in a single drag. Pills are emulated with span highlights.
        let mut spans: Vec<Span<'_, (), iced::Font>> = vec![
            Span::new(theme::format_hhmmss(&entry.timestamp))
                .size(10)
                .color(theme::TEXT_MUTED),
            Span::new("  ").size(10),
            Span::new(&entry.level)
                .size(10)
                .color(level_color)
                .background(level_bg)
                .border(iced::border::rounded(4))
                .padding([1, 6]),
            Span::new("  ").size(10),
        ];
        if !entry.agent_role.is_empty() {
            let (role_color, role_bg) = theme::role_badge_color(&entry.agent_role);
            spans.push(
                Span::new(&entry.agent_role)
                    .size(10)
                    .color(role_color)
                    .background(role_bg)
                    .border(iced::border::rounded(4))
                    .padding([1, 6]),
            );
            spans.push(Span::new("  ").size(10));
        }
        spans.push(
            Span::new(&entry.target)
                .size(11)
                .color(theme::TEXT_SECONDARY),
        );
        if !entry.workspace.is_empty() {
            spans.push(Span::new("  ").size(10));
            spans.push(
                Span::new(&entry.workspace)
                    .size(10)
                    .color(theme::TEXT_MUTED),
            );
        }

        spans.push(Span::new("\n").size(10));
        spans.push(
            Span::new(&entry.message)
                .size(13)
                .color(theme::TEXT_PRIMARY)
                .font(super::JETBRAINS_MONO),
        );

        // Extra fields as key-value tags
        if let Some(obj) = entry.fields.as_object() {
            for (key, value) in obj {
                if key == "message" {
                    continue;
                }
                // Cap the tag text so long fields (e.g. error_chain) render
                // as readable tags instead of overflowing the row; the full
                // value stays queryable in the log store and is copied in
                // full via the row's "Copy" context menu.
                let val_str = crate::util::truncate(&field_value_str(value), 200);
                spans.push(Span::new("\n").size(10));
                spans.push(
                    Span::new(format!("{key}: {val_str}"))
                        .size(10)
                        .color(theme::TEXT_SECONDARY),
                );
            }
        }

        let rich: iced_selection::text::Rich<'_, (), LogMessage, iced::Theme, iced::Renderer> =
            iced_selection::text::Rich::from_iter(spans)
                .style(|_t| iced_selection::text::Style {
                    color: None,
                    selection: theme::ACCENT_DIM,
                })
                .width(Length::Fill);

        if fade_progress < 1.0 {
            // Fade-in: interpolate background/border alpha from 0.6 → 1.0
            container(rich)
                .padding(6)
                .style(move |_theme: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(iced::Color::from_rgba(
                        theme::BG_SURFACE.r,
                        theme::BG_SURFACE.g,
                        theme::BG_SURFACE.b,
                        0.6 + 0.4 * fade_progress,
                    ))),
                    border: iced::Border {
                        radius: 4.0.into(),
                        width: 1.0,
                        color: iced::Color::from_rgba(
                            theme::BORDER.r,
                            theme::BORDER.g,
                            theme::BORDER.b,
                            0.6 + 0.4 * fade_progress,
                        ),
                    },
                    ..container::Style::default()
                })
                .into()
        } else {
            container(rich)
                .padding(6)
                .style(theme::surface_card_style)
                .into()
        }
    }
}

/// Render a stored log-field value as text: strings verbatim, everything else
/// via serde_json's Display (numbers/bools as literals, nested JSON compact).
/// Shared by the row renderer (which then truncates) and the clipboard format
/// (which must stay untruncated).
#[must_use]
fn field_value_str(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Format a [`LogEntry`] for clipboard copy ("Copy" context-menu action): the
/// full raw RFC 3339 timestamp and level on the first line, a header line with
/// the available role/agent/target/workspace parts, then the complete
/// untruncated message, then each remaining field as a `key: value` line.
///
/// Concrete layout (only the parts present in the entry are emitted; the
/// header line is omitted entirely when no role/agent/target/workspace part
/// exists):
///
/// ```text
/// 2026-08-17T20:15:44.581360Z  ERROR
/// role: engineer  agent: ticket_42_engineer  target: run_agent  workspace: mahbot
///
/// The full untruncated message, possibly spanning multiple lines.
/// error_chain: crate::foo at src/foo.rs:42
/// attempt: 3
/// ```
///
/// Field lines iterate the stored fields JSON in **sorted key order** (the
/// format function sorts explicitly, so the output is deterministic regardless
/// of serde_json's `preserve_order` feature state) and skip only the `message`
/// key, mirroring the row renderer. Keys that duplicate header parts (e.g.
/// `agent_id`/`role` inside fields) are kept: the fields section is a faithful
/// dump of the stored JSON, exactly as the row displays them. Non-string
/// values render via [`field_value_str`] — numbers/bools as literals, nested
/// JSON compact.
#[must_use]
fn format_log_entry(entry: &LogEntry) -> String {
    let mut out = String::new();
    out.push_str(&entry.timestamp);
    out.push_str("  ");
    out.push_str(&entry.level);

    // Header: only the parts that are present, fixed order, two-space gaps.
    let mut header: Vec<String> = Vec::new();
    if !entry.agent_role.is_empty() {
        header.push(format!("role: {}", entry.agent_role));
    }
    if !entry.agent_id.is_empty() {
        header.push(format!("agent: {}", entry.agent_id));
    }
    if !entry.target.is_empty() {
        header.push(format!("target: {}", entry.target));
    }
    if !entry.workspace.is_empty() {
        header.push(format!("workspace: {}", entry.workspace));
    }
    if !header.is_empty() {
        out.push('\n');
        out.push_str(&header.join("  "));
    }

    out.push_str("\n\n");
    out.push_str(&entry.message);

    if let Some(obj) = entry.fields.as_object() {
        // Sort explicitly: serde_json's Map is a BTreeMap in this build
        // (sorted), but `preserve_order` could be enabled by feature
        // unification elsewhere in the graph — the sort keeps the copy format
        // deterministic either way.
        let mut keys: Vec<&String> = obj.keys().collect();
        keys.sort();
        for key in keys {
            if key == "message" {
                continue;
            }
            out.push('\n');
            out.push_str(key);
            out.push_str(": ");
            out.push_str(&field_value_str(&obj[key]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry() -> LogEntry {
        LogEntry {
            timestamp: "2026-08-17T20:15:44.581360Z".to_string(),
            level: "ERROR".to_string(),
            target: "run_agent".to_string(),
            message: "Agent failed".to_string(),
            // Written out of order on purpose: fields are emitted in sorted
            // key order, so the fixture order must not leak into the output.
            fields: json!({
                "error_chain": "crate::foo at src/foo.rs:42",
                "message": "not copied",
                "attempt": 3,
                "agent_id": "ticket_42_engineer",
                "role": "engineer",
            }),
            agent_id: "ticket_42_engineer".to_string(),
            agent_role: "engineer".to_string(),
            workspace: "mahbot".to_string(),
        }
    }

    #[test]
    fn format_log_entry_full_layout() {
        let out = format_log_entry(&entry());
        assert_eq!(
            out,
            "2026-08-17T20:15:44.581360Z  ERROR\n\
             role: engineer  agent: ticket_42_engineer  target: run_agent  workspace: mahbot\n\
             \n\
             Agent failed\n\
             agent_id: ticket_42_engineer\n\
             attempt: 3\n\
             error_chain: crate::foo at src/foo.rs:42\n\
             role: engineer"
        );
    }

    #[test]
    fn format_log_entry_minimal_omits_header_and_empty_fields() {
        let entry = LogEntry {
            timestamp: "2026-08-17T20:15:44.581360Z".to_string(),
            level: "INFO".to_string(),
            target: String::new(),
            message: "plain message".to_string(),
            fields: serde_json::Value::Null,
            agent_id: String::new(),
            agent_role: String::new(),
            workspace: String::new(),
        };
        let out = format_log_entry(&entry);
        assert_eq!(
            out,
            "2026-08-17T20:15:44.581360Z  INFO\n\
             \n\
             plain message"
        );
    }
}
