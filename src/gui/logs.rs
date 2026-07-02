//! Logs dashboard page — live log viewing with streaming, filters, pagination,
//! plus a Tool Failures tab for browsing tool error entries.

use crate::logs::{LogEntry, LogQuery, LogStore};

use iced::widget::{
    Column, Row, Space, button, column, container, pick_list, row, scrollable, text, text_input,
    tooltip,
};
use iced::{Alignment, Element, Length, Subscription, Task, window};

use iced_anim::Animated;
use iced_anim::transition::Easing;
use std::time::{Duration, Instant};

use iced_fonts::lucide;

use super::theme;
use super::widgets;
use super::widgets::selectable_text;

/// Tabs within the Logs page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogsTab {
    AllLogs,
    Issues,
    ToolFailures,
}

fn log_stream_producer() -> impl futures_util::Stream<Item = LogMessage> {
    iced::stream::channel(
        1,
        move |mut output: iced::futures::channel::mpsc::Sender<LogMessage>| async move {
            let Some(rx) = super::LOG_BROADCAST.get().and_then(|tx| {
                if tx.receiver_count() > 100 {
                    None
                } else {
                    Some(tx.subscribe())
                }
            }) else {
                return;
            };

            let mut stream = tokio_stream::wrappers::BroadcastStream::new(rx);
            loop {
                match tokio_stream::StreamExt::next(&mut stream).await {
                    Some(Ok(json)) => {
                        if let Ok(entry) = serde_json::from_str::<LogEntry>(&json) {
                            let _ = output.try_send(LogMessage::LiveEntry(entry));
                        }
                    }
                    Some(Err(
                        tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(_n),
                    )) => {
                        let _ = output.try_send(LogMessage::StreamLagged);
                    }
                    None => break,
                }
            }
        },
    )
}

#[derive(Debug, Clone)]
pub enum LogMessage {
    // Data
    Refreshed(Vec<LogEntry>, usize, Vec<super::widgets::PickOption>),
    RefreshError(String),

    // Live stream
    LiveEntry(LogEntry),
    StreamLagged,

    // Filters
    RoleFilterInput(String),
    WorkspaceInput(String),
    SearchInput(String),

    // Tab switching
    TabSelected(LogsTab),

    // Debounced refresh after text input (~300ms)
    DebouncedRefresh(u64),

    // Pagination
    PrevPage,
    NextPage,

    // Pause/Resume
    TogglePause,

    /// Per-frame tick for the fade-in animation.
    AnimTick(Instant),

    /// Dismiss modals/panels (Escape key).
    Escape,

    /// Request toast notification.
    Toast(super::ToastMessage),

    /// Cmd+F keyboard shortcut — highlight the search input.
    FocusSearch,

    /// Bridged Tool Failures sub-messages.
    ToolFailures(super::tool_failures::ToolFailuresMessage),
}

pub struct LogsState {
    entries: Vec<LogEntry>,
    load_state: super::common::AsyncLoadState,

    // Filters
    role_filter: String,
    workspace_filter: String,
    search_filter: String,

    // Dropdown options (populated on refresh)
    role_options: Vec<super::widgets::PickOption>,
    workspace_options: Vec<super::widgets::PickOption>,

    // Pagination
    pagination: super::common::PaginationState,

    // Tab state
    active_tab: LogsTab,
    tool_failures_state: super::tool_failures::ToolFailuresState,

    // Stream control
    paused: bool,

    /// Visual highlight for search input (Cmd+F).
    focus_search: bool,

    /// Timestamp of the most recently received live entry (for fade-in animation).
    newest_entry_timestamp: Option<String>,
    /// Fade progress: 0.0 = just appeared, 1.0 = fully settled.
    fade_anim: Animated<f32>,
    /// Debounce counter for text-input filters. Each keystroke increments this;
    /// only the most recent generation's sleep-task triggers a DB refresh.
    debounce_generation: u64,
    /// True when a debounced refresh is pending (prevents Tick from double-firing).
    debounce_pending: bool,
}

impl LogsState {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            role_filter: String::new(),
            workspace_filter: String::new(),
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
            workspace_options: Vec::new(),
            pagination: super::common::PaginationState::new(50),
            active_tab: LogsTab::AllLogs,
            tool_failures_state: super::tool_failures::ToolFailuresState::new(),
            paused: false,
            focus_search: false,
            newest_entry_timestamp: None,
            fade_anim: Animated::transition(
                0.0f32,
                Easing::EASE_OUT.with_duration(Duration::from_millis(theme::ANIM_LOG_FADE_MS)),
            ),
            debounce_generation: 0,
            debounce_pending: false,
        }
    }

    fn build_query(&self) -> LogQuery {
        let level = match self.active_tab {
            LogsTab::AllLogs => None,
            LogsTab::Issues => Some("ERROR,WARN".to_string()),
            LogsTab::ToolFailures => return LogQuery::default(),
        };

        LogQuery {
            level,
            target: None,
            search: if self.search_filter.is_empty() {
                None
            } else {
                Some(self.search_filter.clone())
            },
            agent_id: None,
            agent_role: if self.role_filter.is_empty() {
                None
            } else {
                Some(self.role_filter.clone())
            },
            workspace: if self.workspace_filter.is_empty() {
                None
            } else {
                Some(self.workspace_filter.clone())
            },
            since: None,
            until: None,
            limit: Some(self.pagination.page_size),
            offset: Some(self.pagination.offset()),
        }
    }

    pub fn refresh(&mut self, log_store: &LogStore) -> Task<LogMessage> {
        self.load_state.start_loading();
        let query = self.build_query();
        let store = log_store.clone();
        Task::perform(
            async move {
                let (log_result, ws_result) = tokio::join!(
                    async { store.query(&query).await.map_err(|e| e.to_string()) },
                    async {
                        crate::workspace::store()
                            .list()
                            .await
                            .map(|ws_list| {
                                ws_list
                                    .into_iter()
                                    .map(|ws| super::widgets::PickOption {
                                        value: ws.path,
                                        label: ws.name,
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default()
                    },
                );
                log_result.map(|(entries, total)| (entries, total, ws_result))
            },
            |res| match res {
                Ok((entries, total, ws_opts)) => LogMessage::Refreshed(entries, total, ws_opts),
                Err(e) => LogMessage::RefreshError(e),
            },
        )
    }

    pub fn subscription(&self) -> Subscription<LogMessage> {
        let stream_sub = if self.paused || self.active_tab != LogsTab::AllLogs {
            Subscription::none()
        } else {
            iced::Subscription::run(log_stream_producer)
        };

        Subscription::batch([stream_sub, window::frames().map(LogMessage::AnimTick)])
    }

    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, msg: LogMessage, log_store: &LogStore) -> Task<LogMessage> {
        match msg {
            LogMessage::Refreshed(entries, total, ws_opts) => {
                self.entries = entries;
                self.pagination.total = total;
                self.load_state.finish_loading();

                // Build workspace options from registry
                self.workspace_options = ws_opts;

                Task::none()
            }
            LogMessage::RefreshError(e) => {
                self.load_state.fail(e);
                Task::none()
            }
            LogMessage::LiveEntry(entry) => {
                // Only prepend live entries when on page 0 (the live view).
                // Other pages are static snapshots from the database.
                if self.pagination.page != 0 {
                    return Task::none();
                }

                // Filter check — only for AllLogs tab; Issues tab uses DB filter.
                // Live entries arrive regardless, but we filter them client-side
                // to match the current active tab and filters.
                let passes = match self.active_tab {
                    LogsTab::AllLogs => true,
                    LogsTab::Issues => entry.level == "ERROR" || entry.level == "WARN",
                    LogsTab::ToolFailures => return Task::none(),
                };
                let passes = passes
                    && (self.role_filter.is_empty()
                        || entry
                            .agent_role
                            .to_lowercase()
                            .contains(&self.role_filter.to_lowercase()));
                let passes = passes
                    && (self.workspace_filter.is_empty()
                        || entry
                            .workspace
                            .to_lowercase()
                            .contains(&self.workspace_filter.to_lowercase()));
                let passes = passes
                    && (self.search_filter.is_empty()
                        || entry
                            .message
                            .to_lowercase()
                            .contains(&self.search_filter.to_lowercase())
                        || entry
                            .target
                            .to_lowercase()
                            .contains(&self.search_filter.to_lowercase()));

                if passes {
                    self.entries.insert(0, entry);
                    self.pagination.total += 1;
                    // Auto-evict: keep exactly page_size entries visible.
                    self.entries.truncate(self.pagination.page_size);
                    // Mark this entry as newest so the view can fade it in.
                    self.newest_entry_timestamp = Some(
                        self.entries
                            .first()
                            .map(|e| e.timestamp.clone())
                            .unwrap_or_default(),
                    );
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
                // On lag, just refresh (but not on ToolFailures tab)
                if self.active_tab == LogsTab::ToolFailures {
                    return Task::none();
                }
                self.refresh(log_store)
            }
            // ── Filter routing based on active tab ─────────────
            LogMessage::RoleFilterInput(v) => match self.active_tab {
                LogsTab::AllLogs | LogsTab::Issues => {
                    self.role_filter = v;
                    self.pagination.reset();
                    self.refresh(log_store)
                }
                LogsTab::ToolFailures => {
                    self.role_filter.clone_from(&v);
                    self.tool_failures_state
                        .update(super::tool_failures::ToolFailuresMessage::RoleFilterInput(
                            v,
                        ))
                        .map(LogMessage::ToolFailures)
                }
            },
            LogMessage::WorkspaceInput(v) => match self.active_tab {
                LogsTab::AllLogs | LogsTab::Issues => {
                    self.workspace_filter = v;
                    self.pagination.reset();
                    self.refresh(log_store)
                }
                LogsTab::ToolFailures => {
                    self.workspace_filter.clone_from(&v);
                    self.tool_failures_state
                        .update(super::tool_failures::ToolFailuresMessage::WorkspaceInput(v))
                        .map(LogMessage::ToolFailures)
                }
            },
            LogMessage::SearchInput(v) => match self.active_tab {
                LogsTab::AllLogs | LogsTab::Issues => {
                    self.search_filter = v;
                    self.pagination.reset();
                    self.debounce_generation = self.debounce_generation.wrapping_add(1);
                    self.debounce_pending = true;
                    let generation = self.debounce_generation;
                    Task::perform(
                        widgets::debounce_sleep(300, generation),
                        LogMessage::DebouncedRefresh,
                    )
                }
                LogsTab::ToolFailures => {
                    self.search_filter.clone_from(&v);
                    self.tool_failures_state
                        .update(super::tool_failures::ToolFailuresMessage::SearchInput(v))
                        .map(LogMessage::ToolFailures)
                }
            },
            LogMessage::DebouncedRefresh(generation) => {
                if self.active_tab == LogsTab::ToolFailures {
                    return Task::none();
                }
                if widgets::debounce_should_process(
                    generation,
                    self.debounce_generation,
                    self.debounce_pending,
                ) {
                    self.debounce_pending = false;
                    return self.refresh(log_store);
                }
                Task::none()
            }
            LogMessage::PrevPage => {
                if self.pagination.prev_page() {
                    return self.refresh(log_store);
                }
                Task::none()
            }
            LogMessage::NextPage => {
                if self.pagination.next_page() {
                    return self.refresh(log_store);
                }
                Task::none()
            }
            LogMessage::TogglePause => {
                self.paused = !self.paused;
                if !self.paused {
                    return self.refresh(log_store);
                }
                Task::none()
            }
            LogMessage::Escape => {
                if self.active_tab == LogsTab::ToolFailures {
                    self.tool_failures_state
                        .update(super::tool_failures::ToolFailuresMessage::Escape)
                        .map(LogMessage::ToolFailures)
                } else {
                    self.focus_search = false;
                    Task::none()
                }
            }
            LogMessage::Toast(_) => Task::none(),
            LogMessage::FocusSearch => {
                if self.active_tab == LogsTab::ToolFailures {
                    self.tool_failures_state
                        .update(super::tool_failures::ToolFailuresMessage::FocusSearch)
                        .map(LogMessage::ToolFailures)
                } else {
                    self.focus_search = true;
                    Task::none()
                }
            }
            LogMessage::TabSelected(tab) => {
                self.active_tab = tab;
                if tab == LogsTab::ToolFailures {
                    // Refresh the tool failures data when switching to that tab
                    self.tool_failures_state
                        .refresh()
                        .map(LogMessage::ToolFailures)
                } else {
                    // Refresh logs when switching to AllLogs or Issues
                    self.refresh(log_store)
                }
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
            container(b)
                .style(|_t: &iced::Theme| container::Style {
                    background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                    ..container::Style::default()
                })
                .into()
        } else {
            container(b).into()
        }
    }

    pub fn view(&self) -> Element<'_, LogMessage> {
        // ── Tab bar ───────────────────────────────────────────────
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

        // ── Shared filter bar ─────────────────────────────────────
        let filter_bar = self.shared_filter_bar();

        // ── Tab content ────────────────────────────────────────────
        let body: Element<'_, LogMessage> = match self.active_tab {
            LogsTab::ToolFailures => self
                .tool_failures_state
                .view()
                .map(LogMessage::ToolFailures),
            LogsTab::AllLogs | LogsTab::Issues => self.logs_view(),
        };

        let content = column![tab_bar, filter_bar, body]
            .width(Length::Fill)
            .height(Length::Fill);

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .style(theme::base_container_style)
            .into()
    }

    /// Render the shared filter bar: role picklist, workspace picklist, search input.
    #[allow(clippy::too_many_lines)]
    fn shared_filter_bar(&self) -> Element<'_, LogMessage> {
        let search_input: Element<'_, LogMessage> = if self.focus_search {
            container(
                text_input("search", &self.search_filter)
                    .on_input(LogMessage::SearchInput)
                    .style(super::widgets::text_input_style)
                    .size(13)
                    .padding(4)
                    .width(Length::Fixed(160.0)),
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
            text_input("search", &self.search_filter)
                .on_input(LogMessage::SearchInput)
                .style(super::widgets::text_input_style)
                .size(13)
                .padding(4)
                .width(Length::Fixed(160.0))
                .into()
        };

        let role_pick_list = {
            let role_selected = self
                .role_options
                .iter()
                .find(|o| o.value == self.role_filter)
                .cloned();
            pick_list(self.role_options.as_slice(), role_selected, |opt| {
                LogMessage::RoleFilterInput(opt.value)
            })
            .placeholder("Role")
            .style(super::widgets::pick_list_style)
            .padding([4, 8])
            .width(Length::Fixed(100.0))
        };

        let workspace_pick_list = {
            let ws_selected = self
                .workspace_options
                .iter()
                .find(|o| o.value == self.workspace_filter)
                .cloned();
            pick_list(self.workspace_options.as_slice(), ws_selected, |opt| {
                LogMessage::WorkspaceInput(opt.value)
            })
            .placeholder("Workspace")
            .style(super::widgets::pick_list_style)
            .padding([4, 8])
            .width(Length::Fixed(120.0))
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

        let filter_row = row![
            // Pause only visible on AllLogs tab
            match self.active_tab {
                LogsTab::AllLogs => {
                    iced::Element::<'_, LogMessage>::from(pause_button)
                }
                _ => iced::Element::<'_, LogMessage>::from(Space::new().width(0)),
            },
            Space::new().width(Length::Fill),
            role_pick_list,
            Space::new().width(Length::Fill),
            workspace_pick_list,
            Space::new().width(Length::Fill),
            search_group,
        ]
        .align_y(Alignment::Center)
        .width(Length::Fill);

        container(filter_row)
            .width(Length::Fill)
            .padding([8, 24])
            .style(theme::surface_container_style)
            .into()
    }

    /// Render the Logs/Issues tab content (entries list + pagination).
    fn logs_view(&self) -> Element<'_, LogMessage> {
        let mut content = Column::new();

        // Error display
        if let Some(err) = self.load_state.error() {
            content = content.push(widgets::error_banner(err));
            content = content.push(Space::new().height(8));
        }

        // Log entries
        if self.load_state.loading() && !self.load_state.has_loaded() {
            content = content.push(text("Loading...").size(14).color(theme::TEXT_MUTED));
        } else if self.entries.is_empty() {
            content = content.push(widgets::empty_state_placeholder(
                lucide::activity::<iced::Theme, iced::Renderer>(),
                "No log entries",
            ));
        } else {
            let entries_view = {
                let fade_progress = *self.fade_anim.value();
                let newest_ts = self.newest_entry_timestamp.clone();
                let scroll = scrollable(
                    Column::with_children(
                        self.entries
                            .iter()
                            .map(|entry| {
                                let is_newest = newest_ts.as_deref() == Some(&entry.timestamp);
                                if is_newest && fade_progress < 1.0 {
                                    // Fade-in: render with animated background opacity
                                    LogsState::render_log_entry(entry, fade_progress)
                                } else {
                                    LogsState::render_log_entry(entry, 1.0)
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                    .spacing(2),
                )
                .height(Length::Fill)
                .direction(theme::vertical_scrollbar())
                .style(theme::scrollbar_style);

                // Stick to bottom when not paused (latest entries at top, but we
                // want to scroll to latest entries which are at position 0).
                // For new live entries, we insert at position 0, so no scrolling needed.

                scroll
            };

            content = content.push(entries_view);
        }

        // Pagination bar
        content = content.push(widgets::pagination_bar(
            self.pagination.page,
            self.pagination.total_pages(),
            LogMessage::PrevPage,
            LogMessage::NextPage,
        ));

        let base = container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(theme::base_container_style);

        base.into()
    }

    #[allow(clippy::too_many_lines)]
    fn render_log_entry(entry: &LogEntry, fade_progress: f32) -> Element<'_, LogMessage> {
        let (level_color, level_bg) = theme::log_level_color(&entry.level);
        let role_color = if entry.agent_role.is_empty() {
            theme::TEXT_MUTED
        } else {
            theme::role_badge_color(&entry.agent_role).0
        };

        let timestamp = if entry.timestamp.len() > 19 {
            &entry.timestamp[11..19] // Extract HH:MM:SS from ISO 8601
        } else {
            &entry.timestamp
        };

        let mut entry_view = column![
            row![
                text(timestamp).size(10).color(theme::TEXT_MUTED),
                Space::new().width(8),
                container(text(&entry.level).size(10).color(level_color))
                    .padding([1, 6])
                    .style(move |_theme: &iced::Theme| container::Style {
                        background: Some(iced::Background::Color(level_bg)),
                        border: iced::Border {
                            radius: 3.0.into(),
                            ..iced::Border::default()
                        },
                        ..container::Style::default()
                    }),
                Space::new().width(8),
                if !entry.agent_role.is_empty() {
                    row![
                        container(text(&entry.agent_role).size(10).color(role_color))
                            .padding([1, 6])
                            .style(move |_theme: &iced::Theme| container::Style {
                                background: Some(iced::Background::Color(iced::Color::from_rgba(
                                    role_color.r,
                                    role_color.g,
                                    role_color.b,
                                    0.1,
                                ),)),
                                border: iced::Border {
                                    radius: 3.0.into(),
                                    ..iced::Border::default()
                                },
                                ..container::Style::default()
                            }),
                        Space::new().width(4),
                    ]
                } else {
                    row![]
                },
                text(&entry.target).size(11).color(theme::TEXT_SECONDARY),
                Space::new().width(Length::Fill),
                if !entry.workspace.is_empty() {
                    text(&entry.workspace).size(10).color(theme::TEXT_MUTED)
                } else {
                    text("")
                },
            ]
            .align_y(Alignment::Center)
            .spacing(2),
            Space::new().height(2),
            selectable_text(&entry.message, theme::TEXT_PRIMARY)
                .size(13)
                .font(super::JETBRAINS_MONO)
                .width(Length::Fill),
        ]
        .spacing(1);

        // Extra fields as key-value tags
        if entry.fields != serde_json::Value::Null {
            if let Some(obj) = entry.fields.as_object() {
                let mut tags = Row::new().spacing(4);
                let mut has_tags = false;
                for (key, value) in obj {
                    if key == "message" {
                        continue;
                    }
                    has_tags = true;
                    let val_str = match value {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    tags = tags.push(
                        container(
                            text(format!("{key}: {val_str}"))
                                .size(10)
                                .color(theme::TEXT_MUTED),
                        )
                        .padding([1, 4])
                        .style(|_theme: &iced::Theme| container::Style {
                            background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                            border: iced::Border {
                                radius: 3.0.into(),
                                ..iced::Border::default()
                            },
                            ..container::Style::default()
                        }),
                    );
                }
                if has_tags {
                    entry_view = column![entry_view, Space::new().height(2), tags];
                }
            }
        }

        if fade_progress < 1.0 {
            // Fade-in: interpolate background/border alpha from 0.6 → 1.0
            container(entry_view)
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
            container(entry_view)
                .padding(6)
                .style(theme::surface_card_style)
                .into()
        }
    }
}
