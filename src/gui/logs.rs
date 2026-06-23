//! Logs dashboard page — live log viewing with streaming, filters, pagination.

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

fn log_stream_producer() -> impl futures_util::Stream<Item = LogMessage> {
    iced::stream::channel(
        1,
        move |mut output: iced::futures::channel::mpsc::Sender<LogMessage>| async move {
            let rx = match super::LOG_BROADCAST.get().and_then(|tx| {
                if tx.receiver_count() > 100 {
                    None
                } else {
                    Some(tx.subscribe())
                }
            }) {
                Some(rx) => rx,
                None => return,
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
    ToggleIssuesOnly,
    RoleFilterInput(String),
    WorkspaceInput(String),
    TargetInput(String),
    SearchInput(String),

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
}

pub struct LogsState {
    entries: Vec<LogEntry>,
    total: usize,
    error: Option<String>,
    loading: bool,
    /// Whether at least one refresh has completed (prevents "Loading..." flicker
    /// on empty datasets when auto-poll Ticks).
    has_loaded: bool,

    // Filters
    issues_only: bool,
    role_filter: String,
    workspace_filter: String,
    target_filter: String,
    search_filter: String,

    // Dropdown options (populated on refresh)
    role_options: Vec<super::widgets::PickOption>,
    workspace_options: Vec<super::widgets::PickOption>,

    // Pagination
    page: usize,
    page_size: usize,

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
            total: 0,
            error: None,
            loading: false,
            has_loaded: false,
            issues_only: false,
            role_filter: String::new(),
            workspace_filter: String::new(),
            target_filter: String::new(),
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
            page: 0,
            page_size: 50,
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
        let level = if self.issues_only {
            Some("ERROR,WARN".to_string())
        } else {
            None
        };

        LogQuery {
            level,
            target: if self.target_filter.is_empty() {
                None
            } else {
                Some(self.target_filter.clone())
            },
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
            limit: Some(self.page_size),
            offset: Some(self.page * self.page_size),
        }
    }

    const fn total_pages(&self) -> usize {
        if self.total == 0 {
            0
        } else {
            (self.total + self.page_size - 1) / self.page_size
        }
    }

    pub fn refresh(&mut self, log_store: &LogStore) -> Task<LogMessage> {
        self.loading = true;
        self.error = None;
        let query = self.build_query();
        let store = log_store.clone();
        Task::perform(
            async move {
                let log_result = store.query(&query).await.map_err(|e| e.to_string());
                let ws_options = crate::workspace::store()
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
                    .unwrap_or_default();
                log_result.map(|(entries, total)| (entries, total, ws_options))
            },
            |res| match res {
                Ok((entries, total, ws_opts)) => LogMessage::Refreshed(entries, total, ws_opts),
                Err(e) => LogMessage::RefreshError(e),
            },
        )
    }

    pub fn subscription(&self) -> Subscription<LogMessage> {
        let stream_sub = if self.paused {
            Subscription::none()
        } else {
            iced::Subscription::run(log_stream_producer)
        };

        Subscription::batch([stream_sub, window::frames().map(LogMessage::AnimTick)])
    }

    pub fn update(&mut self, msg: LogMessage, log_store: &LogStore) -> Task<LogMessage> {
        match msg {
            LogMessage::Refreshed(entries, total, ws_opts) => {
                self.entries = entries;
                self.total = total;
                self.loading = false;
                self.has_loaded = true;

                // Build role options from Role::iter()
                self.role_options = <crate::Role as strum::IntoEnumIterator>::iter()
                    .map(|r| {
                        let name = r.to_string();
                        super::widgets::PickOption {
                            value: name.clone(),
                            label: name,
                        }
                    })
                    .collect();

                // Build workspace options from registry (name → path lookup)
                self.workspace_options = ws_opts;

                Task::none()
            }
            LogMessage::RefreshError(e) => {
                self.error = Some(e);
                self.loading = false;
                Task::none()
            }
            LogMessage::LiveEntry(entry) => {
                // Only prepend live entries when on page 0 (the live view).
                // Other pages are static snapshots from the database.
                if self.page != 0 {
                    return Task::none();
                }

                // Filter check
                let passes = !self.issues_only || entry.level == "ERROR" || entry.level == "WARN";
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
                    && (self.target_filter.is_empty()
                        || entry
                            .target
                            .to_lowercase()
                            .starts_with(&self.target_filter.to_lowercase()));
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
                    self.total += 1;
                    // Auto-evict: keep exactly page_size entries visible.
                    self.entries.truncate(self.page_size);
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
                // On lag, just refresh
                self.refresh(log_store)
            }
            LogMessage::ToggleIssuesOnly => {
                self.issues_only = !self.issues_only;
                self.page = 0;
                self.refresh(log_store)
            }
            LogMessage::RoleFilterInput(v) => {
                self.role_filter = v;
                self.page = 0;
                self.refresh(log_store)
            }
            LogMessage::WorkspaceInput(v) => {
                self.workspace_filter = v;
                self.page = 0;
                self.refresh(log_store)
            }
            LogMessage::TargetInput(v) => {
                self.target_filter = v;
                self.page = 0;
                self.debounce_generation = self.debounce_generation.wrapping_add(1);
                self.debounce_pending = true;
                let generation = self.debounce_generation;
                Task::perform(
                    widgets::debounce_sleep(300, generation),
                    LogMessage::DebouncedRefresh,
                )
            }
            LogMessage::SearchInput(v) => {
                self.search_filter = v;
                self.page = 0;
                self.debounce_generation = self.debounce_generation.wrapping_add(1);
                self.debounce_pending = true;
                let generation = self.debounce_generation;
                Task::perform(
                    widgets::debounce_sleep(300, generation),
                    LogMessage::DebouncedRefresh,
                )
            }
            LogMessage::DebouncedRefresh(generation) => {
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
                if self.page > 0 {
                    self.page -= 1;
                    return self.refresh(log_store);
                }
                Task::none()
            }
            LogMessage::NextPage => {
                if self.page + 1 < self.total_pages() {
                    self.page += 1;
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
                self.focus_search = false;
                Task::none()
            }
            LogMessage::Toast(_) => Task::none(),
            LogMessage::FocusSearch => {
                self.focus_search = true;
                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<'_, LogMessage> {
        let mut content = Column::new();

        // Error display
        if let Some(ref err) = self.error {
            content = content.push(widgets::error_banner(err));
            content = content.push(Space::new().height(8));
        }

        // Filter bar
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

        let issues_toggle = button(if self.issues_only {
            iced::Element::<'_, LogMessage>::from(row![
                lucide::triangle_alert::<iced::Theme, iced::Renderer>()
                    .size(11)
                    .color(theme::TEXT_MUTED),
                Space::new().width(4),
                text("Only Issues").size(12),
            ])
        } else {
            iced::Element::<'_, LogMessage>::from(row![
                lucide::info::<iced::Theme, iced::Renderer>()
                    .size(11)
                    .color(theme::TEXT_MUTED),
                Space::new().width(4),
                text("All Logs").size(12),
            ])
        })
        .style(theme::button_text)
        .on_press(LogMessage::ToggleIssuesOnly);

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

        let target_input = text_input("target", &self.target_filter)
            .on_input(LogMessage::TargetInput)
            .style(super::widgets::text_input_style)
            .size(13)
            .padding(4)
            .width(Length::Fixed(100.0));

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
            .delay(Duration::from_millis(400))
        };

        let filter_row = row![
            issues_toggle,
            Space::new().width(8),
            pause_button,
            Space::new().width(Length::Fill),
            role_pick_list,
            Space::new().width(Length::Fill),
            workspace_pick_list,
            Space::new().width(Length::Fill),
            target_input,
            Space::new().width(Length::Fill),
            search_group,
        ]
        .align_y(Alignment::Center)
        .width(Length::Fill);

        content = content.push(filter_row);
        content = content.push(Space::new().height(8));

        // Log entries
        if self.loading && !self.has_loaded {
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
                .direction(scrollable::Direction::Vertical(theme::thin_scrollbar()))
                .style(theme::scrollbar_style);

                // Stick to bottom when not paused (latest entries at top, but we
                // want to scroll to latest entries which are at position 0).
                // For new live entries, we insert at position 0, so no scrolling needed.

                scroll
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
                        Some(LogMessage::PrevPage)
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
                        Some(LogMessage::NextPage)
                    } else {
                        None
                    }),
            ]
            .align_y(Alignment::Center);

            content = content.push(Space::new().height(8));
            content = content.push(pagination);
        }

        let base = container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(|_theme: &iced::Theme| container::Style {
                background: Some(iced::Background::Color(theme::BG_BASE)),
                ..container::Style::default()
            });

        base.into()
    }

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
}
