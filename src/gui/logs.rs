//! Logs dashboard page — live log viewing with a per-view pagination and
//! search over the log store.
//!
//! The page has no rail entry: the dashboard's Cmd+L (Ctrl+L elsewhere)
//! shortcut is its only way in.
//!
//! Each view (All Logs / Only Issues) keeps its own entries, pagination state
//! and search query, so switching views never reuses another view's page index
//! or search. The bottom bar holds pagination, the switch between the two views,
//! this view's clear button (deletes the view's records from the store, behind
//! the shared confirmation modal) and search.

use crate::logs::{LogEntry, LogQuery, LogStore};

use iced::advanced::text::Span;
use iced::widget::{Space, button, column, container, row, stack, text};
use iced::{Alignment, Element, Length, Subscription, Task, window};
use iced_anim::Animated;
use iced_anim::transition::Easing;
use std::time::{Duration, Instant};

use iced_fonts::lucide;

use super::common::PaginatedViewState;
use super::dialog;
use super::menus::{ContextMenu, MenuItem};
use super::theme;
use super::widgets;

/// The two listings the Logs page switches between.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogsView {
    AllLogs,
    Issues,
}

impl LogsView {
    /// The level filter this view's listing query uses — and therefore the exact
    /// scope of its clear button, which always empties the whole view and never
    /// the searched subset. All Logs means every level.
    fn level_filter(self) -> Option<&'static str> {
        match self {
            Self::Issues => Some("ERROR,WARN"),
            Self::AllLogs => None,
        }
    }

    /// Label of this view's segment in the switch.
    fn switch_label(self) -> &'static str {
        match self {
            Self::AllLogs => "All Logs",
            Self::Issues => "Only Issues",
        }
    }

    /// Label of this view's clear button (also the title of its confirmation).
    fn clear_label(self) -> &'static str {
        match self {
            Self::AllLogs => "Clear logs",
            Self::Issues => "Clear issues",
        }
    }

    /// Consequence paragraphs for this view's clear confirmation: the scope the
    /// button names, including where the deleted records are shown elsewhere.
    fn clear_consequences(self) -> [&'static str; 3] {
        match self {
            Self::AllLogs => [
                "This permanently deletes every stored log record — informational, \
                 warning and error — for every workspace and user.",
                "The Issues view shows warnings and errors from the same records, so \
                 it is emptied too.",
                "Nothing can be recovered.",
            ],
            Self::Issues => [
                "This permanently deletes every stored warning and error record, for \
                 every workspace and user.",
                "Informational records are not touched and stay in the All Logs view.",
                "Nothing can be recovered.",
            ],
        }
    }
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
    // Data — tagged with the originating view and refresh generation so stale
    // async responses (issued before a newer refresh of the same view) are
    // dropped instead of overwriting newer data.
    Refreshed(LogsView, u64, Vec<LogEntry>, usize),
    RefreshError(LogsView, u64, String),

    // Live stream
    LiveEntry(LogEntry),
    StreamLagged,

    // Search
    SearchInput(super::editor_widget::EditorAction),

    // View switching
    ViewSelected(LogsView),

    // Debounced refresh after text input (~300ms). Carries the view whose
    // query changed so the refresh still targets that view if the user
    // switches views within the debounce window.
    DebouncedRefresh(u64, LogsView),

    // Pagination
    PrevPage,
    NextPage,

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

    /// Clear button pressed — ask for confirmation before deleting anything.
    RequestClear,
    /// The confirmation's danger action — delete the pending view's records.
    ClearConfirmed,
    /// The confirmation was dismissed (Keep/backdrop/Escape) — nothing deleted.
    ClearDismissed,
    /// A clear finished. `Ok` = the delete committed; `Err` carries the failure
    /// to surface in that view's error banner. Records already buffered by the
    /// log writer may still land after it.
    Cleared(LogsView, Result<(), String>),
}

/// Per-view state shared by the All Logs and Issues views. Each view keeps its
/// own entries, load state, pagination and search query so switching views
/// never reuses another view's page index or query.
type LogsViewData = PaginatedViewState<LogEntry>;

pub struct LogsState {
    /// All Logs view data (live-streamed).
    all_logs: LogsViewData,
    /// Issues view data (ERROR/WARN only).
    issues: LogsViewData,

    /// The view on screen.
    active_view: LogsView,

    /// Visual highlight for search input (Cmd+F).
    focus_search: bool,

    /// Timestamp of the most recently received live entry (for fade-in animation).
    newest_entry_timestamp: Option<String>,
    /// Fade progress: 0.0 = just appeared, 1.0 = fully settled.
    fade_anim: Animated<f32>,
    /// Debounce state for search-text input filtering.
    debounce: super::common::DebounceState,
    /// Buffer for the shared search input (bound to the active view's query).
    search_buffer: super::common::SingleLineEditorState,

    /// View whose clear button awaits confirmation. `Some` while the confirm
    /// modal is open — nothing is deleted without it.
    pending_clear: Option<LogsView>,
}

impl LogsState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            all_logs: PaginatedViewState::new(50),
            issues: PaginatedViewState::new(50),
            active_view: LogsView::AllLogs,
            focus_search: false,
            newest_entry_timestamp: None,
            fade_anim: Animated::transition(
                0.0f32,
                Easing::EASE_OUT.with_duration(Duration::from_millis(theme::ANIM_LOG_FADE_MS)),
            ),
            debounce: super::common::DebounceState::new(),
            search_buffer: super::common::SingleLineEditorState::new(""),
            pending_clear: None,
        }
    }

    /// Immutable access to one view's data.
    fn view_data(&self, view: LogsView) -> &LogsViewData {
        match view {
            LogsView::AllLogs => &self.all_logs,
            LogsView::Issues => &self.issues,
        }
    }

    /// Mutable access to one view's data.
    fn view_data_mut(&mut self, view: LogsView) -> &mut LogsViewData {
        match view {
            LogsView::AllLogs => &mut self.all_logs,
            LogsView::Issues => &mut self.issues,
        }
    }

    /// The active view's search query (bound to the shared search input).
    fn active_search(&self) -> &str {
        &self.view_data(self.active_view).search
    }

    /// Reset pagination for whichever view is currently active.
    fn reset_pagination_for_active_view(&mut self) {
        self.view_data_mut(self.active_view).pagination.reset();
    }

    /// Refresh whichever view is currently active.
    fn refresh_active_view(&mut self, log_store: &LogStore) -> Task<LogMessage> {
        self.refresh_view(log_store, self.active_view)
    }

    /// Public entry point: refresh the currently active view (used at boot).
    pub fn refresh(&mut self, log_store: &LogStore) -> Task<LogMessage> {
        self.refresh_active_view(log_store)
    }

    /// Issue a refresh for one specific view, tagging the async response with
    /// its origin view and a generation counter so stale responses (issued
    /// before a newer refresh of the same view) are dropped by the handler.
    fn refresh_view(&mut self, log_store: &LogStore, view: LogsView) -> Task<LogMessage> {
        let data = self.view_data_mut(view);
        let generation = data.begin_refresh();
        let query = LogQuery {
            level: view.level_filter().map(str::to_owned),
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
                    LogMessage::Refreshed(view, generation, entries, total)
                }
                Err((generation, e)) => LogMessage::RefreshError(view, generation, e),
            },
        )
    }

    pub fn subscription(&self) -> Subscription<LogMessage> {
        // Live log stream only while the All Logs view is active. Frame ticks are subscribed only while the fade animation is running —
        // an always-on frames() subscription closes a self-sustaining redraw
        // loop (redraw → AnimTick → redraw). A hidden window draws nothing, so
        // the fade waits for the window to come back rather than ticking for a
        // frame nobody sees.
        let mut subs: Vec<Subscription<LogMessage>> = Vec::new();
        if self.active_view == LogsView::AllLogs {
            subs.push(iced::Subscription::run(log_stream_producer));
        }
        if self.fade_anim.is_animating() && !super::app_window::hidden() {
            subs.push(window::frames().map(LogMessage::AnimTick));
        }
        Subscription::batch(subs)
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: LogMessage, log_store: &LogStore) -> Task<LogMessage> {
        match msg {
            LogMessage::Refreshed(view, generation, entries, total) => {
                if self
                    .view_data_mut(view)
                    .handle_refreshed(generation, entries, total)
                {
                    return self.refresh_view(log_store, view);
                }
                Task::none()
            }
            LogMessage::RefreshError(view, generation, e) => {
                self.view_data_mut(view).handle_refresh_error(generation, e);
                Task::none()
            }
            LogMessage::RequestClear => {
                self.pending_clear = Some(self.active_view);
                Task::none()
            }
            LogMessage::ClearDismissed => {
                self.pending_clear = None;
                Task::none()
            }
            LogMessage::ClearConfirmed => {
                // The dialog names the view it was opened for, not whichever view
                // is active now, so a view switch behind the modal cannot retarget it.
                let Some(view) = self.pending_clear.take() else {
                    return Task::none();
                };
                let store = log_store.clone();
                Task::perform(
                    // The clear scope is the view's own level filter: All Logs
                    // carries none and empties the table.
                    async move { store.clear_logs(view.level_filter()).await.map(|_| ()) },
                    move |result| LogMessage::Cleared(view, result.map_err(|e| e.to_string())),
                )
            }
            LogMessage::Cleared(view, result) => match result {
                Ok(()) => {
                    // Post-clear: the pressed view returns to its first page and
                    // keeps its typed search; its count is re-read by the
                    // refreshes below, never adjusted in place. Both views read
                    // the same table, so the other one holds rows this clear just
                    // deleted and is re-read too.
                    self.view_data_mut(view).clear_entries();
                    Task::batch(
                        [LogsView::AllLogs, LogsView::Issues]
                            .map(|view| self.refresh_view(log_store, view)),
                    )
                }
                // This surface has no completion-notice plumbing; the view's
                // error banner is the channel for a failed clear.
                Err(e) => {
                    self.view_data_mut(view).load_state.fail(e);
                    Task::none()
                }
            },
            LogMessage::LiveEntry(entry) => {
                // Live entries only arrive while the All Logs view is active
                // (the subscription is gated in `subscription()`).
                let data = &mut self.all_logs;
                // Only prepend live entries when on page 0 (the live view).
                // Other pages are static snapshots from the database.
                if data.pagination.page != 0 {
                    return Task::none();
                }

                // Client-side search filter matching this view's query: the
                // live filter and the stored-row query cover the same four
                // parts — target, message, workspace and ticket. (The level
                // filter comes from the DB query for the Issues view, which
                // never receives live entries.)
                if entry.matches_search(&data.search) {
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
                self.refresh_active_view(log_store)
            }
            LogMessage::SearchInput(action) => {
                // Single-line Tab moves focus rather than editing.
                if let Some(task) = super::common::focus_navigation_task(&action) {
                    return task;
                }
                // The search input is bound to the active view's own query.
                let changes_text = action.changes_text();
                self.search_buffer.apply_action(action);
                let text = self.search_buffer.text();
                self.view_data_mut(self.active_view).search = text;
                // Only text changes re-trigger the debounced refresh; cursor
                // movement emits actions too but shouldn't restart the timer.
                if !changes_text {
                    return Task::none();
                }
                self.reset_pagination_for_active_view();
                // Tag the debounced refresh with the view whose query changed
                // so it still targets that view if the user switches away
                // within the debounce window.
                let view = self.active_view;
                self.debounce
                    .trigger(300)
                    .map(move |g| LogMessage::DebouncedRefresh(g, view))
            }
            LogMessage::DebouncedRefresh(generation, view) => {
                if !self.debounce.should_process(generation) {
                    return Task::none();
                }
                self.refresh_view(log_store, view)
            }
            LogMessage::PrevPage => {
                if self.view_data_mut(self.active_view).pagination.prev_page() {
                    return self.refresh_active_view(log_store);
                }
                Task::none()
            }
            LogMessage::NextPage => {
                if self.view_data_mut(self.active_view).pagination.next_page() {
                    return self.refresh_active_view(log_store);
                }
                Task::none()
            }
            LogMessage::Escape => {
                // Escape never confirms: while the confirm modal is open it only
                // dismisses it.
                if self.pending_clear.take().is_some() {
                    return Task::none();
                }
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
            LogMessage::ViewSelected(view) => {
                self.active_view = view;
                // Sync the shared search buffer to the newly-active view's query.
                let active_search = self.active_search().to_string();
                self.search_buffer.set_text(&active_search);
                self.refresh_active_view(log_store)
            }
        }
    }

    /// One position of the bottom bar's view switch. Both labels always render;
    /// the active one is accent-colored on an elevated fill, so the current
    /// position is read off the control rather than inferred from a label.
    fn view_switch_segment(view: LogsView, active: LogsView) -> Element<'static, LogMessage> {
        let is_active = view == active;
        let color = if is_active {
            theme::ACCENT
        } else {
            theme::TEXT_FAINT
        };
        let segment = button(text(view.switch_label()).size(theme::TEXT_11).color(color))
            .style(theme::button_text)
            .padding([theme::PAD_2, theme::PAD_8])
            .on_press_maybe((!is_active).then_some(LogMessage::ViewSelected(view)));
        container(segment)
            .style(move |_| {
                let background = if is_active {
                    theme::BG_ELEVATED
                } else {
                    iced::Color::TRANSPARENT
                };
                theme::container_style(background, 4.0, 0.0, iced::Color::TRANSPARENT)
            })
            .into()
    }

    /// The view switch as one hairline-bordered pill sitting in the bottom bar.
    /// Rendered unconditionally: it is the only view switcher on the page, so a
    /// store with nothing to paginate or list must not hide it.
    fn view_switch(active: LogsView) -> Element<'static, LogMessage> {
        container(
            row![
                Self::view_switch_segment(LogsView::AllLogs, active),
                Self::view_switch_segment(LogsView::Issues, active),
            ]
            .spacing(theme::SPACE_2),
        )
        .padding(theme::PAD_1)
        .style(|_| theme::container_style(iced::Color::TRANSPARENT, 5.0, 1.0, theme::BORDER))
        .into()
    }

    pub fn view(&self) -> Element<'_, LogMessage> {
        let body: Element<'_, LogMessage> = self.logs_view();

        // ── Log-writer persistence warning banner ──────────────────
        // Surfaces log-store insert failures (the writer task cannot use
        // tracing to report them without recursing into itself).
        let write_error_banner = Self::write_error_banner();

        // ── Bottom bar: pagination + view switch + clear + search ──
        let bottom_bar = self.bottom_bar();

        let content = column![write_error_banner, body, bottom_bar]
            .width(Length::Fill)
            .height(Length::Fill);

        let page = widgets::page_bare(content);
        // Confirm-dialog overlay: the page is stack child 0, the dialog (or a
        // type-stable placeholder) child 1 — the shapes never change, so the
        // page keeps its widget state as the modal opens/closes (same pattern
        // as the Running page).
        let confirm_layer: Element<'_, LogMessage> = match self.pending_clear {
            Some(view) => Self::clear_confirm_dialog(view),
            None => stack([widgets::empty_stack_placeholder()]).into(),
        };
        stack([page, confirm_layer]).into()
    }

    /// The clear confirmation for `view`: the shared confirm-dialog/backdrop
    /// pair, with the view's scope named in the title, the danger button and the
    /// listed consequences. There is no undo, so nothing is deleted without it.
    fn clear_confirm_dialog(view: LogsView) -> Element<'static, LogMessage> {
        widgets::modal_backdrop(
            dialog::confirm_dialog(
                dialog::dialog_title(format!("{}?", view.clear_label())),
                dialog::dialog_body(view.clear_consequences()),
                [
                    dialog::DialogAction::secondary("Keep records", LogMessage::ClearDismissed),
                    dialog::DialogAction::danger(view.clear_label(), LogMessage::ClearConfirmed),
                ],
            ),
            LogMessage::ClearDismissed,
            0.5,
        )
    }

    /// Render a warning banner for the log store's own bad news: the write
    /// failures observed on it — the log-writer task's failed batches and panics,
    /// and a panic out of one telemetry record's own write — and the telemetry rows
    /// a full queue dropped (visible on the Logs page — the sanctioned
    /// observability surface for log-persistence outages). Each number is reported
    /// on its own sentence, so neither hides the other. Returns an empty element
    /// when there is nothing to report.
    fn write_error_banner() -> Element<'static, LogMessage> {
        let info = crate::logs::log_write_error_info();
        if info.count == 0 && info.dropped == 0 {
            return Space::new().height(0).into();
        }

        let mut report = Vec::new();
        if info.count > 0 {
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
            report.push(format!(
                "Log store write failures: {} — records are not reaching the log store{last}. \
                 Latest error: {detail}{stopped}",
                info.count
            ));
        }
        if info.dropped > 0 {
            report.push(format!(
                "Telemetry rows dropped by a full queue: {} — recording never waits for the \
                 store, so a burst with nowhere to go is dropped.",
                info.dropped
            ));
        }

        container(
            row![
                lucide::triangle_alert::<iced::Theme, iced::Renderer>()
                    .size(theme::TEXT_14)
                    .color(theme::STATUS_WARNING),
                Space::new().width(theme::SPACE_6),
                text(report.join(" "))
                    .size(theme::TEXT_12)
                    .color(theme::TEXT_SECONDARY),
            ]
            .align_y(Alignment::Center),
        )
        .width(Length::Fill)
        .padding([theme::PAD_4, theme::PAD_8])
        .style(|_| {
            theme::container_style(iced::Color::TRANSPARENT, 4.0, 1.0, theme::STATUS_WARNING)
        })
        .into()
    }

    /// Render the bottom bar: pagination controls, the view switch, the active
    /// view's clear button and the search input, all bound to the active view.
    fn bottom_bar(&self) -> Element<'_, LogMessage> {
        let data = self.view_data(self.active_view);
        let (page, total_pages) = (data.pagination.page, data.pagination.total_pages());

        // Pagination cluster — hidden entirely when there are no pages so the
        // "Page X of Y" indicator never shows an invalid page.
        let pagination_cluster: Element<'_, LogMessage> = if total_pages == 0 {
            Space::new().width(0).into()
        } else {
            let prev_button = button(text("← Prev").size(theme::TEXT_12))
                .style(theme::button_text)
                .on_press_maybe(if page > 0 {
                    Some(LogMessage::PrevPage)
                } else {
                    None
                });
            let next_button = button(text("Next →").size(theme::TEXT_12))
                .style(theme::button_text)
                .on_press_maybe(if page + 1 < total_pages {
                    Some(LogMessage::NextPage)
                } else {
                    None
                });
            row![
                prev_button,
                Space::new().width(theme::SPACE_8),
                text(format!("Page {} of {}", page + 1, total_pages))
                    .size(theme::TEXT_12)
                    .color(theme::TEXT_SECONDARY),
                Space::new().width(theme::SPACE_8),
                next_button,
            ]
            .align_y(Alignment::Center)
            .into()
        };

        // This view's clear — always rendered: a view refilling from live
        // activity right after a clear must stay clearable, so this is not tied
        // to the paging state.
        let clear_button = button(text(self.active_view.clear_label()).size(theme::TEXT_12))
            .style(theme::button_text_danger)
            .on_press(LogMessage::RequestClear);

        let search_input: Element<'_, LogMessage> = if self.focus_search {
            container(super::widgets::single_line_editor(
                &self.search_buffer.buffer,
                "search",
                false,
                Length::Fill,
                Some(iced::widget::Id::new("logs_search")),
                LogMessage::SearchInput,
            ))
            .padding(theme::PAD_2)
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
                .size(theme::TEXT_12)
                .color(theme::TEXT_MUTED),
            Space::new().width(theme::SPACE_4),
            search_input,
        ]
        .align_y(Alignment::Center);

        let bottom_row = row![
            pagination_cluster,
            Space::new().width(theme::SPACE_8),
            Self::view_switch(self.active_view),
            Space::new().width(theme::SPACE_8),
            clear_button,
            Space::new().width(Length::Fill),
            search_group,
        ]
        .align_y(Alignment::Center)
        .width(Length::Fill);

        container(bottom_row)
            .width(Length::Fill)
            .padding([theme::PAD_8, theme::PAD_24])
            .style(theme::surface_container_style)
            .into()
    }

    /// Render the active view's content (entries list). Pagination lives in the
    /// shared bottom bar.
    fn logs_view(&self) -> Element<'_, LogMessage> {
        let data = self.view_data(self.active_view);
        // New live entries insert at position 0, so no scroll-to-newest logic
        // is needed. The fade and the newest-entry timestamp are read per row,
        // i.e. only when rows exist — never on the loading or empty paths.
        widgets::paginated_body(
            data,
            lucide::activity::<iced::Theme, iced::Renderer>(),
            "No log entries",
            |entry| {
                let is_newest =
                    self.newest_entry_timestamp.as_deref() == Some(entry.timestamp.as_str());
                let fade_progress = if is_newest {
                    *self.fade_anim.value()
                } else {
                    1.0
                };
                // Per-row context menu: right-click copies the full entry in
                // formatted form. Wrapping only the row — not the column —
                // keeps the inter-row spacing a fall-through (same convention
                // as the per-bubble menus on the Home page). Left-click drag
                // selection on the row is unaffected: ContextMenu forwards all
                // non-right-click events to the underlay.
                ContextMenu::new(
                    LogsState::render_log_entry(entry, fade_progress),
                    vec![MenuItem::new(
                        "Copy".into(),
                        LogMessage::CopyEntry(entry.clone()),
                    )],
                )
                .into()
            },
        )
    }

    fn render_log_entry(entry: &LogEntry, fade_progress: f32) -> Element<'_, LogMessage> {
        let (level_color, level_bg) = theme::log_level_color(&entry.level);

        // One selectable Rich widget per entry so a whole log line can be
        // copied in a single drag. Pills are emulated with span highlights.
        let mut spans: Vec<Span<'_, (), iced::Font>> = vec![
            Span::new(theme::format_hhmmss(&entry.timestamp))
                .size(theme::TEXT_10)
                .color(theme::TEXT_MUTED),
            Span::new("  ").size(theme::TEXT_10),
            Span::new(&entry.level)
                .size(theme::TEXT_10)
                .color(level_color)
                .background(level_bg)
                .border(iced::border::rounded(4))
                .padding([theme::PAD_1, theme::PAD_6]),
            Span::new("  ").size(theme::TEXT_10),
        ];
        if !entry.agent_role.is_empty() {
            let (role_color, role_bg) = theme::role_badge_color(&entry.agent_role);
            spans.push(
                Span::new(&entry.agent_role)
                    .size(theme::TEXT_10)
                    .color(role_color)
                    .background(role_bg)
                    .border(iced::border::rounded(4))
                    .padding([theme::PAD_1, theme::PAD_6]),
            );
            spans.push(Span::new("  ").size(theme::TEXT_10));
        }
        spans.push(
            Span::new(&entry.target)
                .size(theme::TEXT_11)
                .color(theme::TEXT_SECONDARY),
        );
        // The row's remaining scope — the ticket it works on, then the
        // workspace — shares one muted style under the module.
        for scope in [&entry.ticket, &entry.workspace] {
            if scope.is_empty() {
                continue;
            }
            spans.push(Span::new("  ").size(theme::TEXT_10));
            spans.push(
                Span::new(scope)
                    .size(theme::TEXT_10)
                    .color(theme::TEXT_MUTED),
            );
        }

        spans.push(Span::new("\n").size(theme::TEXT_10));
        spans.push(
            Span::new(&entry.message)
                .size(theme::TEXT_13)
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
                spans.push(Span::new("\n").size(theme::TEXT_10));
                spans.push(
                    Span::new(format!("{key}: {val_str}"))
                        .size(theme::TEXT_10)
                        .color(theme::TEXT_SECONDARY),
                );
            }
        }

        let rich: super::selectable::Selectable<'_, (), LogMessage, iced::Theme, iced::Renderer> =
            super::selectable::rich_text(spans)
                .style(|_t| super::selectable::Style {
                    color: None,
                    selection: theme::ACCENT_DIM,
                })
                .width(Length::Fill);

        if fade_progress < 1.0 {
            // Fade-in: interpolate background/border alpha from 0.6 → 1.0
            container(rich)
                .padding(theme::PAD_6)
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
                .padding(theme::PAD_6)
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
/// the available role/agent/target/ticket/workspace parts, then the complete
/// untruncated message, then each remaining field as a `key: value` line.
///
/// Concrete layout (only the parts present in the entry are emitted; the
/// header line is omitted entirely when no role/agent/target/ticket/workspace
/// part exists):
///
/// ```text
/// 2026-08-17T20:15:44.581360Z  ERROR
/// role: engineer  agent: ticket_42_engineer  target: run_agent  ticket: mahbot-42  workspace: mahbot
///
/// The full untruncated message, possibly spanning multiple lines.
/// error_chain: crate::foo at src/foo.rs:42
/// attempt: 3
/// ```
///
/// Field lines iterate the stored fields JSON in **sorted key order** (the
/// format function sorts explicitly, so the output is deterministic regardless
/// of serde_json's `preserve_order` feature state) and skip only the `message`
/// key, mirroring the row renderer. The row's scope is promoted out of the
/// record at write time, so the keys that duplicate header parts — `agent_id`,
/// `role`/`agent_role`, `workspace`/`workspace_name`/`workspace_path`,
/// `ticket`/`ticket_id` — are no longer fields: the fields section is only the
/// record's remaining payload. Non-string values render via
/// [`field_value_str`] — numbers/bools as literals, nested JSON compact.
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
    if !entry.ticket.is_empty() {
        header.push(format!("ticket: {}", entry.ticket));
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
            }),
            agent_id: "ticket_42_engineer".to_string(),
            agent_role: "engineer".to_string(),
            workspace: "mahbot".to_string(),
            ticket: "mahbot-42".to_string(),
        }
    }

    #[test]
    fn format_log_entry_full_layout() {
        let out = format_log_entry(&entry());
        assert_eq!(
            out,
            "2026-08-17T20:15:44.581360Z  ERROR\n\
             role: engineer  agent: ticket_42_engineer  target: run_agent  ticket: mahbot-42  workspace: mahbot\n\
             \n\
             Agent failed\n\
             attempt: 3\n\
             error_chain: crate::foo at src/foo.rs:42"
        );
    }

    #[test]
    fn format_log_entry_minimal_omits_header_and_empty_fields() {
        let entry = LogEntry {
            timestamp: "2026-08-17T20:15:44.581360Z".to_string(),
            level: "INFO".to_string(),
            target: String::new(),
            message: "plain message".to_string(),
            ..Default::default()
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
