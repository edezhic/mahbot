//! Sessions dashboard page — view and manage conversation sessions.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::ChatMessage;
use crate::session::SessionMetadata;

use iced::widget::{
    Column, Id, Row, Space, button, column, container, markdown, mouse_area, responsive, row,
    scrollable, text,
};
use iced::{Alignment, Element, Length, Task};

use iced_anim::Animated;
use iced_anim::transition::Easing;

use iced::window;
use iced_fonts::lucide;

use super::ToastMessage;
use super::media_markers;
use super::menus::{ContextMenu, MenuItem};
use super::session_view::preview::{
    MAX_PREVIEW_LINES, MessageMeasure, measure_message, re_measure, width_bucket,
};
use super::session_view::{self, SessionEntry, ToolCallEntry};
use super::theme;
use super::widgets;
use super::widgets::selectable_text;

/// Fraction of the transcript row width a chat bubble occupies (FillPortion 3
/// of a 3:1 row). The collapse measurement uses this to narrow the measured
/// width to the actual bubble body (container padding + a safety margin).
const BUBBLE_BODY_RATIO: f32 = 0.75;

/// Shared transcript render context: the flat session ledger, its per-entry
/// markdown bodies, the element expansion set, the collapse-measurement
/// cache, and the single text measurement width (bubble body width, used for
/// all collapsible-element measurement).
struct TranscriptCtx<'a> {
    entries: &'a [SessionEntry],
    entry_md: &'a [Option<Vec<markdown::Item>>],
    expanded: &'a HashSet<(usize, usize)>,
    measure_cache: &'a RefCell<HashMap<(usize, usize), MessageMeasure>>,
    text_width: f32,
}

#[derive(Debug, Clone)]
pub(crate) enum SessionsMessage {
    Refreshed(Vec<SessionMetadata>),
    RefreshError(String),
    SelectSession(String),
    SessionMessages(String, Vec<ChatMessage>),
    SessionError(String),
    /// Toggle an element's collapsed preview / full content, keyed by
    /// (entry index, element index within entry).
    ToggleExpand(usize, usize),
    AnimTick(Instant),

    /// Auto-refresh the currently selected session's transcript.
    AutoRefreshMessages,
    /// Result of an auto-refresh message load.
    AutoRefreshResult(String, Vec<ChatMessage>),
    /// Scroll position changed in the transcript viewport.
    ScrollChanged(scrollable::Viewport),

    /// Dismiss modals/panels (Escape key).
    Escape,

    /// A link was clicked in rendered markdown.
    LinkClicked(String),

    /// A toast notification to surface from the dashboard.
    Toast(ToastMessage),
    /// Delete a session (context-menu action).
    DeleteSession(String),
    /// A session delete finished — remove it from the list. The bool is the
    /// store-reported row deletion (`true` when a real removal happened).
    SessionDeleted(String, bool),
}

#[derive(Debug, Clone)]
struct CachedSessionItem {
    key: String,
    /// Rendered key text for the session label.
    label: String,
    /// Pre-formatted message count string.
    msg_count_label: String,
    /// Pre-formatted compact token count (same format as the Running Agents
    /// card), when the session ever recorded a provider-reported length.
    token_label: Option<String>,
    /// Pre-formatted timestamp string.
    timestamp_label: String,
}

pub(crate) struct SessionsState {
    sessions: Vec<SessionMetadata>,
    pub(crate) load_state: super::common::AsyncLoadState,
    selected_session: Option<String>,
    /// The flat session ledger built from the selected session's messages.
    entries: Vec<SessionEntry>,
    /// Per-entry markdown bodies, parallel to `entries` (parsed from the
    /// ledger via `session_view::parse_entry_bodies`).
    entry_md: Vec<Option<Vec<markdown::Item>>>,
    selected_loading: bool,
    /// Elements the user expanded beyond their collapsed preview, keyed by
    /// (entry index, element index within entry): 0 = body/narration,
    /// 1 = thinking block, 2+j = result block of call `j`. Cleared on
    /// session switch.
    expanded: HashSet<(usize, usize)>,
    /// Per-element collapse measurement cache, keyed like `expanded`.
    /// Survives the 1-second auto-refresh — session entries are append-only,
    /// so an existing entry stays valid and only new elements are measured
    /// lazily. Mutated during layout (the bubble width is only known there),
    /// hence `RefCell`.
    measure_cache: RefCell<HashMap<(usize, usize), MessageMeasure>>,
    /// Animated transition for selected row background.
    selected_anim: Animated<f32>,
    /// Cached session list display data. Rebuilt only when `sessions` changes.
    /// `view()` builds widgets from this data on every frame; `selected_progress`
    /// animation is applied at widget-construction time outside the cache.
    cached_session_items: Option<Vec<CachedSessionItem>>,

    // ── Auto-refresh fields ──────────────────────────────────────
    /// Stable scrollable ID for the transcript area, preserves scroll position
    /// across widget rebuilds.
    scrollable_id: Id,
    /// Whether auto-scroll-to-bottom is enabled (user is at or near the bottom).
    auto_scroll_enabled: bool,
    /// Whether the Sessions page is currently visible (controls subscription).
    page_active: bool,
    /// Guard to prevent overlapping auto-refresh tasks.
    messages_refreshing: bool,
}

impl SessionsState {
    pub(crate) fn new() -> Self {
        Self {
            sessions: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            selected_session: None,
            entries: Vec::new(),
            entry_md: Vec::new(),
            selected_loading: false,
            expanded: HashSet::new(),
            measure_cache: RefCell::new(HashMap::new()),
            selected_anim: Animated::transition(
                0.0f32,
                Easing::EASE_OUT.with_duration(Duration::from_millis(theme::ANIM_SELECTED_MS)),
            ),
            cached_session_items: None,
            scrollable_id: Id::new("session_transcript_scroll"),
            auto_scroll_enabled: false,
            page_active: false,
            messages_refreshing: false,
        }
    }

    pub(crate) fn subscription(&self) -> iced::Subscription<SessionsMessage> {
        // Emit a 1-second timer for auto-refresh when the page is active and
        // a session is selected. Frame ticks are subscribed only while the
        // selection fade is animating — an always-on frames() subscription
        // closes a self-sustaining redraw loop (redraw → AnimTick → redraw).
        let mut subs: Vec<iced::Subscription<SessionsMessage>> = Vec::new();
        if self.page_active && self.selected_session.is_some() {
            subs.push(
                iced::time::every(Duration::from_secs(1))
                    .map(|_| SessionsMessage::AutoRefreshMessages),
            );
        }
        if self.selected_anim.is_animating() {
            subs.push(window::frames().map(SessionsMessage::AnimTick));
        }
        iced::Subscription::batch(subs)
    }

    /// Notify the sessions state whether the Sessions page is currently visible.
    /// This controls the auto-refresh subscription — when the page is hidden,
    /// polling stops.
    pub(crate) fn set_page_active(&mut self, active: bool) {
        self.page_active = active;
    }

    pub(crate) fn refresh() -> Task<SessionsMessage> {
        Task::perform(
            async {
                let store = crate::session::store();
                let list = store.list_sessions_with_metadata().await;
                Ok::<_, String>(list)
            },
            |res| match res {
                Ok(sessions) => SessionsMessage::Refreshed(sessions),
                Err(e) => SessionsMessage::RefreshError(e),
            },
        )
    }

    #[expect(clippy::too_many_lines)]
    pub(crate) fn update(&mut self, msg: SessionsMessage) -> Task<SessionsMessage> {
        match msg {
            SessionsMessage::AnimTick(instant) => {
                self.selected_anim.tick(instant);
                Task::none()
            }
            SessionsMessage::Refreshed(sessions) => {
                self.sessions = sessions;
                self.rebuild_session_cache();
                self.load_state.finish_loading();
                Task::none()
            }
            SessionsMessage::RefreshError(e) => {
                self.load_state.fail(e);
                Task::none()
            }
            SessionsMessage::SelectSession(key) => {
                self.selected_session = Some(key.clone());
                // Trigger selected animation
                self.selected_anim.set_target(1.0_f32);
                self.selected_loading = true;
                self.expanded.clear();
                // The measurement cache is cleared in SessionMessages when the
                // new session's messages actually arrive. The loading frames in
                // between show "Loading..." (old messages are not rendered),
                // but the cache must not be cleared here: on SessionError the
                // fallback keeps rendering the previous session's messages,
                // whose cache entries are still valid for that path. `expanded`
                // is cleared eagerly because the design resets collapse state
                // on session switch; on the error path that simply renders the
                // previous transcript collapsed (cosmetic, error-path only).
                // Do NOT set auto_scroll_enabled here — let ScrollChanged
                // determine it from the user's scroll behavior. The initial
                // snap to bottom happens eagerly in SessionMessages instead
                // of being delayed to the next auto-refresh tick.
                Task::perform(
                    async move {
                        let store = crate::session::store();
                        let messages = store.load(&key).await;
                        Ok::<_, String>((key, messages))
                    },
                    |res| match res {
                        Ok((key, messages)) => SessionsMessage::SessionMessages(key, messages),
                        Err(e) => SessionsMessage::SessionError(e),
                    },
                )
            }
            SessionsMessage::SessionMessages(key, messages) => {
                if self.selected_session.as_deref() == Some(&key) {
                    self.entries = session_view::build_ledger(&messages);
                    self.entry_md = session_view::parse_entry_bodies(&self.entries);
                    // Fresh session load: reset the measurement cache (the
                    // previous session's keys are index-stale).
                    self.measure_cache.borrow_mut().clear();
                    self.selected_loading = false;
                    // Snap to bottom so the user sees the most recent messages
                    // immediately, rather than waiting for the next auto-refresh
                    // tick (which would cause a delayed jump).
                    return iced::widget::operation::snap_to_end(self.scrollable_id.clone());
                }
                Task::none()
            }
            SessionsMessage::SessionError(e) => {
                self.load_state.fail(e);
                self.selected_loading = false;
                self.messages_refreshing = false;
                Task::none()
            }
            SessionsMessage::AutoRefreshMessages => {
                // Guard: skip if a refresh is already in-flight or no session selected.
                if self.messages_refreshing {
                    return Task::none();
                }
                let Some(key) = self.selected_session.clone() else {
                    return Task::none();
                };
                self.messages_refreshing = true;
                Task::perform(
                    async move {
                        let store = crate::session::store();
                        let messages = store.load(&key).await;
                        Ok::<_, String>((key, messages))
                    },
                    |res| match res {
                        Ok((key, messages)) => SessionsMessage::AutoRefreshResult(key, messages),
                        Err(e) => SessionsMessage::SessionError(e),
                    },
                )
            }
            SessionsMessage::AutoRefreshResult(key, messages) => {
                // Stale guard: ignore results for a different (deselected/overwritten) session.
                if self.selected_session.as_deref() != Some(&key) {
                    self.messages_refreshing = false;
                    return Task::none();
                }
                // Rebuild the ledger from the (append-only) message list.
                self.entries = session_view::build_ledger(&messages);
                self.entry_md = session_view::parse_entry_bodies(&self.entries);
                // Incremental cache: session entries are append-only, so keep
                // existing measurements and only add entries for new elements
                // lazily.
                self.messages_refreshing = false;

                // Auto-scroll to bottom when the user is already at the bottom.
                if self.auto_scroll_enabled {
                    iced::widget::operation::snap_to_end(self.scrollable_id.clone())
                } else {
                    Task::none()
                }
            }
            SessionsMessage::ScrollChanged(viewport) => {
                let bounds = viewport.bounds();
                let content = viewport.content_bounds();
                let at_bottom = if content.height > bounds.height {
                    viewport.relative_offset().y >= 0.99
                } else {
                    content.height <= bounds.height
                };
                self.auto_scroll_enabled = at_bottom;
                Task::none()
            }
            SessionsMessage::ToggleExpand(i, j) => {
                let key = (i, j);
                if self.expanded.contains(&key) {
                    self.expanded.remove(&key);
                } else {
                    self.expanded.insert(key);
                }
                Task::none()
            }
            SessionsMessage::DeleteSession(key) => Task::perform(
                async move {
                    let store = crate::session::store();
                    let deleted = store.delete(&key).await.map_err(|e| e.to_string())?;
                    Ok::<_, String>((key, deleted))
                },
                |res| match res {
                    Ok((key, deleted)) => SessionsMessage::SessionDeleted(key, deleted),
                    Err(e) => SessionsMessage::Toast(ToastMessage::Error(e)),
                },
            ),
            SessionsMessage::SessionDeleted(key, deleted) => {
                self.sessions.retain(|s| s.agent_id != key);
                self.rebuild_session_cache();
                if self.selected_session.as_deref() == Some(&key) {
                    self.clear_selection();
                }
                // Only claim a "Deleted" success when a real row removal
                // happened; an already-absent session (cleaned up elsewhere)
                // just vanishes from the list without an inaccurate toast.
                if deleted {
                    Task::done(SessionsMessage::Toast(ToastMessage::Deleted))
                } else {
                    Task::none()
                }
            }
            SessionsMessage::Toast(_) | SessionsMessage::LinkClicked(_) => Task::none(),
            SessionsMessage::Escape => {
                self.clear_selection();
                Task::none()
            }
        }
    }

    /// Clear the currently selected session and ALL per-session transcript
    /// state, returning the transcript column to its placeholder.
    fn clear_selection(&mut self) {
        self.selected_session = None;
        self.entries.clear();
        self.entry_md.clear();
        self.selected_loading = false;
        self.expanded.clear();
        self.messages_refreshing = false;
        self.auto_scroll_enabled = false;
        self.measure_cache.borrow_mut().clear();
        self.selected_anim.set_target(0.0);
    }

    /// Rebuild the cached session list display data. Called when `self.sessions`
    /// changes. `view()` builds widgets from this data on every frame, applying
    /// the `selected_progress` animation at widget-construction time.
    fn rebuild_session_cache(&mut self) {
        let items: Vec<CachedSessionItem> = self
            .sessions
            .iter()
            .map(|s| CachedSessionItem {
                key: s.agent_id.clone(),
                label: s.agent_id.clone(),
                msg_count_label: format!("{} msgs", s.message_count),
                token_label: s.token_length.map(theme::format_compact_tokens),
                timestamp_label: theme::format_timestamp(&s.last_activity.to_rfc3339()),
            })
            .collect();
        self.cached_session_items = if items.is_empty() { None } else { Some(items) };
    }

    #[expect(clippy::too_many_lines)]
    pub(crate) fn view(&self) -> Element<'_, SessionsMessage> {
        let mut content = column![];

        content = widgets::push_error_banner(content, self.load_state.error());

        if self.load_state.loading() && !self.load_state.has_loaded() {
            content = content.push(widgets::loading_text());
        } else if self.sessions.is_empty() {
            content = content.push(widgets::empty_state_placeholder(
                lucide::layout_dashboard::<iced::Theme, iced::Renderer>(),
                "No sessions",
            ));
        } else {
            // Session list on the left side — built from cached display data.
            // The cache is rebuilt only when `self.sessions` changes (in
            // `Refreshed`). The `selected_progress` animation is applied at
            // widget-construction time every frame.
            let mut session_list = Column::new().spacing(4);
            let selected_progress = *self.selected_anim.value();
            if let Some(ref cached) = self.cached_session_items {
                for item in cached {
                    let is_selected = self.selected_session.as_deref() == Some(&item.key);

                    let sess_row: Element<'_, SessionsMessage> = ContextMenu::new(
                        container(
                            column![
                                row![
                                    button(
                                        container(
                                            column![
                                                text(&item.label)
                                                    .size(13)
                                                    .color(theme::TEXT_PRIMARY),
                                                {
                                                    // Meta row: message count, then the
                                                    // token length when one was ever
                                                    // recorded (older sessions show no
                                                    // token value), then the timestamp.
                                                    // The 8px `Space` separators (with
                                                    // the row's 4px spacing) preserve
                                                    // the original msg-count → timestamp
                                                    // gap exactly.
                                                    let mut meta_row = row![
                                                        text(&item.msg_count_label)
                                                            .size(11)
                                                            .color(theme::TEXT_MUTED)
                                                    ]
                                                    .spacing(4);
                                                    if let Some(token) = &item.token_label {
                                                        meta_row = meta_row
                                                            .push(Space::new().width(8))
                                                            .push(
                                                                text(token)
                                                                    .size(11)
                                                                    .color(theme::TEXT_MUTED),
                                                            );
                                                    }
                                                    meta_row.push(Space::new().width(8)).push(
                                                        text(&item.timestamp_label)
                                                            .size(11)
                                                            .color(theme::TEXT_MUTED),
                                                    )
                                                },
                                            ]
                                            .spacing(2),
                                        )
                                        .padding(6)
                                        .width(Length::Fill)
                                        .style(
                                            move |_theme: &iced::Theme| container::Style {
                                                background: {
                                                    let t = if is_selected {
                                                        selected_progress
                                                    } else {
                                                        0.0f32
                                                    };
                                                    if t > 0.01 {
                                                        Some(iced::Background::Color(
                                                            iced::Color::from_rgba(
                                                                theme::ACCENT_DIM.r,
                                                                theme::ACCENT_DIM.g,
                                                                theme::ACCENT_DIM.b,
                                                                theme::ACCENT_DIM.a * t,
                                                            ),
                                                        ))
                                                    } else {
                                                        None
                                                    }
                                                },
                                                ..container::Style::default()
                                            }
                                        ),
                                    )
                                    .style(theme::button_text)
                                    .on_press(SessionsMessage::SelectSession(item.key.clone())),
                                ]
                                .align_y(Alignment::Center),
                            ]
                            .spacing(2),
                        )
                        .style(theme::surface_card_style),
                        vec![MenuItem::with_icon(
                            iced_fonts::lucide::advanced_text::trash,
                            "Delete".into(),
                            SessionsMessage::DeleteSession(item.key.clone()),
                        )],
                    )
                    .into();

                    session_list = session_list.push(sess_row);
                }
            }

            let session_scroll = scrollable(session_list)
                .width(Length::Fixed(350.0))
                .direction(theme::vertical_scrollbar())
                .style(theme::scrollbar_style);

            // Transcript on the right side. Wrapped in `responsive` so the
            // collapse measurement uses the real bubble body width, correct on
            // first render and after every window resize.
            let transcript: iced::Element<'_, SessionsMessage> = if self.selected_loading {
                iced::widget::container(
                    iced::widget::text("Loading messages...")
                        .size(13)
                        .color(theme::TEXT_MUTED),
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(16)
                .into()
            } else if let Some(ref _key) = self.selected_session {
                let entries = &self.entries;
                let entry_md = &self.entry_md;
                let expanded = &self.expanded;
                let measure_cache = &self.measure_cache;
                let scrollable_id = self.scrollable_id.clone();
                responsive(move |size| {
                    // The transcript container adds 8px padding per side. Every
                    // transcript round (message or tool round) renders as a chat
                    // bubble; all content measures at the bubble body width = 3/4
                    // of the transcript row minus the bubble's 10px padding per
                    // side, folded in with a +4px safety margin that errs toward
                    // collapse.
                    let text_width = ((size.width - 16.0) * BUBBLE_BODY_RATIO - 24.0).max(0.0);
                    let ctx = TranscriptCtx {
                        entries,
                        entry_md,
                        expanded,
                        measure_cache,
                        text_width,
                    };
                    container(render_transcript(&ctx, &scrollable_id))
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .padding(8)
                        .into()
                })
                .into()
            } else {
                container(
                    text("Select a session to view transcript.")
                        .size(13)
                        .color(theme::TEXT_MUTED),
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(16)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into()
            };

            content = content.push(
                row![session_scroll, Space::new().width(12), transcript].height(Length::Fill),
            );
        }

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(24)
            .style(theme::base_container_style)
            .into()
    }
}

/// Wrap a chat bubble in a 3:1 FillPortion row so it occupies 75% width,
/// aligned to the right for assistant messages or to the left for
/// system/user/tool messages. The caller must set `.width(FillPortion(3))`
/// on the bubble before passing it — this function only creates the spacer row.
fn align_bubble<'a>(
    bubble: impl Into<Element<'a, SessionsMessage>>,
    assistant: bool,
) -> Element<'a, SessionsMessage> {
    let bubble = bubble.into();
    if assistant {
        // Assistant: spacer left, bubble right.
        row![Space::new().width(Length::FillPortion(1)), bubble].into()
    } else {
        // System/user/tool: bubble left, spacer right.
        row![bubble, Space::new().width(Length::FillPortion(1))].into()
    }
}

/// Author label above a bubble, aligned to the bubble's side. One non-accent
/// color for every role, so roles stay visually consistent.
fn author_label<'a>(role: crate::ChatRole, assistant: bool) -> Element<'a, SessionsMessage> {
    let name = match role {
        crate::ChatRole::System => "System",
        crate::ChatRole::User => "User",
        crate::ChatRole::Assistant => "Assistant",
        crate::ChatRole::Tool => "Tool",
    };
    let label = text(name).size(11).color(theme::TEXT_SECONDARY);
    if assistant {
        row![Space::new().width(Length::FillPortion(1)), label].into()
    } else {
        row![label, Space::new().width(Length::FillPortion(1))].into()
    }
}

/// Author-labeled chat-bubble row for one ledger entry: the label sits above
/// the bubble, both aligned to the bubble's side (assistant right, others
/// left). Same background/typography as the Home chat bubbles.
fn bubble_row(
    role: crate::ChatRole,
    assistant: bool,
    body: Column<'_, SessionsMessage>,
) -> Element<'_, SessionsMessage> {
    let bubble = container(body)
        .padding(10)
        .style(theme::bubble_style(
            if assistant {
                theme::BG_ELEVATED
            } else {
                theme::BG_SURFACE
            },
            Some(theme::TEXT_PRIMARY),
        ))
        .width(Length::FillPortion(3));
    column![
        author_label(role, assistant),
        align_bubble(bubble, assistant)
    ]
    .spacing(2)
    .into()
}

/// Bottom-positioned chevron toggle for a collapsed/expanded long element.
/// Does not count toward the line budget.
fn toggle_button<'a>(is_expanded: bool, key: (usize, usize)) -> Element<'a, SessionsMessage> {
    let (icon, label) = if is_expanded {
        (
            lucide::chevron_up::<iced::Theme, iced::Renderer>().size(12),
            " Show less",
        )
    } else {
        (
            lucide::chevron_down::<iced::Theme, iced::Renderer>().size(12),
            " Show more",
        )
    };
    button(
        row![
            icon.color(theme::TEXT_MUTED),
            text(label).size(10).color(theme::TEXT_MUTED),
        ]
        .spacing(2)
        .align_y(Alignment::Center),
    )
    .style(theme::button_text)
    .on_press(SessionsMessage::ToggleExpand(key.0, key.1))
    .into()
}

/// Resolve the collapse measurement for an element at the current
/// `text_width`, measuring on cache miss (cached per element and width; the
/// session ledger is append-only, so a cached entry stays valid across the
/// 1-second auto-refresh). Returns the wrapped-line count and — only when
/// `need_preview` is set (the element is currently collapsed) — its 3-line
/// plain-text preview, so expanded elements do not pay a per-frame preview
/// clone.
///
/// Stale-fingerprint semantics: a cache hit with the same width bucket and
/// `content_len` is reused; the same `content_len` at a different bucket is
/// re-measured via [`re_measure`]; otherwise a fresh measurement runs.
fn element_measurement(
    measure_cache: &RefCell<HashMap<(usize, usize), MessageMeasure>>,
    key: (usize, usize),
    text_width: f32,
    need_preview: bool,
    content: &str,
) -> (u32, Option<String>) {
    let bucket = width_bucket(text_width);
    let content_len = content.len();
    let mut cache = measure_cache.borrow_mut();
    let cache_hit = cache.get(&key);
    let stale = cache_hit.is_none_or(|m| m.width_bucket != bucket || m.content_len != content_len);
    if stale {
        // Cache miss: measure. A pure width change reuses the cached
        // processed display text (an `Arc` clone); otherwise measure from the
        // caller's plain-text content.
        let m = match cache_hit {
            Some(m) if m.width_bucket != bucket && m.content_len == content_len => {
                re_measure(m, text_width)
            }
            _ => measure_message(content, text_width, content_len),
        };
        cache.insert(key, m);
    }
    let m = cache.get(&key).expect("measurement resolved above");
    (
        m.wrapped_lines,
        if need_preview {
            m.preview.clone()
        } else {
            None
        },
    )
}

/// A plain-text collapsible element with the 3-line measured collapse rule:
/// when it renders more than [`MAX_PREVIEW_LINES`] wrapped lines and is not
/// expanded, a plain 3-line preview (wrapped in a click-to-expand button) is
/// shown with a bottom chevron toggle; otherwise the full selectable text
/// (with a leading icon when given) plus the toggle. `leading_icon` is used
/// for the result block (`arrow_down_to_line`); thinking keeps its own header
/// and passes `None`.
fn plain_collapsible<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    content: &'a str,
    color: iced::Color,
    leading_icon: Option<Element<'a, SessionsMessage>>,
) -> Element<'a, SessionsMessage> {
    let is_expanded = ctx.expanded.contains(&key);
    let (wrapped_lines, preview) = element_measurement(
        ctx.measure_cache,
        key,
        ctx.text_width,
        !is_expanded,
        content,
    );
    let collapses = wrapped_lines > MAX_PREVIEW_LINES;
    let mut col = Column::new().spacing(2).width(Length::Fill);

    if collapses && !is_expanded {
        let preview = preview.expect("a collapsing element always has a preview");
        let mut inner = Row::new().spacing(4).align_y(Alignment::Start);
        if let Some(icon) = leading_icon {
            inner = inner.push(icon);
        }
        inner = inner.push(selectable_text(preview, color).size(11));
        col = col.push(
            button(inner)
                .style(theme::button_text)
                .on_press(SessionsMessage::ToggleExpand(key.0, key.1))
                .width(Length::Fill),
        );
    } else {
        let mut inner = Row::new().spacing(4).align_y(Alignment::Start);
        if let Some(icon) = leading_icon {
            inner = inner.push(icon);
        }
        inner = inner.push(selectable_text(content, color).size(11));
        col = col.push(inner);
    }
    if collapses {
        col = col.push(toggle_button(is_expanded, key));
    }
    col.into()
}

/// The collapsible Thinking block: lucide `brain` + "Thinking" header above a
/// [`plain_collapsible`] body with the same 3-line collapse, measured at the
/// bubble body width.
fn thinking_block<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    content: &'a str,
) -> Element<'a, SessionsMessage> {
    let header = row![
        lucide::brain::<iced::Theme, iced::Renderer>()
            .size(11)
            .color(theme::TEXT_MUTED),
        text("Thinking").size(11).color(theme::TEXT_MUTED),
    ]
    .spacing(4)
    .align_y(Alignment::Center);
    let body = plain_collapsible(ctx, key, content, theme::TEXT_MUTED, None);
    column![header, body].spacing(4).into()
}

/// The message/narration body with the 3-line collapse rule: a collapsed
/// element shows a plain clickable preview; an expanded (or short) element
/// shows the markdown body parsed from `md`. `content` is the plain text used
/// for the collapse measurement (at the bubble body width); `md` is the parsed
/// markdown for the expanded view.
fn body_block<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    content: &'a str,
    md: Option<&'a [markdown::Item]>,
) -> Element<'a, SessionsMessage> {
    let is_expanded = ctx.expanded.contains(&key);
    let (wrapped_lines, preview) = element_measurement(
        ctx.measure_cache,
        key,
        ctx.text_width,
        !is_expanded,
        content,
    );
    let collapses = wrapped_lines > MAX_PREVIEW_LINES;
    let mut col = Column::new().spacing(2).width(Length::Fill);

    if collapses && !is_expanded {
        let preview = preview.expect("a collapsing body always has a preview");
        col = col.push(
            button(selectable_text(preview, theme::TEXT_PRIMARY).size(theme::MARKDOWN_TEXT_SIZE))
                .style(theme::button_text)
                .on_press(SessionsMessage::ToggleExpand(key.0, key.1))
                .width(Length::Fill),
        );
    } else if let Some(md) = md {
        let md_el: Element<'a, SessionsMessage> =
            media_markers::selectable_markdown_view(md, theme::markdown_settings())
                .map(SessionsMessage::LinkClicked);
        col = col.push(md_el);
    } else {
        col =
            col.push(selectable_text(content, theme::TEXT_PRIMARY).size(theme::MARKDOWN_TEXT_SIZE));
    }
    if collapses {
        col = col.push(toggle_button(is_expanded, key));
    }
    col.into()
}

/// The tool-call result block: lucide `arrow_down_to_line` prefix over the
/// result content with the 3-line collapse. The result content is plain
/// selectable text (tool output is not markdown). Results render inside the
/// assistant bubble, so they measure at the bubble body width.
fn result_block<'a>(
    ctx: &TranscriptCtx<'a>,
    key: (usize, usize),
    result: &'a str,
) -> Element<'a, SessionsMessage> {
    let icon: Element<'a, SessionsMessage> =
        lucide::arrow_down_to_line::<iced::Theme, iced::Renderer>()
            .size(11)
            .color(theme::TEXT_MUTED)
            .into();
    plain_collapsible(ctx, key, result, theme::TEXT_SECONDARY, Some(icon))
}

/// Render one session round as a single assistant bubble containing the
/// reasoning block, the narration body, and one tool block per call with its
/// collapsible result beneath.
fn render_tool_round<'a>(
    ctx: &TranscriptCtx<'a>,
    i: usize,
    narration: Option<&'a str>,
    reasoning: Option<&'a str>,
    calls: &'a [ToolCallEntry],
) -> Element<'a, SessionsMessage> {
    let md = ctx.entry_md.get(i).and_then(|m| m.as_deref());
    let mut bubble_col = Column::new().spacing(4);

    if let Some(reasoning) = reasoning {
        bubble_col = bubble_col.push(thinking_block(ctx, (i, 1), reasoning));
    }
    if let Some(narration) = narration {
        bubble_col = bubble_col.push(body_block(ctx, (i, 0), narration, md));
    }

    for (j, call) in calls.iter().enumerate() {
        let key = (i, 2 + j);
        // Only a result that actually collapses makes the tool block a click
        // target (toggling a short result's expansion would be a no-op).
        let collapses = call.result.as_deref().is_some_and(|result| {
            let (wrapped_lines, _) =
                element_measurement(ctx.measure_cache, key, ctx.text_width, false, result);
            wrapped_lines > MAX_PREVIEW_LINES
        });
        // The tool block is wrapped in a transparent mouse_area (not a
        // button) so the shared tool block's hover tooltip keeps working;
        // clicking anywhere toggles the associated result's collapse.
        let tool_block = container(session_view::tool_block(&call.tool, true, true))
            .width(Length::Fill)
            .padding([2, 4]);
        if collapses {
            let tool_clickable: Element<'a, SessionsMessage> = mouse_area(tool_block)
                .on_press(SessionsMessage::ToggleExpand(key.0, key.1))
                .into();
            bubble_col = bubble_col.push(tool_clickable);
        } else {
            bubble_col = bubble_col.push(tool_block);
        }

        match call.result.as_deref() {
            Some(result) if !result.is_empty() => {
                bubble_col = bubble_col.push(result_block(ctx, key, result));
            }
            // Empty or missing ToolResult record: the old view's explicit
            // "(no result)" indicator.
            _ => {
                bubble_col = bubble_col.push(
                    row![
                        lucide::arrow_down_to_line::<iced::Theme, iced::Renderer>()
                            .size(11)
                            .color(theme::TEXT_MUTED),
                        text("(no result)").size(10).color(theme::TEXT_MUTED),
                    ]
                    .spacing(4)
                    .align_y(Alignment::Center),
                );
            }
        }
    }

    // One bubble for the whole round, same container as assistant text rounds
    // (right-aligned, 75% width, author label above).
    bubble_row(crate::ChatRole::Assistant, true, bubble_col)
}

/// Render one ledger entry as a chat-bubble row: a `Message` renders its own
/// bubble, and a `ToolRound` renders a single assistant bubble containing the
/// round's reasoning, narration, and tool calls with their results.
fn render_entry<'a>(ctx: &TranscriptCtx<'a>, i: usize) -> Element<'a, SessionsMessage> {
    let md = ctx.entry_md.get(i).and_then(|m| m.as_deref());
    let entry = &ctx.entries[i];
    match entry {
        SessionEntry::Message {
            role,
            content,
            thinking,
        } => {
            let assistant = matches!(role, crate::ChatRole::Assistant);
            let mut bubble_col = Column::new().spacing(4);
            if let Some(thinking) = thinking {
                bubble_col = bubble_col.push(thinking_block(ctx, (i, 1), thinking));
            }
            if let Some(content) = content {
                bubble_col = bubble_col.push(body_block(ctx, (i, 0), content, md));
            }
            if thinking.is_some() || content.is_some() {
                bubble_row(*role, assistant, bubble_col)
            } else {
                // Nothing to show but the author.
                author_label(*role, assistant)
            }
        }
        SessionEntry::ToolRound {
            narration,
            reasoning,
            calls,
        } => render_tool_round(ctx, i, narration.as_deref(), reasoning.as_deref(), calls),
    }
}

/// Render the transcript column for the selected session: build the chat
/// bubbles from the shared ledger and apply the 3-line collapse rule to
/// message bodies, thinking blocks, and tool results.
fn render_transcript<'a>(
    ctx: &TranscriptCtx<'a>,
    scrollable_id: &Id,
) -> Element<'a, SessionsMessage> {
    if ctx.entries.is_empty() {
        return text("No messages in this session.")
            .size(13)
            .color(theme::TEXT_MUTED)
            .into();
    }
    let mut items = Column::new().spacing(6);
    for i in 0..ctx.entries.len() {
        items = items.push(render_entry(ctx, i));
    }
    scrollable(items)
        .id(scrollable_id.clone())
        .on_scroll(SessionsMessage::ScrollChanged)
        .height(Length::Fill)
        .direction(theme::vertical_scrollbar())
        .style(theme::scrollbar_style)
        .into()
}
