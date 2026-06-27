//! Sessions dashboard page — view and manage conversation sessions.

#![allow(
    clippy::too_many_lines,
    clippy::match_same_arms,
    clippy::manual_let_else
)]

use crate::ChatMessage;
use crate::session::{DecodedNativeHistoryMessage, SessionMetadata, decode_native_history_message};

use iced::widget::{Column, Id, Space, button, column, container, markdown, row, scrollable, text};
use iced::{Alignment, Element, Length, Task};

use iced_anim::Animated;
use iced_anim::transition::Easing;
use std::time::{Duration, Instant};

use iced::window;
use iced_fonts::lucide;

use super::theme;
use super::widgets;
use super::widgets::selectable_text;

#[derive(Debug, Clone)]
pub enum SessionsMessage {
    Refreshed(Vec<SessionMetadata>),
    RefreshError(String),
    SelectSession(String),
    SessionMessages(String, Vec<ChatMessage>),
    SessionError(String),
    ToggleToolRound(usize),
    ToggleThinkingBlock(usize),
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
}

#[derive(Debug, Clone)]
struct CachedSessionItem {
    key: String,
    /// Rendered key text for the session label.
    label: String,
    /// Pre-formatted message count string.
    msg_count_label: String,
    /// Pre-formatted timestamp string.
    timestamp_label: String,
}

pub struct SessionsState {
    sessions: Vec<SessionMetadata>,
    pub(crate) load_state: super::common::AsyncLoadState,
    selected_session: Option<String>,
    selected_messages: Vec<ChatMessage>,
    /// Cached parsed markdown items for each message, populated when messages are loaded.
    selected_md_items: Vec<Vec<markdown::Item>>,
    selected_loading: bool,
    expanded_tool_rounds: std::collections::HashSet<usize>,
    expanded_thinking_blocks: std::collections::HashSet<usize>,
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
    pub fn new() -> Self {
        Self {
            sessions: Vec::new(),
            load_state: super::common::AsyncLoadState::new(),
            selected_session: None,
            selected_messages: Vec::new(),
            selected_md_items: Vec::new(),
            selected_loading: false,
            expanded_tool_rounds: std::collections::HashSet::new(),
            expanded_thinking_blocks: std::collections::HashSet::new(),
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

    pub fn subscription(&self) -> iced::Subscription<SessionsMessage> {
        // Emit a 1-second timer for auto-refresh when the page is active and
        // a session is selected.
        if self.page_active && self.selected_session.is_some() {
            iced::Subscription::batch([
                window::frames().map(SessionsMessage::AnimTick),
                iced::time::every(Duration::from_secs(1))
                    .map(|_| SessionsMessage::AutoRefreshMessages),
            ])
        } else {
            window::frames().map(SessionsMessage::AnimTick)
        }
    }

    /// Notify the sessions state whether the Sessions page is currently visible.
    /// This controls the auto-refresh subscription — when the page is hidden,
    /// polling stops.
    pub fn set_page_active(&mut self, active: bool) {
        self.page_active = active;
    }

    pub fn refresh(&self) -> Task<SessionsMessage> {
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

    pub fn update(&mut self, msg: SessionsMessage) -> Task<SessionsMessage> {
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
                self.expanded_thinking_blocks.clear();
                self.expanded_tool_rounds.clear();
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
                    self.selected_md_items = parse_messages_to_md_items(&messages);
                    self.selected_messages = messages;
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
                let key = match self.selected_session.clone() {
                    Some(k) => k,
                    None => return Task::none(),
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
                // Parse markdown for each message (same as SessionMessages but
                // without touching selected_loading, preserving scrollable identity).
                self.selected_md_items = parse_messages_to_md_items(&messages);
                self.selected_messages = messages;
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
            SessionsMessage::ToggleToolRound(idx) => {
                if self.expanded_tool_rounds.contains(&idx) {
                    self.expanded_tool_rounds.remove(&idx);
                } else {
                    self.expanded_tool_rounds.insert(idx);
                }
                Task::none()
            }
            SessionsMessage::ToggleThinkingBlock(idx) => {
                if self.expanded_thinking_blocks.contains(&idx) {
                    self.expanded_thinking_blocks.remove(&idx);
                } else {
                    self.expanded_thinking_blocks.insert(idx);
                }
                Task::none()
            }
            SessionsMessage::Escape => {
                self.selected_session = None;
                self.selected_messages.clear();
                self.expanded_thinking_blocks.clear();
                self.expanded_tool_rounds.clear();
                Task::none()
            }
            SessionsMessage::LinkClicked(_) => Task::none(),
        }
    }

    /// Rebuild the cached session list display data. Called when `self.sessions`
    /// changes. `view()` builds widgets from this data on every frame, applying
    /// the `selected_progress` animation at widget-construction time.
    fn rebuild_session_cache(&mut self) {
        let items: Vec<CachedSessionItem> = self
            .sessions
            .iter()
            .map(|s| CachedSessionItem {
                key: s.key.clone(),
                label: s.key.clone(),
                msg_count_label: format!("{} msgs", s.message_count),
                timestamp_label: theme::format_timestamp(&s.last_activity.to_rfc3339()),
            })
            .collect();
        self.cached_session_items = if items.is_empty() { None } else { Some(items) };
    }

    fn render_transcript<'a>(
        messages: &'a [ChatMessage],
        md_items: &'a [Vec<markdown::Item>],
        expanded_rounds: &'a std::collections::HashSet<usize>,
        expanded_thinking: &'a std::collections::HashSet<usize>,
        scrollable_id: &Id,
    ) -> Element<'a, SessionsMessage> {
        // Inner types used in transcript rendering
        #[derive(Debug, Clone)]
        struct ToolCallInfo {
            id: String,
            name: String,
            arguments: String,
        }

        enum DecodedMsgKind {
            Regular {
                content_parts: ContentParts,
            },
            AssistantToolCalls {
                /// Individual tool calls with their IDs for matching with results.
                calls: Vec<ToolCallInfo>,
                /// Reasoning/thinking text extracted from the assistant message
                /// (already unwrapped, no `[thinking]` markup).
                reasoning_text: Option<String>,
                /// Text content from the assistant message that appeared
                /// alongside the tool calls (before or after them).
                text_content: Option<String>,
            },
            ToolResult {
                tool_call_id: Option<String>,
                content: String,
            },
        }

        struct DecodedMsg {
            role: String,
            role_color: iced::Color,
            kind: DecodedMsgKind,
        }

        enum ContentParts {
            Simple(String),
            WithThinking {
                thinking: String,
                after_thinking: String,
            },
        }

        // Used during tool-call↔result matching in the second pass.
        struct MatchedPair<'a> {
            call: &'a ToolCallInfo,
            result_content: Option<&'a str>,
        }

        /// Parse `[thinking]...[/thinking]` markup from a content string.
        /// Returns `ContentParts::WithThinking` if a complete thinking block
        /// is found, otherwise falls back to `ContentParts::Simple`.
        fn parse_thinking_blocks(content_str: String) -> ContentParts {
            if let Some(thinking_start) = content_str.find("[thinking]") {
                let body_start = thinking_start + "[thinking]".len();
                if let Some(end) = content_str.find("[/thinking]") {
                    let thinking = content_str[body_start..end].trim().to_string();
                    let after = content_str[end + "[/thinking]".len()..].trim().to_string();
                    return ContentParts::WithThinking {
                        thinking,
                        after_thinking: after,
                    };
                }
            }
            ContentParts::Simple(content_str)
        }

        if messages.is_empty() {
            return text("No messages in this session.")
                .size(13)
                .color(theme::TEXT_MUTED)
                .into();
        }

        // First pass: decode all messages
        let mut decoded_msgs: Vec<DecodedMsg> = Vec::new();
        for msg in messages {
            let decoded = decode_native_history_message(msg);
            let role_color = theme::role_badge_color(&msg.role).0;

            let kind = if let Some(ref d) = decoded {
                match d {
                    DecodedNativeHistoryMessage::AssistantToolCalls {
                        content,
                        tool_calls,
                        reasoning,
                    } => {
                        let reasoning_text = reasoning
                            .as_ref()
                            .and_then(|r| r.reasoning.as_deref())
                            .filter(|r| !r.is_empty())
                            .map(ToString::to_string);

                        let calls: Vec<ToolCallInfo> = tool_calls
                            .iter()
                            .map(|tc| ToolCallInfo {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                arguments: crate::util::summarize_args(&tc.arguments),
                            })
                            .collect();

                        let text_content: Option<String> = match content {
                            Some(c) if !c.is_empty() => Some(c.clone()),
                            _ => None,
                        };

                        DecodedMsgKind::AssistantToolCalls {
                            calls,
                            reasoning_text,
                            text_content,
                        }
                    }
                    DecodedNativeHistoryMessage::AssistantReasoning { content, reasoning } => {
                        let mut parts = Vec::new();
                        if let Some(r) = reasoning
                            .as_ref()
                            .and_then(|r| r.reasoning.as_deref())
                            .filter(|r| !r.is_empty())
                        {
                            parts.push(format!("[thinking]\n{r}\n[/thinking]"));
                        }
                        if let Some(c) = content
                            && !c.is_empty()
                        {
                            parts.push(c.clone());
                        }
                        let content_str = parts.join("\n");
                        let content_parts = parse_thinking_blocks(content_str);

                        DecodedMsgKind::Regular { content_parts }
                    }
                    DecodedNativeHistoryMessage::ToolResult {
                        tool_call_id,
                        content,
                    } => DecodedMsgKind::ToolResult {
                        tool_call_id: tool_call_id.clone(),
                        content: content.clone(),
                    },
                }
            } else {
                let content_str = msg.content.clone();
                let content_parts = parse_thinking_blocks(content_str);

                DecodedMsgKind::Regular { content_parts }
            };

            decoded_msgs.push(DecodedMsg {
                role: msg.role.clone(),
                role_color,
                kind,
            });
        }

        // Second pass: group into tool rounds with interleaved call/result pairs
        let len = decoded_msgs.len();
        let mut items = Column::new().spacing(6);
        let mut i = 0;
        let mut round_idx = 0;
        while i < len {
            let dm_role = decoded_msgs[i].role.clone();
            let dm_role_color = decoded_msgs[i].role_color;

            match &decoded_msgs[i].kind {
                DecodedMsgKind::AssistantToolCalls {
                    calls,
                    reasoning_text,
                    text_content,
                } => {
                    // Collect consecutive ToolResult messages after this tool call
                    let mut result_msgs: Vec<(usize, &str, Option<&str>)> = Vec::new();
                    // (msg_index, content, tool_call_id)

                    let mut j = i + 1;
                    while j < len {
                        if let DecodedMsgKind::ToolResult {
                            ref tool_call_id,
                            ref content,
                        } = decoded_msgs[j].kind
                        {
                            result_msgs.push((j, content.as_str(), tool_call_id.as_deref()));
                            j += 1;
                        } else {
                            break;
                        }
                    }

                    // --- Matching: pair calls with results by tool_call_id ---
                    let mut matched: Vec<MatchedPair<'_>> = Vec::new();
                    let mut used_results: std::collections::HashSet<usize> =
                        std::collections::HashSet::new();

                    for call in calls {
                        // Try to find a result with matching tool_call_id
                        let found = result_msgs.iter().position(|(idx, _content, cid)| {
                            cid == &Some(call.id.as_str()) && !used_results.contains(idx)
                        });

                        if let Some(pos) = found {
                            let msg_idx = result_msgs[pos].0;
                            used_results.insert(msg_idx);
                            matched.push(MatchedPair {
                                call,
                                result_content: Some(result_msgs[pos].1),
                            });
                        } else {
                            matched.push(MatchedPair {
                                call,
                                result_content: None,
                            });
                        }
                    }

                    // Unmatched results (not consumed by ID matching) —
                    // try positional fallback for None tool_call_id results
                    let unmatched_results: Vec<(usize, &str)> = result_msgs
                        .iter()
                        .filter(|(idx, _content, _cid)| !used_results.contains(idx))
                        .map(|(idx, content, _cid)| (*idx, *content))
                        .collect();

                    // Positional fallback: pair first unmatched result (with None ID)
                    // with first unmatched call (that had no result). Only applied
                    // when counts of None-ID results and unmatched calls align,
                    // so ordering is unambiguous.
                    let mut fallback_results: Vec<(usize, &str)> = Vec::new();
                    let mut unmatched_calls: Vec<&ToolCallInfo> = Vec::new();

                    for pair in &matched {
                        if pair.result_content.is_none() {
                            unmatched_calls.push(pair.call);
                        }
                    }

                    // Only use positional fallback for None-ID results
                    // when counts align exactly.
                    let none_id_results: Vec<(usize, &str)> = unmatched_results
                        .iter()
                        .filter(|(idx, _content)| {
                            if let DecodedMsgKind::ToolResult { tool_call_id, .. } =
                                &decoded_msgs[*idx].kind
                            {
                                tool_call_id.is_none()
                            } else {
                                false
                            }
                        })
                        .copied()
                        .collect();

                    if none_id_results.len() == unmatched_calls.len() && !none_id_results.is_empty()
                    {
                        // Positional match: pair first-to-first, second-to-second, etc.
                        let mut with_fallback: Vec<MatchedPair<'_>> = Vec::new();
                        let mut fallback_iter = none_id_results.iter();
                        for pair in &matched {
                            if pair.result_content.is_none()
                                && let Some((fb_idx, fb_content)) = fallback_iter.next()
                            {
                                fallback_results.push((*fb_idx, *fb_content));
                                with_fallback.push(MatchedPair {
                                    call: pair.call,
                                    result_content: Some(fb_content),
                                });
                            } else {
                                with_fallback.push(MatchedPair {
                                    call: pair.call,
                                    result_content: pair.result_content,
                                });
                            }
                        }
                        matched = with_fallback;
                    }

                    // Rebuild unmatched results excluding fallback ones
                    let fallback_idxs: std::collections::HashSet<usize> =
                        fallback_results.iter().map(|(idx, _)| *idx).collect();

                    let stray_unmatched_results: Vec<(usize, &str)> = unmatched_results
                        .into_iter()
                        .filter(|(idx, _)| !fallback_idxs.contains(idx))
                        .collect();

                    // Recompute unmatched calls (after fallback)
                    let final_unmatched_calls: Vec<&ToolCallInfo> = matched
                        .iter()
                        .filter(|p| p.result_content.is_none())
                        .map(|p| p.call)
                        .collect();

                    let is_expanded = expanded_rounds.contains(&round_idx);

                    // Compact names list
                    let compact_names: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
                    let compact_names_str = compact_names.join(", ");

                    let header = button(
                        container(
                            row![
                                text("🔧").size(11).color(theme::TEXT_MUTED),
                                Space::new().width(6),
                                text(compact_names_str).size(11).color(theme::TEXT_MUTED),
                                Space::new().width(Length::Fill),
                                text(if is_expanded { "▼" } else { "▶" })
                                    .size(9)
                                    .color(theme::TEXT_MUTED),
                            ]
                            .align_y(Alignment::Center),
                        )
                        .padding(8)
                        .width(Length::Fill),
                    )
                    .style(theme::button_text)
                    .on_press(SessionsMessage::ToggleToolRound(round_idx));

                    let mut contents: Vec<Element<'_, SessionsMessage>> = vec![header.into()];

                    if is_expanded {
                        let mut expanded_col = Column::new().spacing(3);

                        // Text content banner (if assistant had text alongside tool calls)
                        if let Some(tc) = text_content
                            && !tc.is_empty()
                        {
                            expanded_col = expanded_col.push(
                                container(text(tc.clone()).size(11).color(theme::TEXT_MUTED))
                                    .padding([2, 4]),
                            );
                        }

                        // Reasoning/thinking banner
                        if let Some(rt) = reasoning_text
                            && !rt.is_empty()
                        {
                            expanded_col = expanded_col.push(
                                container(
                                    column![
                                        text("🧠 Thinking").size(11).color(theme::TEXT_MUTED),
                                        text(rt.clone()).size(11).color(theme::TEXT_MUTED),
                                    ]
                                    .spacing(2),
                                )
                                .padding([4, 8])
                                .style(|_theme: &iced::Theme| container::Style {
                                    background: Some(iced::Background::Color(theme::BG_SURFACE)),
                                    border: iced::Border {
                                        radius: 3.0.into(),
                                        width: 1.0,
                                        color: theme::BORDER,
                                    },
                                    ..container::Style::default()
                                }),
                            );
                        }

                        // Matched pairs: call → result, interleaved
                        for pair in &matched {
                            // Call line
                            expanded_col = expanded_col.push(
                                container(
                                    selectable_text(
                                        format!("🔧 {}: {}", pair.call.name, pair.call.arguments),
                                        theme::TEXT_SECONDARY,
                                    )
                                    .size(11),
                                )
                                .padding([2, 4]),
                            );

                            // Result line (if matched)
                            if let Some(result) = pair.result_content {
                                if !result.is_empty() {
                                    expanded_col = expanded_col.push(
                                        container(
                                            selectable_text(
                                                format!("📋 {result}"),
                                                theme::TEXT_SECONDARY,
                                            )
                                            .size(11),
                                        )
                                        .padding([2, 4]),
                                    );
                                }
                            }
                        }

                        // Unmatched calls (no result)
                        for call in &final_unmatched_calls {
                            expanded_col = expanded_col.push(
                                container(row![
                                    selectable_text(
                                        format!("🔧 {}: {}", call.name, call.arguments),
                                        theme::TEXT_MUTED,
                                    )
                                    .size(11),
                                    Space::new().width(6),
                                    selectable_text("(no result)", theme::TEXT_MUTED).size(10),
                                ])
                                .padding([2, 4]),
                            );
                        }

                        // Unmatched results rendered inside the round card
                        for (_, content) in &stray_unmatched_results {
                            if !content.is_empty() {
                                expanded_col = expanded_col.push(
                                    container(
                                        selectable_text(
                                            format!("📋 {content}"),
                                            theme::TEXT_SECONDARY,
                                        )
                                        .size(11),
                                    )
                                    .padding([2, 4]),
                                );
                            }
                        }

                        contents.push(container(expanded_col).padding([4, 8]).into());
                    }

                    let round_card =
                        container(Column::with_children(contents).spacing(if is_expanded {
                            2
                        } else {
                            0
                        }))
                        .padding(8)
                        .style(|_theme: &iced::Theme| container::Style {
                            background: Some(iced::Background::Color(theme::BG_ELEVATED)),
                            border: iced::Border {
                                radius: 4.0.into(),
                                width: 1.0,
                                color: theme::BORDER,
                            },
                            ..container::Style::default()
                        });

                    items = items.push(round_card);
                    i = j;
                    round_idx += 1;
                }
                DecodedMsgKind::ToolResult {
                    tool_call_id: _,
                    content,
                } => {
                    // Stray tool result (no preceding tool call) — render as regular message
                    let mut msg_col = Column::new().spacing(2);
                    msg_col = msg_col.push(
                        container(text(dm_role.clone()).size(11).color(dm_role_color))
                            .padding([1, 6])
                            .style(move |t| theme::role_badge_pill_style(t, dm_role_color)),
                    );
                    if !content.is_empty() {
                        msg_col = msg_col.push({
                            let md: iced::Element<
                                '_,
                                SessionsMessage,
                                iced::Theme,
                                iced::Renderer,
                            > = markdown::view(&md_items[i], theme::markdown_settings())
                                .map(SessionsMessage::LinkClicked);
                            md
                        });
                    }
                    items = items.push(
                        container(msg_col)
                            .padding(8)
                            .style(theme::surface_card_style),
                    );
                    i += 1;
                }
                DecodedMsgKind::Regular { content_parts: cp } => {
                    // Regular message — extract owned strings from the content parts
                    let (thinking, after, simple) = match cp {
                        ContentParts::Simple(s) => (
                            None,
                            None,
                            if s.is_empty() { None } else { Some(s.clone()) },
                        ),
                        ContentParts::WithThinking {
                            thinking: t,
                            after_thinking: a,
                        } => {
                            let t_owned = t.clone();
                            let a_owned = if a.is_empty() { None } else { Some(a.clone()) };
                            (Some(t_owned), a_owned, None)
                        }
                    };

                    let mut msg_col = Column::new().spacing(2);
                    msg_col = msg_col.push(
                        container(text(dm_role).size(11).color(dm_role_color))
                            .padding([1, 6])
                            .style(move |t| theme::role_badge_pill_style(t, dm_role_color)),
                    );

                    if let Some(ref t) = thinking {
                        let is_thinking_expanded = expanded_thinking.contains(&i);

                        let thinking_header =
                            button(
                                container(
                                    row![
                                        text("🧠 Thinking").size(11).color(theme::TEXT_MUTED),
                                        Space::new().width(Length::Fill),
                                        text(if is_thinking_expanded { "▼" } else { "▶" })
                                            .size(9)
                                            .color(theme::TEXT_MUTED),
                                    ]
                                    .align_y(Alignment::Center),
                                )
                                .padding([4, 8])
                                .width(Length::Fill)
                                .style(|_theme: &iced::Theme| container::Style {
                                    background: Some(iced::Background::Color(theme::BG_SURFACE)),
                                    border: iced::Border {
                                        radius: 3.0.into(),
                                        width: 1.0,
                                        color: theme::BORDER,
                                    },
                                    ..container::Style::default()
                                }),
                            )
                            .style(theme::button_text)
                            .on_press(SessionsMessage::ToggleThinkingBlock(i));

                        msg_col = msg_col.push(thinking_header);

                        if is_thinking_expanded {
                            msg_col = msg_col.push(
                                container(text(t.clone()).size(11).color(theme::TEXT_MUTED))
                                    .padding([4, 8])
                                    .style(|_theme: &iced::Theme| container::Style {
                                        background: Some(iced::Background::Color(
                                            theme::BG_SURFACE,
                                        )),
                                        border: iced::Border {
                                            radius: 3.0.into(),
                                            width: 1.0,
                                            color: theme::BORDER,
                                        },
                                        ..container::Style::default()
                                    }),
                            );
                        }
                    }
                    if after.is_some() || simple.is_some() {
                        msg_col = msg_col.push({
                            let md: iced::Element<
                                '_,
                                SessionsMessage,
                                iced::Theme,
                                iced::Renderer,
                            > = markdown::view(&md_items[i], theme::markdown_settings())
                                .map(SessionsMessage::LinkClicked);
                            md
                        });
                    }

                    items = items.push(
                        container(msg_col)
                            .padding(8)
                            .style(theme::surface_card_style),
                    );

                    i += 1;
                }
            }
        }

        scrollable(items)
            .id(scrollable_id.clone())
            .on_scroll(SessionsMessage::ScrollChanged)
            .height(Length::Fill)
            .direction(theme::vertical_scrollbar())
            .style(theme::scrollbar_style)
            .into()
    }

    pub fn view(&self) -> Element<'_, SessionsMessage> {
        let mut content = column![];

        if let Some(err) = self.load_state.error() {
            content = content.push(widgets::error_banner(err));
            content = content.push(Space::new().height(12));
        }

        if self.load_state.loading() && !self.load_state.has_loaded() {
            content = content.push(text("Loading...").size(14).color(theme::TEXT_MUTED));
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

                    let sess_row = container(
                        column![
                            row![
                                button(
                                    container(
                                        column![
                                            text(&item.label).size(13).color(theme::TEXT_PRIMARY),
                                            row![
                                                text(&item.msg_count_label)
                                                    .size(11)
                                                    .color(theme::TEXT_MUTED),
                                                Space::new().width(8),
                                                text(&item.timestamp_label)
                                                    .size(11)
                                                    .color(theme::TEXT_MUTED),
                                            ]
                                            .spacing(4),
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
                    .style(theme::surface_card_style);

                    session_list = session_list.push(sess_row);
                }
            }

            let session_scroll = scrollable(session_list)
                .width(Length::Fixed(350.0))
                .direction(theme::vertical_scrollbar())
                .style(theme::scrollbar_style);

            // Transcript on the right side
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
                container(Self::render_transcript(
                    &self.selected_messages,
                    &self.selected_md_items,
                    &self.expanded_tool_rounds,
                    &self.expanded_thinking_blocks,
                    &self.scrollable_id,
                ))
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(8)
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

/// Decode native history messages and parse their content into markdown items.
/// Shared between initial load (`SessionMessages`) and auto-refresh
/// (`AutoRefreshResult`) to keep the decoding logic in a single place.
fn parse_messages_to_md_items(messages: &[ChatMessage]) -> Vec<Vec<markdown::Item>> {
    messages
        .iter()
        .map(|m| {
            let display_text = decode_native_history_message(m)
                .and_then(|d| match d {
                    DecodedNativeHistoryMessage::AssistantToolCalls { content, .. }
                    | DecodedNativeHistoryMessage::AssistantReasoning { content, .. } => content,
                    DecodedNativeHistoryMessage::ToolResult { content, .. } => Some(content),
                })
                .unwrap_or_else(|| m.content.clone());
            markdown::parse(&display_text).collect()
        })
        .collect()
}
