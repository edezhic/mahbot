//! Sessions dashboard page — view and manage conversation sessions.

use std::cell::RefCell;

use crate::ChatMessage;
use crate::session::{DecodedNativeHistoryMessage, SessionMetadata, decode_native_history_message};

use iced::widget::{
    Column, Id, Space, button, column, container, markdown, responsive, row, scrollable, text,
};
use iced::{Alignment, Element, Length, Task};

use iced_anim::Animated;
use iced_anim::transition::Easing;
use std::time::{Duration, Instant};

use iced::window;
use iced_fonts::lucide;

use super::ToastMessage;
use super::menus::{ContextMenu, MenuItem};
use super::session_preview::{
    MAX_PREVIEW_LINES, MessageMeasure, measure_message, re_measure, width_bucket,
};
use super::theme;
use super::widgets;
use super::widgets::selectable_text;

/// Horizontal inset between the `responsive` transcript area width and the
/// markdown body width: the transcript container adds 8px padding per side
/// and each message card adds another 8px per side.
const TRANSCRIPT_CONTENT_INSET_PX: f32 = 32.0;

/// Shared transcript render context: the per-message markdown items, the
/// collapse-measurement cache, the expanded-message set, and the current
/// transcript content width. Bundled so the render helpers do not thread
/// four parameters through every call.
struct TranscriptCtx<'a> {
    md_items: &'a [Vec<markdown::Item>],
    measure_cache: &'a RefCell<Vec<Option<MessageMeasure>>>,
    expanded_messages: &'a std::collections::HashSet<usize>,
    text_width: f32,
}

/// Content of a regular (non-tool-call) message: either plain body text or a
/// `[thinking]` block followed by the visible body text.
enum ContentParts {
    Simple(String),
    WithThinking {
        thinking: String,
        after_thinking: String,
    },
}

/// Parse `[thinking]...[/thinking]` markup from a content string.
/// Returns [`ContentParts::WithThinking`] if a complete thinking block is
/// found, otherwise falls back to [`ContentParts::Simple`].
///
/// The closer is searched only after the opener: a literal `[/thinking]`
/// appearing before the opener must not slice-panic (the historical
/// `content_str.find("[/thinking]")` found a closer at an offset before
/// `body_start`, panicking with "byte range starts at X but ends at Y" — the
/// same crash-class as the RTL measurement panic, reachable from malformed
/// LLM output). Only the first pair is consumed; a later literal block stays
/// visible body text.
///
/// Legacy-divergence note (pre-existing): for messages whose stored content
/// contains a literal `[thinking]` block, the collapse measurement sees the
/// body after the first pair (the collapse rule targets the visible message
/// body — thinking keeps its own collapse) while the expanded markdown body
/// (`md_items`, parsed from the full display text) re-renders the literal
/// block. Such a message can render longer than its measured line count;
/// acceptable per the err-toward-collapse design constraint (the block is
/// not worth re-reading).
fn parse_thinking_blocks(content_str: String) -> ContentParts {
    if let Some(thinking_start) = content_str.find("[thinking]") {
        let body_start = thinking_start + "[thinking]".len();
        if let Some(rel_end) = content_str[body_start..].find("[/thinking]") {
            let end = body_start + rel_end;
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

#[derive(Debug, Clone)]
pub(crate) enum SessionsMessage {
    Refreshed(Vec<SessionMetadata>),
    RefreshError(String),
    SelectSession(String),
    SessionMessages(String, Vec<ChatMessage>),
    SessionError(String),
    ToggleToolRound(usize),
    ToggleThinkingBlock(usize),
    /// Toggle a long message's 3-line preview / full content.
    ToggleMessage(usize),
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
    selected_messages: Vec<ChatMessage>,
    /// Cached parsed markdown items for each message, populated when messages are loaded.
    selected_md_items: Vec<Vec<markdown::Item>>,
    selected_loading: bool,
    expanded_tool_rounds: std::collections::HashSet<usize>,
    expanded_thinking_blocks: std::collections::HashSet<usize>,
    /// Messages the user expanded beyond their 3-line preview. Index-keyed,
    /// survives the periodic refresh, reset when switching sessions.
    expanded_messages: std::collections::HashSet<usize>,
    /// Per-message collapse measurement cache, parallel to `selected_messages`
    /// (entry `i` corresponds to message `i`). Survives the 1-second
    /// auto-refresh — session messages are append-only, so an existing entry
    /// stays valid and only new messages are measured lazily. Mutated during
    /// layout (the transcript width is only known there), hence `RefCell`.
    measure_cache: RefCell<Vec<Option<MessageMeasure>>>,
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
            selected_messages: Vec::new(),
            selected_md_items: Vec::new(),
            selected_loading: false,
            expanded_tool_rounds: std::collections::HashSet::new(),
            expanded_thinking_blocks: std::collections::HashSet::new(),
            expanded_messages: std::collections::HashSet::new(),
            measure_cache: RefCell::new(Vec::new()),
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
                self.expanded_thinking_blocks.clear();
                self.expanded_tool_rounds.clear();
                self.expanded_messages.clear();
                // The measurement cache is cleared+resized in SessionMessages
                // when the new session's messages actually arrive. The
                // loading frames in between show "Loading..." (old messages
                // are not rendered), but the cache must not be cleared here:
                // on SessionError the fallback keeps rendering the previous
                // session's messages, whose cache entries are still
                // index-valid for that path. `expanded_messages` is cleared
                // eagerly because the design resets collapse state on
                // session switch; on the error path that simply renders the
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
                    self.selected_md_items = parse_messages_to_md_items(&messages);
                    // Fresh session load: reset the measurement cache (the
                    // previous session's entries are index-stale).
                    {
                        let mut cache = self.measure_cache.borrow_mut();
                        cache.clear();
                        cache.resize_with(messages.len(), || None);
                    }
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
                // Parse markdown for each message (same as SessionMessages but
                // without touching selected_loading, preserving scrollable identity).
                self.selected_md_items = parse_messages_to_md_items(&messages);
                // Incremental cache: session messages are append-only, so keep
                // existing measurements and only add entries for new messages.
                self.measure_cache
                    .borrow_mut()
                    .resize_with(messages.len(), || None);
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
            SessionsMessage::ToggleMessage(idx) => {
                if self.expanded_messages.contains(&idx) {
                    self.expanded_messages.remove(&idx);
                } else {
                    self.expanded_messages.insert(idx);
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
        self.selected_messages.clear();
        self.selected_md_items.clear();
        self.selected_loading = false;
        self.expanded_tool_rounds.clear();
        self.expanded_thinking_blocks.clear();
        self.expanded_messages.clear();
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

    /// Render the transcript column for the selected session: decode the
    /// messages, group tool-call rounds, and apply the 3-line collapse rule
    /// to regular messages and stray tool results (tool rounds and thinking
    /// blocks keep their own collapse).
    #[expect(clippy::too_many_lines)]
    fn render_transcript<'a>(
        messages: &'a [ChatMessage],
        ctx: &TranscriptCtx<'a>,
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
                tool_call_id: String,
                content: String,
            },
        }

        struct DecodedMsg {
            role: String,
            role_colors: (iced::Color, iced::Color),
            kind: DecodedMsgKind,
        }

        // Used during tool-call↔result matching in the second pass.
        struct MatchedPair<'a> {
            call: &'a ToolCallInfo,
            result_content: Option<&'a str>,
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
            let role_colors = theme::role_badge_color(&msg.role.to_string());

            let kind = if let Some(ref d) = decoded {
                match d {
                    DecodedNativeHistoryMessage::Assistant {
                        content,
                        tool_calls,
                        reasoning,
                    } => {
                        if let Some(tool_calls) = tool_calls {
                            let reasoning_text =
                                crate::providers::plaintext_for_display(reasoning.as_ref());

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
                        } else {
                            let mut parts = Vec::new();
                            if let Some(reasoning_text) =
                                crate::providers::plaintext_for_display(reasoning.as_ref())
                            {
                                parts.push(format!("[thinking]\n{reasoning_text}\n[/thinking]"));
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
                role: msg.role.to_string(),
                role_colors,
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
            let dm_role_colors = decoded_msgs[i].role_colors;

            match &decoded_msgs[i].kind {
                DecodedMsgKind::AssistantToolCalls {
                    calls,
                    reasoning_text,
                    text_content,
                } => {
                    // Collect consecutive ToolResult messages after this tool call
                    let mut result_msgs: Vec<(usize, &str, &str)> = Vec::new();
                    // (msg_index, content, tool_call_id)

                    let mut j = i + 1;
                    while j < len {
                        if let DecodedMsgKind::ToolResult {
                            ref tool_call_id,
                            ref content,
                        } = decoded_msgs[j].kind
                        {
                            result_msgs.push((j, content.as_str(), tool_call_id.as_str()));
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
                            *cid == call.id.as_str() && !used_results.contains(idx)
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

                    // Unmatched results (not consumed by ID matching)
                    let stray_unmatched_results: Vec<(usize, &str)> = result_msgs
                        .iter()
                        .filter(|(idx, _content, _cid)| !used_results.contains(idx))
                        .map(|(idx, content, _cid)| (*idx, *content))
                        .collect();

                    // Unmatched calls (no result)
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
                                container(selectable_text(tc.clone(), theme::TEXT_MUTED).size(11))
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
                                        selectable_text("🧠 Thinking", theme::TEXT_MUTED).size(11),
                                        selectable_text(rt.clone(), theme::TEXT_MUTED).size(11),
                                    ]
                                    .spacing(2),
                                )
                                .padding([4, 8])
                                .style(theme::surface_card_style),
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
                        .style(theme::elevated_card_style);

                    items = items.push(round_card);
                    i = j;
                    round_idx += 1;
                }
                DecodedMsgKind::ToolResult {
                    tool_call_id: _,
                    content,
                } => {
                    // Stray tool result (no preceding tool call) — render as
                    // regular message, subject to the 3-line collapse rule.
                    let mut msg_col = Column::new().spacing(2);
                    msg_col = msg_col.push(widgets::role_badge(
                        dm_role.clone(),
                        dm_role_colors,
                        11,
                        [1, 6],
                        true,
                    ));
                    msg_col = push_message_body(
                        msg_col,
                        ctx,
                        i,
                        (!content.is_empty()).then_some(content.as_str()),
                        messages[i].content.len(),
                    );
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
                        ContentParts::Simple(s) => (None, None, crate::util::none_if_empty(s)),
                        ContentParts::WithThinking {
                            thinking: t,
                            after_thinking: a,
                        } => {
                            let t_owned = t.clone();
                            let a_owned = crate::util::none_if_empty(a);
                            (Some(t_owned), a_owned, None)
                        }
                    };

                    let mut msg_col = Column::new().spacing(2);
                    msg_col = msg_col.push(widgets::role_badge(
                        dm_role,
                        dm_role_colors,
                        11,
                        [1, 6],
                        true,
                    ));

                    if let Some(ref t) = thinking {
                        let is_thinking_expanded = expanded_thinking.contains(&i);

                        let thinking_header = button(
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
                            .style(theme::surface_card_style),
                        )
                        .style(theme::button_text)
                        .on_press(SessionsMessage::ToggleThinkingBlock(i));

                        msg_col = msg_col.push(thinking_header);

                        if is_thinking_expanded {
                            msg_col = msg_col.push(
                                container(selectable_text(t.clone(), theme::TEXT_MUTED).size(11))
                                    .padding([4, 8])
                                    .style(theme::surface_card_style),
                            );
                        }
                    }
                    msg_col = push_message_body(
                        msg_col,
                        ctx,
                        i,
                        simple.as_deref().or(after.as_deref()),
                        messages[i].content.len(),
                    );

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
            // collapse measurement uses the real transcript content width,
            // correct on first render and after every window resize.
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
                let messages = &self.selected_messages;
                let md_items = &self.selected_md_items;
                let expanded_rounds = &self.expanded_tool_rounds;
                let expanded_thinking = &self.expanded_thinking_blocks;
                let expanded_messages = &self.expanded_messages;
                let measure_cache = &self.measure_cache;
                let scrollable_id = self.scrollable_id.clone();
                responsive(move |size| {
                    // The transcript container adds 8px padding each side and
                    // each message card adds another 8px — the markdown body
                    // (and therefore the measurement) sees `size - 32`.
                    let text_width = (size.width - TRANSCRIPT_CONTENT_INSET_PX).max(0.0);
                    let ctx = TranscriptCtx {
                        md_items,
                        measure_cache,
                        expanded_messages,
                        text_width,
                    };
                    container(Self::render_transcript(
                        messages,
                        &ctx,
                        expanded_rounds,
                        expanded_thinking,
                        &scrollable_id,
                    ))
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

/// Decode native history messages and parse their content into markdown items.
/// Shared between initial load (`SessionMessages`) and auto-refresh
/// (`AutoRefreshResult`) to keep the decoding logic in a single place.
///
/// The display text is the decoded assistant/tool-result content (falling
/// back to the raw stored content) — the exact text the collapse measurement
/// sees, so the measured line count matches the rendered body. The markdown
/// body additionally runs [`escape_html_blocks`] first: angle-bracket tag
/// blocks are otherwise classified as HTML and dropped wholesale at parse
/// time (the collapsed preview is plain text, which is why the content is
/// visible collapsed but vanished when expanded via 'Show more').
fn parse_messages_to_md_items(messages: &[ChatMessage]) -> Vec<Vec<markdown::Item>> {
    messages
        .iter()
        .map(|m| {
            let display = decode_native_history_message(m)
                .and_then(|d| match d {
                    DecodedNativeHistoryMessage::Assistant { content, .. } => content,
                    DecodedNativeHistoryMessage::ToolResult { content, .. } => Some(content),
                })
                .unwrap_or_else(|| m.content.clone());
            let processed = super::media_markers::preprocess(&display);
            let escaped = escape_html_blocks(&processed);
            let escaped = super::markdown_breaks::hard_breaks(&escaped);
            markdown::parse(&escaped).collect()
        })
        .collect()
}

/// Escape `<` → `&lt;` only inside the regions the markdown parser classifies
/// as HTML (block-level `Event::Html` plus inline `Event::InlineHtml`), so
/// angle-bracket tag blocks (`<system-notification>`, `<analyze-tool-result>`,
/// `<timestamp>` …) render as visible literal text on the Sessions transcript
/// instead of being dropped wholesale.
///
/// This is deliberately a Sessions-only, `<`-only escape: the shared
/// `media_markers::preprocess` / `selectable_markdown_view` / `theme` helpers
/// used by Home/board/workspaces are untouched, and `util::escape_html` is
/// not reused (it also escapes `>` — breaking blockquotes — and `&`, which
/// would double-encode existing entities).
///
/// Pre-scanning with the same pulldown-cmark parser iced uses confines the
/// change to exactly the bytes that would be dropped: autolinks
/// (`<https://…>`), code spans and fenced code blocks are not HTML events and
/// pass through byte-identical, avoiding both trade-offs of a whole-string
/// escape (dead autolinks, `&lt;` entities inside code). Re-parsing the
/// escaped text also re-interprets the block's inner content as markdown, so
/// bold/lists/code/links keep working.
///
/// Returns a borrowed `Cow` when the text has no `<` at all (the per-second
/// auto-refresh reparse fast path for the common tag-free message) or the
/// scan finds no HTML regions.
fn escape_html_blocks(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains('<') {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for (event, range) in
        pulldown_cmark::Parser::new_ext(text, super::markdown_breaks::iced_markdown_options())
            .into_offset_iter()
    {
        match event {
            pulldown_cmark::Event::Html(_) | pulldown_cmark::Event::InlineHtml(_) => {
                ranges.push(range);
            }
            _ => {}
        }
    }
    if ranges.is_empty() {
        return std::borrow::Cow::Borrowed(text);
    }
    // Events arrive in document order with disjoint ranges; merge overlapping
    // or adjacent ones so multi-line HTML blocks collapse into single
    // replacement regions.
    ranges.sort_by_key(|r| (r.start, r.end));
    let mut merged: Vec<std::ops::Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(last) = merged.last_mut() {
            if range.start <= last.end {
                last.end = last.end.max(range.end);
                continue;
            }
        }
        merged.push(range);
    }
    let mut out = String::with_capacity(text.len() + merged.len() * 4);
    let mut pos = 0;
    for range in &merged {
        out.push_str(&text[pos..range.start]);
        out.push_str(&text[range.clone()].replace('<', "&lt;"));
        pos = range.end;
    }
    out.push_str(&text[pos..]);
    std::borrow::Cow::Owned(out)
}

/// Resolve the collapse measurement for message `i` at the current
/// `text_width`, measuring on cache miss (cached per message and width;
/// session messages are append-only, so a cached entry stays valid across
/// the 1-second auto-refresh). Returns the wrapped-line count and — only
/// when `need_preview` is set (the message is currently collapsed) — its
/// 3-line plain-text preview, so expanded messages do not pay a per-frame
/// preview clone.
///
/// The preview is cloned out of the cache once per frame for each collapsed
/// message (~1KB memcpy-scale, bounded). An `Arc<str>` would not remove that
/// copy: `iced_selection::text::IntoFragment<'a>` accepts `Cow<'a, str>`
/// (`Fragment`'s Borrowed/Owned variants) plus `&str`/`String`/primitives —
/// there is no `Arc<str>` impl — and borrowing `&'a str` out of the
/// `RefCell` cache would be unsound: later cache misses replace entries
/// while widgets from earlier in the frame still hold the borrow.
///
/// `measure_text` is the already-decoded body (both render arms pass it from
/// the decode pass — the measurement never decodes itself; the WithThinking
/// arm passes the body after the first thinking block, so thinking keeps its
/// own collapse and never counts toward the rule, and a later literal block
/// counts as visible body text — err toward collapsing). `content_len` is the
/// raw message content length — a cheap fingerprint so the index-keyed cache
/// re-measures after session compaction replaces a message.
fn message_measurement(
    measure_cache: &RefCell<Vec<Option<MessageMeasure>>>,
    i: usize,
    text_width: f32,
    need_preview: bool,
    measure_text: &str,
    content_len: usize,
) -> (u32, Option<String>) {
    let bucket = width_bucket(text_width);
    let mut cache = measure_cache.borrow_mut();
    let cache_hit = cache.get(i).and_then(Option::as_ref);
    let stale = cache_hit.is_none_or(|m| m.width_bucket != bucket || m.content_len != content_len);
    if stale {
        // Cache miss: measure. A pure width change reuses the cached
        // processed display text (an `Arc` clone — no re-decode, no re-copy,
        // no media preprocessing re-run); otherwise measure from the
        // caller's decoded body.
        let m = match cache_hit {
            Some(m) if m.width_bucket != bucket && m.content_len == content_len => {
                re_measure(m, text_width)
            }
            _ => measure_message(measure_text, text_width, content_len),
        };
        debug_assert!(
            cache.len() > i,
            "cache is resized in lockstep with selected_messages on every load/refresh"
        );
        cache[i] = Some(m);
    }
    let m = cache[i].as_ref().expect("measurement resolved above");
    (
        m.wrapped_lines,
        if need_preview {
            m.preview.clone()
        } else {
            None
        },
    )
}

/// Push the body of message `i` onto `msg_col` with the 3-line collapse rule
/// applied: when the message renders more than [`MAX_PREVIEW_LINES`] wrapped
/// lines and is not expanded, a plain selectable 3-line preview is pushed;
/// otherwise the full markdown view. When the message collapses, a bottom
/// chevron toggle is appended (the toggle and the role badge do not count
/// toward the line budget). Returns the augmented column.
///
/// `measure_text` is the already-decoded body from the first pass (`None` =
/// empty body — the rule never applies to empty content); `content_len` is
/// the raw message content length used as the measurement cache fingerprint.
fn push_message_body<'a>(
    mut msg_col: Column<'a, SessionsMessage>,
    ctx: &TranscriptCtx<'a>,
    i: usize,
    measure_text: Option<&str>,
    content_len: usize,
) -> Column<'a, SessionsMessage> {
    let Some(measure_text) = measure_text else {
        return msg_col;
    };
    let is_expanded = ctx.expanded_messages.contains(&i);
    let (wrapped_lines, preview) = message_measurement(
        ctx.measure_cache,
        i,
        ctx.text_width,
        !is_expanded,
        measure_text,
        content_len,
    );
    let collapses = wrapped_lines > MAX_PREVIEW_LINES;
    if collapses && !is_expanded {
        // Collapsed: plain selectable text of the first lines (media markers
        // shown as placeholders). `preview` is `Some` exactly when the
        // message exceeds the budget (message_measurement returns it only
        // when `need_preview` is set) — fail loud on an invariant break
        // instead of silently rendering an empty message body.
        msg_col = msg_col.push(
            selectable_text(
                preview.expect("a collapsing message always has a preview"),
                theme::TEXT_PRIMARY,
            )
            .size(theme::MARKDOWN_TEXT_SIZE),
        );
    } else {
        msg_col = msg_col.push({
            let md: iced::Element<'_, SessionsMessage, iced::Theme, iced::Renderer> =
                super::media_markers::selectable_markdown_view(
                    &ctx.md_items[i],
                    theme::markdown_settings(),
                )
                .map(SessionsMessage::LinkClicked);
            md
        });
    }
    if collapses {
        msg_col = msg_col.push(message_toggle_button(is_expanded, i));
    }
    msg_col
}

/// Bottom-positioned chevron toggle for a collapsed/expanded long message.
/// Sits below the content (or preview) and does not count toward the
/// 3-line budget.
fn message_toggle_button(is_expanded: bool, idx: usize) -> Element<'static, SessionsMessage> {
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
    .on_press(SessionsMessage::ToggleMessage(idx))
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_thinking_blocks_never_panics_on_reversed_markers() {
        // Regression: a literal "[/thinking]" before "[thinking]" previously
        // sliced `content_str[body_start..end]` with `end < body_start`
        // ("byte range starts at X but ends at Y"), aborting the GUI — the
        // same crash-class as the RTL measurement panic. The closer is
        // searched only after the opener, so this input parses as simple
        // body text instead of crashing.
        let parts = parse_thinking_blocks("[/thinking] before [thinking]".to_string());
        match parts {
            ContentParts::Simple(text) => {
                assert_eq!(text, "[/thinking] before [thinking]");
            }
            ContentParts::WithThinking { .. } => {
                panic!("reversed markers must not form a thinking block");
            }
        }
    }

    #[test]
    fn parse_thinking_blocks_extracts_first_pair_only() {
        let parts = parse_thinking_blocks(
            "intro [thinking] inner [/thinking] mid [thinking] later [/thinking] tail".to_string(),
        );
        match parts {
            ContentParts::WithThinking {
                thinking,
                after_thinking,
            } => {
                assert_eq!(thinking, "inner");
                // Only the first pair is consumed; a later literal block
                // stays visible body text and counts toward the collapse
                // budget — err toward collapsing.
                assert_eq!(after_thinking, "mid [thinking] later [/thinking] tail");
            }
            ContentParts::Simple(_) => panic!("complete first pair must be extracted"),
        }
    }
}
