//! Home page — native GUI chat interface with user impersonation.
//!
//! Users pick an identity from the user picker, select a workspace (sync'd
//! with the Home page workspace picker), and chat with MahBot agents in real time
//! with full markdown rendering and typing indicators.

use crate::ChatDirection;
use crate::Role;
use crate::chat_history::ChatHistoryEntry;
use futures_util::SinkExt;
use iced::widget::rule;
use iced::widget::{
    Column, Id, Space, button, column, container, row, scrollable, text, text_editor,
};
use iced::{Alignment, Element, Length, Task, keyboard};
use iced_fonts::lucide;
use std::collections::HashSet;

use super::ToastMessage;
use super::common::MAX_INPUT_CHARS;
use super::theme;
use super::widgets::PickOption;

/// Maximum number of message IDs to keep in the dedup set before pruning.
const DEDUP_PRUNE_THRESHOLD: usize = 500;

/// Scrollable ID for the chat message list, used for snap-to-end after
/// history loads.
pub(super) const CHAT_SCROLL_ID: Id = Id::new("home_chat_scroll");

/// A displayed chat message in the scroll view.
#[derive(Debug, Clone)]
pub struct DisplayMessage {
    /// Database row ID (Some for history-loaded, None for live arrivals).
    pub id: Option<i64>,
    pub message_id: String,
    pub content: String,
    pub direction: ChatDirection,
    pub agent_role: Option<String>,
    /// Pre-parsed markdown items for rendering.
    pub md_items: Vec<iced::widget::markdown::Item>,
    /// True when this is an optimistic placeholder pushed before the pipeline
    /// confirmation arrives. The `ChatEvent::Message` handler replaces these.
    pub is_optimistic: bool,
}

/// Build a [`DisplayMessage`] from raw parts. Media-preprocesses content and
/// parses markdown; divider directions skip markdown (rendered as rules).
/// `id` is `Some` for history-loaded entries, `None` for live/optimistic
/// messages.
fn display_message(
    id: Option<i64>,
    message_id: String,
    content: String,
    direction: ChatDirection,
    agent_role: Option<String>,
    is_optimistic: bool,
) -> DisplayMessage {
    use iced::widget::markdown;
    let md_items: Vec<markdown::Item> = if direction == ChatDirection::Divider {
        Vec::new()
    } else {
        let processed = super::media_markers::preprocess(&content);
        markdown::parse(&processed).collect()
    };
    DisplayMessage {
        id,
        message_id,
        content,
        direction,
        agent_role,
        md_items,
        is_optimistic,
    }
}

impl From<ChatHistoryEntry> for DisplayMessage {
    fn from(entry: ChatHistoryEntry) -> Self {
        let ChatHistoryEntry {
            id,
            message_id,
            content,
            direction,
            agent_role,
            ..
        } = entry;
        display_message(Some(id), message_id, content, direction, agent_role, false)
    }
}

/// Wrap a chat bubble in a 3:1 FillPortion row so it occupies 75% width,
/// aligned to the right for user messages or to the left for agent/typing.
///
/// The caller must set `.width(Length::FillPortion(3))` on the bubble before
/// passing it — this function only creates the spacer row.
fn align_bubble<'a>(
    bubble: impl Into<Element<'a, HomeMessage>>,
    is_user: bool,
) -> Element<'a, HomeMessage> {
    let bubble = bubble.into();
    if is_user {
        // User: bubble left, spacer right
        row![bubble, Space::new().width(Length::FillPortion(1)),].into()
    } else {
        // Agent: spacer left, bubble right
        row![Space::new().width(Length::FillPortion(1)), bubble,].into()
    }
}

#[derive(Debug, Clone)]
pub enum HomeMessage {
    /// User selected (from picker, Users page icon, or auto-selected at boot).
    UserSelected(String),
    /// Workspace changed (from global picker — propagated via Dashboard).
    WorkspaceChanged(Option<String>),
    /// Text editor content changed.
    InputChanged(text_editor::Action),
    /// Send button pressed or Enter key in editor.
    SendMessage,
    /// Chat history loaded from the store (entries, has_more).
    HistoryLoaded(Vec<ChatHistoryEntry>, bool),
    /// History load failed.
    HistoryLoadError(String),
    /// Live chat event from CHAT_BROADCAST subscription.
    ChatEvent(crate::ChatEvent),
    /// Stream lagged — resync needed.
    StreamLagged,
    /// Scroll position changed in the chat scrollable.
    ScrollChanged(scrollable::Viewport),
    /// User clicked "Load older messages" button.
    LoadOlderMessages,
    /// Older history loaded (entries, has_more, pagination_gen for staleness check).
    OlderHistoryLoaded(Vec<ChatHistoryEntry>, bool, u64),
    /// Older history load failed.
    OlderHistoryLoadError(String),
    /// User list loaded for the picker.
    UsersLoaded(Vec<PickOption>),
    /// Markdown link was clicked.
    LinkClicked(String),
    /// Request a workspace change at the Dashboard level (reverse sync:
    /// DB-stored workspace differs from sidebar). Intercepted by Dashboard;
    /// never reaches Home's own update handler.
    RequestWorkspaceChange(String),
    /// Internal signal: reverse-sync check completed. Carries the
    /// user's DB-stored workspace (None if not set, Some if set).
    /// Proceeds with normal history refresh for the selected user.
    ResolveUserSelected(Option<String>),
    /// User picked a workspace from the Home page picker.
    /// Intercepted by Dashboard; never reaches Home's own update handler.
    WorkspacePicked(String),
    /// Clear chat button pressed — reset session and display.
    ClearChat,
    /// Switch user's active role. Carries (user_name, new_role).
    SwitchRole(String, Role),
    /// Chat history cleared successfully — divider inserted.
    ChatCleared,
    /// Chat history clear failed.
    ChatClearError(String),
    /// Toast notification to show via Dashboard.
    /// Intercepted by Dashboard; never reaches Home's own update handler.
    Toast(ToastMessage),
    /// Typing indicator animation: cycles through 0, 1, 2 → ".", "..", "...".
    TypingTick,
    /// Timeout safety net: if `sending` stays stuck for 30+ seconds,
    /// auto-clear it. Carries the generation counter to prevent stale
    /// timeouts from interfering with a fresh send.
    SendingTimeout(u64),
    /// Undo the last text edit in the chat input.
    Undo,
    /// Redo a previously undone text edit in the chat input.
    Redo,
    /// Keyboard modifiers changed (shift, ctrl, alt, etc.).
    /// Used to track shift state for shift+click selection in the text editor.
    ModifiersChanged(keyboard::Modifiers),
    /// Mic button clicked — start a voice message recording to the active role.
    StartVoiceRecording,
    /// Recording popup: stop recording, transcribe, and send the voice message.
    StopVoiceRecordingSend,
    /// Recording popup: stop recording and discard the voice message.
    StopVoiceRecordingDiscard,
    /// Toggle the composer role dropdown open/closed.
    RoleMenuToggled,
    /// Close the composer role dropdown (on role selection, message send,
    /// chat clear, or user switch).
    RoleMenuClosed,
}

pub struct HomeState {
    /// Currently selected user (sender identifier).
    pub(crate) selected_user: Option<String>,
    /// Currently selected workspace name (synced from dashboard sidebar).
    /// Empty string `""` means the "Personal" workspace — must be resolved
    /// to `personal:<user_name>` before querying chat_history or sessions.
    selected_workspace: Option<String>,
    /// Displayed chat messages.
    messages: Vec<DisplayMessage>,
    /// Deduplication set of seen message IDs.
    seen_ids: HashSet<String>,
    /// Text editor content.
    editor_content: text_editor::Content,
    /// Whether a message is currently being sent / agent is responding.
    sending: bool,
    /// Whether a typing indicator is active.
    typing: bool,
    /// Typing animation dot cycle state: 0=".", 1="..", 2="...".
    typing_tick_state: u8,
    /// Whether the initial history load has happened for the current user+workspace.
    history_loaded: bool,
    /// Generation counter for stale sending timeout detection.
    sending_gen: u64,
    /// True when WorkspaceChanged arrived before a user was selected — the
    /// deferred `refresh_history()` will be triggered by `ResolveUserSelected`.
    pending_workspace_refresh: bool,
    /// Whether auto-scroll is enabled (user is scrolled to the bottom).
    auto_scroll_enabled: bool,
    /// The database ID of the oldest loaded message, if any.
    oldest_loaded_id: Option<i64>,
    /// Whether there are more older messages to load.
    has_more: bool,
    /// Whether an older-messages load is in-flight.
    loading_older: bool,
    /// Generation counter for stale OlderHistoryLoaded callback detection.
    pagination_gen: u64,
    /// Undo/redo stack for the chat input text editor.
    undo_stack: super::common::UndoStack,
    /// Current keyboard modifiers (shift, ctrl, alt, etc.).
    /// Updated from `ModifiersChanged` events. Used to detect shift+click
    /// for extending text selection.
    modifiers: keyboard::Modifiers,
    /// Whether the composer role dropdown is open.
    role_menu_open: bool,
}

impl HomeState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            selected_user: None,
            selected_workspace: None,
            messages: Vec::new(),
            seen_ids: HashSet::new(),
            editor_content: text_editor::Content::new(),
            sending: false,
            typing: false,
            typing_tick_state: 0,
            history_loaded: false,
            sending_gen: 0,
            pending_workspace_refresh: false,
            auto_scroll_enabled: true,
            oldest_loaded_id: None,
            has_more: false,
            loading_older: false,
            pagination_gen: 0,
            undo_stack: super::common::UndoStack::new(),
            modifiers: keyboard::Modifiers::empty(),
            role_menu_open: false,
        }
    }

    /// Load users for the user picker.
    #[allow(clippy::unused_self)]
    pub fn load_users(&self) -> Task<HomeMessage> {
        Task::perform(
            async {
                let Some(store) = crate::users::USER_STORE.get() else {
                    return Vec::new();
                };
                let users = store.list_users().await.unwrap_or_default();
                users
                    .iter()
                    .map(|u| PickOption {
                        value: u.name.clone(),
                        label: u.name.clone(),
                    })
                    .collect()
            },
            HomeMessage::UsersLoaded,
        )
    }

    /// Resolve the workspace name for chat history and session queries.
    /// `None` or empty string (Personal) → `personal:<user_name>`.
    /// Non-empty workspace → the workspace name as-is.
    fn resolve_workspace_name(&self) -> Option<String> {
        match &self.selected_workspace {
            Some(w) if !w.is_empty() => Some(w.clone()),
            _ => {
                let user = self.selected_user.as_ref()?;
                Some(format!("personal:{user}"))
            }
        }
    }

    /// Reverse-sync the DB-stored workspace preference for a user.
    ///
    /// Checks whether the user has a stored workspace preference that differs
    /// from the current sidebar selection. If so, returns
    /// [`RequestWorkspaceChange`] to trigger a Dashboard-level change (which
    /// will cascade to [`WorkspaceChanged`] → `refresh_history`). Otherwise
    /// returns [`ResolveUserSelected`] to proceed with a normal history refresh.
    ///
    /// NOTE: We deliberately do NOT write the sidebar workspace to the
    /// impersonated user's DB record.  The GUI sidebar is a per-session
    /// context — persisting it would silently overwrite the user's real
    /// workspace choice.
    async fn resolve_user_workspace_sync(user: String, current_ws: Option<String>) -> HomeMessage {
        match crate::users::get_raw_selected_workspace(&user).await {
            Ok(Some(ws_name)) => {
                // User has an explicit stored workspace preference.
                // Normalize personal workspaces to the GUI sentinel "".
                let ws_gui = if crate::users::is_personal_workspace(&ws_name) {
                    String::new()
                } else {
                    ws_name
                };
                if Some(&ws_gui) != current_ws.as_ref() {
                    HomeMessage::RequestWorkspaceChange(ws_gui)
                } else {
                    HomeMessage::ResolveUserSelected(current_ws.clone())
                }
            }
            Ok(None) => {
                // User has no stored preference — keep current sidebar selection.
                HomeMessage::ResolveUserSelected(current_ws.clone())
            }
            Err(e) => {
                tracing::warn!("Failed to get raw workspace for user {user}: {e}");
                HomeMessage::ResolveUserSelected(current_ws.clone())
            }
        }
    }

    /// Refresh chat history from the store for the current user + workspace.
    fn refresh_history(&self) -> Task<HomeMessage> {
        let user_name = match &self.selected_user {
            Some(s) => s.clone(),
            None => return Task::none(),
        };
        let Some(workspace) = self.resolve_workspace_name() else {
            return Task::none();
        };
        Task::perform(
            async move {
                let store = crate::chat_history::store();
                store
                    .load_for_user(&user_name, &workspace)
                    .await
                    .map_err(|e| e.to_string())
            },
            |result| match result {
                Ok((entries, has_more)) => HomeMessage::HistoryLoaded(entries, has_more),
                Err(e) => HomeMessage::HistoryLoadError(e),
            },
        )
    }

    /// Push a new chat message to the display. Returns the message's ID for dedup tracking.
    fn push_message(&mut self, entry: ChatHistoryEntry) -> String {
        let msg_id = entry.message_id.clone();
        self.messages.push(entry.into());
        msg_id
    }

    /// Reset pagination and auto-scroll state. Called at all cleanup sites
    /// (user change, workspace change, role change, clear, stream lag).
    const fn reset_pagination_state(&mut self) {
        self.oldest_loaded_id = None;
        self.has_more = false;
        self.loading_older = false;
        self.auto_scroll_enabled = true;
        self.pagination_gen = self.pagination_gen.wrapping_add(1);
    }

    /// Clear chat display state: messages, dedup set, history flag, pagination.
    fn reset_chat_state(&mut self) {
        self.messages.clear();
        self.seen_ids.clear();
        self.history_loaded = false;
        self.role_menu_open = false;
        self.reset_pagination_state();
    }

    /// Produce a snap-to-end task if auto-scroll is enabled.
    fn maybe_snap(&self) -> Task<HomeMessage> {
        if self.auto_scroll_enabled {
            iced::widget::operation::snap_to_end(CHAT_SCROLL_ID)
        } else {
            Task::none()
        }
    }

    /// Replace an optimistic placeholder with a confirmed pipeline message.
    ///
    /// If `optimistic_id` matches a locally-inserted optimistic message
    /// (`is_optimistic && message_id == optimistic_id`), swaps in the real
    /// [`DisplayMessage`], marks the canonical ID as seen, clears `sending`,
    /// and returns `Some(snap_task)` so the caller can early-return.
    /// Returns `None` when no replacement was performed.
    #[allow(clippy::too_many_arguments)]
    fn replace_optimistic(
        &mut self,
        optimistic_id: Option<&str>,
        message_id: &str,
        content: &str,
        direction: ChatDirection,
        agent_role: Option<&str>,
    ) -> Option<Task<HomeMessage>> {
        if let Some(opt_id) = optimistic_id {
            if let Some(pos) = self
                .messages
                .iter()
                .position(|m| m.is_optimistic && m.message_id == *opt_id)
            {
                self.messages[pos] = display_message(
                    None,
                    message_id.to_string(),
                    content.to_string(),
                    direction,
                    agent_role.map(std::string::ToString::to_string),
                    false,
                );
                // Track the canonical ID for dedup — the optimistic ID was
                // never added to seen_ids.
                self.seen_ids.insert(message_id.to_string());
                // User's own message confirmed by pipeline — clear sending
                // so the button re-enables.
                self.sending = false;
                return Some(self.maybe_snap());
            }
        }
        None
    }

    /// Try to deduplicate a message by its ID.
    ///
    /// Returns `true` if the message was already seen (caller should bail).
    /// Inserts fresh IDs into `seen_ids` and prunes the set (keeping the
    /// most recent 200 IDs) when it exceeds [`DEDUP_PRUNE_THRESHOLD`].
    fn try_dedup(&mut self, message_id: &str) -> bool {
        if self.seen_ids.contains(message_id) {
            return true;
        }
        self.seen_ids.insert(message_id.to_string());

        if self.seen_ids.len() > DEDUP_PRUNE_THRESHOLD {
            let retain: HashSet<String> = self
                .messages
                .iter()
                .rev()
                .take(200)
                .map(|m| m.message_id.clone())
                .collect();
            self.seen_ids.retain(|id| retain.contains(id));
        }
        false
    }

    /// Update typing/sending state based on message direction and sender.
    ///
    /// * **Agent** responses for the selected user → clear both `typing`
    ///   and `sending` (the agent has replied).
    /// * **User** message echo for the selected user → clear `sending`
    ///   only (re-enables the send button). Does **not** clear `typing`
    ///   — the typing indicator persists until an agent response arrives.
    ///
    /// Clear sending/typing state based on direction, sender, and workspace.
    ///
    /// Does nothing when `workspace` does not match
    /// [`resolve_workspace_name()`](Self::resolve_workspace_name) — this
    /// prevents an agent response from workspace B from clearing the
    /// typing/sending indicators for workspace A.
    fn update_sending_state(&mut self, direction: ChatDirection, user_name: &str, workspace: &str) {
        if Some(workspace) != self.resolve_workspace_name().as_deref() {
            return;
        }
        if Some(user_name) != self.selected_user.as_deref() {
            return;
        }

        self.sending = false;
        if direction == ChatDirection::Agent {
            self.typing = false;
        }
    }

    /// Append a chat message if it belongs to the selected user + workspace.
    ///
    /// Does nothing when `user_name` is not the selected user, or when
    /// `workspace` does not match [`resolve_workspace_name()`](Self::resolve_workspace_name).
    /// Takes ownership of the message fields so the caller avoids extra
    /// clones on the common (append) path.
    ///
    /// The caller should call [`maybe_snap()`](Self::maybe_snap)
    /// unconditionally after this (snap is always safe when nothing was
    /// appended).
    #[allow(clippy::too_many_arguments)]
    fn append_message(
        &mut self,
        user_name: &str,
        workspace: &str,
        message_id: String,
        content: String,
        direction: ChatDirection,
        agent_role: Option<String>,
    ) {
        if Some(user_name) != self.selected_user.as_deref() {
            return;
        }
        if Some(workspace) != self.resolve_workspace_name().as_deref() {
            return;
        }

        self.messages.push(display_message(
            None, message_id, content, direction, agent_role, false,
        ));
    }

    #[allow(clippy::too_many_lines)]
    pub fn view(
        &self,
        active_role: Option<Role>,
        role_pool: &[Role],
        draining: bool,
    ) -> Element<'_, HomeMessage> {
        // ── Chat message area ────────────────────────────────────
        let chat_area = if self.messages.is_empty() {
            let empty_hint = if self.selected_user.is_none() {
                "No user selected. Create users via the Users page."
            } else if self.selected_workspace.is_none() {
                "No workspace selected."
            } else {
                "No messages yet. Type something below to start."
            };
            container(text(empty_hint).color(theme::TEXT_SECONDARY).size(13))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .style(theme::base_container_style)
        } else {
            // Build message bubbles with typing indicator.
            let mut children: Vec<Element<'_, HomeMessage>> = self
                .messages
                .iter()
                .map(|msg| {
                    // ── Divider marker ────────────────────────────────────
                    if msg.direction == ChatDirection::Divider {
                        // Render as a horizontal rule with a label.
                        let label: Element<'_, HomeMessage> = container(
                            text("─ Session cleared ─")
                                .color(theme::TEXT_MUTED)
                                .size(12),
                        )
                        .center_x(Length::Fill)
                        .into();

                        let divider_rule = |_: &iced::Theme| rule::Style {
                            color: theme::TEXT_MUTED,
                            radius: 0.0.into(),
                            fill_mode: rule::FillMode::Padded(0),
                            snap: true,
                        };

                        let divider = column![
                            rule::horizontal(1).style(divider_rule),
                            label,
                            rule::horizontal(1).style(divider_rule),
                        ]
                        .spacing(4)
                        .padding(8)
                        .width(Length::Fill);

                        return divider.into();
                    }

                    let is_user = msg.direction == ChatDirection::User;

                    // Render markdown content
                    let content: Element<'_, HomeMessage> = if msg.md_items.is_empty() {
                        super::widgets::selectable_text(&msg.content, theme::TEXT_PRIMARY)
                            .size(13)
                            .into()
                    } else {
                        super::media_markers::selectable_markdown_view(
                            &msg.md_items,
                            theme::markdown_settings(),
                        )
                        .map(HomeMessage::LinkClicked)
                    };

                    // Build bubble body: role icon header for agents, or just content for users.
                    let bubble_body: Element<'_, HomeMessage> = if is_user {
                        content
                    } else {
                        // Strip numeric suffix (e.g. "analyst_3" → "analyst") and parse.
                        let maybe_role = msg.agent_role.as_ref().and_then(|r| {
                            let stripped = r
                                .rsplit_once('_')
                                .and_then(|(base, suffix)| {
                                    if suffix.chars().all(|c| c.is_ascii_digit()) {
                                        Some(base)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(r.as_str());
                            stripped.parse::<Role>().ok()
                        });
                        if let Some(role) = maybe_role {
                            let (icon_color, _) = theme::role_badge_color_for(&role);
                            let icon = theme::role_icon(&role).size(14).color(icon_color);
                            column![row![icon].align_y(Alignment::Center), content]
                                .spacing(4)
                                .into()
                        } else {
                            content
                        }
                    };

                    let bubble = container(bubble_body)
                        .padding(10)
                        .style(theme::bubble_style(
                            if is_user {
                                theme::BG_ELEVATED
                            } else {
                                theme::BG_SURFACE
                            },
                            Some(theme::TEXT_PRIMARY),
                        ))
                        .width(Length::FillPortion(3));

                    align_bubble(bubble, is_user)
                })
                .collect();

            if self.typing {
                let dots = match self.typing_tick_state {
                    1 => "..",
                    2 => "...",
                    _ => ".",
                };
                let typing_dots = text(dots).size(20).color(theme::TEXT_MUTED);
                let typing_bubble = container(typing_dots)
                    .padding(10)
                    .style(theme::bubble_style(theme::BG_SURFACE, None))
                    .width(Length::FillPortion(3));

                children.push(align_bubble(typing_bubble, false));
            }

            // Prepend "Load older messages" button when applicable.
            if self.has_more && self.history_loaded {
                let load_text = if self.loading_older {
                    "Loading older messages..."
                } else {
                    "▲ Load older messages"
                };
                let load_btn = button(text(load_text).size(12).color(theme::TEXT_SECONDARY))
                    .style(move |_t: &iced::Theme, _status| {
                        use iced::widget::button;
                        button::Style {
                            background: Some(iced::Background::Color(theme::BG_SURFACE)),
                            border: iced::Border {
                                radius: 4.0.into(),
                                width: 0.0,
                                color: iced::Color::TRANSPARENT,
                            },
                            text_color: theme::TEXT_SECONDARY,
                            ..button::Style::default()
                        }
                    })
                    .width(Length::Fill)
                    .on_press_maybe(if self.loading_older {
                        None
                    } else {
                        Some(HomeMessage::LoadOlderMessages)
                    });
                children.insert(0, container(load_btn).padding(4).into());
            }

            container(
                scrollable(Column::with_children(children).spacing(12).padding(8))
                    .id(CHAT_SCROLL_ID)
                    .on_scroll(HomeMessage::ScrollChanged)
                    .direction(theme::vertical_scrollbar())
                    .style(theme::scrollbar_style)
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .style(theme::base_container_style)
        };

        // ── Input area ───────────────────────────────────────────
        let voice_status = crate::audio::voice::get_status();
        let recording = matches!(
            voice_status,
            crate::audio::voice::VoiceStatus::RecordingManual
        );
        // The Transcribing status is shared between the manual and wake-word
        // paths; only a mic-button recording shows the composer popup.
        let transcribing = matches!(voice_status, crate::audio::voice::VoiceStatus::Transcribing)
            && crate::audio::voice::is_manual_recording();
        // The mic is busy while the pipeline owns the mic for any recording
        // or ASR (manual or wake-word) — the button must not look active
        // when a new recording would be rejected.
        let mic_busy = matches!(
            voice_status,
            crate::audio::voice::VoiceStatus::Recording
                | crate::audio::voice::VoiceStatus::RecordingManual
                | crate::audio::voice::VoiceStatus::Transcribing
        );

        // Right-edge controls column: role selector + mic button.
        let mut controls: Vec<Element<'_, HomeMessage>> = Vec::new();
        let role_icon = match active_role {
            Some(role) => {
                let (fg, _) = theme::role_badge_color_for(&role);
                theme::role_icon(&role).size(15).color(fg)
            }
            None => lucide::bot::<iced::Theme, iced::Renderer>()
                .size(15)
                .color(theme::TEXT_MUTED),
        };
        let role_btn = button(role_icon)
            .on_press_maybe(
                (self.selected_user.is_some() && !role_pool.is_empty())
                    .then_some(HomeMessage::RoleMenuToggled),
            )
            .style(theme::icon_button_style(false))
            .padding(3);
        controls.push(role_btn.into());

        let mic_btn = button(lucide::mic::<iced::Theme, iced::Renderer>().size(14).color(
            if mic_busy {
                theme::TEXT_MUTED
            } else {
                theme::TEXT_SECONDARY
            },
        ))
        .on_press_maybe(
            (self.selected_user.is_some() && !mic_busy).then_some(HomeMessage::StartVoiceRecording),
        )
        .style(theme::icon_button_style(mic_busy))
        .padding(3);
        controls.push(mic_btn.into());

        let input_area = super::widgets::chat_composer(
            &self.editor_content,
            HomeMessage::InputChanged,
            HomeMessage::SendMessage,
            "Type a message... (Enter to send, Shift+Enter for newline)",
            super::widgets::ChatComposerOptions {
                // Input disabled during the graceful drain (decision 3):
                // sends are blocked while the shutdown banner is up.
                sending: self.sending || draining,
                // One line taller than the plain composer so the controls
                // column (role + mic) fits above the send button.
                min_height: 88.0,
                max_height: 330.0,
                controls,
                grey_on_empty: true,
            },
        );

        // ── Role dropdown panel (in-flow, above the composer) ────
        let role_panel: Element<'_, HomeMessage> = if self.role_menu_open && !role_pool.is_empty() {
            let user = self.selected_user.clone().unwrap_or_default();
            let items: Vec<Element<'_, HomeMessage>> = role_pool
                .iter()
                .map(|role| {
                    let (fg, _) = theme::role_badge_color_for(role);
                    let icon = theme::role_icon(role).size(13).color(fg);
                    let is_current = active_role.as_ref() == Some(role);
                    let label = text(crate::role::role_info(role).display_label)
                        .size(13)
                        .color(if is_current {
                            theme::TEXT_PRIMARY
                        } else {
                            theme::TEXT_SECONDARY
                        });
                    let check: Element<'_, HomeMessage> = if is_current {
                        lucide::check::<iced::Theme, iced::Renderer>()
                            .size(12)
                            .color(theme::ACCENT)
                            .into()
                    } else {
                        Space::new().width(12).into()
                    };
                    let content = row![icon, label, check]
                        .spacing(6)
                        .align_y(Alignment::Center);
                    button(content)
                        .on_press_maybe(
                            (!is_current).then_some(HomeMessage::SwitchRole(user.clone(), *role)),
                        )
                        .style(theme::menu_item_button_style())
                        .width(Length::Fill)
                        .padding(5)
                        .into()
                })
                .collect();
            container(Column::with_children(items).spacing(2).padding(5))
                .style(theme::surface_container_style)
                .width(Length::Fixed(170.0))
                .into()
        } else {
            Space::new().height(0).into()
        };

        // ── Recording popup (stop + send / stop + discard) ───────
        // While transcribing, the popup stays visible as a passive
        // "Transcribing…" indicator (no stop controls — the ASR is finalizing).
        let recording_popup: Element<'_, HomeMessage> = if recording {
            let status_label = text("Recording voice message…")
                .size(13)
                .color(theme::STATUS_ERROR);
            let send_btn = button(text("Stop + Send").size(12))
                .on_press(HomeMessage::StopVoiceRecordingSend)
                .style(theme::button_primary)
                .padding(5);
            let discard_btn = button(text("Stop + Discard").size(12))
                .on_press(HomeMessage::StopVoiceRecordingDiscard)
                .style(theme::button_secondary)
                .padding(5);
            container(
                row![
                    status_label,
                    Space::new().width(Length::Fill),
                    send_btn,
                    discard_btn
                ]
                .spacing(8)
                .align_y(Alignment::Center),
            )
            .padding(8)
            .style(theme::surface_container_style)
            .width(Length::Fill)
            .into()
        } else if transcribing {
            container(
                row![
                    text("Transcribing voice message…")
                        .size(13)
                        .color(theme::TEXT_MUTED),
                    Space::new().width(Length::Fill),
                ]
                .spacing(8)
                .align_y(Alignment::Center),
            )
            .padding(8)
            .style(theme::surface_container_style)
            .width(Length::Fill)
            .into()
        } else {
            Space::new().height(0).into()
        };

        // ── Full layout ──────────────────────────────────────────
        column![chat_area, role_panel, recording_popup, input_area,]
            .align_x(Alignment::End)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    #[allow(clippy::unused_self)]
    pub fn subscription(&self) -> iced::Subscription<HomeMessage> {
        let mut subs = vec![
            iced::Subscription::run(chat_stream_producer),
            iced::Subscription::run(typing_tick),
        ];

        // Keyboard shortcuts: Cmd+Z → undo, Cmd+Shift+Z → redo.
        // Also track modifier changes for shift+click text selection.
        subs.push(keyboard::listen().filter_map(|event| {
            super::common::composer_keyboard_event(
                event,
                HomeMessage::ModifiersChanged,
                || HomeMessage::Undo,
                || HomeMessage::Redo,
            )
        }));

        // Reset keyboard modifiers when the window loses focus, preventing
        // stale shift/ctrl/alt state from affecting the editor if the user
        // presses a modifier, switches apps, releases it, and returns.
        subs.push(iced::window::events().filter_map(|(_id, event)| {
            if matches!(event, iced::window::Event::Unfocused) {
                Some(HomeMessage::ModifiersChanged(keyboard::Modifiers::empty()))
            } else {
                None
            }
        }));

        iced::Subscription::batch(subs)
    }

    #[allow(clippy::too_many_lines)]
    pub fn update(&mut self, msg: HomeMessage) -> Task<HomeMessage> {
        // Allow match_same_arms on the entire update() match block: many variants
        // return Task::none() as intercepted/intermediate messages. Narrowing to
        // individual arms would require auditing every no-op each time one changes.
        #[allow(clippy::match_same_arms)]
        match msg {
            HomeMessage::UserSelected(user) => {
                if self.selected_user.as_deref() == Some(&user) {
                    return Task::none();
                }
                self.selected_user = Some(user.clone());
                self.reset_chat_state();

                Task::perform(
                    Self::resolve_user_workspace_sync(user, self.selected_workspace.clone()),
                    |msg| msg,
                )
            }
            HomeMessage::WorkspaceChanged(ws_name) => {
                self.selected_workspace.clone_from(&ws_name);
                self.reset_chat_state();

                // When a user is already selected, refresh history immediately.
                // Otherwise defer — `ResolveUserSelected` will pick it up once
                // a user is chosen (e.g. first boot before UsersLoaded fires).
                if self.selected_user.is_some() {
                    self.pending_workspace_refresh = false;
                    self.refresh_history()
                } else {
                    self.pending_workspace_refresh = true;
                    Task::none()
                }
            }
            HomeMessage::InputChanged(action) => {
                super::common::apply_editor_action(
                    &mut self.editor_content,
                    &mut self.undo_stack,
                    action,
                    self.modifiers.shift(),
                );
                Task::none()
            }
            HomeMessage::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers;
                Task::none()
            }
            HomeMessage::Undo => {
                let snapshot = self.undo_stack.undo(&self.editor_content);
                super::common::restore_undo_snapshot(&mut self.editor_content, snapshot);
                Task::none()
            }
            HomeMessage::Redo => {
                let snapshot = self.undo_stack.redo(&self.editor_content);
                super::common::restore_undo_snapshot(&mut self.editor_content, snapshot);
                Task::none()
            }
            HomeMessage::ResolveUserSelected(workspace) => {
                // Reverse-sync check completed: either the user's DB workspace
                // matches the sidebar (no disagreement), or no DB workspace
                // exists for this user.
                self.selected_workspace = workspace;
                //
                // If WorkspaceChanged arrived before a user was selected
                // (boot timing), it deferred the refresh via the flag.
                // Clear stale state now before loading history.
                if self.pending_workspace_refresh {
                    self.pending_workspace_refresh = false;
                    self.reset_chat_state();
                }
                self.refresh_history()
            }
            HomeMessage::SendMessage => {
                self.role_menu_open = false;
                self.send_message()
            }
            HomeMessage::HistoryLoaded(entries, has_more) => {
                // Track oldest loaded ID and whether more exist for pagination.
                self.oldest_loaded_id = entries.first().map(|e| e.id);
                self.has_more = has_more;
                for entry in entries {
                    let msg_id = self.push_message(entry);
                    self.seen_ids.insert(msg_id);
                }
                self.history_loaded = true;
                // Snap to end only if auto-scroll is enabled.
                self.maybe_snap()
            }
            HomeMessage::HistoryLoadError(e) => {
                tracing::warn!(error = %e, "Home: failed to load chat history");
                Task::none()
            }
            HomeMessage::UsersLoaded(options) => {
                // If no user is selected, auto-select the first one (admin at boot).
                if self.selected_user.is_none() && !options.is_empty() {
                    let first = options[0].value.clone();
                    return Task::done(HomeMessage::UserSelected(first));
                }
                // If the selected user no longer exists in the loaded list
                // (deleted from another session), auto-select the first user.
                if let Some(ref user) = self.selected_user {
                    if !options.iter().any(|opt| opt.value == *user) && !options.is_empty() {
                        let first = options[0].value.clone();
                        return Task::done(HomeMessage::UserSelected(first));
                    }
                }
                Task::none()
            }
            HomeMessage::ClearChat => {
                // Clear messages synchronously first (prevents flash).
                self.messages.clear();
                self.seen_ids.clear();
                self.sending = false;
                self.typing = false;
                self.typing_tick_state = 0;
                self.role_menu_open = false;
                self.reset_pagination_state();

                // Build agent ID and schedule async cleanup.
                let sender = match &self.selected_user {
                    Some(s) => s.clone(),
                    None => return Task::none(),
                };
                let Some(ws) = self.resolve_workspace_name() else {
                    return Task::none();
                };
                Task::perform(
                    async move {
                        // Look up the user's active role from DB (set via the Users page).
                        let role = crate::users::get_active_role(&sender)
                            .await
                            .ok()
                            .flatten()
                            .unwrap_or(Role::Manager.as_str().to_string());
                        // Clear the session.
                        let _ = crate::session::clear_session("gui", &sender, &role, &ws).await;
                        // Insert a divider marker instead of deleting history.
                        let store = crate::chat_history::store();
                        match store.insert_divider(&sender, "gui", &ws).await {
                            Ok(()) => Ok(()),
                            Err(e) => {
                                tracing::warn!(
                                    user = %sender,
                                    workspace = %ws,
                                    error = %e,
                                    "Home: failed to insert chat divider"
                                );
                                Err(e.to_string())
                            }
                        }
                    },
                    |result| match result {
                        Ok(()) => HomeMessage::ChatCleared,
                        Err(e) => HomeMessage::ChatClearError(e),
                    },
                )
            }
            HomeMessage::ChatCleared => {
                let toast = Task::done(HomeMessage::Toast(ToastMessage::SuccessMsg(
                    "Session cleared".to_string(),
                )));
                Task::batch([self.refresh_history(), toast])
            }
            HomeMessage::SwitchRole(user, role) => {
                // Intercepted by Dashboard — no-op in Home
                tracing::debug!("Home: SwitchRole({user}, {role}) — handled by Dashboard");
                Task::none()
            }
            HomeMessage::ChatClearError(e) => {
                Task::done(HomeMessage::Toast(ToastMessage::Error(e)))
            }
            HomeMessage::ChatEvent(event) => match event {
                crate::ChatEvent::Message {
                    message_id,
                    user_name,
                    content,
                    direction,
                    timestamp: _,
                    channel: _,
                    agent_role,
                    workspace,
                    optimistic_id,
                } => {
                    // 1. Replace optimistic placeholder if present.
                    if let Some(task) = self.replace_optimistic(
                        optimistic_id.as_deref(),
                        &message_id,
                        &content,
                        direction,
                        agent_role.as_deref(),
                    ) {
                        return task;
                    }

                    // 2. Deduplicate against already-seen IDs.
                    if self.try_dedup(&message_id) {
                        return Task::none();
                    }

                    // 3. Clear sending/typing state based on direction, sender, and workspace.
                    self.update_sending_state(direction, &user_name, &workspace);

                    // 4. Append the message (filtered by selected user + workspace).
                    self.append_message(
                        &user_name, &workspace, message_id, content, direction, agent_role,
                    );

                    self.maybe_snap()
                }
                crate::ChatEvent::Typing {
                    user_name,
                    is_typing,
                    workspace,
                } => {
                    // Apply user + workspace filter — only show typing indicator
                    // for the selected user in the selected workspace.
                    if Some(&user_name) == self.selected_user.as_ref()
                        && Some(workspace.as_str()) == self.resolve_workspace_name().as_deref()
                    {
                        self.typing = is_typing;
                        if is_typing {
                            self.typing_tick_state = 0;
                        }
                    }
                    Task::none()
                }
            },
            HomeMessage::StreamLagged => {
                // Resync: reload history. Also clear sending as a safety
                // net — if the agent response was dropped due to the lag,
                // this prevents the send button from staying stuck.
                self.sending = false;
                self.seen_ids.clear();
                self.reset_pagination_state();
                self.refresh_history()
            }
            HomeMessage::ScrollChanged(viewport) => {
                // Determine if the user is at the bottom. Two checks:
                // 1. Content is taller than viewport AND relative offset >= 0.99
                // 2. Content fits entirely in viewport (no scrolling needed)
                let at_bottom = {
                    let bounds = viewport.bounds();
                    let content = viewport.content_bounds();
                    if content.height > bounds.height {
                        viewport.relative_offset().y >= 0.99
                    } else {
                        content.height <= bounds.height
                    }
                };
                self.auto_scroll_enabled = at_bottom;
                Task::none()
            }
            HomeMessage::LoadOlderMessages => {
                // Guard against double-clicks.
                if self.loading_older {
                    return Task::none();
                }
                self.loading_older = true;
                let sender = match &self.selected_user {
                    Some(s) => s.clone(),
                    None => return Task::none(),
                };
                let Some(workspace) = self.resolve_workspace_name() else {
                    return Task::none();
                };
                let Some(before_id) = self.oldest_loaded_id else {
                    self.loading_older = false;
                    return Task::none();
                };
                let generation = self.pagination_gen;
                Task::perform(
                    async move {
                        let store = crate::chat_history::store();
                        store
                            .load_older_for_user(&sender, &workspace, before_id)
                            .await
                            .map(|(entries, has_more)| (entries, has_more, generation))
                            .map_err(|e| e.to_string())
                    },
                    |result| match result {
                        Ok((entries, has_more, generation)) => {
                            HomeMessage::OlderHistoryLoaded(entries, has_more, generation)
                        }
                        Err(e) => HomeMessage::OlderHistoryLoadError(e),
                    },
                )
            }
            HomeMessage::OlderHistoryLoaded(display_entries, has_more, generation) => {
                // Guard against stale callbacks.
                if generation != self.pagination_gen {
                    self.loading_older = false;
                    return Task::none();
                }
                // Prepend entries to the beginning of messages.
                let mut prepended: Vec<DisplayMessage> = display_entries
                    .into_iter()
                    .map(DisplayMessage::from)
                    .collect();
                // Track seen_ids for the prepended messages.
                for msg in &prepended {
                    self.seen_ids.insert(msg.message_id.clone());
                }
                prepended.append(&mut self.messages);
                self.messages = prepended;
                // Update oldest_loaded_id and has_more.
                self.oldest_loaded_id = self.messages.first().and_then(|m| m.id);
                self.has_more = has_more;
                self.loading_older = false;
                // Snap to end if auto-scroll enabled.
                self.maybe_snap()
            }
            HomeMessage::OlderHistoryLoadError(msg) => {
                self.loading_older = false;
                Task::done(HomeMessage::Toast(ToastMessage::Error(msg)))
            }
            HomeMessage::RequestWorkspaceChange(_) => {
                // This variant is intercepted by the Dashboard and should
                // never reach Home's update handler.  No-op fallback.
                Task::none()
            }
            HomeMessage::WorkspacePicked(_) => {
                // Intercepted by Dashboard.  No-op fallback.
                Task::none()
            }
            HomeMessage::Toast(_) => {
                // Intercepted by Dashboard.  No-op fallback.
                Task::none()
            }
            HomeMessage::LinkClicked(url) => {
                super::open_url(&url);
                Task::none()
            }
            HomeMessage::TypingTick => {
                if self.typing {
                    self.typing_tick_state = (self.typing_tick_state + 1) % 3;
                }
                Task::none()
            }
            HomeMessage::SendingTimeout(generation) => {
                // Only clear sending if the generation counter matches —
                // a stale timeout from a previous send should be ignored.
                if generation == self.sending_gen && self.sending {
                    self.sending = false;
                }
                Task::none()
            }
            HomeMessage::RoleMenuToggled => {
                self.role_menu_open = !self.role_menu_open;
                Task::none()
            }
            HomeMessage::RoleMenuClosed => {
                self.role_menu_open = false;
                Task::none()
            }
            HomeMessage::StartVoiceRecording => {
                self.role_menu_open = false;
                // Best-effort pre-flight check surfaced by the pipeline itself
                // (single source of truth for the blocked-state mapping). The
                // pipeline remains the authoritative guard — this predicate can
                // drift on transient transitions but covers the common cases.
                if let Some(msg) = crate::audio::voice::manual_recording_blocked_reason() {
                    return Task::done(HomeMessage::Toast(ToastMessage::Warning(msg.to_string())));
                }
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::StartRecording,
                );
                Task::none()
            }
            HomeMessage::StopVoiceRecordingSend => {
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::StopRecordingSend,
                );
                Task::none()
            }
            HomeMessage::StopVoiceRecordingDiscard => {
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::StopRecordingDiscard,
                );
                Task::none()
            }
        }
    }

    /// Construct and send the user's message through the GUI channel.
    fn send_message(&mut self) -> Task<HomeMessage> {
        let text = self.editor_content.text();
        let trimmed = match super::common::send_guard(&text, self.sending, true, |count| {
            Task::done(HomeMessage::Toast(ToastMessage::Warning(format!(
                "Message too long: {count} characters (maximum {MAX_INPUT_CHARS}). Please shorten your message and try again."
            ))))
        }) {
            Ok(t) => t,
            Err(task) => return task,
        };

        let content = trimmed.to_string();

        let sender = match &self.selected_user {
            Some(s) => s.clone(),
            None => return Task::none(),
        };

        // Guard against sending without a selected workspace.
        if self.selected_workspace.is_none() {
            tracing::warn!("Home: attempted to send message without a workspace selected");
            return Task::none();
        }

        // Generate an optimistic ID for non-command messages so the Home page
        // can display the user's message immediately and replace it when the
        // pipeline confirmation arrives. Commands (starting with "/") are NOT
        // optimistically shown because `handle_bot_command` intercepts
        // them before the GUI broadcast in `process_channel_message` — the
        // confirmation never arrives, so an optimistic entry would become an orphan.
        let is_command = content.starts_with('/');
        let optimistic_id = if is_command {
            None
        } else {
            Some(crate::generate_id())
        };

        // Clear the editor.
        self.editor_content = text_editor::Content::new();
        self.undo_stack.clear();
        self.sending = true;

        // Push optimistic message immediately so the user sees their own
        // message without waiting for the pipeline round-trip.
        if let Some(ref opt_id) = optimistic_id {
            self.messages.push(display_message(
                None,
                opt_id.clone(),
                content.clone(),
                ChatDirection::User,
                None,
                true,
            ));
        }

        let msg = crate::ChannelMessage {
            user_name: sender.clone(),
            reply_target: sender,
            content,
            channel: "gui".to_string(),
            workspace: self.selected_workspace.clone().unwrap_or_default(),
            optimistic_id,
            callback_query_id: None,
        };

        // Push to GUI_MESSAGE_TX.
        if let Some(tx) = crate::GUI_MESSAGE_TX.get() {
            if let Err(e) = tx.send(msg) {
                tracing::error!("Home: failed to send message via GUI_MESSAGE_TX: {e}");
                self.sending = false;
                return Task::none();
            }
        } else {
            tracing::error!("Home: GUI_MESSAGE_TX not initialized");
            self.sending = false;
            return Task::none();
        }

        // Spawn a safety timeout: if sending stays true for 30 seconds
        // (silent agent failure, crash, cancellation), auto-clear it.
        // Generation counter prevents a stale timeout from clearing
        // sending during a new send.
        self.sending_gen = self.sending_gen.wrapping_add(1);
        let generation = self.sending_gen;
        let timeout_task = Task::perform(
            async move {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                HomeMessage::SendingTimeout(generation)
            },
            |msg| msg,
        );
        // Snap to end on optimistic push if auto-scroll enabled.
        Task::batch([timeout_task, self.maybe_snap()])
    }
}

/// Stream producer for chat events from CHAT_BROADCAST.
fn chat_stream_producer() -> impl futures_util::Stream<Item = HomeMessage> {
    super::common::broadcast_stream_producer(16, &crate::CHAT_BROADCAST, |output, item| {
        let msg = match item {
            Some(event) => HomeMessage::ChatEvent(event),
            None => HomeMessage::StreamLagged,
        };
        Box::pin(async move {
            let _ = output.send(msg).await;
        })
    })
}

/// Emit `TypingTick` every 500ms for the typing indicator animation.
fn typing_tick() -> impl futures_util::Stream<Item = HomeMessage> {
    iced::stream::channel(
        1,
        move |mut output: iced::futures::channel::mpsc::Sender<HomeMessage>| async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if output.send(HomeMessage::TypingTick).await.is_err() {
                    break;
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_home_state(user: &str, workspace: &str) -> HomeState {
        let mut state = HomeState::new();
        state.selected_user = Some(user.to_string());
        state.selected_workspace = Some(workspace.to_string());
        state
    }

    fn make_msg(
        message_id: &str,
        content: &str,
        direction: ChatDirection,
        agent_role: Option<&str>,
        is_optimistic: bool,
    ) -> DisplayMessage {
        DisplayMessage {
            id: None,
            message_id: message_id.to_string(),
            content: content.to_string(),
            direction,
            agent_role: agent_role.map(String::from),
            md_items: Vec::new(),
            is_optimistic,
        }
    }

    // ------------------------------------------------------------------
    // replace_optimistic
    // ------------------------------------------------------------------

    #[test]
    fn test_replace_optimistic_found() {
        let mut state = make_home_state("alice", "ws1");
        state.messages.push(make_msg(
            "opt-1",
            "(placeholder)",
            ChatDirection::User,
            None,
            true,
        ));

        let task = state.replace_optimistic(
            Some("opt-1"),
            "real-42",
            "Hello!",
            ChatDirection::User,
            None,
        );

        assert!(task.is_some(), "expected Some(task) for found optimistic");
        assert_eq!(state.messages.len(), 1);
        let replaced = &state.messages[0];
        assert_eq!(replaced.message_id, "real-42");
        assert_eq!(replaced.content, "Hello!");
        assert!(!replaced.is_optimistic, "should no longer be optimistic");
        assert!(
            state.seen_ids.contains("real-42"),
            "seen_ids should track canonical ID"
        );
        assert!(!state.sending, "sending should be cleared");
    }

    #[test]
    fn test_replace_optimistic_not_found() {
        let mut state = make_home_state("alice", "ws1");
        state.messages.push(make_msg(
            "opt-1",
            "(placeholder)",
            ChatDirection::User,
            None,
            true,
        ));

        // optimistic_id does not match any message
        let task = state.replace_optimistic(
            Some("wrong-opt"),
            "real-42",
            "Hello!",
            ChatDirection::User,
            None,
        );

        assert!(task.is_none(), "expected None when no optimistic match");
        assert_eq!(state.messages.len(), 1);
        assert_eq!(
            state.messages[0].message_id, "opt-1",
            "original should be untouched"
        );
    }

    #[test]
    fn test_replace_optimistic_no_opt_id() {
        let mut state = make_home_state("alice", "ws1");

        let task = state.replace_optimistic(None, "real-42", "Hello!", ChatDirection::User, None);

        assert!(task.is_none(), "expected None when optimistic_id is None");
    }

    // ------------------------------------------------------------------
    // try_dedup
    // ------------------------------------------------------------------

    #[test]
    fn test_try_dedup_fresh() {
        let mut state = make_home_state("alice", "ws1");
        assert!(!state.try_dedup("msg-1"), "fresh ID should return false");
        assert!(state.seen_ids.contains("msg-1"), "fresh ID should be added");
    }

    #[test]
    fn test_try_dedup_duplicate() {
        let mut state = make_home_state("alice", "ws1");
        state.seen_ids.insert("msg-1".to_string());
        assert!(state.try_dedup("msg-1"), "duplicate should return true");
    }

    #[test]
    fn test_try_dedup_pruning() {
        let mut state = make_home_state("alice", "ws1");
        // Add 500 IDs.
        for i in 0..DEDUP_PRUNE_THRESHOLD {
            state.seen_ids.insert(format!("old-{i}"));
        }
        // Push 200 messages so there is a retain pool.
        for i in 0..200u32 {
            state.messages.push(make_msg(
                &format!("old-{i}"),
                "",
                ChatDirection::User,
                None,
                false,
            ));
        }
        // Add one more (breaches the threshold).
        state.seen_ids.insert("extra".to_string());
        assert_eq!(state.seen_ids.len(), 501);

        // Calling try_dedup on a fresh ID triggers pruning.
        assert!(!state.try_dedup("fresh"));

        // After pruning, seen_ids only has the 200 message IDs.
        // "fresh" and "extra" are dropped because they are not in messages.
        assert_eq!(state.seen_ids.len(), 200);
        assert!(!state.seen_ids.contains("fresh"));
        assert!(!state.seen_ids.contains("extra"));
        // An ID that is in messages is retained.
        assert!(state.seen_ids.contains("old-0"));
        assert!(state.seen_ids.contains("old-199"));
    }

    // ------------------------------------------------------------------
    // update_sending_state
    // ------------------------------------------------------------------

    #[test]
    fn test_update_sending_state() {
        let cases = [
            (
                "agent match",
                ChatDirection::Agent,
                "alice",
                "ws1",
                false,
                false,
            ),
            (
                "user match",
                ChatDirection::User,
                "alice",
                "ws1",
                false,
                true,
            ),
            ("wrong user", ChatDirection::Agent, "bob", "ws1", true, true),
            (
                "wrong workspace",
                ChatDirection::Agent,
                "alice",
                "ws2",
                true,
                true,
            ),
        ];
        for (name, direction, user, workspace, exp_sending, exp_typing) in cases {
            let mut state = make_home_state("alice", "ws1");
            state.sending = true;
            state.typing = true;
            state.update_sending_state(direction, user, workspace);
            assert_eq!(state.sending, exp_sending, "{name}: sending");
            assert_eq!(state.typing, exp_typing, "{name}: typing");
        }
    }

    // ------------------------------------------------------------------
    // append_message
    // ------------------------------------------------------------------

    #[test]
    fn test_append_message_match() {
        let mut state = make_home_state("alice", "ws1");
        assert_eq!(state.messages.len(), 0);

        state.append_message(
            "alice",
            "ws1",
            "msg-1".to_string(),
            "Hello!".to_string(),
            ChatDirection::User,
            None,
        );

        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].message_id, "msg-1");
        assert_eq!(state.messages[0].content, "Hello!");
    }

    #[test]
    fn test_append_message_no_match_user() {
        let mut state = make_home_state("alice", "ws1");

        state.append_message(
            "bob",
            "ws1",
            "msg-1".to_string(),
            "Hello!".to_string(),
            ChatDirection::User,
            None,
        );

        assert_eq!(
            state.messages.len(),
            0,
            "bob's message should be filtered out"
        );
    }

    #[test]
    fn test_append_message_no_match_workspace() {
        let mut state = make_home_state("alice", "ws1");

        state.append_message(
            "alice",
            "ws2",
            "msg-1".to_string(),
            "Hello!".to_string(),
            ChatDirection::User,
            None,
        );

        assert_eq!(
            state.messages.len(),
            0,
            "ws2 message should be filtered out"
        );
    }

    #[test]
    fn test_append_message_agent_response() {
        let mut state = make_home_state("alice", "ws1");

        state.append_message(
            "alice",
            "ws1",
            "msg-agent".to_string(),
            "Agent answer".to_string(),
            ChatDirection::Agent,
            Some("engineer".to_string()),
        );

        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].direction, ChatDirection::Agent);
        assert_eq!(state.messages[0].agent_role.as_deref(), Some("engineer"),);
    }

    // ------------------------------------------------------------------
    // entry_to_display_message
    // ------------------------------------------------------------------

    #[test]
    fn test_display_message_from_user() {
        let entry = ChatHistoryEntry {
            id: 1,
            message_id: "msg-1".to_string(),
            user_name: "alice".to_string(),
            content: "Hello **world**".to_string(),
            direction: ChatDirection::User,
            agent_role: None,
        };
        let msg = DisplayMessage::from(entry);

        assert_eq!(msg.id, Some(1));
        assert_eq!(msg.direction, ChatDirection::User);
        assert_eq!(msg.content, "Hello **world**");
        assert!(
            !msg.md_items.is_empty(),
            "user message should produce markdown items"
        );
        assert!(!msg.is_optimistic);
    }

    #[test]
    fn test_display_message_divider() {
        let entry = ChatHistoryEntry {
            id: 42,
            message_id: "divider-1".to_string(),
            user_name: "alice".to_string(),
            content: "2026-07-17T20:30:00Z".to_string(),
            direction: ChatDirection::Divider,
            agent_role: None,
        };
        let msg = DisplayMessage::from(entry);

        assert_eq!(msg.id, Some(42));
        assert_eq!(msg.direction, ChatDirection::Divider);
        assert_eq!(msg.content, "2026-07-17T20:30:00Z");
        // Dividers should produce NO markdown items — they render as rules.
        assert!(
            msg.md_items.is_empty(),
            "divider entry should produce empty markdown items, got {} items",
            msg.md_items.len()
        );
        assert!(!msg.is_optimistic);
    }

    // ------------------------------------------------------------------
    // ModifiersChanged + shift+click
    // ------------------------------------------------------------------

    #[test]
    fn test_modifiers_changed_updates_state() {
        let mut state = make_home_state("alice", "ws1");

        // Default is empty modifiers
        assert!(!state.modifiers.shift());

        // Simulate Shift pressed
        let shift_mods = keyboard::Modifiers::SHIFT;
        let _task = state.update(HomeMessage::ModifiersChanged(shift_mods));

        assert!(state.modifiers.shift());

        // Simulate reset via Unfocused (empty modifiers)
        let _task = state.update(HomeMessage::ModifiersChanged(keyboard::Modifiers::empty()));

        assert!(!state.modifiers.shift());
    }

    #[test]
    fn test_shift_click_converts_to_drag() {
        use iced::Point;

        let mut state = make_home_state("alice", "ws1");
        state.editor_content = text_editor::Content::with_text("hello world");

        // Click somewhere to position cursor. Even without a font system,
        // hit-testing at (0,0) on non-empty text typically resolves to
        // the first cursor position (line 0, col 0).
        state
            .editor_content
            .perform(text_editor::Action::Click(Point { x: 0.0, y: 0.0 }));

        let cursor_before = state.editor_content.cursor();
        // Click clears selection
        assert!(
            cursor_before.selection.is_none(),
            "Click should clear selection"
        );

        // Now hold Shift
        state.modifiers = keyboard::Modifiers::SHIFT;

        // Dispatch a Click at a different position — should be converted to Drag
        let _task = state.update(HomeMessage::InputChanged(text_editor::Action::Click(
            Point { x: 100.0, y: 0.0 },
        )));

        let cursor_after = state.editor_content.cursor();

        // Drag anchors selection at current cursor when none exists. Even if
        // hit-testing at (100, 0) yields the same or no position, the selection
        // should now be Some — verifying the Click→Drag conversion happened.
        assert!(
            cursor_after.selection.is_some(),
            "shift+click should create selection via Action::Drag conversion; \
             got selection={:?}",
            cursor_after.selection
        );
    }

    // ------------------------------------------------------------------
    // send_message
    // ------------------------------------------------------------------

    #[test]
    fn test_send_message_empty_is_noop() {
        let mut state = make_home_state("alice", "ws1");
        // Empty content — should return Task::none() and not change state.
        state.editor_content = text_editor::Content::new();
        let _task = state.send_message();
        assert!(!state.sending);
        assert!(state.editor_content.text().is_empty());

        // Whitespace-only content should also be treated as empty.
        state.editor_content = text_editor::Content::with_text("   ");
        let _task = state.send_message();
        assert!(!state.sending);
    }

    #[test]
    fn test_send_message_within_limit_clears_editor() {
        let mut state = make_home_state("alice", "ws1");
        state.editor_content = text_editor::Content::with_text("hello world");
        let _task = state.send_message();
        // Editor must be cleared before the GUI_MESSAGE_TX send attempt.
        assert!(
            state.editor_content.text().is_empty(),
            "editor should be cleared after accepting a within-limit message"
        );
        // Assert the optimistic push, not `sending` — GUI_MESSAGE_TX is uninitialized in tests.
        assert!(
            state
                .messages
                .iter()
                .any(|m| m.is_optimistic && m.content == "hello world"),
            "optimistic message should be pushed for accepted non-command text"
        );
    }
}
