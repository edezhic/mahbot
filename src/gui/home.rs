//! Home page — native GUI chat interface with user impersonation.
//!
//! Users pick an identity from the user picker, select a workspace via the
//! Dashboard workspace picker, and chat with MahBot agents in real time
//! with full markdown rendering and typing indicators.

use crate::ChatDirection;
use crate::Role;
use crate::channels::chat_history::ChatHistoryEntry;
use futures_util::SinkExt;
use iced::widget::rule;
use iced::widget::{Column, Id, Space, button, column, container, row, scrollable, text, tooltip};
use iced::{Alignment, Element, Length, Task};
use iced_fonts::lucide;
use std::collections::HashSet;

use super::ToastMessage;
use super::common::MAX_INPUT_CHARS;
use super::menus::{ContextMenu, MenuItem, RoleMenu, RoleMenuItem};
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
    /// Timestamp (from chat_history) used for the relative-time label.
    pub timestamp: Option<String>,
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
    timestamp: Option<String>,
    is_optimistic: bool,
) -> DisplayMessage {
    use iced::widget::markdown;
    let md_items: Vec<markdown::Item> = if direction == ChatDirection::Divider {
        Vec::new()
    } else {
        let processed = super::media_markers::preprocess(&content);
        let processed = super::markdown_breaks::hard_breaks(&processed);
        markdown::parse(&processed).collect()
    };
    DisplayMessage {
        id,
        message_id,
        content,
        direction,
        agent_role,
        timestamp,
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
            timestamp,
            ..
        } = entry;
        display_message(
            Some(id),
            message_id,
            content,
            direction,
            agent_role,
            timestamp,
            false,
        )
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
    /// Workspace changed (from Dashboard workspace picker — propagated via Dashboard).
    WorkspaceChanged(Option<String>),
    /// Text editor content changed.
    InputChanged(super::editor_widget::EditorAction),
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
    /// Internal signal: reverse-sync check completed. Carries the user
    /// (staleness guard for fast user switches), the sidebar workspace to
    /// show (the user's DB-stored workspace, normalized — empty means
    /// Personal) and the user's DB-selected project workspace (the merge
    /// partner for the Personal-picker chat view; `None` when unset or
    /// personal). Proceeds with a normal history refresh for the selected user.
    ResolveUserSelected {
        user: String,
        sidebar_ws: Option<String>,
        project_ws: Option<String>,
    },
    /// Refreshed DB-selected project workspace for a user (re-read on
    /// workspace change so the Personal-picker merge partner never goes
    /// stale). Carries `(user, project_workspace)`.
    ProjectWorkspaceRefreshed(String, Option<String>),
    /// Reset session button pressed — reset session and display.
    ClearChat,
    /// Copy a chat message's raw markdown content to the clipboard.
    /// Carries the exact stored/transmitted text (original media markers
    /// included), not the rendered view.
    CopyMessage(String),
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
    /// Mic button clicked — start a voice message recording to the active role.
    StartVoiceRecording,
    /// Recording popup: stop recording, transcribe, and send the voice message.
    StopVoiceRecordingSend,
    /// Recording popup: stop recording and discard the voice message.
    StopVoiceRecordingDiscard,
    /// The scripted onboarding exchange completed (provider configured) — re-run
    /// the onboarding check (now firing the Phase-2 kickoff).
    OnboardingScriptCompleted,
    /// The scripted onboarding input was invalid — the script stays active.
    OnboardingScriptRePrompt,
    /// The Phase-2 kickoff task finished (marker; the work is done inside the task).
    OnboardingKickoffDone,
    /// Toggle the composer role dropdown open/closed.
    RoleMenuToggled,
    /// Close the composer role dropdown (on role selection, message send,
    /// chat clear, or user switch).
    RoleMenuClosed,
}

pub struct HomeState {
    /// Currently selected user (sender identifier).
    pub(crate) selected_user: Option<String>,
    /// Currently selected workspace name (synced from Dashboard workspace
    /// picker). `Some("personal:{user}")` = the user's "Personal" workspace;
    /// `Some("ws")` = a shared workspace. Resolved to `personal:<user_name>`
    /// before querying chat_history or sessions.
    selected_workspace: Option<String>,
    /// The selected user's DB-stored project workspace (None when unset or
    /// personal). The merge partner for the Personal-picker chat view: at
    /// the Personal picker the chat shows this workspace alongside the
    /// user's personal workspace.
    user_project_workspace: Option<String>,
    /// Displayed chat messages.
    messages: Vec<DisplayMessage>,
    /// Deduplication set of seen message IDs.
    seen_ids: HashSet<String>,
    /// Text editor content.
    editor_content: super::editor_widget::EditorBuffer,
    /// Whether a message is currently being sent / agent is responding.
    sending: bool,
    /// Whether a typing indicator is active.
    typing: bool,
    /// Typing animation dot cycle state: 0=".", 1="..", 2="...".
    typing_tick_state: u8,
    /// Whether the initial history load has happened for the current user+workspace.
    history_loaded: bool,
    /// Whether the Phase-1 scripted onboarding scenario is active (shown while
    /// `provider_configured() == false`).
    onboarding_script_active: bool,
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
    /// Whether the composer role dropdown is open.
    role_menu_open: bool,
}

/// The chat view for the selected user: two workspaces — the picker-resolved
/// one plus a merge partner. Symmetric visibility: the picker only selects
/// the recipient, it never filters the view (personal Assistant/Artist
/// messages show at any picker, and the user's project chat shows at the
/// Personal picker).
#[derive(Debug, Clone, PartialEq, Eq)]
struct VisibleChat {
    /// The picker-resolved workspace: the selected workspace, or
    /// `personal:{user}` at the Personal picker.
    primary: String,
    /// The merge partner: `personal:{user}` at a project picker, or the
    /// user's DB-selected project workspace at the Personal picker. `None`
    /// only when the primary is the personal workspace and the user has no
    /// project workspace (deduplicated).
    merge: Option<String>,
}

impl VisibleChat {
    /// Whether `workspace` is part of this visible chat.
    fn contains(&self, workspace: &str) -> bool {
        self.primary == workspace || self.merge.as_deref() == Some(workspace)
    }
}

impl HomeState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            selected_user: None,
            selected_workspace: None,
            user_project_workspace: None,
            messages: Vec::new(),
            seen_ids: HashSet::new(),
            editor_content: super::editor_widget::EditorBuffer::with_text(
                "",
                Some(super::highlight::HighlightLanguage::Markdown),
            ),
            sending: false,
            typing: false,
            typing_tick_state: 0,
            history_loaded: false,
            onboarding_script_active: false,
            sending_gen: 0,
            pending_workspace_refresh: false,
            auto_scroll_enabled: true,
            oldest_loaded_id: None,
            has_more: false,
            loading_older: false,
            pagination_gen: 0,
            undo_stack: super::common::UndoStack::new(),
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
    /// A `personal:{user}` name is used as-is; `None` or an empty string (a
    /// legacy/edge sentinel) → `personal:<user_name>`. Non-personal names are
    /// the workspace name as-is.
    fn resolve_workspace_name(&self) -> Option<String> {
        match &self.selected_workspace {
            Some(w) if !w.is_empty() => Some(w.clone()),
            _ => {
                let user = self.selected_user.as_ref()?;
                Some(crate::users::personal_workspace_name(user))
            }
        }
    }

    /// Whether `workspace` is the selected user's personal workspace
    /// (`personal:{user}`) — the Assistant/Artist chat shown at any picker.
    fn is_selected_user_personal_workspace(&self, workspace: &str) -> bool {
        self.selected_user
            .as_deref()
            .is_some_and(|user| crate::users::personal_user_name(workspace) == Some(user))
    }

    /// The visible chat set for the selected user (see [`VisibleChat`]), or
    /// `None` when no user is selected.
    fn visible_workspaces(&self) -> Option<VisibleChat> {
        let user = self.selected_user.as_ref()?;
        let personal = crate::users::personal_workspace_name(user);
        let chat = match self.resolve_workspace_name() {
            Some(sel) if !self.is_selected_user_personal_workspace(&sel) => VisibleChat {
                primary: sel,
                merge: Some(personal),
            },
            _ => VisibleChat {
                primary: personal,
                merge: self.user_project_workspace.clone(),
            },
        };
        Some(chat)
    }

    /// Whether a message or typing event in `workspace` belongs to the
    /// selected user's visible chat (see [`VisibleChat::contains`]).
    fn workspace_visible(&self, workspace: &str) -> bool {
        self.visible_workspaces()
            .is_some_and(|chat| chat.contains(workspace))
    }

    /// Whether the picker selects the selected user's personal workspace —
    /// projected from [`Self::visible_workspaces`] (its `primary` is the
    /// personal workspace only at the Personal picker), so the merge-partner
    /// decision keeps a single shape across the refresh paths.
    fn at_personal_picker(&self) -> bool {
        self.visible_workspaces()
            .is_some_and(|chat| self.is_selected_user_personal_workspace(&chat.primary))
    }

    /// Reverse-sync the DB-stored workspace preference for a user.
    ///
    /// Returns [`ResolveUserSelected`] carrying the sidebar workspace to show
    /// and the user's DB-selected project workspace. Home's handler emits
    /// [`RequestWorkspaceChange`] when the sidebar must move (Dashboard-level
    /// change cascading to [`WorkspaceChanged`] → `refresh_history`);
    /// otherwise it refreshes history directly.
    ///
    /// NOTE: The Dashboard-level switch emitted here writes the same value back
    /// to the user's DB record via `select_workspace` — idempotent by design,
    /// since the DB is the single source of truth for the selection.
    async fn resolve_user_workspace_sync(user: String, current_ws: Option<String>) -> HomeMessage {
        match crate::users::get_raw_selected_workspace(&user).await {
            Ok(Some(ws_name)) => {
                // User has an explicit stored workspace preference. Normalize
                // a personal stored value to the GUI-wide `personal:{user}` name;
                // the project workspace is the merge partner at the Personal
                // picker.
                let (sidebar_ws, project_ws) = if crate::users::is_personal_workspace(&ws_name) {
                    (crate::users::personal_workspace_name(&user), None)
                } else {
                    (ws_name.clone(), Some(ws_name))
                };
                HomeMessage::ResolveUserSelected {
                    user,
                    sidebar_ws: Some(sidebar_ws),
                    project_ws,
                }
            }
            Ok(None) => {
                // User has no stored preference — keep current sidebar selection.
                HomeMessage::ResolveUserSelected {
                    user,
                    sidebar_ws: current_ws.clone(),
                    project_ws: None,
                }
            }
            Err(e) => {
                tracing::warn!("Failed to get raw workspace for user {user}: {e}");
                HomeMessage::ResolveUserSelected {
                    user,
                    sidebar_ws: current_ws.clone(),
                    project_ws: None,
                }
            }
        }
    }

    /// The user's DB-selected project workspace (None when unset or
    /// personal). The Personal-picker merge partner — re-read on workspace
    /// changes so it never goes stale (Users-page edits flow through
    /// [`WorkspaceChanged`]).
    async fn project_workspace_for(user: String) -> Option<String> {
        crate::users::get_raw_selected_workspace(&user)
            .await
            .ok()
            .flatten()
            .filter(|ws| !crate::users::is_personal_workspace(ws))
    }

    /// Refresh chat history from the store for the current user's visible
    /// workspaces (the selected workspace plus the user's personal workspace).
    fn refresh_history(&self) -> Task<HomeMessage> {
        let user_name = match &self.selected_user {
            Some(s) => s.clone(),
            None => return Task::none(),
        };
        let Some(chat) = self.visible_workspaces() else {
            return Task::none();
        };
        Task::perform(
            async move {
                let store = crate::channels::chat_history::store();
                store
                    .load_for_user_workspaces(&user_name, &chat.primary, chat.merge.as_deref())
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

    /// Reset session display state: messages, dedup set, history flag, pagination.
    fn reset_chat_state(&mut self) {
        self.messages.clear();
        self.seen_ids.clear();
        self.history_loaded = false;
        self.onboarding_script_active = false;
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

    /// Start the Phase-1 scripted onboarding (no provider) or fire the Phase-2
    /// Support kickoff (provider configured, state Init). Called after the first
    /// history load for a selected user. No-op when the scenario is already active.
    fn maybe_start_onboarding(&mut self) -> Task<HomeMessage> {
        if self.selected_user.is_none() {
            return Task::none();
        }
        // NOTE: the Phase-1 provider-setup script is intentionally NOT gated on
        // the selected user being the admin — the onboarding flow is admin-only,
        // but a non-admin created via the Settings bypass before a provider is
        // set would see it and could set the global provider. That edge is out of
        // scope; only the Phase-2 `kickoff_support` is admin-gated (via its
        // `role_pool` Support check) so a non-admin never consumes the state.
        if !crate::config::provider_configured() {
            if self.onboarding_script_active {
                return Task::none();
            }
            let user = self.selected_user.clone().unwrap_or_default();
            let workspace = match self.visible_workspaces() {
                Some(chat) => chat.primary.clone(),
                None => return Task::none(),
            };
            // Only arm the scenario once a workspace is confirmed visible, so an
            // empty picker can't strand `onboarding_script_active` with no script.
            self.onboarding_script_active = true;
            for (i, msg) in crate::onboarding::intro_messages().into_iter().enumerate() {
                crate::channels::broadcast_transient_event(
                    &format!("onboarding-intro-{i}"),
                    &user,
                    msg,
                    crate::ChatDirection::Agent,
                    "gui",
                    Some("support".to_string()),
                    &workspace,
                    None,
                    &crate::db::now(),
                );
            }
            Task::none()
        } else if crate::config::CONFIG.onboarding_stage() == crate::config::OnboardingState::Init {
            self.onboarding_script_active = false;
            let user = self.selected_user.clone().unwrap_or_default();
            let user_for_task = user.clone();
            Task::perform(
                async move { crate::onboarding::kickoff_support(&user_for_task).await },
                |_res| HomeMessage::OnboardingKickoffDone,
            )
        } else {
            self.onboarding_script_active = false;
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
    fn replace_optimistic(
        &mut self,
        optimistic_id: Option<&str>,
        message_id: &str,
        content: &str,
        direction: ChatDirection,
        agent_role: Option<&str>,
        timestamp: Option<&str>,
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
                    timestamp.map(std::string::ToString::to_string),
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
    /// Does nothing when `workspace` is not visible for the selected user
    /// (see [`Self::workspace_visible`]) — this prevents an agent response
    /// from an unrelated workspace from clearing the typing/sending
    /// indicators for the visible chat.
    fn update_sending_state(&mut self, direction: ChatDirection, user_name: &str, workspace: &str) {
        if !self.workspace_visible(workspace) {
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

    /// Append a chat message if it belongs to the selected user's visible
    /// chat (selected workspace or the user's personal workspace).
    ///
    /// Does nothing when `user_name` is not the selected user, or when
    /// `workspace` is not visible (see [`Self::workspace_visible`]).
    /// Takes ownership of the message fields so the caller avoids extra
    /// clones on the common (append) path.
    ///
    /// The caller should call [`maybe_snap()`](Self::maybe_snap)
    /// unconditionally after this (snap is always safe when nothing was
    /// appended).
    #[expect(clippy::too_many_arguments)]
    fn append_message(
        &mut self,
        user_name: &str,
        workspace: &str,
        message_id: String,
        content: String,
        direction: ChatDirection,
        agent_role: Option<String>,
        timestamp: Option<String>,
    ) {
        if Some(user_name) != self.selected_user.as_deref() {
            return;
        }
        if !self.workspace_visible(workspace) {
            return;
        }

        self.messages.push(display_message(
            None, message_id, content, direction, agent_role, timestamp, false,
        ));
    }

    #[expect(clippy::too_many_lines)]
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
                            let mut icon_row = row![icon].align_y(Alignment::Center).spacing(6);
                            if let Some(ts) = msg.timestamp.as_deref() {
                                let label = theme::format_relative_time(ts, chrono::Local::now());
                                if !label.is_empty() {
                                    icon_row = icon_row
                                        .push(text(label).size(11).color(theme::TEXT_SECONDARY));
                                }
                            }
                            column![icon_row, content].spacing(4).into()
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

                    // Per-bubble context menu: right-clicking the bubble offers
                    // copying the raw markdown content (the exact stored text
                    // with original media markers). Wrapping only the bubble —
                    // not the align_bubble row — keeps the spacer beside it a
                    // fall-through: spacer/empty-space right-clicks reach the
                    // outer "Reset session" menu in gui/mod.rs instead.
                    let bubble: Element<'_, HomeMessage> = ContextMenu::new(
                        bubble,
                        vec![MenuItem::new(
                            "Copy message".into(),
                            HomeMessage::CopyMessage(msg.content.clone()),
                        )],
                    )
                    .into();

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
                    "Loading older messages\u{2026}"
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
                // The 4px is the pill's design accent on top of the bubbles'
                // inset (8 pre-rework column padding, now the wrapper's) —
                // kept untrimmed so the pill stays 4px deeper than bubbles.
                children.insert(0, container(load_btn).padding(4).into());
            }

            container(super::widgets::vscroll_tracked(
                Column::with_children(children)
                    .spacing(super::widgets::CHAT_VERTICAL_RHYTHM)
                    .padding([8, 0]),
                Length::Fill,
                Length::Fill,
                CHAT_SCROLL_ID,
                HomeMessage::ScrollChanged,
            ))
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
        // With local transcription disabled the shared ASR model never loads,
        // so a mic-button recording can never start — present the control as
        // unavailable instead of a loading state that can never complete.
        let transcription_disabled = crate::audio::voice::is_transcription_disabled();
        let recording_unavailable = mic_busy || transcription_disabled;

        // Controls for the composer action toolbar: role selector + mic button.
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
        // Personal-workspace picker hides the Manager role — routing maps
        // Manager→Assistant there, so offering it in the menu is misleading.
        // Use the filtered pool both for the button gate and the menu items.
        let filtered_pool: Vec<Role> = role_pool
            .iter()
            .filter(|role| **role != Role::Manager || !self.at_personal_picker())
            .copied()
            .collect();
        let role_btn = button(role_icon)
            .on_press_maybe(
                (self.selected_user.is_some() && !filtered_pool.is_empty())
                    .then_some(HomeMessage::RoleMenuToggled),
            )
            .style(theme::icon_button_style(false))
            .padding(3);

        // ── Role dropdown (overlay, above the composer) ────────────
        // The role list floats above the whole widget tree via
        // `RoleMenu`'s overlay (Widget::overlay + Overlay trait) anchored
        // to the role button — opening it no longer shifts the chat
        // layout. Items carry the same roles/selection as before; the
        // current role is disabled with a checkmark, and selecting a role
        // publishes SwitchRole, which the Dashboard intercepts and persists
        // (it also sends RoleMenuClosed). Outside-click / Escape dismissal
        // is handled by the popup itself via RoleMenuClosed.
        let role_btn: Element<'_, HomeMessage> = if !filtered_pool.is_empty() {
            let user = self.selected_user.clone().unwrap_or_default();
            let items: Vec<RoleMenuItem<HomeMessage>> = filtered_pool
                .iter()
                .map(|role| {
                    let is_current = active_role.as_ref() == Some(role);
                    RoleMenuItem::new(
                        *role,
                        is_current,
                        (!is_current).then(|| HomeMessage::SwitchRole(user.clone(), *role)),
                    )
                })
                .collect();
            RoleMenu::new(
                role_btn,
                items,
                self.role_menu_open,
                HomeMessage::RoleMenuClosed,
            )
            .into()
        } else {
            role_btn.into()
        };
        controls.push(
            tooltip(
                role_btn,
                text("switch agent").size(11),
                tooltip::Position::Top,
            )
            .style(theme::tooltip_style)
            .into(),
        );

        let mic_btn = tooltip(
            button(lucide::mic::<iced::Theme, iced::Renderer>().size(14).color(
                if recording_unavailable {
                    theme::TEXT_MUTED
                } else {
                    theme::TEXT_SECONDARY
                },
            ))
            .on_press_maybe(
                (self.selected_user.is_some() && !recording_unavailable)
                    .then_some(HomeMessage::StartVoiceRecording),
            )
            .style(theme::icon_button_style(recording_unavailable))
            .padding(3),
            text(if transcription_disabled {
                "voice recording unavailable — local transcription is disabled"
            } else {
                "record voice message"
            })
            .size(11),
            tooltip::Position::Top,
        )
        .style(theme::tooltip_style);
        controls.push(mic_btn.into());

        // Composer strip matches the BG_BASE chat pane so the empty space around
        // the rounded bubble blends with the page instead of showing a gray panel;
        // the 8px horizontal padding matches the chat scrollable wrapper's inset,
        // and the bottom padding matches the shared chat vertical rhythm.
        // The bubble itself keeps its own elevated styling.
        let input_area: Element<'_, HomeMessage> = container(super::widgets::chat_composer(
            &self.editor_content,
            HomeMessage::InputChanged,
            HomeMessage::SendMessage,
            "Type a message... (Enter to send, Shift+Enter for newline)",
            super::widgets::ChatComposerOptions {
                // Input disabled during the graceful drain:
                // sends are blocked while draining.
                sending: self.sending || draining,
                // The action toolbar sits below the editor, so no reserved
                // height is needed.
                min_height: 44.0,
                max_height: 330.0,
                controls,
                grey_on_empty: true,
                send_tooltip: "send text message",
                id: Some(Id::new("home_chat_composer")),
            },
        ))
        .style(theme::base_container_style)
        .padding(iced::Padding::from([0, 8]).bottom(super::widgets::CHAT_VERTICAL_RHYTHM))
        .width(Length::Fill)
        .into();

        // ── Recording popup (stop + send / stop + discard) ───────
        // While transcribing, the popup stays visible as a passive
        // "Transcribing…" indicator (no stop controls — the ASR is finalizing).
        let recording_popup: Element<'_, HomeMessage> = if recording {
            let status_label = text("Recording voice message…")
                .size(13)
                .color(theme::STATUS_ERROR);
            let send_btn = button(text("Stop + Send").size(12))
                .on_press(HomeMessage::StopVoiceRecordingSend)
                .style(theme::button_secondary)
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
        column![chat_area, recording_popup, input_area,]
            .align_x(Alignment::End)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    pub fn subscription(&self) -> iced::Subscription<HomeMessage> {
        let mut subs = vec![iced::Subscription::run(chat_stream_producer)];

        // The typing-indicator tick only advances while the user is actually
        // typing; subscribe it only then to avoid waking the runtime at 2Hz
        // when idle. The handler already no-ops when `self.typing` is false.
        if self.typing {
            subs.push(iced::Subscription::run(typing_tick));
        }

        iced::Subscription::batch(subs)
    }

    #[expect(clippy::too_many_lines)]
    pub fn update(&mut self, msg: HomeMessage) -> Task<HomeMessage> {
        match msg {
            HomeMessage::UserSelected(user) => {
                if self.selected_user.as_deref() == Some(&user) {
                    return Task::none();
                }
                self.selected_user = Some(user.clone());
                self.reset_chat_state();
                self.user_project_workspace = None; // re-resolved below

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
                let Some(user) = self.selected_user.clone() else {
                    self.pending_workspace_refresh = true;
                    return Task::none();
                };
                self.pending_workspace_refresh = false;
                // At the Personal picker the view merges the user's DB-selected
                // project workspace — re-read it so Users-page edits (which
                // flow through WorkspaceChanged) are reflected, then load in
                // one shot. At a project picker the merge partner is the
                // selected workspace itself, so a direct refresh suffices.
                if self.at_personal_picker() {
                    let read_user = user.clone();
                    Task::perform(Self::project_workspace_for(read_user), move |project| {
                        HomeMessage::ProjectWorkspaceRefreshed(user, project)
                    })
                } else {
                    self.refresh_history()
                }
            }
            HomeMessage::ProjectWorkspaceRefreshed(user, project) => {
                // Stale resolve (user switched while reading) — the newer
                // selection owns its own resolution.
                if self.selected_user.as_deref() != Some(&user) {
                    return Task::none();
                }
                let at_personal_picker = self.at_personal_picker();
                self.user_project_workspace = project;
                // Re-load only while the merge partner is part of the view; a
                // later project-picker switch already refreshed.
                if at_personal_picker {
                    self.refresh_history()
                } else {
                    Task::none()
                }
            }
            HomeMessage::InputChanged(action) => {
                super::common::apply_editor_action(
                    &mut self.editor_content,
                    &mut self.undo_stack,
                    action,
                );
                Task::none()
            }
            HomeMessage::ResolveUserSelected {
                user,
                sidebar_ws,
                project_ws,
            } => {
                // Stale resolve (user switched while reading) — the newer
                // selection owns its own resolution.
                if self.selected_user.as_deref() != Some(&user) {
                    return Task::none();
                }
                // The user's DB-selected project workspace is the merge
                // partner for the Personal-picker chat view.
                self.user_project_workspace = project_ws;
                if sidebar_ws != self.selected_workspace {
                    // Sidebar must move to the user's DB workspace — the
                    // Dashboard intercepts RequestWorkspaceChange and cascades
                    // a WorkspaceChanged refresh.
                    return Task::done(HomeMessage::RequestWorkspaceChange(
                        sidebar_ws.unwrap_or_default(),
                    ));
                }
                // Reverse-sync check completed: either the user's DB workspace
                // matches the sidebar (no disagreement), or no DB workspace
                // exists for this user.
                self.selected_workspace = sidebar_ws;
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
                let onboarding = self.maybe_start_onboarding();
                Task::batch([self.maybe_snap(), onboarding])
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
                Task::perform(
                    async move {
                        // Clear the session the user actually talks to — the
                        // same (role, workspace) resolution as routing and
                        // Telegram /clear (see
                        // [`crate::users::resolve_session_target`]): the
                        // starting workspace is the user's DB workspace, never
                        // the GUI picker position, so the Personal picker
                        // clears the project Manager conversation instead of
                        // a phantom Analyst session in the personal workspace.
                        let (effective_role, ws) =
                            crate::users::resolve_session_target(&sender).await;
                        let _ = crate::session::clear_session(
                            &sender,
                            effective_role.as_str(),
                            &ws.name,
                        )
                        .await;
                        // Insert a divider marker instead of deleting history.
                        let store = crate::channels::chat_history::store();
                        match store.insert_divider(&sender, &ws.name).await {
                            Ok(()) => Ok(()),
                            Err(e) => {
                                tracing::warn!(
                                    user = %sender,
                                    workspace = %ws.name,
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
            HomeMessage::CopyMessage(content) => {
                // Raw markdown copy — no toast, matching the editor's
                // copy-path context-menu actions.
                iced::clipboard::write(content)
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
                    timestamp,
                    channel: _,
                    agent_role,
                    workspace,
                    optimistic_id,
                    ..
                } => {
                    // 1. Replace optimistic placeholder if present.
                    if let Some(task) = self.replace_optimistic(
                        optimistic_id.as_deref(),
                        &message_id,
                        &content,
                        direction,
                        agent_role.as_deref(),
                        Some(timestamp.as_str()),
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
                        &user_name,
                        &workspace,
                        message_id,
                        content,
                        direction,
                        agent_role,
                        Some(timestamp),
                    );

                    self.maybe_snap()
                }
                crate::ChatEvent::Typing {
                    user_name,
                    is_typing,
                    workspace,
                } => {
                    // Apply user + workspace filter — only show typing indicator
                    // for the selected user in a visible workspace.
                    if Some(&user_name) == self.selected_user.as_ref()
                        && self.workspace_visible(&workspace)
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
                let Some(chat) = self.visible_workspaces() else {
                    return Task::none();
                };
                let Some(before_id) = self.oldest_loaded_id else {
                    self.loading_older = false;
                    return Task::none();
                };
                let generation = self.pagination_gen;
                Task::perform(
                    async move {
                        let store = crate::channels::chat_history::store();
                        store
                            .load_older_for_user_workspaces(
                                &sender,
                                &chat.primary,
                                chat.merge.as_deref(),
                                before_id,
                            )
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
            HomeMessage::Toast(_) | HomeMessage::RequestWorkspaceChange(_) => {
                // Intercepted by the Dashboard (Toast → toast stack,
                // RequestWorkspaceChange → sidebar switch). No-op fallback.
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
            HomeMessage::OnboardingScriptCompleted => {
                self.sending = false;
                self.onboarding_script_active = false;
                self.maybe_start_onboarding()
            }
            HomeMessage::OnboardingScriptRePrompt => {
                self.sending = false;
                Task::none()
            }
            HomeMessage::OnboardingKickoffDone => Task::none(),
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
        self.editor_content.clear();
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
                None,
                true,
            ));
        }

        // Phase-1 scripted onboarding intercepts the send before the pipeline:
        // the provider entry is parsed and persisted directly, and the
        // exchange is rendered as transient (never persisted) events.
        if self.onboarding_script_active {
            return self.send_scripted_message(content, optimistic_id);
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

    /// Handle a Phase-1 scripted submit: broadcast the user's message
    /// (transient, replacing the optimistic bubble), persist the provider
    /// input, broadcast the success or re-prompt message, then emit the
    /// completion/re-prompt message.
    fn send_scripted_message(
        &mut self,
        content: String,
        optimistic_id: Option<String>,
    ) -> Task<HomeMessage> {
        let user = match &self.selected_user {
            Some(u) => u.clone(),
            None => return Task::none(),
        };
        let workspace = match self.visible_workspaces() {
            Some(chat) => chat.primary.clone(),
            None => return Task::none(),
        };
        let opt = optimistic_id;
        let send_task = Task::perform(
            async move {
                let parsed = crate::onboarding::parse_provider_input(&content);
                // Replace the optimistic bubble with the user's transient message.
                crate::channels::broadcast_transient_event(
                    &crate::generate_id(),
                    &user,
                    &content,
                    crate::ChatDirection::User,
                    "gui",
                    None,
                    &workspace,
                    opt,
                    &crate::db::now(),
                );
                let outcome: Result<(), String> = match parsed {
                    crate::onboarding::ProviderInput::Invalid => {
                        crate::channels::broadcast_transient_event(
                            &crate::generate_id(),
                            &user,
                            crate::onboarding::invalid_message(),
                            crate::ChatDirection::Agent,
                            "gui",
                            Some("support".to_string()),
                            &workspace,
                            None,
                            &crate::db::now(),
                        );
                        Err("invalid provider input".to_string())
                    }
                    valid => match crate::onboarding::persist_provider_input(&valid).await {
                        Ok(()) => {
                            crate::channels::broadcast_transient_event(
                                &crate::generate_id(),
                                &user,
                                crate::onboarding::success_message(),
                                crate::ChatDirection::Agent,
                                "gui",
                                Some("support".to_string()),
                                &workspace,
                                None,
                                &crate::db::now(),
                            );
                            Ok(())
                        }
                        Err(e) => {
                            crate::channels::broadcast_transient_event(
                                &crate::generate_id(),
                                &user,
                                &format!("Couldn't save that: {e:#}"),
                                crate::ChatDirection::Agent,
                                "gui",
                                Some("support".to_string()),
                                &workspace,
                                None,
                                &crate::db::now(),
                            );
                            Err(e.to_string())
                        }
                    },
                };
                match outcome {
                    Ok(()) => HomeMessage::OnboardingScriptCompleted,
                    Err(_) => HomeMessage::OnboardingScriptRePrompt,
                }
            },
            std::convert::identity,
        );
        // Arm the 30s SendingTimeout guard (same as the normal send path) so a
        // stalled persist can't leave `sending=true` forever.
        self.sending_gen = self.sending_gen.wrapping_add(1);
        let generation = self.sending_gen;
        let timeout_task = Task::perform(
            async move {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                HomeMessage::SendingTimeout(generation)
            },
            |msg| msg,
        );
        Task::batch([send_task, timeout_task])
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
    use super::super::editor_widget::{EditorAction, EditorBuffer};
    use super::super::highlight::HighlightLanguage;
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
            timestamp: None,
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

        let task =
            state.replace_optimistic(None, "real-42", "Hello!", ChatDirection::User, None, None);

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
            None,
        );

        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].direction, ChatDirection::Agent);
        assert_eq!(state.messages[0].agent_role.as_deref(), Some("engineer"),);
    }

    // ------------------------------------------------------------------
    // personal-workspace visibility (Assistant/Artist at any picker)
    // ------------------------------------------------------------------

    #[test]
    fn test_visible_chat_symmetric_at_any_picker() {
        // Project picker: project + personal visible; unrelated stays hidden.
        let project = make_home_state("alice", "ws1");
        assert_eq!(
            project.visible_workspaces(),
            Some(VisibleChat {
                primary: "ws1".to_string(),
                merge: Some("personal:alice".to_string()),
            })
        );
        assert!(project.workspace_visible("ws1"));
        assert!(
            project.workspace_visible("personal:alice"),
            "personal Assistant/Artist messages must be visible at any picker"
        );
        assert!(!project.workspace_visible("ws2"));
        assert!(
            !project.workspace_visible("personal:bob"),
            "another user's personal workspace is not visible"
        );

        // Personal picker: personal + the user's DB project workspace visible
        // (symmetric — the picker selects the recipient, not the view).
        let mut personal = make_home_state("alice", "");
        personal.user_project_workspace = Some("ws1".to_string());
        assert_eq!(
            personal.resolve_workspace_name().as_deref(),
            Some("personal:alice"),
            "empty picker selection resolves to the personal workspace"
        );
        assert_eq!(
            personal.visible_workspaces(),
            Some(VisibleChat {
                primary: "personal:alice".to_string(),
                merge: Some("ws1".to_string()),
            }),
            "personal picker merges the user's project workspace"
        );
        assert!(personal.workspace_visible("personal:alice"));
        assert!(
            personal.workspace_visible("ws1"),
            "project messages must be visible at the personal picker"
        );
        assert!(
            !personal.workspace_visible("ws2"),
            "a non-selected project workspace stays hidden at the personal picker"
        );

        // No DB project workspace → personal-only view; no user → no chat.
        let personal_only = make_home_state("alice", "");
        assert_eq!(
            personal_only.visible_workspaces(),
            Some(VisibleChat {
                primary: "personal:alice".to_string(),
                merge: None,
            }),
            "personal picker without a project workspace shows only the personal chat"
        );
        assert!(!personal_only.workspace_visible("ws1"));
        assert_eq!(HomeState::new().visible_workspaces(), None);
    }

    #[test]
    fn test_workspace_changed_at_personal_picker_resets_and_defers() {
        // Switching to the Personal picker resets the chat and re-reads the
        // user's DB project workspace (completion via ProjectWorkspaceRefreshed
        // is covered by test_project_workspace_refreshed_updates_merge_partner).
        let mut state = make_home_state("alice", "ws1");
        state.user_project_workspace = Some("ws1".to_string());
        state
            .messages
            .push(make_msg("m1", "hi", ChatDirection::User, None, false));

        let _task = state.update(HomeMessage::WorkspaceChanged(Some(
            "personal:alice".to_string(),
        )));
        assert_eq!(state.selected_workspace.as_deref(), Some("personal:alice"));
        assert!(
            state.messages.is_empty(),
            "chat state resets on workspace change"
        );
        assert_eq!(
            state.resolve_workspace_name().as_deref(),
            Some("personal:alice"),
            "personal picker selection resolves to the personal workspace"
        );

        // A project-picker change refreshes history directly and keeps the
        // merge partner untouched (it only matters at the Personal picker).
        let mut state = make_home_state("alice", "ws1");
        state.user_project_workspace = Some("ws1".to_string());
        let _task = state.update(HomeMessage::WorkspaceChanged(Some("ws2".to_string())));
        assert_eq!(state.selected_workspace.as_deref(), Some("ws2"));
        assert_eq!(state.user_project_workspace.as_deref(), Some("ws1"));
    }

    #[test]
    fn test_project_workspace_refreshed_updates_merge_partner() {
        // At the Personal picker the refreshed DB project workspace drives
        // the view (the wiring that keeps the merge partner fresh after a
        // Users-page workspace edit).
        let mut state = make_home_state("alice", "");
        state.user_project_workspace = Some("ws1".to_string());

        // A stale read for a different user is ignored.
        let _ = state.update(HomeMessage::ProjectWorkspaceRefreshed(
            "bob".to_string(),
            Some("ws9".to_string()),
        ));
        assert_eq!(state.user_project_workspace.as_deref(), Some("ws1"));

        // The current user's refreshed value applies and the view follows.
        let _ = state.update(HomeMessage::ProjectWorkspaceRefreshed(
            "alice".to_string(),
            Some("ws2".to_string()),
        ));
        assert_eq!(state.user_project_workspace.as_deref(), Some("ws2"));
        assert!(state.workspace_visible("ws2"));
        assert!(!state.workspace_visible("ws1"));
        assert_eq!(
            state.visible_workspaces(),
            Some(VisibleChat {
                primary: "personal:alice".to_string(),
                merge: Some("ws2".to_string()),
            })
        );

        // At a project picker the refreshed value is stored but the view is
        // the selected workspace (no reload needed there).
        let mut state = make_home_state("alice", "ws1");
        state.user_project_workspace = Some("ws1".to_string());
        let _ = state.update(HomeMessage::ProjectWorkspaceRefreshed(
            "alice".to_string(),
            Some("ws2".to_string()),
        ));
        assert_eq!(state.user_project_workspace.as_deref(), Some("ws2"));
        assert_eq!(
            state.visible_workspaces(),
            Some(VisibleChat {
                primary: "ws1".to_string(),
                merge: Some("personal:alice".to_string()),
            })
        );
    }

    #[test]
    fn test_resolve_user_selected_stale_user_guard() {
        // A resolve for a user that is no longer selected must not apply
        // (fast user switch while the DB read was in flight) — same guard as
        // ProjectWorkspaceRefreshed.
        let mut state = make_home_state("alice", "");
        let _ = state.update(HomeMessage::ResolveUserSelected {
            user: "bob".to_string(),
            sidebar_ws: Some("ws_bob".to_string()),
            project_ws: Some("ws_bob".to_string()),
        });
        assert_eq!(state.selected_user.as_deref(), Some("alice"));
        assert_eq!(state.selected_workspace.as_deref(), Some(""));
        assert_eq!(state.user_project_workspace, None);

        // The current user's resolve applies the merge partner.
        let _ = state.update(HomeMessage::ResolveUserSelected {
            user: "alice".to_string(),
            sidebar_ws: Some("ws1".to_string()),
            project_ws: Some("ws1".to_string()),
        });
        assert_eq!(state.user_project_workspace.as_deref(), Some("ws1"));
    }

    #[tokio::test]
    async fn test_project_workspace_for_reads_db_normalized() {
        crate::util::test::init_test_stores().await;
        let user = "home_project_workspace_for";
        let store = crate::users::USER_STORE
            .get()
            .expect("users store initialized");
        store
            .add_user(user, Some("full"), Role::Manager)
            .await
            .expect("add user");

        // Unset → None.
        assert_eq!(
            HomeState::project_workspace_for(user.to_string()).await,
            None
        );

        // Personal DB workspace → None.
        store
            .update_user(
                user,
                crate::users::FieldUpdate::Unchanged,
                crate::users::FieldUpdate::Set("personal:home_project_workspace_for"),
                crate::users::FieldUpdate::Unchanged,
            )
            .await
            .expect("set personal workspace");
        assert_eq!(
            HomeState::project_workspace_for(user.to_string()).await,
            None
        );

        // Project DB workspace → Some(ws).
        crate::util::test::create_test_workspace(
            "/tmp/home_project_workspace_for_ws",
            "ws_home_project_workspace_for",
        )
        .await;
        store
            .update_user(
                user,
                crate::users::FieldUpdate::Unchanged,
                crate::users::FieldUpdate::Set("ws_home_project_workspace_for"),
                crate::users::FieldUpdate::Unchanged,
            )
            .await
            .expect("set project workspace");
        assert_eq!(
            HomeState::project_workspace_for(user.to_string())
                .await
                .as_deref(),
            Some("ws_home_project_workspace_for")
        );
    }

    #[test]
    fn test_call_site_wiring_symmetric_at_any_picker() {
        // (picker, message workspace, agent role, content) — both directions:
        // at the project picker personal Assistant/Artist messages append and
        // clear sending/typing; at the personal picker the same holds for
        // Manager replies in the user's DB project workspace.
        let cases = [
            ("ws1", "personal:alice", "assistant", "Assistant reply"),
            ("", "ws1", "manager", "Manager reply"),
        ];
        for (picker, msg_ws, role, content) in cases {
            let mut state = make_home_state("alice", picker);
            state.user_project_workspace = Some("ws1".to_string());
            state.append_message(
                "alice",
                msg_ws,
                "msg".to_string(),
                content.to_string(),
                ChatDirection::Agent,
                Some(role.to_string()),
                None,
            );
            assert_eq!(state.messages.len(), 1);
            assert_eq!(state.messages[0].content, content);
            state.sending = true;
            state.typing = true;
            state.update_sending_state(ChatDirection::Agent, "alice", msg_ws);
            assert!(!state.sending);
            assert!(!state.typing);
        }

        // A project workspace that is not the user's own stays hidden at the
        // personal picker.
        let mut state = make_home_state("alice", "");
        state.user_project_workspace = Some("ws1".to_string());
        state.append_message(
            "alice",
            "ws2",
            "msg-other".to_string(),
            "other".to_string(),
            ChatDirection::Agent,
            None,
            None,
        );
        assert_eq!(
            state.messages.len(),
            0,
            "messages from an unrelated workspace must not append at the personal picker"
        );
    }

    // ------------------------------------------------------------------
    // entry_to_display_message
    // ------------------------------------------------------------------

    #[test]
    fn test_display_message_from_user() {
        let entry = ChatHistoryEntry {
            id: 1,
            message_id: "msg-1".to_string(),
            content: "Hello **world**".to_string(),
            direction: ChatDirection::User,
            agent_role: None,
            timestamp: None,
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
            content: "2026-07-17T20:30:00Z".to_string(),
            direction: ChatDirection::Divider,
            agent_role: None,
            timestamp: None,
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
    // Selection (shift+click / drag)
    // ------------------------------------------------------------------

    #[test]
    fn test_select_to_creates_selection() {
        let mut state = make_home_state("alice", "ws1");
        state.editor_content =
            EditorBuffer::with_text("hello world", Some(HighlightLanguage::Markdown));

        // Move the cursor away from the anchor, then extend a selection.
        state
            .editor_content
            .perform_action(EditorAction::MoveTo { line: 0, col: 0 });
        let cursor_before = state.editor_content.cursor();
        assert!(
            cursor_before.selection.is_none(),
            "MoveTo should clear selection"
        );

        // Dispatch a SelectTo through the page update.
        let _task = state.update(HomeMessage::InputChanged(EditorAction::SelectTo {
            line: 0,
            col: 5,
        }));

        let cursor_after = state.editor_content.cursor();
        assert!(
            cursor_after.selection.is_some(),
            "SelectTo should create a selection; got selection={:?}",
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
        state.editor_content.clear();
        let _task = state.send_message();
        assert!(!state.sending);
        assert!(state.editor_content.text().is_empty());

        // Whitespace-only content should also be treated as empty.
        state.editor_content = EditorBuffer::with_text("   ", Some(HighlightLanguage::Markdown));
        let _task = state.send_message();
        assert!(!state.sending);
    }

    #[test]
    fn test_send_message_within_limit_clears_editor() {
        let mut state = make_home_state("alice", "ws1");
        state.editor_content =
            EditorBuffer::with_text("hello world", Some(HighlightLanguage::Markdown));
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
