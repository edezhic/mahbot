//! Home page — native GUI chat interface for the app's own admin identity.
//!
//! The page acts as the seeded admin, selects a workspace via the footer
//! workspace picker, and chats with MahBot agents in real time with full
//! markdown rendering and typing indicators.

use crate::ChatDirection;
use crate::Role;
use crate::channels::ReplyReference;
use crate::channels::chat_history::ChatHistoryEntry;
use crate::channels::reply::{agent_author_label, normalize_reply_text};
use futures_util::SinkExt;
use iced::widget::rule;
use iced::widget::{Column, Id, Space, button, column, container, row, scrollable, text, tooltip};
use iced::{Alignment, Element, Length, Task};
use iced_fonts::lucide;
use std::collections::HashSet;
use std::sync::Arc;

use super::ToastMessage;
use super::common::MAX_INPUT_CHARS;
use super::menus::{ContextMenu, MenuItem};
use super::theme;
use super::widgets::{BubbleSide, align_bubble};

/// Dedup-set prune trigger: when `seen_ids` grows past this many IDs, it is
/// pruned down to the most recent [`DEDUP_RETAIN`] IDs.
const DEDUP_PRUNE_THRESHOLD: usize = 500;

/// Number of most-recent message IDs retained after a dedup-set prune.
const DEDUP_RETAIN: usize = 200;

/// Debounce window (ms) after a text/reply change before the draft is
/// persisted to disk.
const DRAFT_SETTLE_MS: u64 = 700;

/// Scrollable ID for the chat message list, used for snap-to-end after
/// history loads.
pub(super) const CHAT_SCROLL_ID: Id = Id::new("home_chat_scroll");

/// Focus id for the chat composer editor, used to refocus it after selecting
/// a reply target.
const CHAT_COMPOSER_ID: Id = Id::new("home_chat_composer");

/// A displayed chat message in the scroll view.
#[derive(Debug, Clone)]
struct DisplayMessage {
    /// Database row ID (Some for history-loaded, None for live arrivals).
    pub id: Option<i64>,
    pub message_id: String,
    pub content: String,
    pub direction: ChatDirection,
    pub agent_role: Option<String>,
    /// Timestamp (from chat_history) used for the relative-time label.
    pub timestamp: Option<String>,
    /// Reference to the message this message replies to, rendered as a quote
    /// header. `None` for messages with no reply target.
    pub reply_reference: Option<ReplyReference>,
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
#[expect(clippy::too_many_arguments)]
fn display_message(
    id: Option<i64>,
    message_id: String,
    content: String,
    direction: ChatDirection,
    agent_role: Option<String>,
    timestamp: Option<String>,
    reply_reference: Option<ReplyReference>,
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
        reply_reference,
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
            reply_reference,
            ..
        } = entry;
        display_message(
            Some(id),
            message_id,
            content,
            direction,
            agent_role,
            timestamp,
            reply_reference,
            false,
        )
    }
}

/// Broadcast a transient GUI chat event, wrapping
/// [`crate::channels::broadcast_transient_event`] with the constants shared by
/// every GUI call site: the `"gui"` channel, no reply reference, and a fresh
/// timestamp. Callers pass the user and the workspace of the visible chat.
fn broadcast_gui_transient(
    user: &str,
    workspace: &str,
    message_id: &str,
    content: &str,
    direction: ChatDirection,
    agent_role: Option<String>,
    optimistic_id: Option<String>,
) {
    crate::channels::broadcast_transient_event(
        message_id,
        user,
        content,
        direction,
        "gui",
        agent_role,
        workspace,
        optimistic_id,
        None,
        &crate::db::now(),
    );
}

/// Derive the [`ReplyReference`] captured for a reply-to action. User messages
/// are the admin's own, so they resolve to the admin's canonical name; agent
/// messages use the shared author-label derivation. Author and snippet both get
/// angle brackets stripped (the snippet via [`normalize_reply_text`], which also
/// HTML-decodes, collapses newlines, maps media markers, and caps the text).
#[must_use]
fn reply_reference_for(
    direction: ChatDirection,
    agent_role: Option<&str>,
    content: &str,
) -> ReplyReference {
    let author = if direction == ChatDirection::User {
        crate::users::ADMIN_USER_NAME.to_string()
    } else {
        agent_author_label(agent_role)
    };
    ReplyReference {
        author: author.replace(['<', '>'], ""),
        snippet: normalize_reply_text(content),
    }
}

/// Shared lead row of both reply renderings: reply glyph, author (caller-chosen
/// color), and muted snippet, already spaced and center-aligned. Callers may
/// `push` extra trailing elements (e.g. the dismiss button).
fn reply_lead_row(
    reply: &ReplyReference,
    author_color: iced::Color,
) -> iced::widget::Row<'_, HomeMessage> {
    row![
        lucide::reply::<iced::Theme, iced::Renderer>()
            .size(theme::TEXT_12)
            .color(theme::ACCENT),
        text(&reply.author).size(theme::TEXT_11).color(author_color),
        text(&reply.snippet)
            .size(theme::TEXT_11)
            .color(theme::TEXT_MUTED),
    ]
    .spacing(theme::SPACE_4)
    .align_y(Alignment::Center)
}

/// Render the compact reply quote header shown above a bubble that carries a
/// [`ReplyReference`]: a muted "_author_: snippet" line with a small reply
/// glyph, aligned to the row the bubble occupies.
fn reply_quote_header(reply: &ReplyReference) -> Element<'_, HomeMessage> {
    reply_lead_row(reply, theme::TEXT_SECONDARY).into()
}

/// Render the reply preview bar shown between the message list and the
/// composer when a reply reference is pending: author + snippet on one line
/// (the snippet's own cap produces the ellipsis) with a dismiss control.
fn reply_preview(reply: &ReplyReference) -> Element<'_, HomeMessage> {
    let dismiss = button(
        lucide::x::<iced::Theme, iced::Renderer>()
            .size(theme::TEXT_12)
            .color(theme::TEXT_MUTED),
    )
    .on_press(HomeMessage::CancelReply)
    .style(theme::icon_button_style(false))
    .padding(theme::PAD_2);
    container(
        reply_lead_row(reply, theme::TEXT_PRIMARY)
            .push(Space::new().width(Length::Fill))
            .push(dismiss),
    )
    .padding([theme::PAD_4, theme::PAD_8])
    .style(theme::surface_container_style)
    .width(Length::Fill)
    .into()
}

#[derive(Debug, Clone)]
pub enum HomeMessage {
    /// Workspace changed (from the footer workspace picker — propagated via Dashboard).
    WorkspaceChanged(Option<String>),
    /// Text editor content changed.
    InputChanged(super::editor_widget::EditorAction),
    /// Send button pressed or Enter key in editor.
    SendMessage,
    /// Chat history loaded from the store (entries, has_more), carrying the
    /// transcript's primary workspace as the read key: a successful read for a
    /// transcript the page has since left must not replace the current one (nor
    /// clear the failure of the one it left behind).
    HistoryLoaded {
        entries: Vec<ChatHistoryEntry>,
        has_more: bool,
        key: String,
    },
    /// History load failed. Carries the key (primary workspace) the read was
    /// issued under, so a failure for a transcript the page has since left is
    /// dropped.
    HistoryLoadError { key: String, error: String },
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
    /// Older history load failed (error, pagination_gen for staleness check).
    OlderHistoryLoadError(String, u64),
    /// Markdown link was clicked.
    LinkClicked(String),
    /// Refreshed DB-selected project workspace (re-read on workspace change so
    /// the Personal-picker merge partner never goes stale).
    ProjectWorkspaceRefreshed(Option<String>),
    /// Reset session button pressed — reset session and display.
    ClearChat,
    /// Copy a chat message's raw markdown content to the clipboard.
    /// Carries the exact stored/transmitted text (original media markers
    /// included), not the rendered view.
    CopyMessage(String),
    /// Reply to a displayed chat message: capture its author + snippet as the
    /// pending reply reference and focus the composer. Carries the message
    /// identity needed to derive author/snippet (direction/agent_role/content)
    /// without any id-based resolution — deliberately not the whole
    /// [`DisplayMessage`], whose parsed markdown would be dead weight.
    ReplyTo {
        direction: ChatDirection,
        agent_role: Option<String>,
        content: String,
    },
    /// Dismiss the pending reply preview (clears the captured reference).
    CancelReply,
    /// A composer-draft debounce has settled — persist the current draft.
    ///
    /// `generation` is the debounce counter captured when the edit was staged;
    /// a settle whose generation no longer matches the current counter is
    /// stale and dropped, so per-keystroke and out-of-order writes are
    /// impossible.
    DraftSaveSettled { generation: u64 },
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
    /// Mic button clicked — start a voice message recording to the Assistant.
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
}

pub struct HomeState {
    /// Currently selected workspace name (synced from the footer workspace
    /// picker). `Some("personal:admin")` = the admin's "Personal" workspace;
    /// `Some("ws")` = a shared workspace. Resolved to `personal:admin`
    /// before querying chat_history or sessions.
    selected_workspace: Option<String>,
    /// The admin's DB-stored project workspace (None when unset or
    /// personal). The merge partner for the Personal-picker chat view: at
    /// the Personal picker the chat shows this workspace alongside the
    /// admin's personal workspace.
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
    /// Whether the initial history load has happened for the current workspace.
    history_loaded: bool,
    /// Whether the Phase-1 scripted onboarding scenario is active (shown while
    /// `provider_configured() == false`).
    onboarding_script_active: bool,
    /// Generation counter for stale sending timeout detection.
    sending_gen: u64,
    /// Whether auto-scroll is enabled (user is scrolled to the bottom).
    auto_scroll_enabled: bool,
    /// The database ID of the oldest loaded message, if any.
    oldest_loaded_id: Option<i64>,
    /// Whether there are more older messages to load.
    has_more: bool,
    /// Whether an older-messages load is in-flight.
    loading_older: bool,
    /// Generation counter for stale OlderHistoryLoaded callback detection.
    /// Advanced only by [`Self::reset_pagination_state`], which clears
    /// `loading_older` with it — so a result dropped as stale can never leave
    /// the "load older" action stuck.
    pagination_gen: u64,
    /// Undo/redo stack for the chat input text editor.
    undo_stack: super::common::UndoStack,
    /// Captured {author, snippet} of the message being replied to. `Some`
    /// while a reply preview is shown above the composer; cleared on send,
    /// cancel, chat clear, or workspace switch.
    pending_reply: Option<ReplyReference>,
    /// Persisted composer-draft store (draft text + pending reply per user
    /// and resolved send-target workspace).
    drafts: Arc<crate::channels::chat_draft::DraftStore>,
    /// Debounce state for the asynchronous draft persistence.
    draft_save: super::common::DebounceState,
    /// Persistent failure of the last chat-history read for the current
    /// context. Rendered in place of the empty-chat hint so a failed read is
    /// never mistaken for a legitimately empty history; cleared on the next
    /// successful load.
    history_error: Option<String>,
    /// Persistent failure of the last "load older messages" read. Rendered
    /// above the transcript so a failed read is never mistaken for "nothing
    /// older exists"; cleared on the next successful load.
    older_history_error: Option<String>,
}

/// The chat view for the admin: two workspaces — the picker-resolved
/// one plus a merge partner. Symmetric visibility: the picker only selects
/// the recipient, it never filters the view (personal Assistant
/// messages show at any picker, and the admin's project chat shows at the
/// Personal picker).
#[derive(Debug, Clone, PartialEq, Eq)]
struct VisibleChat {
    /// The picker-resolved workspace: the selected workspace, or
    /// `personal:admin` at the Personal picker.
    primary: String,
    /// The merge partner: `personal:admin` at a project picker, or the
    /// admin's DB-selected project workspace at the Personal picker. `None`
    /// only when the primary is the personal workspace and the admin has no
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
            auto_scroll_enabled: true,
            oldest_loaded_id: None,
            has_more: false,
            loading_older: false,
            pagination_gen: 0,
            undo_stack: super::common::UndoStack::new(),
            pending_reply: None,
            drafts: crate::channels::chat_draft::DraftStore::global().clone(),
            draft_save: super::common::DebounceState::new(),
            history_error: None,
            older_history_error: None,
        }
    }

    /// Resolve the workspace name for chat history and session queries.
    /// A `personal:{user}` name is used as-is; no selection yet → the admin's
    /// personal workspace. Non-personal names are the workspace name as-is.
    fn resolve_workspace_name(&self) -> String {
        match &self.selected_workspace {
            Some(w) => w.clone(),
            None => crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
        }
    }

    /// Whether `workspace` is the admin's personal workspace
    /// (`personal:admin`) — the Assistant chat shown at any picker.
    fn is_admin_personal_workspace(workspace: &str) -> bool {
        crate::users::personal_user_name(workspace).is_some_and(crate::users::is_admin_name)
    }

    /// The visible chat set for the admin (see [`VisibleChat`]).
    fn visible_workspaces(&self) -> VisibleChat {
        let personal = crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME);
        let sel = self.resolve_workspace_name();
        if Self::is_admin_personal_workspace(&sel) {
            VisibleChat {
                primary: personal,
                merge: self.user_project_workspace.clone(),
            }
        } else {
            VisibleChat {
                primary: sel,
                merge: Some(personal),
            }
        }
    }

    /// The key a chat-history read is issued under: the transcript's primary
    /// workspace. A result whose key no longer matches the current selection
    /// belongs to a transcript the page has left.
    fn history_read_key(&self) -> String {
        self.visible_workspaces().primary
    }

    /// Whether a message or typing event in `workspace` belongs to the
    /// admin's visible chat (see [`VisibleChat::contains`]).
    fn workspace_visible(&self, workspace: &str) -> bool {
        self.visible_workspaces().contains(workspace)
    }

    /// Whether the picker selects the admin's personal workspace —
    /// projected from [`Self::visible_workspaces`] (its `primary` is the
    /// personal workspace only at the Personal picker), so the merge-partner
    /// decision keeps a single shape across the refresh paths.
    fn at_personal_picker(&self) -> bool {
        Self::is_admin_personal_workspace(&self.visible_workspaces().primary)
    }

    /// The admin's DB-selected project workspace (None when unset or
    /// personal). The Personal-picker merge partner — re-read on workspace
    /// changes so it never goes stale (Users-page edits flow through
    /// [`WorkspaceChanged`]).
    async fn project_workspace_for() -> Option<String> {
        crate::users::resolve_selected_workspace_name(crate::users::ADMIN_USER_NAME)
            .await
            .filter(|ws| !crate::users::is_personal_workspace(ws))
    }

    /// Refresh chat history from the store for the admin's visible
    /// workspaces (the selected workspace plus the admin's personal workspace).
    fn refresh_history(&self) -> Task<HomeMessage> {
        let chat = self.visible_workspaces();
        let key = chat.primary.clone();
        Task::perform(
            async move {
                let store = crate::channels::chat_history::store();
                store
                    .load_for_user_workspaces(
                        crate::users::ADMIN_USER_NAME,
                        &chat.primary,
                        chat.merge.as_deref(),
                    )
                    .await
                    .map_err(|e| e.to_string())
            },
            move |result| match result {
                Ok((entries, has_more)) => HomeMessage::HistoryLoaded {
                    entries,
                    has_more,
                    key,
                },
                Err(e) => HomeMessage::HistoryLoadError { error: e, key },
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
    /// (workspace change, clear, stream lag).
    fn reset_pagination_state(&mut self) {
        self.oldest_loaded_id = None;
        self.has_more = false;
        self.loading_older = false;
        self.older_history_error = None;
        self.auto_scroll_enabled = true;
        self.pagination_gen = self.pagination_gen.wrapping_add(1);
    }

    /// Reset session display state: messages, dedup set, history flag, pagination.
    fn reset_chat_state(&mut self) {
        self.messages.clear();
        self.seen_ids.clear();
        self.history_loaded = false;
        self.onboarding_script_active = false;
        self.pending_reply = None;
        // The transcript this failure described is gone; the read that follows
        // reports its own outcome.
        self.history_error = None;
        self.reset_pagination_state();
    }
    /// Snapshot the current composer state (text + pending reply) into the
    /// draft store and schedule a debounced persist after `delay_ms`.
    ///
    /// The in-memory mutation happens synchronously at schedule time — the
    /// file write lags behind the debounce, so the in-memory map is always
    /// current.
    fn capture_draft(&mut self, delay_ms: u64) -> Task<HomeMessage> {
        if self.onboarding_script_active {
            return Task::none();
        }
        let ws = self.resolve_workspace_name();
        let text = self.editor_content.text();
        let reply = self.pending_reply.clone();
        self.drafts
            .set(crate::users::ADMIN_USER_NAME, &ws, text, reply);
        self.draft_save
            .trigger(delay_ms)
            .map(|generation| HomeMessage::DraftSaveSettled { generation })
    }

    /// Restore the persisted composer draft for the current context, or clear
    /// the composer when none exists (per-context swap, not leak).
    fn restore_chat_draft(&mut self) {
        let ws = self.resolve_workspace_name();
        let entry = self.drafts.get(crate::users::ADMIN_USER_NAME, &ws);
        self.editor_content
            .set_text(entry.as_ref().map_or("", |e| e.text.as_str()));
        self.undo_stack.clear();
        self.pending_reply = entry.and_then(|e| e.reply);
    }

    /// Remove the persisted draft for the current context and schedule a
    /// persist — used on send, after the composer has been cleared.
    fn drop_current_draft(&mut self) {
        let ws = self.resolve_workspace_name();
        self.drafts.remove(crate::users::ADMIN_USER_NAME, &ws);
        self.drafts.clone().persist_async();
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
    /// Assistant kickoff (provider configured, state Init). Called after the first
    /// history load. No-op when the scenario is already active.
    fn maybe_start_onboarding(&mut self) -> Task<HomeMessage> {
        if !crate::config::provider_configured() {
            if self.onboarding_script_active {
                return Task::none();
            }
            let workspace = self.visible_workspaces().primary;
            // Only arm the scenario once a workspace is confirmed visible, so an
            // empty picker can't strand `onboarding_script_active` with no script.
            self.onboarding_script_active = true;
            for (i, msg) in crate::onboarding::intro_messages().into_iter().enumerate() {
                broadcast_gui_transient(
                    crate::users::ADMIN_USER_NAME,
                    &workspace,
                    &format!("onboarding-intro-{i}"),
                    msg,
                    crate::ChatDirection::Agent,
                    Some("assistant".to_string()),
                    None,
                );
            }
            Task::none()
        } else if crate::config::CONFIG.onboarding_stage() == crate::config::OnboardingState::Init {
            self.onboarding_script_active = false;
            Task::perform(
                async {
                    crate::onboarding::kickoff_onboarding(crate::users::ADMIN_USER_NAME).await
                },
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
    #[expect(clippy::too_many_arguments)]
    fn replace_optimistic(
        &mut self,
        optimistic_id: Option<&str>,
        message_id: &str,
        content: &str,
        direction: ChatDirection,
        agent_role: Option<&str>,
        timestamp: Option<&str>,
        reply_reference: Option<ReplyReference>,
    ) -> Option<Task<HomeMessage>> {
        if let Some(opt_id) = optimistic_id {
            if let Some(pos) = self
                .messages
                .iter()
                .position(|m| m.is_optimistic && m.message_id == *opt_id)
            {
                // Fall back to the optimistic bubble's own reference when the
                // confirmed event carries none (robustness for agent/derived
                // confirmations that drop the reference).
                let reply = reply_reference.or_else(|| self.messages[pos].reply_reference.clone());
                self.messages[pos] = display_message(
                    None,
                    message_id.to_string(),
                    content.to_string(),
                    direction,
                    agent_role.map(std::string::ToString::to_string),
                    timestamp.map(std::string::ToString::to_string),
                    reply,
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
    /// Inserts fresh IDs into `seen_ids` and, once it exceeds
    /// [`DEDUP_PRUNE_THRESHOLD`], prunes it to the most recent [`DEDUP_RETAIN`]
    /// IDs.
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
                .take(DEDUP_RETAIN)
                .map(|m| m.message_id.clone())
                .collect();
            self.seen_ids.retain(|id| retain.contains(id));
        }
        false
    }

    /// Update typing/sending state based on message direction and sender.
    ///
    /// * **Agent** responses for the admin → clear both `typing`
    ///   and `sending` (the agent has replied).
    /// * **User** message echo for the admin → clear `sending`
    ///   only (re-enables the send button). Does **not** clear `typing`
    ///   — the typing indicator persists until an agent response arrives.
    ///
    /// Does nothing when `workspace` is not visible for the admin
    /// (see [`Self::workspace_visible`]) — this prevents an agent response
    /// from an unrelated workspace from clearing the typing/sending
    /// indicators for the visible chat.
    fn update_sending_state(&mut self, direction: ChatDirection, user_name: &str, workspace: &str) {
        if !self.workspace_visible(workspace) {
            return;
        }
        if !crate::users::is_admin_name(user_name) {
            return;
        }

        self.sending = false;
        if direction == ChatDirection::Agent {
            self.typing = false;
        }
    }

    /// Append a chat message if it belongs to the admin's visible
    /// chat (selected workspace or the admin's personal workspace).
    ///
    /// Does nothing when `user_name` is not the admin, or when
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
        reply_reference: Option<ReplyReference>,
    ) {
        if !crate::users::is_admin_name(user_name) {
            return;
        }
        if !self.workspace_visible(workspace) {
            return;
        }

        self.messages.push(display_message(
            None,
            message_id,
            content,
            direction,
            agent_role,
            timestamp,
            reply_reference,
            false,
        ));
    }

    #[expect(clippy::too_many_lines)]
    pub fn view(&self, draining: bool) -> Element<'_, HomeMessage> {
        // ── Chat message area ────────────────────────────────────
        let chat_area = if self.messages.is_empty() {
            let empty_hint = if self.selected_workspace.is_none() {
                "No workspace selected."
            } else {
                "No messages yet. Type something below to start."
            };
            // A failed read must render in place of the empty hint and stay
            // until a successful read replaces it — a log line or toast is
            // not enough.
            let body: Element<'_, HomeMessage> = if let Some(err) = &self.history_error {
                super::widgets::error_banner(err)
            } else {
                text(empty_hint)
                    .color(theme::TEXT_SECONDARY)
                    .size(theme::TEXT_13)
                    .into()
            };
            container(body)
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .style(theme::base_container_style)
                .into()
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
                                .size(theme::TEXT_12),
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
                        .spacing(theme::SPACE_4)
                        .padding(theme::PAD_8)
                        .width(Length::Fill);

                        return divider.into();
                    }

                    let is_user = msg.direction == ChatDirection::User;

                    // Render markdown content
                    let content: Element<'_, HomeMessage> = if msg.md_items.is_empty() {
                        super::widgets::selectable_text(&msg.content, theme::TEXT_PRIMARY)
                            .size(theme::TEXT_13)
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
                        // Strip the numeric role suffix (e.g. "analyst_3" →
                        // "analyst", shared with reply author labels) and parse.
                        let maybe_role = msg
                            .agent_role
                            .as_deref()
                            .map(crate::channels::reply::strip_agent_role_suffix)
                            .and_then(|stripped| stripped.parse::<Role>().ok());
                        if let Some(role) = maybe_role {
                            let (icon_color, _) = theme::role_badge_color_for(&role);
                            let icon = theme::role_icon(&role)
                                .size(theme::TEXT_14)
                                .color(icon_color);
                            let mut icon_row = row![icon]
                                .align_y(Alignment::Center)
                                .spacing(theme::SPACE_6);
                            if let Some(ts) = msg.timestamp.as_deref() {
                                let label = theme::format_relative_time(ts, chrono::Local::now());
                                if !label.is_empty() {
                                    icon_row = icon_row.push(
                                        text(label)
                                            .size(theme::TEXT_11)
                                            .color(theme::TEXT_SECONDARY),
                                    );
                                }
                            }
                            column![icon_row, content].spacing(theme::SPACE_4).into()
                        } else {
                            content
                        }
                    };

                    let bubble = container(bubble_body)
                        .padding(theme::PAD_10)
                        .style(theme::bubble_style(
                            if is_user {
                                theme::BG_ELEVATED
                            } else {
                                theme::BG_SURFACE
                            },
                            Some(theme::TEXT_PRIMARY),
                        ))
                        .width(Length::FillPortion(3));

                    // Quote header for a replied-to message: rendered above the
                    // bubble (inside the same context-menu region) so a reply
                    // stays visually tied to its target. Direction-agnostic —
                    // agent messages never carry references today, but the
                    // header renders whenever one is present.
                    let bubble_region: Element<'_, HomeMessage> =
                        if let Some(reply) = &msg.reply_reference {
                            column![reply_quote_header(reply), bubble.width(Length::Fill),]
                                .spacing(theme::SPACE_4)
                                .width(Length::FillPortion(3))
                                .into()
                        } else {
                            bubble.into()
                        };

                    // Per-bubble context menu: right-clicking the bubble offers
                    // copying the raw markdown content (the exact stored text
                    // with original media markers) and replying to it (hidden on
                    // optimistic placeholders; divider rows return earlier so
                    // they never get a menu). Wrapping only the bubble region —
                    // not the align_bubble row — keeps the spacer beside it a
                    // fall-through: spacer/empty-space right-clicks reach the
                    // outer "Reset session" menu in gui/mod.rs instead.
                    let mut menu_items = vec![MenuItem::new(
                        "Copy message".into(),
                        HomeMessage::CopyMessage(msg.content.clone()),
                    )];
                    if !msg.is_optimistic {
                        menu_items.push(MenuItem::with_icon(
                            iced_fonts::lucide::advanced_text::reply,
                            "Reply".into(),
                            HomeMessage::ReplyTo {
                                direction: msg.direction,
                                agent_role: msg.agent_role.clone(),
                                content: msg.content.clone(),
                            },
                        ));
                    }
                    let bubble: Element<'_, HomeMessage> =
                        ContextMenu::new(bubble_region, menu_items).into();

                    align_bubble(
                        bubble,
                        if is_user {
                            BubbleSide::Left
                        } else {
                            BubbleSide::Right
                        },
                    )
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
                    .padding(theme::PAD_10)
                    .style(theme::bubble_style(theme::BG_SURFACE, None))
                    .width(Length::FillPortion(3));

                // Typing indicator always renders on the agent (right) side.
                children.push(align_bubble(typing_bubble, BubbleSide::Right));
            }

            // Prepend "Load older messages" button when applicable.
            if self.has_more && self.history_loaded {
                let load_text = if self.loading_older {
                    "Loading older messages\u{2026}"
                } else {
                    "▲ Load older messages"
                };
                let load_btn = button(
                    text(load_text)
                        .size(theme::TEXT_12)
                        .color(theme::TEXT_SECONDARY),
                )
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
                children.insert(0, container(load_btn).padding(theme::PAD_4).into());
            }

            // A failed history read must not leave the transcript looking
            // current: the failure stays above the loaded messages until a read
            // succeeds. (This is the resync case — with an empty transcript the
            // same failure renders in the empty state above.)
            if let Some(err) = &self.history_error {
                children.insert(
                    0,
                    super::widgets::scroll_h_inset(super::widgets::error_banner(err)),
                );
            }

            // A failed "load older" read must not leave the transcript looking
            // as if nothing older existed: the failure stays above the loaded
            // messages until a later load succeeds.
            if let Some(err) = &self.older_history_error {
                children.insert(
                    0,
                    super::widgets::scroll_h_inset(super::widgets::error_banner(err)),
                );
            }

            super::widgets::page_bare(super::widgets::vscroll_tracked(
                Column::with_children(children)
                    .spacing(super::widgets::CHAT_VERTICAL_RHYTHM)
                    .padding([theme::PAD_8, 0.0]),
                Length::Fill,
                Length::Fill,
                CHAT_SCROLL_ID,
                HomeMessage::ScrollChanged,
            ))
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

        // Controls for the composer action toolbar: the mic button.
        let mut controls: Vec<Element<'_, HomeMessage>> = Vec::new();

        let mic_tooltip = if transcription_disabled {
            "voice recording unavailable — local transcription is disabled"
        } else {
            "record voice message"
        };
        controls.push(super::widgets::icon_tooltip_button(
            lucide::mic::<iced::Theme, iced::Renderer>()
                .size(theme::TEXT_14)
                .color(if recording_unavailable {
                    theme::TEXT_MUTED
                } else {
                    theme::TEXT_SECONDARY
                }),
            mic_tooltip,
            (!recording_unavailable).then_some(HomeMessage::StartVoiceRecording),
            theme::PAD_3,
            theme::icon_button_style(recording_unavailable),
            tooltip::Position::Top,
        ));

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
                id: Some(CHAT_COMPOSER_ID),
            },
        ))
        .style(theme::base_container_style)
        .padding(
            iced::Padding::from([0.0, theme::PAD_8]).bottom(super::widgets::CHAT_VERTICAL_RHYTHM),
        )
        .width(Length::Fill)
        .into();

        // ── Recording popup (stop + send / stop + discard) ───────
        // While transcribing, the popup stays visible as a passive
        // "Transcribing…" indicator (no stop controls — the ASR is finalizing).
        let recording_popup: Element<'_, HomeMessage> = if recording {
            let status_label = text("Recording voice message…")
                .size(theme::TEXT_13)
                .color(theme::STATUS_ERROR);
            let send_btn = button(text("Stop + Send").size(theme::TEXT_12))
                .on_press(HomeMessage::StopVoiceRecordingSend)
                .style(theme::button_secondary)
                .padding(theme::PAD_5);
            let discard_btn = button(text("Stop + Discard").size(theme::TEXT_12))
                .on_press(HomeMessage::StopVoiceRecordingDiscard)
                .style(theme::button_secondary)
                .padding(theme::PAD_5);
            container(
                row![
                    status_label,
                    Space::new().width(Length::Fill),
                    send_btn,
                    discard_btn
                ]
                .spacing(theme::SPACE_8)
                .align_y(Alignment::Center),
            )
            .padding(theme::PAD_8)
            .style(theme::surface_container_style)
            .width(Length::Fill)
            .into()
        } else if transcribing {
            container(
                row![
                    text("Transcribing voice message…")
                        .size(theme::TEXT_13)
                        .color(theme::TEXT_MUTED),
                    Space::new().width(Length::Fill),
                ]
                .spacing(theme::SPACE_8)
                .align_y(Alignment::Center),
            )
            .padding(theme::PAD_8)
            .style(theme::surface_container_style)
            .width(Length::Fill)
            .into()
        } else {
            Space::new().height(0).into()
        };

        // ── Reply preview ────────────────────────────────────────
        // Shown between the message list and the composer while a reply target
        // is pending; the dismiss control cancels it.
        let reply_preview: Element<'_, HomeMessage> = match &self.pending_reply {
            Some(reply) => reply_preview(reply),
            None => Space::new().height(0).into(),
        };

        // ── Full layout ──────────────────────────────────────────
        column![chat_area, recording_popup, reply_preview, input_area,]
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
            HomeMessage::WorkspaceChanged(ws_name) => {
                self.selected_workspace.clone_from(&ws_name);
                self.reset_chat_state();
                self.restore_chat_draft();
                // At the Personal picker the view merges the admin's DB-selected
                // project workspace — re-read it so Users-page edits (which
                // flow through WorkspaceChanged) are reflected, then load in
                // one shot. At a project picker the merge partner is the
                // selected workspace itself, so a direct refresh suffices.
                if self.at_personal_picker() {
                    Task::perform(
                        Self::project_workspace_for(),
                        HomeMessage::ProjectWorkspaceRefreshed,
                    )
                } else {
                    self.refresh_history()
                }
            }
            HomeMessage::ProjectWorkspaceRefreshed(project) => {
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
                let changes_text = action.changes_text();
                super::common::apply_editor_action(
                    &mut self.editor_content,
                    &mut self.undo_stack,
                    action,
                );
                if changes_text {
                    self.capture_draft(DRAFT_SETTLE_MS)
                } else {
                    Task::none()
                }
            }
            HomeMessage::SendMessage => self.send_message(),
            HomeMessage::HistoryLoaded {
                entries,
                has_more,
                key,
            } => {
                // A read issued for a transcript the page has since left must not
                // replace the current one — nor clear the failure of the one it
                // left behind.
                if self.history_read_key() != key {
                    return Task::none();
                }
                self.history_error = None;
                // A fresh transcript has not attempted to page further back, so
                // a previous transcript's failed "load older" no longer applies.
                self.older_history_error = None;
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
            HomeMessage::HistoryLoadError { key, error } => {
                // A read issued for a workspace the page has since left must
                // not paint its failure over the current transcript.
                tracing::warn!(error = %error, "Home: failed to load chat history");
                if self.history_read_key() == key {
                    self.history_error = Some(error);
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
                self.pending_reply = None;
                // The transcript this failure described is gone; the reload the
                // `ChatCleared` callback issues reports its own outcome.
                self.history_error = None;
                self.reset_pagination_state();
                // Capture synchronously (not from the ChatCleared callback):
                // the reply is dropped but the draft text is kept, and the
                // entry must be snapshotted under the context that was
                // visible when the clear was requested.
                let draft = self.capture_draft(DRAFT_SETTLE_MS);
                // Build agent ID and schedule async cleanup.
                let sender = crate::users::ADMIN_USER_NAME.to_string();
                let clear_task = Task::perform(
                    async move {
                        // Clear the session the admin actually talks to — the
                        // same resolution as routing and Telegram /clear (see
                        // [`crate::users::resolve_session_target`]): the
                        // workspace resolves from the admin's DB record, never
                        // from the GUI picker position, and the Assistant is
                        // pinned to it — so the cleared session is always the
                        // admin's personal-workspace session.
                        let (effective_role, ws) =
                            crate::users::resolve_session_target(&sender).await;
                        // Fail-closed: a failed abandon aborts the clear — the
                        // session was not cleared, so no divider is inserted.
                        if let Err(e) = crate::session::clear_session(
                            &sender,
                            effective_role.as_str(),
                            &ws.name,
                        )
                        .await
                        {
                            tracing::warn!(
                                user = %sender,
                                error = %e,
                                "Home: clear failed — session kept"
                            );
                            return Err(e.to_string());
                        }
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
                );
                Task::batch([draft, clear_task])
            }
            HomeMessage::ChatCleared => {
                // Belt and suspenders: ClearChat cleared the message list and
                // this callback re-loads history (not via reset_chat_state), so
                // drop any pending reply explicitly.
                self.pending_reply = None;
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
            HomeMessage::ReplyTo {
                direction,
                agent_role,
                content,
            } => {
                self.pending_reply = Some(reply_reference_for(
                    direction,
                    agent_role.as_deref(),
                    &content,
                ));
                let draft = self.capture_draft(DRAFT_SETTLE_MS);
                // Refocus the composer so the user can type the reply
                // immediately, matching the select-to-focus affordance
                // established by the composer's `Id`.
                Task::batch([
                    iced::widget::operation::focus::<HomeMessage>(CHAT_COMPOSER_ID),
                    draft,
                ])
            }
            HomeMessage::CancelReply => {
                self.pending_reply = None;
                self.capture_draft(DRAFT_SETTLE_MS)
            }
            HomeMessage::DraftSaveSettled { generation } => {
                if self.draft_save.should_process(generation) {
                    self.drafts.clone().persist_async();
                }
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
                    reply_reference,
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
                        reply_reference.clone(),
                    ) {
                        return task;
                    }

                    // 2. Deduplicate against already-seen IDs.
                    if self.try_dedup(&message_id) {
                        return Task::none();
                    }

                    // 3. Clear sending/typing state based on direction, sender, and workspace.
                    self.update_sending_state(direction, &user_name, &workspace);

                    // 4. Append the message (filtered by acting user + workspace).
                    self.append_message(
                        &user_name,
                        &workspace,
                        message_id,
                        content,
                        direction,
                        agent_role,
                        Some(timestamp),
                        reply_reference,
                    );

                    self.maybe_snap()
                }
                crate::ChatEvent::Typing {
                    user_name,
                    is_typing,
                    workspace,
                } => {
                    // Apply user + workspace filter — only show typing indicator
                    // for the admin in a visible workspace.
                    if crate::users::is_admin_name(&user_name) && self.workspace_visible(&workspace)
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
                let sender = crate::users::ADMIN_USER_NAME.to_string();
                let chat = self.visible_workspaces();
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
                            .map_err(|e| e.to_string())
                    },
                    move |result| match result {
                        Ok((entries, has_more)) => {
                            HomeMessage::OlderHistoryLoaded(entries, has_more, generation)
                        }
                        Err(e) => HomeMessage::OlderHistoryLoadError(e, generation),
                    },
                )
            }
            HomeMessage::OlderHistoryLoaded(display_entries, has_more, generation) => {
                // Stale-result guard: `pagination_gen` advances only through
                // `reset_pagination_state` (which clears `loading_older`), so a
                // dropped result cannot strand the action — and clearing the flag
                // here could take it away from a newer load already in flight.
                if generation != self.pagination_gen {
                    return Task::none();
                }
                // Prepend entries to the beginning of messages.
                let mut prepended: Vec<DisplayMessage> = display_entries
                    .into_iter()
                    .map(DisplayMessage::from)
                    .collect();
                self.older_history_error = None;
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
            HomeMessage::OlderHistoryLoadError(msg, generation) => {
                // Stale-result guard, mirroring `OlderHistoryLoaded`: a failed
                // read for a transcript the page has left must not pin its
                // failure on the current one (nor touch its in-flight flag).
                if generation != self.pagination_gen {
                    return Task::none();
                }
                // The persistent banner is the signal; a toast here would report
                // the same failure a second time and then vanish.
                self.loading_older = false;
                self.older_history_error = Some(msg);
                Task::none()
            }
            HomeMessage::Toast(_) => {
                // Intercepted by the Dashboard (Toast → toast stack). No-op
                // fallback.
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
            HomeMessage::StartVoiceRecording => {
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

        let sender = crate::users::ADMIN_USER_NAME.to_string();

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
        // message without waiting for the pipeline round-trip. The pending
        // reply reference (if any) attaches to the bubble so the quote header
        // renders live; the scripted onboarding path must never flash one, so
        // the reference is skipped (and left untouched) there.
        if let Some(ref opt_id) = optimistic_id {
            let reply = if self.onboarding_script_active {
                None
            } else {
                self.pending_reply.clone()
            };
            self.messages.push(display_message(
                None,
                opt_id.clone(),
                content.clone(),
                ChatDirection::User,
                None,
                None,
                reply,
                true,
            ));
        }

        // Phase-1 scripted onboarding intercepts the send before the pipeline:
        // the provider entry is parsed and persisted directly, and the
        // exchange is rendered as transient (never persisted) events. The
        // pending reply is dropped with the send — scripted messages never
        // carry one, and a stale reference must not attach to a later send.
        if self.onboarding_script_active {
            self.pending_reply = None;
            return self.send_scripted_message(content, optimistic_id);
        }

        // Capture the pending reply reference for the confirmed message — a
        // command-based message (no optimistic bubble) still carries it so it
        // surfaces on the confirmed bubble via the ChatEvent. Cleared only
        // after the send is queued (see below).
        let reply = self.pending_reply.clone();

        let msg = crate::ChannelMessage {
            user_name: sender.clone(),
            reply_target: sender,
            content,
            channel: "gui".to_string(),
            workspace: self.selected_workspace.clone().unwrap_or_default(),
            optimistic_id,
            reply_reference: reply,
            ..Default::default()
        };

        // Push to GUI_MESSAGE_TX.
        if let Some(tx) = crate::GUI_MESSAGE_TX.get() {
            if let Err(e) = tx.send(msg) {
                tracing::error!("Home: failed to send message via GUI_MESSAGE_TX: {e}");
                self.sending = false;
                self.drop_current_draft();
                return Task::none();
            }
        } else {
            tracing::error!("Home: GUI_MESSAGE_TX not initialized");
            self.sending = false;
            self.drop_current_draft();
            return Task::none();
        }

        // The send was queued — drop the pending reply so the preview clears.
        self.pending_reply = None;
        // The composer is cleared: remove the persisted draft for this context
        // so it cannot resurrect on a later restore.
        self.drop_current_draft();

        // Snap to end on optimistic push if auto-scroll enabled.
        Task::batch([self.sending_timeout_task(), self.maybe_snap()])
    }

    /// Arm the 30-second SendingTimeout guard: if `sending` stays true for 30
    /// seconds (silent agent failure, crash, cancellation), auto-clear it. The
    /// generation counter prevents a stale timeout from clearing `sending`
    /// during a new send.
    fn sending_timeout_task(&mut self) -> Task<HomeMessage> {
        self.sending_gen = self.sending_gen.wrapping_add(1);
        let generation = self.sending_gen;
        Task::perform(
            async move {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                HomeMessage::SendingTimeout(generation)
            },
            |msg| msg,
        )
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
        let user = crate::users::ADMIN_USER_NAME;
        let workspace = self.visible_workspaces().primary;
        let send_task = Task::perform(
            async move {
                let parsed = crate::onboarding::parse_provider_input(&content);
                // Replace the optimistic bubble with the user's transient message.
                broadcast_gui_transient(
                    user,
                    &workspace,
                    &crate::generate_id(),
                    &content,
                    crate::ChatDirection::User,
                    None,
                    optimistic_id,
                );
                let outcome: Result<(), String> = match parsed {
                    crate::onboarding::ProviderInput::Invalid => {
                        broadcast_gui_transient(
                            user,
                            &workspace,
                            &crate::generate_id(),
                            crate::onboarding::invalid_message(),
                            crate::ChatDirection::Agent,
                            Some("assistant".to_string()),
                            None,
                        );
                        Err("invalid provider input".to_string())
                    }
                    valid => match crate::onboarding::persist_provider_input(&valid).await {
                        Ok(()) => {
                            broadcast_gui_transient(
                                user,
                                &workspace,
                                &crate::generate_id(),
                                crate::onboarding::success_message(),
                                crate::ChatDirection::Agent,
                                Some("assistant".to_string()),
                                None,
                            );
                            Ok(())
                        }
                        Err(e) => {
                            broadcast_gui_transient(
                                user,
                                &workspace,
                                &crate::generate_id(),
                                &format!("Couldn't save that: {e:#}"),
                                crate::ChatDirection::Agent,
                                Some("assistant".to_string()),
                                None,
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
        Task::batch([send_task, self.sending_timeout_task()])
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

    fn make_home_state(workspace: &str) -> HomeState {
        let mut state = HomeState::new();
        // Hermetic draft store bound to a throwaway temp dir so tests never
        // touch the real `~/.mahbot/chat-draft.json`.
        state.drafts = hermetic_draft_store();
        state.selected_workspace = Some(workspace.to_string());
        state
    }

    /// A draft store backed by a unique throwaway file under one shared test
    /// temp dir, so tests never touch the real `~/.mahbot/chat-draft.json`.
    fn hermetic_draft_store() -> Arc<crate::channels::chat_draft::DraftStore> {
        static DIR: std::sync::LazyLock<tempfile::TempDir> =
            std::sync::LazyLock::new(|| tempfile::tempdir().unwrap());
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        crate::channels::chat_draft::DraftStore::at(Some(
            DIR.path().join(format!("draft-{n}.json")),
        ))
    }

    #[test]
    fn draft_capture_and_workspace_swap() {
        let mut state = make_home_state("ws1");
        let admin = crate::users::ADMIN_USER_NAME;

        // Typing captures synchronously into the store under the current key.
        let _ = state.update(HomeMessage::InputChanged(EditorAction::Insert('h')));
        assert_eq!(
            state.drafts.get(admin, "ws1").map(|e| e.text).as_deref(),
            Some("h")
        );

        // The workspace swap moves the context: the incoming context restores —
        // no entry means the composer is cleared (per-context swap, not leak).
        let _ = state.update(HomeMessage::WorkspaceChanged(Some("ws2".to_string())));
        assert_eq!(state.editor_content.text(), "");

        // The resolved context captures under its own key...
        let _ = state.update(HomeMessage::InputChanged(EditorAction::Insert('b')));
        assert_eq!(
            state.drafts.get(admin, "ws2").map(|e| e.text).as_deref(),
            Some("b")
        );

        // ...and switching back swaps again, restoring the first draft.
        let _ = state.update(HomeMessage::WorkspaceChanged(Some("ws1".to_string())));
        assert_eq!(state.editor_content.text(), "h");
    }

    #[test]
    fn failed_reads_stay_set_until_a_successful_load() {
        let mut state = HomeState::new();

        // A failed history read is retained; a successful load clears it.
        let key = state.history_read_key();
        let _ = state.update(HomeMessage::HistoryLoadError {
            key: key.clone(),
            error: "db down".to_string(),
        });
        assert_eq!(state.history_error.as_deref(), Some("db down"));

        // A failure whose key no longer matches the current selection belongs to
        // a transcript the page has left: it must not overwrite the current one.
        let stale_key = "personal:bob".to_string();
        let _ = state.update(HomeMessage::HistoryLoadError {
            key: stale_key.clone(),
            error: "stale".to_string(),
        });
        assert_eq!(state.history_error.as_deref(), Some("db down"));

        // Neither may a successful read of that left transcript: applying it
        // would replace the current transcript and clear the failure it shows.
        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: Vec::new(),
            has_more: false,
            key: stale_key,
        });
        assert_eq!(
            state.history_error.as_deref(),
            Some("db down"),
            "a stale successful read must not clear the current failure"
        );

        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: Vec::new(),
            has_more: false,
            key: key.clone(),
        });
        assert!(state.history_error.is_none());

        // A transcript replacement drops the failure with the transcript it
        // described; the read that follows reports its own outcome.
        let _ = state.update(HomeMessage::HistoryLoadError {
            key: key.clone(),
            error: "db down".to_string(),
        });
        state.reset_chat_state();
        assert!(state.history_error.is_none());

        // Same for a cleared chat, which reloads through its own callback.
        let _ = state.update(HomeMessage::HistoryLoadError {
            key,
            error: "db down".to_string(),
        });
        let _ = state.update(HomeMessage::ClearChat);
        assert!(state.history_error.is_none());

        // Same for the older-messages read: the failure survives until a load
        // succeeds.
        let generation = state.pagination_gen;
        let _ = state.update(HomeMessage::OlderHistoryLoadError(
            "db down".to_string(),
            generation,
        ));
        assert_eq!(state.older_history_error.as_deref(), Some("db down"));
        let _ = state.update(HomeMessage::OlderHistoryLoaded(
            Vec::new(),
            false,
            generation,
        ));
        assert!(state.older_history_error.is_none());

        // A paging failure from a transcript the page has left is dropped, like
        // its successful counterpart.
        let _ = state.update(HomeMessage::OlderHistoryLoadError(
            "stale".to_string(),
            generation.wrapping_sub(1),
        ));
        assert!(state.older_history_error.is_none());

        // A fresh transcript drops a stale older-messages failure: it belongs
        // to the transcript that was replaced.
        let _ = state.update(HomeMessage::OlderHistoryLoadError(
            "db down".to_string(),
            state.pagination_gen,
        ));
        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: Vec::new(),
            has_more: false,
            key: state.history_read_key(),
        });
        assert!(state.older_history_error.is_none());
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
            reply_reference: None,
            md_items: Vec::new(),
            is_optimistic,
        }
    }

    // ------------------------------------------------------------------
    // replace_optimistic
    // ------------------------------------------------------------------

    #[test]
    fn test_replace_optimistic_found() {
        let mut state = make_home_state("ws1");
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
        let mut state = make_home_state("ws1");
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
        let mut state = make_home_state("ws1");

        let task = state.replace_optimistic(
            None,
            "real-42",
            "Hello!",
            ChatDirection::User,
            None,
            None,
            None,
        );

        assert!(task.is_none(), "expected None when optimistic_id is None");
    }

    // ------------------------------------------------------------------
    // try_dedup
    // ------------------------------------------------------------------

    #[test]
    fn test_try_dedup_fresh() {
        let mut state = make_home_state("ws1");
        assert!(!state.try_dedup("msg-1"), "fresh ID should return false");
        assert!(state.seen_ids.contains("msg-1"), "fresh ID should be added");
    }

    #[test]
    fn test_try_dedup_duplicate() {
        let mut state = make_home_state("ws1");
        state.seen_ids.insert("msg-1".to_string());
        assert!(state.try_dedup("msg-1"), "duplicate should return true");
    }

    #[test]
    fn test_try_dedup_pruning() {
        let mut state = make_home_state("ws1");
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
                crate::users::ADMIN_USER_NAME,
                "ws1",
                false,
                false,
            ),
            (
                "user match",
                ChatDirection::User,
                crate::users::ADMIN_USER_NAME,
                "ws1",
                false,
                true,
            ),
            ("wrong user", ChatDirection::Agent, "bob", "ws1", true, true),
            (
                "wrong workspace",
                ChatDirection::Agent,
                crate::users::ADMIN_USER_NAME,
                "ws2",
                true,
                true,
            ),
        ];
        for (name, direction, user, workspace, exp_sending, exp_typing) in cases {
            let mut state = make_home_state("ws1");
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
        let mut state = make_home_state("ws1");
        assert_eq!(state.messages.len(), 0);

        state.append_message(
            crate::users::ADMIN_USER_NAME,
            "ws1",
            "msg-1".to_string(),
            "Hello!".to_string(),
            ChatDirection::User,
            None,
            None,
            None,
        );

        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].message_id, "msg-1");
        assert_eq!(state.messages[0].content, "Hello!");
    }

    #[test]
    fn test_append_message_no_match_user() {
        let mut state = make_home_state("ws1");

        state.append_message(
            "bob",
            "ws1",
            "msg-1".to_string(),
            "Hello!".to_string(),
            ChatDirection::User,
            None,
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
        let mut state = make_home_state("ws1");

        state.append_message(
            crate::users::ADMIN_USER_NAME,
            "ws2",
            "msg-1".to_string(),
            "Hello!".to_string(),
            ChatDirection::User,
            None,
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
        let mut state = make_home_state("ws1");

        state.append_message(
            crate::users::ADMIN_USER_NAME,
            "ws1",
            "msg-agent".to_string(),
            "Agent answer".to_string(),
            ChatDirection::Agent,
            Some("engineer".to_string()),
            None,
            None,
        );

        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].direction, ChatDirection::Agent);
        assert_eq!(state.messages[0].agent_role.as_deref(), Some("engineer"),);
    }

    // ------------------------------------------------------------------
    // personal-workspace visibility (Assistant at any picker)
    // ------------------------------------------------------------------

    #[test]
    fn test_visible_chat_symmetric_at_any_picker() {
        let admin = crate::users::ADMIN_USER_NAME;
        let personal_name = crate::users::personal_workspace_name(admin);

        // Project picker: project + personal visible; unrelated stays hidden.
        let project = make_home_state("ws1");
        assert_eq!(
            project.visible_workspaces(),
            VisibleChat {
                primary: "ws1".to_string(),
                merge: Some(personal_name.clone()),
            }
        );
        assert!(project.workspace_visible("ws1"));
        assert!(
            project.workspace_visible(&personal_name),
            "personal Assistant messages must be visible at any picker"
        );
        assert!(!project.workspace_visible("ws2"));
        assert!(
            !project.workspace_visible("personal:bob"),
            "another user's personal workspace is not visible"
        );

        // Personal picker: personal + the admin's DB project workspace visible
        // (symmetric — the picker selects the recipient, not the view).
        let mut personal = make_home_state(&personal_name);
        personal.user_project_workspace = Some("ws1".to_string());
        assert_eq!(
            personal.resolve_workspace_name(),
            personal_name,
            "a personal picker selection resolves to the personal workspace"
        );
        assert_eq!(
            personal.visible_workspaces(),
            VisibleChat {
                primary: personal_name.clone(),
                merge: Some("ws1".to_string()),
            },
            "personal picker merges the admin's project workspace"
        );
        assert!(personal.workspace_visible(&personal_name));
        assert!(
            personal.workspace_visible("ws1"),
            "project messages must be visible at the personal picker"
        );
        assert!(
            !personal.workspace_visible("ws2"),
            "a non-selected project workspace stays hidden at the personal picker"
        );

        // No DB project workspace → personal-only view.
        let personal_only = make_home_state(&personal_name);
        assert_eq!(
            personal_only.visible_workspaces(),
            VisibleChat {
                primary: personal_name,
                merge: None,
            },
            "personal picker without a project workspace shows only the personal chat"
        );
        assert!(!personal_only.workspace_visible("ws1"));
    }

    #[test]
    fn test_workspace_changed_at_personal_picker_resets_and_defers() {
        // Switching to the Personal picker resets the chat and re-reads the
        // admin's DB project workspace (completion via ProjectWorkspaceRefreshed
        // is covered by test_project_workspace_refreshed_updates_merge_partner).
        let mut state = make_home_state("ws1");
        state.user_project_workspace = Some("ws1".to_string());
        state
            .messages
            .push(make_msg("m1", "hi", ChatDirection::User, None, false));

        let _task = state.update(HomeMessage::WorkspaceChanged(Some(
            "personal:admin".to_string(),
        )));
        assert_eq!(state.selected_workspace.as_deref(), Some("personal:admin"));
        assert!(
            state.messages.is_empty(),
            "chat state resets on workspace change"
        );
        assert_eq!(
            state.resolve_workspace_name(),
            "personal:admin",
            "personal picker selection resolves to the personal workspace"
        );

        // A project-picker change refreshes history directly and keeps the
        // merge partner untouched (it only matters at the Personal picker).
        let mut state = make_home_state("ws1");
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
        let mut state = make_home_state("personal:admin");
        state.user_project_workspace = Some("ws1".to_string());

        let _ = state.update(HomeMessage::ProjectWorkspaceRefreshed(Some(
            "ws2".to_string(),
        )));
        assert_eq!(state.user_project_workspace.as_deref(), Some("ws2"));
        assert!(state.workspace_visible("ws2"));
        assert!(!state.workspace_visible("ws1"));
        assert_eq!(
            state.visible_workspaces(),
            VisibleChat {
                primary: "personal:admin".to_string(),
                merge: Some("ws2".to_string()),
            }
        );

        // At a project picker the refreshed value is stored but the view is
        // the selected workspace (no reload needed there).
        let mut state = make_home_state("ws1");
        state.user_project_workspace = Some("ws1".to_string());
        let _ = state.update(HomeMessage::ProjectWorkspaceRefreshed(Some(
            "ws2".to_string(),
        )));
        assert_eq!(state.user_project_workspace.as_deref(), Some("ws2"));
        assert_eq!(
            state.visible_workspaces(),
            VisibleChat {
                primary: "ws1".to_string(),
                merge: Some("personal:admin".to_string()),
            }
        );
    }

    #[tokio::test]
    #[serial_test::serial(gui_admin_workspace)] // writes the shared seeded admin row
    async fn test_project_workspace_for_reads_db_normalized() {
        crate::util::test::init_test_stores().await;
        let store = crate::users::USER_STORE
            .get()
            .expect("users store initialized");

        // The guard restores the shared seeded admin row even when an assertion
        // below panics.
        crate::users::test_util::with_admin_workspace_restored(async {
            // NULL → None.
            store
                .set_selected_workspace(crate::users::ADMIN_USER_NAME, None)
                .await
                .expect("update admin selected_workspace");
            assert_eq!(HomeState::project_workspace_for().await, None);

            // Personal DB workspace → None.
            store
                .set_selected_workspace(
                    crate::users::ADMIN_USER_NAME,
                    Some("personal:home_project_workspace_for"),
                )
                .await
                .expect("update admin selected_workspace");
            assert_eq!(HomeState::project_workspace_for().await, None);

            // Shared DB workspace → Some(ws).
            crate::util::test::create_test_workspace(
                "/tmp/home_project_workspace_for_ws",
                "ws_home_project_workspace_for",
            )
            .await;
            store
                .set_selected_workspace(
                    crate::users::ADMIN_USER_NAME,
                    Some("ws_home_project_workspace_for"),
                )
                .await
                .expect("update admin selected_workspace");
            assert_eq!(
                HomeState::project_workspace_for().await.as_deref(),
                Some("ws_home_project_workspace_for")
            );
        })
        .await;
    }

    #[test]
    fn test_call_site_wiring_symmetric_at_any_picker() {
        // (picker, message workspace, agent role, content) — both directions:
        // at the project picker personal Assistant messages append and
        // clear sending/typing; at the personal picker Assistant messages
        // still append into the admin's DB project workspace (the merge partner).
        let cases = [
            ("ws1", "personal:admin", "assistant", "Assistant reply"),
            (
                "personal:admin",
                "ws1",
                "assistant",
                "Assistant reply in project",
            ),
        ];
        for (picker, msg_ws, role, content) in cases {
            let mut state = make_home_state(picker);
            state.user_project_workspace = Some("ws1".to_string());
            state.append_message(
                crate::users::ADMIN_USER_NAME,
                msg_ws,
                "msg".to_string(),
                content.to_string(),
                ChatDirection::Agent,
                Some(role.to_string()),
                None,
                None,
            );
            assert_eq!(state.messages.len(), 1);
            assert_eq!(state.messages[0].content, content);
            state.sending = true;
            state.typing = true;
            state.update_sending_state(ChatDirection::Agent, crate::users::ADMIN_USER_NAME, msg_ws);
            assert!(!state.sending);
            assert!(!state.typing);
        }

        // A project workspace that is not the admin's own stays hidden at the
        // personal picker.
        let mut state = make_home_state("personal:admin");
        state.user_project_workspace = Some("ws1".to_string());
        state.append_message(
            crate::users::ADMIN_USER_NAME,
            "ws2",
            "msg-other".to_string(),
            "other".to_string(),
            ChatDirection::Agent,
            None,
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
            reply_reference: None,
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
            reply_reference: None,
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

    #[test]
    fn test_display_message_from_entry_with_reply_reference() {
        let entry = ChatHistoryEntry {
            id: 7,
            message_id: "msg-7".to_string(),
            content: "replied text".to_string(),
            direction: ChatDirection::User,
            agent_role: None,
            timestamp: None,
            reply_reference: Some(crate::channels::ReplyReference {
                author: "alice".to_string(),
                snippet: "quote".to_string(),
            }),
        };
        let msg = DisplayMessage::from(entry);

        assert_eq!(
            msg.reply_reference.as_ref().map(|r| r.author.as_str()),
            Some("alice"),
            "From<ChatHistoryEntry> must carry the entry's reply_reference"
        );
        assert_eq!(
            msg.reply_reference.as_ref().map(|r| r.snippet.as_str()),
            Some("quote")
        );
    }

    #[test]
    fn test_reply_reference_for_user_message() {
        let reply = reply_reference_for(
            ChatDirection::User,
            None,
            "Hello <world>\n\nwith [IMAGE:/tmp/photo.png]",
        );

        // A user message is the admin's own; the snippet is normalized (angle
        // brackets stripped, newlines collapsed — each `\n` becomes a single
        // space, so `\n\n` yields two — media marker mapped).
        assert_eq!(reply.author, crate::users::ADMIN_USER_NAME);
        assert_eq!(reply.snippet, "Hello world  with [Photo]");
    }

    #[test]
    fn test_reply_reference_for_agent_message() {
        let reply = reply_reference_for(
            ChatDirection::Agent,
            Some("analyst_3"),
            "line one\n<line two> [IMAGE:/tmp/x.png] and a long tail",
        );

        // Agent direction uses the shared author-label derivation (suffix
        // stripped + role label); snippet normalized the same way.
        assert_eq!(reply.author, "Analyst");
        assert_eq!(reply.snippet, "line one line two [Photo] and a long tail");
    }

    #[test]
    fn test_reply_reference_snippet_caps_at_100() {
        let long = "a".repeat(150);
        let reply = reply_reference_for(ChatDirection::User, None, &long);

        assert!(
            reply.snippet.chars().count() <= crate::channels::reply::REPLY_SNIPPET_MAX_CHARS,
            "snippet must be capped at 100 chars, got {}",
            reply.snippet.chars().count()
        );
        assert!(
            reply.snippet.ends_with('…'),
            "truncated snippet ends with …"
        );
    }

    #[test]
    fn test_replace_optimistic_reply_reference_fallback() {
        // A confirmed event carrying no reference falls back to the target
        // optimistic bubble's own reference.
        let mut state = make_home_state("ws1");
        state.messages.push(DisplayMessage {
            id: None,
            message_id: "opt-9".to_string(),
            content: "(placeholder)".to_string(),
            direction: ChatDirection::User,
            agent_role: None,
            timestamp: None,
            reply_reference: Some(crate::channels::ReplyReference {
                author: "bob".to_string(),
                snippet: "original quote".to_string(),
            }),
            md_items: Vec::new(),
            is_optimistic: true,
        });

        let task = state.replace_optimistic(
            Some("opt-9"),
            "real-9",
            "Hello!",
            ChatDirection::User,
            None,
            None,
            None,
        );
        assert!(task.is_some(), "expected replacement");
        let replaced = &state.messages[0];
        assert_eq!(
            replaced.reply_reference.as_ref().map(|r| r.author.as_str()),
            Some("bob"),
            "event with no reference must keep the optimistic bubble's"
        );

        // A confirmed reference wins over the optimistic one.
        let mut state = make_home_state("ws1");
        state.messages.push(make_msg(
            "opt-10",
            "(placeholder)",
            ChatDirection::User,
            None,
            true,
        ));
        let task = state.replace_optimistic(
            Some("opt-10"),
            "real-10",
            "Hi",
            ChatDirection::User,
            None,
            None,
            Some(crate::channels::ReplyReference {
                author: "carol".to_string(),
                snippet: "confirmed".to_string(),
            }),
        );
        assert!(task.is_some(), "expected replacement");
        let replaced = &state.messages[0];
        assert_eq!(
            replaced.reply_reference.as_ref().map(|r| r.author.as_str()),
            Some("carol")
        );
    }

    // ------------------------------------------------------------------
    // Selection (shift+click / drag)
    // ------------------------------------------------------------------

    #[test]
    fn test_select_to_creates_selection() {
        let mut state = make_home_state("ws1");
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
        let mut state = make_home_state("ws1");
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
        let mut state = make_home_state("ws1");
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
