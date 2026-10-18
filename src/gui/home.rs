//! Home page — native GUI chat interface for the app's own admin identity.
//!
//! The page is the app's own single conversation with the Assistant, living in
//! the admin's personal workspace. The footer workspace picker selects the
//! workspace the work that follows runs in — never the conversation, which is
//! one chat in real time with full markdown rendering and typing indicators.

use crate::ChatDirection;
use crate::Role;
use crate::channels::ReplyReference;
use crate::channels::chat_history::ChatHistoryEntry;
use crate::channels::reply::{agent_author_label, normalize_reply_text};
use chrono::{DateTime, Utc};
use futures_util::SinkExt;
use iced::widget::rule;
#[cfg(target_os = "macos")]
use iced::widget::tooltip;
use iced::widget::{Column, Id, Space, button, column, container, row, scrollable, text};
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
    /// Row id in storage (`Some` for entries read from chat_history) or `None`
    /// for what is only on screen — a live arrival, or the message the user is
    /// sending. Only the `Some` entries are the stored history a load replaces.
    pub id: Option<i64>,
    pub message_id: String,
    pub content: String,
    pub direction: ChatDirection,
    pub agent_role: Option<String>,
    /// Timestamp for the relative-time label and for load reconciliation: the
    /// stored row's own, or — for a message that is only on screen — the instant
    /// of the event that produced it.
    pub timestamp: Option<String>,
    /// Reference to the message this message replies to, rendered as a quote
    /// header. `None` for messages with no reply target.
    pub reply_reference: Option<ReplyReference>,
    /// Pre-parsed markdown items for rendering.
    pub md_items: Vec<iced::widget::markdown::Item>,
    /// `[FILE:…]` markers of the raw content, in order.
    pub file_markers: Vec<super::media_markers::FileMarker>,
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
    let (md_items, file_markers) = if direction == ChatDirection::Divider {
        (Vec::new(), Vec::new())
    } else {
        let processed = super::media_markers::preprocess(&content);
        let processed = super::markdown_breaks::hard_breaks(&processed);
        (
            markdown::parse(&processed).collect(),
            super::media_markers::file_markers(&content),
        )
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
        file_markers,
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
/// timestamp. Callers pass the user and the conversation's own workspace.
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

/// Composer mic button: starts a voice message recording for the Assistant.
///
/// macOS-only, like the rest of the audio UI — the button lives in the composer
/// action toolbar and is the entry point of the manual recording flow whose
/// state (stop + send / stop + discard, transcribing) is rendered by
/// [`recording_popup`].
#[cfg(target_os = "macos")]
fn mic_control(voice_status: &crate::audio::voice::VoiceStatus) -> Element<'static, HomeMessage> {
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

    let mic_tooltip = if transcription_disabled {
        "voice recording unavailable — local transcription is disabled"
    } else {
        "record voice message"
    };
    super::widgets::icon_tooltip_button(
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
    )
}

/// Composer recording popup: the stop + send / stop + discard controls for a
/// mic-button recording, or a passive "Transcribing…" indicator while that
/// recording is being transcribed (no stop controls — the ASR is finalizing).
/// Collapses to zero height when neither state applies.
#[cfg(target_os = "macos")]
fn recording_popup(
    voice_status: &crate::audio::voice::VoiceStatus,
) -> Element<'static, HomeMessage> {
    let recording = matches!(
        voice_status,
        crate::audio::voice::VoiceStatus::RecordingManual
    );
    // The Transcribing status is shared between the manual and wake-word
    // paths; only a mic-button recording shows the composer popup.
    let transcribing = matches!(voice_status, crate::audio::voice::VoiceStatus::Transcribing)
        && crate::audio::voice::is_manual_recording();

    if recording {
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
    }
}

#[derive(Debug, Clone)]
pub enum HomeMessage {
    /// The selected workspace changed (footer picker, Settings, or the
    /// read-back of the stored selection). Recorded only: the conversation does
    /// not depend on it.
    WorkspaceChanged(String),
    /// Text editor content changed.
    InputChanged(super::editor_widget::EditorAction),
    /// Send button pressed or Enter key in editor.
    SendMessage,
    /// The whole conversation loaded from the store — what the cap keeps for it
    /// and nothing else, in one read. `generation` is the load counter the read
    /// was issued under: a result whose generation is no longer current belongs
    /// to a superseded load and is dropped.
    HistoryLoaded {
        entries: Vec<ChatHistoryEntry>,
        generation: u64,
    },
    /// History load failed. Carries the generation the read was issued under, so
    /// a failure from a superseded load is dropped.
    HistoryLoadError { generation: u64, error: String },
    /// Live chat event from CHAT_BROADCAST subscription.
    ChatEvent(crate::ChatEvent),
    /// Stream lagged — resync needed.
    StreamLagged,
    /// Scroll position changed in the chat scrollable.
    ScrollChanged(scrollable::Viewport),
    /// Markdown link was clicked.
    LinkClicked(String),
    /// A `[FILE:…]` reveal attempted from a transcript finished. `Ok` needs no
    /// further action; an error becomes an error toast.
    FileRevealed(Result<(), String>),
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
    #[cfg(target_os = "macos")]
    StartVoiceRecording,
    /// Recording popup: stop recording, transcribe, and send the voice message.
    #[cfg(target_os = "macos")]
    StopVoiceRecordingSend,
    /// Recording popup: stop recording and discard the voice message.
    #[cfg(target_os = "macos")]
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
    /// The footer workspace picker's selection, recorded for the send path (the
    /// workspace the work that follows runs in). The conversation does not
    /// depend on it.
    selected_workspace: Option<String>,
    /// Load counter for the conversation read: advanced by every load. A result
    /// (success or failure) carrying a superseded generation is dropped, so a
    /// slow load can never overwrite a newer one.
    history_gen: u64,
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
    /// Whether the Phase-1 scripted onboarding scenario is active (shown while
    /// `provider_configured() == false`).
    onboarding_script_active: bool,
    /// Generation counter for stale sending timeout detection.
    sending_gen: u64,
    /// Whether auto-scroll is enabled (user is scrolled to the bottom).
    auto_scroll_enabled: bool,
    /// Undo/redo stack for the chat input text editor.
    undo_stack: super::common::UndoStack,
    /// Captured {author, snippet} of the message being replied to. `Some`
    /// while a reply preview is shown above the composer; cleared on send,
    /// cancel, or chat clear.
    pending_reply: Option<ReplyReference>,
    /// Persisted composer draft (text + pending reply-to).
    drafts: Arc<crate::channels::chat_draft::DraftStore>,
    /// Debounce state for the asynchronous draft persistence.
    draft_save: super::common::DebounceState,
    /// Persistent failure of the last conversation read. Rendered in place of
    /// the empty-chat hint so a failed read is never mistaken for a legitimately
    /// empty history; cleared on the next successful load.
    history_error: Option<String>,
}

/// The instant a stored timestamp names, or `None` when it is absent or not a
/// timestamp the clock produced. Compared as instants rather than as text: the
/// strings `db::now()` writes vary in their fractional digits and carry an
/// offset, so their text order is the clock's order only by accident of the
/// format.
fn instant(timestamp: Option<&str>) -> Option<DateTime<Utc>> {
    timestamp.and_then(|ts| crate::db::parse_utc_timestamp(ts).ok())
}

impl HomeState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            selected_workspace: None,
            history_gen: 0,
            messages: Vec::new(),
            seen_ids: HashSet::new(),
            editor_content: super::editor_widget::EditorBuffer::with_text(
                "",
                Some(super::highlight::HighlightLanguage::Markdown),
            ),
            sending: false,
            typing: false,
            typing_tick_state: 0,
            onboarding_script_active: false,
            sending_gen: 0,
            auto_scroll_enabled: true,
            undo_stack: super::common::UndoStack::new(),
            pending_reply: None,
            drafts: crate::channels::chat_draft::DraftStore::global().clone(),
            draft_save: super::common::DebounceState::new(),
            history_error: None,
        }
    }

    /// Whether `workspace` is the admin's own personal workspace (`personal:admin`)
    /// — the only workspace the conversation lives in, and therefore the test for
    /// whether a live event belongs to it.
    fn is_admin_personal_workspace(workspace: &str) -> bool {
        crate::users::personal_user_name(workspace).is_some_and(crate::users::is_admin_name)
    }

    /// The workspace the conversation belongs to: the admin's own personal
    /// space. The Assistant is pinned to it, so everything this chat receives —
    /// the admin's own text, agent replies, the notice written when a turn fails
    /// — is tagged with it, and nothing else is.
    fn conversation_workspace() -> String {
        crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME)
    }

    /// Open the app's own conversation: bring the persisted composer draft back
    /// and load the conversation. This is the app-opening load route — the load
    /// hangs off the screen opening, never off a selection arriving.
    pub fn open(&mut self) -> Task<HomeMessage> {
        self.restore_draft();
        self.refresh_history()
    }

    /// The load counter: a load was issued exactly when it advanced. Read by the
    /// boot test, which proves the app opening is a load route.
    #[cfg(test)]
    pub(crate) fn history_generation(&self) -> u64 {
        self.history_gen
    }

    /// Load the conversation — the admin's own single chat, in their personal
    /// workspace, independent of the selected workspace. One read carries the
    /// whole of what the cap keeps for it: there is nothing older to fetch.
    fn refresh_history(&mut self) -> Task<HomeMessage> {
        self.history_gen = self.history_gen.wrapping_add(1);
        let generation = self.history_gen;
        let workspace = Self::conversation_workspace();
        Task::perform(
            async move {
                let store = crate::channels::chat_history::store();
                store
                    .load_conversation(crate::users::ADMIN_USER_NAME, &workspace)
                    .await
                    .map_err(|e| e.to_string())
            },
            move |result| match result {
                Ok(entries) => HomeMessage::HistoryLoaded {
                    entries,
                    generation,
                },
                Err(e) => HomeMessage::HistoryLoadError {
                    generation,
                    error: e,
                },
            },
        )
    }

    /// Replace the stored history with `entries` — the conversation just read.
    /// What is on screen but not stored (the message being sent, the scripted
    /// introduction) is kept, and nothing is ever shown twice; no task is issued
    /// (the caller batches the snap and the onboarding check).
    fn replace_stored_history(&mut self, entries: Vec<ChatHistoryEntry>) {
        let mut kept: Vec<DisplayMessage> = std::mem::take(&mut self.messages)
            .into_iter()
            .filter(|m| m.id.is_none())
            .collect();
        let stored: Vec<DisplayMessage> = entries.into_iter().map(DisplayMessage::from).collect();
        // Anything the page carries in its own right goes in favour of the
        // stored copy: a live arrival and a pending send already confirmed both
        // show under the pipeline's id.
        let stored_ids: HashSet<&str> = stored.iter().map(|m| m.message_id.as_str()).collect();
        // A pending send still shows under its local id, and a lag can drop
        // exactly its confirmation — leaving the bubble while the page brings the
        // row back under the pipeline's id. Matching the two is a heuristic: the
        // row of that send is a page row with the user's own text, written from
        // here, so not older than the bubble it confirms, while an older row with
        // the same text is the user having sent that text before. A bubble the
        // heuristic does not match — its row never written, or stored under a text
        // enrichment rewrote — goes when its confirmation arrives, which drops it
        // if the page already carries the row.
        kept.retain(|msg| {
            if stored_ids.contains(msg.message_id.as_str()) {
                return false;
            }
            // Only a pending send is reconciled, and only against a bubble that
            // carries the send's own instant — with nothing to compare, the
            // bubble stays.
            if !msg.is_optimistic {
                return true;
            }
            let Some(sent_at) = instant(msg.timestamp.as_deref()) else {
                return true;
            };
            !stored.iter().any(|s| {
                s.direction == ChatDirection::User
                    && s.content == msg.content
                    && instant(s.timestamp.as_deref()).is_some_and(|at| at >= sent_at)
            })
        });
        // What is kept goes back where it belongs: both a scripted transient and
        // a pending send carry the instant of the event that produced them, and
        // go before the first stored message that is newer.
        let mut merged = stored;
        for msg in kept {
            let at = instant(msg.timestamp.as_deref()).and_then(|ts| {
                merged
                    .iter()
                    .position(|m| instant(m.timestamp.as_deref()).is_some_and(|t| t > ts))
            });
            merged.insert(at.unwrap_or(merged.len()), msg);
        }
        self.messages = merged;
        self.seen_ids.clear();
        self.seen_ids
            .extend(self.messages.iter().map(|m| m.message_id.clone()));
    }

    /// Snapshot the current composer state (text + pending reply) into the
    /// draft store and schedule a debounced persist after `delay_ms`.
    ///
    /// The in-memory mutation happens synchronously at schedule time — the
    /// file write lags behind the debounce, so the in-memory draft is always
    /// current.
    fn capture_draft(&mut self, delay_ms: u64) -> Task<HomeMessage> {
        if self.onboarding_script_active {
            return Task::none();
        }
        let text = self.editor_content.text();
        let reply = self.pending_reply.clone();
        self.drafts.set(text, reply);
        self.draft_save
            .trigger(delay_ms)
            .map(|generation| HomeMessage::DraftSaveSettled { generation })
    }

    /// Bring the persisted draft back into the composer; called once when the
    /// chat starts.
    fn restore_draft(&mut self) {
        let draft = self.drafts.get();
        self.editor_content.set_text(&draft.text);
        self.pending_reply = draft.reply;
    }

    /// Remove the stored draft and schedule a persist — used on send, after the
    /// composer has been cleared.
    fn drop_draft(&mut self) {
        self.drafts.remove();
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
            let workspace = Self::conversation_workspace();
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
    /// and returns `Some(snap_task)` so the caller can early-return. When the
    /// stored copy is already on screen (a load brought its row in under the
    /// canonical ID) the bubble is the same message and is dropped instead.
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
                // The canonical copy is already shown, so the bubble is the same
                // message a second time — it goes, the stored copy stays.
                if self.seen_ids.contains(message_id) {
                    self.messages.remove(pos);
                    self.sending = false;
                    return Some(self.maybe_snap());
                }
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
    /// Does nothing when `workspace` is not the conversation's own: live work on
    /// project tickets must not clear the conversation's typing/sending state.
    fn update_sending_state(&mut self, direction: ChatDirection, user_name: &str, workspace: &str) {
        if !Self::is_admin_personal_workspace(workspace) {
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

    /// Append a chat message if it belongs to the conversation: the admin's own
    /// chat in their personal workspace. Nothing else is ever shown here, so
    /// what is shown live can never disagree with what a later load reads.
    ///
    /// Does nothing when `user_name` is not the admin, or when `workspace` is
    /// not the conversation's own (see [`Self::is_admin_personal_workspace`]).
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
        if !Self::is_admin_personal_workspace(workspace) {
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
                            &msg.file_markers,
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

        // One status snapshot per render, so the mic button in the composer
        // toolbar below and the recording popup cannot disagree.
        #[cfg(target_os = "macos")]
        let voice_status = crate::audio::voice::get_status();

        // ── Input area ───────────────────────────────────────────
        // Controls for the composer action toolbar: the mic button.
        #[cfg(target_os = "macos")]
        let controls: Vec<Element<'_, HomeMessage>> = vec![mic_control(&voice_status)];
        #[cfg(not(target_os = "macos"))]
        let controls: Vec<Element<'_, HomeMessage>> = Vec::new();

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
        #[cfg(target_os = "macos")]
        let recording_popup = recording_popup(&voice_status);
        #[cfg(not(target_os = "macos"))]
        let recording_popup: Element<'_, HomeMessage> = Space::new().height(0).into();

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
                // Recorded for the send path only: the selection decides the
                // workspace the work that follows runs in, and the conversation
                // is not its to touch.
                self.selected_workspace = Some(ws_name);
                Task::none()
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
                generation,
            } => {
                // A superseded load belongs to a transcript the page has left.
                if generation != self.history_gen {
                    return Task::none();
                }
                self.history_error = None;
                self.replace_stored_history(entries);
                let onboarding = self.maybe_start_onboarding();
                Task::batch([self.maybe_snap(), onboarding])
            }
            HomeMessage::HistoryLoadError { generation, error } => {
                tracing::warn!(error = %error, "Home: failed to load chat history");
                // A failure from a superseded load belongs to a transcript the
                // page has left: it must not paint itself over the current one.
                if generation == self.history_gen {
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
                // A load in flight describes the transcript this clear just
                // dropped: superseding it keeps it from landing in front of the
                // reload.
                self.history_gen = self.history_gen.wrapping_add(1);
                // A deliberate clear empties the screen (including a message
                // being sent) and re-arms the no-provider script, so the reset
                // conversation is shown from its own beginning at the bottom.
                self.onboarding_script_active = false;
                self.auto_scroll_enabled = true;
                // Capture synchronously (not from the ChatCleared callback): the
                // reply is dropped but the draft text is kept.
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
                // Belt and suspenders: ClearChat already cleared the message
                // list and this callback reloads the conversation, so drop any
                // pending reply explicitly.
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

                    // 2. Clear sending/typing state based on direction, sender, and workspace.
                    //    Before the dedup: a confirmation whose row a load already
                    //    brought in is dropped below, and it must still free the
                    //    composer rather than leave the send guard holding it.
                    self.update_sending_state(direction, &user_name, &workspace);

                    // 3. Deduplicate against already-seen IDs.
                    if self.try_dedup(&message_id) {
                        return Task::none();
                    }

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
                    // for the admin in the conversation's own workspace.
                    if crate::users::is_admin_name(&user_name)
                        && Self::is_admin_personal_workspace(&workspace)
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
                // Resync through the one load path. Clearing `sending` is the
                // safety net for an agent response the lag swallowed —
                // otherwise the send button stays stuck.
                self.sending = false;
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
            HomeMessage::Toast(_) => {
                // Intercepted by the Dashboard (Toast → toast stack). No-op
                // fallback.
                Task::none()
            }
            HomeMessage::LinkClicked(url) => {
                if let Some(target) = super::media_markers::file_link_target(&url) {
                    let target = target.to_string();
                    let workspace =
                        crate::users::personal_workspace_path(crate::users::ADMIN_USER_NAME);
                    // The reveal stats and spawns a launcher: keep it out of the
                    // update handler.
                    Task::perform(
                        async move {
                            super::media_markers::reveal_file_marker(&target, Some(&workspace))
                        },
                        HomeMessage::FileRevealed,
                    )
                } else {
                    super::open_url(&url);
                    Task::none()
                }
            }
            HomeMessage::FileRevealed(result) => match result {
                Ok(()) => Task::none(),
                Err(error) => Task::done(HomeMessage::Toast(ToastMessage::Error(error))),
            },
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
            #[cfg(target_os = "macos")]
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
            #[cfg(target_os = "macos")]
            HomeMessage::StopVoiceRecordingSend => {
                crate::audio::voice::send_command(
                    crate::audio::voice::VoiceCommand::StopRecordingSend,
                );
                Task::none()
            }
            #[cfg(target_os = "macos")]
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
            // Carries the instant it was sent: the load reconciles it against
            // the stored copy of the same send (see `replace_stored_history`).
            // A user bubble renders no time label, so this is not displayed.
            self.messages.push(display_message(
                None,
                opt_id.clone(),
                content.clone(),
                ChatDirection::User,
                None,
                Some(crate::db::now()),
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
                self.drop_draft();
                return Task::none();
            }
        } else {
            tracing::error!("Home: GUI_MESSAGE_TX not initialized");
            self.sending = false;
            self.drop_draft();
            return Task::none();
        }

        // The send was queued — drop the pending reply so the preview clears.
        self.pending_reply = None;
        // The composer is cleared: remove the persisted draft so it cannot
        // resurrect on a later restore.
        self.drop_draft();

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
        let workspace = Self::conversation_workspace();
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
        // touch the real draft file.
        state.drafts = hermetic_draft_store();
        // The selection is recorded for the send path only: the conversation
        // does not depend on it.
        state.selected_workspace = Some(workspace.to_string());
        state
    }

    fn stored_entry(id: i64, message_id: &str, content: &str) -> ChatHistoryEntry {
        ChatHistoryEntry {
            id,
            message_id: message_id.to_string(),
            content: content.to_string(),
            direction: ChatDirection::User,
            agent_role: None,
            timestamp: None,
            reply_reference: None,
        }
    }

    /// A draft store backed by a unique throwaway file under one shared test
    /// temp dir, so tests never touch the real draft file.
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
    fn the_chat_has_one_draft_and_a_workspace_switch_leaves_it_alone() {
        let mut state = make_home_state("ws1");

        // Typing captures synchronously into the one draft.
        let _ = state.update(HomeMessage::InputChanged(EditorAction::Insert('h')));
        assert_eq!(state.drafts.get().text, "h");

        // The workspace swap does not touch the composer: the text stays, and
        // the draft is still the same single draft.
        let _ = state.update(HomeMessage::WorkspaceChanged("ws2".to_string()));
        assert_eq!(state.editor_content.text(), "h");
        let _ = state.update(HomeMessage::WorkspaceChanged("ws1".to_string()));
        assert_eq!(state.editor_content.text(), "h");
        assert_eq!(state.drafts.get().text, "h");
    }

    #[test]
    fn restore_brings_back_the_text_and_its_pending_reply() {
        let mut state = make_home_state("ws1");
        let reply = reply_reference_for(ChatDirection::Agent, Some("assistant"), "hi there");
        state.drafts.set("typed".to_string(), Some(reply.clone()));

        // The app's opening brings the stored draft back; the boot selection
        // push lands after it and leaves the composer as it is.
        let _ = state.open();
        let _ = state.update(HomeMessage::WorkspaceChanged("ws2".to_string()));
        assert_eq!(state.editor_content.text(), "typed");
        assert_eq!(state.pending_reply, Some(reply));
        assert_eq!(state.history_gen, 1, "the opening issues exactly one load");

        // ...and a later picker change leaves it alone.
        let _ = state.update(HomeMessage::WorkspaceChanged("ws1".to_string()));
        assert_eq!(state.editor_content.text(), "typed");
        assert!(state.pending_reply.is_some());
    }

    #[test]
    fn failed_reads_stay_set_until_a_successful_load() {
        let mut state = HomeState::new();
        let generation = state.history_gen;

        // A failed read of the current load is retained.
        let _ = state.update(HomeMessage::HistoryLoadError {
            generation,
            error: "db down".to_string(),
        });
        assert_eq!(state.history_error.as_deref(), Some("db down"));

        // A failure from a superseded load is dropped.
        let _ = state.update(HomeMessage::HistoryLoadError {
            generation: generation.wrapping_sub(1),
            error: "stale".to_string(),
        });
        assert_eq!(state.history_error.as_deref(), Some("db down"));

        // A superseded successful load must not replace the window nor clear
        // the current failure.
        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: Vec::new(),
            generation: generation.wrapping_sub(1),
        });
        assert_eq!(
            state.history_error.as_deref(),
            Some("db down"),
            "a superseded successful load must not clear the current failure"
        );
        assert!(
            state.messages.is_empty(),
            "a superseded load must not replace the window"
        );

        // A successful load of the current generation clears it.
        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: Vec::new(),
            generation,
        });
        assert!(state.history_error.is_none());

        // A clear drops the failure with the transcript it described.
        let _ = state.update(HomeMessage::HistoryLoadError {
            generation,
            error: "db down".to_string(),
        });
        assert!(state.history_error.is_some());
        // ...and supersedes a load still in flight, so nothing lands behind it.
        let in_flight = state.history_gen;
        let _ = state.update(HomeMessage::ClearChat);
        assert!(state.history_error.is_none());
        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: vec![stored_entry(1, "old-1", "old")],
            generation: in_flight,
        });
        assert!(
            state.messages.is_empty(),
            "a pre-clear load must not land behind the clear"
        );
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
            file_markers: Vec::new(),
            is_optimistic,
        }
    }

    /// The admin's own message as the live stream delivers it: the confirmation
    /// of a composer send (`optimistic_id`) or a plain echo.
    fn own_message(message_id: &str, optimistic_id: Option<&str>) -> HomeMessage {
        HomeMessage::ChatEvent(crate::ChatEvent::Message {
            message_id: message_id.to_string(),
            user_name: crate::users::ADMIN_USER_NAME.to_string(),
            content: "hello".to_string(),
            direction: ChatDirection::User,
            timestamp: crate::db::now(),
            channel: "gui".to_string(),
            agent_role: None,
            workspace: HomeState::conversation_workspace(),
            optimistic_id: optimistic_id.map(std::string::ToString::to_string),
            transient: false,
            reply_reference: None,
        })
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

    #[test]
    fn test_replace_optimistic_drops_a_bubble_the_page_already_carries() {
        let mut state = make_home_state("ws1");
        let mut row = make_msg("pipeline-9", "hello", ChatDirection::User, None, false);
        row.id = Some(9);
        state.messages.push(row);
        state
            .messages
            .push(make_msg("opt-1", "hello", ChatDirection::User, None, true));
        state.seen_ids.insert("pipeline-9".to_string());
        state.sending = true;

        let task = state.replace_optimistic(
            Some("opt-1"),
            "pipeline-9",
            "hello",
            ChatDirection::User,
            None,
            None,
            None,
        );

        assert!(task.is_some(), "the removal still snaps to the end");
        assert_eq!(state.messages.len(), 1, "one message, not two");
        assert_eq!(state.messages[0].message_id, "pipeline-9");
        assert!(!state.sending, "sending should be cleared");
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
        let personal = crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME);
        let cases = [
            (
                "agent in the conversation",
                ChatDirection::Agent,
                crate::users::ADMIN_USER_NAME,
                personal.as_str(),
                false,
                false,
            ),
            (
                "user echo in the conversation",
                ChatDirection::User,
                crate::users::ADMIN_USER_NAME,
                personal.as_str(),
                false,
                true,
            ),
            (
                "another user",
                ChatDirection::Agent,
                "bob",
                personal.as_str(),
                true,
                true,
            ),
            (
                "a project workspace",
                ChatDirection::Agent,
                crate::users::ADMIN_USER_NAME,
                "ws1",
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
            &crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
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
            "a project workspace's message is not the conversation's and must never be appended"
        );
    }

    #[test]
    fn test_append_message_agent_response() {
        let mut state = make_home_state("ws1");

        state.append_message(
            crate::users::ADMIN_USER_NAME,
            &crate::users::personal_workspace_name(crate::users::ADMIN_USER_NAME),
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
    // one conversation, scoped to the admin's personal workspace
    // ------------------------------------------------------------------

    #[test]
    fn a_selection_change_leaves_the_conversation_alone() {
        let mut state = make_home_state("ws1");
        state
            .messages
            .push(make_msg("m1", "one", ChatDirection::User, None, false));
        state.messages.push(make_msg(
            "m2",
            "two",
            ChatDirection::Agent,
            Some("assistant"),
            false,
        ));
        state.seen_ids = state
            .messages
            .iter()
            .map(|m| m.message_id.clone())
            .collect();
        state.sending = true;
        state.typing = true;
        state.auto_scroll_enabled = false; // the user has scrolled up
        let generation = state.history_gen;

        // A selection change is recorded and nothing else — in any order, back
        // and forth.
        for picker in ["ws2", "ws1", "personal:admin"] {
            let task = state.update(HomeMessage::WorkspaceChanged(picker.to_string()));
            assert_eq!(task.units(), 0, "a selection change must issue no read");
            assert_eq!(state.selected_workspace.as_deref(), Some(picker));
            assert_eq!(generation, state.history_gen, "no load was issued");
            let ids_and_contents: Vec<(&str, &str)> = state
                .messages
                .iter()
                .map(|m| (m.message_id.as_str(), m.content.as_str()))
                .collect();
            assert_eq!(ids_and_contents, [("m1", "one"), ("m2", "two")]);
            assert!(
                state.sending && state.typing,
                "live-work state is not the selection's to reset"
            );
            assert!(
                !state.auto_scroll_enabled,
                "a scrolled-up view must not be snapped back"
            );
        }
    }

    #[test]
    fn live_work_follows_the_conversation_not_the_selection() {
        let personal = HomeState::conversation_workspace();
        let message = |workspace: &str, id: &str| {
            HomeMessage::ChatEvent(crate::ChatEvent::Message {
                message_id: id.to_string(),
                user_name: crate::users::ADMIN_USER_NAME.to_string(),
                content: format!("from {workspace}"),
                direction: ChatDirection::Agent,
                timestamp: crate::db::now(),
                channel: "gui".to_string(),
                agent_role: Some("assistant".to_string()),
                workspace: workspace.to_string(),
                optimistic_id: None,
                transient: false,
                reply_reference: None,
            })
        };
        let typing = |workspace: &str| {
            HomeMessage::ChatEvent(crate::ChatEvent::Typing {
                user_name: crate::users::ADMIN_USER_NAME.to_string(),
                is_typing: true,
                workspace: workspace.to_string(),
            })
        };

        // The project the picker selects is where the work that follows runs —
        // not where the conversation's live traffic lands.
        let mut state = make_home_state("ws1");
        state.sending = true;
        let _ = state.update(message("ws1", "project-1"));
        let _ = state.update(typing("ws1"));
        assert!(
            state.messages.is_empty(),
            "a project's message is never the conversation's"
        );
        assert!(
            !state.typing && state.sending,
            "a project's live work is not the conversation's cue either"
        );

        let _ = state.update(message(&personal, "own-1"));
        let _ = state.update(typing(&personal));
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].message_id, "own-1");
        assert!(state.typing, "the conversation's own typing cue is shown");
        assert!(!state.sending, "its own live traffic clears the send");
    }

    #[test]
    fn a_reload_replaces_the_stored_history_and_keeps_what_is_not_stored() {
        let mut state = HomeState::new();
        let mut old = make_msg("old-1", "old", ChatDirection::User, None, false);
        old.id = Some(1);
        state.messages.push(make_msg(
            "live-1",
            "live",
            ChatDirection::Agent,
            Some("assistant"),
            false,
        ));
        state.messages.push(old);
        state.messages.push(make_msg(
            "opt-1",
            "optimistic",
            ChatDirection::User,
            None,
            true,
        ));
        state.seen_ids = state
            .messages
            .iter()
            .map(|m| m.message_id.clone())
            .collect();

        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: vec![
                stored_entry(1, "old-1", "old"),
                stored_entry(2, "new-2", "new"),
            ],
            generation: state.history_gen,
        });

        // The stored page in chronological order, then what was kept, in its
        // previous order.
        let ids_and_contents: Vec<(&str, &str)> = state
            .messages
            .iter()
            .map(|m| (m.message_id.as_str(), m.content.as_str()))
            .collect();
        assert_eq!(
            ids_and_contents,
            [
                ("old-1", "old"),
                ("new-2", "new"),
                ("live-1", "live"),
                ("opt-1", "optimistic"),
            ]
        );
        assert_eq!(state.messages[0].id, Some(1));
        assert_eq!(state.messages[1].id, Some(2));
        assert!(state.messages[2].id.is_none());
        assert!(state.messages[3].id.is_none() && state.messages[3].is_optimistic);
        let unique: HashSet<&str> = state
            .messages
            .iter()
            .map(|m| m.message_id.as_str())
            .collect();
        assert_eq!(unique.len(), state.messages.len(), "no duplicated ids");
    }

    #[test]
    fn a_reload_puts_a_kept_transient_message_back_among_the_stored_ones() {
        let mut state = HomeState::new();
        // The scripted onboarding exchange is transient — never stored — and
        // carries the timestamp of the event that produced it.
        let mut transient = make_msg(
            "intro-1",
            "intro",
            ChatDirection::Agent,
            Some("assistant"),
            false,
        );
        transient.timestamp = Some("2026-01-01T10:00:00.000000000+00:00".to_string());
        state.messages.push(transient);

        let mut reply = stored_entry(5, "reply", "the reply");
        reply.timestamp = Some("2026-01-01T10:00:05.000000000+00:00".to_string());
        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: vec![reply],
            generation: state.history_gen,
        });

        let ids: Vec<&str> = state
            .messages
            .iter()
            .map(|m| m.message_id.as_str())
            .collect();
        assert_eq!(
            ids,
            ["intro-1", "reply"],
            "a kept message stays in front of the stored one it preceded"
        );
    }

    #[test]
    fn a_lost_confirmation_does_not_double_a_pending_send() {
        let sent = "2026-01-01T10:00:00.000000000+00:00";
        let mut confirmed = stored_entry(9, "pipeline-9", "hello");
        confirmed.timestamp = Some("2026-01-01T10:00:01.000000000+00:00".to_string());

        let mut state = HomeState::new();
        let mut bubble = make_msg("opt-1", "hello", ChatDirection::User, None, true);
        bubble.timestamp = Some(sent.to_string());
        state.messages.push(bubble);
        state.messages.push(make_msg(
            "live-1",
            "later",
            ChatDirection::Agent,
            Some("assistant"),
            false,
        ));

        // The resync page brings the pending send back under the pipeline's own
        // id — its confirmation was dropped with the lag.
        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: vec![confirmed],
            generation: state.history_gen,
        });
        let ids: Vec<&str> = state
            .messages
            .iter()
            .map(|m| m.message_id.as_str())
            .collect();
        assert_eq!(
            ids,
            ["pipeline-9", "live-1"],
            "the pending send appears once, as the stored row"
        );

        // A row of the same text written before the send is a message of its
        // own, so the pending send is not swallowed by it.
        let mut state = HomeState::new();
        let mut bubble = make_msg("opt-2", "hello", ChatDirection::User, None, true);
        bubble.timestamp = Some(sent.to_string());
        state.messages.push(bubble);
        let mut older_row = stored_entry(1, "old-1", "hello");
        older_row.timestamp = Some("2026-01-01T09:00:00.000000000+00:00".to_string());
        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: vec![older_row],
            generation: state.history_gen,
        });
        let ids: Vec<&str> = state
            .messages
            .iter()
            .map(|m| m.message_id.as_str())
            .collect();
        assert_eq!(
            ids,
            ["old-1", "opt-2"],
            "an older row of the same text is not the pending send's copy"
        );
    }

    #[test]
    fn a_deduplicated_confirmation_still_frees_the_composer() {
        // The row is already on screen and the bubble is gone, so the
        // confirmation only repeats what is shown: it adds nothing and still
        // releases the send guard.
        let mut state = HomeState::new();
        state.sending = true;
        let mut row = make_msg("pipeline-9", "hello", ChatDirection::User, None, false);
        row.id = Some(9);
        state.messages.push(row);
        state.seen_ids.insert("pipeline-9".to_string());

        let _ = state.update(own_message("pipeline-9", None));
        assert_eq!(state.messages.len(), 1);
        assert!(
            !state.sending,
            "a deduplicated confirmation must not block the composer"
        );
    }

    #[test]
    fn a_reload_never_doubles_history() {
        let mut state = HomeState::new();
        state.messages.push(make_msg(
            "in-flight",
            "text",
            ChatDirection::Agent,
            Some("assistant"),
            false,
        ));
        state.seen_ids.insert("in-flight".to_string());

        // A live arrival whose copy the loaded page also carries appears
        // exactly once — the stored copy, not the live one.
        let _ = state.update(HomeMessage::HistoryLoaded {
            entries: vec![stored_entry(5, "in-flight", "text")],
            generation: state.history_gen,
        });
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].id, Some(5));
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
            file_markers: Vec::new(),
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
