pub(crate) mod chat_draft;
pub(crate) mod chat_history;
mod enrichment;
pub(crate) mod reply;
pub mod telegram;
pub use enrichment::{EnrichmentStrategy, enrich_links, enrich_message, has_only_audio_markers};
pub use reply::{ReplyReference, apply_reply_marker};
pub use telegram::mirror_gui_message_to_telegram;

use crate::channels::chat_history::ChatHistoryInsert;
use crate::db;
use crate::util::UnwrapPoison;
use crate::{Channel, ChannelMessage, ChatDirection, SendMessage};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const CHANNEL_TYPING_REFRESH_INTERVAL_SECS: u64 = 4;

/// Entry for a single chat message that should be both broadcast to the GUI
/// dashboard and persisted to chat_history. Fields map directly to the
/// [`crate::ChatEvent::Message`] and [`ChatHistoryInsert`] parameters.
#[derive(Debug, Clone)]
struct BroadcastPersistEntry {
    user_name: String,
    channel: String,
    content: String,
    direction: ChatDirection,
    agent_role: Option<String>,
    workspace: String,
    optimistic_id: Option<String>,
    reply_reference: Option<crate::channels::ReplyReference>,
}

impl BroadcastPersistEntry {
    /// Broadcast this entry to [`crate::CHAT_BROADCAST`] and persist it to
    /// `chat_history`.
    async fn broadcast_and_persist(self) {
        debug_assert!(
            self.direction != ChatDirection::Agent || self.agent_role.is_some(),
            "BroadcastPersistEntry: direction=Agent but agent_role is None"
        );

        let message_id = crate::generate_id();
        let timestamp = db::now();

        let db_direction = match self.direction {
            ChatDirection::Agent => "agent".to_string(),
            ChatDirection::User => "user".to_string(),
            ChatDirection::Divider => {
                unreachable!("Divider markers should not go through broadcast_and_persist")
            }
        };

        broadcast_chat_event(
            &message_id,
            &self.user_name,
            &self.content,
            self.direction,
            &self.channel,
            self.agent_role.clone(),
            &self.workspace,
            self.optimistic_id.clone(),
            self.reply_reference.clone(),
            &timestamp,
            false,
        );

        let store = crate::channels::chat_history::store();
        let _ = store
            .insert(&ChatHistoryInsert {
                message_id,
                user_name: self.user_name,
                direction: db_direction,
                content: self.content,
                agent_role: self.agent_role,
                workspace: self.workspace,
                timestamp: Some(timestamp),
                reply_reference: self.reply_reference,
            })
            .await;
    }
}

/// Broadcast an agent response to CHAT_BROADCAST for live GUI display and
/// persist it to chat_history. This is the canonical entry point for all
/// agent responses — used by the per-agent consumer loop in
/// [`crate::agent::message_router`] and by the raw reply-target delivery path.
///
/// TTS audio playback is handled separately by [`crate::audio::tts::init_listener()`],
/// which subscribes to [`CHAT_BROADCAST`](crate::CHAT_BROADCAST) and triggers
/// speech for matching agent messages.  This function does not itself invoke
/// any TTS logic.
///
/// Takes explicit `user_name` (canonical user name), `channel` (e.g. "telegram", "gui"),
/// and primitive fields — does **not** depend on [`crate::SendMessage`], so it can be used
/// from the per-agent consumer loop which works from [`crate::users::UserRecord`].
pub(crate) async fn broadcast_and_persist_agent_response(
    user_name: &str,
    channel: &str,
    content: &str,
    agent_role: Option<String>,
    workspace: &str,
) {
    BroadcastPersistEntry {
        user_name: user_name.to_string(),
        channel: channel.to_string(),
        content: content.to_string(),
        direction: ChatDirection::Agent,
        agent_role, // moved — no clone needed
        workspace: workspace.to_string(),
        optimistic_id: None,   // agent messages must not carry one
        reply_reference: None, // agent responses never carry a reply reference
    }
    .broadcast_and_persist()
    .await;
}

/// Send a [`ChatEvent::Message`] to the broadcast channel.
///
/// This is the single shared entry point for all broadcast operations,
/// ensuring consistent message construction across user messages, agent
/// responses, and any future message types.  The caller is responsible
/// for generating a stable [`message_id`] and [`timestamp`] if they need
/// to correlate the broadcast event with a persist operation.
///
/// `transient` marks an event destined for live GUI display only: it is
/// never persisted to chat_history (e.g. the Phase-1 scripted onboarding
/// exchange). Pass `false` for every persisted path.
#[expect(clippy::too_many_arguments)]
pub(crate) fn broadcast_chat_event(
    message_id: &str,
    user_name: &str,
    content: &str,
    direction: ChatDirection,
    channel: &str,
    agent_role: Option<String>,
    workspace: &str,
    optimistic_id: Option<String>,
    reply_reference: Option<crate::channels::ReplyReference>,
    timestamp: &str,
    transient: bool,
) {
    use crate::ChatEvent;

    if let Some(tx) = crate::CHAT_BROADCAST.get() {
        let _ = tx.send(ChatEvent::Message {
            message_id: message_id.to_string(),
            user_name: user_name.to_string(),
            content: content.to_string(),
            direction,
            timestamp: timestamp.to_string(),
            channel: channel.to_string(),
            agent_role,
            workspace: workspace.to_string(),
            optimistic_id,
            transient,
            reply_reference,
        });
    }
}

/// Broadcast a transient (non-persisted) chat event for the GUI. Used by the
/// Phase-1 scripted onboarding scenario; the event is shown live but never
/// written to chat_history.
#[expect(clippy::too_many_arguments)]
pub(crate) fn broadcast_transient_event(
    message_id: &str,
    user_name: &str,
    content: &str,
    direction: ChatDirection,
    channel: &str,
    agent_role: Option<String>,
    workspace: &str,
    optimistic_id: Option<String>,
    reply_reference: Option<crate::channels::ReplyReference>,
    timestamp: &str,
) {
    broadcast_chat_event(
        message_id,
        user_name,
        content,
        direction,
        channel,
        agent_role,
        workspace,
        optimistic_id,
        reply_reference,
        timestamp,
        true,
    );
}

/// Broadcast an incoming user message to the GUI and persist it to chat_history,
/// mirroring it to Telegram in parallel. `broadcast_content` is the enriched text
/// sent to the GUI (e.g. audio transcriptions, renderable data URIs), while
/// `persist_content` is stored in chat_history and mirrored to Telegram — the raw
/// original text (no data-URI bloat), except for audio-only messages which carry
/// the enriched transcription (icon + text) so no temp file path is persisted.
/// The GUI bubble is broadcast synchronously before the async persist + mirror
/// join begins.
///
/// Forwards [`ChannelMessage::optimistic_id`] so the GUI can replace its optimistic
/// bubble with the real one.
pub async fn broadcast_and_persist_incoming_message(
    msg: &ChannelMessage,
    broadcast_content: &str,
    persist_content: &str,
) {
    let message_id = crate::generate_id();
    let timestamp = db::now();

    broadcast_chat_event(
        &message_id,
        &msg.user_name,
        broadcast_content,
        ChatDirection::User,
        &msg.channel,
        None,
        &msg.workspace,
        msg.optimistic_id.clone(),
        msg.reply_reference.clone(),
        &timestamp,
        false,
    );

    tokio::join!(
        async {
            let store = crate::channels::chat_history::store();
            let _ = store
                .insert(&ChatHistoryInsert {
                    message_id,
                    user_name: msg.user_name.clone(),
                    direction: "user".to_string(),
                    content: persist_content.to_string(),
                    agent_role: None,
                    workspace: msg.workspace.clone(),
                    timestamp: Some(timestamp),
                    reply_reference: msg.reply_reference.clone(),
                })
                .await;
        },
        async {
            let mut mirror_msg = msg.clone();
            mirror_msg.content = persist_content.to_string();
            mirror_gui_message_to_telegram(&mirror_msg).await;
        },
    );
}

#[must_use]
pub fn spawn_scoped_typing_task(
    recipient: String,
    channel: String,
    cancellation_token: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let refresh_interval = std::time::Duration::from_secs(CHANNEL_TYPING_REFRESH_INTERVAL_SECS);
    tokio::spawn(async move {
        let Some(ch) = crate::channel_registry().get(&channel) else {
            tracing::warn!(
                channel = %channel,
                "Channel not found in registry — skipping typing indicator"
            );
            return;
        };
        let mut interval = tokio::time::interval(refresh_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                () = cancellation_token.cancelled() => break,
                _ = interval.tick() => {
                    if let Err(e) = ch.start_typing(&recipient).await {
                        tracing::debug!("Failed to start typing on {}: {e}", ch.name());
                    }
                }
            }
        }
    })
}

/// Cancel the typing task (via token) and await its completion.
pub async fn stop_typing(handle: tokio::task::JoinHandle<()>) {
    if let Err(error) = handle.await {
        tracing::error!("Typing task crashed: {error}");
    }
}

// ── GuiChannel ────────────────────────────────────────────

/// The GUI channel — always registered, even in dashboard-only mode.
///
/// Unlike Telegram which has its own async listener loop, the GUI channel uses
/// an internal mpsc pair: the Iced UI pushes `ChannelMessage` values into
/// `GUI_MESSAGE_TX`, and `listen()` reads them from the paired receiver,
/// forwarding each one into the shared pipeline `tx`.
///
/// Outgoing agent responses are broadcast to the GUI dashboard and persisted
/// to chat_history centrally via [`broadcast_and_persist_agent_response`].
/// `GuiChannel::send()` is pure transport (no-op) — all broadcast+persist
/// happens in the canonical function so every path gets consistent treatment.
pub struct GuiChannel {
    /// The internal receiver, stored so `listen()` can consume it.
    gui_rx: std::sync::Mutex<Option<mpsc::UnboundedReceiver<ChannelMessage>>>,
}

impl GuiChannel {
    /// Create a new GuiChannel with an internal mpsc pair.
    ///
    /// Returns `(Self, gui_tx)`. The caller must:
    /// 1. Store `gui_tx` in `GUI_MESSAGE_TX` globally
    /// 2. Register this channel in the channel registry
    /// 3. Call `listen(tx)` to start consuming from the internal receiver
    #[must_use]
    pub fn new() -> (Self, mpsc::UnboundedSender<ChannelMessage>) {
        let (gui_tx, gui_rx) = mpsc::unbounded_channel::<ChannelMessage>();
        let channel = Self {
            gui_rx: std::sync::Mutex::new(Some(gui_rx)),
        };
        (channel, gui_tx)
    }
}

#[async_trait]
impl Channel for GuiChannel {
    /// Pure transport — broadcast and persistence are handled centrally
    /// by [`broadcast_and_persist_agent_response`].
    async fn send(&self, _message: &SendMessage) -> anyhow::Result<()> {
        Ok(())
    }

    async fn listen(&self, tx: mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        // Take the receiver from the internal mutex. After this, we own it
        // and can drop the &self reference.
        let mut gui_rx = self
            .gui_rx
            .lock()
            .unwrap_poison()
            .take()
            .expect("GuiChannel::listen() called twice");
        // Mutex guard is dropped here.

        // Forward each GUI-originated message into the shared pipeline.
        // Broadcast+persist is handled centrally in process_channel_message().
        while let Some(msg) = gui_rx.recv().await {
            if tx.send(msg).await.is_err() {
                tracing::info!("GuiChannel: pipeline closed — shutting down listener");
                break;
            }
        }

        tracing::info!("GuiChannel: listener stopped");
        Ok(())
    }

    fn name(&self) -> &'static str {
        "gui"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// GUI users are addressed by sender name.
    fn resolve_recipient(&self, user_name: &str, _reply_target: &str) -> Option<String> {
        Some(user_name.to_string())
    }
}

// ── VoiceChannel ──────────────────────────────────────────

/// The voice channel — registered so the message routing system can
/// resolve the `"voice"` channel name when delivering agent responses.
///
/// There is no outbound voice transport — agent responses are broadcast
/// to the GUI and persisted to chat_history by the standard response
/// delivery path. The `send()` method is a no-op, matching the pattern
/// used by [`GuiChannel`].
///
/// The voice pipeline runs its own mic-capture loop independently;
/// `listen()` is a no-op because incoming voice commands flow through
/// `crate::audio::voice::route_to_agent`, not through a channel listener.
pub struct VoiceChannel;

#[async_trait]
impl Channel for VoiceChannel {
    /// No-op — voice has no outbound transport. Agent responses are
    /// broadcast to the GUI and persisted to chat_history by
    /// [`broadcast_and_persist_agent_response`].
    async fn send(&self, _message: &SendMessage) -> anyhow::Result<()> {
        Ok(())
    }

    /// No-op — the voice pipeline manages its own mic-capture loop.
    async fn listen(&self, _tx: mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "voice"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Register the voice channel in the global channel registry.
///
/// Called during bootstrap from `init_message_pipeline`. The channel
/// registry must already be initialised — callers should ensure
/// [`crate::CHANNEL_REGISTRY`] has been set before invoking this.
///
/// The `VoiceChannel` has a no-op `send()` (agent responses are
/// delivered via broadcast+persist independently of the registry) and
/// a no-op `listen()` (the voice pipeline runs its own mic-capture
/// loop). Registration resolves the `"voice"` channel name so the
/// message routing system can look it up when constructing replies.
pub fn register_global() {
    let channel: Arc<dyn Channel> = Arc::new(VoiceChannel);
    crate::channel_registry().register(channel);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CHANNEL_REGISTRY, ChannelRegistry, channel_registry};

    /// Verify that [`register_global`] correctly registers the voice
    /// channel — i.e. the same function called by the production
    /// bootstrap path in `init_message_pipeline`.
    #[test]
    fn test_voice_channel_registration() {
        // Initialise the registry (idempotent — reuses if already set).
        CHANNEL_REGISTRY.get_or_init(ChannelRegistry::default);

        // Call the same registration function used by production.
        register_global();

        let found = channel_registry().get("voice");
        assert!(
            found.is_some(),
            "VoiceChannel should be findable by 'voice' name"
        );
    }
}
