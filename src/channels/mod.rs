//! The shared middle of every conversation: the helpers that move a message
//! between a transport, the persistence layer and the GUI, plus the pieces
//! every transport shares.
//!
//! Owns broadcast-and-persist fan-out, link enrichment, reply references, GUI
//! chat history and drafts, and the Telegram transport and its intake ledger
//! under `telegram*`. Not the message routing into an agent (that is
//! `crate::agent::message_router`) and not the Telegram command handlers,
//! which stay in the binary.

pub(crate) mod chat_draft;
pub(crate) mod chat_history;
mod enrichment;
pub(crate) mod reply;
pub mod telegram;
mod telegram_group;
pub(crate) mod telegram_intake;
pub use chat_history::run_cap_loop;
#[cfg(target_os = "macos")]
pub(crate) use enrichment::marked_transcript;
pub use enrichment::{enrich_links, enrich_message};
pub use reply::{ReplyReference, apply_reply_marker};
pub use telegram::mirror_gui_message_to_telegram;
pub use telegram_group::compose_group_content;

use crate::channels::chat_history::ChatHistoryInsert;
use crate::db;
use crate::util::UnwrapPoison;
use crate::{Channel, ChannelMessage, ChatDirection, SendMessage};
use async_trait::async_trait;
use enrichment::has_inbound_temp_marker;
// The voice channel (macOS-only) is the only `Arc` user in this module.
#[cfg(target_os = "macos")]
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const CHANNEL_TYPING_REFRESH_INTERVAL_SECS: u64 = 4;

/// Broadcast an agent response to CHAT_BROADCAST for live GUI display and
/// persist it to chat_history. This is the canonical entry point for all
/// agent responses: the delivery paths in [`crate::agent::message_router`]
/// (registered users and raw reply targets) and the macOS voice notice in
/// `crate::audio::voice`.
///
/// TTS audio playback is handled separately by `crate::audio::tts::init_listener()`
/// (macOS-only), which subscribes to [`CHAT_BROADCAST`](crate::CHAT_BROADCAST)
/// and triggers speech for the admin's own assistant messages.  This function
/// does not itself invoke any TTS logic.
///
/// Takes primitive fields rather than [`crate::SendMessage`], so the per-agent
/// consumer loop can call it with a [`crate::users::UserRecord`].
pub(crate) async fn broadcast_and_persist_agent_response(
    user_name: &str,
    channel: &str,
    content: &str,
    agent_role: String,
    workspace: &str,
    broadcast_id: Option<String>,
) {
    let message_id = crate::generate_id();
    let timestamp = db::now();

    broadcast_chat_event(
        &message_id,
        user_name,
        content,
        ChatDirection::Agent,
        channel,
        Some(agent_role.clone()),
        workspace,
        None, // agent messages must not carry one
        None, // agent responses never carry a reply reference
        &timestamp,
        false,
    );

    let store = crate::channels::chat_history::store();
    let _ = store
        .insert(&ChatHistoryInsert {
            message_id,
            user_name: user_name.to_string(),
            direction: "agent".to_string(),
            content: content.to_string(),
            agent_role: Some(agent_role),
            broadcast_id,
            workspace: workspace.to_string(),
            timestamp: Some(timestamp),
            reply_reference: None,
        })
        .await;
}

/// Send a [`ChatEvent::Message`] to the broadcast channel.
///
/// This is the single shared entry point for every message broadcast (typing
/// indicators go straight to the channel), ensuring consistent message
/// construction across user messages, agent responses, and any future message
/// types.  The caller is responsible for generating a stable [`message_id`]
/// and [`timestamp`] if they need to correlate the broadcast event with a
/// persist operation.
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

/// Chat-history content for an inbound message: normally the pre-enrichment
/// `original` text (which avoids storing the large data URIs image enrichment
/// produces), but the data-URI-stripped `enriched` content when the original
/// carried a marker for an inbound attachment — those raw markers name temp
/// files the daemon deletes right after enrichment and must never be persisted.
///
/// A hand-typed audio marker names no inbound attachment, so it is kept as
/// written rather than replaced by the annotation it enriches to.
#[must_use]
pub fn persist_content(original: &str, enriched: &str) -> String {
    if has_inbound_temp_marker(original) {
        crate::util::strip_data_uris(enriched)
    } else {
        original.to_string()
    }
}

/// Broadcast an incoming user message to the GUI and persist it to chat_history,
/// mirroring it to Telegram in parallel. `broadcast_content` is the enriched text
/// sent to the GUI (e.g. audio transcriptions, renderable data URIs);
/// `content_for_history` is what is stored and mirrored — callers derive it with
/// [`persist_content`].
/// The GUI bubble is broadcast synchronously before the async persist + mirror
/// join begins.
///
/// Forwards [`ChannelMessage::optimistic_id`] so the GUI can replace its optimistic
/// bubble with the real one.
pub async fn broadcast_and_persist_incoming_message(
    msg: &ChannelMessage,
    broadcast_content: &str,
    content_for_history: &str,
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
                    content: content_for_history.to_string(),
                    agent_role: None,
                    broadcast_id: None,
                    workspace: msg.workspace.clone(),
                    timestamp: Some(timestamp),
                    reply_reference: msg.reply_reference.clone(),
                })
                .await;
        },
        async {
            let mut mirror_msg = msg.clone();
            mirror_msg.content = content_for_history.to_string();
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

/// Await a typing task's shutdown, logging a crash. Cancellation belongs to the
/// caller: cancel the token the task was spawned with, then await this.
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

// ── VoiceChannel (macOS-only) ─────────────────────────────
// The voice channel is the transport for the macOS-only voice pipeline, which
// is its only producer, so nothing resolves the `"voice"` channel name off
// macOS — the whole section is gated with the audio subsystem it serves.

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
#[cfg(target_os = "macos")]
struct VoiceChannel;

#[cfg(target_os = "macos")]
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
#[cfg(target_os = "macos")]
pub fn register_global() {
    let channel: Arc<dyn Channel> = Arc::new(VoiceChannel);
    crate::channel_registry().register(channel);
}

#[cfg(target_os = "macos")]
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
