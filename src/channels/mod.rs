//! The shared middle of every conversation: the helpers that move a message between a
//! transport, the persistence layer and the GUI, plus the pieces every transport shares.
//!
//! Owns broadcast-and-persist fan-out, link enrichment, reply references, GUI chat history and
//! drafts, and the Telegram transport and its intake ledger under `telegram*`. Not the message
//! routing into an agent (`crate::agent::message_router`) and not the Telegram command
//! handlers, which stay in the binary.

pub(crate) mod chat_draft;
pub(crate) mod chat_history;
mod enrichment;
pub(crate) mod reply;
pub mod telegram;
mod telegram_group;
pub(crate) mod telegram_intake;
pub(crate) mod telegram_local;
pub use chat_history::run_cap_loop;
pub(crate) use enrichment::marked_transcript;
pub use enrichment::{enrich_links, enrich_message};
pub use reply::{ReplyReference, apply_reply_marker};
pub use telegram::mirror_gui_message_to_telegram;
pub use telegram_group::compose_group_content;

use crate::channels::chat_history::ChatHistoryInsert;
use crate::db;
use crate::util::UnwrapPoison;
use crate::{Channel, ChannelMessage, ChatDirection, SendFailure, SendMessage, SendOutcome};
use async_trait::async_trait;
use enrichment::has_inbound_temp_marker;
use std::borrow::Cow;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const CHANNEL_TYPING_REFRESH_INTERVAL_SECS: u64 = 4;

/// Take the urgent mark off the text a user typed: a leading `!`, after any spaces, tabs and
/// blank lines, marks the message urgent and is consumed with that whitespace; every other
/// message is returned exactly as written.
///
/// Read on the typed text and nothing else — a caption rather than the attachment markers the
/// surface composes around it, never a transcript or a forward prefix — so no product-added text
/// can make a message urgent or an urgent one ordinary. There is no escape: an ordinary message
/// opening with the mark is urgent, and loses its `!`.
#[must_use]
pub(crate) fn take_urgent(typed: &str) -> (Cow<'_, str>, bool) {
    let rest = typed.trim_start();
    match rest.strip_prefix('!') {
        Some(marked) => (Cow::Owned(marked.trim_start().to_string()), true),
        None => (Cow::Borrowed(typed), false),
    }
}

/// Keep the self-hosted Telegram Bot API server alive, updated and clean, and finish any switch
/// a crash interrupted. A no-op in cloud mode beyond one cheap check a minute.
pub async fn run_telegram_server_supervisor() {
    telegram_local::run_supervisor().await;
}

/// Called from the exits that skip destructors as well as the ordinary shutdown.
pub async fn stop_telegram_server() {
    telegram_local::stop_for_exit().await;
}

/// Broadcast an agent response to CHAT_BROADCAST for live GUI display and persist it to
/// chat_history. The canonical entry point for all agent responses: the delivery paths in
/// [`crate::agent::message_router`] and the voice pipeline's notice in `crate::audio::voice`.
///
/// No speech happens here or on any event this broadcasts: the only way the product speaks is
/// the admin Assistant's `say` tool, which calls `crate::speech::synth` itself.
///
/// Primitive fields rather than [`crate::SendMessage`], so the per-agent consumer loop can call
/// it with a [`crate::users::UserRecord`].
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

/// Send a [`ChatEvent::Message`] to the broadcast channel — the single shared entry point for
/// every message broadcast (typing indicators go straight to the channel), so construction stays
/// consistent. The caller generates a stable [`message_id`] and [`timestamp`] if it must
/// correlate the broadcast with a persist.
///
/// `transient` marks an event for live GUI display only: it is never persisted to chat_history
/// (e.g. the Phase-1 scripted onboarding exchange). Pass `false` for every persisted path.
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

/// Normally the pre-enrichment `original` text (avoiding the large data URIs image enrichment
/// produces), but the data-URI-stripped `enriched` content when the original carried a marker for
/// an inbound attachment — those markers name temp files the daemon deletes right after enrichment.
///
/// A hand-typed audio marker names no inbound attachment, so it is kept as written.
#[must_use]
pub fn persist_content(original: &str, enriched: &str) -> String {
    if has_inbound_temp_marker(original) {
        crate::util::strip_data_uris(enriched)
    } else {
        original.to_string()
    }
}

/// Broadcast an incoming user message to the GUI and persist it to chat_history, mirroring it to
/// Telegram in parallel. `broadcast_content` is the enriched text sent to the GUI;
/// `content_for_history` is what is stored and mirrored — callers derive it with
/// [`persist_content`]. The GUI bubble is broadcast synchronously before the async persist +
/// mirror join begins. Forwards [`ChannelMessage::optimistic_id`] so the GUI can replace its
/// optimistic bubble with the real one.
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

/// Log a typing task's crash. Cancellation belongs to the caller: cancel the token the task was
/// spawned with, then await this.
pub async fn stop_typing(handle: tokio::task::JoinHandle<()>) {
    if let Err(error) = handle.await {
        tracing::error!("Typing task crashed: {error}");
    }
}

/// The GUI channel — always registered, even in dashboard-only mode.
///
/// Unlike Telegram's async listener loop, this uses an internal mpsc pair: the Iced UI pushes
/// `ChannelMessage` values into `GUI_MESSAGE_TX`, and `listen()` reads them from the paired
/// receiver, forwarding each into the shared pipeline `tx`. `send()` is pure transport (a no-op):
/// outgoing responses are broadcast and persisted centrally by
/// [`broadcast_and_persist_agent_response`].
pub struct GuiChannel {
    /// The internal receiver, stored so `listen()` can consume it.
    gui_rx: std::sync::Mutex<Option<mpsc::UnboundedReceiver<ChannelMessage>>>,
}

impl GuiChannel {
    /// Returns `(Self, gui_tx)`. The caller must store `gui_tx` in `GUI_MESSAGE_TX`, register this
    /// channel in the registry, and call `listen(tx)`.
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
    /// Pure transport — broadcast and persistence are handled centrally by
    /// [`broadcast_and_persist_agent_response`].
    async fn send(&self, _message: &SendMessage) -> Result<SendOutcome, SendFailure> {
        Ok(SendOutcome::default())
    }

    async fn listen(&self, tx: mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        // Take the receiver from the internal mutex: after this we own it and can drop the &self
        // reference.
        let mut gui_rx = self
            .gui_rx
            .lock()
            .unwrap_poison()
            .take()
            .expect("GuiChannel::listen() called twice");
        // Mutex guard is dropped here.

        // Forward each GUI-originated message into the shared pipeline.
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

// VoiceChannel: the transport for the voice pipeline, its only producer — the pipeline runs on
// every platform, so the `"voice"` channel name always resolves.

/// Registered so routing can resolve the `"voice"` channel name when delivering agent responses.
///
/// There is no outbound voice transport: responses are broadcast to the GUI and persisted by the
/// standard delivery path, so `send()` is a no-op like [`GuiChannel`]. The voice pipeline runs
/// its own mic-capture loop, and a recording's transcript never arrives through a channel
/// listener: it joins the Home composer's message, delivered on the GUI channel.
struct VoiceChannel;

#[async_trait]
impl Channel for VoiceChannel {
    /// No-op — voice has no outbound transport.
    async fn send(&self, _message: &SendMessage) -> Result<SendOutcome, SendFailure> {
        Ok(SendOutcome::default())
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

/// Called during bootstrap from `init_message_pipeline`; [`crate::CHANNEL_REGISTRY`] must
/// already be set.
pub fn register_global() {
    let channel: Arc<dyn Channel> = Arc::new(VoiceChannel);
    crate::channel_registry().register(channel);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CHANNEL_REGISTRY, ChannelRegistry, channel_registry};

    /// The bootstrap registration must leave the voice channel findable by its name.
    #[test]
    fn test_voice_channel_registration() {
        CHANNEL_REGISTRY.get_or_init(ChannelRegistry::default);
        register_global();
        assert!(
            channel_registry().get("voice").is_some(),
            "VoiceChannel should be findable by 'voice' name"
        );
    }
}

#[cfg(test)]
mod urgent_tests {
    use super::take_urgent;

    /// Only a `!` opening the typed text — after spaces, tabs and blank lines — is the mark.
    #[test]
    fn take_urgent_reads_only_the_leading_mark() {
        let bare_spaces = "   \t\n  !foo";
        let multi_line = "\n\t  !foo";
        for (typed, expected, urgent) in [
            ("!foo", "foo", true),
            (bare_spaces, "foo", true),
            ("foo", "foo", false),
            ("foo!bar", "foo!bar", false),
            ("!foo !bar", "foo !bar", true),
            ("!", "", true),
            (multi_line, "foo", true),
        ] {
            let (text, got) = take_urgent(typed);
            let text: &str = text.as_ref();
            assert_eq!(text, expected, "typed {typed:?}");
            assert_eq!(got, urgent, "typed {typed:?}");
        }
    }
}
