mod enrichment;
pub mod gui;
pub mod telegram;
pub mod voice;
pub use enrichment::{EnrichmentStrategy, enrich_links, enrich_message};
pub use telegram::mirror_gui_message_to_telegram;

use crate::chat_history::ChatHistoryInsert;
use crate::turso;
use crate::{ChannelMessage, ChatDirection, SendMessage};
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
        let timestamp = turso::now();

        let (db_role, db_direction) = match self.direction {
            ChatDirection::Agent => (
                self.agent_role.as_deref().unwrap_or("").to_string(),
                "agent".to_string(),
            ),
            ChatDirection::User => ("user".to_string(), "user".to_string()),
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
            &timestamp,
        );

        let store = crate::chat_history::store();
        let _ = store
            .insert(&ChatHistoryInsert {
                message_id,
                user_name: self.user_name,
                channel: self.channel,
                role: db_role,
                direction: db_direction,
                content: self.content,
                agent_role: self.agent_role,
                workspace: self.workspace,
                created_at: timestamp,
            })
            .await;
    }
}

/// Broadcast an agent response to CHAT_BROADCAST for live GUI display and
/// persist it to chat_history. This is the canonical entry point for all
/// agent responses — both the non-Manager path
/// ([`send_channel_reply`]) and the per-agent consumer loop
/// in [`crate::message_router`].
///
/// TTS audio playback is handled separately by [`crate::tts::init_listener()`],
/// which subscribes to [`CHAT_BROADCAST`](crate::CHAT_BROADCAST) and triggers
/// speech for matching agent messages.  This function does not itself invoke
/// any TTS logic.
///
/// Takes explicit `user_name` (canonical user name), `channel` (e.g. "telegram", "gui"),
/// and primitive fields — does **not** depend on [`SendMessage`], so it can be used
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
        optimistic_id: None, // agent messages must not carry one
    }
    .broadcast_and_persist()
    .await;
}

/// Broadcast a user message to the GUI and persist it to chat_history.
///
/// Symmetric to [`broadcast_and_persist_agent_response`] — provides the
/// same convenience for user-originated messages without requiring a
/// [`ChannelMessage`] struct.  The `channel` field records the message
/// source (e.g. `"voice"`, `"gui"`, `"telegram"`).
pub(crate) async fn broadcast_and_persist_user_message(
    user_name: &str,
    channel: &str,
    content: &str,
    workspace: &str,
) {
    BroadcastPersistEntry {
        user_name: user_name.to_string(),
        channel: channel.to_string(),
        content: content.to_string(),
        direction: ChatDirection::User,
        agent_role: None,
        workspace: workspace.to_string(),
        optimistic_id: None,
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn broadcast_chat_event(
    message_id: &str,
    user_name: &str,
    content: &str,
    direction: ChatDirection,
    channel: &str,
    agent_role: Option<String>,
    workspace: &str,
    optimistic_id: Option<String>,
    timestamp: &str,
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
        });
    }
}

/// Broadcast an incoming user message to CHAT_BROADCAST for immediate GUI display,
/// without persisting to chat_history. Use [`persist_incoming_message`] to persist
/// separately — this allows broadcasting enriched content while persisting the original
/// (e.g. to avoid storing large data URIs in chat_history).
///
/// The `message_id` and `timestamp` should be the same values used in the corresponding
/// [`persist_incoming_message`] call so the broadcast event and chat_history record are
/// correlated.
pub fn broadcast_incoming_message(
    msg: &ChannelMessage,
    content: &str,
    message_id: &str,
    timestamp: &str,
) {
    broadcast_chat_event(
        message_id,
        &msg.user_name,
        content,
        ChatDirection::User,
        &msg.channel,
        None,
        &msg.workspace,
        msg.optimistic_id.clone(),
        timestamp,
    );
}

/// Persist an incoming user message to chat_history, without broadcasting to GUI.
/// Use [`broadcast_incoming_message`] to broadcast separately — this allows persisting
/// the original content while broadcasting enriched content (e.g. to avoid storing
/// large data URIs in chat_history).
///
/// The `message_id` and `timestamp` should be the same values used in the corresponding
/// [`broadcast_incoming_message`] call so the chat_history record and broadcast event are
/// correlated.
pub async fn persist_incoming_message(
    msg: &ChannelMessage,
    content: &str,
    message_id: &str,
    timestamp: &str,
) {
    let store = crate::chat_history::store();
    let _ = store
        .insert(&ChatHistoryInsert {
            message_id: message_id.to_string(),
            user_name: msg.user_name.clone(),
            channel: msg.channel.clone(),
            role: "user".to_string(),
            direction: "user".to_string(),
            content: content.to_string(),
            agent_role: None,
            workspace: msg.workspace.clone(),
            created_at: timestamp.to_string(),
        })
        .await;
}

/// Send a reply through a channel.
pub async fn send_channel_reply(content: String, msg: &ChannelMessage, agent_role: Option<String>) {
    // ── Broadcast agent response for live GUI display and chat_history ──
    // Must happen before the channel registry check -- broadcast_and_persist
    // does not depend on the channel object, only on fields from `msg`.
    broadcast_and_persist_agent_response(
        &msg.user_name,
        &msg.channel,
        &content,
        agent_role,
        &msg.workspace,
    )
    .await;

    let Some(channel) = crate::channel_registry().get(&msg.channel) else {
        tracing::warn!(
            channel = %msg.channel,
            "Channel not found in registry -- reply not delivered via transport (already broadcast & persisted)"
        );
        return;
    };

    let reply = SendMessage {
        content,
        recipient: msg.reply_target.clone(),
        reply_markup: None,
    };

    if let Err(e) = channel.send(&reply).await {
        tracing::error!("Failed to reply on {}: {e}", channel.name());
    }
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
