mod enrichment;
pub mod gui;
pub mod telegram;
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
    channel_name: String,
    content: String,
    direction: ChatDirection,
    agent_role: Option<String>,
    workspace: String,
    optimistic_id: Option<String>,
}

impl BroadcastPersistEntry {
    /// Broadcast this entry to [`crate::CHAT_BROADCAST`] and persist it to
    /// `chat_history`.
    ///
    /// Fields shared between the broadcast and DB insert are cloned for the
    /// broadcast (which consumes via `tx.send`), then the originals are moved
    /// into the DB insert.  Fields only used in the broadcast
    /// (`optimistic_id`) are moved directly without an unnecessary clone.
    async fn broadcast_and_persist(self) {
        use crate::ChatEvent;

        // Invariant: direction=Agent must carry a non-None agent_role
        debug_assert!(
            self.direction != ChatDirection::Agent || self.agent_role.is_some(),
            "BroadcastPersistEntry: direction=Agent but agent_role is None"
        );

        let message_id = crate::generate_id();
        let timestamp = turso::now();

        // Compute db_role/db_direction while self.agent_role is still
        // available (borrowed temporarily via as_deref before being moved
        // into the DB insert below).
        let (db_role, db_direction) = match self.direction {
            ChatDirection::Agent => (
                self.agent_role.as_deref().unwrap_or("").to_string(),
                "agent".to_string(),
            ),
            ChatDirection::User => ("user".to_string(), "user".to_string()),
        };

        // ── Broadcast ──────────────────────────────────────────────
        // Clone fields shared with the DB insert; move optimistic_id
        // (only used in the event, not persisted).
        if let Some(tx) = crate::CHAT_BROADCAST.get() {
            let _ = tx.send(ChatEvent::Message {
                message_id: message_id.clone(),
                user_name: self.user_name.clone(),
                content: self.content.clone(),
                direction: self.direction,
                timestamp: timestamp.clone(),
                agent_role: self.agent_role.clone(),
                workspace: self.workspace.clone(),
                optimistic_id: self.optimistic_id,
            });
        }

        // ── Persist to chat_history ────────────────────────────────
        let store = crate::chat_history::store();
        let _ = store
            .insert(&ChatHistoryInsert {
                message_id,
                user_name: self.user_name,
                channel: self.channel_name,
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
/// ([`send_channel_reply`]) and the Manager queue consumer
/// in [`crate::manager_queue`].
///
/// Takes explicit `user_name` (canonical user name), `channel` (e.g. "telegram", "gui"),
/// and primitive fields — does **not** depend on [`SendMessage`], so it can be used
/// from the Manager queue which works from [`crate::users::UserRecord`].
pub async fn broadcast_and_persist_agent_response(
    user_name: &str,
    channel: &str,
    content: &str,
    agent_role: Option<String>,
    workspace: &str,
) {
    BroadcastPersistEntry {
        user_name: user_name.to_string(),
        channel_name: channel.to_string(),
        content: content.to_string(),
        direction: ChatDirection::Agent,
        agent_role,
        workspace: workspace.to_string(),
        optimistic_id: None, // agent messages must not carry one
    }
    .broadcast_and_persist()
    .await;
}

/// Write an incoming user message to CHAT_BROADCAST for immediate GUI display
/// and persist it to chat_history. Uses `msg.source_channel` for the channel
/// field so it works for both Telegram and GUI-originated messages.
pub async fn write_incoming_to_broadcast(msg: &ChannelMessage) {
    BroadcastPersistEntry {
        user_name: msg.user_name.clone(),
        channel_name: msg.source_channel.clone(),
        content: msg.content.clone(),
        direction: ChatDirection::User,
        agent_role: None, // user messages have no agent role
        workspace: msg.workspace.clone(),
        optimistic_id: msg.optimistic_id.clone(), // GUI uses this for replacement
    }
    .broadcast_and_persist()
    .await;
}

/// Send a reply through a channel.
pub async fn send_channel_reply(content: String, msg: &ChannelMessage, agent_role: Option<String>) {
    // ── Broadcast agent response for live GUI display and chat_history ──
    // Must happen before the channel registry check -- broadcast_and_persist
    // does not depend on the channel object, only on fields from `msg`.
    broadcast_and_persist_agent_response(
        &msg.user_name,
        &msg.source_channel,
        &content,
        agent_role,
        &msg.workspace,
    )
    .await;

    let Some(channel) = crate::channel_registry().get(&msg.source_channel) else {
        tracing::warn!(
            source_channel = %msg.source_channel,
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
    source_channel: String,
    cancellation_token: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let refresh_interval = std::time::Duration::from_secs(CHANNEL_TYPING_REFRESH_INTERVAL_SECS);
    tokio::spawn(async move {
        let Some(channel) = crate::channel_registry().get(&source_channel) else {
            tracing::warn!(
                source_channel = %source_channel,
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
                    if let Err(e) = channel.start_typing(&recipient).await {
                        tracing::debug!("Failed to start typing on {}: {e}", channel.name());
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
