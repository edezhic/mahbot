mod enrichment;
pub mod gui;
pub mod telegram;
pub use enrichment::{EnrichmentStrategy, enrich_links, enrich_message};

use crate::chat_history::ChatHistoryInsert;
use crate::turso;
use crate::util::MEDIA_MARKER_RE;
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
    /// Ownership flow: `self` is destructured at the top so that each field's
    /// value is cloned only for the broadcast (which consumes via `tx.send`),
    /// then the original owned value is moved into the DB insert.  This
    /// eliminates the redundant clones that would otherwise be needed when the
    /// same field appears in both the [`ChatEvent::Message`] and
    /// [`ChatHistoryInsert`] struct literals.
    async fn broadcast_and_persist(self) {
        use crate::ChatEvent;

        // Destructure self to obtain owned fields.  The broadcast (tx.send)
        // receives clones; the DB insert receives the originals.
        let BroadcastPersistEntry {
            user_name,
            channel_name,
            content,
            direction,
            agent_role,
            workspace,
            optimistic_id,
        } = self;

        // Invariant: direction=Agent must carry a non-None agent_role
        debug_assert!(
            direction != ChatDirection::Agent || agent_role.is_some(),
            "BroadcastPersistEntry: direction=Agent but agent_role is None"
        );

        let message_id = crate::generate_id();
        let timestamp = turso::now();

        // Compute db_role/db_direction as owned strings before agent_role is
        // moved into the DB insert below.  (We borrow agent_role here via
        // as_deref, so the owned strings must be computed first.)
        let (db_role, db_direction) = match direction {
            ChatDirection::Agent => (
                agent_role.as_deref().unwrap_or("").to_string(),
                "agent".to_string(),
            ),
            ChatDirection::User => ("user".to_string(), "user".to_string()),
        };

        // ── Broadcast ──────────────────────────────────────────────
        // Broadcast precedes the DB insert: values cloned here are later
        // moved (originals) into ChatHistoryInsert below.
        if let Some(tx) = crate::CHAT_BROADCAST.get() {
            let _ = tx.send(ChatEvent::Message {
                message_id: message_id.clone(),
                user_name: user_name.clone(),
                content: content.clone(),
                direction,
                timestamp: timestamp.clone(),
                agent_role: agent_role.clone(),
                workspace: workspace.clone(),
                optimistic_id,
            });
        }

        // ── Persist to chat_history ────────────────────────────────
        // All fields receive the original owned values (moved, not cloned).
        let store = crate::chat_history::store();
        let _ = store
            .insert(&ChatHistoryInsert {
                message_id,
                user_name,
                channel: channel_name,
                role: db_role,
                direction: db_direction,
                content,
                agent_role,
                workspace,
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
    let stop_signal = cancellation_token;
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
                () = stop_signal.cancelled() => break,
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

/// Mirror a GUI user's message to their Telegram chats as a blockquote, so conversation history is readable from both surfaces.
///
/// This should be called before enrichment to preserve the original
/// user-typed text (pre-link-summary, pre-transcription).
///
/// # Guards
///
/// * Only mirrors messages where `source_channel == "gui"` (prevents echo loops).
/// * Skips empty or whitespace-only messages.
/// * Silently returns when no Telegram channel is registered or the user has no
///   Telegram binding with a `reply_target` (no error, no crash).
/// * Sends to **all** Telegram bindings if the user has multiple.
///
/// # Quote format
///
/// Uses `<blockquote>` HTML tags, which `markdown_to_telegram_html` in the
/// Telegram channel's `send()` pipeline passes through unchanged. The user's
/// text retains markdown formatting through the standard inline parser.
/// Media markers (`[IMAGE:...]`, `[AUDIO:...]`, `[VIDEO:...]`) are stripped
/// so raw marker syntax does not appear in the quote; purely media-only
/// messages are skipped entirely.
pub async fn mirror_gui_message_to_telegram(msg: &ChannelMessage) {
    // Guard: only mirror GUI-originated user messages (prevents echo loops).
    if msg.source_channel != "gui" {
        return;
    }

    // Guard: skip empty or whitespace-only messages.
    let trimmed = msg.content.trim();
    if trimmed.is_empty() {
        return;
    }

    // Guard: Telegram channel must be available.
    let Some(channel) = crate::channel_registry().get("telegram") else {
        return;
    };

    // Look up the user's channel bindings.
    let bindings = match crate::users::store()
        .get_user_channels(&msg.user_name)
        .await
    {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                user = %msg.user_name,
                error = %e,
                "Failed to look up user channels for GUI message mirror"
            );
            return;
        }
    };

    // Filter to Telegram bindings (reply_target checked per binding below).
    let telegram_bindings: Vec<_> = bindings
        .into_iter()
        .filter(|b| b.channel == "telegram")
        .collect();

    if telegram_bindings.is_empty() {
        return; // No Telegram binding — silently skip.
    }

    // Strip media markers so users don't see raw `[IMAGE:...]` syntax in the quote.
    let content = MEDIA_MARKER_RE.replace_all(trimmed, "").to_string();
    let content = content.trim().to_string();
    if content.is_empty() {
        return; // Media-only message — nothing to quote.
    }

    // Wrap in <blockquote> — these tags pass through markdown_to_telegram_html
    // unchanged, while the user's text retains markdown formatting.
    let quoted = format!("<blockquote>\n{content}\n</blockquote>");

    for binding in &telegram_bindings {
        let Some(reply_target) = &binding.reply_target else {
            continue; // skip bindings without a reply target
        };
        let reply = SendMessage {
            content: quoted.clone(),
            recipient: reply_target.clone(),
            reply_markup: None,
        };

        if let Err(e) = channel.send(&reply).await {
            tracing::error!(
                user = %msg.user_name,
                recipient = %reply_target,
                error = %e,
                "Failed to mirror GUI message to Telegram"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── GUI message → Telegram mirror tests ──────────────────────

    use async_trait::async_trait;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::OnceLock;

    /// Serialization lock for all mirror tests — these tests share the global
    /// [`CHANNEL_REGISTRY`] and store singletons, so they must run one at a time.
    /// Uses `tokio::sync::Mutex` to avoid blocking worker threads while held
    /// across await points.
    static MIRROR_TEST_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

    async fn acquire_mirror_lock() -> tokio::sync::MutexGuard<'static, ()> {
        MIRROR_TEST_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await
    }

    use crate::util::UnwrapPoison;

    /// A spy channel that records sent messages in a shared Vec.
    struct SpyChannel {
        sent: Arc<Mutex<Vec<SendMessage>>>,
    }

    #[async_trait]
    impl crate::Channel for SpyChannel {
        async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
            self.sent.lock().unwrap_poison().push(message.clone());
            Ok(())
        }

        async fn listen(
            &self,
            _tx: tokio::sync::mpsc::Sender<ChannelMessage>,
        ) -> anyhow::Result<()> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "telegram"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Set up the channel registry with a spy Telegram channel and return a
    /// shared sent-messages buffer. Idempotent — safe to call from every test.
    fn setup_spy_channel() -> &'static Arc<Mutex<Vec<SendMessage>>> {
        static SPY_SENT: OnceLock<Arc<Mutex<Vec<SendMessage>>>> = OnceLock::new();
        SPY_SENT.get_or_init(|| {
            let sent = Arc::new(Mutex::new(Vec::new()));
            let registry = crate::CHANNEL_REGISTRY.get_or_init(crate::ChannelRegistry::default);
            registry.register(Arc::new(SpyChannel {
                sent: Arc::clone(&sent),
            }) as Arc<dyn crate::Channel>);
            sent
        })
    }

    /// Ensure the user store has a test user with a Telegram binding and
    /// reply_target. Idempotent.
    async fn setup_user_with_telegram_binding(user_name: &str, reply_target: &str) {
        use crate::users::store;
        let store = store();
        store
            .add_user(user_name, Some("full"))
            .await
            .expect("add_user");
        store
            .bind_channel(user_name, "telegram", user_name)
            .await
            .expect("bind_channel");
        store
            .update_channel_contact("telegram", user_name, reply_target)
            .await
            .expect("update_channel_contact");
    }

    /// Three-line preamble shared by all mirror tests: acquire the serialization
    /// lock, initialise test stores, and set up the spy channel. Returns the spy
    /// sent-messages buffer and the lock guard (kept alive for the test duration).
    async fn setup_mirror_test_env() -> (
        &'static Arc<Mutex<Vec<SendMessage>>>,
        tokio::sync::MutexGuard<'static, ()>,
    ) {
        let lock = acquire_mirror_lock().await;
        crate::util::test::init_test_stores().await;
        let sent = setup_spy_channel();
        (sent, lock)
    }

    fn gui_msg(user_name: &str, content: &str) -> ChannelMessage {
        ChannelMessage {
            user_name: user_name.to_string(),
            reply_target: String::new(),
            content: content.to_string(),
            source_channel: "gui".to_string(),
            workspace: "test".to_string(),
            optimistic_id: None,
            callback_query_id: None,
        }
    }

    fn telegram_msg(user_name: &str, content: &str) -> ChannelMessage {
        ChannelMessage {
            user_name: user_name.to_string(),
            reply_target: "chat:thread".to_string(),
            content: content.to_string(),
            source_channel: "telegram".to_string(),
            workspace: "test".to_string(),
            optimistic_id: None,
            callback_query_id: None,
        }
    }

    // ── Guard tests: early-return conditions ─────────────────────
    //
    // These tests verify that `mirror_gui_message_to_telegram` returns
    // early (without sending) for each guard condition. They are serialized
    // via [`MIRROR_TEST_LOCK`] because the channel registry and store
    // singletons are global. Each uses a unique reply target so assertions
    // filter only the current test's messages from the shared spy buffer.

    #[tokio::test]
    async fn skip_non_gui_source() {
        let (sent, _lock) = setup_mirror_test_env().await;
        setup_user_with_telegram_binding("skip_telegram", "target_non_gui").await;

        let msg = telegram_msg("skip_telegram", "hello from telegram");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "target_non_gui")
            .collect();
        assert!(our_msgs.is_empty(), "non-GUI source should not send");
    }

    #[tokio::test]
    async fn skip_empty_or_whitespace_content() {
        // Both inputs exercise the same guard — `msg.content.trim().is_empty()`.
        // Each iteration acquires the serialization lock independently; this
        // is safe because the global stores (OnceCell) and the spy channel
        // (OnceLock) are identical across calls to `setup_mirror_test_env()`.
        for content in ["", "   \t\n  "] {
            let (sent, _lock) = setup_mirror_test_env().await;
            setup_user_with_telegram_binding("skip_ew", "target_empty_ws").await;
            let msg = gui_msg("skip_ew", content);
            super::mirror_gui_message_to_telegram(&msg).await;
            let guard = sent.lock().unwrap_poison();
            let our_msgs: Vec<_> = guard
                .iter()
                .filter(|m| m.recipient == "target_empty_ws")
                .collect();
            assert!(
                our_msgs.is_empty(),
                "content {content:?} should not send, got {} message(s)",
                our_msgs.len()
            );
        }
    }

    #[tokio::test]
    async fn skip_user_with_no_bindings() {
        let (sent, _lock) = setup_mirror_test_env().await;
        // Create user but DO NOT bind a Telegram channel.
        let store = crate::users::store();
        store.add_user("no_binding", None).await.expect("add_user");

        // Use the user's name as the recipient filter — no bindings means
        // no messages should be sent for this user at all.
        let user_name = "no_binding";
        let msg = gui_msg(user_name, "hello");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard.iter().filter(|m| m.recipient == user_name).collect();
        assert!(our_msgs.is_empty(), "user with no bindings should not send");
    }

    #[tokio::test]
    async fn skip_binding_without_reply_target() {
        let (sent, _lock) = setup_mirror_test_env().await;
        // Bind a Telegram channel but don't set reply_target.
        let store = crate::users::store();
        store.add_user("no_target", None).await.expect("add_user");
        store
            .bind_channel("no_target", "telegram", "no_target")
            .await
            .expect("bind_channel");
        // Note: skip update_channel_contact → reply_target stays NULL.

        let msg = gui_msg("no_target", "hello");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "no_target")
            .collect();
        assert!(
            our_msgs.is_empty(),
            "binding without reply_target should not send"
        );
    }

    #[tokio::test]
    async fn skip_media_only_content() {
        let (sent, _lock) = setup_mirror_test_env().await;
        setup_user_with_telegram_binding("media_only", "target_media").await;

        let msg = gui_msg("media_only", "[IMAGE:/path/to/img.png]");
        super::mirror_gui_message_to_telegram(&msg).await;
        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "target_media")
            .collect();
        assert!(our_msgs.is_empty(), "media-only content should not send");
    }

    // ── Happy path tests ─────────────────────────────────────────

    #[tokio::test]
    async fn sends_blockquote_to_single_binding() {
        let (sent, _lock) = setup_mirror_test_env().await;
        setup_user_with_telegram_binding("single_user", "unique_single").await;

        let msg = gui_msg("single_user", "Hello, world!");
        super::mirror_gui_message_to_telegram(&msg).await;

        let guard = sent.lock().unwrap_poison();
        // Filter to our test's messages by recipient.
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "unique_single")
            .collect();
        assert_eq!(our_msgs.len(), 1, "expected exactly one message");
        assert_eq!(
            our_msgs[0].content,
            "<blockquote>\nHello, world!\n</blockquote>"
        );
        assert!(our_msgs[0].reply_markup.is_none());
    }

    #[tokio::test]
    async fn sends_to_multiple_telegram_bindings() {
        let (sent, _lock) = setup_mirror_test_env().await;
        let store = crate::users::store();
        store.add_user("multi_user", None).await.expect("add_user");
        // Bind two Telegram accounts with unique recipients.
        store
            .bind_channel("multi_user", "telegram", "multi_user_1")
            .await
            .expect("bind_channel_1");
        store
            .bind_channel("multi_user", "telegram", "multi_user_2")
            .await
            .expect("bind_channel_2");
        store
            .update_channel_contact("telegram", "multi_user_1", "unique_multi_a")
            .await
            .expect("update_channel_contact_1");
        store
            .update_channel_contact("telegram", "multi_user_2", "unique_multi_b")
            .await
            .expect("update_channel_contact_2");

        let msg = gui_msg("multi_user", "Hi both!");
        super::mirror_gui_message_to_telegram(&msg).await;

        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "unique_multi_a" || m.recipient == "unique_multi_b")
            .collect();
        assert_eq!(our_msgs.len(), 2, "expected two messages (one per binding)");
        // Both should have the same content.
        for m in &our_msgs {
            assert_eq!(m.content, "<blockquote>\nHi both!\n</blockquote>");
        }
        let recipients: Vec<&str> = our_msgs.iter().map(|m| m.recipient.as_str()).collect();
        assert!(recipients.contains(&"unique_multi_a"));
        assert!(recipients.contains(&"unique_multi_b"));
    }

    #[tokio::test]
    async fn strips_media_markers_from_content() {
        let (sent, _lock) = setup_mirror_test_env().await;
        setup_user_with_telegram_binding("strip_markers", "unique_markers").await;

        let msg = gui_msg(
            "strip_markers",
            "Check this [IMAGE:/tmp/screenshot.png] and my [AUDIO:/tmp/recording.mp3]",
        );
        super::mirror_gui_message_to_telegram(&msg).await;

        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "unique_markers")
            .collect();
        assert_eq!(our_msgs.len(), 1);
        // Markers should be stripped entirely (trailing whitespace is trimmed).
        assert_eq!(
            our_msgs[0].content,
            "<blockquote>\nCheck this  and my\n</blockquote>"
        );
    }

    #[tokio::test]
    async fn preserves_markdown_formatting_in_blockquote() {
        let (sent, _lock) = setup_mirror_test_env().await;
        setup_user_with_telegram_binding("md_user", "unique_md").await;

        let msg = gui_msg("md_user", "**bold** and `code` and *italic*");
        super::mirror_gui_message_to_telegram(&msg).await;

        let guard = sent.lock().unwrap_poison();
        let our_msgs: Vec<_> = guard
            .iter()
            .filter(|m| m.recipient == "unique_md")
            .collect();
        assert_eq!(our_msgs.len(), 1);
        // Markdown syntax inside the blockquote passes through — the Telegram
        // channel's markdown_to_telegram_html will handle formatting later.
        assert_eq!(
            our_msgs[0].content,
            "<blockquote>\n**bold** and `code` and *italic*\n</blockquote>"
        );
    }
}
