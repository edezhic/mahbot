//! Chat history persistence — stores all chat messages (both user and agent)
//! for GUI display and history loading. Messages are written by the central
//! broadcast-and-persist paths: `broadcast_and_persist_incoming_message` for
//! incoming user messages and `broadcast_and_persist_agent_response` /
//! `BroadcastPersistEntry` for outgoing agent responses.
//!
//! Each message gets a NanoID for deduplication.

use crate::ChatDirection;
use crate::channels::ReplyReference;
use crate::db::{self, Row};
use anyhow::Result;

crate::define_store! {
    /// Global chat history store.
    pub(crate) static CHAT_HISTORY: ChatHistoryStore,
    expect = "CHAT_HISTORY not initialized — call init_all_stores() first",
}

/// Parameters for inserting a chat history entry.
///
/// This struct bundles the fields needed by [`ChatHistoryStore::insert`].
/// Owned `String` fields match the pattern established by
/// [`LogEntry`](crate::logs::LogEntry).
#[derive(Debug, Clone)]
pub struct ChatHistoryInsert {
    pub message_id: String,
    pub user_name: String,
    pub direction: String,
    pub content: String,
    pub agent_role: Option<String>,
    /// Identifier of the delivery dispatch that wrote this row where the writer
    /// tags one (the agent delivery paths do); `None` on a user's own message,
    /// on the voice delivery row, and on legacy data.
    pub broadcast_id: Option<String>,
    pub workspace: String,
    pub timestamp: Option<String>,
    /// Optional reference to the replied-to message, rendered as a quote
    /// header by the GUI.
    pub reply_reference: Option<ReplyReference>,
}

/// A single chat message record for history display.
#[derive(Debug, Clone)]
pub struct ChatHistoryEntry {
    pub id: i64,
    pub message_id: String,
    pub content: String,
    pub direction: ChatDirection,
    pub agent_role: Option<String>,
    pub timestamp: Option<String>,
    /// Optional reference to the replied-to message for quote-header rendering.
    pub reply_reference: Option<ReplyReference>,
}

/// Maximum number of history entries to load at once.
const HISTORY_LIMIT: usize = 100;

// Column definitions for `chat_history` SELECT queries.
crate::columns! {
    CHAT_HISTORY_COLUMNS [CH] {
        ID           => "id",
        MESSAGE_ID   => "message_id",
        CONTENT      => "content",
        DIRECTION    => "direction",
        AGENT_ROLE   => "agent_role",
        TIMESTAMP    => "timestamp",
        REPLY_AUTHOR => "reply_author",
        REPLY_SNIPPET => "reply_snippet",
    }
}

/// Convert a database row to a [`ChatHistoryEntry`] using the column-index
/// constants from [`CHAT_HISTORY_COLUMNS`].
fn chat_history_entry_from_row(row: &Row) -> Result<ChatHistoryEntry> {
    let reply_author = row.get::<Option<String>>(COL_CH_REPLY_AUTHOR)?;
    let reply_snippet = row.get::<Option<String>>(COL_CH_REPLY_SNIPPET)?;
    let reply_reference = match (reply_author, reply_snippet) {
        (Some(author), Some(snippet)) => Some(ReplyReference { author, snippet }),
        _ => None,
    };
    Ok(ChatHistoryEntry {
        id: row.get::<i64>(COL_CH_ID)?,
        message_id: row.get::<String>(COL_CH_MESSAGE_ID)?,
        content: row.get::<String>(COL_CH_CONTENT)?,
        direction: match row.get::<String>(COL_CH_DIRECTION)?.as_str() {
            "agent" => ChatDirection::Agent,
            "divider" => ChatDirection::Divider,
            _ => ChatDirection::User,
        },
        agent_role: row.get::<Option<String>>(COL_CH_AGENT_ROLE)?,
        timestamp: row.get::<Option<String>>(COL_CH_TIMESTAMP)?,
        reply_reference,
    })
}

/// Process rows from a query that over-fetched by one row (limit = history
/// limit + 1) into a page of entries with a `has_more` flag. Entries are
/// returned in chronological order. The extra over-fetched row (if any) is
/// dropped from the front (oldest entries) so the returned vector contains
/// at most [`HISTORY_LIMIT`] entries.
fn rows_to_page(rows: Vec<Row>) -> Result<(Vec<ChatHistoryEntry>, bool)> {
    let mut entries: Vec<ChatHistoryEntry> = Vec::with_capacity(rows.len());
    for row in rows {
        entries.push(chat_history_entry_from_row(&row)?);
    }
    entries.reverse();
    let has_more = entries.len() > HISTORY_LIMIT;
    if has_more {
        // Entries are in chronological order (oldest first).
        // We over-fetched by 1 to detect has_more; remove the oldest entry
        // (at the front) so we return exactly HISTORY_LIMIT entries.
        entries.drain(0..(entries.len() - HISTORY_LIMIT));
    }
    Ok((entries, has_more))
}

impl ChatHistoryStore {
    /// Insert a message into the history. `message_id` is a NanoID for dedup.
    /// Silently ignores duplicate `message_id` values (`INSERT OR IGNORE`).
    pub async fn insert(&self, entry: &ChatHistoryInsert) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO chat_history \
                 (message_id, user_name, direction, content, agent_role, workspace, timestamp, \
                  reply_author, reply_snippet, broadcast_id) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                db::params![
                    entry.message_id.clone(),
                    entry.user_name.clone(),
                    entry.direction.clone(),
                    entry.content.clone(),
                    entry.agent_role.clone(),
                    entry.workspace.clone(),
                    entry.timestamp.clone(),
                    entry.reply_reference.as_ref().map(|r| r.author.clone()),
                    entry.reply_reference.as_ref().map(|r| r.snippet.clone()),
                    entry.broadcast_id.clone(),
                ],
            )
            .await?;
        Ok(())
    }

    /// Shared paging body for the `load_*` methods. Reads exactly one
    /// conversation — a user in one workspace. `before_id = None` loads the
    /// most recent page; `Some(id)` loads only messages older than `id`.
    /// Ordered by row id, so the composite index
    /// `idx_chat_history_user_ws_id (user_name, workspace, id)` serves the
    /// predicate and the ordering directly.
    async fn load_page(
        &self,
        user_name: &str,
        workspace: &str,
        before_id: Option<i64>,
    ) -> Result<(Vec<ChatHistoryEntry>, bool)> {
        #[expect(clippy::cast_possible_wrap)]
        let query_limit = HISTORY_LIMIT as i64 + 1; // fetch one extra to detect has_more
        let rows = self
            .conn
            .query(
                &format!(
                    "SELECT {CHAT_HISTORY_COLUMNS} \
                     FROM chat_history \
                     WHERE user_name = ?1 AND workspace = ?2 \
                       AND (?3 IS NULL OR id < ?3) \
                     ORDER BY id DESC \
                     LIMIT ?4",
                ),
                db::params![user_name, workspace, before_id, query_limit],
            )
            .await?;
        rows_to_page(rows)
    }

    /// Load the most recent page of one conversation (a user in one
    /// workspace). Entries are in chronological order (oldest first);
    /// `has_more` is true when older entries exist beyond the loaded window.
    pub async fn load_recent(
        &self,
        user_name: &str,
        workspace: &str,
    ) -> Result<(Vec<ChatHistoryEntry>, bool)> {
        self.load_page(user_name, workspace, None).await
    }

    /// Load the page of one conversation older than `before_id`. Entries are
    /// in chronological order (oldest first); `has_more` is true when even
    /// older entries exist beyond the loaded window.
    pub async fn load_older(
        &self,
        user_name: &str,
        workspace: &str,
        before_id: i64,
    ) -> Result<(Vec<ChatHistoryEntry>, bool)> {
        self.load_page(user_name, workspace, Some(before_id)).await
    }

    /// Insert a divider marker row into chat history to indicate where a
    /// session clear occurred. The row uses `direction='divider'` so the GUI
    /// can detect it and render a visible separator instead of a chat bubble.
    pub async fn insert_divider(&self, user_name: &str, workspace: &str) -> Result<()> {
        let message_id = crate::generate_id();
        self.insert(&ChatHistoryInsert {
            message_id,
            user_name: user_name.to_string(),
            direction: "divider".to_string(),
            content: db::now(),
            agent_role: None,
            broadcast_id: None,
            workspace: workspace.to_string(),
            timestamp: None,
            reply_reference: None,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db;
    use tempfile::TempDir;

    async fn test_setup() -> (ChatHistoryStore, TempDir) {
        crate::open_test_store!(ChatHistoryStore, "chat_history")
    }

    #[tokio::test]
    async fn test_open_smoke() {
        let (store, _tmp) = test_setup().await;

        // Verify there is no agent_id column.
        let rows = store
            .conn
            .query(
                "SELECT 1 FROM pragma_table_info('chat_history') WHERE name = 'agent_id'",
                db::params![],
            )
            .await
            .expect("Failed to check column existence");
        assert!(rows.is_empty(), "agent_id should not exist in fresh schema");

        // Verify basic insert and load work.
        store
            .insert(&ChatHistoryInsert {
                message_id: "msg-1".to_string(),
                user_name: "user".to_string(),
                direction: "user".to_string(),
                content: "hello".to_string(),
                agent_role: None,
                broadcast_id: None,
                workspace: "ws".to_string(),
                timestamp: None,
                reply_reference: None,
            })
            .await
            .expect("insert should succeed");
        let (history, has_more) = store
            .load_recent("user", "ws")
            .await
            .expect("load should succeed");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "hello");
        assert!(!has_more);
    }

    #[tokio::test]
    async fn test_insert_divider_roundtrip() {
        let (store, _tmp) = test_setup().await;

        // Insert a divider marker.
        store
            .insert_divider("alice", "ws1")
            .await
            .expect("insert_divider should succeed");

        // Load history for the same user+workspace.
        let (history, has_more) = store
            .load_recent("alice", "ws1")
            .await
            .expect("load should succeed");

        // Should have exactly one entry: the divider.
        assert_eq!(history.len(), 1, "should have exactly one divider entry");
        assert!(!has_more, "no more entries beyond the divider");

        let entry = &history[0];

        // Verify the entry is detected as a divider.
        assert_eq!(
            entry.direction,
            ChatDirection::Divider,
            "divider entry should have ChatDirection::Divider"
        );

        // agent_role should be None for divider rows.
        assert!(
            entry.agent_role.is_none(),
            "divider should have no agent_role"
        );

        // content should be a non-empty timestamp (RFC 3339).
        assert!(
            !entry.content.is_empty(),
            "divider content (timestamp) should not be empty"
        );
        assert!(
            entry.content.contains('T'),
            "divider content should be an ISO 8601 timestamp, got: {}",
            entry.content
        );

        // Verify the divider is *not* present in another user's history.
        let (other_history, _) = store
            .load_recent("bob", "ws1")
            .await
            .expect("other user load should succeed");
        assert!(
            other_history.is_empty(),
            "divider inserted for alice should not appear in bob's history"
        );

        // Verify the divider is *not* present in another workspace's history.
        let (other_ws_history, _) = store
            .load_recent("alice", "ws2")
            .await
            .expect("other workspace load should succeed");
        assert!(
            other_ws_history.is_empty(),
            "divider inserted for ws1 should not appear in ws2's history"
        );
    }

    #[tokio::test]
    async fn test_insert_multiple_dividers() {
        let (store, _tmp) = test_setup().await;

        // Insert two dividers.
        store
            .insert_divider("alice", "ws1")
            .await
            .expect("first divider should succeed");
        store
            .insert_divider("alice", "ws1")
            .await
            .expect("second divider should succeed");

        let (history, has_more) = store
            .load_recent("alice", "ws1")
            .await
            .expect("load should succeed");

        // Both dividers should be present.
        assert_eq!(history.len(), 2, "should have two dividers");
        assert!(!has_more);
        assert_eq!(history[0].direction, ChatDirection::Divider);
        assert_eq!(history[1].direction, ChatDirection::Divider);

        // The first inserted divider should be older (lower id, ordered chronologically).
        assert!(
            history[0].id < history[1].id,
            "first inserted divider should have a lower id"
        );
    }

    #[tokio::test]
    async fn test_divider_mixed_with_messages() {
        let (store, _tmp) = test_setup().await;

        // Insert a regular user message first.
        store
            .insert(&ChatHistoryInsert {
                message_id: "msg-1".to_string(),
                user_name: "alice".to_string(),
                direction: "user".to_string(),
                content: "hello".to_string(),
                agent_role: None,
                broadcast_id: None,
                workspace: "ws1".to_string(),
                timestamp: None,
                reply_reference: None,
            })
            .await
            .expect("insert should succeed");

        // Insert a divider.
        store
            .insert_divider("alice", "ws1")
            .await
            .expect("insert_divider should succeed");

        // Insert another message after the divider.
        store
            .insert(&ChatHistoryInsert {
                message_id: "msg-2".to_string(),
                user_name: "alice".to_string(),
                direction: "user".to_string(),
                content: "world".to_string(),
                agent_role: None,
                broadcast_id: None,
                workspace: "ws1".to_string(),
                timestamp: None,
                reply_reference: None,
            })
            .await
            .expect("insert should succeed");

        // Load all three.
        let (history, has_more) = store
            .load_recent("alice", "ws1")
            .await
            .expect("load should succeed");

        // Load limit is 100, all three should fit.
        assert_eq!(history.len(), 3, "should have all three entries");
        assert!(!has_more);

        // Chronological order (oldest first).
        assert_eq!(history[0].direction, ChatDirection::User);
        assert_eq!(history[0].content, "hello");

        assert_eq!(history[1].direction, ChatDirection::Divider);

        assert_eq!(history[2].direction, ChatDirection::User);
        assert_eq!(history[2].content, "world");
    }

    #[tokio::test]
    async fn test_load_older_paginates_in_one_conversation() {
        let (store, _tmp) = test_setup().await;

        // 6 messages in the conversation's own workspace.
        for i in 0..6 {
            store
                .insert(&ChatHistoryInsert {
                    message_id: format!("msg-{i}"),
                    user_name: "alice".to_string(),
                    direction: "user".to_string(),
                    content: format!("c{i}"),
                    agent_role: None,
                    broadcast_id: None,
                    workspace: "personal:alice".to_string(),
                    timestamp: None,
                    reply_reference: None,
                })
                .await
                .expect("insert should succeed");
        }

        // Noise that must never leak into the conversation read: the same user
        // in another workspace, and another user in the conversation's workspace.
        for (message_id, user, ws) in [
            ("noise-0", "alice", "project"),
            ("noise-1", "alice", "project"),
            ("noise-2", "bob", "personal:alice"),
        ] {
            store
                .insert(&ChatHistoryInsert {
                    message_id: message_id.to_string(),
                    user_name: user.to_string(),
                    direction: "user".to_string(),
                    content: message_id.to_string(),
                    agent_role: None,
                    broadcast_id: None,
                    workspace: ws.to_string(),
                    timestamp: None,
                    reply_reference: None,
                })
                .await
                .expect("insert should succeed");
        }

        let (history, has_more) = store
            .load_recent("alice", "personal:alice")
            .await
            .expect("load should succeed");
        assert_eq!(history.len(), 6);
        assert!(!has_more);
        let contents: Vec<&str> = history.iter().map(|e| e.content.as_str()).collect();
        assert_eq!(contents, ["c0", "c1", "c2", "c3", "c4", "c5"]);
        let oldest_id = history[0].id;

        // Load older than the oldest id — nothing left in the conversation.
        let (older, has_more_older) = store
            .load_older("alice", "personal:alice", oldest_id)
            .await
            .expect("load older should succeed");
        assert!(older.is_empty());
        assert!(!has_more_older);

        // Loading older than the 4th message returns the 3 older ones.
        let (older, _) = store
            .load_older("alice", "personal:alice", history[3].id)
            .await
            .expect("load older should succeed");
        let contents: Vec<&str> = older.iter().map(|e| e.content.as_str()).collect();
        assert_eq!(contents, ["c0", "c1", "c2"]);
    }

    #[tokio::test]
    async fn test_reply_reference_roundtrip() {
        let (store, _tmp) = test_setup().await;

        store
            .insert(&ChatHistoryInsert {
                message_id: "msg-reply".to_string(),
                user_name: "alice".to_string(),
                direction: "user".to_string(),
                content: "hello".to_string(),
                agent_role: None,
                broadcast_id: None,
                workspace: "ws1".to_string(),
                timestamp: Some(crate::db::now()),
                reply_reference: Some(crate::channels::ReplyReference {
                    author: "bob".to_string(),
                    snippet: "earlier message".to_string(),
                }),
            })
            .await
            .expect("insert should succeed");

        let (history, _) = store
            .load_recent("alice", "ws1")
            .await
            .expect("load should succeed");
        assert_eq!(history.len(), 1);
        let reply = history[0]
            .reply_reference
            .as_ref()
            .expect("reply reference should survive the round-trip");
        assert_eq!(reply.author, "bob");
        assert_eq!(reply.snippet, "earlier message");
    }
}
