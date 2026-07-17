//! Chat history persistence — stores all chat messages (both user and agent)
//! for GUI display and history loading. Messages are written at the point of
//! delivery: incoming user messages from the GUI send path, outgoing agent
//! responses from `GuiChannel::send()`.
//!
//! Each message gets a NanoID for deduplication.

use crate::ChatDirection;
use crate::turso::{self, Row};
use anyhow::Result;

crate::define_store! {
    /// Global chat history store.
    pub static CHAT_HISTORY: ChatHistoryStore,
    db_name = "chat_history",
    schema = SCHEMA,
    expect = "CHAT_HISTORY not initialized — call init_global() first",
}

const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    user_name TEXT NOT NULL,
    channel TEXT NOT NULL,
    role TEXT NOT NULL,
    direction TEXT NOT NULL,
    content TEXT NOT NULL,
    agent_role TEXT,
    workspace TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_name, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_workspace ON chat_history(workspace, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_channel ON chat_history(channel, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_user_ws_id ON chat_history(user_name, workspace, id);
";

/// Parameters for inserting a chat history entry.
///
/// This struct bundles the 9 fields needed by [`ChatHistoryStore::insert`],
/// replacing the previous positional-parameter signature. Owned `String` fields
/// match the pattern established by [`LogEntry`](crate::logs::LogEntry).
#[derive(Debug, Clone)]
pub struct ChatHistoryInsert {
    pub message_id: String,
    pub user_name: String,
    pub channel: String,
    pub role: String,
    pub direction: String,
    pub content: String,
    pub agent_role: Option<String>,
    pub workspace: String,
    pub created_at: String,
}

/// A single chat message record for history display.
#[derive(Debug, Clone)]
pub struct ChatHistoryEntry {
    pub id: i64,
    pub message_id: String,
    /// The user's canonical name for both user and agent messages. Use `agent_role` to identify which role produced an agent message.
    pub user_name: String,
    pub content: String,
    pub direction: ChatDirection,
    pub agent_role: Option<String>,
}

/// Maximum number of history entries to load at once.
const HISTORY_LIMIT: usize = 100;

// Column definitions for `chat_history` SELECT queries.
crate::columns! {
    CHAT_HISTORY_COLUMNS [CH] {
        ID          => "id",
        MESSAGE_ID  => "message_id",
        USER_NAME   => "user_name",
        CONTENT     => "content",
        DIRECTION   => "direction",
        AGENT_ROLE  => "agent_role",
    }
}

/// Convert a database row to a [`ChatHistoryEntry`] using the column-index
/// constants from [`CHAT_HISTORY_COLUMNS`].
fn chat_history_entry_from_row(row: &Row) -> Result<ChatHistoryEntry> {
    Ok(ChatHistoryEntry {
        id: row.get::<i64>(COL_CH_ID)?,
        message_id: row.get::<String>(COL_CH_MESSAGE_ID)?,
        user_name: row.get::<String>(COL_CH_USER_NAME)?,
        content: row.get::<String>(COL_CH_CONTENT)?,
        direction: match row.get::<String>(COL_CH_DIRECTION)?.as_str() {
            "agent" => ChatDirection::Agent,
            "divider" => ChatDirection::Divider,
            _ => ChatDirection::User,
        },
        agent_role: row.get::<Option<String>>(COL_CH_AGENT_ROLE)?,
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
    /// Silently ignores duplicate `message_id` values (UPSERT no-op).
    pub async fn insert(&self, entry: &ChatHistoryInsert) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO chat_history \
                 (message_id, user_name, channel, role, direction, \
                  content, agent_role, workspace, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                turso::params![
                    entry.message_id.clone(),
                    entry.user_name.clone(),
                    entry.channel.clone(),
                    entry.role.clone(),
                    entry.direction.clone(),
                    entry.content.clone(),
                    entry.agent_role.clone(),
                    entry.workspace.clone(),
                    entry.created_at.clone(),
                ],
            )
            .await?;
        Ok(())
    }

    /// Load the most recent messages for a user + workspace pair,
    /// returned in chronological order (oldest first).
    /// Returns `(entries, has_more)` where `has_more` is `true` if older
    /// entries exist beyond the loaded window.
    pub async fn load_for_user(
        &self,
        user_name: &str,
        workspace: &str,
    ) -> Result<(Vec<ChatHistoryEntry>, bool)> {
        #[allow(clippy::cast_possible_wrap)]
        let query_limit = HISTORY_LIMIT as i64 + 1; // fetch one extra to detect has_more
        let rows = self
            .conn
            .query(
                &format!(
                    "SELECT {CHAT_HISTORY_COLUMNS} \
                     FROM chat_history \
                     WHERE user_name = ?1 AND workspace = ?2 \
                     ORDER BY id DESC \
                     LIMIT ?3",
                ),
                turso::params![user_name, workspace, query_limit],
            )
            .await?;
        rows_to_page(rows)
    }

    /// Load messages older than `before_id` for a user + workspace pair.
    /// Returns `(entries, has_more)` where `has_more` is `true` if even older
    /// entries exist beyond the loaded window. Returns entries in chronological
    /// order (oldest first).
    pub async fn load_older_for_user(
        &self,
        user_name: &str,
        workspace: &str,
        before_id: i64,
    ) -> Result<(Vec<ChatHistoryEntry>, bool)> {
        #[allow(clippy::cast_possible_wrap)]
        let query_limit = HISTORY_LIMIT as i64 + 1; // fetch one extra to detect has_more
        let rows = self
            .conn
            .query(
                &format!(
                    "SELECT {CHAT_HISTORY_COLUMNS} \
                     FROM chat_history \
                     WHERE user_name = ?1 AND workspace = ?2 AND id < ?3 \
                     ORDER BY id DESC \
                     LIMIT ?4",
                ),
                turso::params![user_name, workspace, before_id, query_limit],
            )
            .await?;
        rows_to_page(rows)
    }

    /// Insert a divider marker row into chat history to indicate where a
    /// session clear occurred. The row uses `role='divider'` and
    /// `direction='divider'` so the GUI can detect it and render a visible
    /// separator instead of a chat bubble.
    pub async fn insert_divider(
        &self,
        user_name: &str,
        channel: &str,
        workspace: &str,
    ) -> Result<()> {
        let message_id = crate::generate_id();
        let created_at = turso::now();
        self.conn
            .execute(
                "INSERT INTO chat_history \
                 (message_id, user_name, channel, role, direction, \
                  content, agent_role, workspace, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                turso::params![
                    message_id,
                    user_name,
                    channel,
                    "divider",          // role
                    "divider",          // direction
                    created_at.clone(), // content — stores the timestamp
                    None::<String>,     // agent_role
                    workspace,
                    created_at,
                ],
            )
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turso;
    use tempfile::TempDir;

    async fn test_setup() -> (ChatHistoryStore, TempDir) {
        crate::open_test_store!(ChatHistoryStore, "chat_history")
    }

    #[tokio::test]
    async fn test_open_smoke() {
        let (store, _tmp) = test_setup().await;

        // Verify there is no session_key column.
        let rows = store
            .conn
            .query(
                "SELECT 1 FROM pragma_table_info('chat_history') WHERE name = 'session_key'",
                turso::params![],
            )
            .await
            .expect("Failed to check column existence");
        assert!(
            rows.is_empty(),
            "session_key should not exist in fresh schema"
        );

        // Verify basic insert and load work.
        store
            .insert(&ChatHistoryInsert {
                message_id: "msg-1".to_string(),
                user_name: "user".to_string(),
                channel: "test".to_string(),
                role: "user".to_string(),
                direction: "user".to_string(),
                content: "hello".to_string(),
                agent_role: None,
                workspace: "ws".to_string(),
                created_at: "now".to_string(),
            })
            .await
            .expect("insert should succeed");
        let (history, has_more) = store
            .load_for_user("user", "ws")
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
            .insert_divider("alice", "gui", "ws1")
            .await
            .expect("insert_divider should succeed");

        // Load history for the same user+workspace.
        let (history, has_more) = store
            .load_for_user("alice", "ws1")
            .await
            .expect("load_for_user should succeed");

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
            .load_for_user("bob", "ws1")
            .await
            .expect("other user load should succeed");
        assert!(
            other_history.is_empty(),
            "divider inserted for alice should not appear in bob's history"
        );

        // Verify the divider is *not* present in another workspace's history.
        let (other_ws_history, _) = store
            .load_for_user("alice", "ws2")
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
            .insert_divider("alice", "gui", "ws1")
            .await
            .expect("first divider should succeed");
        store
            .insert_divider("alice", "gui", "ws1")
            .await
            .expect("second divider should succeed");

        let (history, has_more) = store
            .load_for_user("alice", "ws1")
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
                channel: "gui".to_string(),
                role: "user".to_string(),
                direction: "user".to_string(),
                content: "hello".to_string(),
                agent_role: None,
                workspace: "ws1".to_string(),
                created_at: turso::now(),
            })
            .await
            .expect("insert should succeed");

        // Insert a divider.
        store
            .insert_divider("alice", "gui", "ws1")
            .await
            .expect("insert_divider should succeed");

        // Insert another message after the divider.
        store
            .insert(&ChatHistoryInsert {
                message_id: "msg-2".to_string(),
                user_name: "alice".to_string(),
                channel: "gui".to_string(),
                role: "user".to_string(),
                direction: "user".to_string(),
                content: "world".to_string(),
                agent_role: None,
                workspace: "ws1".to_string(),
                created_at: turso::now(),
            })
            .await
            .expect("insert should succeed");

        // Load all three.
        let (history, has_more) = store
            .load_for_user("alice", "ws1")
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
}
