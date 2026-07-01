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
const HISTORY_LIMIT: i64 = 100;

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
            _ => ChatDirection::User,
        },
        agent_role: row.get::<Option<String>>(COL_CH_AGENT_ROLE)?,
    })
}

/// Parse rows into [`ChatHistoryEntry`], then reverse for chronological order.
/// Shared helper for `load_for_user` and `load_older_for_user`.
fn rows_to_history_entries(rows: Vec<Row>) -> Result<Vec<ChatHistoryEntry>> {
    let mut entries = Vec::with_capacity(rows.len());
    for row in rows {
        entries.push(chat_history_entry_from_row(&row)?);
    }
    entries.reverse();
    Ok(entries)
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

    /// Load the 100 most recent messages for a user + workspace pair,
    /// returned in chronological order (oldest first).
    pub async fn load_for_user(
        &self,
        user_name: &str,
        workspace: &str,
    ) -> Result<Vec<ChatHistoryEntry>> {
        // Query newest first (DESC), then reverse in memory for display order.
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
                turso::params![user_name, workspace, HISTORY_LIMIT],
            )
            .await?;
        rows_to_history_entries(rows)
    }

    /// Load messages older than `before_id` for a user + workspace pair.
    /// Queries `LIMIT limit + 1` to detect whether more entries exist.
    /// Returns entries in chronological order (oldest first).
    pub async fn load_older_for_user(
        &self,
        user_name: &str,
        workspace: &str,
        before_id: i64,
    ) -> Result<Vec<ChatHistoryEntry>> {
        let limit = HISTORY_LIMIT + 1; // fetch one extra to detect has_more
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
                turso::params![user_name, workspace, before_id, limit],
            )
            .await?;
        rows_to_history_entries(rows)
    }

    /// Delete all chat history entries for a specific user + workspace pair.
    /// Used by the Home page Clear button to ensure cleared sessions don't
    /// reappear on history refresh.
    pub async fn delete_for_user(&self, user_name: &str, workspace: &str) -> Result<u64> {
        let deleted = self
            .conn
            .execute(
                "DELETE FROM chat_history WHERE user_name = ?1 AND workspace = ?2",
                turso::params![user_name, workspace],
            )
            .await?;
        Ok(deleted)
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
        let history = store
            .load_for_user("user", "ws")
            .await
            .expect("load should succeed");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "hello");
    }
}
