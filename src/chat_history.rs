//! Chat history persistence — stores all chat messages (both user and agent)
//! for GUI display and history loading. Messages are written at the point of
//! delivery: incoming user messages from the GUI send path, outgoing agent
//! responses from `GuiChannel::send()`.
//!
//! Each message gets a NanoID for deduplication.

use crate::ChatDirection;
use crate::global_store;
use crate::turso::{self, Connection, Row};
use anyhow::Result;
use std::path::Path;

global_store! {
    /// Global chat history store.
    pub static CHAT_HISTORY: ChatHistoryStore,
    constructor = ChatHistoryStore::open,
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

/// A single chat message record for history display.
#[derive(Debug, Clone)]
pub struct ChatHistoryEntry {
    pub id: i64,
    pub message_id: String,
    /// The user's canonical name for user messages; agent role name for agent messages.
    pub user_name: String,
    pub content: String,
    pub direction: ChatDirection,
    pub agent_role: Option<String>,
}

/// Maximum number of history entries to load at once.
const HISTORY_LIMIT: i64 = 100;

/// Column list for chat history SELECT queries.
///
/// The column order here must match the positional indices defined in
/// [`COL_CH_ID`] through [`COL_CH_AGENT_ROLE`], which are used in
/// [`entry_from_row`].
///
/// Note: The column order differs from the schema declaration order
/// (id, message_id, user_name, channel, role, direction, content,
/// agent_role, workspace, created_at). The `COL_CH_*` constants are
/// the source of truth for field mapping; the
/// [`chat_history_columns_count_matches_column_constants`] test
/// catches count drift.
const CHAT_HISTORY_COLUMNS: &str = "id, message_id, user_name, content, direction, agent_role";

/// Column-index constants for [`CHAT_HISTORY_COLUMNS`].
///
/// These replace hardcoded positional indices in [`entry_from_row`].
/// With named constants, the compiler catches references to undefined
/// column constants — for instance, removing a constant but forgetting to
/// update a `row.get()` call produces a compile error rather than a silent
/// field mapping bug.
const COL_CH_ID: usize = 0;
const COL_CH_MESSAGE_ID: usize = 1;
const COL_CH_USER_NAME: usize = 2;
const COL_CH_CONTENT: usize = 3;
const COL_CH_DIRECTION: usize = 4;
const COL_CH_AGENT_ROLE: usize = 5;

/// Convert a database row to a [`ChatHistoryEntry`] using the column-index
/// constants from [`CHAT_HISTORY_COLUMNS`].
fn entry_from_row(row: &Row) -> Result<ChatHistoryEntry> {
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

/// Turso-backed chat history storage.
#[derive(Clone, Debug)]
pub struct ChatHistoryStore {
    pub(crate) conn: Connection,
}

impl ChatHistoryStore {
    /// Open (or create) the chat history database at `root/db/chat_history.db`.
    pub async fn open(root: &Path) -> Result<Self> {
        let db_path = root.join("db/chat_history.db");
        let conn = turso::open_with_schema(&db_path, SCHEMA).await?;
        Ok(Self { conn })
    }

    /// Insert a message into the history. `message_id` is a NanoID for dedup.
    /// Silently ignores duplicate `message_id` values (UPSERT no-op).
    #[allow(clippy::too_many_arguments)]
    pub async fn insert(
        &self,
        message_id: &str,
        user_name: &str,
        channel: &str,
        role: &str,
        direction: &str,
        content: &str,
        agent_role: Option<&str>,
        workspace: &str,
        created_at: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO chat_history \
                 (message_id, user_name, channel, role, direction, \
                  content, agent_role, workspace, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                turso::params![
                    message_id, user_name, channel, role, direction, content, agent_role,
                    workspace, created_at,
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
        let mut entries = Vec::new();
        for row in rows {
            entries.push(entry_from_row(&row)?);
        }
        entries.reverse();
        Ok(entries)
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
        let mut entries = Vec::new();
        for row in rows {
            entries.push(entry_from_row(&row)?);
        }
        // Reverse for chronological display order.
        entries.reverse();
        Ok(entries)
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
    use crate::chat_history::ChatHistoryStore;
    use crate::turso;
    use tempfile::TempDir;

    /// Verify that the number of columns in [`CHAT_HISTORY_COLUMNS`] matches the highest
    /// column-index constant + 1. If this test fails, a column was added or removed
    /// from the string list without updating the corresponding `COL_CH_*` constants,
    /// or vice versa — a silent data corruption hazard.
    #[test]
    fn chat_history_columns_count_matches_column_constants() {
        crate::assert_column_count!(CHAT_HISTORY_COLUMNS, COL_CH_AGENT_ROLE);
    }

    fn test_setup() -> (TempDir, std::path::PathBuf) {
        let tmp = TempDir::new().expect("failed to create test temp dir");
        let root = tmp.path().to_path_buf();
        (tmp, root)
    }

    #[tokio::test]
    async fn test_open_smoke() {
        let (_tmp, root) = test_setup();

        let store = ChatHistoryStore::open(&root)
            .await
            .expect("ChatHistoryStore::open should succeed on fresh database");

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
            .insert(
                "msg-1", "user", "test", "user", "user", "hello", None, "ws", "now",
            )
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
