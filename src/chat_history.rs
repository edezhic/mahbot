//! Chat history persistence — stores all chat messages (both user and agent)
//! for GUI display and history loading. Messages are written at the point of
//! delivery: incoming user messages from the GUI send path, outgoing agent
//! responses from `GuiChannel::send()`.
//!
//! Each message gets a NanoID for deduplication.

use crate::ChatDirection;
use crate::config::CONFIG;
use crate::turso::{self, Connection};
use anyhow::{Context, Result};
use std::path::Path;
use tokio::sync::OnceCell;

/// Global chat history store.
pub static CHAT_HISTORY: OnceCell<ChatHistoryStore> = OnceCell::const_new();

/// Initialize the global chat history store.
pub async fn init_global() -> Result<()> {
    let root = CONFIG.global_storage_root();
    turso::register_global_store(&CHAT_HISTORY, "CHAT_HISTORY", || {
        ChatHistoryStore::open(&root)
    })
    .await
}

/// Get a reference to the global chat history store.
///
/// # Panics
///
/// Panics if the chat history store has not been initialized. All production
/// code initializes the store before any access, so this is a programming error.
#[must_use]
pub fn store() -> &'static ChatHistoryStore {
    CHAT_HISTORY
        .get()
        .expect("CHAT_HISTORY not initialized — call init_global() first")
}

/// Schema for fresh databases. The deprecated `session_key` column (always
/// inserted as `''` and never queried) was removed in migration v1 — see
/// [`ChatHistoryStore::open()`].
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
    pub workspace: String,
    pub created_at: String,
}

/// Maximum number of history entries to load at once.
const HISTORY_LIMIT: i64 = 100;

/// Turso-backed chat history storage.
#[derive(Clone, Debug)]
pub struct ChatHistoryStore {
    pub(crate) conn: Connection,
}

impl ChatHistoryStore {
    /// Open (or create) the chat history database at `root/db/chat_history.db`.
    ///
    /// ## Migration v1
    ///
    /// On first open after upgrade, drops the deprecated `session_key` column
    /// (always inserted as `''` and never queried). The column was removed from
    /// the SCHEMA constant, so fresh databases never create it.
    ///
    /// ### DDL auto-commit note
    ///
    /// `ALTER TABLE` is a DDL statement that implicitly commits any active
    /// transaction. At this point in `open()` no transaction is active, so
    /// there is nothing to implicitly commit. If the migration is interrupted
    /// (e.g., process crash) after `DROP INDEX` but before `ALTER TABLE`, the
    /// database is recoverable — `DROP INDEX IF EXISTS` is idempotent on retry
    /// and the `ALTER TABLE` then completes.
    ///
    /// ### Downgrade risk
    ///
    /// An old binary (pre-migration) that opens a migrated database will fail
    /// on INSERT because its SQL references the dropped `session_key` column.
    /// This is acceptable given MahBot's atomic self-update pattern.
    pub async fn open(root: &Path) -> Result<Self> {
        let db_path = root.join("db/chat_history.db");
        let conn = turso::open_with_schema(&db_path, SCHEMA).await?;

        // ── Schema migration v1 ────────────────────────────────────────
        //
        // PRAGMA user_version returns 0 for unmigrated databases (SQLite
        // default). We use this as a migration stamp to ensure the migration
        // runs exactly once.

        let user_version: i64 = conn
            .query_row("PRAGMA user_version", turso::params![], |row| {
                row.get::<Option<i64>>(0)
            })
            .await
            .context("Failed to read PRAGMA user_version")?
            .unwrap_or(0);

        if user_version < 1 {
            // Check if the deprecated session_key column exists (from a
            // pre-migration database created by an older binary).
            let has_session_key = {
                let rows = conn
                    .query(
                        "SELECT 1 FROM pragma_table_info('chat_history') \
                         WHERE name = 'session_key'",
                        turso::params![],
                    )
                    .await?;
                !rows.is_empty()
            };

            if has_session_key {
                // DROP INDEX must precede ALTER TABLE DROP COLUMN — SQLite
                // refuses to drop a column that is part of an index. This
                // guards the pre-v0.8 upgrade path where the original
                // unconditional DROP INDEX may not have run.
                conn.execute(
                    "DROP INDEX IF EXISTS idx_chat_history_session",
                    turso::params![],
                )
                .await?;

                conn.execute(
                    "ALTER TABLE chat_history DROP COLUMN session_key",
                    turso::params![],
                )
                .await
                .context(
                    "Failed to drop session_key column — verify the underlying \
                     engine supports ALTER TABLE DROP COLUMN",
                )?;
            }

            // Stamp the migration version. Must run unconditionally so that
            // fresh databases (which never had session_key) also get stamped.
            conn.execute("PRAGMA user_version = 1", turso::params![])
                .await
                .context("Failed to set PRAGMA user_version = 1")?;
        }

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
                "SELECT id, message_id, user_name, content, direction, agent_role, \
                 created_at, workspace \
                 FROM chat_history \
                 WHERE user_name = ?1 AND workspace = ?2 \
                 ORDER BY id DESC \
                 LIMIT ?3",
                turso::params![user_name, workspace, HISTORY_LIMIT],
            )
            .await?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(ChatHistoryEntry {
                id: row.get::<i64>(0)?,
                message_id: row.get::<String>(1)?,
                user_name: row.get::<String>(2)?,
                content: row.get::<String>(3)?,
                direction: match row.get::<String>(4)?.as_str() {
                    "agent" => ChatDirection::Agent,
                    _ => ChatDirection::User,
                },
                agent_role: row.get::<Option<String>>(5)?,
                created_at: row.get::<String>(6)?,
                workspace: row.get::<String>(7)?,
            });
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
                "SELECT id, message_id, user_name, content, direction, agent_role, \
                 created_at, workspace \
                 FROM chat_history \
                 WHERE user_name = ?1 AND workspace = ?2 AND id < ?3 \
                 ORDER BY id DESC \
                 LIMIT ?4",
                turso::params![user_name, workspace, before_id, limit],
            )
            .await?;
        let mut entries = Vec::new();
        for row in rows {
            entries.push(ChatHistoryEntry {
                id: row.get::<i64>(0)?,
                message_id: row.get::<String>(1)?,
                user_name: row.get::<String>(2)?,
                content: row.get::<String>(3)?,
                direction: match row.get::<String>(4)?.as_str() {
                    "agent" => ChatDirection::Agent,
                    _ => ChatDirection::User,
                },
                agent_role: row.get::<Option<String>>(5)?,
                created_at: row.get::<String>(6)?,
                workspace: row.get::<String>(7)?,
            });
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

/// Legacy schema for testing migration from a pre-v1 database.
/// Includes the deprecated `session_key` column and its optional index.
///
/// # Sync requirement
///
/// This constant must be kept in sync with [`SCHEMA`] for the index
/// statements — if a new index is added to [`SCHEMA`], it must also be
/// added here. There is no compile-time enforcement of this invariant.
#[cfg(test)]
const OLD_SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    session_key TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_key, id);
";

#[cfg(test)]
mod tests {
    use crate::chat_history::ChatHistoryStore;
    use crate::turso;
    use tempfile::TempDir;

    fn test_setup() -> (TempDir, std::path::PathBuf) {
        let tmp = TempDir::new().expect("failed to create test temp dir");
        let root = tmp.path().to_path_buf();
        (tmp, root)
    }

    #[tokio::test]
    async fn test_migration_from_old_schema() {
        let (_tmp, root) = test_setup();
        let db_path = root.join("db/chat_history.db");

        let conn = turso::open_with_schema(&db_path, super::OLD_SCHEMA)
            .await
            .expect("Failed to create legacy database");
        let has_session_key = conn
            .query(
                "SELECT 1 FROM pragma_table_info('chat_history') WHERE name = 'session_key'",
                turso::params![],
            )
            .await
            .expect("Failed to check column existence");
        assert!(
            !has_session_key.is_empty(),
            "session_key must exist in legacy schema"
        );
        drop(conn);

        // Migration v1 should drop the column and stamp user_version.
        let store = ChatHistoryStore::open(&root)
            .await
            .expect("ChatHistoryStore::open should succeed on legacy database");

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
            "session_key should have been dropped by migration v1"
        );

        let version: i64 = store
            .conn
            .query_row("PRAGMA user_version", turso::params![], |row| {
                row.get::<Option<i64>>(0)
            })
            .await
            .expect("Failed to read PRAGMA user_version")
            .unwrap_or(0);
        assert_eq!(version, 1, "user_version should be 1 after migration");

        // Verify insert() works on the migrated database.
        store
            .insert(
                "msg-1", "user", "test", "user", "user", "hello", None, "ws", "now",
            )
            .await
            .expect("insert should succeed after migration");
        let history = store
            .load_for_user("user", "ws")
            .await
            .expect("load should succeed");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "hello");

        drop(store);

        // Re-open — verify idempotency.
        let store2 = ChatHistoryStore::open(&root)
            .await
            .expect("Re-open should succeed");
        let version2: i64 = store2
            .conn
            .query_row("PRAGMA user_version", turso::params![], |row| {
                row.get::<Option<i64>>(0)
            })
            .await
            .expect("Failed to read PRAGMA user_version")
            .unwrap_or(0);
        assert_eq!(version2, 1, "user_version should still be 1 after re-open");
    }

    #[tokio::test]
    async fn test_fresh_schema() {
        let (_tmp, root) = test_setup();

        let store = ChatHistoryStore::open(&root)
            .await
            .expect("ChatHistoryStore::open should succeed on fresh database");

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

        let version: i64 = store
            .conn
            .query_row("PRAGMA user_version", turso::params![], |row| {
                row.get::<Option<i64>>(0)
            })
            .await
            .expect("Failed to read PRAGMA user_version")
            .unwrap_or(0);
        assert_eq!(version, 1, "user_version should be 1 on fresh database");
    }

    #[tokio::test]
    async fn test_already_migrated() {
        let (_tmp, root) = test_setup();
        let db_path = root.join("db/chat_history.db");

        let conn = turso::open_with_schema(&db_path, super::SCHEMA)
            .await
            .expect("Failed to create database");
        conn.execute("PRAGMA user_version = 1", turso::params![])
            .await
            .expect("Failed to set PRAGMA user_version");
        drop(conn);

        // Migration should be skipped since user_version >= 1.
        let store = ChatHistoryStore::open(&root)
            .await
            .expect("ChatHistoryStore::open should succeed on pre-stamped database");

        let version: i64 = store
            .conn
            .query_row("PRAGMA user_version", turso::params![], |row| {
                row.get::<Option<i64>>(0)
            })
            .await
            .expect("Failed to read PRAGMA user_version")
            .unwrap_or(0);
        assert_eq!(version, 1, "user_version should remain 1");
    }
}
