//! Chat history persistence — the messages of every conversation (one user in
//! one workspace), for GUI display.
//!
//! One conversation keeps its newest [`CHAT_HISTORY_CAP`] messages and nothing
//! older. The cap is a standing state, held by two halves: a message is capped
//! where it is written ([`ChatHistoryStore::insert`], the single write entry
//! every writer goes through — user messages, agent responses and the
//! session-reset divider alike) and [`run_cap_loop`] repairs what a store
//! already holds, including what a copy brought up from an older one holds.
//! What is over the cap is REMOVED from the store, never merely left out of the
//! read; and the read is bounded by that same value, so a conversation the
//! removal has not reached yet still shows the window the cap keeps for it.
//!
//! Messages are written by the central broadcast-and-persist paths:
//! `broadcast_and_persist_incoming_message` for incoming user messages and
//! `broadcast_and_persist_agent_response` for outgoing agent responses.
//!
//! Each message gets a NanoID for deduplication.

use crate::ChatDirection;
use crate::channels::ReplyReference;
use crate::db::{self, Row};
use anyhow::Result;
use std::time::Duration;
use tracing::{debug, warn};

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

/// The number of messages one conversation keeps — and the number the chat
/// window reads. One value for both: the window shows what the cap keeps, so no
/// second bound can drift from the cap and quietly hide what the cap failed to
/// remove.
const CHAT_HISTORY_CAP: usize = 100;

/// How many messages one removal statement may take. The engine rejects
/// `DELETE … LIMIT`, so a removal bounds itself with a `LIMIT`ed id subquery and
/// leaves the rest to the next statement — the write path trims the
/// conversation it just wrote, and the sweep re-trims it on its next round.
const TRIM_BATCH: u64 = 1_000;

/// How often [`run_cap_loop`] repairs every conversation it finds over the cap.
const SWEEP_INTERVAL: Duration = Duration::from_mins(10);

/// Pause between the statements of a sweep that removed something, so a sweep
/// never holds the store across the whole of it.
const SWEEP_PAUSE: Duration = Duration::from_millis(50);

/// The cap's removal: the messages of one conversation that are NOT among its
/// newest [`CHAT_HISTORY_CAP`], oldest first, at most `?4` of them. A
/// conversation at or below the cap matches nothing. One statement, so the
/// boundary it removes to cannot go stale between reading and deleting it.
const TRIM_SQL: &str = "\
DELETE FROM chat_history WHERE id IN (
    SELECT id FROM chat_history
     WHERE user_name = ?1 AND workspace = ?2
       AND id NOT IN (
           SELECT id FROM chat_history
            WHERE user_name = ?1 AND workspace = ?2
            ORDER BY id DESC
            LIMIT ?3
       )
     ORDER BY id
     LIMIT ?4
)";

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

/// Convert a query's rows — newest first, as a conversation read returns them —
/// into entries in the order the chat window shows them (oldest first).
fn rows_to_entries(rows: Vec<Row>) -> Result<Vec<ChatHistoryEntry>> {
    let mut entries: Vec<ChatHistoryEntry> = Vec::with_capacity(rows.len());
    for row in rows {
        entries.push(chat_history_entry_from_row(&row)?);
    }
    entries.reverse();
    Ok(entries)
}

impl ChatHistoryStore {
    /// Insert a message into the history. `message_id` is a NanoID for dedup.
    /// Silently ignores duplicate `message_id` values (`INSERT OR IGNORE`).
    ///
    /// The message just written is what pushes its conversation over the cap, so
    /// the removal that keeps the cap runs here — after the row is written,
    /// never before it.
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
        // The removal's failure is traced where it runs, never silently: the
        // message is written either way, and the next write to this
        // conversation — or the sweep — tries the removal again.
        let _ = self
            .trim_conversation(&entry.user_name, &entry.workspace)
            .await;
        Ok(())
    }

    /// Remove the messages one conversation holds past its newest
    /// [`CHAT_HISTORY_CAP`] — at most [`TRIM_BATCH`] of them, the oldest first.
    /// Returns how many went; a return short of [`TRIM_BATCH`] means the
    /// conversation is within the cap. A failure is traced here, so a caller
    /// only decides what to do next — the conversation stays over the cap until
    /// a later attempt takes the removal up again.
    async fn trim_conversation(&self, user_name: &str, workspace: &str) -> Result<u64> {
        #[expect(clippy::cast_possible_wrap)]
        let (cap, batch) = (CHAT_HISTORY_CAP as i64, TRIM_BATCH as i64);
        let removed = self
            .conn
            .execute(TRIM_SQL, db::params![user_name, workspace, cap, batch])
            .await
            .map_err(|e| {
                warn!(
                    user = %user_name,
                    workspace = %workspace,
                    error = %e,
                    "Chat history cap: could not remove the messages past it — \
                     the conversation stays over it until a later attempt"
                );
                e
            })?;
        Ok(removed)
    }

    /// Read a conversation — one user in one workspace — in the order the chat
    /// window shows it (oldest first). This is the whole of what is kept for it:
    /// the cap leaves nothing older, and [`CHAT_HISTORY_CAP`] bounds the read
    /// because it is the same value that bounds the store. Ordered by row id,
    /// so the composite index `idx_chat_history_user_ws_id (user_name,
    /// workspace, id)` serves the predicate and the ordering directly.
    pub async fn load_conversation(
        &self,
        user_name: &str,
        workspace: &str,
    ) -> Result<Vec<ChatHistoryEntry>> {
        #[expect(clippy::cast_possible_wrap)]
        let cap = CHAT_HISTORY_CAP as i64;
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
                db::params![user_name, workspace, cap],
            )
            .await?;
        rows_to_entries(rows)
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

/// Remove the messages every conversation in `store` holds past the cap — the
/// repair half, for conversations nothing is being written to: records written
/// before the cap, and what a store brought up from an older copy carries.
/// Returns how many were removed; an error means the sweep could not complete,
/// which the next one takes up again. Each removal is one bounded statement, so
/// a conversation over the cap by more than one batch is taken in several, and
/// a removal that fails is traced where it ran and left to the next sweep.
async fn trim_over_cap_conversations(store: &ChatHistoryStore) -> Result<u64> {
    let rows = store
        .conn
        .query(
            "SELECT DISTINCT user_name, workspace FROM chat_history",
            db::params![],
        )
        .await?;
    let mut removed = 0;
    for row in rows {
        let user_name: String = row.get(0)?;
        let workspace: String = row.get(1)?;
        let before = removed;
        loop {
            match store.trim_conversation(&user_name, &workspace).await {
                Ok(n) => {
                    removed += n;
                    if n < TRIM_BATCH {
                        break;
                    }
                }
                // Traced where the statement ran: this sweep gives up on the
                // conversation, and the next one — or the next message written
                // to it — takes the removal up again.
                Err(_) => break,
            }
            // Only a saturated statement is worth a pause: the conversation has
            // more than one batch to lose, and the rest can wait.
            if !crate::shutdown::sleep_or_shutdown_or_drain(SWEEP_PAUSE).await {
                return Ok(removed);
            }
        }
        // The pause between conversations follows work only, so a sweep over
        // conversations already within the cap costs the statements alone.
        if removed > before && !crate::shutdown::sleep_or_shutdown_or_drain(SWEEP_PAUSE).await {
            return Ok(removed);
        }
    }
    Ok(removed)
}

/// Background loop: hold the cap of every conversation the store holds — one
/// sweep per [`SWEEP_INTERVAL`], the first one a whole interval after start so
/// the repair never competes with the store bring-up.
pub async fn run_cap_loop() {
    loop {
        if !crate::shutdown::sleep_or_shutdown_or_drain(SWEEP_INTERVAL).await {
            break;
        }
        match trim_over_cap_conversations(store()).await {
            Ok(0) => debug!("Chat history cap: every conversation is within it"),
            Ok(removed) => debug!(removed, "Chat history cap: removed messages past it"),
            Err(e) => warn!("Chat history cap: the sweep did not complete: {e}"),
        }
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

    async fn insert_message(store: &ChatHistoryStore, user: &str, ws: &str, message_id: &str) {
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

    /// What the store holds for one conversation, as the read hands it over.
    async fn kept(store: &ChatHistoryStore, user: &str, ws: &str) -> Vec<ChatHistoryEntry> {
        store
            .load_conversation(user, ws)
            .await
            .expect("load should succeed")
    }

    async fn contents(store: &ChatHistoryStore, user: &str, ws: &str) -> Vec<String> {
        kept(store, user, ws)
            .await
            .into_iter()
            .map(|entry| entry.content)
            .collect()
    }

    /// How many rows the store holds for one conversation — the stored count,
    /// which the read alone could hide.
    async fn rows(store: &ChatHistoryStore, user: &str, ws: &str) -> usize {
        let stored: i64 = store
            .conn
            .query_row(
                "SELECT COUNT(*) FROM chat_history WHERE user_name = ?1 AND workspace = ?2",
                db::params![user, ws],
                |row| row.get::<i64>(0),
            )
            .await
            .expect("count should succeed");
        usize::try_from(stored).expect("a row count fits in usize")
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
        insert_message(&store, "user", "ws", "msg-1").await;
        let history = kept(&store, "user", "ws").await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content, "msg-1");
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
        let history = kept(&store, "alice", "ws1").await;

        // Should have exactly one entry: the divider.
        assert_eq!(history.len(), 1, "should have exactly one divider entry");

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
        assert!(
            kept(&store, "bob", "ws1").await.is_empty(),
            "divider inserted for alice should not appear in bob's history"
        );

        // Verify the divider is *not* present in another workspace's history.
        assert!(
            kept(&store, "alice", "ws2").await.is_empty(),
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

        let history = kept(&store, "alice", "ws1").await;

        // Both dividers should be present.
        assert_eq!(history.len(), 2, "should have two dividers");
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
        insert_message(&store, "alice", "ws1", "msg-1").await;

        // Insert a divider.
        store
            .insert_divider("alice", "ws1")
            .await
            .expect("insert_divider should succeed");

        // Insert another message after the divider.
        insert_message(&store, "alice", "ws1", "msg-2").await;

        // Load all three.
        let history = kept(&store, "alice", "ws1").await;

        assert_eq!(history.len(), 3, "should have all three entries");

        // Chronological order (oldest first).
        assert_eq!(history[0].direction, ChatDirection::User);
        assert_eq!(history[0].content, "msg-1");

        assert_eq!(history[1].direction, ChatDirection::Divider);

        assert_eq!(history[2].direction, ChatDirection::User);
        assert_eq!(history[2].content, "msg-2");
    }

    /// The cap is held where a message is written: the row that pushes a
    /// conversation over it is the one that takes the removal, and what goes is
    /// the oldest row — a session-reset divider counts toward the cap like any
    /// other row of the conversation, and no other conversation is touched.
    #[tokio::test]
    async fn a_conversation_keeps_only_its_newest_messages() {
        const WS: &str = "personal:alice";
        let (store, _tmp) = test_setup().await;

        for i in 0..CHAT_HISTORY_CAP {
            insert_message(&store, "alice", WS, &format!("msg-{i}")).await;
        }
        // Nothing is over the cap yet, so the last write took nothing.
        assert_eq!(rows(&store, "alice", WS).await, CHAT_HISTORY_CAP);

        // The same user in another workspace, and another user in this one —
        // neither is the conversation being capped.
        insert_message(&store, "alice", "project", "other-workspace").await;
        insert_message(&store, "bob", WS, "other-user").await;

        // The divider is the newest row and counts toward the cap: writing it is
        // what takes the oldest message.
        store
            .insert_divider("alice", WS)
            .await
            .expect("insert_divider should succeed");

        let history = kept(&store, "alice", WS).await;
        assert_eq!(history.len(), CHAT_HISTORY_CAP);
        assert_eq!(history[0].content, "msg-1");
        assert_eq!(history[0].direction, ChatDirection::User);
        assert_eq!(
            history[CHAT_HISTORY_CAP - 1].direction,
            ChatDirection::Divider
        );
        assert_eq!(rows(&store, "alice", WS).await, CHAT_HISTORY_CAP);

        assert_eq!(
            contents(&store, "alice", "project").await,
            ["other-workspace"]
        );
        assert_eq!(contents(&store, "bob", WS).await, ["other-user"]);
    }

    /// Seed `count` rows straight into the table for one conversation — the way
    /// the rows of a store that never had the cap were written, bypassing the
    /// write entry and its removal.
    async fn seed_rows(store: &ChatHistoryStore, ws: &str, count: usize) {
        for i in 0..count {
            store
                .conn
                .execute(
                    "INSERT INTO chat_history (message_id, user_name, direction, content, workspace) \
                     VALUES (?1, 'alice', 'user', ?1, ?2)",
                    db::params![format!("msg-{i}"), ws],
                )
                .await
                .expect("seed should succeed");
        }
    }

    /// The repair half: a store that already holds more than the cap — rows
    /// written before it existed, or carried in by a copy restored from an older
    /// one — is brought within the cap by the sweep alone, with no write needed.
    #[tokio::test]
    async fn the_sweep_removes_what_a_store_already_holds() {
        const WS: &str = "personal:alice";
        const OVER: usize = 5;
        let (store, _tmp) = test_setup().await;

        seed_rows(&store, WS, CHAT_HISTORY_CAP + OVER).await;
        assert_eq!(rows(&store, "alice", WS).await, CHAT_HISTORY_CAP + OVER);
        // The read alone would mask it: the rows are stored and the sweep has
        // not run.
        assert_eq!(contents(&store, "alice", WS).await.len(), CHAT_HISTORY_CAP);

        assert_eq!(
            trim_over_cap_conversations(&store)
                .await
                .expect("the sweep should complete"),
            OVER as u64,
            "exactly the messages past the cap go"
        );
        assert_eq!(rows(&store, "alice", WS).await, CHAT_HISTORY_CAP);
        let history = kept(&store, "alice", WS).await;
        assert_eq!(history[0].content, format!("msg-{OVER}"));
        assert_eq!(
            history[CHAT_HISTORY_CAP - 1].content,
            format!("msg-{}", CHAT_HISTORY_CAP + OVER - 1)
        );

        // A second sweep finds nothing left to repair.
        assert_eq!(
            trim_over_cap_conversations(&store)
                .await
                .expect("the sweep should complete"),
            0
        );
    }

    /// One removal statement takes at most [`TRIM_BATCH`] messages, so a
    /// conversation far past the cap is brought down over several of them: what
    /// one write's own removal leaves, the sweep's rounds carry to completion.
    #[tokio::test]
    async fn a_backlog_past_one_statement_goes_over_several() {
        const WS: &str = "personal:alice";
        const OVER: usize = 1_500;
        let (store, _tmp) = test_setup().await;

        seed_rows(&store, WS, CHAT_HISTORY_CAP + OVER).await;

        // One statement's worth — what a single write to the conversation takes.
        assert_eq!(
            store.trim_conversation("alice", WS).await.unwrap(),
            TRIM_BATCH
        );
        assert_eq!(
            rows(&store, "alice", WS).await,
            CHAT_HISTORY_CAP + OVER - usize::try_from(TRIM_BATCH).unwrap()
        );

        // The sweep keeps taking batches until the conversation is within the
        // cap, whatever the backlog was.
        assert_eq!(
            trim_over_cap_conversations(&store)
                .await
                .expect("the sweep should complete"),
            OVER as u64 - TRIM_BATCH
        );
        assert_eq!(rows(&store, "alice", WS).await, CHAT_HISTORY_CAP);
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

        let history = kept(&store, "alice", "ws1").await;
        assert_eq!(history.len(), 1);
        let reply = history[0]
            .reply_reference
            .as_ref()
            .expect("reply reference should survive the round-trip");
        assert_eq!(reply.author, "bob");
        assert_eq!(reply.snippet, "earlier message");
    }
}
