//! Chat history persistence — the messages of every conversation (one user in one
//! workspace), for GUI display.
//!
//! A conversation keeps its newest [`CHAT_HISTORY_CAP`] messages and nothing older. The cap is
//! held by two halves: a message is capped where it is written
//! ([`ChatHistoryStore::insert`], the single write entry every writer goes through — user
//! messages, agent responses and the session-reset divider alike) and [`run_cap_loop`] repairs
//! what a store already holds, including what a copy from an older one brought up. What is over
//! the cap is REMOVED from the store, never merely left out of the read; the read is bounded by
//! the same value, so a conversation the removal has not reached still shows the same window.
//!
//! Messages are written by the central broadcast-and-persist paths
//! (`broadcast_and_persist_incoming_message` and `broadcast_and_persist_agent_response`). Each
//! gets a NanoID for deduplication.

use crate::ChatDirection;
use crate::channels::ReplyReference;
use crate::db::{self, Row};
use anyhow::Result;
use std::time::Duration;
use tracing::{debug, warn};

crate::define_store! {
    pub(crate) static CHAT_HISTORY: ChatHistoryStore,
    expect = "CHAT_HISTORY not initialized — call init_all_stores() first",
}

/// Parameters for inserting a chat history entry, bundled for [`ChatHistoryStore::insert`].
#[derive(Debug, Clone)]
pub struct ChatHistoryInsert {
    pub message_id: String,
    pub user_name: String,
    pub direction: String,
    pub content: String,
    pub agent_role: Option<String>,
    /// The delivery dispatch that wrote this row where the writer tags one (the agent delivery
    /// paths do); `None` on a user's own message, the voice delivery row, and legacy data.
    pub broadcast_id: Option<String>,
    pub workspace: String,
    pub timestamp: Option<String>,
    /// Optional reference to the replied-to message, rendered as a quote header by the GUI.
    pub reply_reference: Option<ReplyReference>,
}

#[derive(Debug, Clone)]
pub struct ChatHistoryEntry {
    pub id: i64,
    pub message_id: String,
    pub content: String,
    pub direction: ChatDirection,
    pub agent_role: Option<String>,
    pub timestamp: Option<String>,
    pub reply_reference: Option<ReplyReference>,
}

/// The number of messages one conversation keeps — and the number the chat window reads. One value
/// for both, so no second bound can drift from the cap and hide what the cap failed to remove.
const CHAT_HISTORY_CAP: usize = 100;

/// How many messages one removal statement may take. The engine rejects `DELETE … LIMIT`, so a
/// removal bounds itself with a `LIMIT`ed id subquery and leaves the rest to the next statement.
const TRIM_BATCH: u64 = 1_000;

/// How often [`run_cap_loop`] repairs every conversation over the cap.
const SWEEP_INTERVAL: Duration = Duration::from_mins(10);

/// Pause between the statements of a sweep that removed something, so a sweep never holds the
/// store across the whole of it.
const SWEEP_PAUSE: Duration = Duration::from_millis(50);

/// The messages of one conversation NOT among its newest [`CHAT_HISTORY_CAP`], oldest first, at
/// most `?4`. A conversation within the cap matches nothing. One statement, so the boundary it
/// removes to cannot go stale between reading and deleting.
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

// Column definitions for the `chat_history` SELECTs.
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

/// A query's rows — newest first — in the order the chat window shows them.
fn rows_to_entries(rows: Vec<Row>) -> Result<Vec<ChatHistoryEntry>> {
    let mut entries: Vec<ChatHistoryEntry> = Vec::with_capacity(rows.len());
    for row in rows {
        entries.push(chat_history_entry_from_row(&row)?);
    }
    entries.reverse();
    Ok(entries)
}

impl ChatHistoryStore {
    /// `message_id` is a NanoID for dedup; duplicates are ignored (`INSERT OR IGNORE`).
    ///
    /// The row just written is what pushes its conversation over the cap, so the removal runs
    /// after the write, never before it.
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
        // Traced where it runs, never silently: the message is written either way, and the next
        // write to this conversation — or the sweep — retries the removal.
        let _ = self
            .trim_conversation(&entry.user_name, &entry.workspace)
            .await;
        Ok(())
    }

    /// Remove one conversation's messages past its newest [`CHAT_HISTORY_CAP`] — at most
    /// [`TRIM_BATCH`], oldest first. Returns how many went; short of [`TRIM_BATCH`] means it is
    /// within the cap. A failure is traced here, and the conversation stays over the cap until a
    /// later attempt takes the removal up again.
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

    /// One conversation, oldest first: the whole of what is kept for it, since the read is
    /// bounded by the same [`CHAT_HISTORY_CAP`]. Ordered by row id, so the composite index
    /// `idx_chat_history_user_ws_id` serves predicate and ordering directly.
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

    /// A session-clear marker, stored with `direction='divider'` so the GUI renders a separator
    /// rather than a bubble.
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

/// The repair half, for conversations nothing is being written to: records written before the
/// cap, and what a store brought up from an older copy carries. Returns how many were removed;
/// an error means the sweep could not complete, which the next one takes up. Each removal is one
/// bounded statement, so a bigger backlog is taken in several.
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
                // Traced where the statement ran: this sweep gives up on the conversation and
                // the next one — or the next message written to it — tries again.
                Err(_) => break,
            }
            // Only a saturated statement is worth a pause: the conversation has more than one
            // batch to lose.
            if !crate::shutdown::sleep_or_shutdown_or_drain(SWEEP_PAUSE).await {
                return Ok(removed);
            }
        }
        // The pause follows work only, so a sweep over conversations within the cap costs the
        // statements alone.
        if removed > before && !crate::shutdown::sleep_or_shutdown_or_drain(SWEEP_PAUSE).await {
            return Ok(removed);
        }
    }
    Ok(removed)
}

/// One sweep per [`SWEEP_INTERVAL`], the first a whole interval after start so the repair never
/// competes with store bring-up.
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

    /// The stored row count, which the read alone could hide.
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

    /// A divider is its own entry: a timestamp in its content, no agent role, its own
    /// conversation and workspace only — and the timeline reads back in write order, dividers
    /// between messages included.
    #[tokio::test]
    async fn test_dividers_are_stored_ordered_and_scoped() {
        let (store, _tmp) = test_setup().await;

        // A single divider: the row handed back is a divider, not a message, and it speaks only
        // to the conversation that wrote it.
        store
            .insert_divider("alice", "ws1")
            .await
            .expect("insert_divider should succeed");
        let history = kept(&store, "alice", "ws1").await;
        assert_eq!(history.len(), 1, "should have exactly one divider entry");
        let entry = &history[0];
        assert_eq!(
            entry.direction,
            ChatDirection::Divider,
            "divider entry should have ChatDirection::Divider"
        );
        assert!(
            entry.agent_role.is_none(),
            "divider should have no agent_role"
        );
        assert!(
            !entry.content.is_empty(),
            "divider content (timestamp) should not be empty"
        );
        assert!(
            entry.content.contains('T'),
            "divider content should be an ISO 8601 timestamp, got: {}",
            entry.content
        );
        assert!(
            kept(&store, "bob", "ws1").await.is_empty(),
            "divider inserted for alice should not appear in bob's history"
        );
        assert!(
            kept(&store, "alice", "ws2").await.is_empty(),
            "divider inserted for ws1 should not appear in ws2's history"
        );

        // Two dividers are two entries, oldest first.
        store
            .insert_divider("alice", "ws1")
            .await
            .expect("second divider should succeed");
        let history = kept(&store, "alice", "ws1").await;
        assert_eq!(history.len(), 2, "should have two dividers");
        assert_eq!(history[0].direction, ChatDirection::Divider);
        assert_eq!(history[1].direction, ChatDirection::Divider);
        assert!(
            history[0].id < history[1].id,
            "first inserted divider should have a lower id"
        );

        // A divider between two messages keeps its place in the timeline.
        let (store, _tmp) = test_setup().await;
        insert_message(&store, "alice", "ws1", "msg-1").await;
        store
            .insert_divider("alice", "ws1")
            .await
            .expect("insert_divider should succeed");
        insert_message(&store, "alice", "ws1", "msg-2").await;
        let history = kept(&store, "alice", "ws1").await;
        assert_eq!(history.len(), 3, "should have all three entries");
        assert_eq!(history[0].direction, ChatDirection::User);
        assert_eq!(history[0].content, "msg-1");
        assert_eq!(history[1].direction, ChatDirection::Divider);
        assert_eq!(history[2].direction, ChatDirection::User);
        assert_eq!(history[2].content, "msg-2");
    }

    /// Held where a message is written: the row that pushes a conversation over the cap takes
    /// the removal, and the oldest row goes — a divider counts like any other row, and no other
    /// conversation is touched.
    #[tokio::test]
    async fn a_conversation_keeps_only_its_newest_messages() {
        const WS: &str = "personal:alice";
        let (store, _tmp) = test_setup().await;

        for i in 0..CHAT_HISTORY_CAP {
            insert_message(&store, "alice", WS, &format!("msg-{i}")).await;
        }
        // Nothing is over the cap yet, so the last write took nothing.
        assert_eq!(rows(&store, "alice", WS).await, CHAT_HISTORY_CAP);

        // The same user in another workspace, and another user in this one — neither is the
        // conversation being capped.
        insert_message(&store, "alice", "project", "other-workspace").await;
        insert_message(&store, "bob", WS, "other-user").await;

        // The divider is the newest row and counts: writing it takes the oldest message.
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

    /// Seed `count` rows straight into the table, the way a store that never had the cap was
    /// written — bypassing the write entry and its removal.
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

    /// The repair half: a store already over the cap — rows written before it existed, or
    /// carried in by an older copy — is brought within it by the sweep alone, with no write.
    #[tokio::test]
    async fn the_sweep_removes_what_a_store_already_holds() {
        const WS: &str = "personal:alice";
        const OVER: usize = 5;
        let (store, _tmp) = test_setup().await;

        seed_rows(&store, WS, CHAT_HISTORY_CAP + OVER).await;
        assert_eq!(rows(&store, "alice", WS).await, CHAT_HISTORY_CAP + OVER);
        // The read alone would mask it: the rows are stored and the sweep has not run.
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

    /// One statement takes at most [`TRIM_BATCH`], so a conversation far past the cap comes down
    /// over several: what one write's own removal leaves, the sweep's rounds carry to completion.
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
