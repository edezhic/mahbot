//! Chat history persistence — stores all chat messages (both user and agent)
//! for GUI display and history loading. Messages are written at the point of
//! delivery: incoming user messages from the GUI send path, outgoing agent
//! responses from `GuiChannel::send()`.
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
/// This struct bundles the 8 fields needed by [`ChatHistoryStore::insert`].
/// Owned `String` fields match the pattern established by
/// [`LogEntry`](crate::logs::LogEntry).
#[derive(Debug, Clone)]
pub struct ChatHistoryInsert {
    pub message_id: String,
    pub user_name: String,
    pub direction: String,
    pub content: String,
    pub agent_role: Option<String>,
    /// Shared id tagging all per-user copies of one broadcast dispatch;
    /// `None` for single-user rows and legacy data. Enables exact dedup when
    /// reading across user partitions.
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

/// One row of the workspace-wide chat stream for the manager-chat read tool.
pub(crate) struct WorkspaceChatRow {
    pub user_name: String,
    pub direction: ChatDirection,
    pub agent_role: Option<String>,
    pub content: String,
}

/// Maximum number of history entries to load at once.
const HISTORY_LIMIT: usize = 100;

/// Bounds pathological all-duplicate tables: the fetch window stops growing here.
const MAX_FETCH_LIMIT: usize = 4096;

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
    /// Silently ignores duplicate `message_id` values (UPSERT no-op).
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

    /// Shared paging body for the `load_*` methods. `ws2` is an optional
    /// second workspace to merge; `None` loads a single workspace only (and
    /// keeps the composite index `idx_chat_history_user_ws_id` usable).
    /// `before_id = None` loads the most recent page; `Some(id)` loads only
    /// messages older than `id`.
    async fn load_page(
        &self,
        user_name: &str,
        ws1: &str,
        ws2: Option<&str>,
        before_id: Option<i64>,
    ) -> Result<(Vec<ChatHistoryEntry>, bool)> {
        #[expect(clippy::cast_possible_wrap)]
        let query_limit = HISTORY_LIMIT as i64 + 1; // fetch one extra to detect has_more
        let rows = match ws2 {
            // Two-workspace merge (selected + personal): OR predicate, merged
            // chronologically by global row id.
            Some(ws2) => {
                self.conn
                    .query(
                        &format!(
                            "SELECT {CHAT_HISTORY_COLUMNS} \
                             FROM chat_history \
                             WHERE user_name = ?1 AND (workspace = ?2 OR workspace = ?3) \
                               AND (?4 IS NULL OR id < ?4) \
                             ORDER BY id DESC \
                             LIMIT ?5",
                        ),
                        db::params![user_name, ws1, ws2, before_id, query_limit],
                    )
                    .await?
            }
            // Single-workspace: plain equality keeps the composite
            // idx_chat_history_user_ws_id directly usable (the OR form would
            // need a broader scan).
            None => {
                self.conn
                    .query(
                        &format!(
                            "SELECT {CHAT_HISTORY_COLUMNS} \
                             FROM chat_history \
                             WHERE user_name = ?1 AND workspace = ?2 \
                               AND (?3 IS NULL OR id < ?3) \
                             ORDER BY id DESC \
                             LIMIT ?4",
                        ),
                        db::params![user_name, ws1, before_id, query_limit],
                    )
                    .await?
            }
        };
        rows_to_page(rows)
    }

    /// Load the most recent messages for a user across one or two workspaces
    /// (`ws2` is an optional second workspace to merge). Returns entries in
    /// chronological order (oldest first) with a `has_more` flag for older
    /// entries beyond the loaded window.
    pub async fn load_for_user_workspaces(
        &self,
        user_name: &str,
        ws1: &str,
        ws2: Option<&str>,
    ) -> Result<(Vec<ChatHistoryEntry>, bool)> {
        self.load_page(user_name, ws1, ws2, None).await
    }

    /// Load messages older than `before_id` for a user across one or two
    /// workspaces. Returns `(entries, has_more)` where `has_more` is `true`
    /// if even older entries exist beyond the loaded window, in chronological
    /// order (oldest first).
    pub async fn load_older_for_user_workspaces(
        &self,
        user_name: &str,
        ws1: &str,
        ws2: Option<&str>,
        before_id: i64,
    ) -> Result<(Vec<ChatHistoryEntry>, bool)> {
        self.load_page(user_name, ws1, ws2, Some(before_id)).await
    }

    /// Most recent `limit` unique messages across ALL user partitions of one
    /// workspace, chronological. Per-user copies of one broadcast dispatch share
    /// a `broadcast_id` (fresh NanoID per copy otherwise), so agent-direction rows
    /// are deduped exactly by `broadcast_id`; legacy rows (NULL, pre-delta-33)
    /// fall back to (agent_role, content) — best-effort only for historic data.
    /// The query excludes divider rows so they cannot crowd the window — every
    /// fetched row is a real message.
    ///
    /// The fetch window grows on shortfall: it starts at `limit * 4` (one query
    /// covers the common ≤3-copier fan-out) and doubles until it yields at least
    /// `limit` unique messages or the table window is exhausted. That makes the
    /// "at least `limit` logical messages" guarantee structural, independent of
    /// any user-count estimate. The guarantee is exact for rows carrying a
    /// `broadcast_id`; legacy pre-delta-33 rows (NULL broadcast_id) dedup
    /// best-effort on `(agent_role, content)`, so a window dense with
    /// identical-content legacy agent rows may return fewer than `limit` messages.
    /// [`MAX_FETCH_LIMIT`] caps the growth for pathological all-duplicate tables.
    #[expect(clippy::cast_possible_wrap)]
    pub(crate) async fn load_workspace_stream(
        &self,
        workspace: &str,
        limit: usize,
    ) -> Result<Vec<WorkspaceChatRow>> {
        let mut fetch_limit = limit.saturating_mul(4);
        loop {
            // Column order below: user_name(0), direction(1), agent_role(2),
            // content(3), broadcast_id(4).
            let rows = self
                .conn
                .query(
                    "SELECT user_name, direction, agent_role, content, broadcast_id \
                     FROM chat_history WHERE workspace = ?1 AND direction <> 'divider' \
                     ORDER BY id DESC LIMIT ?2",
                    db::params![workspace, fetch_limit as i64],
                )
                .await?;
            let mut out: Vec<WorkspaceChatRow> = Vec::new();
            let mut seen_broadcast: std::collections::HashSet<String> =
                std::collections::HashSet::default();
            let mut seen_legacy_agent: std::collections::HashSet<(Option<String>, String)> =
                std::collections::HashSet::default();
            for row in &rows {
                let direction = match row.get::<String>(1)?.as_str() {
                    "agent" => ChatDirection::Agent,
                    _ => ChatDirection::User,
                };
                let agent_role = row.get::<Option<String>>(2)?;
                let content = row.get::<String>(3)?;
                let broadcast_id = row.get::<Option<String>>(4)?;
                if direction == ChatDirection::Agent {
                    let is_duplicate = match broadcast_id {
                        Some(id) => !seen_broadcast.insert(id),
                        None => !seen_legacy_agent.insert((agent_role.clone(), content.clone())),
                    };
                    if is_duplicate {
                        continue;
                    }
                }
                out.push(WorkspaceChatRow {
                    user_name: row.get::<String>(0)?,
                    direction,
                    agent_role,
                    content,
                });
            }
            if out.len() >= limit {
                out.truncate(limit);
                out.reverse();
                return Ok(out);
            }
            if rows.len() < fetch_limit || fetch_limit >= MAX_FETCH_LIMIT {
                out.reverse();
                return Ok(out);
            }
            fetch_limit = fetch_limit.saturating_mul(2).min(MAX_FETCH_LIMIT);
        }
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
            .load_for_user_workspaces("user", "ws", None)
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
            .load_for_user_workspaces("alice", "ws1", None)
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
            .load_for_user_workspaces("bob", "ws1", None)
            .await
            .expect("other user load should succeed");
        assert!(
            other_history.is_empty(),
            "divider inserted for alice should not appear in bob's history"
        );

        // Verify the divider is *not* present in another workspace's history.
        let (other_ws_history, _) = store
            .load_for_user_workspaces("alice", "ws2", None)
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
            .load_for_user_workspaces("alice", "ws1", None)
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
            .load_for_user_workspaces("alice", "ws1", None)
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
    async fn test_load_across_workspaces_merges_chronologically() {
        let (store, _tmp) = test_setup().await;

        for (i, (content, ws)) in [
            ("p1", "project"),
            ("me1", "personal:alice"),
            ("p2", "project"),
            ("me2", "personal:alice"),
        ]
        .into_iter()
        .enumerate()
        {
            store
                .insert(&ChatHistoryInsert {
                    message_id: format!("msg-{i}"),
                    user_name: "alice".to_string(),
                    direction: "user".to_string(),
                    content: content.to_string(),
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
            .load_for_user_workspaces("alice", "project", Some("personal:alice"))
            .await
            .expect("load should succeed");
        assert_eq!(history.len(), 4, "all four entries from both workspaces");
        assert!(!has_more);
        // Chronological order by id across workspaces.
        let contents: Vec<&str> = history.iter().map(|e| e.content.as_str()).collect();
        assert_eq!(contents, ["p1", "me1", "p2", "me2"]);

        // A single-workspace load still filters correctly via the same API.
        let (only_personal, _) = store
            .load_for_user_workspaces("alice", "personal:alice", None)
            .await
            .expect("load should succeed");
        assert_eq!(only_personal.len(), 2);
        assert_eq!(only_personal[0].content, "me1");
        assert_eq!(only_personal[1].content, "me2");
    }

    #[tokio::test]
    async fn test_load_older_across_workspaces_paginates() {
        let (store, _tmp) = test_setup().await;

        // Two workspaces, 3 messages each, inserted alternately.
        for i in 0..6 {
            let ws = if i % 2 == 0 {
                "project"
            } else {
                "personal:alice"
            };
            store
                .insert(&ChatHistoryInsert {
                    message_id: format!("msg-{i}"),
                    user_name: "alice".to_string(),
                    direction: "user".to_string(),
                    content: format!("c{i}"),
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
            .load_for_user_workspaces("alice", "project", Some("personal:alice"))
            .await
            .expect("load should succeed");
        assert_eq!(history.len(), 6);
        assert!(!has_more);
        let oldest_id = history[0].id;

        // Load older than the oldest id — nothing left in either workspace.
        let (older, has_more_older) = store
            .load_older_for_user_workspaces("alice", "project", Some("personal:alice"), oldest_id)
            .await
            .expect("load older should succeed");
        assert!(older.is_empty());
        assert!(!has_more_older);

        // Loading older than the 3rd message returns the 3 older ones, merged.
        let (older, _) = store
            .load_older_for_user_workspaces(
                "alice",
                "project",
                Some("personal:alice"),
                history[3].id,
            )
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
            .load_for_user_workspaces("alice", "ws1", None)
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

    /// Insert a chat_history row directly (fills every field), so tests can
    /// control `broadcast_id` exactly.
    async fn insert_row(
        store: &ChatHistoryStore,
        message_id: &str,
        user_name: &str,
        direction: &str,
        content: &str,
        agent_role: Option<&str>,
        broadcast_id: Option<&str>,
    ) {
        store
            .insert(&ChatHistoryInsert {
                message_id: message_id.to_string(),
                user_name: user_name.to_string(),
                direction: direction.to_string(),
                content: content.to_string(),
                agent_role: agent_role.map(str::to_string),
                broadcast_id: broadcast_id.map(str::to_string),
                workspace: "ws".to_string(),
                timestamp: None,
                reply_reference: None,
            })
            .await
            .expect("insert should succeed");
    }

    #[tokio::test]
    async fn workspace_stream_dedups_manager_copies_by_broadcast_id() {
        let (store, _tmp) = test_setup().await;
        insert_row(&store, "u1-msg", "u1", "user", "hello", None, None).await;
        insert_row(
            &store,
            "mgr-u1",
            "u1",
            "agent",
            "manager reply",
            Some("manager"),
            Some("b1"),
        )
        .await;
        insert_row(
            &store,
            "mgr-u2",
            "u2",
            "agent",
            "manager reply",
            Some("manager"),
            Some("b1"),
        )
        .await;

        let rows = store.load_workspace_stream("ws", 5).await.expect("load");
        // Exactly 2 unique logical messages, chronological: the user row, then
        // ONE manager copy (u1's copy shares b1 and is deduped away).
        assert_eq!(rows.len(), 2, "user row + one manager copy");
        assert_eq!(rows[0].user_name, "u1");
        assert_eq!(rows[0].direction, ChatDirection::User);
        assert_eq!(rows[0].content, "hello");
        assert_eq!(rows[1].user_name, "u2");
        assert_eq!(rows[1].direction, ChatDirection::Agent);
        assert_eq!(rows[1].agent_role.as_deref(), Some("manager"));
        assert_eq!(rows[1].content, "manager reply");
    }

    #[tokio::test]
    async fn workspace_stream_growing_window_recovers_from_copies() {
        let (store, _tmp) = test_setup().await;
        // Three unique messages, oldest first: two user rows + one manager reply.
        insert_row(&store, "u1", "u1", "user", "msg-a", None, None).await;
        insert_row(&store, "u2", "u2", "user", "msg-b", None, None).await;
        insert_row(
            &store,
            "resp",
            "u3",
            "agent",
            "resp",
            Some("manager"),
            Some("b1"),
        )
        .await;
        // The newest rows are a flood of duplicated manager copies sharing b1.
        // A naive fixed window (e.g. limit*2, or even the limit*4 start) would
        // fetch only copies and under-return; the loader doubles its window
        // until the unique messages emerge.
        for i in 0..14 {
            insert_row(
                &store,
                &format!("dup-{i}"),
                "u3",
                "agent",
                "resp",
                Some("manager"),
                Some("b1"),
            )
            .await;
        }

        let rows = store.load_workspace_stream("ws", 3).await.expect("load");
        assert_eq!(
            rows.len(),
            3,
            "growing window must recover the full set of unique messages"
        );
        assert_eq!(rows[0].content, "msg-a");
        assert_eq!(rows[1].content, "msg-b");
        assert_eq!(rows[2].content, "resp");
    }

    #[tokio::test]
    async fn workspace_stream_keeps_distinct_identical_broadcasts() {
        let (store, _tmp) = test_setup().await;
        insert_row(&store, "u1", "u1", "user", "hi", None, None).await;
        insert_row(
            &store,
            "m1",
            "u1",
            "agent",
            "same",
            Some("manager"),
            Some("b1"),
        )
        .await;
        insert_row(
            &store,
            "m2",
            "u2",
            "agent",
            "same",
            Some("manager"),
            Some("b2"),
        )
        .await;

        let rows = store.load_workspace_stream("ws", 5).await.expect("load");
        assert_eq!(rows.len(), 3, "distinct broadcast ids must both survive");
        let agent_count = rows
            .iter()
            .filter(|r| r.direction == ChatDirection::Agent)
            .count();
        assert_eq!(
            agent_count, 2,
            "both identical-content broadcasts are kept (no content-key collision)"
        );
    }

    #[tokio::test]
    async fn workspace_stream_legacy_rows_fall_back_to_content_dedup() {
        let (store, _tmp) = test_setup().await;
        insert_row(&store, "u1", "u1", "user", "hi", None, None).await;
        insert_row(
            &store,
            "legacy-u1",
            "u1",
            "agent",
            "resp",
            Some("manager"),
            None,
        )
        .await;
        insert_row(
            &store,
            "legacy-u2",
            "u2",
            "agent",
            "resp",
            Some("manager"),
            None,
        )
        .await;

        let rows = store.load_workspace_stream("ws", 5).await.expect("load");
        assert_eq!(
            rows.len(),
            2,
            "legacy NULL-broadcast agent rows dedup to one via (agent_role, content)"
        );
        assert!(rows.iter().any(|r| r.content == "hi"));
        assert!(rows.iter().any(|r| r.content == "resp"));
    }

    #[tokio::test]
    async fn workspace_stream_excludes_dividers() {
        let (store, _tmp) = test_setup().await;
        insert_row(&store, "one", "u1", "user", "one", None, None).await;
        store
            .insert_divider("u1", "ws")
            .await
            .expect("insert_divider should succeed");
        insert_row(&store, "two", "u1", "user", "two", None, None).await;

        let rows = store.load_workspace_stream("ws", 5).await.expect("load");
        assert_eq!(rows.len(), 2, "divider rows are excluded from the stream");
        assert!(rows.iter().all(|r| r.direction != ChatDirection::Divider));
        assert_eq!(rows[0].content, "one");
        assert_eq!(rows[1].content, "two");
    }
}
