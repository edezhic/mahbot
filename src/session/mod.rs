//! Session persistence — Turso-backed store + native history decoding.

pub mod manager;
pub use manager::Session;

use crate::turso::{self, IntoParams, Row, TxGuard, Value, params};
use crate::{ChatMessage, ChatRole, Reasoning, ToolCall, ToolResultPayload};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};

// ── Summarization constants ──────────────────────────────────
//
// The summarization LLM call lives in `crate::Agent::summarize` so that all
// parameters (model, temperature, reasoning_effort, tools, provider routing)
// are byte-identical to the agent's work loop.  This section keeps only the
// constants and helpers used by `Session::apply_summary`.

/// History-length threshold (in estimated tokens) that triggers summarization.
///
/// This is a conservative default chosen to work across models with varying
/// context window sizes (128K–1M).  The value of **100,000** estimated tokens
/// translates to roughly 400K characters of message content under the rough
/// `estimate_tokens` formula (~4 chars/token + 4 tokens per-message overhead).
///
/// ## Why 100K?
///
/// The actual token consumption at request time is higher than `estimate_tokens`
/// suggests for several reasons:
///
/// * **Tokenization ratio** — Code- and JSON-heavy agent conversations (tool
///   calls, structured outputs) can tokenize at ~2.5 chars/token rather than
///   the estimate's 4 chars/token, making the real token count ~1.6× higher.
/// * **Tool schemas** — The tool definitions injected by `build_chat_request`
///   consume ~10–20K actual tokens that are **not** counted by `estimate_tokens`
///   (they live in the `tools` field of the request, not in `messages`).
/// * **System prompt overhead** — The role instruction + workspace context +
///   ticket context are part of `history` and *are* counted, but for large
///   workspaces they add non-trivial context consumption.
/// * **Intra-turn growth** — After summarization the agent loop can add several
///   more tool-call rounds (each adding assistant + tool-result messages) before
///   the next threshold check at the start of the following turn.
///
pub(crate) const SUMMARIZATION_THRESHOLD: usize = 100_000;

/// Stored session rows and second `history` entry after compaction use this prefix so channel
/// orchestration can re-inject the summary on later turns (baseline `system` rows stay excluded).
pub(crate) const PREVIOUS_CONVERSATION_SUMMARY_PREFIX: &str = "Previous conversation summary:\n\n";

/// Rough token count for history (~4 chars/token + 4 tokens per-message overhead)
#[must_use]
pub(crate) fn estimate_tokens(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| m.content.len().div_ceil(4) + 4)
        .sum()
}

crate::define_store! {
    /// Global session store.
    pub(crate) static SESSIONS: SessionStore,
    db_name = "sessions",
    schema = SCHEMA,
    post_open = after_open,
    expect = "SESSIONS not initialized",
}

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_metadata (
    agent_id      TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    last_activity TEXT NOT NULL
);";

// ── Column index constants ──────────────────────────────────

// Session messages (2-column SELECT: role, content)
crate::columns! {
    SESSION_MESSAGE_COLUMNS [SM] {
        ROLE    => "role",
        CONTENT => "content",
    }
}

// Session list with metadata (3-column SELECT: sm.agent_id, sm.last_activity,
// COUNT(s.id))
crate::columns! {
    SESSION_LIST_COLUMNS [SL] {
        AGENT_ID      => "sm.agent_id",
        LAST_ACTIVITY  => "sm.last_activity",
        MESSAGE_COUNT  => "COUNT(s.id)",
    }
}

/// Agent ID prefixes for transient (background-only, non-user-facing) agent sessions.
///
/// These agents are created automatically (analysts, engineers, maintainer,
/// discovery, etc.) and their sessions are cleaned up periodically by
/// [`cleanup_old_transient_sessions`].
///
/// User-facing agents — those the user can directly converse with — persist
/// indefinitely and are intentionally excluded:
/// - Direct chat: `{channel}_{user_name}_{ws_name}_{role}`
/// - Manager: `manager_{ws_name}` — the Manager session carries both chat conversation
///   and notification context and must never be added here.
///
/// If a new agent role is added that can talk to users directly, its agent ID prefix
/// must also be excluded from this list.
pub(crate) const TRANSIENT_AGENT_ID_PREFIXES: &[&str] =
    &["ticket_", "ask_", "maintainer_", "discovery_"];

#[derive(Debug, Clone)]
pub(crate) struct SessionMetadata {
    pub agent_id: String,
    pub last_activity: DateTime<Utc>,
    pub message_count: usize,
}

/// Parse an RFC 3339 timestamp string, falling back to `Utc::now()` on failure.
///
/// Logs a warning with the field name, the raw value, and the parse error
/// when falling back.
#[must_use]
fn parse_ts_or_now(s: &str, label: &str) -> DateTime<Utc> {
    turso::parse_utc_timestamp(s).unwrap_or_else(|e| {
        tracing::warn!(
            field = %label,
            value = %s,
            error = %e,
            "Failed to parse timestamp {label}, falling back to Utc::now()",
        );
        Utc::now()
    })
}

fn session_metadata_from_row(agent_id: &str, activity_str: &str, count: i64) -> SessionMetadata {
    SessionMetadata {
        agent_id: agent_id.to_string(),
        last_activity: parse_ts_or_now(activity_str, "last_activity"),
        message_count: usize::try_from(count).unwrap_or(0),
    }
}

/// Insert messages into `sessions` and upsert `session_metadata` within an existing transaction.
/// Shared helper used by [`SessionStore::append_messages`].
async fn insert_messages_in_transaction(
    tx: &TxGuard<'_>,
    agent_id: &str,
    messages: &[ChatMessage],
) -> Result<()> {
    let now = turso::now();
    for msg in messages {
        tx.execute(
            "INSERT INTO sessions (agent_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![
                agent_id,
                msg.role.to_string(),
                msg.content.clone(),
                now.clone()
            ],
        )
        .await?;
    }
    tx.execute(
        "INSERT INTO session_metadata (agent_id, created_at, last_activity) \
         VALUES (?1, ?2, ?3) \
         ON CONFLICT(agent_id) DO UPDATE SET \
         last_activity = excluded.last_activity",
        params![agent_id, now.clone(), now],
    )
    .await?;
    Ok(())
}

/// Execute a `query_map`, logging warnings on failure and skipping unparseable rows.
/// Returns an empty [`Vec`] on query error.
///
/// `agent_id` is passed as a structured tracing field; when `None`, tracing
/// automatically suppresses it from the output.
async fn query_map_collect<T, E>(
    conn: &turso::Connection,
    sql: &str,
    params: impl IntoParams + Send + 'static,
    row_parser: impl FnMut(&Row) -> std::result::Result<T, E> + Send + 'static,
    warn_context: &str,
    agent_id: Option<&str>,
) -> Vec<T>
where
    T: Send + 'static,
    E: std::fmt::Display + Send + Sync + 'static,
{
    let rows = match conn.query_map(sql, params, row_parser).await {
        Ok(rows) => rows,
        Err(e) => {
            tracing::warn!(error = %e, agent_id, "{warn_context}: query failed, returning empty");
            return Vec::new();
        }
    };
    rows.into_iter()
        .filter_map(|r| match r {
            Ok(val) => Some(val),
            Err(e) => {
                tracing::warn!(error = %e, agent_id, "{warn_context}: row decode failed, skipping");
                None
            }
        })
        .collect()
}

// ── Methods — callable on the static ──────────────────────────

impl SessionStore {
    pub(crate) async fn load(&self, agent_id: &str) -> Vec<ChatMessage> {
        query_map_collect(
            &self.conn,
            &format!(
                "SELECT {SESSION_MESSAGE_COLUMNS} FROM sessions WHERE agent_id = ?1 ORDER BY id ASC"
            ),
            params![agent_id],
            |row| {
                Ok::<_, anyhow::Error>(ChatMessage {
                    role: row
                        .get::<String>(COL_SM_ROLE)?
                        .parse::<ChatRole>()
                        .map_err(|e| anyhow!(e))?,
                    content: row.get(COL_SM_CONTENT)?,
                })
            },
            "load session",
            Some(agent_id),
        )
        .await
    }

    pub(crate) async fn append(&self, agent_id: &str, message: &ChatMessage) -> Result<()> {
        self.batch_append(agent_id, std::slice::from_ref(message))
            .await
    }

    async fn append_messages(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
        replace: bool,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        if replace {
            tx.execute(
                "DELETE FROM sessions WHERE agent_id = ?1",
                params![agent_id],
            )
            .await?;
        }
        insert_messages_in_transaction(&tx, agent_id, messages).await?;
        tx.commit().await?;
        Ok(())
    }

    pub(crate) async fn batch_append(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
    ) -> Result<()> {
        self.append_messages(agent_id, messages, false).await
    }

    pub(crate) async fn replace_messages(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
    ) -> Result<()> {
        self.append_messages(agent_id, messages, true).await
    }

    pub(crate) async fn delete(&self, agent_id: &str) -> Result<bool> {
        let tx = self.conn.begin_tx().await?;
        let deleted = tx
            .execute(
                "DELETE FROM sessions WHERE agent_id = ?1",
                params![agent_id],
            )
            .await?;
        tx.execute(
            "DELETE FROM session_metadata WHERE agent_id = ?1",
            params![agent_id],
        )
        .await?;
        tx.commit().await?;
        Ok(deleted > 0)
    }

    pub(crate) async fn list_sessions_with_metadata(&self) -> Vec<SessionMetadata> {
        query_map_collect(
            &self.conn,
            &format!(
                "SELECT {SESSION_LIST_COLUMNS} \
                 FROM session_metadata sm \
                 LEFT JOIN sessions s ON s.agent_id = sm.agent_id \
                 GROUP BY sm.agent_id \
                 ORDER BY sm.last_activity DESC",
            ),
            (),
            |row| {
                Ok::<_, anyhow::Error>(session_metadata_from_row(
                    &row.get::<String>(COL_SL_AGENT_ID)?,
                    &row.get::<String>(COL_SL_LAST_ACTIVITY)?,
                    row.get::<i64>(COL_SL_MESSAGE_COUNT)?,
                ))
            },
            "list sessions",
            None,
        )
        .await
    }
}

// ── Schema migration (rename session_key to agent_id) ─────────────

impl SessionStore {
    /// Post-open setup: run schema migrations, then ensure indexes.
    async fn after_open(&self) -> anyhow::Result<()> {
        run_session_migrations(&self.conn).await?;
        // Index must be created AFTER migration so the column name matches.
        self.conn
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_sessions_agent_id \
                 ON sessions(agent_id, id);",
            )
            .await
            .context("Failed to create sessions index")?;
        Ok(())
    }
}

/// Run schema migrations for the `sessions` and `session_metadata` tables.
///
/// Uses `PRAGMA user_version` for versioning (following the board.rs pattern).
/// Migration v1: rename `session_key` column to `agent_id` in both tables.
async fn run_session_migrations(conn: &turso::Connection) -> anyhow::Result<()> {
    let version_rows = conn
        .query("PRAGMA user_version", ())
        .await
        .context("Failed to read PRAGMA user_version for schema migration")?;
    let current_version: i64 = version_rows
        .first()
        .and_then(|row| row.get::<i64>(0).ok())
        .unwrap_or(0);

    if current_version < 1 {
        // Check whether the old `session_key` column still exists in sessions table.
        let table_info = conn
            .query("PRAGMA table_info('sessions')", ())
            .await
            .context("Failed to read PRAGMA table_info for sessions table")?;

        let has_session_key = table_info
            .iter()
            .any(|row| row.get::<String>(1).ok().as_deref() == Some("session_key"));

        if has_session_key {
            tracing::info!("Schema migration: renaming sessions.session_key to sessions.agent_id");
            conn.execute(
                "ALTER TABLE sessions RENAME COLUMN session_key TO agent_id",
                (),
            )
            .await
            .context(
                "Schema migration failed: unable to rename sessions.session_key to sessions.agent_id",
            )?;
        }

        // Same for session_metadata table.
        let meta_table_info = conn
            .query("PRAGMA table_info('session_metadata')", ())
            .await
            .context("Failed to read PRAGMA table_info for session_metadata table")?;

        let meta_has_session_key = meta_table_info
            .iter()
            .any(|row| row.get::<String>(1).ok().as_deref() == Some("session_key"));

        if meta_has_session_key {
            tracing::info!(
                "Schema migration: renaming session_metadata.session_key to session_metadata.agent_id"
            );
            conn.execute(
                "ALTER TABLE session_metadata RENAME COLUMN session_key TO agent_id",
                (),
            )
            .await
            .context(
                "Schema migration failed: unable to rename session_metadata.session_key to session_metadata.agent_id",
            )?;
        }

        // PRAGMA user_version is NOT transaction-atomic in SQLite — set it
        // after the ALTER TABLE (which has already auto-committed).
        conn.execute("PRAGMA user_version = 1", ())
            .await
            .context("Schema migration failed: unable to set PRAGMA user_version to 1")?;

        conn.checkpoint().await.context(
            "Schema migration failed: unable to checkpoint after renaming session_key columns",
        )?;

        if has_session_key || meta_has_session_key {
            tracing::info!(
                "Schema migration complete: renamed session_key to agent_id (version 1)"
            );
        }
    }

    Ok(())
}

/// Delete all transient (background-only) sessions whose `last_activity` is older than
/// the given RFC 3339 `cutoff`. Returns the number of deleted session metadata rows.
///
/// Transient agent IDs start with the prefixes listed in
/// `TRANSIENT_AGENT_ID_PREFIXES`.
///
/// Both `sessions` and `session_metadata` tables are cleaned up in a single transaction.
pub async fn cleanup_old_transient_sessions(cutoff: &str) -> Result<u64> {
    let session_store = store();
    let tx = session_store.conn.begin_tx().await?;

    let likes = TRANSIENT_AGENT_ID_PREFIXES
        .iter()
        .map(|_| "agent_id LIKE ?")
        .collect::<Vec<_>>()
        .join(" OR ");
    let prefix_patterns = format!("({likes})");

    let build_params = {
        let mut p = vec![Value::Text(cutoff.to_string())];
        p.extend(
            TRANSIENT_AGENT_ID_PREFIXES
                .iter()
                .map(|prefix| Value::Text(format!("{prefix}%"))),
        );
        p
    };

    // Delete session messages for matching transient sessions
    tx.execute(
        &format!(
            "DELETE FROM sessions WHERE agent_id IN ( \
             SELECT agent_id FROM session_metadata \
             WHERE last_activity < ? AND {prefix_patterns})"
        ),
        build_params.clone(),
    )
    .await?;

    // Delete the metadata entries themselves
    let deleted = tx
        .execute(
            &format!("DELETE FROM session_metadata WHERE last_activity < ? AND {prefix_patterns}"),
            build_params.clone(),
        )
        .await?;

    tx.commit().await?;

    Ok(deleted)
}

/// Construct an agent ID for direct user-to-agent chat.
///
/// Format: `{channel}_{user_name}_{ws_name}_{role}`
/// Role is the last segment for consistent identification in logs and
/// debugging. The role-last format is immune to underscores in user/workspace
/// names since the role is always the final `_`-delimited segment, but note
/// that the router no longer parses agent ID strings — the role is embedded
/// directly in [`AgentJob`](crate::message_router::AgentJob).
/// This ID is stable across messages — the same ID is used for every message
/// in the same channel/user/role/workspace combination, accumulating conversation
/// history within a single session.
#[must_use]
pub fn direct_agent_id(channel: &str, user_name: &str, role: &str, ws_name: &str) -> String {
    format!("{channel}_{user_name}_{ws_name}_{role}")
}

/// Construct a base agent ID for ticket-driven agent work.
///
/// The base ID format is `ticket_{ticket_id}_{role}`.
///
/// ## Usage
///
/// * **Singular dispatch** (e.g., Engineer at `dispatch_engineer`): the base
///   ID is used directly — no suffix is appended.
///
/// * **Parallel agents** (analysts, reviewers, QA via
///   `run_parallel_agents`): the caller appends `_{index}_{suffix}`
///   for disambiguation, producing IDs like
///   `ticket_{ticket_id}_0_nano_{role}` (role last).
#[must_use]
pub(crate) fn ticket_agent_id(ticket_id: &str, role: &str) -> String {
    format!("ticket_{ticket_id}_{role}")
}

/// Construct an agent ID for Manager agents (workspace-scoped).
///
/// Format: `manager_{ws_name}`
#[must_use]
pub fn manager_agent_id(ws_name: &str) -> String {
    format!("manager_{ws_name}")
}

/// Construct an agent ID for a user message, dispatching to the appropriate
/// format based on role.
///
/// - **Manager** agents use workspace-scoped IDs (`manager_{ws_name}`).
/// - **Non-Manager** agents use channel-scoped IDs
///   (`{channel}_{user_name}_{ws_name}_{role}`).
///
/// This is a convenience wrapper around [`manager_agent_id`] and
/// [`direct_agent_id`] that selects the right format based on
/// whether `role` is `"manager"`.
///
/// # Parameter order
///
/// Matches [`direct_agent_id`]: `channel` first, then `user_name`,
/// `role`, and `ws_name` last.
#[must_use]
pub fn resolve_agent_id(channel: &str, user_name: &str, role: &str, ws_name: &str) -> String {
    if role == "manager" {
        manager_agent_id(ws_name)
    } else {
        direct_agent_id(channel, user_name, role, ws_name)
    }
}

/// Construct an agent ID for Maintainer agents (workspace-scoped, unique per run).
///
/// Format: `maintainer_{ws_name}_{suffix}`
/// Each run gets a fresh ID (via random suffix) — maintainer runs should not
/// accumulate conversation history across maintenance cycles.
#[must_use]
pub(crate) fn maintainer_agent_id(ws_name: &str) -> String {
    format!("maintainer_{}_{}", ws_name, crate::generate_suffix())
}

/// Construct an agent ID for sub-agent asks (Engineer/Maintainer → sub-agent).
///
/// Format: `ask_{ws_name}_{suffix}_{role}`
/// Role is the LAST segment — see [`direct_agent_id`] for rationale.
#[must_use]
pub(crate) fn ask_agent_id(ws_name: &str, role: &str) -> String {
    format!("ask_{}_{}_{}", ws_name, crate::generate_suffix(), role)
}

/// Construct an agent ID for workspace role discovery.
///
/// Format: `discovery_{ws_name}_{suffix}_{role}`
/// Role is the LAST segment — see [`direct_agent_id`] for rationale.
#[must_use]
pub(crate) fn discovery_agent_id(ws_name: &str, role: &str) -> String {
    format!(
        "discovery_{}_{}_{}",
        ws_name,
        crate::generate_suffix(),
        role
    )
}

// ── Existing tests ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_ID: AtomicU32 = AtomicU32::new(0);

    fn unique_key() -> String {
        format!("s{}", TEST_ID.fetch_add(1, Ordering::Relaxed))
    }

    #[tokio::test]
    async fn session_store_create_and_load() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store()
            .append(&k, &ChatMessage::user("hello"))
            .await
            .unwrap();
        let msgs = store().load(&k).await;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "hello");
    }

    #[tokio::test]
    async fn session_store_replace_messages() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store().append(&k, &ChatMessage::user("old")).await.unwrap();
        store()
            .replace_messages(&k, &[ChatMessage::user("new")])
            .await
            .unwrap();
        let msgs = store().load(&k).await;
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "new");
    }

    #[tokio::test]
    async fn session_store_delete() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store().append(&k, &ChatMessage::user("a")).await.unwrap();
        assert!(store().delete(&k).await.unwrap());
        assert!(!store().delete(&k).await.unwrap());
    }
}

/// Validate that the `session_key` → `agent_id` column rename migration works correctly:
///   1. Creates a database with the old schema (`session_key` columns)
///   2. Inserts sample rows via raw SQL
///   3. Opens via [`SessionStore`], which triggers migration in `after_open`
///   4. Verifies data survived intact
///   5. Verifies columns are now named `agent_id`
///   6. Verifies `PRAGMA user_version = 1`
///   7. Re-opens to verify idempotency
#[cfg(test)]
mod migration_tests {
    use super::*;
    use tempfile::TempDir;

    /// Old DDL with `session_key` columns (pre-migration schema).
    const OLD_SESSION_SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_session_key ON sessions(session_key, id);

CREATE TABLE IF NOT EXISTS session_metadata (
    session_key   TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    last_activity TEXT NOT NULL
);";

    #[tokio::test]
    async fn test_session_key_to_agent_id_migration() {
        let tmp = TempDir::new().expect("temp dir for migration test");

        // ── 1. Create a database with the old schema (`session_key` columns) ──
        let db_path = tmp.path().join("db").join("sessions.db");
        std::fs::create_dir_all(db_path.parent().unwrap()).expect("create db directory");

        let old_conn = crate::turso::open_with_schema(&db_path, OLD_SESSION_SCHEMA)
            .await
            .expect("open database with old schema");

        // ── 2. Insert sample rows using the old column layout ───────────────
        // Insert into sessions table
        old_conn
            .execute(
                "INSERT INTO sessions (session_key, role, content, created_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                crate::turso::params!["key-1", "analyst", "Hello", "2024-01-01T00:00:00Z"],
            )
            .await
            .expect("insert session message key-1");

        old_conn
            .execute(
                "INSERT INTO sessions (session_key, role, content, created_at) \
                 VALUES (?1, ?2, ?3, ?4)",
                crate::turso::params!["key-2", "engineer", "World", "2024-01-02T00:00:00Z"],
            )
            .await
            .expect("insert session message key-2");

        // Insert into session_metadata table
        old_conn
            .execute(
                "INSERT INTO session_metadata (session_key, created_at, last_activity) \
                 VALUES (?1, ?2, ?3)",
                crate::turso::params!["key-1", "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z",],
            )
            .await
            .expect("insert session metadata key-1");

        // Checkpoint and close the old connection.
        old_conn
            .checkpoint()
            .await
            .expect("checkpoint old connection");
        drop(old_conn);

        // ── 3. Open via SessionStore — triggers migration in after_open() ───
        let store = SessionStore::open(tmp.path())
            .await
            .expect("open session store (should trigger migration)");

        // ── 4. Verify data survived intact ─────────────────────────────────
        let rows = store
            .conn
            .query(
                "SELECT id, agent_id, role, content FROM sessions ORDER BY id",
                (),
            )
            .await
            .expect("query migrated sessions");
        assert_eq!(
            rows.len(),
            2,
            "should have 2 session messages after migration"
        );
        assert_eq!(rows[0].get::<String>(1).unwrap(), "key-1");
        assert_eq!(rows[0].get::<String>(2).unwrap(), "analyst");
        assert_eq!(rows[0].get::<String>(3).unwrap(), "Hello");
        assert_eq!(rows[1].get::<String>(1).unwrap(), "key-2");
        assert_eq!(rows[1].get::<String>(2).unwrap(), "engineer");
        assert_eq!(rows[1].get::<String>(3).unwrap(), "World");

        // Verify session_metadata also migrated
        let meta_rows = store
            .conn
            .query(
                "SELECT agent_id, created_at FROM session_metadata ORDER BY agent_id",
                (),
            )
            .await
            .expect("query migrated session_metadata");
        assert_eq!(
            meta_rows.len(),
            1,
            "should have 1 metadata row after migration"
        );
        assert_eq!(meta_rows[0].get::<String>(0).unwrap(), "key-1");

        // ── 5. Verify columns are now named `agent_id`, not `session_key` ──
        // Check sessions table
        let sess_info = store
            .conn
            .query("PRAGMA table_info('sessions')", ())
            .await
            .expect("query table_info for sessions");
        let sess_col_names: Vec<String> = sess_info
            .iter()
            .filter_map(|r| r.get::<String>(1).ok())
            .collect();
        assert!(
            !sess_col_names.iter().any(|n| n == "session_key"),
            "column 'session_key' should not exist in sessions after migration; \
             found: {sess_col_names:?}",
        );
        assert!(
            sess_col_names.iter().any(|n| n == "agent_id"),
            "column 'agent_id' must exist in sessions after migration; \
             found: {sess_col_names:?}",
        );

        // Check session_metadata table
        let meta_info = store
            .conn
            .query("PRAGMA table_info('session_metadata')", ())
            .await
            .expect("query table_info for session_metadata");
        let meta_col_names: Vec<String> = meta_info
            .iter()
            .filter_map(|r| r.get::<String>(1).ok())
            .collect();
        assert!(
            !meta_col_names.iter().any(|n| n == "session_key"),
            "column 'session_key' should not exist in session_metadata after migration; \
             found: {meta_col_names:?}",
        );
        assert!(
            meta_col_names.iter().any(|n| n == "agent_id"),
            "column 'agent_id' must exist in session_metadata after migration; \
             found: {meta_col_names:?}",
        );

        // ── 6. Verify PRAGMA user_version = 1 ──────────────────────────────
        let ver_rows = store
            .conn
            .query("PRAGMA user_version", ())
            .await
            .expect("query user_version after migration");
        let version: i64 = ver_rows[0].get(0).expect("get user_version value");
        assert_eq!(version, 1, "user_version should be 1 after migration");

        // ── 7. Re-open to verify idempotency ──────────────────────────────
        drop(store);
        let store2 = SessionStore::open(tmp.path())
            .await
            .expect("re-open session store (idempotent migration)");

        // Data still intact
        let rows2 = store2
            .conn
            .query(
                "SELECT id, agent_id, role, content FROM sessions ORDER BY id",
                (),
            )
            .await
            .expect("query sessions after re-open");
        assert_eq!(rows2.len(), 2, "should still have 2 sessions after re-open");

        // user_version still 1
        let ver_rows2 = store2
            .conn
            .query("PRAGMA user_version", ())
            .await
            .expect("query user_version after re-open");
        let version2: i64 = ver_rows2[0].get(0).expect("get user_version value");
        assert_eq!(version2, 1, "user_version should remain 1 after re-open");
    }
}

// ── TRANSIENT AGENT ID PREFIX GUARDS ──────────────────────────
//
// [`TRANSIENT_AGENT_ID_PREFIXES`] controls which sessions are cleaned up by
// [`cleanup_old_transient_sessions`] (SQL `LIKE '{prefix}%'`, equivalent to
// `key.starts_with(prefix)`).
//
// Two invariants:
// 1. **Forward (no collision)**: User-facing agent IDs must never start with
//    a transient prefix or the periodic cleanup would silently delete user history.
// 2. **Reverse (inclusion)**: Transient agent ID builders must produce IDs
//    starting with a prefix registered in [`TRANSIENT_AGENT_ID_PREFIXES`];
//    an unregistered prefix means transient sessions never get cleaned up (leak).
//
// Limitations: `forward_no_collision_with_user_facing_agent_ids` covers
// `direct_agent_id()` and `manager_agent_id()` patterns.
// `reverse_transient_builders_use_registered_prefixes` covers all transient
// builders (ticket, ask, maintainer, discovery). If a new transient role
// adds an agent ID builder, add it to the reverse test.
// Channel-name collision (a channel registered as "ticket" or "ask") is an
// orthogonal risk — `starts_with` matches the first key segment (channel
// name), which cannot be guarded by assertion because channel names are
// dynamic. Awareness during channel registration is required.
//
// All builders are pure string functions — these are cheap synchronous tests.
// Assertion `Fix:` messages guide corrective action when an invariant breaks.

#[cfg(test)]
mod transient_prefix_tests {
    use super::*;

    /// Known channel identifiers in the system. Must never produce agent IDs
    /// matching a transient prefix.
    const SAFE_CHANNELS: &[&str] = &["telegram", "gui"];

    #[test]
    fn forward_no_collision_with_user_facing_agent_ids() {
        // For every transient prefix, verify that none of the user-facing
        // agent ID patterns start with it. Direct IDs have the format
        // {channel}_{user}_{ws}_{role}, and `starts_with` only checks the
        // first segment (channel name). Since safe channels ("telegram",
        // "gui") don't match any transient prefix, the workspace and role
        // segments have no effect on the assertion outcome — a single role
        // and workspace suffice.
        for prefix in TRANSIENT_AGENT_ID_PREFIXES {
            // Manager uses a separate ID format (manager_{ws_name}).
            let manager_key = manager_agent_id("test-ws");
            assert!(
                !manager_key.starts_with(prefix),
                "MANAGER AGENT ID COLLISION: \
                 prefix='{prefix}' matches id='{manager_key}'. \
                 Fix: remove '{prefix}' from TRANSIENT_AGENT_ID_PREFIXES \
                 or change the manager_agent_id pattern.",
            );

            // Direct chat IDs across all safe channels.
            for channel in SAFE_CHANNELS {
                let key = direct_agent_id(channel, "testuser", "analyst", "test-ws");
                assert!(
                    !key.starts_with(prefix),
                    "DIRECT AGENT ID COLLISION: prefix='{prefix}' \
                     matches id='{key}' (channel='{channel}'). \
                     Fix: remove '{prefix}' from TRANSIENT_AGENT_ID_PREFIXES \
                     or change the agent ID pattern.",
                );
            }
        }
    }

    fn assert_transient_key(key: &str, expected_prefix: &str, builder_expr: &str) {
        assert!(
            key.starts_with(expected_prefix),
            "{builder_expr} = '{key}' does not start with '{expected_prefix}'.\n\
             Fix: update {builder_expr} to produce IDs starting with '{expected_prefix}'.",
        );
        assert!(
            TRANSIENT_AGENT_ID_PREFIXES.contains(&expected_prefix),
            "TRANSIENT_AGENT_ID_PREFIXES is missing '{expected_prefix}' — \
             {builder_expr} sessions will never be cleaned up.\n\
             Fix: add \"{expected_prefix}\" to TRANSIENT_AGENT_ID_PREFIXES.",
        );
    }

    #[test]
    fn reverse_transient_builders_use_registered_prefixes() {
        // Each transient agent ID builder must produce IDs starting with a
        // prefix that is actually registered in TRANSIENT_AGENT_ID_PREFIXES.
        assert_transient_key(
            &ticket_agent_id("abc123", "analyst"),
            "ticket_",
            "ticket_agent_id('abc123', 'analyst')",
        );
        assert_transient_key(
            &ask_agent_id("ws", "coder"),
            "ask_",
            "ask_agent_id('ws', 'coder')",
        );
        assert_transient_key(
            &maintainer_agent_id("ws"),
            "maintainer_",
            "maintainer_agent_id('ws')",
        );
        assert_transient_key(
            &discovery_agent_id("ws", "analyst"),
            "discovery_",
            "discovery_agent_id('ws', 'analyst')",
        );
    }

    #[test]
    fn resolve_agent_id_manager_dispatch() {
        // Manager role produces a manager-scoped ID.
        let key = resolve_agent_id("telegram", "alice", "manager", "my-workspace");
        assert_eq!(key, "manager_my-workspace");
    }

    #[test]
    fn resolve_agent_id_non_manager_dispatch() {
        // Non-Manager role produces a direct channel-scoped ID.
        // Role is the LAST segment.
        let key = resolve_agent_id("discord", "bob", "engineer", "my-workspace");
        assert_eq!(key, "discord_bob_my-workspace_engineer");
    }

    #[test]
    fn resolve_agent_id_lowercase_manager() {
        // The dispatching uses string comparison `"manager"` — verify it works
        // (matches Role::Manager.as_str() which is lowercase).
        let key = resolve_agent_id("gui", "carol", "Manager", "ws");
        assert_ne!(key, "manager_ws", "capital-M 'Manager' should NOT match");
        assert_eq!(key, "gui_carol_ws_Manager");
    }
}

#[test]
fn parse_ts_or_now_invalid_fallback() {
    let before = Utc::now();
    let ts = parse_ts_or_now("garbage-input", "test_invalid");
    let after = Utc::now();
    assert!(
        ts >= before - chrono::Duration::seconds(1),
        "fallback ts {ts} should not be before {before}",
    );
    assert!(
        ts <= after + chrono::Duration::seconds(1),
        "fallback ts {ts} should not be after {after}",
    );
}

// ── Native history decoding ────────────────────────────────────

#[derive(Debug)]
pub(crate) enum DecodedNativeHistoryMessage {
    Assistant {
        content: Option<String>,
        tool_calls: Option<Vec<ToolCall>>,
        reasoning: Option<Reasoning>,
    },
    ToolResult {
        tool_call_id: Option<String>,
        content: String,
    },
}

/// Decode a `ChatMessage` whose `content` is a JSON-wrapped native message.
/// Returns `None` if the message doesn't look like a native/session-persisted message.
pub(crate) fn decode_native_history_message(
    message: &ChatMessage,
) -> Option<DecodedNativeHistoryMessage> {
    let parsed = serde_json::from_str::<serde_json::Value>(&message.content).ok();

    if message.role == ChatRole::Assistant
        && let Some(value) = parsed.as_ref()
    {
        let content = value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

        // Extract reasoning fields for the Assistant variant.
        let (r, rc, rd) =
            crate::providers::reasoning_roundtrip::json_lossless_assistant_reasoning_fields(value);
        let reasoning = Reasoning::from_optional_parts(r, rc, rd);

        let tool_calls = value
            .get("tool_calls")
            .and_then(|v| serde_json::from_value::<Vec<ToolCall>>(v.clone()).ok())
            .map(|mut parsed_calls| {
                for call in &mut parsed_calls {
                    if let Some(s) = call.arguments.as_str()
                        && let Ok(v) = serde_json::from_str::<serde_json::Value>(s)
                    {
                        call.arguments = v;
                    }
                }
                parsed_calls
            });

        return Some(DecodedNativeHistoryMessage::Assistant {
            content,
            tool_calls,
            reasoning,
        });
    }

    if message.role == ChatRole::Tool
        && let Ok(payload) = serde_json::from_str::<ToolResultPayload>(&message.content)
    {
        return Some(DecodedNativeHistoryMessage::ToolResult {
            tool_call_id: Some(payload.tool_call_id),
            content: payload.content,
        });
    }

    None
}
