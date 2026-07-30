//! Session persistence — Turso-backed store + native history decoding.

pub mod dead_session;
pub mod manager;
pub use manager::Session;

use crate::turso::{self, IntoParams, Row, TxGuard, Value, params};
use crate::{ChatMessage, ChatRole, Reasoning, ToolCall, ToolResultPayload};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};

// The summarization LLM call lives in `crate::Agent::summarize` so that all
// parameters (model, temperature, reasoning_effort, tools, provider routing)
// are byte-identical to the agent's work loop.  This section keeps only the
// constants and helpers used by `Session::apply_summary`.

/// History-length threshold (in estimated tokens) that triggers summarization.
///
/// This is a conservative default chosen to work across models with varying
/// context window sizes (250K–1M). The value of **200,000** estimated tokens
/// translates to roughly 800K characters of message content under the rough
/// `estimate_tokens` formula (~4 chars/token + 4 tokens per-message overhead).
///
/// ## Why 200K?
///
/// The actual token consumption at request time can be higher than `estimate_tokens`
/// suggests for several reasons:
///
/// * **Tokenization ratio** — Code- and JSON-heavy agent conversations (tool
///   calls, structured outputs) can tokenize at ~2.5 chars/token rather than
///   the estimate's 4 chars/token.
/// * **Tool schemas** — The tool definitions injected by `build_chat_request`
///   consume ~10–20K actual tokens that are **not** counted by `estimate_tokens`
///   (they live in the `tools` field of the request, not in `messages`).
///
/// Every modern model (as of 2026) supports 250K+ context windows making 200K safe.
pub(crate) const SUMMARIZATION_THRESHOLD: usize = 200_000;

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
    last_activity TEXT NOT NULL,
    channel       TEXT,
    user_name     TEXT,
    workspace_name TEXT,
    role          TEXT
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

/// Context data stored alongside a session for recovery purposes.
///
/// Populated when a user initiates a direct agent session so the dead-session
/// recovery poller can reconstruct an [`AgentJob`](crate::message_router::AgentJob)
/// without parsing the agent ID string.
///
/// # Column naming note
///
/// The `role` field is persisted in the `session_metadata.role` column, which
/// stores the **agent role** (e.g. `"engineer"`, `"analyst"`, `"reviewer"`).
/// This is semantically distinct from the `sessions.role` column, which stores
/// the **message role** (`"user"`, `"assistant"`, `"tool"`, `"system"`) — even
/// though both columns happen to share the name `role` in different tables.
#[derive(Debug, Clone)]
pub(crate) struct SessionContext {
    pub channel: String,
    pub user_name: String,
    pub workspace_name: String,
    /// Agent role (e.g. `"engineer"`, `"analyst"`, `"reviewer"`).
    ///
    /// Contrast with `sessions.role` which stores the message role
    /// (`"user"`, `"assistant"`, `"tool"`, `"system"`).  The two are
    /// semantically unrelated despite sharing a column name.
    pub role: String,
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
///
/// When `context` is `Some((channel, user_name, workspace_name, role))`, the context columns
/// are set atomically alongside the messages — closing the atomicity gap described in
/// [`SessionStore::set_session_context`].  This is the preferred path for new sessions
/// and subsequent turns.
async fn insert_messages_in_transaction(
    tx: &TxGuard<'_>,
    agent_id: &str,
    messages: &[ChatMessage],
    context: Option<(&str, &str, &str, &str)>,
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
    match context {
        Some((channel, user_name, workspace_name, role)) => {
            tx.execute(
                "INSERT INTO session_metadata (agent_id, created_at, last_activity, \
                 channel, user_name, workspace_name, role) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) \
                 ON CONFLICT(agent_id) DO UPDATE SET \
                 last_activity = excluded.last_activity, \
                 channel = excluded.channel, \
                 user_name = excluded.user_name, \
                 workspace_name = excluded.workspace_name, \
                 role = excluded.role",
                params![
                    agent_id,
                    now.clone(),
                    now,
                    channel,
                    user_name,
                    workspace_name,
                    role,
                ],
            )
            .await?;
        }
        None => {
            tx.execute(
                "INSERT INTO session_metadata (agent_id, created_at, last_activity) \
                 VALUES (?1, ?2, ?3) \
                 ON CONFLICT(agent_id) DO UPDATE SET \
                 last_activity = excluded.last_activity",
                params![agent_id, now.clone(), now],
            )
            .await?;
        }
    }
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
        context: Option<(&str, &str, &str, &str)>,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        if replace {
            tx.execute(
                "DELETE FROM sessions WHERE agent_id = ?1",
                params![agent_id],
            )
            .await?;
        }
        insert_messages_in_transaction(&tx, agent_id, messages, context).await?;
        tx.commit().await?;
        Ok(())
    }

    pub(crate) async fn batch_append(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
    ) -> Result<()> {
        self.append_messages(agent_id, messages, false, None).await
    }

    /// Like [`batch_append`], but also sets session context (`channel`,
    /// `user_name`, `workspace_name`, `role`) in the same transaction as
    /// the message insert, eliminating the atomicity gap between message
    /// persistence and context persistence.
    pub(crate) async fn batch_append_with_context(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
        channel: &str,
        user_name: &str,
        workspace_name: &str,
        role: &str,
    ) -> Result<()> {
        self.append_messages(
            agent_id,
            messages,
            false,
            Some((channel, user_name, workspace_name, role)),
        )
        .await
    }

    /// Like [`append`], but also sets session context in the same transaction.
    pub(crate) async fn append_with_context(
        &self,
        agent_id: &str,
        message: &ChatMessage,
        channel: &str,
        user_name: &str,
        workspace_name: &str,
        role: &str,
    ) -> Result<()> {
        self.batch_append_with_context(
            agent_id,
            std::slice::from_ref(message),
            channel,
            user_name,
            workspace_name,
            role,
        )
        .await
    }

    pub(crate) async fn replace_messages(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
    ) -> Result<()> {
        self.append_messages(agent_id, messages, true, None).await
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

    /// Like [`list_sessions_with_metadata`], but excludes sessions whose
    /// `agent_id` starts with any of the given prefixes by adding a `WHERE`
    /// clause (e.g., `"manager_"`, `"ticket_"`).
    ///
    /// Uses parameterised `NOT LIKE ?N` placeholders with the prefix patterns
    /// passed as query parameters — no string interpolation into SQL.
    pub(crate) async fn list_sessions_with_metadata_excluding(
        &self,
        exclude_prefixes: &[&str],
    ) -> Vec<SessionMetadata> {
        if exclude_prefixes.is_empty() {
            return self.list_sessions_with_metadata().await;
        }

        let where_clause = exclude_prefixes
            .iter()
            .enumerate()
            .map(|(i, _)| format!("sm.agent_id NOT LIKE ?{}", i + 1))
            .collect::<Vec<_>>()
            .join(" AND ");

        let params: Vec<turso::Value> = exclude_prefixes
            .iter()
            .map(|p| turso::Value::Text(format!("{p}%")))
            .collect();

        query_map_collect(
            &self.conn,
            &format!(
                "SELECT {SESSION_LIST_COLUMNS} \
                 FROM session_metadata sm \
                 LEFT JOIN sessions s ON s.agent_id = sm.agent_id \
                 WHERE {where_clause} \
                 GROUP BY sm.agent_id \
                 ORDER BY sm.last_activity DESC",
            ),
            params,
            |row| {
                Ok::<_, anyhow::Error>(session_metadata_from_row(
                    &row.get::<String>(COL_SL_AGENT_ID)?,
                    &row.get::<String>(COL_SL_LAST_ACTIVITY)?,
                    row.get::<i64>(COL_SL_MESSAGE_COUNT)?,
                ))
            },
            "list sessions (excluding prefixes)",
            None,
        )
        .await
    }

    /// Lightweight query: get the role of the last message in a session.
    /// Returns `None` if the session has no messages.
    pub(crate) async fn get_last_message_role(&self, agent_id: &str) -> Option<ChatRole> {
        let rows = self
            .conn
            .query(
                "SELECT role FROM sessions WHERE agent_id = ?1 ORDER BY id DESC LIMIT 1",
                params![agent_id],
            )
            .await
            .ok()?;
        rows.first().and_then(|row| {
            let role_str: String = row.get(0).ok()?;
            role_str.parse::<ChatRole>().ok()
        })
    }

    /// Store session context (channel, user_name, workspace_name, role)
    /// alongside the session metadata for use by the dead-session recovery
    /// poller.
    ///
    /// Uses an upsert so it works regardless of whether a metadata row already
    /// exists.
    ///
    /// # Atomicity note
    ///
    /// Prefer [`batch_append_with_context`](SessionStore::batch_append_with_context)
    /// or [`append_with_context`](SessionStore::append_with_context) when
    /// setting context alongside message persistence — they write both in a
    /// single transaction.  This standalone method is retained for external
    /// callers that may need to set context independently of message insertion.
    ///
    /// When called separately from message persistence, a crash between the
    /// message write and this method leaves the session with messages but no
    /// context (unrecoverable by the dead-session poller).  This is a narrow
    /// window and accepted for callers that cannot batch the write.
    #[allow(dead_code)]
    pub(crate) async fn set_session_context(
        &self,
        agent_id: &str,
        channel: &str,
        user_name: &str,
        workspace_name: &str,
        role: &str,
    ) -> Result<()> {
        let now = turso::now();
        self.conn
            .execute(
                "INSERT INTO session_metadata (agent_id, created_at, last_activity, channel, user_name, workspace_name, role)
                 VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(agent_id) DO UPDATE SET
                 channel = excluded.channel,
                 user_name = excluded.user_name,
                 workspace_name = excluded.workspace_name,
                 role = excluded.role",
                params![agent_id, now, channel, user_name, workspace_name, role],
            )
            .await
            .context("Failed to set session context")?;
        Ok(())
    }

    /// Retrieve stored session context for a given agent ID.
    /// Returns `None` if the session has no metadata or the context columns
    /// are null (pre-migration sessions).
    pub(crate) async fn get_session_context(&self, agent_id: &str) -> Option<SessionContext> {
        let rows = self
            .conn
            .query(
                "SELECT channel, user_name, workspace_name, role FROM session_metadata WHERE agent_id = ?1",
                params![agent_id],
            )
            .await
            .ok()?;
        rows.first().and_then(|row| {
            let channel: Option<String> = row.get(0).ok();
            let user_name: Option<String> = row.get(1).ok();
            let workspace_name: Option<String> = row.get(2).ok();
            let role: Option<String> = row.get(3).ok();
            Some(SessionContext {
                channel: channel?,
                user_name: user_name?,
                workspace_name: workspace_name?,
                role: role?,
            })
        })
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
/// Migration v2: add context columns (`channel`, `user_name`, `workspace_name`, `role`).
#[allow(clippy::too_many_lines)]
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

    if current_version < 2 {
        tracing::info!("Schema migration: adding context columns to session_metadata (version 2)");

        // Add columns for session context (channel, user_name, workspace_name, role).
        // Each ADD COLUMN is auto-committed in SQLite.
        let table_info = conn
            .query("PRAGMA table_info('session_metadata')", ())
            .await
            .context("Failed to read PRAGMA table_info for session_metadata migration")?;

        let existing: std::collections::HashSet<String> = table_info
            .iter()
            .filter_map(|row| row.get::<String>(1).ok())
            .collect();

        for col in &["channel", "user_name", "workspace_name", "role"] {
            if !existing.contains(*col) {
                conn.execute(
                    &format!("ALTER TABLE session_metadata ADD COLUMN {col} TEXT"),
                    (),
                )
                .await
                .with_context(|| {
                    format!(
                        "Schema migration failed: unable to add column {col} to session_metadata"
                    )
                })?;
            }
        }

        conn.execute("PRAGMA user_version = 2", ())
            .await
            .context("Schema migration failed: unable to set PRAGMA user_version to 2")?;

        conn.checkpoint().await.context(
            "Schema migration failed: unable to checkpoint after adding context columns",
        )?;

        tracing::info!(
            "Schema migration complete: added context columns to session_metadata (version 2)"
        );

        // Warn about pre-migration sessions that will be unrecoverable
        // by the dead-session recovery poller (NULL context columns).
        let null_count: i64 = conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE channel IS NULL",
                (),
            )
            .await
            .context("Failed to count pre-migration sessions")?
            .first()
            .and_then(|row| row.get::<i64>(0).ok())
            .unwrap_or(0);
        if null_count > 0 {
            tracing::warn!(
                count = null_count,
                "Pre-migration sessions have NULL context columns and will \
                 not be recovered by the dead-session recovery poller. \
                 These sessions have no stored channel/user_name/workspace/role."
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

    #[tokio::test]
    async fn session_context_roundtrip() {
        crate::util::test::init_test_stores().await;
        let agent_id = unique_key();

        // Initially, no context should exist.
        assert!(store().get_session_context(&agent_id).await.is_none());

        // Store context.
        store()
            .set_session_context(&agent_id, "gui", "alice", "work", "engineer")
            .await
            .unwrap();

        // Retrieve and verify.
        let ctx = store()
            .get_session_context(&agent_id)
            .await
            .expect("should have context after set");
        assert_eq!(ctx.channel, "gui");
        assert_eq!(ctx.user_name, "alice");
        assert_eq!(ctx.workspace_name, "work");
        assert_eq!(ctx.role, "engineer");

        // Overwrite with different values.
        store()
            .set_session_context(&agent_id, "telegram", "bob", "project-x", "analyst")
            .await
            .unwrap();

        let ctx = store()
            .get_session_context(&agent_id)
            .await
            .expect("should have updated context");
        assert_eq!(ctx.channel, "telegram");
        assert_eq!(ctx.user_name, "bob");
        assert_eq!(ctx.workspace_name, "project-x");
        assert_eq!(ctx.role, "analyst");
    }

    #[tokio::test]
    async fn session_get_last_message_role() {
        crate::util::test::init_test_stores().await;
        let agent_id = unique_key();

        // Empty session → None.
        assert!(store().get_last_message_role(&agent_id).await.is_none());

        // Append a user message → User.
        store()
            .append(&agent_id, &ChatMessage::user("hello"))
            .await
            .unwrap();
        assert_eq!(
            store().get_last_message_role(&agent_id).await,
            Some(ChatRole::User)
        );

        // Append an assistant message → Assistant.
        store()
            .append(&agent_id, &ChatMessage::assistant("world"))
            .await
            .unwrap();
        assert_eq!(
            store().get_last_message_role(&agent_id).await,
            Some(ChatRole::Assistant)
        );
    }

    #[tokio::test]
    async fn session_list_excluding_prefixes() {
        crate::util::test::init_test_stores().await;

        // Create sessions with different agent ID patterns.
        let direct_id = unique_key();
        let manager_id = format!("manager_{}", unique_key());
        let ticket_id = format!("ticket_{}", unique_key());
        let ask_id = format!("ask_{}", unique_key());
        let maintainer_id = format!("maintainer_{}", unique_key());

        for id in &[&direct_id, &manager_id, &ticket_id, &ask_id, &maintainer_id] {
            store().append(id, &ChatMessage::user("msg")).await.unwrap();
            // list_sessions_with_metadata joins with session_metadata, so we
            // need metadata rows too (append doesn't create them).
            store()
                .set_session_context(id, "test_channel", "test_user", "test_ws", "engineer")
                .await
                .unwrap();
        }

        // Without exclusions, all 5 sessions should be listed.
        let all = store().list_sessions_with_metadata().await;
        let all_ids: Vec<&str> = all.iter().map(|s| s.agent_id.as_str()).collect();
        assert!(
            all_ids.contains(&direct_id.as_str()),
            "direct session should be in full list"
        );

        // Exclude manager_ + transient prefixes → only the direct session remains.
        let excluded = store()
            .list_sessions_with_metadata_excluding(&["manager_", "ticket_", "ask_", "maintainer_"])
            .await;
        let excluded_ids: Vec<&str> = excluded.iter().map(|s| s.agent_id.as_str()).collect();
        assert!(
            excluded_ids.contains(&direct_id.as_str()),
            "direct session should survive exclusion"
        );
        assert!(
            !excluded_ids.contains(&manager_id.as_str()),
            "manager_ prefix should be excluded"
        );
        assert!(
            !excluded_ids.contains(&ticket_id.as_str()),
            "ticket_ prefix should be excluded"
        );
        assert!(
            !excluded_ids.contains(&ask_id.as_str()),
            "ask_ prefix should be excluded"
        );
        assert!(
            !excluded_ids.contains(&maintainer_id.as_str()),
            "maintainer_ prefix should be excluded"
        );
    }

    /// Empty messages are not appended by [`Session::init`].  Recovery retries
    /// pass an empty message so the agent re-runs against the existing session
    /// history without adding a new user turn.
    #[tokio::test]
    async fn session_init_empty_message_no_append() {
        crate::util::test::init_test_stores().await;
        let agent_id = unique_key();
        let ws = crate::workspace::test_ws_named("/_test_empty_session_init", "empty_test");
        let role = crate::Role::Assistant;

        // First turn: init with a real message creates the session.
        let mut session = Session::default();
        session
            .init(&agent_id, "hello", &ws, &role, None, "gui", "tester")
            .await
            .unwrap();
        let len_after_real = session.history().len();
        assert!(
            len_after_real >= 2,
            "real message should produce system prompt + user message (got {len_after_real})"
        );

        // Second turn: init with empty message should NOT append.
        let mut session = Session::default();
        session
            .init(&agent_id, "", &ws, &role, None, "gui", "tester")
            .await
            .unwrap();
        assert_eq!(
            session.history().len(),
            len_after_real,
            "empty message must not append to session history",
        );
    }
}

///   1. Creates a database with the old schema (`session_key` columns)
///   2. Inserts sample rows via raw SQL
///   3. Opens via [`SessionStore`], which triggers migration in `after_open`
///   4. Verifies data survived intact
///   5. Verifies columns are now named `agent_id`
///   6. Verifies `PRAGMA user_version = 2` (v1 + v2 migrations)
///   7. Verifies v2 context columns were added: `channel`, `user_name`,
///      `workspace_name`, `role`
///   8. Re-opens to verify idempotency
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

        // ── 6. Verify PRAGMA user_version = 2 ──────────────────────────────
        let ver_rows = store
            .conn
            .query("PRAGMA user_version", ())
            .await
            .expect("query user_version after migration");
        let version: i64 = ver_rows[0].get(0).expect("get user_version value");
        assert_eq!(
            version, 2,
            "user_version should be 2 after migration (v1 + v2)"
        );

        // ── 7. Verify v2 context columns exist ──────────────────────────────
        let v2_info = store
            .conn
            .query("PRAGMA table_info('session_metadata')", ())
            .await
            .expect("query table_info for session_metadata v2 check");
        let v2_col_names: Vec<String> = v2_info
            .iter()
            .filter_map(|r| r.get::<String>(1).ok())
            .collect();
        for col in &["channel", "user_name", "workspace_name", "role"] {
            assert!(
                v2_col_names.iter().any(|n| n == col),
                "v2 column '{col}' should exist in session_metadata after migration; \
                 found: {v2_col_names:?}",
            );
        }

        // ── 8. Re-open to verify idempotency ──────────────────────────────
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

        // user_version still 2
        let ver_rows2 = store2
            .conn
            .query("PRAGMA user_version", ())
            .await
            .expect("query user_version after re-open");
        let version2: i64 = ver_rows2[0].get(0).expect("get user_version value");
        assert_eq!(version2, 2, "user_version should remain 2 after re-open");
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
