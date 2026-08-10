//! Session persistence — Turso-backed store + native history decoding.

pub mod dead_session;
pub mod manager;
pub use manager::Session;

use crate::turso::{self, IntoParams, Row, TxGuard, Value, params};
use crate::{ChatMessage, ChatRole, Reasoning, ToolCall, ToolResultPayload};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};

// The summarization LLM call lives in `crate::Agent::summarize` so that all
// parameters (model, reasoning_effort, tools, provider routing)
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

/// Number of latest user messages / assistant answers retained per side
/// after summarization compaction.
pub(crate) const RETENTION_PER_SIDE: usize = 3;

/// Assistant messages carrying tool-call payloads are tool traffic — never
/// counted toward the retention window. Mirrors the app-wide history-rendering
/// discriminator ([`decode_native_history_message`]).
fn is_tool_call_frame(msg: &ChatMessage) -> bool {
    matches!(
        decode_native_history_message(msg),
        Some(DecodedNativeHistoryMessage::Assistant {
            tool_calls: Some(_),
            ..
        })
    )
}

/// Select the latest [`RETENTION_PER_SIDE`] user messages and assistant answers
/// from `history`, merged in chronological order. Tool-call frames and tool
/// results are excluded from both sides. The triggering (in-flight) user
/// message is the newest entry, so it always lands last — callers must not
/// re-append it separately.
#[must_use]
pub(crate) fn select_retention_window(history: &[ChatMessage]) -> Vec<ChatMessage> {
    let mut users: Vec<(usize, &ChatMessage)> = Vec::new();
    let mut assistants: Vec<(usize, &ChatMessage)> = Vec::new();
    for (idx, msg) in history.iter().enumerate().rev() {
        match msg.role {
            ChatRole::User if users.len() < RETENTION_PER_SIDE => users.push((idx, msg)),
            ChatRole::Assistant
                if !is_tool_call_frame(msg) && assistants.len() < RETENTION_PER_SIDE =>
            {
                assistants.push((idx, msg));
            }
            _ => {}
        }
        if users.len() == RETENTION_PER_SIDE && assistants.len() == RETENTION_PER_SIDE {
            break;
        }
    }
    let mut selected: Vec<(usize, &ChatMessage)> = users.into_iter().chain(assistants).collect();
    selected.sort_by_key(|(idx, _)| *idx);
    selected.into_iter().map(|(_, m)| m.clone()).collect()
}

/// Build a user message with the current datetime prepended.
#[must_use]
pub(crate) fn user_msg_with_datetime(content: &str) -> ChatMessage {
    let now = chrono::Local::now();
    ChatMessage::user(format!(
        "<timestamp>{} ({})</timestamp>\n\n{}",
        now.format("%Y-%m-%d %H:%M:%S"),
        now.format("%Z"),
        content
    ))
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
    role          TEXT,
    active_models TEXT
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
    &["ticket_", "ask_", "research_", "maintainer_", "discovery_"];

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
/// are set atomically alongside the messages — closing the atomicity gap of separate
/// context writes.  This is the preferred path for new sessions
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

/// Run the session-listing query body (metadata + message count join) with an
/// optional `WHERE` fragment. Shared by [`SessionStore::list_sessions_with_metadata`]
/// and [`SessionStore::list_sessions_with_metadata_excluding`].
async fn list_sessions_where(
    conn: &turso::Connection,
    where_clause: &str,
    params: impl IntoParams + Send + 'static,
    warn_context: &str,
) -> Vec<SessionMetadata> {
    query_map_collect(
        conn,
        &format!(
            "SELECT {SESSION_LIST_COLUMNS} \
             FROM session_metadata sm \
             LEFT JOIN sessions s ON s.agent_id = sm.agent_id \
             {where_clause} \
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
        warn_context,
        None,
    )
    .await
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
        list_sessions_where(&self.conn, "", (), "list sessions").await
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

        list_sessions_where(
            &self.conn,
            &format!("WHERE {where_clause}"),
            params,
            "list sessions (excluding prefixes)",
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

    /// Persist the `<active-models-opts>` snapshot (rendered model ids) for
    /// mid-session change detection; `None` clears the baseline (no block
    /// rendered — fail-open). Upserts so a missing metadata row (e.g. a
    /// session without a preceding message append) still records the baseline.
    pub(crate) async fn set_active_models(
        &self,
        agent_id: &str,
        snapshot: Option<&str>,
    ) -> Result<()> {
        let now = turso::now();
        self.conn
            .execute(
                "INSERT INTO session_metadata (agent_id, created_at, last_activity, active_models) \
                 VALUES (?1, ?2, ?3, ?4) \
                 ON CONFLICT(agent_id) DO UPDATE SET active_models = excluded.active_models",
                params![agent_id, now.clone(), now, snapshot],
            )
            .await?;
        Ok(())
    }

    /// Read the last persisted `<active-models-opts>` snapshot, if any.
    /// Returns `None` when no baseline exists (no block rendered, or a
    /// session started before this feature).
    pub(crate) async fn get_active_models(&self, agent_id: &str) -> Option<String> {
        match self
            .conn
            .query_optional(
                "SELECT active_models FROM session_metadata WHERE agent_id = ?1",
                params![agent_id],
                |row| row.get::<Option<String>>(0),
            )
            .await
        {
            // A NULL column and a missing metadata row both mean "no baseline".
            Ok(Some(snapshot)) => snapshot,
            Ok(None) => None,
            // A read failure silently disables change detection — log it so
            // the outage is visible rather than looking like a missing block.
            Err(e) => {
                tracing::warn!(agent_id = %agent_id, error = %e, "Failed to read active-models snapshot");
                None
            }
        }
    }
}

impl SessionStore {
    /// Post-open setup: reject legacy schemas, then ensure indexes.
    async fn after_open(&self) -> anyhow::Result<()> {
        // Legacy-format DBs (pre-migration) fail fast with an actionable error
        // instead of failing later at index creation or on no-such-column queries.
        if turso::column_exists(&self.conn, "sessions", "session_key").await?
            || turso::column_exists(&self.conn, "session_metadata", "session_key").await?
        {
            return Err(anyhow!(
                "sessions.db has a legacy schema (session_key present); migrations \
                 were removed — restore a backup created by a current mahbot version"
            ));
        }

        // Index must exist before sessions are queried.
        self.conn
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_sessions_agent_id \
                 ON sessions(agent_id, id);",
            )
            .await
            .context("Failed to create sessions index")?;

        // active_models stores the rendered <active-models-opts> snapshot
        // (model ids) for mid-session change detection. Guarded ALTER follows
        // the done_at/bounce_count precedents in board.rs.
        if !turso::column_exists(&self.conn, "session_metadata", "active_models").await? {
            self.conn
                .execute(
                    "ALTER TABLE session_metadata ADD COLUMN active_models TEXT",
                    (),
                )
                .await
                .context("Failed to add active_models column to session_metadata")?;
        }
        Ok(())
    }
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

/// Clear the session for a channel/user/role/workspace, returning the result message.
pub async fn clear_session(channel: &str, user_name: &str, role: &str, ws_name: &str) -> String {
    Session::delete(&resolve_agent_id(channel, user_name, role, ws_name)).await
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

/// Construct an agent ID for a deep-research sub-agent (decomposers,
/// round-1 researchers, gap-round analysts, verification analysts).
///
/// Format: `research_{ws_name}_{suffix}_{label}`
#[must_use]
pub(crate) fn research_agent_id(ws_name: &str, label: &str) -> String {
    format!(
        "research_{}_{}_{}",
        ws_name,
        crate::generate_suffix(),
        label
    )
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
            .batch_append(&k, &[ChatMessage::user("hello")])
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
        store()
            .batch_append(&k, &[ChatMessage::user("old")])
            .await
            .unwrap();
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
        store()
            .batch_append(&k, &[ChatMessage::user("a")])
            .await
            .unwrap();
        assert!(store().delete(&k).await.unwrap());
        assert!(!store().delete(&k).await.unwrap());
    }

    #[tokio::test]
    async fn session_context_roundtrip() {
        crate::util::test::init_test_stores().await;
        let agent_id = unique_key();

        // Initially, no context should exist.
        assert!(store().get_session_context(&agent_id).await.is_none());

        // Store context alongside a message.
        store()
            .append_with_context(
                &agent_id,
                &ChatMessage::user("hello"),
                "gui",
                "alice",
                "work",
                "engineer",
            )
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
            .append_with_context(
                &agent_id,
                &ChatMessage::user("hello again"),
                "telegram",
                "bob",
                "project-x",
                "analyst",
            )
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
            .batch_append(&agent_id, &[ChatMessage::user("hello")])
            .await
            .unwrap();
        assert_eq!(
            store().get_last_message_role(&agent_id).await,
            Some(ChatRole::User)
        );

        // Append an assistant message → Assistant.
        store()
            .batch_append(&agent_id, &[ChatMessage::assistant("world")])
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
            // list_sessions_with_metadata joins with session_metadata, so the
            // context columns are needed too (append alone doesn't create them).
            store()
                .append_with_context(
                    id,
                    &ChatMessage::user("msg"),
                    "test_channel",
                    "test_user",
                    "test_ws",
                    "engineer",
                )
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

    // ── Retention-window selection ───────────────────────────────────

    fn tool_call_frame() -> ChatMessage {
        let tc = ToolCall {
            id: "t1".into(),
            name: "read".into(),
            arguments: serde_json::json!({}),
        };
        ChatMessage::assistant(
            crate::providers::reasoning_roundtrip::assistant_replay_payload(
                Some("prologue"),
                std::slice::from_ref(&tc),
                None,
            )
            .to_string(),
        )
    }

    #[test]
    fn retention_window_keeps_latest_per_side_excluding_tools() {
        let frame = tool_call_frame();
        let messages = vec![
            ChatMessage::system("prompt"),
            ChatMessage::user("u1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("u2"),
            frame.clone(),
            ChatMessage::tool_result("t1", "file contents"),
            ChatMessage::assistant("a2"),
            ChatMessage::user("u3"), // in-flight — newest, always retained
        ];
        let window = select_retention_window(&messages);
        let contents: Vec<&str> = window.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(contents, vec!["u1", "a1", "u2", "a2", "u3"]);
        // In-flight message appears exactly once — no duplicate re-append.
        assert_eq!(contents.iter().filter(|c| **c == "u3").count(), 1);
    }

    #[test]
    fn retention_window_drops_oldest_beyond_three() {
        let messages: Vec<ChatMessage> = (0..5)
            .flat_map(|i| {
                vec![
                    ChatMessage::user(format!("u{i}")),
                    ChatMessage::assistant(format!("a{i}")),
                ]
            })
            .collect();
        let window = select_retention_window(&messages);
        let contents: Vec<&str> = window.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(contents, vec!["u2", "a2", "u3", "a3", "u4", "a4"]);
    }

    #[test]
    fn retention_window_short_history_and_json_answers() {
        // First turn: no assistant answers → users only.
        let window = select_retention_window(&[ChatMessage::user("u1")]);
        assert_eq!(window.len(), 1);
        assert_eq!(window[0].content, "u1");

        // Reasoning-only JSON payloads and JSON-looking plain answers are
        // answers, not tool traffic.
        let reasoning = Reasoning {
            reasoning: Some("thinking".into()),
            reasoning_content: None,
            reasoning_details: None,
        };
        let reasoning_msg = ChatMessage::assistant(
            crate::providers::reasoning_roundtrip::assistant_replay_payload(
                Some(""),
                &[],
                Some(&reasoning),
            )
            .to_string(),
        );
        let window =
            select_retention_window(&[reasoning_msg, ChatMessage::assistant("{\"result\": 42}")]);
        assert_eq!(window.len(), 2);
    }

    #[tokio::test]
    async fn finalize_appends_only_new_assistant_answer() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        // Persisted history ending with an assistant answer (e.g. a compacted
        // session whose retained window ends with one).
        store()
            .batch_append(
                &k,
                &[
                    ChatMessage::system("prompt"),
                    ChatMessage::user("u1"),
                    ChatMessage::assistant("a1"),
                ],
            )
            .await
            .unwrap();
        let ws = crate::workspace::test_ws_named("/_test_finalize_guard", "finalize_test");
        let mut session = Session::default();
        session
            .init(&k, "", &ws, &crate::Role::Assistant, None, "gui", "tester")
            .await
            .unwrap();

        // No new assistant output this turn → the persisted trailing answer
        // must not be re-appended.
        session.finalize(&k).await.unwrap();
        assert_eq!(store().load(&k).await.len(), 3);

        // A genuinely new answer IS appended.
        session.push_assistant("a2".to_string());
        session.finalize(&k).await.unwrap();
        let msgs = store().load(&k).await;
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[3].content, "a2");
    }

    /// A failed incoming-message persist queues the gap; the next successful
    /// persist (tool round) flushes it ahead of its own batch — no loss, no
    /// duplicate frames.
    #[tokio::test]
    async fn failed_drain_gap_flushed_by_later_persist() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store()
            .batch_append(
                &k,
                &[
                    ChatMessage::system("prompt"),
                    ChatMessage::user("u1"),
                    ChatMessage::assistant("a1"),
                ],
            )
            .await
            .unwrap();
        let ws = crate::workspace::test_ws_named("/_test_gap_flush", "gap_flush");
        let mut session = Session::default();
        session
            .init(&k, "", &ws, &crate::Role::Assistant, None, "gui", "tester")
            .await
            .unwrap();

        // Failed drain: comment delivered to history only.
        session.push_messages_unpersisted(&[ChatMessage::user("C1")]);

        // Tool round: the gap must be flushed in the same transaction.
        session
            .persist_messages(
                &k,
                &[
                    ChatMessage::assistant("A"),
                    ChatMessage::tool_result("t1", "R1"),
                ],
            )
            .await
            .unwrap();

        // Final answer.
        session.push_assistant("final".into());
        session.finalize(&k).await.unwrap();

        let msgs = store().load(&k).await;
        let contents: Vec<&str> = msgs.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(
            contents,
            [
                "prompt",
                "u1",
                "a1",
                "C1",
                "A",
                "{\"tool_call_id\":\"t1\",\"content\":\"R1\"}",
                "final"
            ]
        );
    }

    /// An aborted turn (no final answer) still flushes the failed-drain gap
    /// so delivered comments survive in the DB for a recovery retry.
    #[tokio::test]
    async fn failed_drain_gap_flushed_on_aborted_turn() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store()
            .batch_append(
                &k,
                &[ChatMessage::system("prompt"), ChatMessage::user("u1")],
            )
            .await
            .unwrap();
        let ws = crate::workspace::test_ws_named("/_test_gap_abort", "gap_abort");
        let mut session = Session::default();
        session
            .init(&k, "", &ws, &crate::Role::Assistant, None, "gui", "tester")
            .await
            .unwrap();

        session.push_messages_unpersisted(&[ChatMessage::user("C1")]);
        session.finalize(&k).await.unwrap();

        let msgs = store().load(&k).await;
        let contents: Vec<&str> = msgs.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(contents, ["prompt", "u1", "C1"]);
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
            &research_agent_id("ws", "decomposer"),
            "research_",
            "research_agent_id('ws', 'decomposer')",
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
