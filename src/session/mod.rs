//! Session persistence — Turso-backed store + native history decoding.
//!
//! Also hosts the lock-free live transcript snapshot accessor
//! ([`transcript`]) that the Running Agents GUI reads to show a running
//! agent's in-memory conversation including the unpersisted tail.

mod activity;
mod compaction_dump;
pub mod dead_session;
pub(crate) mod image_strip;
mod manager;
pub(crate) mod personal_changes;
pub(crate) mod transcript;
pub(crate) use manager::FinalizeOutcome;
pub(crate) use manager::RewriteOutcome;
pub(crate) use manager::Session;
pub(crate) use transcript::{TRANSCRIPT_REGISTRY, TranscriptSnapshot};

use crate::db::{self, IntoParams, Row, TxGuard, Value, params};
use crate::{ChatMessage, ChatRole, Reasoning, ToolCall, ToolResultPayload};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use std::borrow::Cow;

// The summarization LLM call lives in `crate::Agent::summarize` so that all
// parameters (model, reasoning_effort, tools, provider routing)
// are byte-identical to the agent's work loop.

/// History-length threshold (in REAL provider-reported tokens) that triggers
/// summarization.
///
/// The session's current length in tokens is the input + output tokens of the
/// most recent successful agent LLM call (see
/// [`crate::Agent::record_session_usage`]), persisted per session in
/// `session_metadata.token_length` and loaded at session init. `None` — no
/// successful usage-bearing call ever recorded a value (new sessions,
/// pre-migration sessions; approved no-backfill semantics) or the last
/// compaction cleared it — only disables this token trigger; the
/// image-accumulation trigger ([`SUMMARIZATION_IMAGE_COUNT`]) still applies.
///
/// **200,000** is a conservative default chosen to work across models with
/// varying context window sizes (250K–1M).
pub(crate) const SUMMARIZATION_THRESHOLD: u64 = 200_000;

/// Summarization also fires on image accumulation: when the session's user
/// messages carry at least this many DISTINCT data-URI image markers. The
/// count is a prefix-gate superset of the markers promoted to provider
/// vision parts — it never undercounts, but a malformed `data:image/*`
/// payload may be counted too. `>=` deliberately differs from the strict
/// token threshold: images are a hard floor on context quality regardless
/// of how a provider tokenizes them.
pub(crate) const SUMMARIZATION_IMAGE_COUNT: usize = 20;

/// Number of latest user messages / assistant answers retained per side
/// after summarization compaction.
const RETENTION_PER_SIDE: usize = 3;

/// Decode the tool-call payload of a native assistant history frame — the one
/// calls-only view of [`decode_native_history_message`].
///
/// Returns `Some(calls)` only when the message decodes as an assistant frame
/// carrying a tool-call payload; `None` covers non-assistant frames, decode
/// failures, and frames without a payload. `Some([])` (payload present but
/// empty) stays distinguishable from `None` — a present-but-empty call list is
/// a tool-call frame with nothing to run, not an absence of one.
///
/// Callers span unrelated concerns — retention-window selection, dead-session
/// dangling-tail detection, tool-result settling, and query/command telemetry
/// over persisted history — so a semantic change here reaches all of them.
pub(crate) fn decoded_tool_calls(msg: &ChatMessage) -> Option<Vec<ToolCall>> {
    match decode_native_history_message(msg) {
        Some(DecodedNativeHistoryMessage::Assistant { tool_calls, .. }) => tool_calls,
        _ => None,
    }
}

/// Assistant messages carrying tool-call payloads are tool traffic — never
/// counted toward the retention window. Mirrors the app-wide history-rendering
/// discriminator ([`decode_native_history_message`]).
fn is_tool_call_frame(msg: &ChatMessage) -> bool {
    decoded_tool_calls(msg).is_some()
}

/// True when the message decodes as a native assistant tool-call frame
/// carrying at least one tool call — the dangling-tail signal used by the
/// dead-session recovery poller. Mirrors [`Session::pending_tool_frame`]:
/// a frame that is the session TAIL can have no following result rows, so
/// non-empty calls there means unresolved calls.
fn is_dangling_tool_call_frame(role: ChatRole, content: &str) -> bool {
    decoded_tool_calls(&ChatMessage {
        role,
        content: content.to_string(),
    })
    .is_some_and(|calls| !calls.is_empty())
}

/// Select the latest [`RETENTION_PER_SIDE`] user messages and assistant answers
/// from `history`, merged in chronological order. Tool-call frames and tool
/// results are excluded from both sides. The newest user message is always
/// among them — callers must not re-append it separately.
#[must_use]
fn select_retention_window(history: &[ChatMessage]) -> Vec<ChatMessage> {
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

/// Render the local datetime in the user-message timestamp format.
#[must_use]
pub(crate) fn render_timestamp() -> String {
    let now = chrono::Local::now();
    format!("{} ({})", now.format("%Y-%m-%d %H:%M:%S"), now.format("%Z"))
}

/// Build a user message with a `<timestamp>` block appended. `round_ts`
/// (pre-rendered via [`render_timestamp`]) pins one value per round so
/// parallel members share a byte-identical first message; `None` stamps now
/// (mid-round injected content).
#[must_use]
pub(crate) fn user_msg_with_ts(content: &str, round_ts: Option<&str>) -> ChatMessage {
    let ts = round_ts.map_or_else(render_timestamp, str::to_string);
    ChatMessage::user(format!("{content}\n\n<timestamp>{ts}</timestamp>"))
}

crate::define_store! {
    /// Global session store.
    pub(crate) static SESSIONS: SessionStore,
    expect = "SESSIONS not initialized — call init_all_stores() first",
}

// ── Column index constants ──────────────────────────────────
crate::columns! {
    SESSION_MESSAGE_COLUMNS [SM] {
        ROLE    => "role",
        CONTENT => "content",
    }
}

// Session list with metadata (10-column SELECT: sm.agent_id, sm.last_activity,
// sm.message_count, sm.token_length, sm.role, sm.user_name, sm.workspace_name,
// sm.custom_role, sm.custom_identifier, sm.launch_stopped).
// Counts are read from the denormalized metadata column — NOT from a JOIN over
// the `sessions` message table (the list re-reads on every runtime tick while
// the Sessions page is open, and from the dead-session recovery poller; scanning
// the full message table per refresh was the largest repeated query in the
// system). The count is maintained in the same transaction as message writes
// (see `insert_messages_in_transaction`). It is NOT backfilled: sessions that
// predate the metadata column default to 0. Schema: `db/migrations.rs`.
crate::columns! {
    SESSION_LIST_COLUMNS [SL] {
        AGENT_ID          => "sm.agent_id",
        LAST_ACTIVITY     => "sm.last_activity",
        MESSAGE_COUNT     => "sm.message_count",
        TOKEN_LENGTH      => "sm.token_length",
        ROLE              => "sm.role",
        USER_NAME         => "sm.user_name",
        WORKSPACE_NAME    => "sm.workspace_name",
        CUSTOM_ROLE       => "sm.custom_role",
        CUSTOM_IDENTIFIER => "sm.custom_identifier",
        LAUNCH_STOPPED    => "sm.launch_stopped",
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
/// - Direct chat: `{user}_{ws_name}_{role}` (or the deduped
///   `{user}_personal:{role}` for the user's own personal workspace) — a real
///   user whose name collides with a reserved prefix is escaped with a `user_`
///   prefix (see [`safe_user_segment`]).
/// - Manager: `manager_{ws_name}` — the Manager session carries both chat conversation
///   and notification context and must never be added here.
/// - A custom role's continuable branch: `role_{name}` / `role_{name}:{identifier}`
///   ([`custom_role_agent_id`]) — an administrator-defined conversation the user
///   comes back to, so it persists like a direct one. Only its one-off siblings
///   (`rolerun_`, [`custom_role_run_agent_id`]) are transient.
///
/// If a new agent role is added that can talk to users directly, its agent ID prefix
/// must also be excluded from this list.
const TRANSIENT_AGENT_ID_PREFIXES: &[&str] = &[
    "ticket_",
    "analyze_",
    "research_",
    "cleanup_",
    "maintainer_",
    "discovery_",
    "rolerun_",
];

/// Agent-id prefix of a custom role's continuable branch — `role_{name}` (the
/// record's default branch) or `role_{name}:{identifier}` (a named branch). The
/// `:` separator is what makes the two forms injective: names and identifiers
/// may both contain `_`, so a `_` separator would alias `(a_b, None)` with
/// `(a, Some(b))` — the `personal:{user}` workspace key already establishes `:`
/// in agent ids.
///
/// Deliberately NOT in [`TRANSIENT_AGENT_ID_PREFIXES`]: the branch is a
/// user-facing conversation the administrator defined and comes back to, so it
/// is never TTL-purged and is eligible for dead-session recovery. It is in the
/// wider [`reserved_agent_id_prefixes`] union, so no direct user id can start
/// with it (a user literally named `role` is escaped to `user_role`).
pub(crate) const CUSTOM_ROLE_AGENT_ID_PREFIX: &str = "role_";

/// Agent-id prefix of a custom role's ONE-OFF run — `rolerun_{name}:{suffix}`.
/// The same `:` separator, and the same injectivity rule, as
/// [`CUSTOM_ROLE_AGENT_ID_PREFIX`] — here the [`crate::generate_suffix`] NanoID
/// suffix itself may contain `_`, which is exactly why that rule applies too.
/// Registered in [`TRANSIENT_AGENT_ID_PREFIXES`] so it expires with the
/// ordinary session lifetime.
pub(crate) const CUSTOM_ROLE_RUN_AGENT_ID_PREFIX: &str = "rolerun_";

/// Agent-id prefix of a worker branch — `worker_{branch}`. A worker branch is a
/// continuable conversation like a custom role's branch: it is deliberately NOT
/// in [`TRANSIENT_AGENT_ID_PREFIXES`] (never TTL-purged) and IS recovered by the
/// dead-session poller, but it is in the wider [`reserved_agent_id_prefixes`]
/// union so no direct user id can shadow it. It is NOT in
/// [`non_user_facing_agent_id_prefixes`]: the branch is user-facing.
pub(crate) const WORKER_BRANCH_AGENT_ID_PREFIX: &str = "worker_";

#[derive(Debug, Clone)]
pub(crate) struct SessionMetadata {
    pub agent_id: String,
    pub last_activity: DateTime<Utc>,
    pub message_count: usize,
    /// Real provider-reported session length (input + output tokens of the
    /// last successful agent LLM call), if ever recorded. Older sessions are
    /// intentionally never backfilled — `None` renders no token value.
    pub token_length: Option<u64>,
    /// Session context columns (see `SessionContext`); `None` when the row was
    /// written without context (transient/background sessions).
    pub role: Option<String>,
    pub user_name: Option<String>,
    pub workspace_name: Option<String>,
    /// The custom role this session belongs to, when it is a custom-role
    /// session; `None` for a worker branch (whose branch name is in
    /// `custom_identifier`).
    pub custom_role: Option<String>,
    /// The named branch this continuable session continues, when the branch was
    /// named — a custom role's identifier, or a worker branch's name.
    pub custom_identifier: Option<String>,
    /// Whether the launch was stopped before it produced anything.
    pub launch_stopped: bool,
}

/// Context data stored alongside a session for recovery purposes.
///
/// Written on every non-empty turn for all roles (see
/// [`Session::append_turn_message`](crate::session::Session::append_turn_message)),
/// so the dead-session recovery poller can always reconstruct an
/// [`AgentJob`](crate::agent::message_router::AgentJob) without parsing the
/// agent ID string.
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
    pub role: String,
    /// The custom role this session belongs to, when it is a custom-role
    /// session; `None` for a worker branch (whose branch name is in
    /// `custom_identifier`).
    pub custom_role: Option<String>,
    /// The named branch this continuable session continues, when the branch was
    /// named — a custom role's identifier, or a worker branch's name.
    pub custom_identifier: Option<String>,
    /// The conversation that asked for this branch's launch, when one did — the
    /// asking custom role's branch and the folder its turn ran in, so a worker
    /// branch's report returns there (see [`crate::Launch::reply_to`]). `None`
    /// for an owner- or automation-started launch, and for every session recorded
    /// before the `reply_to` column existed.
    pub reply_to: Option<crate::agent::message_router::Asker>,
}

#[expect(clippy::too_many_arguments)]
fn session_metadata_from_row(
    agent_id: &str,
    activity_str: &str,
    count: i64,
    token_length: Option<i64>,
    role: Option<String>,
    user_name: Option<String>,
    workspace_name: Option<String>,
    custom_role: Option<String>,
    custom_identifier: Option<String>,
    launch_stopped: Option<String>,
) -> Result<SessionMetadata> {
    let last_activity = db::parse_utc_timestamp(activity_str).with_context(|| {
        format!("invalid last_activity {activity_str:?} for session {agent_id}")
    })?;
    Ok(SessionMetadata {
        agent_id: agent_id.to_string(),
        last_activity,
        message_count: usize::try_from(count).unwrap_or(0),
        token_length: token_length.and_then(|t| u64::try_from(t).ok()),
        role,
        user_name,
        workspace_name,
        custom_role,
        custom_identifier,
        launch_stopped: launch_stopped.is_some_and(|flag| flag == "1"),
    })
}

/// Row insert shared by the append and settle paths — the only two writers of
/// `sessions` rows.
const INSERT_SESSION_ROW: &str =
    "INSERT INTO sessions (agent_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4)";

/// The session metadata a message batch writes in the SAME transaction as its
/// rows: the describing columns (only a batch that also sets them — a session's
/// first turn) and the leading context block's length (only the batch that wrote
/// the block).
#[derive(Clone, Copy, Default)]
struct BatchMetadata<'a> {
    context: Option<(&'a str, &'a str, &'a str, &'a str)>,
    block_len: Option<usize>,
    /// Clear the recorded provider length in this same transaction — the
    /// compaction rewrite, see [`SessionStore::replace_messages`].
    clear_token_length: bool,
}

impl<'a> BatchMetadata<'a> {
    /// A batch that also writes the session's describing columns.
    const fn context(
        channel: &'a str,
        user_name: &'a str,
        workspace_name: &'a str,
        role: &'a str,
    ) -> Self {
        Self {
            context: Some((channel, user_name, workspace_name, role)),
            block_len: None,
            clear_token_length: false,
        }
    }

    /// The length of the leading context block this same batch wrote — `Some`
    /// only for the batch that wrote the block.
    const fn block_len(self, len: Option<usize>) -> Self {
        Self {
            block_len: len,
            ..self
        }
    }
}

/// Insert messages into `sessions` and upsert `session_metadata` within an existing transaction.
/// Shared helper used by [`SessionStore::append_messages`].
///
/// When `meta.context` is `Some((channel, user_name, workspace_name, role))`, the context columns
/// are set atomically alongside the messages — closing the atomicity gap of separate
/// context writes.  This is the preferred path for new sessions
/// and subsequent turns. `meta.block_len` records the leading context block's
/// length in that same transaction when the batch wrote the block.
///
/// `replace` selects the denormalized `session_metadata.message_count`
/// semantics: `false` (append) increments the stored count by the batch
/// length; `true` (replace — summarization compaction deletes then inserts)
/// SETs it to the batch length. A naive shared increment would double-count
/// the replace path, since the old rows are gone before the insert.
async fn insert_messages_in_transaction(
    tx: &TxGuard<'_>,
    agent_id: &str,
    messages: &[ChatMessage],
    meta: BatchMetadata<'_>,
    replace: bool,
) -> Result<()> {
    let now = db::now();
    for msg in messages {
        tx.execute(
            INSERT_SESSION_ROW,
            params![
                agent_id,
                msg.role.to_string(),
                msg.content.clone(),
                now.clone()
            ],
        )
        .await?;
    }
    // Message count = the number of `sessions` rows for this agent (system
    // prompts, tool-call frames, and tool results all count — one row per
    // message, matching the historical COUNT(s.id) list semantics). On a
    // fresh metadata row the INSERT branch carries the batch length directly;
    // on an existing row the update branch adds (append) or overwrites
    // (replace). The INSERT branches rely on the `NOT NULL DEFAULT 0`
    // declaration for rows created by other paths (e.g. `set_token_length`).
    let count = i64::try_from(messages.len()).context("message batch exceeds i64")?;
    match meta.context {
        Some((channel, user_name, workspace_name, role)) => {
            // `created_at` stamps the session_metadata row only on creation
            // (the session's creation timestamp); the update branch below must
            // never touch it.
            let created_at = now.clone();
            let count_clause = message_count_clause(replace);
            tx.upsert_row(
                &format!(
                    "UPDATE session_metadata SET last_activity = ?2, channel = ?4, \
                     user_name = ?5, workspace_name = ?6, role = ?7, {count_clause} \
                     WHERE agent_id = ?1"
                ),
                || {
                    params![
                        agent_id,
                        now.as_str(),
                        count,
                        channel,
                        user_name,
                        workspace_name,
                        role,
                    ]
                },
                "INSERT INTO session_metadata (agent_id, last_activity, message_count, \
                 channel, user_name, workspace_name, role, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
                 ON CONFLICT(agent_id) DO NOTHING",
                params![
                    agent_id,
                    now.as_str(),
                    count,
                    channel,
                    user_name,
                    workspace_name,
                    role,
                    created_at.as_str(),
                ],
            )
            .await?;
        }
        None => upsert_message_count(tx, agent_id, count, &now, replace).await?,
    }
    // A batch that wrote the session's leading context block records its length
    // in the SAME transaction: a failed write can then never leave a recorded
    // length that disagrees with the block it describes.
    if let Some(len) = meta.block_len {
        let len = i64::try_from(len).context("context block length exceeds i64")?;
        set_metadata_value_in_tx(
            tx,
            agent_id,
            MetadataColumn::ContextBlockLen,
            Value::Integer(len),
        )
        .await?;
    }
    // A compaction rewrite clears the recorded provider length in the SAME
    // transaction, for the same reason: the value described the history this
    // batch replaces, and one left behind would re-fire the compaction on the
    // session's next turn.
    if meta.clear_token_length {
        set_metadata_value_in_tx(tx, agent_id, MetadataColumn::TokenLength, Value::Null).await?;
    }
    Ok(())
}

/// The message-count clause shared by both `session_metadata` write paths:
/// `replace=true` overwrites (the old rows were already deleted, so
/// a shared increment would double-count), `replace=false` appends. The batch
/// length is bound as `?3` on both paths. This must
/// stay separate from the [`SessionStore::set_metadata_value`] write, which deliberately
/// never touches `last_activity`.
fn message_count_clause(replace: bool) -> &'static str {
    if replace {
        "message_count = ?3"
    } else {
        "message_count = message_count + ?3"
    }
}

/// Upsert one `session_metadata` column inside a caller-owned transaction.
/// `created_at` is stamped only on row creation (never on an existing row).
async fn set_metadata_value_in_tx(
    tx: &TxGuard<'_>,
    agent_id: &str,
    column: MetadataColumn,
    value: Value,
) -> Result<()> {
    let col = column.as_str();
    let now = db::now();
    tx.upsert_row(
        &format!("UPDATE session_metadata SET {col} = ?1 WHERE agent_id = ?2"),
        || params![value.clone(), agent_id],
        &format!(
            "INSERT INTO session_metadata (agent_id, last_activity, {col}, created_at) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(agent_id) DO NOTHING"
        ),
        params![agent_id, now.as_str(), value.clone(), now.as_str()],
    )
    .await?;
    Ok(())
}

/// Upsert the context-free `session_metadata` count row: `last_activity` is
/// stamped, `message_count` is added on append/settle (`replace=false`) or
/// overwritten on the replace path (`replace=true`). `created_at` is
/// INSERT-only: it stamps the session's creation time and is never touched
/// by the update.
async fn upsert_message_count(
    tx: &TxGuard<'_>,
    agent_id: &str,
    count: i64,
    now: &str,
    replace: bool,
) -> Result<()> {
    let count_clause = message_count_clause(replace);
    tx.upsert_row(
        &format!(
            "UPDATE session_metadata SET last_activity = ?2, {count_clause} \
             WHERE agent_id = ?1"
        ),
        || params![agent_id, now, count],
        "INSERT INTO session_metadata (agent_id, last_activity, message_count, created_at) \
         VALUES (?1, ?2, ?3, ?4) \
         ON CONFLICT(agent_id) DO NOTHING",
        params![agent_id, now, count, now],
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
    conn: &db::Connection,
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

/// The session-list SQL with an optional `WHERE` fragment. Shared by the
/// infallible and checked session-list readers.
fn sessions_list_sql(where_clause: &str) -> String {
    format!(
        "SELECT {SESSION_LIST_COLUMNS} \
         FROM session_metadata sm \
         {where_clause} \
         ORDER BY sm.last_activity DESC",
    )
}

/// The session-transcript SQL for one agent, oldest message first. Shared by
/// the infallible [`SessionStore::load`] and strict
/// [`SessionStore::load_checked`] readers, so the two read contracts cannot
/// read different statements.
fn session_messages_sql() -> String {
    format!("SELECT {SESSION_MESSAGE_COLUMNS} FROM sessions WHERE agent_id = ?1 ORDER BY id ASC")
}

/// Decode one [`SESSION_LIST_COLUMNS`] row. Shared by the infallible and
/// checked session-list readers.
fn session_metadata_row(row: &Row) -> Result<SessionMetadata> {
    session_metadata_from_row(
        &row.get::<String>(COL_SL_AGENT_ID)?,
        &row.get::<String>(COL_SL_LAST_ACTIVITY)?,
        row.get::<i64>(COL_SL_MESSAGE_COUNT)?,
        row.get::<Option<i64>>(COL_SL_TOKEN_LENGTH)?,
        row.get::<Option<String>>(COL_SL_ROLE)?,
        row.get::<Option<String>>(COL_SL_USER_NAME)?,
        row.get::<Option<String>>(COL_SL_WORKSPACE_NAME)?,
        row.get::<Option<String>>(COL_SL_CUSTOM_ROLE)?,
        row.get::<Option<String>>(COL_SL_CUSTOM_IDENTIFIER)?,
        row.get::<Option<String>>(COL_SL_LAUNCH_STOPPED)?,
    )
}

/// Run the session-listing query body (metadata columns only, no message-table
/// join) with an optional `WHERE` fragment. Shared by
/// [`SessionStore::list_sessions_with_metadata`] and
/// [`SessionStore::list_sessions_with_metadata_excluding`].
///
/// The message count is read from the denormalized `session_metadata.message_count`
/// column (maintained in the same transaction as message writes; NOT backfilled —
/// sessions predating the metadata column default to 0). The historical
/// LEFT JOIN + GROUP BY over the full `sessions`
/// table was the largest repeated query in the system (this list re-reads on
/// every runtime tick while the Sessions page is open, and from the dead-session
/// recovery poller) and has been removed. Ordering and filtering are unchanged.
async fn list_sessions_where(
    conn: &db::Connection,
    where_clause: &str,
    params: impl IntoParams + Send + 'static,
    warn_context: &str,
) -> Vec<SessionMetadata> {
    query_map_collect(
        conn,
        &sessions_list_sql(where_clause),
        params,
        session_metadata_row,
        warn_context,
        None,
    )
    .await
}

/// Decode one [`SESSION_MESSAGE_COLUMNS`] row into a [`ChatMessage`]. Shared
/// by the infallible [`SessionStore::load`] and strict
/// [`SessionStore::load_checked`] reads.
fn session_message_from_row(row: &Row) -> Result<ChatMessage> {
    Ok(ChatMessage {
        role: row
            .get::<String>(COL_SM_ROLE)?
            .parse::<ChatRole>()
            .map_err(|e| anyhow!(e))?,
        content: row.get(COL_SM_CONTENT)?,
    })
}

/// A `session_metadata` column the read/write helpers may target; the closed
/// enum makes the `format!` interpolation compile-time safe.
enum MetadataColumn {
    ActiveModels,
    UserActivity,
    TokenLength,
    SleepEnded,
    CustomRole,
    CustomIdentifier,
    ReplyTo,
    LaunchStopped,
    WorkspaceName,
    ContextBlockLen,
}

impl MetadataColumn {
    fn as_str(&self) -> &'static str {
        match self {
            Self::ActiveModels => "active_models",
            Self::UserActivity => "user_activity",
            Self::TokenLength => "token_length",
            Self::SleepEnded => "sleep_ended",
            Self::CustomRole => "custom_role",
            Self::CustomIdentifier => "custom_identifier",
            Self::ReplyTo => "reply_to",
            Self::LaunchStopped => "launch_stopped",
            Self::WorkspaceName => "workspace_name",
            Self::ContextBlockLen => "context_block_len",
        }
    }
}

// ── Methods — callable on the static ──────────────────────────

/// One row of the final post-frame sequence returned by
/// [`SessionStore::settle_tool_results`] — freshly settled results, follow-up
/// rows and previously captured rows in their final order; the caller replays
/// it verbatim into memory.
#[derive(Debug, Clone)]
pub(crate) struct SettledRow {
    pub role: String,
    pub content: String,
    pub created_at: String,
}

/// Build the final post-frame session-row sequence for a settle. The last
/// tool-call frame's tool results must appear in FRAME CALL ORDER (provider
/// validity), so:
///
/// 1. Captured tool rows are keyed by their decoded `tool_call_id`.
/// 2. For each frame call id IN ORDER: the new result for that id if present,
///    else the captured tool row for that id if present.
/// 3. Then the `follow_up` rows (synthetic user image messages).
///
/// The frame-order interleaving (steps 1–2) is the contract. The only
/// production caller runs before `append_turn_message`, so the dangling frame
/// is always the session tail and captured rows are only frame sibling tool
/// rows — each is either consumed in step 2 or dropped by the post-frame
/// rewrite.
fn build_settle_sequence(
    captured: &[SettledRow],
    frame_calls: &[crate::ToolCall],
    results: &[(String, String)],
    follow_up: &[ChatMessage],
    now: &str,
) -> Result<Vec<SettledRow>> {
    let mut captured_by_call_id: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (i, row) in captured.iter().enumerate() {
        if row.role == "tool"
            && let Ok(payload) = serde_json::from_str::<crate::ToolResultPayload>(&row.content)
        {
            captured_by_call_id.insert(payload.tool_call_id, i);
        }
    }

    let new_results: std::collections::HashMap<&str, &str> = results
        .iter()
        .map(|(id, content)| (id.as_str(), content.as_str()))
        .collect();

    let mut sequence = Vec::new();
    for call in frame_calls {
        if let Some(content) = new_results.get(call.id.as_str()) {
            let payload = crate::ToolResultPayload {
                tool_call_id: call.id.clone(),
                content: (*content).to_string(),
            };
            sequence.push(SettledRow {
                role: "tool".to_string(),
                content: serde_json::to_string(&payload)?,
                created_at: now.to_string(),
            });
        } else if let Some(&i) = captured_by_call_id.get(&call.id) {
            sequence.push(captured[i].clone());
        }
    }
    for msg in follow_up {
        sequence.push(SettledRow {
            role: msg.role.to_string(),
            content: msg.content.clone(),
            created_at: now.to_string(),
        });
    }
    Ok(sequence)
}

impl SessionStore {
    pub(crate) async fn load(&self, agent_id: &str) -> Vec<ChatMessage> {
        query_map_collect(
            &self.conn,
            &session_messages_sql(),
            params![agent_id],
            session_message_from_row,
            "load session",
            Some(agent_id),
        )
        .await
    }

    /// Strict [`Self::load`]: a query error or the first undecodable row is
    /// returned as an `Err`. Used by the Sessions transcript, which must render
    /// a failed read as an error rather than as an empty transcript. The trade
    /// is deliberate and total: one undecodable row replaces the whole read, so
    /// the rows that were readable are not shown either — skipping it would
    /// render a transcript that silently omits what the store holds.
    pub(crate) async fn load_checked(&self, agent_id: &str) -> Result<Vec<ChatMessage>> {
        self.conn
            .query_map_strict(
                &session_messages_sql(),
                params![agent_id],
                session_message_from_row,
            )
            .await
    }

    /// Session-non-emptiness check (resume dispatch rule): true when the
    /// agent has at least one non-empty message row. Avoids materializing the
    /// full history just to test `.is_empty()` — the engineer's accumulated
    /// session can exceed 200k tokens.
    pub(crate) async fn has_content(&self, agent_id: &str) -> bool {
        self.conn
            .query_optional(
                "SELECT 1 FROM sessions WHERE agent_id = ?1 AND length(content) > 0 LIMIT 1",
                params![agent_id],
                |_| Ok::<(), anyhow::Error>(()),
            )
            .await
            .ok()
            .flatten()
            .is_some()
    }

    /// `created_at` of the session's last assistant message row.
    ///
    /// That row is the session's dangling tool-call frame whenever one exists:
    /// the frame is persisted BEFORE its calls run (emission-time durability)
    /// and the agent cannot emit another assistant message past an unsettled
    /// frame. `None` when the session has no assistant row at all.
    pub(crate) async fn last_assistant_created_at(&self, agent_id: &str) -> Result<Option<String>> {
        self.conn
            .query_optional(
                "SELECT created_at FROM sessions \
                 WHERE agent_id = ?1 AND role = 'assistant' ORDER BY id DESC LIMIT 1",
                params![agent_id],
                |row| row.get::<String>(0),
            )
            .await
            .context("read the session's last assistant timestamp")
    }

    async fn append_messages(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
        replace: bool,
        meta: BatchMetadata<'_>,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        if replace {
            tx.execute(
                "DELETE FROM sessions WHERE agent_id = ?1",
                params![agent_id],
            )
            .await?;
        }
        insert_messages_in_transaction(&tx, agent_id, messages, meta, replace).await?;
        tx.commit().await?;
        Ok(())
    }

    async fn batch_append(&self, agent_id: &str, messages: &[ChatMessage]) -> Result<()> {
        self.append_messages(agent_id, messages, false, BatchMetadata::default())
            .await
    }

    /// Like [`batch_append`], but writes `meta` in the same transaction as the
    /// message insert, eliminating the atomicity gap between the rows and the
    /// session metadata they belong to.
    async fn batch_append_with_context(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
        meta: BatchMetadata<'_>,
    ) -> Result<()> {
        self.append_messages(agent_id, messages, false, meta).await
    }

    /// Like [`batch_append`], but also sets session context in the same transaction.
    async fn append_with_context(
        &self,
        agent_id: &str,
        message: &ChatMessage,
        meta: BatchMetadata<'_>,
    ) -> Result<()> {
        self.batch_append_with_context(agent_id, std::slice::from_ref(message), meta)
            .await
    }

    /// Rewrite the whole history, recording the leading context block's length in
    /// the same transaction when this rewrite wrote the block (see
    /// [`crate::session::Session::refresh_context_block`]).
    ///
    /// `clear_token_length` is set by the compaction rewrite
    /// ([`crate::session::Session::apply_summary`]); the recorded length went with
    /// the history it described.
    async fn replace_messages(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
        block_len: Option<usize>,
        clear_token_length: bool,
    ) -> Result<()> {
        self.append_messages(
            agent_id,
            messages,
            true,
            BatchMetadata {
                context: None,
                block_len,
                clear_token_length,
            },
        )
        .await
    }

    /// Rewrite the content of the most recent `user`-role message row for the
    /// agent — the durable half of the input-image-rejection strip (see
    /// [`crate::session::image_strip`]). A single-row positional UPDATE: the caller's
    /// in-memory "most recent User-role message" corresponds to this row only
    /// while that message is within the persisted prefix; the in-memory guard
    /// lives in [`crate::session::Session::rewrite_last_user_message`].
    ///
    /// Returns an error when no `user`-role row exists (0 rows affected), so
    /// a caller that believes it found one learns the write did not land.
    pub(crate) async fn rewrite_last_user_message(
        &self,
        agent_id: &str,
        content: &str,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        let changed = tx
            .execute(
                "UPDATE sessions SET content = ?1 WHERE agent_id = ?2 AND id = (
                    SELECT MAX(id) FROM sessions WHERE agent_id = ?2 AND role = 'user'
                )",
                params![content, agent_id],
            )
            .await?;
        tx.commit().await?;
        if changed == 0 {
            anyhow::bail!("no user message row to rewrite for agent {agent_id}");
        }
        Ok(())
    }

    /// Test-only thin wrapper: begin a transaction, run the tx-core settle, and
    /// commit. Production uses [`Self::settle_tool_results_tx`] inside the
    /// agent's atomic settle+terminalize transaction.
    #[cfg(test)]
    pub(crate) async fn settle_tool_results(
        &self,
        agent_id: &str,
        results: &[(String, String)],
        follow_up: &[ChatMessage],
    ) -> Result<Vec<SettledRow>> {
        let tx = self.conn.begin_tx().await?;
        let rows = self
            .settle_tool_results_tx(&tx, agent_id, results, follow_up)
            .await?;
        tx.commit().await?;
        Ok(rows)
    }

    /// Settle (re-)persisted tool results directly after the session's last
    /// tool-call frame, in one transaction: the rows that follow that frame are
    /// captured and deleted, the new tool results are inserted at the frame
    /// boundary in frame call order (preserving any captured sibling tool
    /// rows), then the follow-up rows (synthetic user image messages).
    /// `session_metadata.message_count` is bumped by
    /// `results.len() + follow_up.len()` — exact only because a call with a
    /// committed result is never pending (the pending scan is the single settle
    /// source), so no captured frame sibling is ever superseded here; a
    /// double-settle of the same frame would drift the count and is out of
    /// contract. Returns the full final post-frame sequence so the caller can
    /// mirror the rewrite in memory. A no-op (warn + empty) when no tool-call
    /// frame exists.
    ///
    /// The production caller runs before `append_turn_message`, so the dangling
    /// frame is always the session tail and the captured rows are only the
    /// frame's sibling tool rows; the frame-order interleaving is the contract
    /// (see [`build_settle_sequence`]).
    ///
    /// The frame-locate SELECTs run on the SAME tx as the delete/rebuild so the
    /// locate+write is one unit — a crash (or concurrent writer) can never
    /// observe a half-settled frame. The caller owns `commit`/`rollback`.
    pub(crate) async fn settle_tool_results_tx(
        &self,
        tx: &TxGuard<'_>,
        agent_id: &str,
        results: &[(String, String)],
        follow_up: &[ChatMessage],
    ) -> Result<Vec<SettledRow>> {
        // Frame location: scan assistant rows carrying the tool-call marker
        // newest-first and pick the FIRST that structurally decodes as a
        // tool-call frame. A later assistant text row may merely contain the
        // substring — skip it. Tool results carry "tool_call_id", never
        // "tool_calls" — the role filter already excludes them. Content is
        // fetched in the same scan so the frame's ordered calls can be decoded
        // without a second SELECT.
        let candidates: Vec<(i64, String)> = tx
            .query(
                "SELECT id, content FROM sessions \
                 WHERE agent_id = ?1 AND role = 'assistant' AND content LIKE '%\"tool_calls\"%' \
                 ORDER BY id DESC",
                params![agent_id],
            )
            .await
            .context("find tool-call frame candidates")?
            .into_iter()
            .map(|r| Result::<_, turso::Error>::Ok((r.get::<i64>(0)?, r.get::<String>(1)?)))
            .collect::<Result<Vec<_>, _>>()
            .context("decode tool-call frame candidate rows")?;

        let mut frame: Option<(i64, Vec<ToolCall>)> = None;
        for (id, content) in candidates {
            let msg = ChatMessage {
                role: ChatRole::Assistant,
                content,
            };
            if let Some(calls) = decoded_tool_calls(&msg) {
                // Frame call order is the provider-valid result order.
                frame = Some((id, calls));
                break;
            }
        }
        let Some((frame_row_id, frame_calls)) = frame else {
            tracing::warn!(
                agent_id,
                "settle_tool_results: no tool-call frame found — no-op"
            );
            return Ok(Vec::new());
        };

        let captured: Vec<SettledRow> = tx
            // Raw column indices: `created_at` has no COL_SM_* constant (it is
            // not part of SESSION_MESSAGE_COLUMNS), so the row mapping cannot
            // go through the shared column list.
            .query_map_strict(
                "SELECT role, content, created_at FROM sessions WHERE agent_id = ?1 AND id > ?2 ORDER BY id",
                params![agent_id, frame_row_id],
                |row| {
                    Ok::<_, anyhow::Error>(SettledRow {
                        role: row.get(0)?,
                        content: row.get(1)?,
                        created_at: row.get(2)?,
                    })
                },
            )
            .await
            .context("capture rows after tool-call frame")?;

        // Delete the rows that follow the frame (the frame itself stays). The
        // sequence below replaces them.
        tx.execute(
            "DELETE FROM sessions WHERE agent_id = ?1 AND id > ?2",
            params![agent_id, frame_row_id],
        )
        .await
        .context("delete rows after tool-call frame")?;

        let now = db::now();
        let sequence = build_settle_sequence(&captured, &frame_calls, results, follow_up, &now)?;
        for row in &sequence {
            tx.execute(
                INSERT_SESSION_ROW,
                params![
                    agent_id,
                    row.role.clone(),
                    row.content.clone(),
                    row.created_at.clone()
                ],
            )
            .await
            .context("insert settled session row")?;
        }

        // Message count: the captured rows were deleted + re-inserted (net
        // zero); the new results and follow-up rows add exactly `results.len()
        // + follow_up.len()`.
        let count = i64::try_from(results.len() + follow_up.len())
            .context("settle result count exceeds i64")?;
        upsert_message_count(tx, agent_id, count, &now, false)
            .await
            .context("bump settle message count")?;

        Ok(sequence)
    }

    pub(crate) async fn delete(&self, agent_id: &str) -> Result<bool> {
        // Every deletion path (the `/clear` command, the Sessions page, the
        // maintainer's self-cleanup) goes through here, so this is where a
        // launch folder's search engine — held for the session's lifetime and
        // referenced by nothing else (see `workspace_index::release_path_named_engine`)
        // is released, and where the computer tool's session ends: the conversation
        // this id addressed is gone, so the application/window ids minted under it are
        // too (see `tools::computer::end_session`). Read the recorded folder before the
        // rows go. The ids go even when the conversation had no row to delete — a
        // reset is what it is, whether or not the session had stored a message — while
        // the engine is released only for a real removal.
        let workspace = self.session_workspace_name(agent_id).await;
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
        crate::tools::computer::end_session(agent_id);
        if deleted > 0
            && let Some(workspace) = workspace
        {
            crate::workspace_index::release_path_named_engine(&workspace);
        }
        Ok(deleted > 0)
    }

    /// Delete the `index`-th message row (0-based row ordinal in `id` order —
    /// the order [`Self::load_checked`] returns) and every later row for
    /// `agent_id`: the Sessions page's manual truncate. The cut is inclusive;
    /// an index past the end deletes nothing. Returns the number of rows
    /// removed.
    ///
    /// Deliberately unguarded — no coordination with a running agent. Accepted
    /// consequences: a cut can leave an in-flight round committing tool results
    /// without their tool-call frame row, and can leave a tail the dead-session
    /// recovery poller treats as resumable — `last_activity` is deliberately
    /// untouched, since bumping it would suppress exactly that recovery.
    /// `token_length` is left stale: a recount is not free, and summarization
    /// firing early is intended. `message_count` is adjusted from this DELETE's
    /// own row count.
    pub(crate) async fn truncate_from(&self, agent_id: &str, index: usize) -> Result<usize> {
        let offset = i64::try_from(index).context("truncate index exceeds i64")?;
        let tx = self.conn.begin_tx().await?;
        let deleted = tx
            .execute(
                "DELETE FROM sessions WHERE agent_id = ?1 AND id >= \
                 (SELECT id FROM sessions WHERE agent_id = ?1 ORDER BY id ASC LIMIT 1 OFFSET ?2)",
                params![agent_id, offset],
            )
            .await?;
        if deleted > 0 {
            let removed = i64::try_from(deleted).context("removed row count exceeds i64")?;
            tx.execute(
                "UPDATE session_metadata SET message_count = MAX(message_count - ?2, 0) \
                 WHERE agent_id = ?1",
                params![agent_id, removed],
            )
            .await?;
        }
        let removed = usize::try_from(deleted).context("removed row count exceeds usize")?;
        tx.commit().await?;
        Ok(removed)
    }

    /// The full session list, infallible by contract: whatever could be read is
    /// returned (a query error yields an empty list). Kept for readers with no
    /// person in front of them — the exclusion helper's no-prefix case, where an
    /// empty list is a safe input — and for tests; the Sessions page reads
    /// [`Self::list_sessions_with_metadata_checked`], which renders a failed read
    /// as a failure rather than as an empty list.
    pub(crate) async fn list_sessions_with_metadata(&self) -> Vec<SessionMetadata> {
        list_sessions_where(&self.conn, "", (), "list sessions").await
    }

    /// Strict [`Self::list_sessions_with_metadata`]: a query error or the first
    /// undecodable row is returned as an `Err`. Used by the Sessions list,
    /// which must render a failed read as a failure rather than as an empty
    /// list. The trade is deliberate and total: one undecodable row replaces
    /// the whole read rather than silently shortening the list.
    pub(crate) async fn list_sessions_with_metadata_checked(&self) -> Result<Vec<SessionMetadata>> {
        self.conn
            .query_map_strict(&sessions_list_sql(""), (), session_metadata_row)
            .await
    }

    /// Like [`list_sessions_with_metadata`], but excludes sessions whose
    /// `agent_id` starts with any of the given prefixes by adding a `WHERE`
    /// clause (e.g., `"manager_"`, `"ticket_"`).
    ///
    /// Uses parameterised `NOT LIKE ?N` placeholders with the prefix patterns
    /// passed as query parameters — no string interpolation into SQL.
    async fn list_sessions_with_metadata_excluding(
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

        let params: Vec<db::Value> = exclude_prefixes
            .iter()
            .map(|p| db::Value::Text(format!("{p}%")))
            .collect();

        list_sessions_where(
            &self.conn,
            &format!("WHERE {where_clause}"),
            params,
            "list sessions (excluding prefixes)",
        )
        .await
    }

    /// Lightweight query: role and raw content of the session's last message
    /// row (`None` for an empty session). Powers content-aware tail
    /// classification (the dead-session poller's dangling-frame check).
    pub(crate) async fn get_last_message_tail(&self, agent_id: &str) -> Option<(ChatRole, String)> {
        let rows = self
            .conn
            .query(
                &format!(
                    "SELECT {SESSION_MESSAGE_COLUMNS} FROM sessions \
                     WHERE agent_id = ?1 ORDER BY id DESC LIMIT 1"
                ),
                params![agent_id],
            )
            .await
            .ok()?;
        rows.first().and_then(|row| {
            let role_str: String = row.get(COL_SM_ROLE).ok()?;
            let content: String = row.get(COL_SM_CONTENT).ok()?;
            Some((role_str.parse::<ChatRole>().ok()?, content))
        })
    }

    /// Retrieve stored session context for a given agent ID.
    /// Returns `None` if the session has no metadata or the context columns
    /// are null.
    async fn get_session_context(&self, agent_id: &str) -> Option<SessionContext> {
        let rows = self
            .conn
            .query(
                "SELECT channel, user_name, workspace_name, role, custom_role, custom_identifier, \
                 reply_to FROM session_metadata WHERE agent_id = ?1",
                params![agent_id],
            )
            .await
            .ok()?;
        rows.first().and_then(|row| {
            let channel: Option<String> = row.get(0).ok();
            let user_name: Option<String> = row.get(1).ok();
            let workspace_name: Option<String> = row.get(2).ok();
            let role: Option<String> = row.get(3).ok();
            let custom_role: Option<String> = row.get(4).ok().flatten();
            let custom_identifier: Option<String> = row.get(5).ok().flatten();
            let reply_to = row
                .get::<Option<String>>(6)
                .ok()
                .flatten()
                .and_then(|json| serde_json::from_str(&json).ok());
            Some(SessionContext {
                channel: channel?,
                user_name: user_name?,
                workspace_name: workspace_name?,
                role: role?,
                custom_role,
                custom_identifier,
                reply_to,
            })
        })
    }

    /// Upsert one `session_metadata` column for `agent_id`; `last_activity` is
    /// stamped only when the row is first created (the TTL key), never on an
    /// existing row.
    async fn set_metadata_value(
        &self,
        agent_id: &str,
        column: MetadataColumn,
        value: Value,
    ) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        set_metadata_value_in_tx(&tx, agent_id, column, value).await?;
        tx.commit().await?;
        Ok(())
    }

    /// Run one `agent_id`-keyed read that keeps the three-way contract — `None`
    /// is a missing row, `Some(None)` a row without the value, `Some(Some(_))` a
    /// value — and fails open: a read error is logged with `warn_context` and
    /// reads as `None`, so a caller that establishes missing state retries it
    /// rather than acting on a half-read value.
    async fn read_optional_row<X>(
        &self,
        sql: &str,
        agent_id: &str,
        warn_context: &str,
        map: impl FnOnce(&Row) -> std::result::Result<Option<X>, ::turso::Error> + Send + 'static,
    ) -> Option<Option<X>>
    where
        X: Send + 'static,
    {
        match self.conn.query_optional(sql, params![agent_id], map).await {
            Ok(value) => value,
            Err(e) => {
                tracing::warn!(agent_id = %agent_id, error = %e, "{warn_context}");
                None
            }
        }
    }

    /// Read one `session_metadata` column via `map`. `None` if the row is
    /// missing, the column is NULL, or the read fails (logged via `warn_context`).
    async fn get_metadata_value<X>(
        &self,
        agent_id: &str,
        column: MetadataColumn,
        warn_context: &str,
        map: impl FnOnce(&Row) -> std::result::Result<Option<X>, ::turso::Error> + Send + 'static,
    ) -> Option<X>
    where
        X: Send + 'static,
    {
        let sql = format!(
            "SELECT {} FROM session_metadata WHERE agent_id = ?1",
            column.as_str()
        );
        self.read_optional_row(&sql, agent_id, warn_context, map)
            .await
            .flatten()
    }

    /// Persist the `<active-models-opts>` snapshot (rendered model ids) for
    /// mid-session change detection; `None` clears the baseline (no block
    /// rendered — fail-open). Upserts so a missing metadata row (e.g. a
    /// session without a preceding message append) still records the baseline.
    async fn set_active_models(&self, agent_id: &str, snapshot: Option<&str>) -> Result<()> {
        self.set_metadata_value(agent_id, MetadataColumn::ActiveModels, snapshot.into())
            .await
    }

    /// Read the last persisted `<active-models-opts>` snapshot, if any.
    /// Returns `None` when no baseline exists (no block rendered, or a
    /// session started before this feature).
    async fn get_active_models(&self, agent_id: &str) -> Option<String> {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::ActiveModels,
            "Failed to read active-models snapshot",
            |row| row.get::<Option<String>>(0),
        )
        .await
    }

    /// Record the user-activity state the admin Assistant was last shown, so the
    /// next message reports a change against what was actually seen. Upserts, like
    /// the snapshot above.
    pub(crate) async fn set_user_activity(&self, agent_id: &str, location: &str) -> Result<()> {
        self.set_metadata_value(agent_id, MetadataColumn::UserActivity, location.into())
            .await
    }

    /// The user-activity state the admin Assistant was last shown; `None` when the
    /// session has none recorded yet (nothing shown, a read error, or no session
    /// row) — the next known state is then set silently instead of being reported.
    pub(crate) async fn user_activity(&self, agent_id: &str) -> Option<String> {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::UserActivity,
            "Failed to read the user-activity reference",
            |row| row.get::<Option<String>>(0),
        )
        .await
    }

    /// Persist the personal-workspace change baseline (JSON: relative path →
    /// size/mtime/content-digest) for the Assistant's session.
    ///
    /// The baseline is a row of its own table, a child of the session's own
    /// `session_metadata` row (`ON DELETE CASCADE`), split out by migration `62`
    /// so a bookkeeping update of that row no longer rewrites it. The update
    /// branch runs first and the insert is reached only when the session has no
    /// baseline yet; the engine rejects the insert when the session row is gone
    /// (a `/clear` landing between the refresh's read and this write), so a
    /// deleted session's identifier can never be handed a stale baseline.
    pub(crate) async fn update_personal_changes(&self, agent_id: &str, state: &str) -> Result<()> {
        let tx = self.conn.begin_tx().await?;
        tx.upsert_row(
            "UPDATE session_personal_changes SET baseline = ?1 WHERE agent_id = ?2",
            || params![state, agent_id],
            "INSERT INTO session_personal_changes (agent_id, baseline) VALUES (?1, ?2) \
             ON CONFLICT(agent_id) DO NOTHING",
            params![agent_id, state],
        )
        .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Read the personal-workspace change baseline (JSON: relative path →
    /// size/mtime/content-digest) for the Assistant's session.
    ///
    /// `None` means there is no session row at all — nothing could be recorded
    /// for it, so a refresh must not walk the folder. `Some(None)` is a session
    /// row whose baseline is not recorded yet — a new session, or one whose
    /// baseline the `/clear` race never wrote — and `Some(Some(…))` a stored
    /// one; both mean "no block to render". The session's own row is what the
    /// join is anchored on precisely so the two cases stay distinguishable. A
    /// read error reads as `None` after a warning, failing the block open.
    pub(crate) async fn personal_changes_baseline(&self, agent_id: &str) -> Option<Option<String>> {
        self.read_optional_row(
            "SELECT spc.baseline FROM session_metadata sm \
             LEFT JOIN session_personal_changes spc ON spc.agent_id = sm.agent_id \
             WHERE sm.agent_id = ?1",
            agent_id,
            "Failed to read personal-changes baseline",
            |row| row.get::<Option<String>>(0),
        )
        .await
    }

    /// Persist the real provider-reported session length (input + output
    /// tokens of the last successful agent LLM call). `None` clears the
    /// value — the column stays empty for sessions that never recorded usage
    /// (approved no-backfill semantics for pre-migration sessions). Upserts
    /// so a missing metadata row still records the length.
    pub(crate) async fn set_token_length(
        &self,
        agent_id: &str,
        token_length: Option<u64>,
    ) -> Result<()> {
        // turso binds integers as i64 — token counts are far below i64::MAX.
        let bound: Option<i64> = token_length.map(i64::try_from).transpose()?;
        self.set_metadata_value(agent_id, MetadataColumn::TokenLength, bound.into())
            .await
    }

    /// Read the last persisted provider-reported session length, if any.
    /// `None` when the session never recorded a successful usage-bearing
    /// agent call (new sessions, pre-migration sessions — approved
    /// no-backfill semantics) or the value was explicitly cleared.
    async fn get_token_length(&self, agent_id: &str) -> Option<u64> {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::TokenLength,
            "Failed to read session token length",
            |row| {
                row.get::<Option<i64>>(0)
                    .map(|opt| opt.and_then(|v| u64::try_from(v).ok()))
            },
        )
        .await
    }

    /// Persist the sleep-ended marker (`"1"`) or clear it (`"0"`) for a session
    /// that ended its turn via the `sleep` tool. Read by the dead-session
    /// recovery poller to exclude intentional turn-ends.
    pub(crate) async fn set_sleep_ended(&self, agent_id: &str, ended: bool) -> Result<()> {
        let flag = if ended { "1" } else { "0" };
        self.set_metadata_value(
            agent_id,
            MetadataColumn::SleepEnded,
            db::Value::Text(flag.into()),
        )
        .await
    }

    /// Whether the session's last turn ended via the `sleep` tool.
    pub(crate) async fn get_sleep_ended(&self, agent_id: &str) -> bool {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::SleepEnded,
            "Failed to read sleep-ended flag",
            |row| row.get::<Option<String>>(0),
        )
        .await
        .is_some_and(|flag| flag == "1")
    }

    /// Record this session's branch kind and address — a custom role's name and
    /// the named branch it continues, or a worker branch's `None` role name and
    /// its branch name — plus the conversation that asked for it, so the session
    /// can be resumed as the same branch without parsing the agent ID. All three
    /// columns are written in ONE upsert, on every appending turn of a branch
    /// session, so a single failed write cannot leave the branch without a
    /// recorded role.
    ///
    /// `reply_to` is the asking conversation this launch's work reports back to
    /// (see [`crate::Launch::reply_to`]); it is stored as JSON, `None` as NULL.
    pub(crate) async fn set_branch(
        &self,
        agent_id: &str,
        role_name: Option<&str>,
        identifier: Option<&str>,
        reply_to: Option<&crate::agent::message_router::Asker>,
    ) -> Result<()> {
        let now = db::now();
        let created_at = now.clone();
        let role_col = MetadataColumn::CustomRole.as_str();
        let id_col = MetadataColumn::CustomIdentifier.as_str();
        let reply_col = MetadataColumn::ReplyTo.as_str();
        let reply_json = reply_to
            .map(serde_json::to_string)
            .transpose()
            .context("Failed to encode the branch's reply-to")?;
        self.conn
            .upsert_row(
                &format!(
                    "UPDATE session_metadata SET {role_col} = ?1, {id_col} = ?2, {reply_col} = ?3 \
                     WHERE agent_id = ?4"
                ),
                || params![role_name, identifier, reply_json.clone(), agent_id],
                &format!(
                    "INSERT INTO session_metadata \
                     (agent_id, last_activity, {role_col}, {id_col}, {reply_col}, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
                     ON CONFLICT(agent_id) DO NOTHING"
                ),
                params![
                    agent_id,
                    now.as_str(),
                    role_name,
                    identifier,
                    reply_json.clone(),
                    created_at.as_str()
                ],
            )
            .await?;
        Ok(())
    }

    /// Persist the launch-stopped marker (`"1"`) or clear it (`"0"`) for a branch
    /// session whose launch was stopped before it produced anything.
    pub(crate) async fn set_launch_stopped(&self, agent_id: &str, stopped: bool) -> Result<()> {
        let flag = if stopped { "1" } else { "0" };
        self.set_metadata_value(
            agent_id,
            MetadataColumn::LaunchStopped,
            db::Value::Text(flag.into()),
        )
        .await
    }

    /// Whether a branch session's launch was stopped before it produced
    /// anything.
    pub(crate) async fn get_launch_stopped(&self, agent_id: &str) -> bool {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::LaunchStopped,
            "Failed to read launch-stopped flag",
            |row| row.get::<Option<String>>(0),
        )
        .await
        .is_some_and(|flag| flag == "1")
    }

    /// The workspace address recorded for a session, if any — used to release a
    /// stopped branch's launch-folder engine. Best-effort: a failed
    /// read returns `None` (the engine simply stays registered).
    pub(crate) async fn session_workspace_name(&self, agent_id: &str) -> Option<String> {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::WorkspaceName,
            "Failed to read the session's recorded workspace name",
            |row| row.get::<Option<String>>(0),
        )
        .await
    }

    /// The length of the leading context block recorded for a launch session,
    /// if any — what was WRITTEN when the block was cached. `None` when the
    /// session predates the column; a failed read or a negative/garbage value
    /// also returns `None` (the caller falls back to the pre-pinning re-render).
    pub(crate) async fn context_block_len(&self, agent_id: &str) -> Option<usize> {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::ContextBlockLen,
            "Failed to read the session's recorded context-block length",
            |row| row.get::<Option<i64>>(0),
        )
        .await
        .and_then(|len| usize::try_from(len).ok())
    }

    /// The branch list of one custom role: every session (`agent_id`) whose
    /// metadata records `custom_role = name`. Used by
    /// [`crate::custom_role::delete`] to stop them all.
    pub(crate) async fn custom_role_sessions(&self, name: &str) -> Result<Vec<String>> {
        let rows = self
            .conn
            .query(
                "SELECT agent_id FROM session_metadata WHERE custom_role = ?1",
                params![name],
            )
            .await?;
        let mut agent_ids = Vec::with_capacity(rows.len());
        for row in &rows {
            agent_ids.push(row.get::<String>(0)?);
        }
        Ok(agent_ids)
    }
}

/// Inactivity window of a transient session: one silent for longer than this
/// is removed by [`cleanup_old_transient_sessions`] on the next pass.
///
/// The clock is `session_metadata.last_activity` — bumped by message traffic
/// only, so metadata-only work does not extend a session's life.
///
/// Deliberately independent of the windows around it: the stale-job purge
/// keeps [`crate::jobs::PURGE_CUTOFF_HOURS`] and the log store keeps its own,
/// and neither follows this one.
const TRANSIENT_SESSION_TTL_HOURS: i64 = 2;

/// Delete all transient (background-only) sessions inactive for longer than
/// [`TRANSIENT_SESSION_TTL_HOURS`]. Returns the number of deleted session
/// metadata rows.
///
/// Transient agent IDs start with the prefixes listed in
/// `TRANSIENT_AGENT_ID_PREFIXES`.
///
/// Both `sessions` and `session_metadata` tables are cleaned up in a single transaction.
/// The launch folders of the purged rows are released with them — this is where a
/// one-off custom-role run's search engine dies (see
/// [`crate::workspace_index::release_path_named_engine`]).
pub async fn cleanup_old_transient_sessions() -> Result<u64> {
    let cutoff = (Utc::now() - ChronoDuration::hours(TRANSIENT_SESSION_TTL_HOURS)).to_rfc3339();
    let session_store = store();
    let tx = session_store.conn.begin_tx().await?;

    let likes = TRANSIENT_AGENT_ID_PREFIXES
        .iter()
        .map(|_| "agent_id LIKE ?")
        .collect::<Vec<_>>()
        .join(" OR ");
    let prefix_patterns = format!("({likes})");

    let mut build_params = vec![Value::Text(cutoff)];
    build_params.extend(
        TRANSIENT_AGENT_ID_PREFIXES
            .iter()
            .map(|prefix| Value::Text(format!("{prefix}%"))),
    );

    // Safety-critical predicate shared by both DELETEs. The agents table IS
    // the protection marker: a session referenced by any agents row is never
    // purged. For ticket-phase jobs that protection ends when the phase purge
    // removes the job row; launched non-phase jobs are never time-deleted, so
    // their protection is indefinite by design (released by completion or
    // explicit abandon).
    let safety_predicate = format!(
        "last_activity < ? AND {prefix_patterns} \
         AND agent_id NOT IN (SELECT agent_id FROM agents)"
    );

    // The launch folders of the rows about to go: a one-off custom-role run's
    // engine is held for the run and referenced by nothing else (see
    // `workspace_index::release_path_named_engine`), and this bulk purge is the one
    // path that ends such a session without going through `delete`.
    let purged_folders: Vec<String> = tx
        .query(
            &format!("SELECT workspace_name FROM session_metadata WHERE {safety_predicate}"),
            build_params.clone(),
        )
        .await?
        .iter()
        .filter_map(|row| row.get::<Option<String>>(0).ok().flatten())
        .collect();

    tx.execute(
        &format!(
            "DELETE FROM sessions WHERE agent_id IN ( \
             SELECT agent_id FROM session_metadata WHERE {safety_predicate})"
        ),
        build_params.clone(),
    )
    .await?;

    // Delete the metadata entries themselves
    let deleted = tx
        .execute(
            &format!("DELETE FROM session_metadata WHERE {safety_predicate}"),
            build_params,
        )
        .await?;

    tx.commit().await?;

    // After the commit: the purged sessions are gone, so the launch folders they
    // named are no longer referenced by anything.
    for workspace in purged_folders {
        crate::workspace_index::release_path_named_engine(&workspace);
    }

    Ok(deleted)
}

/// The prefixes of every non-user-facing agent id: `manager_` plus every
/// [`TRANSIENT_AGENT_ID_PREFIXES`] entry.
///
/// These identify session keys a user never talks to directly (Manager /
/// transient / background). The dead-session poller excludes this whole union,
/// while [`cleanup_old_transient_sessions`] purges only the transient entries
/// (`manager_` is intentionally excluded there). Both match via SQL
/// `LIKE 'prefix_%'`, which Turso applies case-insensitively for ASCII.
pub(crate) fn non_user_facing_agent_id_prefixes() -> impl Iterator<Item = &'static str> {
    std::iter::once("manager_").chain(TRANSIENT_AGENT_ID_PREFIXES.iter().copied())
}

/// The union a **user-facing direct agent id** must never start with: every
/// non-user-facing prefix ([`non_user_facing_agent_id_prefixes`]) plus
/// [`CUSTOM_ROLE_AGENT_ID_PREFIX`] and [`WORKER_BRANCH_AGENT_ID_PREFIX`], which a
/// direct id must also not shadow.
///
/// A real user whose name collides with any of these is escaped with a `user_`
/// prefix by [`safe_user_segment`] — so a user literally named `role` becomes
/// `user_role`, protecting the start of a custom-role or worker-branch address.
pub(crate) fn reserved_agent_id_prefixes() -> impl Iterator<Item = &'static str> {
    non_user_facing_agent_id_prefixes()
        .chain([CUSTOM_ROLE_AGENT_ID_PREFIX, WORKER_BRANCH_AGENT_ID_PREFIX])
}

/// Case-insensitive (ASCII) [`str::starts_with`], matching Turso's `LIKE`
/// semantics for the reserved-prefix exclusion.
fn starts_with_ignore_ascii_case(s: &str, prefix: &str) -> bool {
    s.get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

/// Normalize a routed user identity, enforcing the never-empty-user invariant.
///
/// Returns the seeded `admin` name for an empty input, otherwise the input
/// unchanged. `context` labels the calling layer in the [`tracing::warn!`]
/// canary so a malformed `admin` fallback is traceable. This is the single
/// source of the `admin` fallback, shared by the ID chokepoint
/// ([`direct_agent_id`]) and the delivery-payload layers
/// ([`crate::agent::message_router::route_user_message`],
/// [`crate::agent::message_router::deliver_unregistered_user_response`]).
pub(crate) fn normalize_user_name<'a>(user_name: &'a str, context: &str) -> &'a str {
    if user_name.is_empty() {
        tracing::warn!(
            context = %context,
            "empty user_name — falling back to seeded 'admin'",
        );
        crate::users::ADMIN_USER_NAME
    } else {
        user_name
    }
}

/// Does a user name collide with a reserved agent-ID prefix when it becomes the
/// leading segment of a direct agent ID? Such a name would otherwise be purged by
/// the SQL exclusion `LIKE 'prefix_%'` (case-insensitive for ASCII, `_` = any one
/// char), so `ticket`, `ticketx`, and `TicketBob` all collide. Names already
/// starting with `user_` are also treated as colliding to keep [`safe_user_segment`]
/// injective.
#[must_use]
fn user_segment_collides(user_name: &str) -> bool {
    let touches_reserved = reserved_agent_id_prefixes()
        .any(|prefix| starts_with_ignore_ascii_case(user_name, prefix.trim_end_matches('_')));
    touches_reserved || starts_with_ignore_ascii_case(user_name, "user_")
}

/// Escape a user name for the leading segment of a direct agent ID by prefixing
/// colliding names ([`user_segment_collides`]) with `user_`, so their session is
/// never mistaken for a transient/background one. The escape is injective: a name
/// already starting with `user_` is ALSO escaped (`user_ticket` →
/// `user_user_ticket`), so `ticket` → `user_ticket` can never collide with a
/// distinct real user named `user_ticket`.
#[must_use]
fn safe_user_segment(user_name: &str) -> Cow<'_, str> {
    if user_segment_collides(user_name) {
        Cow::Owned(format!("user_{user_name}"))
    } else {
        Cow::Borrowed(user_name)
    }
}

/// Construct an agent ID for direct user-to-agent chat.
///
/// Format: `{user}_{ws_name}_{role}` — channel-agnostic: one conversation per
/// (user + workspace + role) regardless of the originating channel. For the
/// user's OWN personal workspace (`personal:{user}`) the duplicated user
/// segment is dropped, producing `{user}_personal:{role}` (the role is
/// `:`-delimited after the `personal` marker).
/// Role is the last segment for consistent identification in logs and
/// debugging. The role-last format is immune to underscores in user/workspace
/// names since the role is always the final `_`-delimited (or `:`-delimited for
/// a personal workspace) segment, but note that the router no longer parses
/// agent ID strings — the role is embedded directly in
/// [`AgentJob`](crate::agent::message_router::AgentJob).
/// This ID is stable across messages — the same ID is used for every message
/// in the same user/role/workspace combination, accumulating conversation
/// history within a single session.
///
/// A real user whose name collides with a reserved transient/manager prefix is
/// escaped with a `user_` prefix (see [`safe_user_segment`]) so their
/// conversation is never mistaken for a transient/background agent.
#[must_use]
fn direct_agent_id(user_name: &str, role: &str, ws_name: &str) -> String {
    // Invariant: the routed user identity is never an empty string
    // ([`normalize_user_name`]). Guard the ID itself so no bare "_ws_role" key
    // can ever be produced — the single chokepoint for every direct ID builder
    // (resolve_agent_id, envelope_target). route_user_message also normalizes
    // the delivery payload; the two layers are complementary (that one feeds
    // `job.user_name`, this one the ID key), not redundant.
    let user_name = normalize_user_name(user_name, "direct_agent_id");
    // Dedup the user for the user's OWN personal workspace (`personal:{user}`)
    // so the ID reads `{user}_personal:{role}`. Key strictly on the `personal:`
    // prefix AND the embedded user matching the routed raw user — a project
    // named `personal_work` or another user's personal workspace keeps the
    // full `{user}_{ws_name}_{role}` form. The leading segment stays escaped
    // (`safe_user_segment`), comparing the embedded user raw so reserved names
    // like `manager` yield `user_manager_personal:{role}`.
    //
    // Orphaned persisted sessions keyed `{user}_personal:support` are
    // intentionally NOT migrated or cleaned up — the Support role is gone and
    // those rows are dead weight by design.
    if ws_name.strip_prefix("personal:") == Some(user_name) {
        return format!("{}_personal:{role}", safe_user_segment(user_name));
    }
    format!("{}_{}_{}", safe_user_segment(user_name), ws_name, role)
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
pub(crate) fn manager_agent_id(ws_name: &str) -> String {
    format!("manager_{ws_name}")
}

/// Construct an agent ID for a user message, dispatching to the appropriate
/// format based on role.
///
/// - **Manager** agents use workspace-scoped IDs (`manager_{ws_name}`).
/// - **Non-Manager** agents use channel-agnostic per (user + workspace + role)
///   IDs (`{user}_{ws_name}_{role}`, or the deduped `{user}_personal:{role}`
///   for the user's own personal workspace).
///
/// This is a convenience wrapper around [`manager_agent_id`] and
/// [`direct_agent_id`] that selects the right format based on
/// whether `role` is `"manager"`.
///
/// # Parameter order
///
/// Matches [`direct_agent_id`]: `user_name` first, then `role`,
/// and `ws_name` last.
#[must_use]
pub(crate) fn resolve_agent_id(user_name: &str, role: &str, ws_name: &str) -> String {
    if role == "manager" {
        manager_agent_id(ws_name)
    } else {
        direct_agent_id(user_name, role, ws_name)
    }
}

/// The agent address of `user_name`'s Assistant session.
///
/// The single source of the owner's Assistant address — used by automation arming
/// and by notices addressed to the assistant, never re-derived ad hoc.
#[must_use]
pub(crate) fn assistant_agent_id(user_name: &str) -> String {
    resolve_agent_id(
        user_name,
        crate::Role::Assistant.as_str(),
        &crate::users::personal_workspace_name(user_name),
    )
}

/// Whether `agent_id` addresses a custom-role session: a continuable branch
/// ([`CUSTOM_ROLE_AGENT_ID_PREFIX`]) or a one-off run
/// ([`CUSTOM_ROLE_RUN_AGENT_ID_PREFIX`]).
#[must_use]
pub(crate) fn is_custom_role_agent_id(agent_id: &str) -> bool {
    agent_id.starts_with(CUSTOM_ROLE_AGENT_ID_PREFIX)
        || agent_id.starts_with(CUSTOM_ROLE_RUN_AGENT_ID_PREFIX)
}

/// Clear the session for a user/role/workspace, returning the confirmation
/// message. Fail-closed: the session's unfinished sync jobs are abandoned
/// FIRST, and a failed abandon aborts the clear so the caller can surface the
/// error — deleting the session rows despite a failed abandon would orphan the
/// jobs with no caller session left to resume them. Internal cleanup paths
/// (maintainer self-cleanup) call `store().delete` directly, and the transient
/// TTL cleanup issues a bulk SQL DELETE — all deliberately skip the abandon.
pub async fn clear_session(user_name: &str, role: &str, ws_name: &str) -> anyhow::Result<String> {
    let agent_id = resolve_agent_id(user_name, role, ws_name);
    crate::jobs::abandon_session_jobs(&agent_id).await?;
    crate::session::store().delete(&agent_id).await?;
    Ok("Session cleared. Starting fresh.".to_string())
}

/// Build a transient agent ID shared by the suffixed builder family
/// (`analyze_`, `research_`, `discovery_`).
///
/// Format: `{prefix}{ws_name}_{suffix}_{label}`. The `prefix` must carry its
/// trailing underscore (as stored in [`TRANSIENT_AGENT_ID_PREFIXES`]) so the
/// result stays byte-identical to the historical literal builders — a bare
/// prefix plus an inserted separator would emit a double underscore.
#[must_use]
fn transient_agent_id(prefix: &str, ws_name: &str, label: &str) -> String {
    let suffix = crate::generate_suffix();
    format!("{prefix}{ws_name}_{suffix}_{label}")
}

/// Construct the Maintainer agent ID for a workspace (deterministic, workspace-scoped).
///
/// Format: `maintainer_{ws_name}` — same stable-ID precedent as `manager_{ws_name}`.
/// A stable ID makes runs resumable: the next maintainer cycle (including after a
/// daemon restart) picks up the interrupted session instead of restarting from
/// scratch. Keep `maintainer_` registered in [`TRANSIENT_AGENT_ID_PREFIXES`] so the
/// transient TTL purge still applies to stale maintainer sessions.
#[must_use]
pub(crate) fn maintainer_agent_id(ws_name: &str) -> String {
    format!("maintainer_{ws_name}")
}

/// Construct an agent ID for sub-agent analyze rounds (Engineer/Maintainer → sub-agent).
///
/// Format: `analyze_{ws_name}_{suffix}_{role}`
/// Role is the LAST segment — see [`direct_agent_id`] for rationale.
#[must_use]
pub(crate) fn analyze_agent_id(ws_name: &str, role: &str) -> String {
    transient_agent_id("analyze_", ws_name, role)
}

/// Construct an agent ID for a deep-research sub-agent (decomposers,
/// round-1 researchers, gap-round analysts, verification analysts).
///
/// Format: `research_{ws_name}_{suffix}_{label}`
#[must_use]
pub(crate) fn research_agent_id(ws_name: &str, label: &str) -> String {
    transient_agent_id("research_", ws_name, label)
}

/// Construct an agent ID for workspace role discovery.
///
/// Format: `discovery_{ws_name}_{suffix}_{role}`
/// Role is the LAST segment — see [`direct_agent_id`] for rationale.
#[must_use]
pub(crate) fn discovery_agent_id(ws_name: &str, role: &str) -> String {
    transient_agent_id("discovery_", ws_name, role)
}

/// Construct the agent ID of a custom role's continuable branch.
///
/// Format: `role_{name}` for the record's default branch, or
/// `role_{name}:{identifier}` for a named branch. Deterministic, so the next
/// message on the same branch resumes the same session; the branch persists
/// like a direct conversation (see [`CUSTOM_ROLE_AGENT_ID_PREFIX`]).
#[must_use]
pub(crate) fn custom_role_agent_id(name: &str, identifier: Option<&str>) -> String {
    match identifier {
        Some(identifier) => format!("{CUSTOM_ROLE_AGENT_ID_PREFIX}{name}:{identifier}"),
        None => format!("{CUSTOM_ROLE_AGENT_ID_PREFIX}{name}"),
    }
}

/// Construct the agent ID of a custom role's ONE-OFF run.
///
/// Format: `rolerun_{name}:{suffix}` — fresh per launch and registered as
/// transient (see [`CUSTOM_ROLE_RUN_AGENT_ID_PREFIX`]).
#[must_use]
pub(crate) fn custom_role_run_agent_id(name: &str) -> String {
    format!(
        "{CUSTOM_ROLE_RUN_AGENT_ID_PREFIX}{name}:{}",
        crate::generate_suffix()
    )
}

/// Construct the agent ID of a worker branch.
///
/// Format: `worker_{branch}` — deterministic, so the next message on the same
/// branch resumes the same session; the branch persists like a direct
/// conversation (see [`WORKER_BRANCH_AGENT_ID_PREFIX`]).
#[must_use]
pub(crate) fn worker_branch_agent_id(branch: &str) -> String {
    format!("{WORKER_BRANCH_AGENT_ID_PREFIX}{branch}")
}

/// Whether `agent_id` addresses a worker branch ([`WORKER_BRANCH_AGENT_ID_PREFIX`]).
#[must_use]
pub(crate) fn is_worker_branch_agent_id(agent_id: &str) -> bool {
    agent_id.starts_with(WORKER_BRANCH_AGENT_ID_PREFIX)
}

/// The branch name inside a worker branch's agent id, when the id is one.
#[must_use]
pub(crate) fn worker_branch_name(agent_id: &str) -> Option<&str> {
    agent_id.strip_prefix(WORKER_BRANCH_AGENT_ID_PREFIX)
}

/// Whether a custom role's `name` may not be used: it IS a built-in
/// agent-address namespace word, or it begins that namespace's address family
/// (the namespace prefix including its `_`), so the role's id
/// ([`custom_role_agent_id`] / [`custom_role_run_agent_id`]) cannot read as a
/// built-in one.
///
/// The `_` is part of an address, not the SQL `LIKE` wildcard the user-name
/// escaper ([`user_segment_collides`]) must defend against: `ticket`, `role`
/// and the `ticket_*` / `role_*` families are refused, while `ticketx`,
/// `rolex` and `roleplay` shadow nothing and are allowed.
#[must_use]
pub(crate) fn custom_role_name_is_reserved(name: &str) -> bool {
    reserved_custom_role_prefixes().any(|prefix| {
        let word = prefix.trim_end_matches('_');
        name.eq_ignore_ascii_case(word) || starts_with_ignore_ascii_case(name, prefix)
    })
}

/// Every agent-id prefix a custom role name must not shadow, with its `_` — the
/// set [`custom_role_name_is_reserved`] tests and the one its refusal names.
/// Deduped: `rolerun_` is both a transient prefix and a custom-role one.
pub(crate) fn reserved_custom_role_prefixes() -> impl Iterator<Item = &'static str> {
    let mut seen = std::collections::BTreeSet::new();
    non_user_facing_agent_id_prefixes()
        .chain([CUSTOM_ROLE_AGENT_ID_PREFIX, CUSTOM_ROLE_RUN_AGENT_ID_PREFIX])
        .filter(move |prefix| seen.insert(*prefix))
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

    /// Build a freshly-initialized [`Session`] with the fixed test arguments
    /// (Assistant role, no ticket, "gui"/"tester" channel/user, no round_ts).
    /// Message may be empty (recovery-retry semantics) or a real user turn —
    /// appended via `append_turn_message` after `init` (the split mirrors
    /// `Agent::work`).
    async fn session_with_init(agent_id: &str, msg: &str, ws: &crate::Workspace) -> Session {
        let mut session = Session::default();
        session.init(agent_id).await.unwrap();
        session
            .append_turn_message(
                agent_id,
                msg,
                ws,
                &crate::Role::Assistant,
                None,
                "gui",
                "tester",
                None,
                false,
            )
            .await
            .unwrap();
        session
    }

    /// The checked readers back the person-facing GUI: a query/decode failure
    /// must be an `Err` (rendered as a failure) while the infallible readers
    /// keep their degrade-and-skip contract for non-GUI callers.
    #[tokio::test]
    async fn checked_readers_propagate_undecodable_rows() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store()
            .batch_append(&k, &[ChatMessage::user("hello")])
            .await
            .unwrap();

        // Happy paths: both checked readers return the same data as the
        // infallible ones.
        let checked = store().load_checked(&k).await.unwrap();
        assert_eq!(checked.len(), 1);
        assert_eq!(checked[0].content, "hello");
        let listed = store().list_sessions_with_metadata_checked().await.unwrap();
        assert!(listed.iter().any(|s| s.agent_id == k));

        // A row with an unknown role is undecodable: the strict reader
        // surfaces it, the infallible reader skips just that row.
        store()
            .conn
            .execute(
                "INSERT INTO sessions (agent_id, role, content, created_at) \
                 VALUES (?1, 'bogus', 'x', ?2)",
                params![k.as_str(), db::now()],
            )
            .await
            .unwrap();
        assert!(
            store().load_checked(&k).await.is_err(),
            "undecodable row must be an error for the checked reader"
        );
        assert_eq!(
            store().load(&k).await.len(),
            1,
            "the infallible reader keeps its skip-undecodable-rows contract"
        );
    }

    #[tokio::test]
    async fn session_store_rewrite_last_user_message_targets_only_user_rows() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store()
            .batch_append(
                &k,
                &[
                    ChatMessage::system("role"),
                    ChatMessage::user("[IMAGE:/tmp/a.png] earlier user"),
                    ChatMessage::user("[IMAGE:/tmp/b.png] last user"),
                    ChatMessage::assistant("last row is assistant"),
                ],
            )
            .await
            .unwrap();
        store()
            .rewrite_last_user_message(&k, "rewritten")
            .await
            .unwrap();
        let msgs = store().load(&k).await;
        assert_eq!(msgs[0].content, "role", "the system row is untouched");
        assert_eq!(
            msgs[1].content, "[IMAGE:/tmp/a.png] earlier user",
            "an earlier user row is untouched"
        );
        assert_eq!(
            msgs[2].content, "rewritten",
            "last USER row rewritten, not the trailing assistant row"
        );
        assert_eq!(msgs[3].content, "last row is assistant");
    }

    /// The baseline lives and dies with its session: a written row reads back
    /// verbatim, the session's delete cascades to it (leaving no orphan, invisible
    /// or otherwise), and a refresh racing that delete cannot insert one back —
    /// the child insert has no parent key to satisfy — so a later session reusing
    /// the identifier inherits nothing.
    #[tokio::test]
    async fn delete_takes_the_personal_changes_baseline_with_it() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store()
            .batch_append(&k, &[ChatMessage::user("a")])
            .await
            .unwrap();
        let json = r#"{"a.txt":{"size":1,"mtime_ns":1,"digest":1}}"#;
        store().update_personal_changes(&k, json).await.unwrap();
        assert_eq!(
            store().personal_changes_baseline(&k).await,
            Some(Some(json.to_string())),
            "the stored baseline must read back verbatim"
        );

        assert!(store().delete(&k).await.unwrap());
        assert_eq!(
            store().personal_changes_baseline(&k).await,
            None,
            "a deleted session has no baseline to render"
        );
        let orphans: i64 = store()
            .conn
            .query(
                "SELECT COUNT(*) FROM session_personal_changes WHERE agent_id = ?1",
                params![k.clone()],
            )
            .await
            .unwrap()[0]
            .get::<i64>(0)
            .unwrap();
        assert_eq!(
            orphans, 0,
            "the delete must cascade to the baseline row, leaving no orphan"
        );
        assert!(
            store().update_personal_changes(&k, "{}").await.is_err(),
            "a refresh racing the delete must not insert the baseline back"
        );
    }

    #[tokio::test]
    async fn session_store_truncate_from_cuts_inclusively() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store()
            .batch_append(
                &k,
                &[
                    ChatMessage::user("a"),
                    ChatMessage::assistant("b"),
                    ChatMessage::user("c"),
                    ChatMessage::assistant("d"),
                ],
            )
            .await
            .unwrap();

        assert_eq!(store().truncate_from(&k, 2).await.unwrap(), 2);
        let kept: Vec<String> = store()
            .load(&k)
            .await
            .into_iter()
            .map(|m| m.content)
            .collect();
        assert_eq!(kept, ["a", "b"]);
        let meta = store()
            .list_sessions_with_metadata()
            .await
            .into_iter()
            .find(|s| s.agent_id == k)
            .expect("session listed");
        assert_eq!(meta.message_count, 2, "count follows the deleted rows");

        // Past the end: nothing to cut, nothing deleted.
        assert_eq!(store().truncate_from(&k, 9).await.unwrap(), 0);
        assert_eq!(store().load(&k).await.len(), 2);
    }

    #[tokio::test]
    async fn clear_session_abandons_owned_sync_jobs() {
        crate::util::test::init_test_stores().await;
        let conn = &crate::session::store().conn;
        // clear_session resolves the pin internally from user/role/workspace, so
        // the sync job must be owned by that resolved agent id.
        let agent_id = resolve_agent_id("clear_user", "engineer", "clear_ws");

        crate::jobs::spawn_job(
            conn,
            "clear_sess_sync",
            "task",
            "clear_ws",
            "clear_user",
            "telegram",
            crate::Role::Engineer,
            &[],
            &crate::jobs::SpawnChild::Analyze,
            Some(&agent_id),
            None,
            None,
        )
        .await
        .unwrap();

        crate::session::clear_session("clear_user", "engineer", "clear_ws")
            .await
            .unwrap();

        let rows = conn
            .query("SELECT id FROM jobs WHERE id = 'clear_sess_sync'", ())
            .await
            .unwrap();
        assert!(
            rows.is_empty(),
            "clear_session abandoned the session's sync job"
        );
    }

    #[tokio::test]
    async fn session_get_last_message_tail() {
        crate::util::test::init_test_stores().await;
        let agent_id = unique_key();

        // Empty session → None.
        assert!(store().get_last_message_tail(&agent_id).await.is_none());

        // Append a user message → (User, "hello").
        store()
            .batch_append(&agent_id, &[ChatMessage::user("hello")])
            .await
            .unwrap();
        assert_eq!(
            store().get_last_message_tail(&agent_id).await,
            Some((ChatRole::User, "hello".into()))
        );

        // Append an assistant message → (Assistant, "world").
        store()
            .batch_append(&agent_id, &[ChatMessage::assistant("world")])
            .await
            .unwrap();
        assert_eq!(
            store().get_last_message_tail(&agent_id).await,
            Some((ChatRole::Assistant, "world".into()))
        );

        // Emission-time assistant tool-call frame: the persisted content is the
        // JSON replay payload, and the tail is the dangling-frame signal.
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            &[crate::ToolCall {
                id: "call_tail".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "x"}),
            }],
            None,
        )
        .to_string();
        store()
            .batch_append(
                &agent_id,
                &[
                    ChatMessage::user("read the file"),
                    ChatMessage::assistant(frame.clone()),
                ],
            )
            .await
            .unwrap();

        let (role, content) = store()
            .get_last_message_tail(&agent_id)
            .await
            .expect("tail should be readable");
        assert_eq!(role, ChatRole::Assistant);
        assert_eq!(content, frame);
        assert!(super::is_dangling_tool_call_frame(
            ChatRole::Assistant,
            &content
        ));
    }

    // ── Denormalized message_count consistency ─────────────────────────
    //
    // The Sessions page list reads counts from `session_metadata.message_count`
    // instead of scanning the `sessions` table. The counter must match the
    // historical COUNT(s.id) definition exactly — one row per message (system
    // prompts, tool-call frames, and tool results all count) — across the
    // append, replace, TTL-cleanup, and delete write paths.

    #[tokio::test]
    async fn message_count_append_matches_rows_and_token_length_flows() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();

        // Append with context — creates the metadata row with a count of 1.
        store()
            .append_with_context(
                &k,
                &ChatMessage::user("u1"),
                BatchMetadata::context("test_channel", "test_user", "test_ws", "engineer"),
            )
            .await
            .unwrap();
        // Batch append 2 more (an assistant answer + a tool frame).
        store()
            .batch_append(
                &k,
                &[
                    ChatMessage::assistant("a1"),
                    ChatMessage::tool_result("t1", "r1"),
                ],
            )
            .await
            .unwrap();

        let mine = store()
            .list_sessions_with_metadata()
            .await
            .into_iter()
            .find(|s| s.agent_id == k)
            .expect("session listed");
        assert_eq!(
            mine.message_count,
            store().load(&k).await.len(),
            "denormalized count must equal the session row count"
        );
        assert_eq!(mine.message_count, 3);
        // No provider-reported length recorded yet → no token value.
        assert_eq!(mine.token_length, None);

        // Once recorded, the real length flows through the list (the value
        // the Sessions card renders next to the message count).
        store().set_token_length(&k, Some(12_300)).await.unwrap();
        let mine = store()
            .list_sessions_with_metadata()
            .await
            .into_iter()
            .find(|s| s.agent_id == k)
            .expect("session listed");
        assert_eq!(mine.token_length, Some(12_300));
        // The set_token_length upsert must not disturb the count.
        assert_eq!(mine.message_count, 3);
    }

    #[tokio::test]
    async fn message_count_replace_sets_not_increments() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();

        store()
            .batch_append(
                &k,
                &[
                    ChatMessage::user("u1"),
                    ChatMessage::assistant("a1"),
                    ChatMessage::tool_result("t1", "r1"),
                ],
            )
            .await
            .unwrap();
        let count = async |agent: &str| {
            let s = store()
                .list_sessions_with_metadata()
                .await
                .into_iter()
                .find(|s| s.agent_id == agent)
                .expect("session listed");
            s.message_count
        };
        assert_eq!(count(&k).await, 3);

        // Summarization compaction: delete-then-insert. The count must be SET
        // to the new batch length (2), never incremented to 3 + 2 = 5.
        store()
            .replace_messages(
                &k,
                &[ChatMessage::system("prompt"), ChatMessage::user("u1")],
                None,
                false,
            )
            .await
            .unwrap();
        assert_eq!(count(&k).await, 2);
        assert_eq!(store().load(&k).await.len(), 2);

        // A second compaction to a larger batch also SETs.
        store()
            .replace_messages(
                &k,
                &[
                    ChatMessage::system("p"),
                    ChatMessage::user("u"),
                    ChatMessage::assistant("a"),
                    ChatMessage::assistant("a2"),
                ],
                None,
                false,
            )
            .await
            .unwrap();
        assert_eq!(count(&k).await, 4);
        assert_eq!(store().load(&k).await.len(), 4);
    }

    #[tokio::test]
    async fn message_count_ttl_cleanup_and_delete_remove_counts_with_sessions() {
        crate::util::test::init_test_stores().await;

        // Delete path (e.g. `/new`): messages and metadata go in one
        // transaction — the count vanishes with them, nothing to drift.
        let transient = format!("ticket_{}", unique_key());
        store()
            .append_with_context(
                &transient,
                &ChatMessage::user("u1"),
                BatchMetadata::context("test_channel", "test_user", "test_ws", "engineer"),
            )
            .await
            .unwrap();
        store()
            .batch_append(&transient, &[ChatMessage::assistant("a1")])
            .await
            .unwrap();
        assert!(store().delete(&transient).await.unwrap());
        assert!(
            !store()
                .list_sessions_with_metadata()
                .await
                .iter()
                .any(|s| s.agent_id == transient),
            "deleted session (and its count) must leave the list"
        );

        // TTL cleanup: a stale transient session is removed entirely — its
        // count cannot linger because the metadata row goes in the same
        // transaction as the messages.
        let stale = format!("ticket_{}", unique_key());
        store()
            .append_with_context(
                &stale,
                &ChatMessage::user("u1"),
                BatchMetadata::context("test_channel", "test_user", "test_ws", "engineer"),
            )
            .await
            .unwrap();
        store()
            .batch_append(
                &stale,
                &[ChatMessage::assistant("a1"), ChatMessage::assistant("a2")],
            )
            .await
            .unwrap();
        // Backdate past the TTL window — the literal is deliberate, so a window
        // widened back to the old eight hours fails here.
        let past = (chrono::Utc::now() - chrono::Duration::hours(3)).to_rfc3339();
        store()
            .conn
            .execute(
                "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = ?2",
                crate::db::params![past, stale.as_str()],
            )
            .await
            .unwrap();

        // A transient session still inside the window must survive: the reaper
        // removes what is older than the window, not whatever is transient.
        let fresh = format!("ticket_{}", unique_key());
        store()
            .append_with_context(
                &fresh,
                &ChatMessage::user("u1"),
                BatchMetadata::context("test_channel", "test_user", "test_ws", "engineer"),
            )
            .await
            .unwrap();

        let deleted = cleanup_old_transient_sessions().await.unwrap();
        assert!(deleted >= 1, "TTL cleanup must remove the stale session");
        let remaining = store().list_sessions_with_metadata().await;
        assert!(
            !remaining.iter().any(|s| s.agent_id == stale),
            "TTL-cleaned session (and its count) must leave the list"
        );
        assert!(
            remaining.iter().any(|s| s.agent_id == fresh),
            "a transient session inside the TTL window must survive"
        );
    }

    #[tokio::test]
    async fn session_init_loads_persisted_token_length() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();

        store().set_token_length(&k, Some(123_456)).await.unwrap();
        let ws = crate::workspace::test_ws_named("/_test_token_length", "token_length");
        let session = session_with_init(&k, "", &ws).await;
        // The persisted real length is loaded so `maybe_summarize` and the
        // Running Agents card see it from the very start of the turn.
        assert_eq!(session.token_length(), Some(123_456));
    }

    #[tokio::test]
    async fn session_list_excluding_prefixes() {
        // Use an ISOLATED store (not the shared `store()` backed by
        // `test_root()`): the shared global DB is mutated by parallel tests
        // (e.g. management/jobs purging `ticket_`/`analyze_`/`research_`
        // session metadata), which deletes this test's just-appended rows and
        // makes exact-set assertions flaky under high test-thread counts.
        let (store, _dir) = crate::open_test_store!(crate::session::SessionStore, "session");

        // Direct session plus one session per excluded prefix — the union of
        // `manager_` and [`TRANSIENT_AGENT_ID_PREFIXES`], plus the custom-role
        // continuable `role_` branch the wider reserved set adds.
        let direct_id = unique_key();
        let excluded_prefixes: Vec<&str> = reserved_agent_id_prefixes().collect();
        let prefixed_ids: Vec<String> = excluded_prefixes
            .iter()
            .map(|p| format!("{p}{}", unique_key()))
            .collect();
        // The SQL exclusion is `NOT LIKE 'prefix_%'`, and LIKE is ASCII
        // case-insensitive — so case-variants of reserved words are excluded
        // too.  These runtime-built rows keep the case-insensitivity of the
        // live SQL path covered.
        let case_variant_ids: Vec<String> = ["Manager_", "Ticket_", "tIcKeT_"]
            .iter()
            .map(|p| format!("{p}{}", unique_key()))
            .collect();

        for id in std::iter::once(&direct_id)
            .chain(prefixed_ids.iter())
            .chain(case_variant_ids.iter())
        {
            // list_sessions_with_metadata reads FROM session_metadata directly;
            // the append path upserts the row (context columns stay NULL
            // when context is None).
            store
                .append_with_context(
                    id,
                    &ChatMessage::user("msg"),
                    BatchMetadata::context("test_channel", "test_user", "test_ws", "engineer"),
                )
                .await
                .unwrap();
        }

        // Without exclusions, all sessions should be listed.
        let all = store.list_sessions_with_metadata().await;
        let all_ids: Vec<&str> = all.iter().map(|s| s.agent_id.as_str()).collect();
        assert!(
            all_ids.contains(&direct_id.as_str()),
            "direct session should be in full list"
        );
        for (prefix, id) in excluded_prefixes.iter().zip(&prefixed_ids) {
            assert!(
                all_ids.contains(&id.as_str()),
                "{prefix} session should be in full list"
            );
        }
        for id in &case_variant_ids {
            assert!(
                all_ids.contains(&id.as_str()),
                "case-variant '{id}' should be in full list"
            );
        }

        // Excluding manager_ + every transient prefix → only the direct session remains.
        let excluded = store
            .list_sessions_with_metadata_excluding(&excluded_prefixes)
            .await;
        let excluded_ids: Vec<&str> = excluded.iter().map(|s| s.agent_id.as_str()).collect();
        assert!(
            excluded_ids.contains(&direct_id.as_str()),
            "direct session should survive exclusion"
        );
        for (prefix, id) in excluded_prefixes.iter().zip(&prefixed_ids) {
            assert!(
                !excluded_ids.contains(&id.as_str()),
                "{prefix} session should be excluded"
            );
        }
        for id in &case_variant_ids {
            assert!(
                !excluded_ids.contains(&id.as_str()),
                "case-variant '{id}' should be excluded"
            );
        }
    }

    /// Empty messages are not appended by [`Session::init`].  Recovery retries
    /// pass an empty message so the agent re-runs against the existing session
    /// history without adding a new user turn.
    #[tokio::test]
    async fn session_init_empty_message_no_append() {
        crate::util::test::init_test_stores().await;
        let agent_id = unique_key();
        let ws = crate::workspace::test_ws_named("/_test_empty_session_init", "empty_test");

        // First turn: init with a real message creates the session.
        let session = session_with_init(&agent_id, "hello", &ws).await;
        let len_after_real = session.history().len();
        assert!(
            len_after_real >= 2,
            "real message should produce system prompt + user message (got {len_after_real})"
        );

        // Second turn: init with empty message should NOT append.
        let session = session_with_init(&agent_id, "", &ws).await;
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
            crate::providers::reasoning::assistant_replay_payload(
                Some("prologue"),
                std::slice::from_ref(&tc),
                None,
            )
            .to_string(),
        )
    }

    /// The retention window keeps the latest three user messages and the latest
    /// three assistant answers, drops everything older, never counts tool traffic
    /// (a tool-call frame or its result) as an answer, never re-appends the
    /// in-flight message, and reads a reasoning-only or JSON-shaped answer as an
    /// answer rather than as tool traffic.
    #[test]
    fn retention_window_keeps_the_latest_three_per_side_excluding_tool_traffic() {
        let frame = tool_call_frame();
        let mixed = vec![
            ChatMessage::system("prompt"),
            ChatMessage::user("u1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("u2"),
            frame.clone(),
            ChatMessage::tool_result("t1", "file contents"),
            ChatMessage::assistant("a2"),
            ChatMessage::user("u3"), // in-flight — newest, always retained
        ];
        let pairs: Vec<ChatMessage> = (0..5)
            .flat_map(|i| {
                vec![
                    ChatMessage::user(format!("u{i}")),
                    ChatMessage::assistant(format!("a{i}")),
                ]
            })
            .collect();
        let cases: [(Vec<ChatMessage>, &[&str]); 2] = [
            (mixed, &["u1", "a1", "u2", "a2", "u3"]),
            (pairs, &["u2", "a2", "u3", "a3", "u4", "a4"]),
        ];
        for (messages, expected) in &cases {
            let window = select_retention_window(messages);
            let contents: Vec<&str> = window.iter().map(|m| m.content.as_str()).collect();
            assert_eq!(contents, *expected, "unexpected retention window");
        }
        // In-flight message appears exactly once — no duplicate re-append.
        assert_eq!(
            select_retention_window(&cases[0].0)
                .iter()
                .filter(|m| m.content == "u3")
                .count(),
            1
        );

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
            crate::providers::reasoning::assistant_replay_payload(Some(""), &[], Some(&reasoning))
                .to_string(),
        );
        let window =
            select_retention_window(&[reasoning_msg, ChatMessage::assistant("{\"result\": 42}")]);
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn decode_native_history_message_gates_on_native_shape() {
        // Bare-JSON answer carries neither key → stays plain text (None).
        assert!(
            decode_native_history_message(&ChatMessage::assistant(r#"{"verdict": "bad"}"#))
                .is_none()
        );
        // Reasoning-only keyless JSON is not a frame — no producer emits it.
        assert!(
            decode_native_history_message(&ChatMessage::assistant(
                r#"{"reasoning_content": "think"}"#
            ))
            .is_none()
        );

        // Native frames carry a "content" key (even null) — decode to Assistant.
        let decoded =
            decode_native_history_message(&ChatMessage::assistant(r#"{"content": null}"#));
        assert!(matches!(
            decoded,
            Some(DecodedNativeHistoryMessage::Assistant { content: None, .. })
        ));
        let decoded =
            decode_native_history_message(&ChatMessage::assistant(r#"{"content": "hi"}"#));
        assert!(matches!(
            decoded,
            Some(DecodedNativeHistoryMessage::Assistant {
                content: Some(_),
                ..
            })
        ));

        // Tool-call frame built by the producer decodes to Assistant with calls.
        let tc = ToolCall {
            id: "t1".into(),
            name: "read".into(),
            arguments: serde_json::json!({}),
        };
        let frame = crate::providers::reasoning::assistant_replay_payload(
            Some(""),
            std::slice::from_ref(&tc),
            None,
        )
        .to_string();
        let decoded = decode_native_history_message(&ChatMessage::assistant(frame));
        assert!(matches!(
            decoded,
            Some(DecodedNativeHistoryMessage::Assistant {
                tool_calls: Some(_),
                ..
            })
        ));

        // Tool-result payloads decode to ToolResult.
        let decoded = decode_native_history_message(&ChatMessage::tool_result("t1", "r"));
        assert!(matches!(
            decoded,
            Some(DecodedNativeHistoryMessage::ToolResult {
                tool_call_id,
                content
            }) if tool_call_id == "t1" && content == "r"
        ));
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
        let mut session = session_with_init(&k, "", &ws).await;

        // No new assistant output this turn → the persisted trailing answer
        // must not be re-appended; the empty-tail no-op is reported.
        assert_eq!(
            session.finalize(&k).await.unwrap(),
            FinalizeOutcome::NoUnpersistedTail
        );
        assert_eq!(store().load(&k).await.len(), 3);

        // A genuinely new answer IS appended.
        session.push_assistant("a2".to_string());
        assert_eq!(
            session.finalize(&k).await.unwrap(),
            FinalizeOutcome::Flushed
        );
        let msgs = store().load(&k).await;
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[3].content, "a2");
    }

    /// A failed incoming-message persist queues the gap; the next successful
    /// persist (tool round) flushes it ahead of its own batch — no loss, no
    /// duplicate frames — and an aborted turn (no final answer) still flushes it
    /// at the turn end, so delivered comments survive in the DB for a recovery
    /// retry.
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
        let mut session = session_with_init(&k, "", &ws).await;

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

        // The same gap on a turn that ends without a final answer: the turn-end
        // flush still lands it.
        let aborted = unique_key();
        store()
            .batch_append(
                &aborted,
                &[ChatMessage::system("prompt"), ChatMessage::user("u1")],
            )
            .await
            .unwrap();
        let mut session = session_with_init(&aborted, "", &ws).await;
        session.push_messages_unpersisted(&[ChatMessage::user("C1")]);
        session.finalize(&aborted).await.unwrap();

        let contents: Vec<String> = store()
            .load(&aborted)
            .await
            .into_iter()
            .map(|m| m.content)
            .collect();
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
// `direct_agent_id()` and `manager_agent_id()` patterns. Direct IDs start with
// the user segment, so a user name that collides with a reserved prefix
// (`manager_` or any transient prefix) is escaped by [`safe_user_segment`].
// `reverse_transient_builders_use_registered_prefixes` covers all transient
// builders (one per prefix in TRANSIENT_AGENT_ID_PREFIXES). If a new transient
// role adds an agent ID builder, add it to the reverse test.
//
// All builders are pure string functions — these are cheap synchronous tests.
// Assertion `Fix:` messages guide corrective action when an invariant breaks.

#[cfg(test)]
mod transient_prefix_tests {
    use super::*;

    #[test]
    fn forward_no_collision_with_user_facing_agent_ids() {
        // Every reserved prefix (manager_ plus all transient prefixes) is a
        // possible leading segment of a direct agent ID. A real user whose
        // name collides with a reserved prefix is escaped with a `user_`
        // prefix by `safe_user_segment`, so their conversation is never
        // mistaken for a transient/background agent.
        let reserved_prefixes: Vec<&str> = reserved_agent_id_prefixes().collect();

        for prefix in &reserved_prefixes {
            let bare_word = prefix.trim_end_matches('_');

            // Capitalized variant (uppercase only the first char) covers the
            // ASCII case-insensitivity of Turso's SQLite-compatible `LIKE`,
            // e.g. `Ticket`, `Manager`.
            let capitalized = {
                let mut chars = bare_word.chars();
                match chars.next() {
                    Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                    None => bare_word.to_string(),
                }
            };

            // Forms 1–5: bare word, underscore separator, the no-underscore
            // SQL-LIKE gap (e.g. "ticketx"), a capitalized variant, and a
            // capitalized+suffixed variant. Each must be escaped by
            // `safe_user_segment` so the produced ID never starts
            // (case-insensitively) with the bare reserved word.
            let user_names: [String; 5] = [
                bare_word.to_string(),
                format!("{bare_word}_bob"),
                format!("{bare_word}x"),
                capitalized.clone(),
                format!("{capitalized}Bob"),
            ];
            for user_name in &user_names {
                let key = direct_agent_id(user_name, "analyst", "test-ws");
                assert!(
                    !starts_with_ignore_ascii_case(&key, bare_word),
                    "DIRECT AGENT ID COLLISION: bare-word='{bare_word}' \
                     (case-insensitive) matches id='{key}' (user='{user_name}'). \
                     Fix: safe_user_segment must escape any user name starting \
                     (case-insensitively) with '{bare_word}'.",
                );
            }

            // Form 6 (injective escape): a real user whose name already begins
            // with `user_` is escaped AGAIN, so `user_{bare_word}` (the escape
            // of the bare reserved word) can never collide with a distinct
            // real user literally named `user_{bare_word}`.
            let user_word = format!("user_{bare_word}");
            let escaped_key = direct_agent_id(&user_word, "analyst", "test-ws");
            assert!(
                escaped_key.starts_with("user_user_"),
                "real user '{user_word}' should be double-escaped to start \
                 with 'user_user_', got '{escaped_key}'",
            );
            assert_ne!(
                escaped_key,
                direct_agent_id(bare_word, "analyst", "test-ws"),
                "escape of '{bare_word}' collides with escape of '{user_word}'",
            );
        }

        // Manager uses a separate ID format (manager_{ws_name}) — it must
        // never collide with a transient prefix.
        for prefix in TRANSIENT_AGENT_ID_PREFIXES {
            let manager_key = manager_agent_id("test-ws");
            assert!(
                !manager_key.starts_with(prefix),
                "MANAGER AGENT ID COLLISION: prefix='{prefix}' \
                 matches id='{manager_key}'. \
                 Fix: remove '{prefix}' from TRANSIENT_AGENT_ID_PREFIXES \
                 or change the manager_agent_id pattern.",
            );
        }

        // A normal user is NOT escaped — the id keeps its raw user segment.
        assert_eq!(
            direct_agent_id("alice", "analyst", "test-ws"),
            "alice_test-ws_analyst",
        );

        // A bare reserved word IS escaped with a `user_` prefix so its session
        // is never mistaken for the manager/transient session with the same
        // prefix.
        for prefix in &reserved_prefixes {
            let bare_word = prefix.trim_end_matches('_');
            let key = direct_agent_id(bare_word, "analyst", "test-ws");
            assert!(
                key.starts_with("user_"),
                "bare reserved word '{bare_word}' should be escaped to start \
                 with 'user_', got '{key}'",
            );
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
            &analyze_agent_id("ws", "worker"),
            "analyze_",
            "analyze_agent_id('ws', 'worker')",
        );
        assert_transient_key(
            &research_agent_id("ws", "decomposer"),
            "research_",
            "research_agent_id('ws', 'decomposer')",
        );
        // The research-cleanup Sanitation agent id is built by the shared
        // `cleanup_agent_id` builder in research_cleanup.rs (used by both the
        // fresh dispatch and the boot-resume path) — asserting the REAL builder
        // keeps the transient-prefix invariant honest (a literal would silently
        // pass if the builder changed, leaking one session per research run).
        assert_transient_key(
            &crate::research_cleanup::cleanup_agent_id("run_abc123"),
            "cleanup_",
            "cleanup_agent_id('run_abc123')",
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
        assert_transient_key(
            &custom_role_run_agent_id("ws"),
            "rolerun_",
            "custom_role_run_agent_id('ws')",
        );
    }

    /// A one-off run's address must keep its role name recoverable: with a `_`
    /// separator, a role named `a_b` and the NanoID suffix's own `_` would make
    /// role `a_b` + suffix `c` and role `a` + suffix `b_c` one identical id.
    #[test]
    fn one_off_run_address_separates_the_name_from_its_suffix() {
        let id = custom_role_run_agent_id("a_b");
        let (name, suffix) = id
            .strip_prefix(CUSTOM_ROLE_RUN_AGENT_ID_PREFIX)
            .expect("the one-off prefix")
            .split_once(':')
            .expect("a `:` between the name and the generated suffix");
        assert_eq!(name, "a_b");
        assert!(!suffix.is_empty(), "the suffix must survive the separator");
    }

    /// Dispatch is on the role string, matched exactly as [`crate::Role`]
    /// spells it: lowercase `manager` addresses the workspace's Manager, while
    /// every other role — a capital-M `Manager` included — is a direct,
    /// channel-agnostic id with the role as its LAST segment.
    #[test]
    fn resolve_agent_id_dispatches_on_the_lowercase_manager_role() {
        let cases = [
            ("alice", "manager", "my-workspace", "manager_my-workspace"),
            (
                "bob",
                "engineer",
                "my-workspace",
                "bob_my-workspace_engineer",
            ),
            ("carol", "Manager", "ws", "carol_ws_Manager"),
        ];
        for (user, role, ws, expected) in cases {
            assert_eq!(resolve_agent_id(user, role, ws), expected);
        }
        assert_ne!(
            resolve_agent_id("carol", "Manager", "ws"),
            "manager_ws",
            "capital-M 'Manager' must not address the Manager"
        );
    }

    #[test]
    fn direct_agent_id_personal_workspace_dedup() {
        // A user's OWN personal workspace (`personal:{user}`) dedups the
        // duplicated user segment, so role is `:`-delimited after the marker.
        assert_eq!(
            direct_agent_id("alice", "assistant", "personal:alice"),
            "alice_personal:assistant",
        );

        // Another user's personal workspace keeps the full form.
        assert_eq!(
            direct_agent_id("bob", "assistant", "personal:admin"),
            "bob_personal:admin_assistant",
        );

        // A project named `personal_work` (no `personal:` prefix) is NOT
        // deduped.
        assert_eq!(
            direct_agent_id("alice", "analyst", "personal_work"),
            "alice_personal_work_analyst",
        );

        // A reserved-name user still gets the `user_` escape on the leading
        // segment while the embedded user is compared raw — so the ID stays
        // collision-safe yet deduped.
        assert_eq!(
            direct_agent_id("manager", "engineer", "personal:manager"),
            "user_manager_personal:engineer",
        );

        // A user already starting with `user_` is injectively double-escaped.
        assert_eq!(
            direct_agent_id("user_ticket", "analyst", "personal:user_ticket"),
            "user_user_ticket_personal:analyst",
        );
    }
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
        tool_call_id: String,
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
        // Native-frame gate: producers always emit a "content" or "tool_calls"
        // key; bare JSON answers carry neither and stay plain text.
        && (value.get("content").is_some() || value.get("tool_calls").is_some())
    {
        let content = value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

        // Extract reasoning fields for the Assistant variant.
        let (r, rc, rd) =
            crate::providers::reasoning::json_lossless_assistant_reasoning_fields(value);
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
            tool_call_id: payload.tool_call_id,
            content: payload.content,
        });
    }

    None
}
