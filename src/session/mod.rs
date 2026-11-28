//! Session persistence — Turso-backed store + native history decoding.
//!
//! Also hosts the lock-free live transcript snapshot ([`transcript`]) the Running Agents GUI
//! reads to show a running agent's in-memory conversation including the unpersisted tail.

mod activity;
mod compaction_dump;
pub(crate) mod control_token_strip;
pub mod dead_session;
pub(crate) mod image_cap;
pub(crate) mod image_strip;
mod manager;
pub(crate) mod personal_changes;
pub(crate) mod transcript;
pub(crate) mod unreadable;
pub(crate) use manager::RewriteOutcome;
pub(crate) use manager::Session;
pub(crate) use manager::SessionFlush;
pub(crate) use transcript::{TRANSCRIPT_REGISTRY, TranscriptSnapshot};

use crate::db::{self, IntoParams, Row, TxGuard, Value, params};
use crate::{ChatMessage, ChatRole, Reasoning, ToolCall, ToolResultPayload};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Duration as ChronoDuration, Utc};

// The summarization LLM call lives in `crate::Agent::summarize` so its parameters (model,
// reasoning_effort, tools, pinned provider) stay byte-identical to the work loop's.

/// Token threshold (REAL provider-reported input + output of the last successful agent call,
/// persisted in `session_metadata.token_length`) that triggers summarization. `None` — never
/// recorded (new/pre-migration sessions; no backfill) or cleared by the last compaction — disables
/// this trigger only; [`SUMMARIZATION_IMAGE_COUNT`] still applies. 200k is a conservative default
/// across the 250K–1M windows models ship with.
pub(crate) const SUMMARIZATION_THRESHOLD: u64 = 200_000;

/// Image-accumulation trigger: DISTINCT `data:image/*` markers in user messages. `>=` (unlike the
/// strict token threshold) because images are a hard floor on context quality regardless of how a
/// provider tokenizes them.
///
/// Request cap ([`image_cap`]): the pictures ONE REQUEST may carry, counted by OCCURRENCE — a
/// provider refuses a request past its own undocumented limit, so this bounds what is sent, not
/// what the conversation keeps. 20 sits well below the only limit ever observed (30).
///
/// Both count a prefix-gate superset of the markers promoted to provider vision parts: never
/// undercounts, but a malformed payload may count. An editor of one predicate must check the other.
pub(crate) const SUMMARIZATION_IMAGE_COUNT: usize = 20;

/// Latest user messages / assistant answers kept per side after compaction.
const RETENTION_PER_SIDE: usize = 3;

/// Tool calls of a native assistant history frame — the calls-only view of
/// [`decode_native_history_message`]. `Some([])` (present-but-empty payload) stays distinct from
/// `None` (not an assistant frame, decode failure, no payload). Callers span retention-window
/// selection, dead-session dangling-tail detection, tool-result settling and history telemetry, so
/// a semantic change here reaches all of them.
pub(crate) fn decoded_tool_calls(msg: &ChatMessage) -> Option<Vec<ToolCall>> {
    match decode_native_history_message(msg) {
        Some(DecodedNativeHistoryMessage::Assistant { tool_calls, .. }) => tool_calls,
        _ => None,
    }
}

/// Tool-call payload frames are tool traffic — never counted toward the retention window.
/// Mirrors the app-wide history-rendering discriminator ([`decode_native_history_message`]).
fn is_tool_call_frame(msg: &ChatMessage) -> bool {
    decoded_tool_calls(msg).is_some()
}

/// Native assistant tool-call frame carrying at least one call — the dangling-tail signal for the
/// dead-session poller: a frame that is the session TAIL has no following result rows, so
/// non-empty calls mean unresolved calls ([`Session::pending_tool_frame`]).
fn is_dangling_tool_call_frame(role: ChatRole, content: &str) -> bool {
    decoded_tool_calls(&ChatMessage {
        role,
        content: content.to_string(),
    })
    .is_some_and(|calls| !calls.is_empty())
}

/// Latest [`RETENTION_PER_SIDE`] user messages and assistant answers from `history`, merged
/// chronologically, plus everything from `keep_from` on — the index the in-flight turn starts at,
/// so nothing a multi-message turn carries is summarised away before it is read (pass
/// `history.len()` when idle). Tool frames and results are excluded from both sides; the newest
/// user message is always among them, so callers must not re-append it.
#[must_use]
fn select_retention_window(history: &[ChatMessage], keep_from: usize) -> Vec<ChatMessage> {
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
    selected.extend((keep_from.min(history.len())..history.len()).map(|idx| (idx, &history[idx])));
    selected.sort_by_key(|(idx, _)| *idx);
    selected.dedup_by_key(|(idx, _)| *idx);
    selected.into_iter().map(|(_, m)| m.clone()).collect()
}

/// Render the local datetime in the user-message timestamp format.
#[must_use]
pub(crate) fn render_timestamp() -> String {
    let now = chrono::Local::now();
    format!("{} ({})", now.format("%Y-%m-%d %H:%M:%S"), now.format("%Z"))
}

/// Build a user message with a `<timestamp>` block appended. `round_ts` (pre-rendered via
/// [`render_timestamp`]) pins one value per round so parallel members share a byte-identical first
/// message; `None` stamps now (mid-round injected content).
#[must_use]
pub(crate) fn user_msg_with_ts(content: &str, round_ts: Option<&str>) -> ChatMessage {
    let ts = round_ts.map_or_else(render_timestamp, str::to_string);
    ChatMessage::user(format!("{content}\n\n<timestamp>{ts}</timestamp>"))
}

crate::define_store! {
    pub(crate) static SESSIONS: SessionStore,
    expect = "SESSIONS not initialized — call init_all_stores() first",
}

crate::columns! {
    SESSION_MESSAGE_COLUMNS [SM] {
        ROLE    => "role",
        CONTENT => "content",
    }
}

// Session list metadata (10-column SELECT). Counts come from the denormalized metadata column —
// NOT a JOIN over `sessions`: the list re-reads on every runtime tick while the Sessions page is
// open and from the dead-session poller, and scanning the message table per refresh was the
// largest repeated query in the system. Maintained in the same transaction as message writes
// (`insert_messages_in_transaction`), never backfilled — pre-metadata sessions default to 0.
// Schema: `db/migrations.rs`.
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

#[derive(Debug, Clone)]
pub(crate) struct SessionMetadata {
    pub agent_id: String,
    pub last_activity: DateTime<Utc>,
    pub message_count: usize,
    /// Real provider-reported session length (input + output tokens of the last successful agent
    /// call), if ever recorded. Older sessions are intentionally never backfilled — `None` renders
    /// no token value.
    pub token_length: Option<u64>,
    /// Session context columns (see `SessionContext`); `None` for transient/background sessions.
    pub role: Option<String>,
    pub user_name: Option<String>,
    pub workspace_name: Option<String>,
    /// The custom role this session belongs to; `None` for a worker branch (whose branch name is
    /// in `custom_identifier`).
    pub custom_role: Option<String>,
    /// The named branch this continuable session continues, when named — a custom role's
    /// identifier, or a worker branch's name.
    pub custom_identifier: Option<String>,
    /// Whether the launch was stopped before it produced anything.
    pub launch_stopped: bool,
}

/// Context stored alongside a session so the dead-session recovery poller can reconstruct an
/// [`AgentJob`](crate::agent::message_router::AgentJob) without parsing the agent ID string.
/// Written on every non-empty turn for all roles
/// ([`Session::append_turn_message`](crate::session::Session::append_turn_message)).
/// `session_metadata.role` stores the **agent role** (`"engineer"`, `"analyst"`, `"reviewer"`),
/// semantically distinct from `sessions.role`'s **message role** — same column name, different
/// tables.
#[derive(Debug, Clone)]
pub(crate) struct SessionContext {
    pub channel: String,
    pub user_name: String,
    pub workspace_name: String,
    /// Agent role (e.g. `"engineer"`, `"analyst"`, `"reviewer"`).
    pub role: String,
    /// The custom role this session belongs to; `None` for a worker branch.
    pub custom_role: Option<String>,
    /// The named branch this continuable session continues, when named — a custom role's
    /// identifier, or a worker branch's name.
    pub custom_identifier: Option<String>,
    /// The conversation that asked for this branch's launch, when one did — the asking custom
    /// role's branch and the folder its turn ran in, so a worker branch's report returns there
    /// ([`crate::Launch::reply_to`]). `None` for owner- or automation-started launches, and for
    /// rows predating the `reply_to` column.
    pub reply_to: Option<crate::agent::message_router::Asker>,
    /// The chain this branch is a step of, when it is one: the recovered turn hands its answer on
    /// exactly as the interrupted one would have. An unreadable record abandons the branch instead
    /// of running it as an ordinary delegation.
    pub chain: crate::launch::Stored,
    /// The schema this branch's answer must come back as, when its caller asked for one: a
    /// recovered turn transforms the answer exactly as the interrupted one would have.
    pub extract: Option<crate::extract::Extract>,
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

/// Row insert shared by the append and settle paths — the only two writers of `sessions` rows.
const INSERT_SESSION_ROW: &str =
    "INSERT INTO sessions (agent_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4)";

/// The session metadata a message batch writes in the SAME transaction as its rows: the describing
/// columns (only a batch that also sets them — a session's first turn) and the leading context
/// block's length (only the batch that wrote the block).
#[derive(Clone, Copy, Default)]
struct BatchMetadata<'a> {
    context: Option<(&'a str, &'a str, &'a str, &'a str)>,
    block_len: Option<usize>,
    /// Clear the recorded provider length in this same transaction — the compaction rewrite, see
    /// [`SessionStore::replace_messages`].
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

    /// The length of the leading context block this same batch wrote — `Some` only for the batch
    /// that wrote the block.
    const fn block_len(self, len: Option<usize>) -> Self {
        Self {
            block_len: len,
            ..self
        }
    }
}

/// Insert messages into `sessions` and upsert `session_metadata` within an existing transaction
/// (shared by [`SessionStore::append_messages`]).
///
/// `meta.context` set means the describing columns are written atomically with the messages,
/// closing the atomicity gap of separate context writes. `replace` picks the count semantics:
/// `false` increments by the batch, `true` SETs it — a shared increment would double-count the
/// compaction path, whose old rows are gone before the insert.
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
    // Message count = the number of `sessions` rows for this agent (system prompts, tool-call
    // frames and tool results all count — one row per message, matching the historical COUNT
    // semantics). The batch's pictures land in the SAME statement: the row must be created with
    // both counts, because an increment on a NULL `image_count` would keep it unknown forever.
    let count = i64::try_from(messages.len()).context("message batch exceeds i64")?;
    let images = i64::try_from(image_cap::count(messages)).context("image count exceeds i64")?;
    match meta.context {
        Some((channel, user_name, workspace_name, role)) => {
            // `created_at` stamps the session_metadata row only on creation; the update branch
            // below must never touch it.
            let created_at = now.clone();
            let message_clause = counter_clause(MetadataColumn::MessageCount, 3, replace);
            let image_clause = counter_clause(MetadataColumn::ImageCount, 4, replace);
            tx.upsert_row(
                &format!(
                    "UPDATE session_metadata SET last_activity = ?2, channel = ?5, \
                     user_name = ?6, workspace_name = ?7, role = ?8, {message_clause}, \
                     {image_clause} WHERE agent_id = ?1"
                ),
                || {
                    params![
                        agent_id,
                        now.as_str(),
                        count,
                        images,
                        channel,
                        user_name,
                        workspace_name,
                        role,
                    ]
                },
                "INSERT INTO session_metadata (agent_id, last_activity, message_count, image_count, \
                 channel, user_name, workspace_name, role, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) \
                 ON CONFLICT(agent_id) DO NOTHING",
                params![
                    agent_id,
                    now.as_str(),
                    count,
                    images,
                    channel,
                    user_name,
                    workspace_name,
                    role,
                    created_at.as_str(),
                ],
            )
            .await?;
        }
        None => upsert_counts(tx, agent_id, count, images, &now, replace).await?,
    }
    // A batch that wrote the leading context block records its length in the SAME transaction, so
    // a failed write can never leave a recorded length that disagrees with the block.
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
    // A compaction rewrite clears the recorded provider length in the SAME transaction: the value
    // described the history this batch replaces, and one left behind would re-fire compaction on
    // the next turn.
    if meta.clear_token_length {
        set_metadata_value_in_tx(tx, agent_id, MetadataColumn::TokenLength, Value::Null).await?;
    }
    Ok(())
}

/// One `session_metadata` counter's SET fragment: `replace=true` overwrites (the old rows were
/// already deleted, so a shared increment would double-count), `replace=false` adds. `param` is the
/// bound value's index.
fn counter_clause(column: MetadataColumn, param: usize, replace: bool) -> String {
    let col = column.as_str();
    if replace {
        format!("{col} = ?{param}")
    } else {
        format!("{col} = {col} + ?{param}")
    }
}

/// Upsert one `session_metadata` column inside a caller-owned transaction. `created_at` is stamped
/// only on row creation (never on an existing row).
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

/// Upsert both `session_metadata` counts in a caller-owned transaction: `last_activity` is stamped,
/// each column is added on append/settle (`replace=false`) or overwritten on the rewrite path
/// (`replace=true`). `created_at` is INSERT-only. Both counts are written together because the row
/// creation must carry them: a NULL `image_count` stays NULL under an increment, so a row created by
/// a message-only write would never learn its pictures.
async fn upsert_counts(
    tx: &TxGuard<'_>,
    agent_id: &str,
    messages: i64,
    images: i64,
    now: &str,
    replace: bool,
) -> Result<()> {
    let message_clause = counter_clause(MetadataColumn::MessageCount, 3, replace);
    let image_clause = counter_clause(MetadataColumn::ImageCount, 4, replace);
    tx.upsert_row(
        &format!(
            "UPDATE session_metadata SET last_activity = ?2, {message_clause}, {image_clause} \
             WHERE agent_id = ?1"
        ),
        || params![agent_id, now, messages, images],
        "INSERT INTO session_metadata (agent_id, last_activity, message_count, image_count, \
         created_at) VALUES (?1, ?2, ?3, ?4, ?5) \
         ON CONFLICT(agent_id) DO NOTHING",
        params![agent_id, now, messages, images, now],
    )
    .await?;
    Ok(())
}

/// Execute a `query_map`, logging warnings on failure and skipping unparseable rows. Returns an
/// empty [`Vec`] on query error. `agent_id` is a structured tracing field; `None` suppresses it.
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

/// The session-list SQL with an optional `WHERE` fragment — shared by the infallible and checked
/// session-list readers.
fn sessions_list_sql(where_clause: &str) -> String {
    format!(
        "SELECT {SESSION_LIST_COLUMNS} \
         FROM session_metadata sm \
         {where_clause} \
         ORDER BY sm.last_activity DESC",
    )
}

/// The session-transcript SQL for one agent, oldest first — shared by [`SessionStore::load`] and
/// [`SessionStore::load_checked`], so the two read contracts cannot diverge.
fn session_messages_sql() -> String {
    format!("SELECT {SESSION_MESSAGE_COLUMNS} FROM sessions WHERE agent_id = ?1 ORDER BY id ASC")
}

/// Decode one [`SESSION_LIST_COLUMNS`] row — shared by the infallible and checked list readers.
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

/// Decode one [`SESSION_MESSAGE_COLUMNS`] row into a [`ChatMessage`] — shared by the infallible
/// [`SessionStore::load`] and strict [`SessionStore::load_checked`] reads.
fn session_message_from_row(row: &Row) -> Result<ChatMessage> {
    Ok(ChatMessage {
        role: row
            .get::<String>(COL_SM_ROLE)?
            .parse::<ChatRole>()
            .map_err(|e| anyhow!(e))?,
        content: row.get(COL_SM_CONTENT)?,
    })
}

/// A `session_metadata` column the read/write helpers may target; the closed enum makes the
/// `format!` interpolation compile-time safe.
#[derive(Clone, Copy)]
enum MetadataColumn {
    ActiveModels,
    UserActivity,
    TokenLength,
    SleepEnded,
    CustomRole,
    CustomIdentifier,
    ReplyTo,
    Chain,
    Extract,
    LaunchStopped,
    WorkspaceName,
    ContextBlockLen,
    MessageCount,
    ImageCount,
}

impl MetadataColumn {
    fn as_str(self) -> &'static str {
        match self {
            Self::ActiveModels => "active_models",
            Self::UserActivity => "user_activity",
            Self::TokenLength => "token_length",
            Self::SleepEnded => "sleep_ended",
            Self::CustomRole => "custom_role",
            Self::CustomIdentifier => "custom_identifier",
            Self::ReplyTo => "reply_to",
            Self::Chain => "chain",
            Self::Extract => "extract",
            Self::LaunchStopped => "launch_stopped",
            Self::WorkspaceName => "workspace_name",
            Self::ContextBlockLen => "context_block_len",
            Self::MessageCount => "message_count",
            Self::ImageCount => "image_count",
        }
    }
}

/// One row of the final post-frame sequence returned by
/// [`SessionStore::settle_tool_results`] — freshly settled results, follow-up rows and previously
/// captured rows in their final order; the caller replays it verbatim into memory.
#[derive(Debug, Clone)]
pub(crate) struct SettledRow {
    pub role: String,
    pub content: String,
    pub created_at: String,
}

/// Build the final post-frame session-row sequence for a settle. The last tool-call frame's tool
/// results must appear in FRAME CALL ORDER (provider validity), so: captured tool rows are keyed
/// by decoded `tool_call_id`; for each frame call id in order take the new result else the captured
/// row; then append the `follow_up` rows (synthetic user image messages).
/// The frame-order interleaving is the contract. The only production caller runs before
/// `append_turn_message`, so the dangling frame is always the session tail and captured rows are
/// only frame sibling tool rows — each is either consumed above or dropped by the post-frame
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

    /// Strict [`Self::load`]: a query error or the first undecodable row is returned as an `Err`.
    /// Used by the Sessions transcript, which must render a failed read as an error rather than as
    /// an empty transcript. The trade is deliberate and total: one undecodable row replaces the
    /// whole read, so the readable rows are not shown either — skipping it would render a
    /// transcript that silently omits what the store holds.
    pub(crate) async fn load_checked(&self, agent_id: &str) -> Result<Vec<ChatMessage>> {
        self.conn
            .query_map_strict(
                &session_messages_sql(),
                params![agent_id],
                session_message_from_row,
            )
            .await
    }

    /// Session-non-emptiness check (resume dispatch rule): true when the agent has at least one
    /// non-empty message row. Avoids materializing the full history just to test `.is_empty()` —
    /// the engineer's accumulated session can exceed 200k tokens.
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

    /// `created_at` of the session's last assistant message row — the dangling tool-call frame
    /// whenever one exists, since the frame is persisted BEFORE its calls run (emission-time
    /// durability) and the agent cannot emit another assistant message past an unsettled frame.
    /// `None` when the session has no assistant row at all.
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

    /// Like [`batch_append`], but writes `meta` in the same transaction as the message insert,
    /// eliminating the atomicity gap between the rows and their session metadata.
    async fn batch_append_with_context(
        &self,
        agent_id: &str,
        messages: &[ChatMessage],
        meta: BatchMetadata<'_>,
    ) -> Result<()> {
        self.append_messages(agent_id, messages, false, meta).await
    }

    /// Rewrite the whole history, recording the leading context block's length in the same
    /// transaction when this rewrite wrote the block
    /// ([`crate::session::Session::refresh_context_block`]).
    /// `clear_token_length` is set by the compaction rewrite
    /// ([`crate::session::Session::apply_summary`]): the recorded length went with the history it
    /// described.
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

    /// Rewrite the content of the history row at `index` — the durable half of the history repairs
    /// ([`crate::session::image_strip`], [`crate::session::control_token_strip`]). A single-row
    /// positional UPDATE: `history[..persisted_len]` mirrors the rows contiguously in `id` order, so
    /// an index into that prefix addresses the same row here and in memory. The prefix guard lives
    /// in [`crate::session::Session::rewrite_message`].
    ///
    /// Errors when no row sits at `index` (0 rows affected), so a caller that believes it addressed
    /// one learns the write did not land.
    ///
    /// The positional addressing inherits `truncate_from`'s precedent: it is exact only while no
    /// concurrent truncation lands between detection and this UPDATE — a GUI `truncate_from` would
    /// shift the row at `index` and this rewrite would land on its neighbour.
    pub(crate) async fn rewrite_message_at(
        &self,
        agent_id: &str,
        index: usize,
        content: &str,
    ) -> Result<()> {
        let offset = i64::try_from(index).context("message index does not fit in i64")?;
        let tx = self.conn.begin_tx().await?;
        let changed = tx
            .execute(
                "UPDATE sessions SET content = ?1 WHERE agent_id = ?2 AND id = (
                    SELECT id FROM sessions WHERE agent_id = ?2 ORDER BY id ASC LIMIT 1 OFFSET ?3
                )",
                params![content, agent_id, offset],
            )
            .await?;
        tx.commit().await?;
        if changed == 0 {
            anyhow::bail!("no message row at index {index} to rewrite for agent {agent_id}");
        }
        Ok(())
    }

    /// Test-only thin wrapper: begin a transaction, run the tx-core settle, commit. Production uses
    /// [`Self::settle_tool_results_tx`] inside the agent's atomic settle+terminalize transaction.
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

    /// Settle (re-)persisted tool results directly after the session's last tool-call frame, in one
    /// transaction: rows following that frame are captured and deleted, the new results are
    /// inserted at the frame boundary in frame call order (preserving captured sibling tool rows),
    /// then the follow-up rows. `message_count` is bumped by `results.len() + follow_up.len()` —
    /// exact only because a call with a committed result is never pending, so no captured sibling
    /// is superseded; a double-settle would drift the count and is out of contract. Returns the
    /// final post-frame sequence so the caller can mirror it in memory; a no-op (warn + empty) when
    /// no tool-call frame exists. The frame-locate SELECTs run on the SAME tx as the
    /// delete/rebuild, so a crash or concurrent writer can never observe a half-settled frame. The
    /// caller owns `commit`/`rollback`.
    pub(crate) async fn settle_tool_results_tx(
        &self,
        tx: &TxGuard<'_>,
        agent_id: &str,
        results: &[(String, String)],
        follow_up: &[ChatMessage],
    ) -> Result<Vec<SettledRow>> {
        // Frame location: scan assistant rows carrying the tool-call marker newest-first and pick
        // the FIRST that structurally decodes as a tool-call frame — a later assistant text row may
        // merely contain the substring. Tool results carry "tool_call_id", never "tool_calls", so
        // the role filter already excludes them. Content is fetched in the same scan so the frame's
        // ordered calls decode without a second SELECT.
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
            // Raw column indices: `created_at` has no COL_SM_* constant (it is not part of
            // SESSION_MESSAGE_COLUMNS), so the mapping cannot use the shared column list.
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

        // Delete the rows that follow the frame (the frame itself stays); the sequence below
        // replaces them.
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

        // Counts: the captured rows were deleted and re-inserted (net zero); the new results and
        // follow-up rows add exactly `results.len() + follow_up.len()` messages, and only the
        // follow-up rows (synthetic user image messages) add pictures. A captured row the sequence
        // cannot place leaves the picture count high — the safe direction, since a high count only
        // costs the walk.
        let count = i64::try_from(results.len() + follow_up.len())
            .context("settle result count exceeds i64")?;
        let images = i64::try_from(image_cap::count(follow_up))
            .context("follow-up image count exceeds i64")?;
        upsert_counts(tx, agent_id, count, images, &now, false)
            .await
            .context("bump settle counts")?;

        Ok(sequence)
    }

    pub(crate) async fn delete(&self, agent_id: &str) -> Result<bool> {
        // Every deletion path (`/clear`, the Sessions page, the maintainer's self-cleanup, a
        // discarded pipeline session) goes through here, so this is where a launch folder's search
        // engine is released (`workspace_index::release_path_named_engine`), where the computer
        // tool's session ends (`tools::computer::end_session`), and where the conversation's
        // in-memory memories — announced failures, unreadable facts — are forgotten with its rows.
        // Read the recorded folder before the rows go; the engine is released only for a real
        // removal.
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
        crate::agent::message_router::forget_announced_media_failures(agent_id);
        crate::session::unreadable::forget(agent_id);
        if deleted > 0
            && let Some(workspace) = workspace
        {
            crate::workspace_index::release_path_named_engine(&workspace);
        }
        Ok(deleted > 0)
    }

    /// Delete the `index`-th message row (0-based in `id` order — the order [`Self::load_checked`]
    /// returns) and every later row: the Sessions page's manual truncate. The cut is inclusive; an
    /// index past the end deletes nothing.
    ///
    /// Deliberately unguarded — no coordination with a running agent. Accepted: a cut can leave an
    /// in-flight round committing tool results without their frame row, and a tail the dead-session
    /// poller treats as resumable (`last_activity` is untouched on purpose). `token_length` is left
    /// stale — a recount is not free and summarizing early is intended; the picture count is left
    /// unknown for the same reason, re-established from the loaded history on the next use.
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
            // Back to "never counted": the deleted rows may have carried pictures, and NULL makes
            // the next use establish the value from what the history actually holds.
            tx.execute(
                "UPDATE session_metadata SET image_count = NULL WHERE agent_id = ?1",
                params![agent_id],
            )
            .await?;
        }
        let removed = usize::try_from(deleted).context("removed row count exceeds usize")?;
        tx.commit().await?;
        Ok(removed)
    }

    /// The full session list, infallible by contract: whatever could be read is returned (a query
    /// error yields an empty list). Test-only — every production reader needs a failed read to be
    /// visible and uses [`Self::list_sessions_with_metadata_checked`] instead.
    #[cfg(test)]
    pub(crate) async fn list_sessions_with_metadata(&self) -> Vec<SessionMetadata> {
        query_map_collect(
            &self.conn,
            &sessions_list_sql(""),
            (),
            session_metadata_row,
            "list sessions",
            None,
        )
        .await
    }

    /// The strict full session list: a query error or the first undecodable row is returned as an
    /// `Err`, because the Sessions page must render a failed read as a failure. The trade is
    /// deliberate and total: one undecodable row replaces the whole read rather than silently
    /// shortening the list.
    pub(crate) async fn list_sessions_with_metadata_checked(&self) -> Result<Vec<SessionMetadata>> {
        self.conn
            .query_map_strict(&sessions_list_sql(""), (), session_metadata_row)
            .await
    }

    /// Like [`Self::list_sessions_with_metadata_checked`], but excludes sessions whose `agent_id`
    /// starts with any of the given prefixes (`"manager_"`, `"ticket_"`, …) via parameterised
    /// `NOT LIKE ?N` placeholders — no string interpolation into SQL.
    ///
    /// A session that is a chain step is always listed: its conversation is the step's only durable
    /// record, and a one-off role's transient `rolerun_` conversation would otherwise lose its chain
    /// entirely.
    ///
    /// Strict, unlike the plain list: the dead-session poller prunes its records against this list,
    /// so a failed read must not look like "no sessions".
    pub(crate) async fn list_sessions_with_metadata_excluding(
        &self,
        exclude_prefixes: &[&str],
    ) -> Result<Vec<SessionMetadata>> {
        if exclude_prefixes.is_empty() {
            return self.list_sessions_with_metadata_checked().await;
        }

        let excluded = exclude_prefixes
            .iter()
            .enumerate()
            .map(|(i, _)| format!("sm.agent_id NOT LIKE ?{}", i + 1))
            .collect::<Vec<_>>()
            .join(" AND ");

        let params: Vec<db::Value> = exclude_prefixes
            .iter()
            .map(|p| db::Value::Text(format!("{p}%")))
            .collect();

        self.conn
            .query_map_strict(
                &sessions_list_sql(&format!("WHERE ({excluded}) OR sm.chain IS NOT NULL")),
                params,
                session_metadata_row,
            )
            .await
    }

    /// Lightweight query: role and raw content of the session's last message row (`None` for an
    /// empty session). Powers content-aware tail classification (the dead-session poller's
    /// dangling-frame check).
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

    /// Stored session context, or `None` when the session has no metadata or the context columns
    /// are null.
    async fn get_session_context(&self, agent_id: &str) -> Option<SessionContext> {
        let rows = self
            .conn
            .query(
                "SELECT channel, user_name, workspace_name, role, custom_role, custom_identifier, \
                 reply_to, chain, extract FROM session_metadata WHERE agent_id = ?1",
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
            let chain = crate::launch::Stored::parse(row.get::<Option<String>>(7).ok().flatten());
            let extract =
                crate::extract::Extract::from_stored(row.get::<Option<String>>(8).ok().flatten());
            Some(SessionContext {
                channel: channel?,
                user_name: user_name?,
                workspace_name: workspace_name?,
                role: role?,
                custom_role,
                custom_identifier,
                reply_to,
                chain,
                extract,
            })
        })
    }

    /// Upsert one `session_metadata` column for `agent_id`; `last_activity` is stamped only when
    /// the row is first created (the TTL key), never on an existing row.
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

    /// Run one `agent_id`-keyed read keeping the three-way contract — `None` is a missing row,
    /// `Some(None)` a row without the value, `Some(Some(_))` a value — and fail open: a read error
    /// is logged with `warn_context` and reads as `None`, so a caller that establishes missing
    /// state retries it rather than acting on a half-read value.
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

    /// Read one `session_metadata` column via `map`. `None` if the row is missing, the column is
    /// NULL, or the read fails (logged via `warn_context`).
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

    /// Persist the `<active-models-opts>` snapshot (rendered model ids) for mid-session change
    /// detection; `None` clears the baseline (no block rendered — fail-open). Upserts so a missing
    /// metadata row still records the baseline.
    async fn set_active_models(&self, agent_id: &str, snapshot: Option<&str>) -> Result<()> {
        self.set_metadata_value(agent_id, MetadataColumn::ActiveModels, snapshot.into())
            .await
    }

    /// Read the last persisted `<active-models-opts>` snapshot. `None` when no baseline exists (no
    /// block rendered, or a session started before this feature).
    async fn get_active_models(&self, agent_id: &str) -> Option<String> {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::ActiveModels,
            "Failed to read active-models snapshot",
            |row| row.get::<Option<String>>(0),
        )
        .await
    }

    /// The provider this conversation holds, with the model it was chosen for. `None` when none was
    /// recorded yet (never selected, or a session predating the columns).
    ///
    /// Both columns are read by ONE query because the pair is the unit: a row carrying one of them
    /// is not a choice, and half of it must not be handed to a selection that would then treat a
    /// stale provider as current. Fail open like the single-column readers — a read error reads as
    /// "no provider", which selects again on this run.
    pub(crate) async fn get_provider(
        &self,
        agent_id: &str,
    ) -> Option<crate::providers::selection::StoredProvider> {
        self.read_optional_row(
            "SELECT provider, provider_model FROM session_metadata WHERE agent_id = ?1",
            agent_id,
            "Failed to read the session's provider",
            |row| {
                let provider = row.get::<Option<String>>(0)?;
                let model = row.get::<Option<String>>(1)?;
                Ok(provider.zip(model).map(|(provider, model)| {
                    crate::providers::selection::StoredProvider { model, provider }
                }))
            },
        )
        .await
        .flatten()
    }

    /// Record (or clear, with `None`) the conversation's provider, returning whether the write
    /// matched a `session_metadata` row — `false` means it landed nowhere, so the caller keeps its
    /// in-memory value unrecorded and writes again once the session has a row.
    ///
    /// Deliberately an UPDATE and never an upsert: an upsert would create a `session_metadata` row
    /// for a session that has no history yet (and resurrect one `/clear` removed), which the
    /// Sessions list would then decode as a conversation. [`crate::session::Session::init`]'s
    /// `image_count` note carries the same reasoning.
    pub(crate) async fn set_provider(
        &self,
        agent_id: &str,
        stored: Option<&crate::providers::selection::StoredProvider>,
    ) -> Result<bool> {
        let (provider, model) = match stored {
            Some(stored) => (
                Value::Text(stored.provider.clone()),
                Value::Text(stored.model.clone()),
            ),
            None => (Value::Null, Value::Null),
        };
        let matched = self
            .conn
            .execute(
                "UPDATE session_metadata SET provider = ?1, provider_model = ?2 WHERE agent_id = ?3",
                params![provider, model, agent_id],
            )
            .await
            .context("Failed to record the session's provider")?;
        Ok(matched > 0)
    }

    /// Record the user-activity state the admin Assistant was last shown, so the next message
    /// reports a change against what was actually seen. Upserts, like the snapshot above.
    pub(crate) async fn set_user_activity(&self, agent_id: &str, location: &str) -> Result<()> {
        self.set_metadata_value(agent_id, MetadataColumn::UserActivity, location.into())
            .await
    }

    /// The user-activity state the admin Assistant was last shown; `None` when none is recorded yet
    /// (nothing shown, a read error, or no session row) — the next known state is then set silently
    /// instead of being reported.
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
    /// The baseline lives in its own table, a child of the session's `session_metadata` row
    /// (`ON DELETE CASCADE`), so a bookkeeping update of that row no longer rewrites it. The update
    /// branch runs first, the insert only when the session has no baseline yet; the engine rejects
    /// the insert when the session row is gone (a `/clear` landing between the refresh's read and
    /// this write), so a deleted session's identifier can never be handed a stale baseline.
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
    /// size/mtime/content-digest).
    ///
    /// `None` means there is no session row at all — nothing could be recorded for it, so a refresh
    /// must not walk the folder. `Some(None)` is a session row without a baseline yet (a new
    /// session, or one whose baseline the `/clear` race never wrote) and `Some(Some(…))` a stored
    /// one; both mean "no block to render", and the join is anchored on the session's own row
    /// precisely so the two stay distinguishable. A read error reads as `None` after a warning,
    /// failing the block open.
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

    /// Persist the real provider-reported session length (input + output tokens of the last
    /// successful agent call). `None` clears the value — the column stays empty for sessions that
    /// never recorded usage (approved no-backfill semantics). Upserts so a missing metadata row
    /// still records the length.
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

    /// Read the last persisted provider-reported session length. `None` when the session never
    /// recorded a usage-bearing call (no backfill) or the value was explicitly cleared.
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

    /// Record the pictures the session's history holds (absolute, see [`crate::session::image_cap`]).
    async fn set_image_count(&self, agent_id: &str, count: usize) -> Result<()> {
        // turso binds integers as i64 — a history's picture count is far below i64::MAX.
        let bound = i64::try_from(count).context("image count exceeds i64")?;
        self.set_metadata_value(agent_id, MetadataColumn::ImageCount, bound.into())
            .await
    }

    /// The pictures the session's persisted history holds; `None` when the session predates the
    /// count and has not been used since.
    async fn get_image_count(&self, agent_id: &str) -> Option<usize> {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::ImageCount,
            "Failed to read session image count",
            |row| {
                row.get::<Option<i64>>(0)
                    .map(|opt| opt.and_then(|v| usize::try_from(v).ok()))
            },
        )
        .await
    }

    /// Persist the sleep-ended marker (`"1"`) or clear it (`"0"`) for a session that ended its turn
    /// via the `sleep` tool. The dead-session recovery poller reads it to exclude intentional
    /// turn-ends.
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

    /// Record this session's branch kind and address — a custom role's name plus the named branch
    /// it continues, or a worker branch's `None` role name plus its branch name — the conversation
    /// that asked for it, and the chain step it is running, so the session can be resumed as the
    /// same branch without parsing the agent ID. All four columns are written in ONE upsert on every
    /// appending turn of a branch session, so a single failed write cannot leave the branch without
    /// a recorded role.
    ///
    /// `reply_to` is the asking conversation this launch's work reports back to
    /// ([`crate::Launch::reply_to`]); stored as JSON, `None` as NULL. `extract` is the schema the
    /// launch's answer must come back as ([`crate::Launch::extract`]), so a turn resumed after a
    /// restart still knows what was asked of it.
    pub(crate) async fn set_branch(
        &self,
        agent_id: &str,
        role_name: Option<&str>,
        identifier: Option<&str>,
        reply_to: Option<&crate::agent::message_router::Asker>,
        chain: Option<&crate::launch::Chain>,
        extract: Option<&crate::extract::Extract>,
    ) -> Result<()> {
        let now = db::now();
        let created_at = now.clone();
        let role_col = MetadataColumn::CustomRole.as_str();
        let id_col = MetadataColumn::CustomIdentifier.as_str();
        let reply_col = MetadataColumn::ReplyTo.as_str();
        let chain_col = MetadataColumn::Chain.as_str();
        let extract_col = MetadataColumn::Extract.as_str();
        let reply_json = reply_to
            .map(serde_json::to_string)
            .transpose()
            .context("Failed to encode the branch's reply-to")?;
        let chain_json = chain
            .map(serde_json::to_string)
            .transpose()
            .context("Failed to encode the branch's chain")?;
        let extract_json = extract
            .map(serde_json::to_string)
            .transpose()
            .context("Failed to encode the branch's schema")?;
        self.conn
            .upsert_row(
                &format!(
                    "UPDATE session_metadata SET {role_col} = ?1, {id_col} = ?2, {reply_col} = ?3, \
                     {chain_col} = ?4, {extract_col} = ?5 WHERE agent_id = ?6"
                ),
                || {
                    params![
                        role_name,
                        identifier,
                        reply_json.clone(),
                        chain_json.clone(),
                        extract_json.clone(),
                        agent_id
                    ]
                },
                &format!(
                    "INSERT INTO session_metadata \
                     (agent_id, last_activity, {role_col}, {id_col}, {reply_col}, {chain_col}, \
                     {extract_col}, created_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) \
                     ON CONFLICT(agent_id) DO NOTHING"
                ),
                params![
                    agent_id,
                    now.as_str(),
                    role_name,
                    identifier,
                    reply_json.clone(),
                    chain_json.clone(),
                    extract_json.clone(),
                    created_at.as_str()
                ],
            )
            .await?;
        Ok(())
    }

    /// Clear the schema a turn asked its answer to be shaped into. The step is over once its answer
    /// is delivered, so the recorded state must not shape a later turn or a recovery that asked for
    /// none.
    pub(crate) async fn clear_extract(&self, agent_id: &str) -> Result<()> {
        self.set_metadata_value(agent_id, MetadataColumn::Extract, Value::Null)
            .await
    }

    /// Persist the launch-stopped marker (`"1"`) or clear it (`"0"`) for a branch session whose
    /// launch was stopped before it produced anything.
    pub(crate) async fn set_launch_stopped(&self, agent_id: &str, stopped: bool) -> Result<()> {
        let flag = if stopped { "1" } else { "0" };
        self.set_metadata_value(
            agent_id,
            MetadataColumn::LaunchStopped,
            db::Value::Text(flag.into()),
        )
        .await
    }

    /// Record (or clear) the chain step a conversation is running. It is the step's durable record
    /// when the step has no round of its own — a custom role's or a worker branch's — so a restart
    /// resumes the step WITH its chain instead of delivering an intermediate answer as an ordinary
    /// delegation. Stored as JSON, `None` as NULL.
    pub(crate) async fn set_chain(
        &self,
        agent_id: &str,
        chain: Option<&crate::launch::Chain>,
    ) -> Result<()> {
        let value = chain
            .map(serde_json::to_string)
            .transpose()
            .context("Failed to encode a chain step")?
            .map_or(db::Value::Null, db::Value::Text);
        self.set_metadata_value(agent_id, MetadataColumn::Chain, value)
            .await
    }

    /// Whether a conversation still exists: its session row is there. A launcher whose conversation
    /// was purged (a one-off run after its TTL) has nowhere to receive a reply.
    pub(crate) async fn session_exists(&self, agent_id: &str) -> bool {
        self.read_optional_row(
            "SELECT 1 FROM session_metadata WHERE agent_id = ?1",
            agent_id,
            "Failed to read whether a session still exists",
            |row| row.get::<Option<i64>>(0),
        )
        .await
        .is_some()
    }

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

    /// The workspace address recorded for a session, if any — used to release a stopped branch's
    /// launch-folder engine. Best-effort: a failed read returns `None` (the engine stays
    /// registered).
    pub(crate) async fn session_workspace_name(&self, agent_id: &str) -> Option<String> {
        self.get_metadata_value(
            agent_id,
            MetadataColumn::WorkspaceName,
            "Failed to read the session's recorded workspace name",
            |row| row.get::<Option<String>>(0),
        )
        .await
    }

    /// The leading context block length recorded for a launch session, if any — what was WRITTEN
    /// when the block was cached. `None` when the session predates the column; a failed read or a
    /// negative/garbage value also yields `None` (the caller falls back to the pre-pinning
    /// re-render).
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

    /// The branch list of one custom role: every session whose metadata records `custom_role =
    /// name`. Used by [`crate::custom_role::delete`] to stop them all.
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

/// Inactivity window of a transient session: one silent for longer than this is removed by
/// [`cleanup_old_transient_sessions`] on the next pass.
///
/// The clock is `session_metadata.last_activity`, bumped by message traffic only, so metadata-only
/// work does not extend a session's life. Deliberately independent of the neighbouring windows —
/// the stale-job purge keeps [`crate::jobs::PURGE_CUTOFF_HOURS`] and the log store its own.
const TRANSIENT_SESSION_TTL_HOURS: i64 = 2;

/// Delete all transient (background-only) sessions inactive for longer than
/// [`TRANSIENT_SESSION_TTL_HOURS`]. Returns the number of deleted metadata rows.
///
/// Both `sessions` and `session_metadata` are cleaned up in a single transaction, and the purged
/// rows' launch folders are released with them — this is where a one-off custom-role run's search
/// engine dies ([`crate::workspace_index::release_path_named_engine`]). This purge is the other
/// path that ends a conversation without `delete`, so it forgets the announced markers too.
pub async fn cleanup_old_transient_sessions() -> Result<u64> {
    let cutoff = (Utc::now() - ChronoDuration::hours(TRANSIENT_SESSION_TTL_HOURS)).to_rfc3339();
    let session_store = store();
    let tx = session_store.conn.begin_tx().await?;

    let likes = crate::address::TRANSIENT_AGENT_ID_PREFIXES
        .iter()
        .map(|_| "agent_id LIKE ?")
        .collect::<Vec<_>>()
        .join(" OR ");
    let prefix_patterns = format!("({likes})");

    let mut build_params = vec![Value::Text(cutoff)];
    build_params.extend(
        crate::address::TRANSIENT_AGENT_ID_PREFIXES
            .iter()
            .map(|prefix| Value::Text(format!("{prefix}%"))),
    );

    // Safety-critical predicate shared by both DELETEs. The agents table IS the protection marker:
    // a session referenced by any agents row is never purged. For ticket-phase jobs that protection
    // ends when the phase purge removes the job row; a launched non-phase job's protection ends when
    // the job ends, and a settled record's when its retention deletes the record row.
    let safety_predicate = format!(
        "last_activity < ? AND {prefix_patterns} \
         AND agent_id NOT IN (SELECT agent_id FROM agents)"
    );

    // The launch folders of the rows about to go: a one-off custom-role run's engine is held for
    // the run and referenced by nothing else (`workspace_index::release_path_named_engine`), and
    // this bulk purge is the one path that ends such a session without going through `delete`.
    // The agent id travels with it so the conversation's announcement memory is forgotten too.
    let purged: Vec<(String, Option<String>)> = tx
        .query(
            &format!(
                "SELECT agent_id, workspace_name FROM session_metadata WHERE {safety_predicate}"
            ),
            build_params.clone(),
        )
        .await?
        .iter()
        .filter_map(|row| {
            let agent_id = row.get::<String>(0).ok()?;
            let workspace = row.get::<Option<String>>(1).ok().flatten();
            Some((agent_id, workspace))
        })
        .collect();

    tx.execute(
        &format!(
            "DELETE FROM sessions WHERE agent_id IN ( \
             SELECT agent_id FROM session_metadata WHERE {safety_predicate})"
        ),
        build_params.clone(),
    )
    .await?;

    let deleted = tx
        .execute(
            &format!("DELETE FROM session_metadata WHERE {safety_predicate}"),
            build_params,
        )
        .await?;

    tx.commit().await?;

    // After the commit: the purged sessions are gone, so the launch folders they named are no
    // longer referenced by anything, and their conversations are new again.
    for (agent_id, workspace) in purged {
        crate::agent::message_router::forget_announced_media_failures(&agent_id);
        crate::session::unreadable::forget(&agent_id);
        if let Some(workspace) = workspace {
            crate::workspace_index::release_path_named_engine(&workspace);
        }
    }

    Ok(deleted)
}

/// Normalize a routed user identity, enforcing the never-empty-user invariant.
///
/// Returns the seeded `admin` name for an empty input, otherwise the input
/// unchanged. `context` labels the calling layer in the [`tracing::warn!`]
/// canary so a malformed `admin` fallback is traceable. This is the single
/// source of the `admin` fallback, shared by the ID chokepoint
/// ([`crate::address::AgentAddress::Direct`]) and the delivery-payload layers
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

/// Clear the session for a user/role/workspace, returning the confirmation message. Fail-closed:
/// the session's unfinished sync jobs are abandoned FIRST, and a failed abandon aborts the clear so
/// the caller can surface the error — deleting the rows anyway would orphan the jobs with no caller
/// session left to resume them. Internal cleanup paths (maintainer self-cleanup, the transient TTL
/// bulk DELETE) deliberately skip the abandon.
pub async fn clear_session(user_name: &str, role: &str, ws_name: &str) -> anyhow::Result<String> {
    let agent_id = crate::address::AgentAddress::routed(user_name, role, ws_name).id();
    crate::jobs::abandon_session_jobs(&agent_id).await?;
    crate::session::store().delete(&agent_id).await?;
    Ok("Session cleared. Starting fresh.".to_string())
}

/// Whether a custom role's `name` may not be used: it IS a built-in agent-address namespace word,
/// or it begins that namespace's address family (the prefix including its `_`), so the role's id
/// ([`crate::address::AgentAddress::CustomRole`]) cannot read as a built-in one.
///
/// The `_` is part of an address, not the SQL `LIKE` wildcard the user-name escaper
/// ([`crate::address::safe_user_segment`]) must defend against: `ticket`, `role` and the
/// `ticket_*` / `role_*` families are refused, while `ticketx`, `rolex` and `roleplay` shadow
/// nothing.
#[must_use]
pub(crate) fn custom_role_name_is_reserved(name: &str) -> bool {
    reserved_custom_role_prefixes().any(|prefix| {
        let word = prefix.trim_end_matches('_');
        name.eq_ignore_ascii_case(word)
            || crate::address::starts_with_ignore_ascii_case(name, prefix)
    })
}

/// Every agent-id prefix a custom role name must not shadow, with its `_` — the set
/// [`custom_role_name_is_reserved`] tests and the one its refusal names. Deduped: `rolerun_` is
/// both a transient prefix and a custom-role one.
pub(crate) fn reserved_custom_role_prefixes() -> impl Iterator<Item = &'static str> {
    let mut seen = std::collections::BTreeSet::new();
    crate::address::non_user_facing_agent_id_prefixes()
        .chain([
            crate::address::CUSTOM_ROLE_AGENT_ID_PREFIX,
            crate::address::CUSTOM_ROLE_RUN_AGENT_ID_PREFIX,
        ])
        .filter(move |prefix| seen.insert(*prefix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_ID: AtomicU32 = AtomicU32::new(0);

    fn unique_key() -> String {
        format!("s{}", TEST_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Build a freshly-initialized [`Session`] with the fixed test arguments (Assistant role, no
    /// ticket, "gui"/"tester" channel/user, no round_ts). `msg` may be empty (recovery-retry
    /// semantics) or a real user turn, appended via `append_turn_message` after `init` — the split
    /// mirrors `Agent::work`.
    async fn session_with_init(agent_id: &str, msg: &str, ws: &crate::Workspace) -> Session {
        let mut session = Session::default();
        session.init(agent_id).await.unwrap();
        session
            .append_turn_message(
                agent_id,
                msg,
                ws,
                &crate::Role::Assistant,
                &crate::work_unit::WorkUnit::Turn,
                "gui",
                "tester",
                None,
                false,
                true,
            )
            .await
            .unwrap();
        session
    }

    /// The checked readers back the person-facing GUI: a query/decode failure must be an `Err`
    /// (rendered as a failure) while the infallible readers keep their degrade-and-skip contract
    /// for non-GUI callers.
    #[tokio::test]
    async fn checked_readers_propagate_undecodable_rows() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store()
            .batch_append(&k, &[ChatMessage::user("hello")])
            .await
            .unwrap();

        // Happy paths: both checked readers return the same data as the infallible ones.
        let checked = store().load_checked(&k).await.unwrap();
        assert_eq!(checked.len(), 1);
        assert_eq!(checked[0].content, "hello");
        let listed = store().list_sessions_with_metadata_checked().await.unwrap();
        assert!(listed.iter().any(|s| s.agent_id == k));

        // A row with an unknown role is undecodable: the strict reader surfaces it, the infallible
        // reader skips just that row.
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
    async fn session_store_rewrite_message_at_targets_the_addressed_index() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        store()
            .batch_append(
                &k,
                &[
                    ChatMessage::system("role"),
                    ChatMessage::user("[IMAGE:/tmp/a.png] earlier user"),
                    ChatMessage::assistant("middle row is assistant"),
                    ChatMessage::user("[IMAGE:/tmp/b.png] last user"),
                ],
            )
            .await
            .unwrap();
        store()
            .rewrite_message_at(&k, 2, "rewritten")
            .await
            .unwrap();
        let msgs = store().load(&k).await;
        assert_eq!(msgs[0].content, "role", "the system row is untouched");
        assert_eq!(
            msgs[1].content, "[IMAGE:/tmp/a.png] earlier user",
            "a neighbouring row is untouched"
        );
        assert_eq!(
            msgs[2].content, "rewritten",
            "the ASSISTANT row at the addressed index is rewritten, not the last user row"
        );
        assert_eq!(msgs[3].content, "[IMAGE:/tmp/b.png] last user");

        // An index past the end addresses no row: 0 rows changed is an error, so a caller that
        // believes it found a target learns the write did not land.
        let err = store()
            .rewrite_message_at(&k, 9, "rewritten")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("no message row at index 9"),
            "out-of-range index errors: {err}"
        );
    }

    /// The baseline lives and dies with its session: it reads back verbatim, the session's delete
    /// cascades to it, and a refresh racing that delete cannot insert one back (the child insert
    /// has no parent key to satisfy) — so a later session reusing the identifier inherits nothing.
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
        let agent_id =
            crate::address::AgentAddress::routed("clear_user", "engineer", "clear_ws").id();

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

        assert!(store().get_last_message_tail(&agent_id).await.is_none());

        store()
            .batch_append(&agent_id, &[ChatMessage::user("hello")])
            .await
            .unwrap();
        assert_eq!(
            store().get_last_message_tail(&agent_id).await,
            Some((ChatRole::User, "hello".into()))
        );

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

    // Denormalized message_count consistency
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

        store()
            .batch_append_with_context(
                &k,
                &[ChatMessage::user("u1")],
                BatchMetadata::context("test_channel", "test_user", "test_ws", "engineer"),
            )
            .await
            .unwrap();
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
        assert_eq!(mine.token_length, None);

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

        // Summarization compaction: delete-then-insert. The count must be SET to the new batch
        // length (2), never incremented to 3 + 2 = 5.
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
    async fn image_count_follows_the_rows_and_init_establishes_an_unknown_value() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        let picture = "[IMAGE:data:image/png;base64,AAAA]";

        // Production order: the turn's init runs before any message lands. An empty session gets no
        // `session_metadata` row — the first append writes it with the count in the same
        // transaction, so an empty session never surfaces in the session list.
        let mut session = Session::default();
        session.init(&k).await.unwrap();
        assert_eq!(session.image_count(), 0);
        assert_eq!(store().get_image_count(&k).await, None);

        store()
            .batch_append(
                &k,
                &[
                    ChatMessage::user(format!("first {picture}")),
                    ChatMessage::user(format!("second {picture}")),
                ],
            )
            .await
            .unwrap();
        assert_eq!(
            store().get_image_count(&k).await,
            Some(2),
            "the batch's pictures are recorded with its rows"
        );

        let mut session = Session::default();
        session.init(&k).await.unwrap();
        assert_eq!(session.image_count(), 2);

        // A count the truncate path cleared reads as unknown and is re-established from the loaded
        // history on the next init.
        assert_eq!(store().truncate_from(&k, 1).await.unwrap(), 1);
        assert_eq!(store().get_image_count(&k).await, None);

        let mut session = Session::default();
        session.init(&k).await.unwrap();
        assert_eq!(session.image_count(), 1);
        assert_eq!(store().get_image_count(&k).await, Some(1));
    }

    #[tokio::test]
    async fn message_count_ttl_cleanup_and_delete_remove_counts_with_sessions() {
        crate::util::test::init_test_stores().await;

        // Delete path (e.g. `/new`): messages and metadata go in one transaction — the count
        // vanishes with them, nothing to drift.
        let transient = format!("ticket_{}", unique_key());
        store()
            .batch_append_with_context(
                &transient,
                &[ChatMessage::user("u1")],
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

        // TTL cleanup: a stale transient session is removed entirely — its count cannot linger
        // because the metadata row goes in the same transaction as the messages.
        let stale = format!("ticket_{}", unique_key());
        store()
            .batch_append_with_context(
                &stale,
                &[ChatMessage::user("u1")],
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
        // Backdate past the TTL window — the literal is deliberate, so a window widened back to
        // the old eight hours fails here.
        let past = (chrono::Utc::now() - chrono::Duration::hours(3)).to_rfc3339();
        store()
            .conn
            .execute(
                "UPDATE session_metadata SET last_activity = ?1 WHERE agent_id = ?2",
                crate::db::params![past, stale.as_str()],
            )
            .await
            .unwrap();

        // A transient session still inside the window must survive: the reaper removes what is
        // older than the window, not whatever is transient.
        let fresh = format!("ticket_{}", unique_key());
        store()
            .batch_append_with_context(
                &fresh,
                &[ChatMessage::user("u1")],
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
        // The persisted real length is loaded so `maybe_summarize` and the Running Agents card
        // see it from the very start of the turn.
        assert_eq!(session.token_length(), Some(123_456));
    }

    /// The provider belongs to the conversation, not to the process: a fresh [`Session`] over the same
    /// agent id loads the provider its predecessor recorded. That is the restart / resumed-run /
    /// re-dispatched-agent guarantee the whole feature rests on.
    #[tokio::test]
    async fn a_provider_round_trips_through_a_restart() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        let ws = crate::workspace::test_ws_named("/_test_stored_provider", "stored_provider");
        // A recorded provider needs the metadata row the append path creates.
        let mut session = session_with_init(&k, "hello", &ws).await;
        let stored = crate::providers::selection::StoredProvider {
            model: "claude-sonnet-4".to_string(),
            provider: "anthropic".to_string(),
        };

        session.set_provider(&k, Some(stored.clone())).await;
        assert_eq!(
            session.provider(),
            Some(&stored),
            "the provider is held in memory"
        );

        let restarted = session_with_init(&k, "", &ws).await;
        assert_eq!(
            restarted.provider(),
            Some(&stored),
            "a fresh session must read back the provider AND the model it was chosen for"
        );
    }

    /// `None` clears it on both sides: a later run then selects from the top of the list instead of
    /// holding a provider the caller gave up on.
    #[tokio::test]
    async fn clearing_the_provider_clears_it_durably() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        let ws = crate::workspace::test_ws_named("/_test_provider_clear", "provider_clear");
        let mut session = session_with_init(&k, "hello", &ws).await;
        session
            .set_provider(
                &k,
                Some(crate::providers::selection::StoredProvider {
                    model: "claude-sonnet-4".to_string(),
                    provider: "anthropic".to_string(),
                }),
            )
            .await;

        session.set_provider(&k, None).await;
        assert_eq!(
            session.provider(),
            None,
            "the in-memory provider is cleared"
        );
        assert_eq!(
            store().get_provider(&k).await,
            None,
            "both columns are cleared, so a fresh session sees no provider"
        );
        assert_eq!(session_with_init(&k, "", &ws).await.provider(), None);
    }

    /// A provider write must never create the metadata row: an upsert would give an empty session a
    /// conversation the Sessions list decodes (and resurrect one `/clear` removed). An unmatched
    /// write therefore advances nothing, so the next model call retries it once the first append
    /// has created the row.
    #[tokio::test]
    async fn an_unmatched_provider_write_records_nothing_and_is_retried() {
        crate::util::test::init_test_stores().await;
        let k = unique_key();
        let stored = crate::providers::selection::StoredProvider {
            model: "claude-sonnet-4".to_string(),
            provider: "anthropic".to_string(),
        };

        assert!(
            !store().set_provider(&k, Some(&stored)).await.unwrap(),
            "a session with no metadata row must report the write landed nowhere"
        );
        let mut session = Session::default();
        session.init(&k).await.unwrap();
        session.set_provider(&k, Some(stored.clone())).await;
        assert_eq!(
            session.provider(),
            None,
            "an unmatched write advances nothing, so the next model call writes it again"
        );

        let rows: i64 = store()
            .conn
            .query(
                "SELECT COUNT(*) FROM session_metadata WHERE agent_id = ?1",
                params![k.as_str()],
            )
            .await
            .unwrap()[0]
            .get(0)
            .unwrap();
        assert_eq!(
            rows, 0,
            "no provider write may create a session_metadata row"
        );
    }

    #[tokio::test]
    async fn session_list_excluding_prefixes() {
        // Use an ISOLATED store (not the shared `store()`): the shared global DB is mutated by
        // parallel tests (management/jobs purging `ticket_`/`analyze_`/`research_` metadata), which
        // deletes this test's just-appended rows and makes exact-set assertions flaky.
        let (store, _dir) = crate::open_test_store!(crate::session::SessionStore, "session");

        // Direct session plus one per excluded prefix — the union of `manager_` and
        // [`crate::address::TRANSIENT_AGENT_ID_PREFIXES`], plus the custom-role `role_` branch the
        // reserved set
        // adds.
        let direct_id = unique_key();
        let excluded_prefixes: Vec<&str> = crate::address::reserved_agent_id_prefixes().collect();
        let prefixed_ids: Vec<String> = excluded_prefixes
            .iter()
            .map(|p| format!("{p}{}", unique_key()))
            .collect();
        // The SQL exclusion is `NOT LIKE 'prefix_%'`, and LIKE is ASCII case-insensitive — so
        // case-variants of reserved words are excluded too. These runtime-built rows keep that
        // covered.
        let case_variant_ids: Vec<String> = ["Manager_", "Ticket_", "tIcKeT_"]
            .iter()
            .map(|p| format!("{p}{}", unique_key()))
            .collect();

        for id in std::iter::once(&direct_id)
            .chain(prefixed_ids.iter())
            .chain(case_variant_ids.iter())
        {
            // list_sessions_with_metadata reads FROM session_metadata directly; the append path
            // upserts the row.
            store
                .batch_append_with_context(
                    id,
                    &[ChatMessage::user("msg")],
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
            .await
            .unwrap();
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

    /// A chain step is the exception to the exclusion: its conversation is the step's only durable
    /// record, so a one-off role's transient `rolerun_` session must still be listed while its chain
    /// is in flight — and must not be, once it is no longer a step.
    #[tokio::test]
    async fn a_chain_step_is_a_recovery_candidate_whatever_its_prefix() {
        let (store, _dir) = crate::open_test_store!(crate::session::SessionStore, "session");
        let excluded_prefixes: Vec<&str> = crate::address::reserved_agent_id_prefixes().collect();
        let step_id = format!(
            "{}{}",
            crate::address::CUSTOM_ROLE_RUN_AGENT_ID_PREFIX,
            unique_key()
        );
        store
            .batch_append_with_context(
                &step_id,
                &[ChatMessage::user("msg")],
                BatchMetadata::context("test_channel", "test_user", "test_ws", "custom"),
            )
            .await
            .unwrap();
        store
            .set_chain(
                &step_id,
                Some(&crate::launch::Chain {
                    id: unique_key(),
                    task: "draft it".to_string(),
                    user: "test_user".to_string(),
                    channel: "test_channel".to_string(),
                    workspace: "test_ws".to_string(),
                    inherited: None,
                    current: crate::launch::Step::worker(),
                    step: 1,
                    next: Vec::new(),
                }),
            )
            .await
            .unwrap();
        let listed = |sessions: Vec<SessionMetadata>| {
            sessions
                .into_iter()
                .any(|session| session.agent_id == step_id)
        };
        assert!(
            listed(
                store
                    .list_sessions_with_metadata_excluding(&excluded_prefixes)
                    .await
                    .unwrap()
            ),
            "a chain step's conversation is a recovery candidate whatever its prefix"
        );
        store.set_chain(&step_id, None).await.unwrap();
        assert!(
            !listed(
                store
                    .list_sessions_with_metadata_excluding(&excluded_prefixes)
                    .await
                    .unwrap()
            ),
            "and a transient conversation that is no chain step stays excluded"
        );
    }

    /// The schema a branch's answer must come back as rides the conversation, so a turn resumed
    /// after a restart still knows what was asked of it. A branch's FIRST write records it too: the
    /// conversation has no row yet, and the schema must land with everything else in that one write.
    #[tokio::test]
    async fn a_branch_records_the_schema_its_answer_must_come_back_as() {
        let (store, _dir) = crate::open_test_store!(crate::session::SessionStore, "session");
        let agent_id = crate::address::AgentAddress::worker_branch(unique_key()).id();
        let extract = crate::extract::Extract::from_text(
            r#"{"type":"object","properties":{"summary":{"type":"string"}}}"#,
        )
        .expect("the schema is accepted");
        // The branch's own first write, before any message: this is the row's creation.
        store
            .set_branch(&agent_id, None, Some("night"), None, None, Some(&extract))
            .await
            .expect("the conversation records the schema");
        let rows = store
            .conn
            .query(
                "SELECT extract FROM session_metadata WHERE agent_id = ?1",
                params![agent_id.as_str()],
            )
            .await
            .expect("read the conversation's row");
        assert_eq!(
            rows.first()
                .and_then(|row| row.get::<Option<String>>(0).ok().flatten()),
            Some(extract.canonical()),
            "a row created by this write carries the schema"
        );

        // The branch runs, and the schema is still there for the resume that reads it back.
        store
            .batch_append_with_context(
                &agent_id,
                &[ChatMessage::user("draft the summary")],
                BatchMetadata::context("gui", "test_user", "test_ws", "worker"),
            )
            .await
            .unwrap();
        let ctx = store
            .get_session_context(&agent_id)
            .await
            .expect("the conversation reads back");
        assert_eq!(ctx.extract.as_ref(), Some(&extract));

        // A later turn with no schema records none, so an ordinary turn is never shaped by an
        // earlier one's.
        store
            .set_branch(&agent_id, None, Some("night"), None, None, None)
            .await
            .expect("the conversation is rewritten");
        assert!(
            store
                .get_session_context(&agent_id)
                .await
                .expect("the conversation reads back")
                .extract
                .is_none(),
            "a turn with no schema records none"
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

        let session = session_with_init(&agent_id, "hello", &ws).await;
        let len_after_real = session.history().len();
        assert!(
            len_after_real >= 2,
            "real message should produce system prompt + user message (got {len_after_real})"
        );

        let session = session_with_init(&agent_id, "", &ws).await;
        assert_eq!(
            session.history().len(),
            len_after_real,
            "empty message must not append to session history",
        );
    }

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
            let window = select_retention_window(messages, messages.len());
            let contents: Vec<&str> = window.iter().map(|m| m.content.as_str()).collect();
            assert_eq!(contents, *expected, "unexpected retention window");
        }
        assert_eq!(
            select_retention_window(&cases[0].0, cases[0].0.len())
                .iter()
                .filter(|m| m.content == "u3")
                .count(),
            1
        );

        // A turn carrying several messages keeps all of them, whatever the
        // per-side quota: nothing it carries is summarised away before it is read.
        let combined = vec![
            ChatMessage::system("prompt"),
            ChatMessage::user("u1"),
            ChatMessage::assistant("a1"),
            ChatMessage::user("turn-1"),
            ChatMessage::user("turn-2"),
            ChatMessage::user("turn-3"),
            ChatMessage::user("turn-4"),
        ];
        let turn_start = combined.len() - 4;
        let kept: Vec<String> = select_retention_window(&combined, turn_start)
            .iter()
            .map(|m| m.content.clone())
            .collect();
        assert_eq!(kept, ["a1", "turn-1", "turn-2", "turn-3", "turn-4"]);

        let window = select_retention_window(&[ChatMessage::user("u1")], 1);
        assert_eq!(window.len(), 1);
        assert_eq!(window[0].content, "u1");

        let reasoning = Reasoning {
            reasoning: Some("thinking".into()),
            reasoning_content: None,
            reasoning_details: None,
        };
        let reasoning_msg = ChatMessage::assistant(
            crate::providers::reasoning::assistant_replay_payload(Some(""), &[], Some(&reasoning))
                .to_string(),
        );
        let window = select_retention_window(
            &[reasoning_msg, ChatMessage::assistant("{\"result\": 42}")],
            2,
        );
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn decode_native_history_message_gates_on_native_shape() {
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
        // Persisted history ending with an assistant answer (e.g. a compacted session whose
        // retained window ends with one).
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

        // No new assistant output this turn → the persisted trailing answer must not be
        // re-appended; the empty-tail no-op is reported.
        assert_eq!(
            session.finalize(&k).await.unwrap(),
            SessionFlush::NoUnpersistedTail
        );
        assert_eq!(store().load(&k).await.len(), 3);

        session.push_assistant("a2".to_string());
        assert_eq!(session.finalize(&k).await.unwrap(), SessionFlush::Flushed);
        let msgs = store().load(&k).await;
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[3].content, "a2");
    }

    /// A failed incoming-message persist queues the gap; the next successful persist (tool round)
    /// flushes it ahead of its own batch — no loss, no duplicate frames — and an aborted turn (no
    /// final answer) still flushes it at the turn end, so delivered comments survive in the DB for
    /// a recovery retry.
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

        // The same gap on a turn that ends without a final answer: the turn-end flush still lands
        // it.
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

/// Decode a `ChatMessage` whose `content` is a JSON-wrapped native message. `None` when the
/// message does not look like a native/session-persisted one.
pub(crate) fn decode_native_history_message(
    message: &ChatMessage,
) -> Option<DecodedNativeHistoryMessage> {
    let parsed = serde_json::from_str::<serde_json::Value>(&message.content).ok();

    if message.role == ChatRole::Assistant
        && let Some(value) = parsed.as_ref()
        // Native-frame gate: producers always emit a "content" or "tool_calls" key; bare JSON
        // answers carry neither and stay plain text.
        && (value.get("content").is_some() || value.get("tool_calls").is_some())
    {
        let content = value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

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
