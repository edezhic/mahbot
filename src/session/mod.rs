//! Session persistence — Turso-backed store + native history decoding.

pub mod dead_session;
pub mod manager;
pub use manager::Session;

use crate::turso::{self, IntoParams, Row, TxGuard, Value, params};
use crate::{ChatMessage, ChatRole, Reasoning, ToolCall, ToolResultPayload, ToolSpec};
use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use regex::Regex;
use std::sync::LazyLock;

// The summarization LLM call lives in `crate::Agent::summarize` so that all
// parameters (model, reasoning_effort, tools, provider routing)
// are byte-identical to the agent's work loop.

/// History-length threshold (in estimated tokens) that triggers summarization.
///
/// Estimated via [`estimate_tokens`], a weighted, content-aware estimate:
/// image parts at a fixed per-image cost, base64 data-URI payloads at base64
/// density, code/JSON at code density, prose at prose density, plus the tool
/// schemas injected into every chat request (previously uncounted).
///
/// **200,000** is a conservative default chosen to work across models with
/// varying context window sizes (250K–1M). The estimator carries a slight
/// underestimate bias on prose so the first compaction of an average session
/// lands at ~200–220K real tokens rather than before the threshold.
pub(crate) const SUMMARIZATION_THRESHOLD: usize = 200_000;

/// Estimated tokens consumed by one image part (vision input cost). A single
/// constant across models — model-specific per-image pricing is out of scope.
const IMAGE_TOKENS: usize = 4_000;

/// Prose density — characters per token. The Manager's Cyrillic-prose mix
/// measured ~4.38; rounding up keeps the slight-underestimate bias.
const PROSE_CHARS_PER_TOKEN: f64 = 4.4;

/// Code/JSON density — characters per token.
const CODE_CHARS_PER_TOKEN: f64 = 2.5;

/// Base64 text density — characters per token (measured 1.3–1.5).
const BASE64_CHARS_PER_TOKEN: f64 = 1.4;

/// Per-message overhead tokens (role line, framing, separators).
const PER_MESSAGE_TOKENS: usize = 4;

/// Per-tool overhead tokens — JSON wrapper `{"type":"function","function":{...}}`
/// (≈33 chars at code density; name/description/parameters counted separately).
const PER_TOOL_SCHEMA_TOKENS: usize = 12;

/// Data-URI base64 payload: `data:<mime>;base64,<payload>`.
static DATA_URI_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"data:[a-zA-Z0-9./+-]+;base64,[A-Za-z0-9+/]+={0,2}")
        .expect("DATA_URI_RE must compile")
});

/// Weighted token estimate for a session's messages plus the tool schemas the
/// provider injects into every request. Mirrors what the provider actually
/// sends ([`crate::providers::compatible::to_message_content`]): with
/// `allow_image_parts` user `[IMAGE:...]` markers become fixed-cost image
/// parts, otherwise the raw marker stays text and its data-URI payload is
/// charged at base64 density; native assistant frames contribute content +
/// tool-call arguments + reasoning, tool results their decoded content.
#[must_use]
pub(crate) fn estimate_tokens(
    messages: &[ChatMessage],
    tools: &[ToolSpec],
    allow_image_parts: bool,
) -> usize {
    let history: usize = messages
        .iter()
        .map(|msg| estimate_message_tokens(msg, allow_image_parts))
        .sum();
    history + estimate_tool_schema_tokens(tools)
}

fn estimate_message_tokens(msg: &ChatMessage, allow_image_parts: bool) -> usize {
    let content_tokens = match decode_native_history_message(msg) {
        Some(DecodedNativeHistoryMessage::Assistant {
            content,
            tool_calls,
            reasoning,
        }) => {
            let text = content.as_deref().map_or(0, estimate_text_tokens);
            let calls =
                tool_calls.map_or(0, |calls| calls.iter().map(estimate_tool_call_tokens).sum());
            // The provider may synthesize a reasoning_content copy from
            // reasoning/details (native_reasoning_triple_for_replay) — counted
            // once here, a slight undercount within the best-effort envelope.
            let reasoning_tokens = reasoning.map_or(0, |r| {
                r.reasoning.as_deref().map_or(0, estimate_text_tokens)
                    + r.reasoning_content
                        .as_deref()
                        .map_or(0, estimate_text_tokens)
                    + r.reasoning_details.as_ref().map_or(0, |d| {
                        estimate_text_tokens(&serde_json::to_string(d).unwrap_or_default())
                    })
            });
            text + calls + reasoning_tokens
        }
        Some(DecodedNativeHistoryMessage::ToolResult { content, .. }) => {
            estimate_text_tokens(&content)
        }
        None if msg.role == ChatRole::User => {
            if allow_image_parts && msg.content.contains("[IMAGE:") {
                // Image markers become image parts at request time — charge the
                // fixed per-image cost and drop the (possibly base64) marker text.
                // Fast path mirrors to_message_content: skip the marker regex
                // scan when the message carries no image markers.
                let (cleaned, refs) =
                    crate::providers::compatible::parse_image_markers(&msg.content);
                refs.len() * IMAGE_TOKENS + estimate_text_tokens(&cleaned)
            } else {
                // Without image parts the raw marker stays text (e.g. ticket
                // comments bypassing enrichment) — count it like any other text.
                estimate_text_tokens(&msg.content)
            }
        }
        None => estimate_text_tokens(&msg.content),
    };
    content_tokens + PER_MESSAGE_TOKENS
}

/// Tool-call arguments are JSON text in the request — the provider never
/// converts tool-call fields into image parts, so data-URI references in them
/// stay text and are charged at base64 density.
fn estimate_tool_call_tokens(call: &ToolCall) -> usize {
    let args = serde_json::to_string(&call.arguments).unwrap_or_default();
    estimate_text_tokens(&call.name) + estimate_text_tokens(&args)
}

/// Tool schemas are real request tokens (`tools` field) that the old estimator
/// ignored (~5–7K for the Manager's 8-tool set).
fn estimate_tool_schema_tokens(tools: &[ToolSpec]) -> usize {
    tools
        .iter()
        .map(|t| {
            estimate_text_tokens(&t.name)
                + estimate_text_tokens(&t.description)
                + estimate_text_tokens(&serde_json::to_string(&t.parameters).unwrap_or_default())
                + PER_TOOL_SCHEMA_TOKENS
        })
        .sum()
}

/// Estimate a raw text blob: data-URI base64 payloads are charged at base64
/// density, everything else at code or prose density by structural content.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn estimate_text_tokens(text: &str) -> usize {
    let mut total = 0usize;
    let mut last = 0usize;
    for caps in DATA_URI_RE.captures_iter(text) {
        let m = caps.get(0).expect("DATA_URI_RE match carries group 0");
        total += plain_text_token_count(&text[last..m.start()]);
        let uri = m.as_str();
        let (prefix, payload) = uri
            .split_once(";base64,")
            .expect("DATA_URI_RE match always contains ;base64,");
        total += (payload.chars().count() as f64 / BASE64_CHARS_PER_TOKEN) as usize;
        total += plain_text_token_count(prefix);
        last = m.end();
    }
    total + plain_text_token_count(&text[last..])
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn plain_text_token_count(text: &str) -> usize {
    let chars = text.chars().count();
    if chars == 0 {
        return 0;
    }
    (chars as f64 / classify_density(text)) as usize
}

/// Choose a density from the punctuation share: code/JSON is punctuation-dense,
/// prose is letter-dense (digits are neutral — dates and numbers appear in
/// prose too). Best-effort: bare base64 without a `data:` prefix is ~3%
/// punctuation and lands in prose (~3× undercount) — real base64 arrives as
/// data URIs, which [`estimate_text_tokens`] charges at base64 density.
#[allow(clippy::cast_precision_loss)]
fn classify_density(text: &str) -> f64 {
    let mut punctuation = 0u64;
    let mut alphabetic = 0u64;
    for ch in text.chars() {
        if ch.is_ascii_punctuation() {
            punctuation += 1;
        } else if ch.is_alphabetic() {
            alphabetic += 1;
        }
    }
    let total = punctuation + alphabetic;
    if total > 0 && punctuation as f64 / total as f64 > 0.3 {
        CODE_CHARS_PER_TOKEN
    } else {
        PROSE_CHARS_PER_TOKEN
    }
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
);

-- ── Durability/resume substrate (see src/jobs.rs) ─────────────────────
-- jobs must be declared BEFORE agents (lazy FK resolution).
CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    kind           TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'launched',
    task           TEXT NOT NULL DEFAULT '',
    workspace_name TEXT NOT NULL,
    user_name      TEXT NOT NULL DEFAULT '',
    channel        TEXT NOT NULL DEFAULT '',
    role           TEXT NOT NULL,
    retry_count    INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_kind_status ON jobs(kind, status);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);

CREATE TABLE IF NOT EXISTS agents (
    job_id     TEXT REFERENCES jobs(id) ON DELETE CASCADE,
    agent_id   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    idx        INTEGER,
    role       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'launched',
    outcome    TEXT,
    task       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, agent_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agents_anchor ON agents(agent_id) WHERE job_id IS NULL;

CREATE TABLE IF NOT EXISTS pending_jobs (
    id              TEXT PRIMARY KEY,
    target_agent_id TEXT NOT NULL,
    kind            TEXT NOT NULL,
    envelope        TEXT NOT NULL,
    workspace_name  TEXT NOT NULL DEFAULT '',
    user_name       TEXT NOT NULL DEFAULT '',
    channel         TEXT NOT NULL DEFAULT '',
    role            TEXT NOT NULL,
    reply_target    TEXT NOT NULL DEFAULT '',
    -- `started` and `attempts` are schema-locked write-only columns: delivery
    -- dedup (suffix + created_at) is the sole in-session discriminator.
    started         INTEGER NOT NULL DEFAULT 0,
    attempts        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_jobs_agent_created ON pending_jobs(target_agent_id, created_at);

CREATE TABLE IF NOT EXISTS ticket_stage_jobs (
    id          TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    ticket_id   TEXT NOT NULL,
    stage       TEXT NOT NULL,
    phase       TEXT NOT NULL,
    round       INTEGER NOT NULL,
    review_base INTEGER,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ask_jobs (
    id         TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS manager_jobs (
    id             TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    workspace_name TEXT NOT NULL,
    message_kind   TEXT NOT NULL,
    user_name      TEXT NOT NULL DEFAULT '',
    channel        TEXT NOT NULL DEFAULT '',
    reply_target   TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_jobs (
    id           TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    question     TEXT NOT NULL,
    -- Inert: spawn-time values only (resume data lives in `state`).
    stage        TEXT NOT NULL,
    round_index  INTEGER NOT NULL,
    budget_spent INTEGER NOT NULL,
    state        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
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

fn session_metadata_from_row(
    agent_id: &str,
    activity_str: &str,
    count: i64,
) -> Result<SessionMetadata> {
    let last_activity = turso::parse_utc_timestamp(activity_str).with_context(|| {
        format!("invalid last_activity {activity_str:?} for session {agent_id}")
    })?;
    Ok(SessionMetadata {
        agent_id: agent_id.to_string(),
        last_activity,
        message_count: usize::try_from(count).unwrap_or(0),
    })
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
            session_metadata_from_row(
                &row.get::<String>(COL_SL_AGENT_ID)?,
                &row.get::<String>(COL_SL_LAST_ACTIVITY)?,
                row.get::<i64>(COL_SL_MESSAGE_COUNT)?,
            )
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

    /// O(1) session-non-emptiness check (resume dispatch rule): true when the
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

    /// Like [`batch_append`], but also sets session context in the same transaction.
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
    /// are null.
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
    /// Post-open setup: ensure indexes.
    async fn after_open(&self) -> anyhow::Result<()> {
        // Index must exist before sessions are queried.
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

    // Delete session messages for matching transient sessions. The agents
    // table IS the marker: live sessions referenced by unfinished jobs are
    // NEVER purged (agents rows cascade-delete when the job goes terminal, so
    // protection self-heals ≤10 min after the next tick).
    tx.execute(
        &format!(
            "DELETE FROM sessions WHERE agent_id IN ( \
             SELECT agent_id FROM session_metadata \
             WHERE last_activity < ? AND {prefix_patterns} \
               AND agent_id NOT IN (SELECT agent_id FROM agents))"
        ),
        build_params.clone(),
    )
    .await?;

    // Delete the metadata entries themselves
    let deleted = tx
        .execute(
            &format!(
                "DELETE FROM session_metadata WHERE last_activity < ? AND {prefix_patterns} \
                 AND agent_id NOT IN (SELECT agent_id FROM agents)"
            ),
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

    /// The user-message timestamp block lives at the END of the message
    /// (suffix format, `\n\n` separator), so the task text is byte-stable
    /// across rounds — a changed timestamp only invalidates the tail of the
    /// provider prefix-cache. `round_ts` pins one value per round; `None`
    /// stamps now.
    #[test]
    fn user_msg_timestamp_suffix_format() {
        let msg = user_msg_with_ts("task text", Some("2026-01-01 00:00:00 (UTC)"));
        assert_eq!(
            msg.content,
            "task text\n\n<timestamp>2026-01-01 00:00:00 (UTC)</timestamp>"
        );
        let fresh = user_msg_with_ts("task text", None);
        assert!(
            fresh.content.starts_with("task text\n\n<timestamp>")
                && fresh.content.ends_with("</timestamp>"),
            "fresh stamp must still be a suffix: {}",
            fresh.content
        );
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

        // Direct session plus one session per excluded prefix — the union of
        // `manager_` and [`TRANSIENT_AGENT_ID_PREFIXES`].
        let direct_id = unique_key();
        let excluded_prefixes: Vec<&str> = std::iter::once("manager_")
            .chain(TRANSIENT_AGENT_ID_PREFIXES.iter().copied())
            .collect();
        let prefixed_ids: Vec<String> = excluded_prefixes
            .iter()
            .map(|p| format!("{p}{}", unique_key()))
            .collect();

        for id in std::iter::once(&direct_id).chain(prefixed_ids.iter()) {
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

        // Without exclusions, all 7 sessions should be listed.
        let all = store().list_sessions_with_metadata().await;
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

        // Excluding manager_ + every transient prefix → only the direct session remains.
        let excluded = store()
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
            .init(&agent_id, "hello", &ws, &role, None, "gui", "tester", None)
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
            .init(&agent_id, "", &ws, &role, None, "gui", "tester", None)
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
            .init(
                &k,
                "",
                &ws,
                &crate::Role::Assistant,
                None,
                "gui",
                "tester",
                None,
            )
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
            .init(
                &k,
                "",
                &ws,
                &crate::Role::Assistant,
                None,
                "gui",
                "tester",
                None,
            )
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
            .init(
                &k,
                "",
                &ws,
                &crate::Role::Assistant,
                None,
                "gui",
                "tester",
                None,
            )
            .await
            .unwrap();

        session.push_messages_unpersisted(&[ChatMessage::user("C1")]);
        session.finalize(&k).await.unwrap();

        let msgs = store().load(&k).await;
        let contents: Vec<&str> = msgs.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(contents, ["prompt", "u1", "C1"]);
    }
}

#[cfg(test)]
mod estimate_tests {
    use super::*;

    /// Realistic base64 payload for a 1×1 PNG (valid charset, `==` padding).
    const PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    #[test]
    fn image_markers_charge_fixed_cost_regardless_of_payload_size() {
        let small = format!("[IMAGE:data:image/png;base64,{PNG_BASE64}]");
        let huge = format!("[IMAGE:data:image/png;base64,{}]", "A".repeat(200_000));
        let est_small = estimate_tokens(&[ChatMessage::user(&small)], &[], true);
        let est_huge = estimate_tokens(&[ChatMessage::user(&huge)], &[], true);
        assert_eq!(
            est_small, est_huge,
            "image cost must not scale with base64 payload size"
        );
        assert_eq!(est_small, IMAGE_TOKENS + PER_MESSAGE_TOKENS);
    }

    #[test]
    fn cyrillic_prose_charged_by_chars_not_bytes() {
        let prose = "Проверь статус задачи и обнови описание по результатам обсуждения";
        let est = estimate_tokens(&[ChatMessage::user(prose)], &[], false);
        // UTF-8 Cyrillic is 2 bytes/char — the old bytes/4 formula estimated
        // ~1.5× higher for the Manager's mix, compacting prematurely.
        assert!(
            est < prose.len() / 4,
            "estimate {est} must stay below the byte-based formula {}",
            prose.len() / 4
        );
    }

    #[test]
    fn code_and_json_estimate_denser_than_prose() {
        let json = ChatMessage::user(r#"{"ticket_id":"t-1045","phase":"InReview","lines":42}"#);
        let prose = ChatMessage::user("Проверь статус задачи и обнови описание по результатам");
        let json_est = estimate_tokens(&[json], &[], false);
        let prose_est = estimate_tokens(&[prose], &[], false);
        assert!(
            json_est > prose_est,
            "JSON ({json_est}) must beat prose ({prose_est})"
        );
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn bare_data_uri_in_text_charges_base64_density() {
        // A data URI outside an image marker stays text (e.g. inside tool-call
        // JSON) — the payload is charged at base64 density, not prose density.
        let text = format!("image ref: data:image/png;base64,{PNG_BASE64}");
        let est = estimate_tokens(&[ChatMessage::assistant(&text)], &[], false);
        let payload_tokens = (PNG_BASE64.chars().count() as f64 / BASE64_CHARS_PER_TOKEN) as usize;
        assert!(
            est > payload_tokens && est < payload_tokens + 20,
            "base64 payload should dominate the estimate (est {est}, payload {payload_tokens})"
        );
    }

    #[test]
    fn tool_schemas_add_to_estimate() {
        let spec = ToolSpec {
            name: "search_files".into(),
            description: "Search workspace files by pattern and return matches".into(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        };
        let msg = ChatMessage::user("x");
        let with_tools = estimate_tokens(std::slice::from_ref(&msg), &[spec], false);
        let without = estimate_tokens(std::slice::from_ref(&msg), &[], false);
        assert!(with_tools > without, "tool schemas must be counted");
    }

    #[test]
    fn assistant_tool_call_frame_counts_content_and_arguments() {
        let replay = crate::providers::reasoning_roundtrip::assistant_replay_payload;
        let empty = replay(None, &[], None).to_string();
        let content = replay(
            Some("Проверю статус тикета и вернусь с результатом"),
            &[],
            None,
        )
        .to_string();
        let with_call = replay(
            Some("Проверю статус тикета и вернусь с результатом"),
            &[ToolCall {
                id: "call_1".into(),
                name: "get_ticket".into(),
                arguments: serde_json::json!({"ticket_id": "t-1045", "phase": "InReview"}),
            }],
            None,
        )
        .to_string();
        let est = |s: &str| estimate_tokens(&[ChatMessage::assistant(s)], &[], false);
        assert!(
            est(&content) > est(&empty),
            "assistant content must be counted"
        );
        assert!(
            est(&with_call) > est(&content),
            "tool-call arguments must be counted"
        );
    }

    #[test]
    fn retained_image_messages_do_not_re_trigger_summarization() {
        // Post-compaction retention keeps the 3 newest Artist user messages —
        // full base64 data URIs must stay far below the compaction threshold.
        let history: Vec<ChatMessage> = (0..3)
            .map(|_| {
                ChatMessage::user(format!(
                    "[IMAGE:data:image/png;base64,{}] keep this style",
                    "A".repeat(140_000)
                ))
            })
            .collect();
        let est = estimate_tokens(&history, &[], true);
        assert!(
            est < SUMMARIZATION_THRESHOLD / 10,
            "retained image messages must not re-trigger compaction (est {est})"
        );
    }

    #[test]
    fn artist_session_estimates_near_target_band() {
        // ~50 images at the fixed per-image cost plus prose turns must land just
        // above the threshold — the old formula estimated ~1.5M for the same
        // history, compacting at real 578–598K tokens.
        let mut history: Vec<ChatMessage> = (0..50)
            .map(|_| {
                ChatMessage::user(format!(
                    "[IMAGE:data:image/png;base64,{}]",
                    "A".repeat(120_000)
                ))
            })
            .collect();
        for i in 0..10 {
            history.push(ChatMessage::assistant(format!(
                "Изображение {i} готово, путь сохранён в workspace"
            )));
        }
        let est = estimate_tokens(&history, &[], true);
        assert!(
            (SUMMARIZATION_THRESHOLD..=SUMMARIZATION_THRESHOLD + 20_000).contains(&est),
            "Artist estimate {est} must trigger in the 200–220K band"
        );
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
// builders (one per prefix in TRANSIENT_AGENT_ID_PREFIXES). If a new transient
// role adds an agent ID builder, add it to the reverse test.
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
            tool_call_id: payload.tool_call_id,
            content: payload.content,
        });
    }

    None
}
